import logging
import math
import os
import time
import timeit
from typing import Optional, Callable, Dict

import numpy as np
import tensorflow as tf
from sonnet.src.types import TensorNest

from sc2_imitation_learning.agents import Agent, build_saved_agent
from sc2_imitation_learning.common import utils
from sc2_imitation_learning.common.progress_logger import ConsoleProgressLogger, TensorboardProgressLogger
from sc2_imitation_learning.common.utils import make_dummy_batch, swap_leading_axes
from sc2_imitation_learning.environment.environment import ObservationSpace, ActionSpace

logger = logging.getLogger(__file__)


def compute_correct_predictions(target_actions, learner_actions, label_mask_value: Optional[int] = -1):
    target_actions = tf.cast(target_actions, dtype=tf.int32)
    learner_actions = tf.cast(learner_actions, dtype=tf.int32)
    correct_predictions = tf.equal(target_actions, learner_actions)
    if label_mask_value is not None:
        masks = tf.not_equal(target_actions, label_mask_value)
        correct_predictions = tf.logical_and(correct_predictions, masks)
        num_samples = tf.math.count_nonzero(masks, dtype=tf.int32)
    else:
        num_samples = tf.size(target_actions, dtype=tf.int32)
    num_correct_predictions = tf.math.count_nonzero(correct_predictions, dtype=tf.int32)
    return num_correct_predictions, num_samples


def compute_neg_log_probs(labels, logits, label_mask_value: Optional[int] = -1):
    """ Computes negative log probabilities of labels given logits, where labels equal to `label_mask_value`
    are zero-masked """
    if label_mask_value is not None:
        # mask labels to prevent invalid (e.g. negative) label values
        mask = tf.math.not_equal(labels, label_mask_value)
        labels *= tf.cast(mask, dtype=labels.dtype)

    # calculate neg log probabilities
    neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    if label_mask_value is not None:
        # mask neg_log_probs with pre calculated mask
        neg_log_probs *= tf.cast(mask, dtype=neg_log_probs.dtype)

    return neg_log_probs


def compute_cross_entropy_loss(labels, logits, label_mask_value: Optional[int] = -1):
    """ Computes the cross entropy loss, where labels equal to `label_mask_value` are ignored. """
    neg_log_probs = tf.nest.map_structure(
        lambda x, y: compute_neg_log_probs(x, y, label_mask_value),
        labels,
        logits)
    # sum negative log probabilities and average across time dimension
    return tf.reduce_mean(sum(tf.nest.flatten(neg_log_probs)), axis=0)


def evaluate_gradients(
        trajectory_ids: tf.Tensor,
        trajectories: TensorNest,
        global_batch_size: int,
        agent: Agent,
        agent_states: utils.Aggregator,
        l2_regularization=0.):
    trajectories = tf.nest.map_structure(swap_leading_axes, trajectories)  # B x T -> T x B
    env_outputs = (trajectories['reward'], trajectories['done'], trajectories['observation'])

    prev_agent_states = agent_states.read(trajectory_ids)

    with tf.GradientTape() as tape:
        agent_outputs, curr_agent_states = agent(
            prev_actions=trajectories['prev_action'],
            env_outputs=env_outputs,
            core_state=prev_agent_states,
            unroll=True,
            teacher_actions=trajectories['action'])

        crosse_entropy_loss = tf.nn.compute_average_loss(
            per_example_loss=compute_cross_entropy_loss(trajectories['action'], agent_outputs.logits),
            global_batch_size=global_batch_size)

        if l2_regularization > 0.:
            l2_loss = tf.nn.scale_regularization_loss(
                regularization_loss=sum([tf.nn.l2_loss(v) for v in agent.trainable_variables]))
        else:
            l2_loss = 0.

        loss = crosse_entropy_loss + l2_regularization * l2_loss

    # Update current state.
    agent_states.replace(trajectory_ids, curr_agent_states)

    gradients = tape.gradient(loss, agent.trainable_variables)
    grad_norm = tf.linalg.global_norm(gradients) * (1 / tf.distribute.get_strategy().num_replicas_in_sync)

    correct_predictions = tf.nest.map_structure(
        compute_correct_predictions, trajectories['action'], agent_outputs.actions)

    summary = {
        'loss': {
            'loss': loss,
            'ce': crosse_entropy_loss,
            'l2': l2_loss,
        },
        'grad_norm': grad_norm,
        'num_correct': {
            action_name: num_correct for action_name, (num_correct, _) in correct_predictions.items()
        },
        'num_samples': {
            action_name: num_samples for action_name, (_, num_samples) in correct_predictions.items()
        },
    }

    return gradients, summary


def accumulate_gradients(
        accumulated_gradients: tf.Tensor,
        trajectory_ids: tf.Tensor,
        trajectories: TensorNest,
        global_batch_size: int,
        agent: Agent,
        agent_states: utils.Aggregator,
        l2_regularization=0.):
    gradients, summary = evaluate_gradients(
        trajectory_ids=trajectory_ids, trajectories=trajectories, global_batch_size=global_batch_size, agent=agent,
        agent_states=agent_states, l2_regularization=l2_regularization)

    for t, g in zip(accumulated_gradients, gradients):
        t.assign_add(g)

    return summary


def apply_gradients(
        accumulated_gradients: tf.Tensor,
        agent: Agent,
        update_frequency: int,
        optimizer: tf.optimizers.Optimizer):
    gradients = tuple([g / float(update_frequency) for g in accumulated_gradients])

    optimizer.apply_gradients(zip(gradients, agent.trainable_variables))

    for v in accumulated_gradients:
        v.assign(tf.zeros_like(v))


def train_step(trajectory_ids: tf.Tensor,
               trajectories: TensorNest,
               global_batch_size: int,
               agent: Agent,
               optimizer: tf.optimizers.Optimizer,
               agent_states: utils.Aggregator,
               l2_regularization=0.):
    gradients, summary = evaluate_gradients(
        trajectory_ids=trajectory_ids, trajectories=trajectories, global_batch_size=global_batch_size, agent=agent,
        agent_states=agent_states, l2_regularization=l2_regularization)

    optimizer.apply_gradients(zip(gradients, agent.trainable_variables))

    return summary


def learner_loop(log_dir: str,
                 observation_space: ObservationSpace,
                 action_space: ActionSpace,
                 training_strategy: tf.distribute.Strategy,
                 dataset_fn: Callable[[tf.distribute.InputContext], tf.data.Dataset],
                 agent_fn: Callable[[], Agent],
                 optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
                 total_train_samples: int,
                 batch_size: int,
                 sequence_size: int,
                 l2_regularization: float,
                 update_frequency: int,
                 num_episodes: int,
                 eval_fn: Callable[[Agent], Dict],
                 eval_interval: int,
                 max_to_keep_checkpoints: int = None,
                 save_checkpoint_interval: float = 1800.,  # every 30 minutes
                 tensorboard_log_interval: float = 10.,
                 console_log_interval: float = 60.) -> None:

    batch_samples = batch_size*sequence_size
    total_steps = math.ceil(total_train_samples/float(batch_samples))
    eval_interval_steps = math.ceil(eval_interval/float(batch_samples))
    global_step = tf.Variable(0, dtype=tf.int64)
    last_checkpoint_time = None

    with training_strategy.scope():
        agent = agent_fn()
        optimizer = optimizer_fn()

        # initialize agent variables by feeding dummy batch:
        initial_agent_state = agent.initial_state(1)
        prev_actions, env_outputs = make_dummy_batch(observation_space, action_space)
        agent(prev_actions=prev_actions, env_outputs=env_outputs, core_state=initial_agent_state, unroll=True)

        # initialize all optimizer variables:
        _ = optimizer.iterations
        optimizer._create_hypers()
        optimizer._create_slots(agent.trainable_variables)

        checkpoint = tf.train.Checkpoint(agent=agent, optimizer=optimizer, step=global_step)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, log_dir, max_to_keep=max_to_keep_checkpoints)
        if checkpoint_manager.latest_checkpoint:
            logging.info(f'Restoring checkpoint: {checkpoint_manager.latest_checkpoint}')
            checkpoint.restore(checkpoint_manager.latest_checkpoint).assert_consumed()

    # agent states and accumulated gradients should not be shared between replicas:
    agent_state_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
    agent_states = utils.Aggregator(num_episodes, agent_state_specs, 'agent_states')
    if update_frequency > 1:
        accumulated_gradients = [tf.Variable(tf.zeros_like(v), trainable=False) for v in agent.trainable_variables]
    else:
        accumulated_gradients = None

    dataset = training_strategy.experimental_distribute_datasets_from_function(dataset_fn)

    @tf.function
    def distributed_train_step(trajectory_ids, sequences):
        if update_frequency > 1:
            per_replica_summary = training_strategy.run(accumulate_gradients, kwargs={
                'accumulated_gradients': accumulated_gradients,
                'trajectory_ids': trajectory_ids,
                'trajectories': sequences,
                'global_batch_size': batch_size,
                'agent': agent,
                'agent_states': agent_states,
                'l2_regularization': l2_regularization,
            })
            if tf.math.mod(global_step, update_frequency) == 0:
                training_strategy.run(apply_gradients, kwargs={
                    'accumulated_gradients': accumulated_gradients,
                    'agent': agent,
                    'update_frequency': update_frequency,
                    'optimizer': optimizer,
                })
        else:
            per_replica_summary = training_strategy.run(train_step, kwargs={
                'trajectory_ids': trajectory_ids,
                'trajectories': sequences,
                'global_batch_size': batch_size,
                'agent': agent,
                'optimizer': optimizer,
                'agent_states': agent_states,
                'l2_regularization': l2_regularization
            })
        summary = tf.nest.map_structure(lambda t: training_strategy.reduce("SUM", t, axis=None), per_replica_summary)
        return summary

    def should_evaluate(_step):
        return _step % eval_interval_steps == 0

    def should_save_checkpoint(_time):
        return last_checkpoint_time is None or _time - last_checkpoint_time >= save_checkpoint_interval

    def iter_dataset(_dataset):
        dataset_iterator = iter(_dataset)
        while global_step.numpy() < total_steps:
            yield next(dataset_iterator)

    console_logger = ConsoleProgressLogger(
        final_step=total_steps,
        batch_samples=batch_samples,
        logging_interval=console_log_interval,
        initial_step=global_step.numpy())
    console_logger.start()

    tensorboard_logger = TensorboardProgressLogger(
        summary_writer=tf.summary.create_file_writer(log_dir),
        logging_interval=tensorboard_log_interval,
        initial_step=global_step.numpy())
    tensorboard_logger.start()

    last_step_time = timeit.default_timer()

    for batch in iter_dataset(dataset):
        step = global_step.numpy()

        train_summary = distributed_train_step(*batch)

        current_time = timeit.default_timer()
        step_duration = current_time - last_step_time
        last_step_time = current_time

        train_summary = tf.nest.map_structure(lambda s: s.numpy(), train_summary)
        train_summary['samples'] = (step+1) * batch_samples
        train_summary['samples_per_second'] = batch_samples / float(step_duration)
        train_summary['learning_rate'] = optimizer._decayed_lr('float32').numpy()
        train_summary['accuracy'] = {
            action_name: np.true_divide(train_summary['num_correct'][action_name], num_samples)
            for action_name, num_samples in train_summary['num_samples'].items() if num_samples > 0
        }
        console_logger.log_dict(train_summary, step)
        tensorboard_logger.log_dict(train_summary, step)

        if should_evaluate(step):
            checkpoint_manager.save()
            saved_agent = build_saved_agent(agent, observation_space, action_space)
            tf.saved_model.save(saved_agent, os.path.join(log_dir, 'saved_model'))
            eval_summary = eval_fn(os.path.join(log_dir, 'saved_model'))
            tensorboard_logger.log_dict(eval_summary, step)

        now = time.time()
        if should_save_checkpoint(now):
            checkpoint_manager.save()
            saved_agent = build_saved_agent(agent, observation_space, action_space)
            tf.saved_model.save(saved_agent, os.path.join(log_dir, 'saved_model'))
            last_checkpoint_time = now

        global_step.assign_add(1)

    checkpoint_manager.save()
    saved_agent = build_saved_agent(agent, observation_space, action_space)
    tf.saved_model.save(saved_agent, os.path.join(log_dir, 'saved_model'))

    console_logger.shutdown()
    tensorboard_logger.shutdown()
