import concurrent.futures
import datetime
import functools
import logging
import multiprocessing
import os
from typing import Type, Callable, List, Optional, Union, Dict

import gin
import numpy as np
import tensorflow as tf
from absl import app, flags

from sc2_imitation_learning.agents import Agent
from sc2_imitation_learning.behaviour_cloning.learner import learner_loop
from sc2_imitation_learning.common.evaluator import evaluate_on_multiple_envs
from sc2_imitation_learning.common.utils import gin_register_external_configurables, gin_config_str_to_dict
from sc2_imitation_learning.dataset.dataset import DataLoader
from sc2_imitation_learning.environment.environment import ObservationSpace, ActionSpace
from sc2_imitation_learning.environment.sc2_environment import SC2SingleAgentEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

flags.DEFINE_string('logdir', f"./experiments/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                    'Experiment logging directory.')
flags.DEFINE_multi_string('gin_file', ['./configs/1v1/behaviour_cloning.gin'], 'List of paths to Gin config files.')
flags.DEFINE_multi_string('gin_param', None, 'List of Gin parameter bindings.')

# logger config
flags.DEFINE_bool('wandb_logging_enabled', False, 'If wandb logging should be enabled.')
flags.DEFINE_string('wandb_project', 'sc2-il', 'Name of the wandb project.')
flags.DEFINE_string('wandb_entity', None, 'Name of the wandb entity.')
flags.DEFINE_list('wandb_tags', ['behaviour_cloning'], 'List of wandb tags.')

FLAGS = flags.FLAGS

gin_register_external_configurables()


def agent_fn(saved_model_path, *args, **kwargs) -> Agent:
    return tf.saved_model.load(saved_model_path)


@gin.configurable
def evaluate(saved_model_path: str,
             envs: List[Type[SC2SingleAgentEnv]] = gin.REQUIRED,
             num_episodes: int = gin.REQUIRED,
             random_seed: int = gin.REQUIRED,
             num_evaluators: int = gin.REQUIRED) -> Dict[str, Union[int, float, np.ndarray]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        available_gpus = executor.submit(tf.config.list_physical_devices, 'GPU').result()

    episode_stats = evaluate_on_multiple_envs(
        agent_fn=functools.partial(agent_fn, saved_model_path),
        envs=envs,
        num_episodes=num_episodes,
        num_evaluators=num_evaluators,
        random_seed=random_seed,
        replay_dir=os.path.abspath(os.path.join(FLAGS.logdir, 'replays')),
        available_gpus=available_gpus)

    return {
        matchup: {
            'num_episodes': len(stats),
            'episode_frames/mean': np.mean([s.num_frames for s in stats]),
            'episode_steps/mean': np.mean([s.num_steps for s in stats]),
            'episode_reward/mean': np.mean([s.reward for s in stats]),
            'episode_frames/min': np.min([s.num_frames for s in stats]),
            'episode_steps/min': np.min([s.num_steps for s in stats]),
            'episode_reward/min': np.min([s.reward for s in stats]),
            'episode_frames/max': np.max([s.num_frames for s in stats]),
            'episode_steps/max': np.max([s.num_steps for s in stats]),
            'episode_reward/max': np.max([s.reward for s in stats]),
        }
        for matchup, stats in episode_stats.items()
    }


@gin.configurable
def train(action_space: ActionSpace = gin.REQUIRED,
          observation_space: ObservationSpace = gin.REQUIRED,
          data_loader: DataLoader = gin.REQUIRED,
          batch_size: int = gin.REQUIRED,
          sequence_length: int = gin.REQUIRED,
          total_train_samples: int = gin.REQUIRED,
          l2_regularization: float = gin.REQUIRED,
          update_frequency: int = gin.REQUIRED,
          agent_fn: Callable[[], Agent] = gin.REQUIRED,
          optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer] = gin.REQUIRED,
          eval_interval: int = gin.REQUIRED,
          max_to_keep_checkpoints: Optional[int] = None,
          save_checkpoint_interval: float = 1800,
          tensorboard_log_interval: float = 10,
          console_log_interval: float = 60):

    def dataset_fn(ctx: tf.distribute.InputContext) -> tf.data.Dataset:
        num_episodes = data_loader.num_episodes // ctx.num_input_pipelines
        start_index = ctx.input_pipeline_id * num_episodes
        dataset = data_loader.load(
            batch_size=ctx.get_per_replica_batch_size(global_batch_size=batch_size),
            sequence_length=sequence_length,
            offset_episodes=start_index,
            num_episodes=num_episodes,
            num_workers=min(num_episodes, os.cpu_count()))
        return dataset.prefetch(buffer_size=ctx.num_replicas_in_sync)

    training_strategy = tf.distribute.MirroredStrategy([])

    learner_loop(log_dir=FLAGS.logdir,
                 observation_space=observation_space,
                 action_space=action_space,
                 training_strategy=training_strategy,
                 dataset_fn=dataset_fn,
                 agent_fn=agent_fn,
                 optimizer_fn=optimizer_fn,
                 total_train_samples=total_train_samples,
                 batch_size=batch_size,
                 sequence_size=sequence_length,
                 l2_regularization=l2_regularization,
                 update_frequency=update_frequency,
                 num_episodes=data_loader.num_episodes,
                 eval_fn=evaluate,
                 eval_interval=eval_interval,
                 max_to_keep_checkpoints=max_to_keep_checkpoints,
                 save_checkpoint_interval=save_checkpoint_interval,
                 tensorboard_log_interval=tensorboard_log_interval,
                 console_log_interval=console_log_interval)


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    os.makedirs(FLAGS.logdir, exist_ok=True)

    gin_config_str = gin.config_str(max_line_length=120)

    print("Loaded configuration:")
    print(gin_config_str)

    with open(os.path.join(FLAGS.logdir, 'config.gin'), mode='w') as f:
        f.write(gin_config_str)

    if FLAGS.wandb_logging_enabled:
        import wandb
        experiment_name = os.path.basename(FLAGS.logdir.rstrip("/"))
        job_type = 'train'
        wandb.init(
            id=f"{experiment_name}-{job_type}",
            name=f"{experiment_name}-{job_type}",
            group=experiment_name,
            job_type=job_type,
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            tags=FLAGS.wandb_tags,
            resume="allow",
            config=gin_config_str_to_dict(gin_config_str))
        wandb.tensorboard.patch(save=False, tensorboardX=False)

    train()


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    multiprocessing.set_start_method('spawn')
    app.run(main)
