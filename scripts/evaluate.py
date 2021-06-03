import concurrent.futures
import datetime
import functools
import json
import logging
import multiprocessing
import os
from typing import Type, List

import gin
import tensorflow as tf
import wandb
from absl import app, flags

from sc2_imitation_learning.agents import Agent
from sc2_imitation_learning.common.evaluator import evaluate_on_multiple_envs
from sc2_imitation_learning.common.utils import compute_stats_dict, gin_register_external_configurables
from sc2_imitation_learning.environment.sc2_environment import SC2SingleAgentEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

flags.DEFINE_string('logdir', './experiments/2021-03-21_03-00-19', 'Experiment directory.')
flags.DEFINE_multi_string('gin_file', ['./configs/1v1/evaluate.gin'], 'List of paths to Gin config files.')
flags.DEFINE_multi_string('gin_param', None, 'List of Gin parameter bindings.')

# logger config
flags.DEFINE_bool('wandb_logging_enabled', False, 'If wandb logging should be enabled.')
flags.DEFINE_string('wandb_project', 'sc2-il', 'Name of the wandb project.')
flags.DEFINE_string('wandb_entity', None, 'Name of the wandb entity.')
flags.DEFINE_list('wandb_tags', ['behaviour_cloning'], 'List of wandb tags.')

FLAGS = flags.FLAGS


gin_register_external_configurables()


def agent_fn(experiment_path, *args, **kwargs) -> Agent:
    return tf.saved_model.load(os.path.join(experiment_path, 'saved_model'))


@gin.configurable
def evaluate(experiment_path: str,
             envs: List[Type[SC2SingleAgentEnv]] = gin.REQUIRED,
             num_episodes: int = gin.REQUIRED,
             random_seed: int = gin.REQUIRED,
             num_evaluators: int = gin.REQUIRED):

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        available_gpus = executor.submit(tf.config.list_physical_devices, 'GPU').result()

    episode_stats = evaluate_on_multiple_envs(
        agent_fn=functools.partial(agent_fn, experiment_path),
        envs=envs,
        num_episodes=num_episodes,
        num_evaluators=num_evaluators,
        random_seed=random_seed,
        replay_dir=os.path.abspath(os.path.join(FLAGS.logdir, 'replays', 'eval')),
        available_gpus=available_gpus)

    return {
        matchup: {
            'num_episodes': len(stats),
            'episode_frames': compute_stats_dict([s.num_frames for s in stats]),
            'episode_steps': compute_stats_dict([s.num_steps for s in stats]),
            'episode_reward': compute_stats_dict([s.reward for s in stats]),
        }
        for matchup, stats in episode_stats.items()
    }


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    if not os.path.exists(FLAGS.logdir):
        raise ValueError(f"Logdir '{FLAGS.logdir}' does not exist exists.")

    if FLAGS.wandb_logging_enabled:
        experiment_name = os.path.basename(FLAGS.logdir.rstrip("/"))
        job_type = 'test'
        wandb.init(
            id=f"{experiment_name}-{job_type}",
            name=f"{experiment_name}-{job_type}",
            group=experiment_name,
            job_type=job_type,
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            tags=FLAGS.wandb_tags,
            resume="allow")
        wandb.tensorboard.patch(save=False, tensorboardX=False)

    eval_outcome = evaluate(FLAGS.logdir)

    with open(os.path.join(FLAGS.logdir, f'eval_outcome_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'), 'w') as f:
        json.dump(eval_outcome, f, indent=4)

    if FLAGS.wandb_logging_enabled:
        wandb.log(eval_outcome)


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    multiprocessing.set_start_method('spawn')
    app.run(main)
