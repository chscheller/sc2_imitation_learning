import logging
import os
import traceback
from typing import Type

import gin
import tensorflow as tf
from absl import app
from absl import flags

from sc2_imitation_learning.common.utils import gin_register_external_configurables, make_dummy_action
from sc2_imitation_learning.environment.sc2_environment import SC2SingleAgentEnv

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

flags.DEFINE_string('agent_dir', default=None, help='Path to the directory where the agent is stored.')
flags.DEFINE_multi_string('gin_file', ['configs/1v1/play_agent_vs_bot.gin'], 'List of paths to Gin config files.')
flags.DEFINE_multi_string('gin_param', None, 'List of Gin parameter bindings.')

FLAGS = flags.FLAGS


gin_register_external_configurables()


@gin.configurable
def play(env_fn: Type[SC2SingleAgentEnv] = gin.REQUIRED, num_episodes: int = gin.REQUIRED) -> None:

    agent = tf.saved_model.load(FLAGS.agent_dir)
    agent_state = agent.initial_state(1)

    env = env_fn()
    env.launch()
    for episode in range(num_episodes):
        episode_reward = 0.
        episode_frames = 0
        episode_steps = 0
        try:
            reward, done, observation = 0., False, env.reset()
            action = make_dummy_action(env.action_space, num_batch_dims=1)
            while not done:
                env_outputs = (
                    tf.constant([reward], dtype=tf.float32),
                    tf.constant([episode_steps == 0], dtype=tf.bool),
                    tf.nest.map_structure(lambda o: tf.constant([o], dtype=tf.dtypes.as_dtype(o.dtype)), observation))
                agent_output, agent_state = agent(action, env_outputs, agent_state)
                action = tf.nest.map_structure(lambda t: t.numpy(), agent_output.actions)
                reward, _, done, observation = env.step(action)
                episode_reward += reward
                episode_frames += action['step_mul'] + 1
                episode_steps += 1
        except Exception as e:
            logger.error(f"Failed to play episode {episode} (stacktrace below).")
            traceback.print_exc()
        finally:
            logger.info(f"Episode completed: total reward={episode_reward}, frames={episode_frames}, "
                        f"steps={episode_steps}")
            env.close()


def main(_):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    play()


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    app.run(main)
