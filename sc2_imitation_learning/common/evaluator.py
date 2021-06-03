import logging
import multiprocessing
import sys
import traceback
from collections import defaultdict
from queue import Queue
from typing import Type, Optional, Callable, NamedTuple, List, Dict

import numpy as np
import tensorflow as tf
from absl.flags import FLAGS

from sc2_imitation_learning.agents import Agent
from sc2_imitation_learning.common.utils import make_dummy_action
from sc2_imitation_learning.environment.sc2_environment import SC2SingleAgentEnv

logger = logging.getLogger(__file__)


class EvalEpisode(NamedTuple):
    matchup: str
    stats: 'EpisodeStats'


class EpisodeStats(NamedTuple):
    num_frames: int
    num_steps: int
    reward: float


class Evaluator(multiprocessing.Process):

    def __init__(self,
                 agent_fn: Callable[[], Agent],
                 env: SC2SingleAgentEnv,
                 queue_in: Queue,
                 queue_out: Queue,
                 device: Optional = None) -> None:
        super().__init__()
        self._agent_fn = agent_fn
        self._env = env
        self._queue_in = queue_in
        self._queue_out = queue_out
        self._device = device

    def run(self) -> None:
        FLAGS(sys.argv)

        if self._device is not None:
            tf.config.set_visible_devices(self._device, 'GPU')

        agent = self._agent_fn()
        agent_state = agent.initial_state(1)

        seed = self._queue_in.get()
        while seed is not None:
            env: SC2SingleAgentEnv = self._env
            env.restart(seed)

            try:
                total_reward, frame, step = 0., 0, 0
                reward, done, observation = 0., False, env.reset()
                action = make_dummy_action(env.action_space, num_batch_dims=1)
                while not done:
                    env_outputs = (
                        tf.constant([reward], dtype=tf.float32),
                        tf.constant([step == 0], dtype=tf.bool),
                        tf.nest.map_structure(lambda o: tf.constant([o], dtype=tf.dtypes.as_dtype(o.dtype)), observation))
                    agent_output, agent_state = agent(action, env_outputs, agent_state)
                    action = tf.nest.map_structure(lambda t: t.numpy(), agent_output.actions)
                    reward, _, done, observation = env.step(action)
                    total_reward += reward
                    frame += action['step_mul'] + 1
                    step += 1

                self._queue_out.put(
                    EvalEpisode(matchup=env.level_name,
                                stats=EpisodeStats(num_frames=frame, num_steps=step, reward=total_reward)))

            except Exception as e:
                logger.error(f"Failed to evaluate episode (stacktrace below). Restart env.")
                traceback.print_exc()
                continue
            finally:
                env.close()

            seed = self._queue_in.get()


def evaluate_on_single_env(agent_fn: Callable[[], Agent],
                           env_cls: Type[SC2SingleAgentEnv],
                           num_episodes: int,
                           num_evaluators: int,
                           random_seed: int = None,
                           replay_dir: Optional[str] = None,
                           available_gpus: Optional[List] = None) -> Dict[str, EpisodeStats]:
    logger.info(f"Start evaluation for {num_episodes} episodes using {num_evaluators} evaluator threads.")

    queue_in = multiprocessing.Queue()
    queue_out = multiprocessing.Queue()

    rngesus: np.random.Generator = np.random.default_rng(seed=random_seed)
    random_seeds = rngesus.integers(low=0, high=np.iinfo(np.int32).max, size=num_episodes)
    for random_seed in random_seeds:
        queue_in.put(int(random_seed))

    for _ in range(num_evaluators):
        queue_in.put(None)

    env = env_cls(replay_dir=replay_dir)

    evaluators = [
        Evaluator(agent_fn=agent_fn, env=env, queue_in=queue_in, queue_out=queue_out,
                  device=None if available_gpus is None else available_gpus[i % len(available_gpus)])
        for i in range(num_evaluators)]

    for evaluator in evaluators:
        evaluator.start()

    logger.info("All evaluator threads started.")

    all_episode_stats = defaultdict(list)
    total_episodes = 0
    while total_episodes < num_episodes:
        eval_episode: EvalEpisode = queue_out.get()
        all_episode_stats[eval_episode.matchup].append(eval_episode.stats)
        total_episodes += 1
        logger.info(f"Episode {total_episodes} completed: "
                    f"matchup={eval_episode.matchup}, "
                    f"reward={eval_episode.stats.reward:.2f}, "
                    f"frames={eval_episode.stats.num_frames}, "
                    f"matchup_rewards_mean={np.mean([s.reward for s in all_episode_stats[eval_episode.matchup]]):.2f}")

    logger.info(f"Evaluation completed:\n\t" + "\n\t".join([
        f"matchup={matchup}, "
        f"mean_reward={np.mean([s.reward for s in stats]):.2f} (std={np.std([s.reward for s in stats]):.2f})"
        for matchup, stats in all_episode_stats.items()
    ]))

    for evaluator in evaluators:
        evaluator.join()

    logger.info("All evaluator threads stopped.")

    return dict(all_episode_stats)


def evaluate_on_multiple_envs(agent_fn: Callable[[], Agent],
                              envs: List[Type[SC2SingleAgentEnv]],
                              num_episodes: int,
                              num_evaluators: int,
                              random_seed: int = None,
                              replay_dir: Optional[str] = None,
                              available_gpus: Optional[List] = None) -> Dict[str, EpisodeStats]:
    all_episode_stats = defaultdict(list)

    for env_cls in envs:
        eval_episodes = evaluate_on_single_env(
            agent_fn=agent_fn,
            env_cls=env_cls,
            num_episodes=num_episodes,
            num_evaluators=num_evaluators,
            random_seed=random_seed,
            replay_dir=replay_dir,
            available_gpus=available_gpus)

        for matchup, stats in eval_episodes.items():
            all_episode_stats[matchup].extend(stats)

    return dict(all_episode_stats)
