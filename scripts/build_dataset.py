import logging
import os
import sys
from typing import Iterator, NamedTuple, List, Dict, Text, AbstractSet

import gin
import pypeln as pl
from absl import app
from absl import flags
from pypeln.process import IterableQueue
from pypeln.process.api.filter import FilterFn
from pypeln.process.api.map import MapFn
from pysc2.env.environment import StepType
from pysc2.env.sc2_env import Race
from tqdm import tqdm

from sc2_imitation_learning.common.replay_processor import ReplayProcessor, get_replay_info
from sc2_imitation_learning.common.utils import retry
from sc2_imitation_learning.dataset.dataset import ActionTimeStep, store_episode_to_hdf5, get_dataset_specs
from sc2_imitation_learning.dataset.sc2_dataset import SC2REPLAY_RACES
from sc2_imitation_learning.environment.environment import ActionSpace, ObservationSpace
from sc2_imitation_learning.environment.sc2_environment import SC2ActionSpace, SC2ObservationSpace, SC2InterfaceConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

flags.DEFINE_string('replays_path', default='./data/replays/4.7.1/',
                    help='Path to the directory where the replays are stored.')
flags.DEFINE_string('dataset_path', default='./data/datasets/1v1/v1',
                    help='Path to the directory where the dataset will be stored.')
flags.DEFINE_integer('num_workers', os.cpu_count(), help='Number of parallel workers.')
flags.DEFINE_multi_string('gin_file', ['./configs/1v1/build_dataset.gin'], help='List of paths to Gin config files.')
flags.DEFINE_multi_string('gin_param', None, help='List of Gin parameter bindings.')

FLAGS = flags.FLAGS


class ReplayMeta(NamedTuple):
    observed_player_id: int
    replay_info: Dict
    replay_path: Text


class Replay(NamedTuple):
    time_steps: List[ActionTimeStep]
    replay_meta: ReplayMeta


def is_not_none(x): return x is not None


def find_replays(replay_path: Text) -> Iterator[str]:
    for entry in os.scandir(os.path.abspath(replay_path)):
        if entry.name.endswith('.SC2Replay'):
            yield entry.path


def load_replay_meta(replay_path: Text) -> List[ReplayMeta]:
    replay_info = get_replay_info(replay_path)
    return [ReplayMeta(player['PlayerID'], replay_info, replay_path) for player in replay_info['Players']]


@gin.register
class FilterReplay(FilterFn):
    def __init__(self,
                 min_duration: float = 0.,
                 min_mmr: int = 0,
                 min_apm: int = 0,
                 observed_player_races: AbstractSet[Race] = frozenset((Race.protoss, Race.terran, Race.zerg)),
                 opponent_player_races: AbstractSet[Race] = frozenset((Race.protoss, Race.terran, Race.zerg)),
                 wins_only: bool = False) -> None:
        super().__init__()
        self.min_duration = min_duration
        self.min_mmr = min_mmr
        self.min_apm = min_apm
        self.observed_player_races = observed_player_races
        self.opponent_player_races = opponent_player_races
        self.wins_only = wins_only

    def __call__(self, replay_meta: ReplayMeta, **kwargs) -> bool:
        observed_player_info = next(
            filter(lambda p: p['PlayerID'] == replay_meta.observed_player_id, replay_meta.replay_info['Players']))
        if len(replay_meta.replay_info['Players']) > 1:
            opponent_player_info = next(
                filter(lambda p: p['PlayerID'] != replay_meta.observed_player_id, replay_meta.replay_info['Players']))
        else:
            opponent_player_info = None
        return (replay_meta.replay_info['Duration'] >= self.min_duration
                and observed_player_info.get('MMR', 0) >= self.min_mmr
                and observed_player_info['APM'] >= self.min_apm
                and SC2REPLAY_RACES[observed_player_info['AssignedRace']] in self.observed_player_races
                and (opponent_player_info is None or
                     SC2REPLAY_RACES[opponent_player_info['AssignedRace']] in self.opponent_player_races)
                and (not self.wins_only or observed_player_info['Result'] == 'Win'))


@gin.register
class ProcessReplay(MapFn):
    def __init__(self,
                 interface_config: SC2InterfaceConfig = gin.REQUIRED,
                 action_space: SC2ActionSpace = gin.REQUIRED,
                 observation_space: SC2ObservationSpace = gin.REQUIRED,
                 sc2_version: str = gin.REQUIRED) -> None:
        super().__init__()
        self.interface_config = interface_config
        self.action_space = action_space
        self.observation_space = observation_space
        self.sc2_version = sc2_version

    @retry(max_tries=3)
    def __call__(self, replay_meta: ReplayMeta, **kwargs) -> Replay:
        if not FLAGS.is_parsed():
            FLAGS(sys.argv)

        def _valid_or_fallback_action(o: dict, a: Dict):
            if o['scalar_features']['available_actions'][a['action_type']] == 0:
                return self.action_space.no_op()  # action_type not available
            elif 'build_queue_length' in o['scalar_features'] and \
                    o['scalar_features']['build_queue_length'] <= a['build_queue_id']:
                return self.action_space.no_op()  # build_queue_id not available
            elif 'multi_select_length' in o['scalar_features'] and \
                    o['scalar_features']['multi_select_length'] <= a['select_unit_id']:
                return self.action_space.no_op()  # select_unit_id not available
            elif 'cargo_length' in o['scalar_features'] and \
                    o['scalar_features']['cargo_length'] <= a['unload_id']:
                return self.action_space.no_op()  # unload_id not available
            else:
                return a

        with ReplayProcessor(
                replay_path=replay_meta.replay_path,
                interface_config=self.interface_config,
                observation_space=self.observation_space,
                action_space=self.action_space,
                observed_player_id=replay_meta.observed_player_id,
                version=self.sc2_version) as replay_processor:
            sampled_replay: List[ActionTimeStep] = []
            reward = 0.
            for curr_ts, curr_act in replay_processor.iterator():
                action = _valid_or_fallback_action(curr_ts.observation, curr_act)
                reward += curr_ts.reward
                if (                                            # add timestep to replay if:
                        len(sampled_replay) == 0                # a) it is the first timestep of an episode,
                        or curr_ts.step_type == StepType.LAST   # b) it is the last timestep of an episode,
                        or action['action_type'] != 0           # c) an action other than noop is executed or
                        or sampled_replay[-1].action['step_mul'] == self.interface_config.max_step_mul - 1  # d) max_step_mul is reached
                ):
                    sampled_replay.append(ActionTimeStep(observation=curr_ts.observation, action=action, reward=reward,
                                                         done=len(sampled_replay) == 0))
                    reward = 0.
                else:  # if timestep is skipped, increment step_mul of most recent action
                    sampled_replay[-1].action['step_mul'] += 1

        return Replay(time_steps=sampled_replay, replay_meta=replay_meta)


@gin.register
class StoreReplay(MapFn):

    def __init__(self,
                 dataset_path: str,
                 action_space: ActionSpace = gin.REQUIRED,
                 observation_space: ObservationSpace = gin.REQUIRED) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.action_space = action_space
        self.observation_space = observation_space

    def __call__(self, replay: Replay, **kwargs) -> str:
        replay_name = os.path.splitext(os.path.basename(replay.replay_meta.replay_path))[0]
        replay_name = f"{replay_name}_{replay.replay_meta.observed_player_id}"
        specs = get_dataset_specs(self.action_space, self.observation_space)
        file_name = store_episode_to_hdf5(
            path=self.dataset_path,
            name=replay_name,
            episode=replay.time_steps,
            episode_info={
                'observed_player_id': replay.replay_meta.observed_player_id,
                'replay_path': replay.replay_meta.replay_path,
                'replay_info': replay.replay_meta.replay_info
            },
            specs=specs)
        return file_name


def patch_iterable_queue():
    """ Patches __getstate__ and __setstate__ of IterableQueues such that namespace and exception_queue attributes get
    exported/restored. See PR: https://github.com/cgarciae/pypeln/pull/74 """
    orig_getstate = IterableQueue.__getstate__
    orig_setstate = IterableQueue.__setstate__

    def new_getstate(self):
        return orig_getstate(self) + (self.namespace, self.exception_queue)

    def new_setstate(self, state):
        orig_setstate(self, state[:-2])
        self.namespace, self.exception_queue = state[-2:]

    IterableQueue.__getstate__ = new_getstate
    IterableQueue.__setstate__ = new_setstate

    logger.info("Pickle patch for IterableQueue applied.")


patch_iterable_queue()


def main(_):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    os.makedirs(FLAGS.dataset_path, exist_ok=True)
    assert len(os.listdir(FLAGS.dataset_path)) == 0, f'dataset_path directory ({FLAGS.dataset_path}) must be empty.'

    gin_config_str = gin.config_str(max_line_length=120)

    print("Loaded configuration:")
    print(gin_config_str)

    with open(os.path.join(FLAGS.dataset_path, 'config.gin'), mode='w') as f:
        f.write(gin_config_str)

    filter_replay = gin.get_configurable(FilterReplay)()
    process_replay = gin.get_configurable(ProcessReplay)()
    store_replay = gin.get_configurable(StoreReplay)(dataset_path=FLAGS.dataset_path)

    dataset_files = []
    for dataset_file in tqdm(
            find_replays(FLAGS.replays_path)
            | pl.process.flat_map(load_replay_meta, workers=FLAGS.num_workers, maxsize=0)
            | pl.process.filter(filter_replay, workers=1, maxsize=0)
            | pl.process.map(process_replay, workers=FLAGS.num_workers, maxsize=FLAGS.num_workers)
            | pl.process.filter(is_not_none, workers=1, maxsize=0)
            | pl.process.map(store_replay, workers=FLAGS.num_workers, maxsize=0)
    ):
        dataset_files.append(dataset_file)


if __name__ == '__main__':
    app.run(main)
