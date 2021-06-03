import glob
import glob
import logging
import multiprocessing
import os
from typing import Optional
from typing import Set, AbstractSet

import gin
import tensorflow as tf
from pysc2 import run_configs
from pysc2.env.sc2_env import Race

from sc2_imitation_learning.common.utils import load_json
from sc2_imitation_learning.dataset.dataset import DataLoader, get_dataset_specs, load_dataset_from_hdf5
from sc2_imitation_learning.environment.sc2_environment import SC2ActionSpace, SC2ObservationSpace, SC2Maps

logger = logging.getLogger(__name__)


SC2REPLAY_RACES = {
    'Prot': Race.protoss,
    'Terr': Race.terran,
    'Zerg': Race.zerg
}


@gin.register
class SC2DataLoader(DataLoader):
    def __init__(self,
                 path: str,
                 action_space: SC2ActionSpace,
                 observation_space: SC2ObservationSpace,
                 min_duration: float = 0.,
                 min_mmr: int = 0,
                 min_apm: int = 0,
                 observed_player_races: AbstractSet[Race] = frozenset((Race.protoss, Race.terran, Race.zerg)),
                 opponent_player_races: AbstractSet[Race] = frozenset((Race.protoss, Race.terran, Race.zerg)),
                 map_names: Optional[Set[str]] = None) -> None:
        super().__init__()
        assert os.path.isdir(path), f"Not a valid dataset path: '{path}'"

        sc2_maps = SC2Maps(run_configs.get().data_dir)

        def filter_replay_info(episode_info):
            replay_info = episode_info['replay_info']
            observed_player_info = next(
                filter(lambda p: p['PlayerID'] == episode_info['observed_player_id'], replay_info['Players']))
            if len(replay_info['Players']) > 1:
                opponent_player_info = next(
                    filter(lambda p: p['PlayerID'] != episode_info['observed_player_id'], replay_info['Players']))
            return (replay_info['Duration'] >= min_duration
                    and observed_player_info.get('MMR', 0) >= min_mmr
                    and observed_player_info['APM'] >= min_apm
                    and SC2REPLAY_RACES[observed_player_info['AssignedRace']] in observed_player_races
                    and (len(replay_info['Players']) == 1 or
                         SC2REPLAY_RACES[opponent_player_info['AssignedRace']] in opponent_player_races)
                    and (map_names is None or sc2_maps.normalize_map_name(replay_info['Title']) in map_names))

        with multiprocessing.Pool() as p:
            meta_infos = p.map(load_json, glob.glob(os.path.join(path, '*.meta')))

        logger.info(f"Found {len(meta_infos)} episodes.")

        meta_infos = [meta_info for meta_info in meta_infos if filter_replay_info(meta_info['episode_info'])]

        logger.info(f"Filtered {len(meta_infos)} episodes (Filter: "
                    f"min_duration={min_duration}, "
                    f"min_mmr={min_mmr}, "
                    f"min_apm={min_apm}, "
                    f"observed_player_races={list(observed_player_races)}, "
                    f"opponent_player_races={list(opponent_player_races)}, "
                    f"map_names={map_names if map_names is None else list(map_names)}).")

        self._file_paths = [os.path.join(path, meta_info['data_file']) for meta_info in meta_infos]
        self._num_samples = sum([meta_info['episode_length'] for meta_info in meta_infos])
        self._num_episodes = len(self._file_paths)

        assert self._num_episodes > 0, "Empty dataset"

        logger.info(f"Loaded dataset with {self._num_episodes} episodes ({self._num_samples} samples).")

        self._specs = get_dataset_specs(action_space=action_space, observation_space=observation_space)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def num_episodes(self) -> int:
        return self._num_episodes

    def load(self,
             batch_size: int,
             sequence_length: int,
             offset_episodes: int = 0,
             num_episodes: int = 0,
             num_workers: int = os.cpu_count(),
             chunk_size: int = 4,
             seed: Optional[int] = None) -> tf.data.Dataset:
        if num_episodes > 0:
            file_paths = self._file_paths[offset_episodes:offset_episodes+num_episodes]
        else:
            file_paths = self._file_paths[offset_episodes:]
        return load_dataset_from_hdf5(
            file_paths, self._specs, batch_size, sequence_length, num_workers, chunk_size, seed)
