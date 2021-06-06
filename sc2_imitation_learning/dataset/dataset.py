import functools
import json
import logging
import multiprocessing
import os
import pickle
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Iterator
from typing import NamedTuple, Dict, Optional, Sequence

import gym
import h5py
import numpy as np
import tensorflow as tf
import tqdm

from sc2_imitation_learning.common.utils import unflatten_nested_dicts, flatten_nested_dicts, load_json
from sc2_imitation_learning.environment.environment import ObservationSpace, ActionSpace

logger = logging.getLogger(__name__)


class ActionTimeStep(NamedTuple):
    observation: Dict
    action: Dict
    reward: float
    done: bool


class EpisodeSlice(NamedTuple):
    episode_id: int
    episode_path: str
    start: int
    length: int
    wrap_at_end: bool


class EpisodeIterator(Iterator):
    def __init__(self, episode_id: int, episode_path: str, episode_length: int, sequence_length: int,
                 start_index: int = 0) -> None:
        super().__init__()
        self._episode_id = episode_id
        self._episode_path = episode_path
        self._episode_length = episode_length
        self._sequence_length = sequence_length
        self._index = start_index

    def __next__(self) -> EpisodeSlice:
        episode_slice = EpisodeSlice(
            episode_id=self._episode_id, episode_path=self._episode_path, start=self._index,
            length=self._sequence_length, wrap_at_end=True)
        self._index = (self._index + self._sequence_length) % self._episode_length
        return episode_slice


class Batcher(object):
    def __init__(self, batch_size: int, sequence_length: int, max_queue_size: int, seed: Optional[int] = None) -> None:
        super().__init__()
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._queue_out = multiprocessing.Queue(maxsize=max_queue_size)
        self._seed = seed

    def __call__(self, file_paths: Sequence[str]) -> Iterator[List[EpisodeSlice]]:
        process = multiprocessing.Process(target=self._run, args=(file_paths,), daemon=True)
        try:
            process.start()
            while True:
                yield self._queue_out.get()
        finally:
            process.terminate()

    def _run(self, file_paths: Sequence[str]):
        rng = random.Random(self._seed)
        episode_iterators = []
        for i, path in enumerate(tqdm.tqdm(file_paths, total=len(file_paths))):
            # with open(path.replace('.hdf5', '.pkl'), mode='rb') as f:
            #     meta = pickle.load(f)
            with open(path.replace('.hdf5', '.meta'), mode='r') as f:
                meta = json.load(f)
            if meta['episode_length'] >= self._sequence_length:
                episode_iterators.append(EpisodeIterator(
                    i, path, meta['episode_length'], self._sequence_length, rng.randint(0, meta['episode_length'] - 1)))
        while True:
            batch_episodes = rng.sample(episode_iterators, k=self._batch_size)
            batch = [next(it) for it in batch_episodes]
            self._queue_out.put(batch)


def h5py_dataset_iterator(g, prefix=None):
    for key in g.keys():
        item = g[key]
        path = key
        if prefix is not None:
            path = f'{prefix}/{path}'
        if isinstance(item, h5py.Dataset):  # test for dataset
            yield path, item
        elif isinstance(item, h5py.Group):  # test for group (go down)
            yield from h5py_dataset_iterator(item, path)


def load_episode_slice(episode_slice: EpisodeSlice) -> Tuple[int, Dict]:
    with h5py.File(episode_slice.episode_path, 'r') as f:
        episode_length = f['reward'].shape[0]
        if episode_slice.wrap_at_end and episode_slice.start + episode_slice.length > episode_length:
            sequence = {
                key: np.concatenate([
                    dataset[episode_slice.start:],
                    dataset[:(episode_slice.start + episode_slice.length) % episode_length]
                ], axis=0)
                for key, dataset in h5py_dataset_iterator(f)}
        else:
            sequence = {
                key: dataset[episode_slice.start:episode_slice.start+episode_slice.length]
                for key, dataset in h5py_dataset_iterator(f)}
    for k, v in sequence.items():
        assert v.shape[0] == episode_slice.length, f"{k}, {v.shape}, {episode_length}, {episode_slice}"
    return episode_slice.episode_id, unflatten_nested_dicts(sequence)


def load_batch(semaphore, batch: List[EpisodeSlice]) -> Dict:
    semaphore.acquire()
    batch = map(load_episode_slice, batch)
    return tf.nest.map_structure(lambda *x: np.stack(x), *batch)


def load_dataset_from_hdf5(file_paths: Sequence[str],
                           specs: Dict[str, gym.spaces.Space],
                           batch_size: int,
                           sequence_length: int,
                           num_workers: int = os.cpu_count(),
                           chunk_size: int = 4,
                           seed: Optional[int] = None) -> tf.data.Dataset:

    output_types = (tf.int32, unflatten_nested_dicts(tf.nest.map_structure(lambda s: s.dtype, specs)))
    output_shapes = ((batch_size,), unflatten_nested_dicts(tf.nest.map_structure(
        lambda s: tf.TensorShape([batch_size, sequence_length]).concatenate(s.shape), specs)))

    def _gen():
        batcher = Batcher(batch_size, sequence_length=sequence_length, max_queue_size=num_workers, seed=seed)
        manager = multiprocessing.Manager()
        load_batch_semaphore = manager.Semaphore(2*chunk_size*num_workers)
        load_batch_ = functools.partial(load_batch, load_batch_semaphore)
        with multiprocessing.Pool(processes=num_workers) as pool:
            for batch in pool.imap(load_batch_, batcher(file_paths=file_paths), chunksize=chunk_size):
                yield batch
                load_batch_semaphore.release()
    return tf.data.Dataset.from_generator(_gen, args=[], output_types=output_types, output_shapes=output_shapes)


def get_dataset_specs(action_space: ActionSpace, observation_space: ObservationSpace):
    return {
        **{
            f'observation/{key}': space
            for key, space in flatten_nested_dicts(observation_space.specs).items()
        },
        **{
            f'action/{key}': space
            for key, space in flatten_nested_dicts(action_space.specs).items()
        },
        **{
            f'prev_action/{key}': space
            for key, space in flatten_nested_dicts(action_space.specs).items()
        },
        'reward': gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
        'prev_reward': gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
        'done': gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.bool),
    }


def store_episode_to_hdf5(path: str,
                          name: str,
                          episode: List[ActionTimeStep],
                          episode_info: Dict,
                          specs: Dict[str, gym.spaces.Space]) -> str:
    os.makedirs(path, exist_ok=True)

    assert os.path.exists(os.path.join(path, f'{name}.hdf5')) is False, \
        f"'{name}.hdf5' already exists in '{path}'."

    with h5py.File(os.path.join(path, f'{name}.hdf5'), mode='w') as f:
        datasets = {
            key: f.create_dataset(name=key, shape=(len(episode),) + space.shape, dtype=space.dtype)
            for key, space in specs.items()
        }
        for i, time_step in enumerate(episode):
            for key, value in flatten_nested_dicts(time_step.observation).items():
                datasets[f'observation/{key}'][i] = np.asarray(value)
            for key, value in flatten_nested_dicts(time_step.action).items():
                datasets[f'action/{key}'][i] = np.asarray(value)
            for key, value in flatten_nested_dicts(time_step.action).items():
                datasets[f'prev_action/{key}'][i] = -1 if i == 0 else datasets[f'action/{key}'][i - 1]
            datasets['reward'][i] = time_step.reward
            datasets['prev_reward'][i] = 0. if i == 0 else datasets['reward'][i - 1]
            datasets['done'][i] = time_step.done

    with open(os.path.join(path, f'{name}.meta'), mode='w') as f:
        json.dump({
            'data_file': f'{name}.hdf5',
            'episode_return': sum([float(time_step.reward) for time_step in episode]),
            'episode_length': len(episode),
            'episode_info': episode_info
        }, f, indent=4)

    return os.path.join(path, f'{name}')


class DataLoader(ABC):
    @property
    @abstractmethod
    def num_samples(self) -> int:
        pass

    @property
    @abstractmethod
    def num_episodes(self) -> int:
        pass

    @abstractmethod
    def load(self,
             batch_size: int,
             sequence_length: int,
             offset_episodes: int = 0,
             num_episodes: int = 0,
             num_workers: int = os.cpu_count(),
             chunk_size: int = 4,
             seed: Optional[int] = None) -> tf.data.Dataset:
        pass
