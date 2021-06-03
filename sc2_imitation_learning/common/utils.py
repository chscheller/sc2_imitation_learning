import collections
import functools
import json
import logging
import re
import traceback
from typing import Iterable, Callable, Type, Union

import gin
import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow_addons as tfa
import yaml

from sc2_imitation_learning.common.types import ShapeLike
from sc2_imitation_learning.environment.environment import ObservationSpace, ActionSpace, Space


def swap_leading_axes(tensor: tf.Tensor) -> tf.Tensor:
    return tf.transpose(tensor, perm=[1, 0] + list(range(2, len(tensor.get_shape()))))


def prepend_leading_dims(shape: ShapeLike, leading_dims: ShapeLike) -> tf.TensorShape:
    return tf.TensorShape(leading_dims).concatenate(shape)


def flatten_nested_dicts(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dicts(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_nested_dicts(d, sep='/'):
    out_dict = {}
    for k, v in d.items():
        dict_pointer = out_dict
        key_path = list(k.split(sep))
        if len(key_path) > 1:
            for sub_k in key_path[:-1]:
                if sub_k not in dict_pointer:
                    dict_pointer[sub_k] = {}
                dict_pointer = dict_pointer[sub_k]
        dict_pointer[key_path[-1]] = v
    return out_dict


class Aggregator(tf.Module):
    """Utility module for keeping state and statistics for individual actors.
    Copied from:
    https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/common/utils.py"""

    def __init__(self, num_actors, specs, name='Aggregator'):
        """Inits an Aggregator.

        Args:
          num_actors: int, number of actors.
          specs: Structure (as defined by tf.nest) of tf.TensorSpecs that will be
            stored for each actor.
          name: Name of the scope for the operations.
        """
        super(Aggregator, self).__init__(name=name)

        def create_variable(spec):
            z = tf.zeros([num_actors] + spec.shape.dims, dtype=spec.dtype)
            return tf.Variable(z, trainable=False, name=spec.name)

        self._state = tf.nest.map_structure(create_variable, specs)

    @tf.Module.with_name_scope
    def reset(self, actor_ids):
        """Fills the tensors for the given actors with zeros."""
        with tf.name_scope('Aggregator_reset'):
            for s in tf.nest.flatten(self._state):
                s.scatter_update(tf.IndexedSlices(0, actor_ids))

    @tf.Module.with_name_scope
    def add(self, actor_ids, values):
        """In-place adds values to the state associated to the given actors.

        Args:
          actor_ids: 1D tensor with the list of actor IDs we want to add values to.
          values: A structure of tensors following the input spec, with an added
            first dimension that must either have the same size as 'actor_ids', or
            should not exist (in which case, the value is broadcasted to all actor
            ids).
        """
        tf.nest.assert_same_structure(values, self._state)
        for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
            s.scatter_add(tf.IndexedSlices(v, actor_ids))

    @tf.Module.with_name_scope
    def read(self, actor_ids):
        """Reads the values corresponding to a list of actors.

        Args:
          actor_ids: 1D tensor with the list of actor IDs we want to read.

        Returns:
          A structure of tensors with the same shapes as the input specs. A
          dimension is added in front of each tensor, with size equal to the number
          of actor_ids provided.
        """
        return tf.nest.map_structure(lambda s: s.sparse_read(actor_ids),
                                     self._state)

    @tf.Module.with_name_scope
    def replace(self, actor_ids, values):
        """Replaces the state associated to the given actors.

        Args:
          actor_ids: 1D tensor with the list of actor IDs.
          values: A structure of tensors following the input spec, with an added
            first dimension that must either have the same size as 'actor_ids', or
            should not exist (in which case, the value is broadcasted to all actor
            ids).
        """
        tf.nest.assert_same_structure(values, self._state)
        for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
            s.scatter_update(tf.IndexedSlices(v, actor_ids))


def retry(max_tries: int, exceptions: Iterable[Type[Exception]] = (Exception,), exception_on_failure: bool = False):
    def wrapped(fn: Callable):
        @functools.wraps(fn)
        def _retry(*args, **kwargs):
            num_tries = 0
            while num_tries < max_tries:
                try:
                    return fn(*args, **kwargs)
                except tuple(exceptions) as e:
                    logging.warning(f"Failed to call '{fn.__name__}': {e}\n{traceback.format_exc()}")
                    num_tries += 1
            logging.error(f"Failed to call '{fn.__name__}', retried {num_tries} of {max_tries} times.")
            if exception_on_failure:
                raise RuntimeError(f"Failed to call '{fn.__name__}', retried {num_tries} of {max_tries} times.")
            return None
        return _retry
    return wrapped


def positional_encoding(max_position, embedding_size, add_batch_dim=False):
    positions = np.arange(max_position)
    angle_rates = 1 / np.power(10000, (2 * (np.arange(embedding_size)//2)) / np.float32(embedding_size))
    angle_rads = positions[:, np.newaxis] * angle_rates[np.newaxis, :]

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    if add_batch_dim:
        angle_rads = angle_rads[np.newaxis, ...]

    return tf.cast(angle_rads, dtype=tf.float32)


def make_dummy(space: Space, num_batch_dims: int = 1):
    return tf.nest.map_structure(lambda s: tf.zeros((1,)*num_batch_dims + s.shape, s.dtype), space.specs)


def make_dummy_action(action_space: ActionSpace, num_batch_dims: int = 1):
    return make_dummy(action_space, num_batch_dims)


def make_dummy_observation(observation_space: ObservationSpace, num_batch_dims: int = 1):
    return make_dummy(observation_space, num_batch_dims)


def make_dummy_batch(observation_space: ObservationSpace, action_space: ActionSpace, num_batch_dims: int = 2):
    prev_actions = make_dummy_action(action_space, num_batch_dims=num_batch_dims)
    rewards = tf.zeros((1,)*num_batch_dims, dtype=tf.float32)
    dones = tf.zeros((1,)*num_batch_dims, dtype=tf.bool)
    observations = make_dummy_observation(observation_space, num_batch_dims=num_batch_dims)
    return prev_actions, (rewards, dones, observations)


def compute_stats_dict(samples: Union[list, np.ndarray]):
    conf_int = scipy.stats.t.interval(0.95, len(samples)-1, loc=np.mean(samples), scale=scipy.stats.sem(samples))
    return {
        'samples': [float(x) for x in samples],
        'mean': float(np.mean(samples)),
        'median': float(np.median(samples)),
        'std': float(np.std(samples)),
        'min': float(min(samples)),
        'max': float(max(samples)),
        'mean_ci_95': [float(x) for x in conf_int],
    }


def gin_register_external_configurables():
    gin.external_configurable(tf.nn.relu, 'tf.nn.relu')
    gin.external_configurable(tf.keras.layers.LSTMCell, 'tf.keras.layers.LSTMCell')
    gin.external_configurable(tf.keras.layers.StackedRNNCells, 'tf.keras.layers.StackedRNNCells')
    gin.external_configurable(tfa.rnn.LayerNormLSTMCell, 'tfa.rnn.LayerNormLSTMCell')
    gin.external_configurable(tf.keras.optimizers.Adam, 'tf.keras.optimizers.Adam')
    gin.external_configurable(tf.keras.optimizers.schedules.PolynomialDecay,
                              'tf.keras.optimizers.schedules.PolynomialDecay')
    gin.external_configurable(tf.keras.optimizers.schedules.ExponentialDecay,
                              'tf.keras.optimizers.schedules.ExponentialDecay')


def gin_config_str_to_dict(gin_config_str: str) -> dict:
    gin_config_str = "\n".join([x for x in gin_config_str.split("\n") if not x.startswith("import")])
    gin_config_str = re.compile(r"\\\n[^\S\r\n]+").sub(' ', gin_config_str)  # collapse indented newlines to single line
    gin_config_str = gin_config_str.replace("@", "").replace(" = %", ": ").replace(" = ", ": ")
    gin_config_dict = yaml.safe_load(gin_config_str)
    return unflatten_nested_dicts(gin_config_dict)


def load_json(file_path: str):
    with open(file_path) as f:
        return json.load(f)
