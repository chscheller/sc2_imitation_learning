import collections
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Text

import sonnet as snt
import tensorflow as tf
import tree
from sonnet.src import types

from sc2_imitation_learning.environment.environment import ActionSpace, ObservationSpace

AgentOutput = collections.namedtuple('AgentOutput', ['logits', 'actions', 'values'])


class Agent(snt.RNNCore, ABC):
    def __call__(self, prev_actions, env_outputs, core_state, unroll=False, teacher_actions=None) -> Tuple[AgentOutput, Tuple]:
        if not unroll:
            # Add time dimension.
            prev_actions, env_outputs = tf.nest.map_structure(
                lambda t: tf.expand_dims(t, 0), (prev_actions, env_outputs))

        outputs, core_state = self._unroll(prev_actions, env_outputs, core_state, teacher_actions)

        if not unroll:
            # Remove time dimension.
            outputs = tf.nest.map_structure(lambda t: None if t is None else tf.squeeze(t, 0), outputs)

        return outputs, core_state

    @abstractmethod
    def _unroll(self, prev_actions, env_outputs, core_state, teacher_actions=None) -> Tuple[AgentOutput, Tuple]:
        pass


def build_saved_agent(agent: Agent, observation_space: ObservationSpace, action_space: ActionSpace) -> tf.Module:
    call_input_signature = [
        tree.map_structure_with_path(
            lambda path, s: tf.TensorSpec((None,) + s.shape, s.dtype, name='action/' + '/'.join(path)),
            action_space.specs),
        (
            tf.TensorSpec((None,), dtype=tf.float32, name='reward'),
            tf.TensorSpec((None,), dtype=tf.bool, name='done'),
            tree.map_structure_with_path(
                lambda path, s: tf.TensorSpec((None,) + s.shape, s.dtype, name='observation/' + '/'.join(path)),
                observation_space.specs)
        ),
        tree.map_structure(
            lambda t: tf.TensorSpec((None,) + t.shape[1:], t.dtype, name='agent_state'), agent.initial_state(1))
    ]

    initial_state_input_signature = [
        tf.TensorSpec(shape=(), dtype=tf.int32, name='batch_size'),
    ]

    class SavedAgent(tf.Module):
        def __init__(self, agent: Agent, name=None):
            super().__init__(name)
            self._agent = agent

        @tf.function(input_signature=call_input_signature)
        def __call__(self, prev_action, env_outputs, agent_state):
            return self._agent(prev_action, env_outputs, agent_state)

        @tf.function(input_signature=initial_state_input_signature)
        def initial_state(self, batch_size):
            return self._agent.initial_state(batch_size)

    return SavedAgent(agent)
