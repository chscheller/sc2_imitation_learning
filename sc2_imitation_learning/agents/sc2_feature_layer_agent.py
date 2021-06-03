from typing import Tuple, List, Text, Union, Optional, Dict

import gin
import sonnet as snt
import tensorflow as tf

from sc2_imitation_learning.agents import Agent, AgentOutput
from sc2_imitation_learning.agents.common.policy_head import AutoregressivePolicyHead, PolicyContextFeatures
from sc2_imitation_learning.agents.common.scalar_encoder import ScalarEncoder
from sc2_imitation_learning.agents.common.spatial_encoder import SpatialEncoder
from sc2_imitation_learning.agents.common.unit_group_encoder import UnitGroupEncoder


class SC2FeatureLayerAgentHead(snt.Module):
    def __init__(self,
                 autoregressive_embed_dim: int,
                 policy_heads: List[AutoregressivePolicyHead],
                 name: Optional[Text] = None):
        super().__init__(name)
        self._core_outputs_embed = snt.Linear(autoregressive_embed_dim)  # todo glu!
        self._policy_heads = policy_heads

    def __call__(self,
                 core_outputs: tf.Tensor,
                 scalar_context: tf.Tensor,
                 unit_groups: Dict[str, tf.Tensor],
                 available_actions: tf.Tensor,
                 screen_skip: List[tf.Tensor],
                 minimap_skip: List[tf.Tensor],
                 teacher_actions: Optional[Dict[str, tf.Tensor]] = None) -> AgentOutput:

        context = PolicyContextFeatures(
            scalar_context=scalar_context,
            unit_groups=unit_groups,
            available_actions=available_actions,
            map_skip={
                'screen': screen_skip,
                'minimap': minimap_skip
            })

        autoregressive_embedding = self._core_outputs_embed(core_outputs)

        action, logits = {}, {}
        for policy_head in self._policy_heads:
            poliy_outputs, autoregressive_embedding = policy_head(
                autoregressive_embedding=autoregressive_embedding,
                context=context,
                partial_action=action if teacher_actions is None else teacher_actions,
                teacher_action=None if teacher_actions is None else teacher_actions[policy_head.action_name])
            action[policy_head.action_name] = poliy_outputs.action
            logits[policy_head.action_name] = poliy_outputs.logits

        return AgentOutput(logits=logits, actions=action, values=None)


@gin.register
class SC2FeatureLayerAgent(Agent):
    """ An agent that operates on scalar features and spatial feature layers as sensory inputs. """

    def __init__(self,
                 scalar_encoder: ScalarEncoder = gin.REQUIRED,
                 unit_group_encoder: Optional[UnitGroupEncoder] = None,
                 screen_encoder: SpatialEncoder = gin.REQUIRED,
                 minimap_encoder: SpatialEncoder = gin.REQUIRED,
                 core: Union[tf.keras.layers.LSTMCell, tf.keras.layers.StackedRNNCells] = gin.REQUIRED,
                 autoregressive_embed_dim: int = gin.REQUIRED,
                 policy_heads: List[AutoregressivePolicyHead] = gin.REQUIRED,
                 ):
        """ Constructs a feature layer agent.

        Args:
            scalar_encoder: A scalar encoder module.
            unit_group_encoder: A unit group encoder module.
            screen_encoder: A screen encoder module.
            minimap_encoder:  A minimap encoder module.
            core: An LSTM core module. Both single layer (LSTMCell) and deep (StackedRNNCells) LSTMs are supported.
            autoregressive_embed_dim: The size of the autoregressive embedding vector used during action decoding.
            policy_heads: A list of autoregressive policy heads. IMPORTANT: ordering matters! The actions are decoded
                exactly in the provided order.
        """
        super().__init__()
        self._scalar_encoder = snt.BatchApply(scalar_encoder)
        if unit_group_encoder is not None:
            self._unit_group_encoder = snt.BatchApply(unit_group_encoder)
        else:
            self._unit_group_encoder = None
        self._screen_encoder = snt.BatchApply(screen_encoder)
        self._minimap_encoder = snt.BatchApply(minimap_encoder)
        self._core = core
        self._head = snt.BatchApply(SC2FeatureLayerAgentHead(autoregressive_embed_dim, policy_heads))

    def initial_state(self, batch_size: int, **kwargs):
        if isinstance(self._core, snt.RNNCore):
            return self._core.initial_state(batch_size=batch_size, **kwargs)
        else:
            return self._core.get_initial_state(batch_size=batch_size, dtype=tf.float32)

    def _unroll(self, prev_actions, env_outputs, core_state, teacher_actions=None) -> Tuple[AgentOutput, Tuple]:
        rewards, done, observations = env_outputs

        embedded_scalar, scalar_context = self._scalar_encoder(
            features=observations['scalar_features'],
            prev_actions=prev_actions)

        if self._unit_group_encoder is not None:
            _, unit_groups = self._unit_group_encoder(features=observations['scalar_features'])
        else:
            unit_groups = dict()

        embedded_screen, screen_skip = self._screen_encoder(features=observations['screen_features'])

        embedded_minimap, minimap_skip = self._minimap_encoder(features=observations['minimap_features'])

        embedded_observations = tf.concat(values=[embedded_scalar, embedded_screen, embedded_minimap], axis=-1)

        initial_core_state = self.initial_state(batch_size=tf.shape(done)[1])

        core_output_list = []
        for input_, d in zip(tf.unstack(embedded_observations), tf.unstack(done)):
            # If the episode ended, the core state should be reset before the next.
            core_state = tf.nest.map_structure(
                lambda x, y, d=d: tf.where(tf.reshape(d, [tf.shape(d)[0]] + [1] * (x.shape.rank - 1)), x, y),
                initial_core_state,
                core_state)
            core_output, core_state = self._core(input_, core_state)
            core_output_list.append(core_output)

        core_outputs = tf.stack(core_output_list)

        agent_outputs: AgentOutput = self._head(
            core_outputs=core_outputs,
            scalar_context=scalar_context,
            unit_groups=unit_groups,
            available_actions=observations['scalar_features']['available_actions'],
            screen_skip=screen_skip,
            minimap_skip=minimap_skip,
            teacher_actions=teacher_actions)

        return agent_outputs, core_state
