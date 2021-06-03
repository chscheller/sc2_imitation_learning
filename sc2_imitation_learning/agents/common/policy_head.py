from abc import abstractmethod, ABC
from typing import Tuple, Sequence, List, Text, Callable, Union, Optional, Dict, NamedTuple, Type

import gin
import pysc2.lib.actions
import sonnet as snt
import tensorflow as tf

from sc2_imitation_learning.agents.common.spatial_decoder import SpatialDecoder
from sc2_imitation_learning.common.conv import ConvNet2DTranspose
from sc2_imitation_learning.common.mlp import MLP, ResMLP
from sc2_imitation_learning.environment.sc2_environment import SC2ActionSpace


class PolicyContextFeatures(NamedTuple):
    scalar_context: tf.Tensor
    unit_groups: Dict[str, tf.Tensor]
    available_actions: tf.Tensor
    map_skip: Dict[str, List[tf.Tensor]]


class PolicyHeadOutputs(NamedTuple):
    action: tf.Tensor
    logits: tf.Tensor


class PolicyHead(snt.Module, ABC):
    """ A policy head module for categorical actions that produces logits and action samples from a latent
    representation."""

    def __init__(self, num_actions: int, name: Optional[Text] = None):
        """ Constructs the policy head.

        Args:
            num_actions: A scalar that defines the number of categorical actions.
            name: An optional module name.
        """
        super().__init__(name)
        self._num_actions = num_actions

    @property
    def num_actions(self):
        return self._num_actions

    def __call__(self, inputs: tf.Tensor, context: PolicyContextFeatures) -> PolicyHeadOutputs:
        """ Applies the policy head to provided inputs and policy context.

        Args:
            inputs: A 1D tensor from which logits and action samples are calculated.
            context: `PolicyContextFeatures` that contain context information used by certain policy heads.

        Returns:
            A `PolicyHeadOutputs` tuple with sampled action and logits.
        """
        logits = self._compute_logits(inputs, context)
        if logits.shape.rank > 2:
            action = snt.BatchApply(lambda t: tf.random.categorical(logits=t, num_samples=1, dtype=tf.int32))(logits)
        else:
            action = tf.random.categorical(logits=logits, num_samples=1, dtype=tf.int32)
        return PolicyHeadOutputs(action=tf.squeeze(action, axis=-1), logits=logits)

    @abstractmethod
    def _compute_logits(self, inputs: tf.Tensor, context: PolicyContextFeatures) -> tf.Tensor:
        pass


@gin.register()
class ActionTypePolicyHead(PolicyHead):
    """ A policy head for action types that supports masking of invalid actions.
    Inputs are first processed by a decoder network and subsequently projected to a 1D logits tensor with the size
    of `num_actions`. """

    def __init__(self,
                 num_actions: int,
                 # output_size: int = gin.REQUIRED,
                 # num_blocks: int = gin.REQUIRED,
                 # with_layer_norm: bool = gin.REQUIRED,
                 decoder: snt.Module = gin.REQUIRED,
                 name: Optional[Text] = None):
        """ Constructs the action type policy head module.

        Args:
            num_actions: A scalar that defines the number of categorical actions.
            output_size: A scalar that defines the hidden size of each ResBlock.
            num_blocks: A scalar that defines the number of ResBlocks.
            with_layer_norm: A boolean that defines whether ResBlocks use layer norm.
            name: An optional module name.
        """
        super().__init__(num_actions, name)
        self._policy_head = snt.Sequential([
            decoder,
            # snt.Linear(output_size=output_size),
            # ResMLP(output_size=output_size, num_blocks=num_blocks, with_layer_norm=with_layer_norm,
            #        activate_final=True),
            snt.Linear(output_size=self.num_actions)])

    def _compute_logits(self, inputs: tf.Tensor, context: PolicyContextFeatures) -> tf.Tensor:
        logits = self._policy_head(inputs)
        logits_mask = tf.cast(context.available_actions, logits.dtype)
        logits = (logits * logits_mask) + (tf.abs(logits_mask - 1) * logits.dtype.min)
        return logits


@gin.register()
class ScalarPolicyHead(PolicyHead):
    """ A policy head for categorical actions that operates on 1D inputs.
    Inputs are first processed by a decoder network and subsequently projected to a 1D logits tensor with the size
    of `num_actions`. """

    def __init__(self,
                 num_actions: int,
                 # output_sizes: Sequence[int] = gin.REQUIRED,
                 # with_layer_norm: bool = gin.REQUIRED,
                 decoder: snt.Module = gin.REQUIRED,
                 name: Optional[Text] = None):
        """ Constructs the scalar policy head module.

        Args:
            num_actions: A scalar that defines the number of categorical actions.
            output_sizes: A sequence of scalars that defines the number of hidden units in each linear layer.
            with_layer_norm: A boolean that defines whether ResBlocks use layer norm.
            name: An optional module name.
        """
        super().__init__(num_actions, name)
        self._policy_head = snt.Sequential([
            # MLP(output_sizes=output_sizes, with_layer_norm=with_layer_norm, activate_final=True),
            decoder,
            snt.Linear(output_size=self.num_actions)])

    def _compute_logits(self, inputs: tf.Tensor, context: PolicyContextFeatures) -> tf.Tensor:
        return self._policy_head(inputs)


@gin.register()
class SpatialPolicyHead(PolicyHead):
    """ A policy head for categorical actions that operates on 2D inputs.
    Input tensors, together with the tensors in map_skip, are first processed by a spatial decoder. The decoder outputs
    are then upsampled by a sequence of transpose convolutions and flattened to produce a 1D logits tensor. """

    def __init__(self,
                 num_actions: int,
                 decoder: SpatialDecoder = gin.REQUIRED,
                 upsample_conv_net: ConvNet2DTranspose = gin.REQUIRED,
                 map_skip: str = gin.REQUIRED,
                 name: Optional[Text] = None):
        """ Constructs a spatial policy head module.

        Args:
            num_actions: A scalar that defines the number of categorical actions.
            decoder: A `SpatialDecoder` module that encodes inputs together with `map_skip` context features.
            upsample_conv_net: A `ConvNet2DTranspose` module that upsamples decoder outputs to the final spatial shape.
            map_skip: The name of the map_skip to be used by the decoder (either "screen" or "minimap")
            name: An optional module name.
        """
        super().__init__(num_actions, name)
        self._map_skip = map_skip
        self._decoder = decoder
        self._upsample_conv_net = upsample_conv_net
        self._logits_projection = snt.Sequential([
            snt.Conv2D(output_channels=1, kernel_shape=1, stride=1),
            snt.Flatten()])

    def _compute_logits(self, inputs: tf.Tensor, context: PolicyContextFeatures) -> tf.Tensor:
        map_skip = context.map_skip[self._map_skip]
        decoder_out = self._decoder(inputs, map_skip)
        upsample_out = self._upsample_conv_net(decoder_out)
        logits = self._logits_projection(upsample_out)
        return logits


@gin.register()
class UnitGroupPointerPolicyHead(PolicyHead):
    """ A policy head for pointer actions into unit groups (e.g. control groups, multi select, ...). """

    def __init__(self,
                 num_actions: int,
                 query_embedding_output_sizes: List[int] = gin.REQUIRED,
                 key_embedding_output_sizes: List[int] = gin.REQUIRED,
                 target_group: Text = gin.REQUIRED,
                 mask_zeros: bool = False,
                 name: Optional[Text] = None):
        super().__init__(num_actions, name)
        self._target_group = target_group
        self._mask_zeros = mask_zeros
        self._query_embed = MLP(output_sizes=query_embedding_output_sizes)
        self._key_embed = MLP(output_sizes=key_embedding_output_sizes)
        # self._v = snt.Linear(output_size=1, with_bias=False)

    def _compute_logits(self, inputs: tf.Tensor, context: PolicyContextFeatures) -> tf.Tensor:
        unit_group = context.unit_groups[self._target_group]  # B x T x E
        unit_group_length = context.unit_groups.get(f'{self._target_group}_length', None)  # B x 1
        query_embedding = self._query_embed(inputs)  # B x K
        key_embeddings = self._key_embed(unit_group)  # B x T x K

        # Additive attention (Bahdanau2015):
        # logits = self._v(tf.nn.tanh(tf.expand_dims(query_embedding, axis=1) + key_embeddings))  # B x T x 1
        # logits = tf.squeeze(logits, axis=-1)  # B x T

        # Dot-Product attention (Luong2015):
        logits = tf.reduce_sum(tf.expand_dims(query_embedding, axis=1) * key_embeddings, axis=-1)  # B x T

        if self._mask_zeros:
            sequence_mask = tf.cast(tf.not_equal(tf.reduce_sum(unit_group, axis=-1), 0.), dtype=logits.dtype)  # B x T
            logits = (logits * sequence_mask) + (tf.abs(sequence_mask - 1) * logits.dtype.min)  # B x T

        if unit_group_length is not None:
            if unit_group.shape.rank - unit_group_length.shape.rank < 2:
                unit_group_length = tf.squeeze(unit_group_length, axis=-1)  # B
            sequence_mask = tf.sequence_mask(
                unit_group_length, maxlen=tf.shape(key_embeddings)[1], dtype=logits.dtype)  # B x T
            logits = (logits * sequence_mask) + (tf.abs(sequence_mask - 1) * logits.dtype.min)  # B x T

        return logits


@gin.register()
class ActionEmbedding(snt.Module, ABC):
    """ An action embedding module that embeds categorical actions into continuous space by one-hot encoding followed
    by an MLP. """

    def __init__(self,
                 num_actions: int,
                 output_sizes: Sequence[int] = gin.REQUIRED,
                 activation: Union[Callable[[tf.Tensor], tf.Tensor], Text] = tf.nn.relu,
                 with_layer_norm: bool = False,
                 name: Optional[Text] = None):
        super().__init__(name)
        self._num_actions = num_actions
        self._mlp = MLP(
            output_sizes=output_sizes,
            with_layer_norm=with_layer_norm,
            activation=activation,
            activate_final=True)

    def __call__(self, actions: tf.Tensor) -> tf.Tensor:
        embedded_actions = tf.one_hot(indices=actions, depth=self._num_actions, dtype=tf.float32)
        embedded_actions = self._mlp(embedded_actions)
        return embedded_actions


@gin.register()
class ActionArgumentMask(snt.Module):
    """ An action argument mask module that masks SC2 action arguments that are not required by the current action
    type. """

    def __init__(self, argument_name: str, action_mask_value: int = -1, name: Optional[Text] = None):
        super().__init__(name)
        assert argument_name in [arg.name for arg in pysc2.lib.actions.TYPES]
        self._mask = tf.constant([argument_name in [a.name for a in f.args] for f in pysc2.lib.actions.FUNCTIONS])
        self._action_mask_value = action_mask_value

    def __call__(self, partial_action: Dict[str, tf.Tensor], action: tf.Tensor):
        mask = tf.gather(self._mask, tf.stop_gradient(partial_action['action_type']))
        mask = tf.cast(mask, dtype=action.dtype)
        action = (action * mask) + (tf.abs(mask - 1) * self._action_mask_value)
        return action


@gin.register()
class AutoregressivePolicyHead(snt.Module):
    """ An autoregressive policy head that wraps a PolicyHead, masks sampled action arguments and updates the
    autoregressive embedding. """

    def __init__(self,
                 action_space: SC2ActionSpace,
                 action_name: str = gin.REQUIRED,
                 policy_head: Type[PolicyHead] = gin.REQUIRED,
                 action_embed: Type[ActionEmbedding] = None,
                 action_mask: Type[ActionArgumentMask] = None,
                 action_mask_value: int = -1,
                 name: Optional[Text] = None):
        """ Constructs the autoregressive policy head module.

        Args:
            action_space: The full SC2 action space.
            action_name: The name of the action argument.
            policy_head: A policy head that maps latent representation to action arguments and corresponding logits.
            action_embed: An optional action embedding module that embeds discrete action argumentss into continuous
                space. If provided, the embedding is used to update the autoregressive_embedding tensor, otherwise the
                autoregressive_embedding will be returned unchanged.
            action_mask: An optional action argument mask module that masks the current action argument given the
                partial action constructed so far.
            action_mask_value: The value for masked actions.
            name: An optional module name.
        """
        super().__init__(name)
        self._action_name = action_name
        self._num_actions = action_space.specs[action_name].n
        self._policy_head = policy_head(self._num_actions)
        self._action_embed = action_embed(self._num_actions) if action_embed is not None else None
        self._action_mask = action_mask(action_name, action_mask_value) if action_mask is not None else None
        self._action_mask_value = action_mask_value

    @property
    def action_name(self):
        return self._action_name

    def __call__(self,
                 autoregressive_embedding: tf.Tensor,
                 context: PolicyContextFeatures,
                 partial_action: Dict[str, tf.Tensor],
                 teacher_action: Optional[tf.Tensor] = None) -> Tuple[PolicyHeadOutputs, tf.Tensor]:
        """ Queries the policy head with the given autoregressive embedding and context tensors and masks sampled
        actions if necessary. Updates the autoregressive embedding with the current action choice of a action embed
        module is configured.

        Args:
            autoregressive_embedding: A 1D tensor representing the autoregressive embedding.
            context: A 1D context vector.
            partial_action: The partial action constructed so far.
            teacher_action: An optional teacher action. If provided, it will be used instead of the sampled action to
                update the autoregressive emedding.

        Returns:
            A tuple containing `PolicyHeadOutputs` and the new autoregressive embedding.
        """
        # apply policy head
        action, logits = self._policy_head(inputs=autoregressive_embedding, context=context)

        # mask actions that are not required by the current action type
        if self._action_mask is not None:
            action = self._action_mask(partial_action, action)

        policy_head_outputs = PolicyHeadOutputs(action=action, logits=logits)

        # update autoregressive embedding with embedded actions
        if self._action_embed is not None:
            action = tf.stop_gradient(action) if teacher_action is None else teacher_action
            mask = tf.not_equal(action, self._action_mask_value)

            # embed action, apply masking first to prevent invalid action values (e.g. -1)
            embedded_action = self._action_embed(action * tf.cast(mask, action.dtype))
            embedded_action *= tf.expand_dims(tf.cast(mask, embedded_action.dtype), axis=-1)
            autoregressive_embedding += embedded_action
        return policy_head_outputs, autoregressive_embedding