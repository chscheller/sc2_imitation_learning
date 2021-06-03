from abc import ABC, abstractmethod
from typing import Callable, Optional, Text

import gin
import sonnet as snt
import tensorflow as tf
from pysc2.env.sc2_env import Race
from pysc2.lib.static_data import UNIT_TYPES, ABILITIES

from sc2_imitation_learning.common.layers import SparseOneHot, SparseEmbed
from sc2_imitation_learning.common.utils import positional_encoding


class FeatureEncoder(snt.Module, ABC):
    """ General base class for encoder modules that encode raw observations features. """

    @abstractmethod
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Applies the feature encoder to the inputs.

        Args:
            inputs: A tf.Tensor containing the raw feature observations.

        Returns:
            A tf.Tensor with the encoded feature observations.
        """
        pass


@gin.register
class IdentityEncoder(FeatureEncoder):
    """ Identity mapping encoder. """

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs


@gin.register
class PlayerEncoder(FeatureEncoder):
    """ Encodes player observations using a log transformation, followed by a linear transformation and ReLUs. """

    def __init__(self, embedding_size: int = 64, name: Optional[Text] = None):
        """ Initializes the player encoder module.

        Args:
            embedding_size: The output size of the linear transformation.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._linear = snt.Linear(output_size=embedding_size)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        player_embed = tf.math.log1p(tf.cast(inputs, dtype=tf.float32))
        player_embed = self._linear(player_embed)
        player_embed = tf.nn.relu(player_embed)
        return player_embed


@gin.register
class RaceEncoder(FeatureEncoder):
    """ Encodes race observations using a one-hot encoding with maximum |Race|, followed by a linear transformation
    and ReLUs. """

    def __init__(self, embedding_size: int = 64, name: Optional[Text] = None):
        """ Initializes the race encoder module.

        Args:
            embedding_size: The output size of the linear transformation.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._linear = snt.Linear(output_size=embedding_size)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(tf.squeeze(inputs, axis=-1), tf.int32)
        race_embed = tf.one_hot(indices=inputs, depth=len(Race), dtype=tf.float32)
        race_embed = self._linear(race_embed)
        race_embed = tf.nn.relu(race_embed)
        return race_embed


@gin.register
class UpgradesEncoder(FeatureEncoder):
    """ Encodes upgrade observations (bool values, whether upgrades are available or not) using a linear transformation
    and ReLUs. """

    def __init__(self, embedding_size: int = 32, name: Optional[Text] = None):
        """ Initializes the upgrade encoder module.

        Args:
            embedding_size: The output size of the linear transformation.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._linear = snt.Linear(output_size=embedding_size)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        upgrades_embed = tf.cast(inputs, dtype=tf.float32)
        upgrades_embed = self._linear(upgrades_embed)
        upgrades_embed = tf.nn.relu(upgrades_embed)
        return upgrades_embed


@gin.register
class GameLoopEncoder(FeatureEncoder):
    """ Encodes game loop observations (the current play time in number of game loops) using a transformer positional
    encoding. """

    def __init__(self, embedding_size: int = 64, max_game_loop: int = 100000, name: Optional[Text] = None):
        """ Initializes the game loop encoder module.

        Args:
            embedding_size: The size of the positional encoding vector.
            max_game_loop: The maximum game loop value.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._encoding_lookup = positional_encoding(max_position=max_game_loop, embedding_size=embedding_size)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        game_loop_embed = tf.gather_nd(self._encoding_lookup, tf.cast(inputs, tf.int32))
        return game_loop_embed


@gin.register
class AvailableActionsEncoder(FeatureEncoder):
    """ Encodes upgrade observations (bool values, whether actions are available or not) using a linear transformation
    and ReLUs. """

    def __init__(self, embedding_size: int = 64, name: Optional[Text] = None):
        """ Initializes the upgrade encoder module.

        Args:
            embedding_size: The output size of the linear transformation.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._linear = snt.Linear(output_size=embedding_size)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        available_actions_embed = tf.cast(inputs, dtype=tf.float32)
        available_actions_embed = self._linear(available_actions_embed)
        available_actions_embed = tf.nn.relu(available_actions_embed)
        return available_actions_embed


@gin.register
class UnitCountsEncoder(FeatureEncoder):
    """ Encodes unit count observations using a square-root transformation, followed by a linear transformation
    and ReLUs. """

    def __init__(self, embedding_size: int = 32, name: Optional[Text] = None):
        """ Initializes the unit counts encoder module.

        Args:
            embedding_size: The output size of the linear transformation.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._linear = snt.Linear(output_size=embedding_size)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        unit_counts_embed = tf.math.sqrt(tf.cast(inputs, dtype=tf.float32))
        unit_counts_embed = self._linear(unit_counts_embed)
        unit_counts_embed = tf.nn.relu(unit_counts_embed)
        return unit_counts_embed


@gin.register
class ActionEncoder(FeatureEncoder):
    """ Encodes action observations (categorical) using a one-hot encoding with size `num_actions`, followed by
    a linear transformation and ReLUs. Encodings of actions with their value equal to `mask_action_value` are
    zero-masked (optional). """

    def __init__(self, num_actions: int, embedding_size: int = 64, mask_action_value: Optional[int] = None,
                 name: Optional[Text] = None):
        """ Initializes the action encoder module.

        Args:
            num_actions: The total number of possible actions.
            embedding_size: The output size of the linear transformation.
            mask_action_value: An optional action value that will be masked with zeros. `None` means no masking.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._num_actions = num_actions
        self._mask_action_value = mask_action_value
        self._linear = snt.Linear(output_size=embedding_size)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        action = tf.cast(inputs, tf.int32)
        action_embed = tf.one_hot(indices=action, depth=self._num_actions, dtype=tf.float32)
        action_embed = self._linear(action_embed)
        action_embed = tf.nn.relu(action_embed)
        if self._mask_action_value is not None:
            action_mask = tf.not_equal(action, self._mask_action_value)
            action_embed *= tf.expand_dims(tf.cast(action_mask, dtype=tf.float32), axis=-1)
        return action_embed


@gin.register
class ControlGroupEncoder(FeatureEncoder):
    """ Encodes control group observations (sequence of unit type and unit count pairs).
    Unit types are one-hot encoded, concatenated with log transformed unit counts and further processed by an
    encoder module."""

    def __init__(self,
                 encoder: Callable[[tf.Tensor], tf.Tensor],
                 max_position: int = 10,
                 mask_value: int = -1,
                 name: Optional[Text] = None):
        """ Initializes the control group encoder module.

        Args:
            encoder: A encoder module that encodes preprocessed control group observations.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._unit_type_one_hot = SparseOneHot(vocab=UNIT_TYPES)
        self._pos_encoding = positional_encoding(max_position, len(UNIT_TYPES) + 2, add_batch_dim=True)
        self._encoder = encoder
        self._mask_value = mask_value

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        sequence_length = tf.shape(inputs)[1]
        mask = tf.expand_dims(tf.reduce_all(tf.not_equal(inputs, self._mask_value), axis=-1), axis=-1)
        inputs *= tf.cast(mask, dtype=inputs.dtype)
        unit_types = self._unit_type_one_hot(tf.cast(inputs[:, :, 0], dtype=tf.int32))
        unit_counts = tf.expand_dims(tf.math.log1p(tf.cast(inputs[:, :, 1], dtype=tf.float32)), axis=-1)
        control_group_embed = tf.concat([unit_types, unit_counts], axis=-1)
        control_group_embed += self._pos_encoding[:, :sequence_length, :]
        control_group_embed *= tf.cast(mask, dtype=control_group_embed.dtype)
        control_group_embed = self._encoder(control_group_embed)
        return control_group_embed


@gin.register
class ProductionQueueEncoder(FeatureEncoder):
    """ Encodes production queue observations (sequence of ability id and build progress pairs).
    Ability ids are one-hot encoded with maximum of |ABILITIES| and build progresses are scaled by 1/100.
    Subsequently, the two resulting tensors are concatenated and further processed by an encoder module."""

    def __init__(self,
                 encoder: Callable[[tf.Tensor], tf.Tensor],
                 max_position: int = 10,
                 mask_value: int = -1,
                 name: Optional[Text] = None):
        """ Initializes the production queue encoder module.

        Args:
            encoder: A encoder module that encodes preprocessed production queue observations.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._ability_one_hot = SparseOneHot(vocab=ABILITIES)
        self._pos_encoding = positional_encoding(max_position, len(ABILITIES) + 2, add_batch_dim=True)
        self._encoder = encoder
        self._mask_value = mask_value

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        sequence_length = tf.shape(inputs)[1]
        mask = tf.expand_dims(tf.reduce_all(tf.not_equal(inputs, self._mask_value), axis=-1), axis=-1)
        inputs *= tf.cast(mask, dtype=inputs.dtype)
        ability = self._ability_one_hot(tf.cast(inputs[:, :, 0], dtype=tf.int32))
        build_progress = tf.expand_dims(tf.cast(inputs[:, :, 1], dtype=tf.float32) * (1 / 100.), axis=-1)
        production_queue_embed = tf.concat([ability, build_progress], axis=-1)
        production_queue_embed += self._pos_encoding[:, :sequence_length, :]
        production_queue_embed *= tf.cast(mask, dtype=production_queue_embed.dtype)
        production_queue_embed = self._encoder(production_queue_embed)
        return production_queue_embed


@gin.register
class UnitSelectionEncoder(FeatureEncoder):
    """ Encodes raw selection observations (set of unit feature vectors).
    Unit feature vectors are preprocessed as follows:
        unit_type: one-hot with maximum of |UNIT_TYPES|
        player_relative: one-hot with maximum 5
        statistics: log transformation
        transport_slots_taken: one-hot with maximum 8
        build_progress: scaled by 1/100
    The resulting tensors are subsequently concatenated and further processed by an encoder module. """

    def __init__(self,
                 encoder: Callable[[tf.Tensor], tf.Tensor],
                 max_position: int = 512,
                 mask_value: int = -1,
                 name: Optional[Text] = None):
        """ Initializes the unit selection encoder module.

        Args:
            encoder: A encoder module that encodes preprocessed unit selection observations .
            name: An optional name for the module.
        """
        super().__init__(name)
        self._unit_type_one_hot = SparseOneHot(vocab=UNIT_TYPES)
        self._pos_encoding = positional_encoding(max_position, len(UNIT_TYPES) + 19, add_batch_dim=True)
        self._encoder = encoder
        self._mask_value = mask_value

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        sequence_length = tf.shape(inputs)[1]
        mask = tf.expand_dims(tf.reduce_all(tf.not_equal(inputs, self._mask_value), axis=-1), axis=-1)
        inputs *= tf.cast(mask, dtype=inputs.dtype)
        unit_type = self._unit_type_one_hot(tf.cast(inputs[:, :, 0], dtype=tf.int32))  # B x T x U
        player_relative = tf.one_hot(tf.cast(inputs[:, :, 1], dtype=tf.int32), depth=5)  # B x T x 5
        statistics = tf.math.log1p(tf.cast(inputs[:, :, 2:5], dtype=tf.float32))   # B x T x 3
        transport_slots_taken = tf.one_hot(tf.minimum(tf.cast(inputs[:, :, 5], dtype=tf.int32), 8), depth=9) # B x T x 9
        build_progress = tf.expand_dims(tf.cast(inputs[:, :, 6], dtype=tf.float32) / 100., axis=-1)   # B x T x 1
        unit_selection_embed = tf.concat([unit_type, player_relative, statistics, transport_slots_taken,
                                          build_progress], axis=-1)    # B x T x (U+18)
        unit_selection_embed += self._pos_encoding[:, :sequence_length, :]
        unit_selection_embed *= tf.cast(mask, dtype=unit_selection_embed.dtype)
        unit_selection_embed = self._encoder(unit_selection_embed)   # B x T x E
        return unit_selection_embed


@gin.register
class OneHotEncoder(FeatureEncoder):
    """ One-hot encodes features. """

    def __init__(self, depth: int = gin.REQUIRED, name: Optional[Text] = None):
        """ Initializes the one-hot encoder module.

        Args:
            depth: A scalar defining the depth of the one-hot dimension.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._depth = depth

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        one_hot_embed = tf.one_hot(indices=tf.cast(inputs, dtype=tf.int32), depth=self._depth, dtype=tf.float32)
        return one_hot_embed


@gin.register
class ScaleEncoder(FeatureEncoder):
    """ Encodes features by scaling them with a constant scalar factor. """

    def __init__(self, factor, dtype: tf.DType = tf.float32, name: Optional[Text] = None):
        """ Initializes the scale encoder module.

        Args:
            factor: A scalar defining the scaling.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._factor = factor
        self._dtype = dtype

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, dtype=self._dtype)
        return tf.expand_dims(inputs * self._factor, -1)


@gin.register
class LogScaleEncoder(FeatureEncoder):
    """ Encodes features by applying a log transformation. """

    def __init__(self, log_fn: Callable[[tf.Tensor], tf.Tensor] = tf.math.log1p, name: Optional[Text] = None):
        """ Initializes the log scale encoder module.

        Args:
            log_fn: Log transformation to apply.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._log_fn = log_fn

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, dtype=tf.float32)
        inputs = tf.maximum(inputs, 0)
        log_scale_embed = tf.expand_dims(self._log_fn(inputs), -1)
        return log_scale_embed


@gin.register
class UnitTypeEncoder(FeatureEncoder):
    """ Encodes spatial unit type feature layer observations.

    To capitalize on the sparsity of the unit type feature layers, upto `max_unit_count` unit types are first extracted
    into a flattened dense tensor. Unit types are then embedded into continuous space with dimension defined by
    `embed_dim` and further processed by an encoder module. Finally, the encoded unit type vectors are scattered into a
    feature layer to their original spatial locations.
    """

    def __init__(self,
                 encoder: Callable[[tf.Tensor], tf.Tensor],
                 embed_dim=32,
                 max_unit_count: int = 512,
                 name: Optional[Text] = None):
        """ Initializes the unit type encoder module.

        Args:
            encoder: A encoder module that encodes preprocessed unit type observations.
            name: An optional name for the module.
        """
        super().__init__(name)
        self._max_unit_count = max_unit_count
        # self._unit_type_one_hot = SparseOneHot(vocab=UNIT_TYPES)
        self._unit_type_embed = SparseEmbed(vocab=UNIT_TYPES, embed_dim=embed_dim, densify_gradients=True)
        self._encoder = encoder

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]

        # flatten and get original indices
        inputs_flat = tf.reshape(inputs, (-1, tf.reduce_prod(tf.shape(inputs)[1:])))
        indices_flat = tf.broadcast_to(
            tf.expand_dims(tf.range(tf.shape(inputs_flat)[-1], dtype=tf.int32), 0), shape=tf.shape(inputs_flat))

        # mask and extract unit types that are not 0
        mask_flat = tf.math.not_equal(inputs_flat, 0)
        masked_inputs_flat = tf.ragged.boolean_mask(inputs_flat, mask=mask_flat)
        masked_indices_flat = tf.ragged.boolean_mask(indices_flat, mask=mask_flat)
        masked_inputs_flat_lengths = masked_inputs_flat.row_lengths(axis=1)

        # ragged tensor to dense (so we can operate on them), with zero padding upto max_unit_count length
        masked_inputs_flat = masked_inputs_flat.to_tensor(shape=tf.concat(([batch_size], [self._max_unit_count]), axis=0))
        masked_indices_flat = masked_indices_flat.to_tensor(shape=tf.concat(([batch_size], [self._max_unit_count]), axis=0))

        # encode inputs
        masked_inputs_flat_embed = self._unit_type_embed(tf.cast(masked_inputs_flat, dtype=tf.int32))
        masked_inputs_flat_embed = self._encoder(masked_inputs_flat_embed)

        # re-apply mask to encoded units so that we have zero padded tensors again
        sequence_mask = tf.sequence_mask(masked_inputs_flat_lengths, maxlen=tf.shape(masked_inputs_flat_embed)[1])
        masked_inputs_flat_embed *= tf.cast(tf.expand_dims(sequence_mask, -1), dtype=masked_inputs_flat_embed.dtype)

        # add batch indices
        masked_indices_flat = tf.stack([
            tf.broadcast_to(tf.expand_dims(tf.range(batch_size), -1), shape=tf.shape(masked_indices_flat)),
            masked_indices_flat
        ], axis=-1)

        # scatter vectors, paddings will lead to zero updates (no affect on the result)
        unit_type_embed = tf.scatter_nd(
            indices=masked_indices_flat, updates=masked_inputs_flat_embed,
            shape=tf.concat([tf.shape(inputs_flat), [tf.shape(masked_inputs_flat_embed)[-1]]], axis=0))

        # reshape to spatial input shape
        unit_type_embed = tf.reshape(
            tensor=unit_type_embed,
            shape=tf.concat([tf.shape(inputs), [tf.shape(masked_inputs_flat_embed)[-1]]], axis=0))

        return unit_type_embed
