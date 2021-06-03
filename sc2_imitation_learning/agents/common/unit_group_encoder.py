from abc import ABC, abstractmethod
from typing import Dict, NamedTuple, Type

import gin
import sonnet as snt
import tensorflow as tf

from sc2_imitation_learning.agents.common.feature_encoder import FeatureEncoder
from sc2_imitation_learning.common.layers import MaskedGlobalAveragePooling1D


class UnitGroupsEncoderOutputs(NamedTuple):
    embedded_unit_group: tf.Tensor
    unit_group_embeddings: Dict[str, tf.Tensor]


def mask_unit_group(unit_group: tf.Tensor, unit_group_length: tf.Tensor, mask_value=0) -> tf.Tensor:
    """ Masks unit groups according to their length.

    Args:
        unit_group: A tensor of rank 3 with a sequence of unit feature vectors.
        unit_group_length: The length of the unit group (assumes all unit feature vectors upfront).
        mask_value: The mask value.

    Returns:
        A tensor of rank 3 where indices beyond unit_group_length are zero-masked.

    """
    if unit_group_length is not None:
        # get rid of last dimensions with size 1
        if unit_group.shape.rank - unit_group_length.shape.rank < 2:
            unit_group_length = tf.squeeze(unit_group_length, axis=-1)  # B

        # mask with mask_value
        unit_group_mask = tf.sequence_mask(
            tf.cast(unit_group_length, tf.int32), maxlen=unit_group.shape[1], dtype=unit_group.dtype)  # B x T
        unit_group_mask = tf.expand_dims(unit_group_mask, axis=-1)
        unit_group *= unit_group_mask
        if mask_value != 0:
            mask_value = tf.convert_to_tensor(mask_value)
            unit_group = tf.cast(unit_group, mask_value.dtype)
            unit_group_mask = tf.cast(unit_group_mask, mask_value.dtype)
            unit_group += (1 - unit_group_mask) * mask_value
    return unit_group


class UnitGroupEncoder(snt.Module, ABC):
    """ Encoder module for unit group features. """

    @abstractmethod
    def __call__(self, features: Dict[str, tf.Tensor]) -> UnitGroupsEncoderOutputs:
        """ Encodes the unit group features

        Args:
            features: A Dict with raw scalar features.

        Returns:
            A namedtuple with:
                - embedded_unit_group: An embedded unit group vector
                - unit_group_embeddings: A Dict of unit group embeddings.
        """
        pass


@gin.register
class ConcatAverageUnitGroupEncoder(UnitGroupEncoder):
    """ Unit group encoder module that encodes unit groups by concatenating their average embedding vectors """
    def __init__(self,
                 embedding_size: int = gin.REQUIRED,
                 feature_encoders: Dict[str, Type[FeatureEncoder]] = gin.REQUIRED):
        super().__init__()
        self._feature_encoders = {key: enc() for key, enc in feature_encoders.items()}
        self._unit_group_embed = {
            key: snt.Sequential([

                MaskedGlobalAveragePooling1D(mask_value=0),  # assume encoded unit group are zero masked before.
                snt.Linear(output_size=embedding_size),
                tf.nn.relu
            ])
            for key in self._feature_encoders.keys()
        }

    def __call__(self, features: Dict[str, tf.Tensor]) -> UnitGroupsEncoderOutputs:
        unit_group_embeddings = {
            key: enc(mask_unit_group(features[key], features.get(f'{key}_length', None), -1))
            for key, enc in self._feature_encoders.items()}

        embedded_unit_groups = {
            key: emb(unit_group_embeddings[key])
            for key, emb in self._unit_group_embed.items()
        }
        embedded_unit_groups = tf.concat(tf.nest.flatten(embedded_unit_groups), axis=-1)

        return UnitGroupsEncoderOutputs(
            embedded_unit_group=embedded_unit_groups, unit_group_embeddings=unit_group_embeddings)
