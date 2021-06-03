from abc import abstractmethod, ABC
from typing import Dict, Sequence, NamedTuple, Type

import gin
import sonnet as snt
import tensorflow as tf

from sc2_imitation_learning.agents.common.feature_encoder import ActionEncoder, FeatureEncoder
from sc2_imitation_learning.environment.sc2_environment import SC2ActionSpace


class ScalarEncoderOutputs(NamedTuple):
    embedded_scalar: tf.Tensor
    scalar_context: tf.Tensor


@gin.register
class ScalarEncoder(snt.Module, ABC):
    """ Encoder module for scalar features. """

    @abstractmethod
    def __call__(self, features: Dict[str, tf.Tensor], prev_actions: Dict[str, tf.Tensor]) -> ScalarEncoderOutputs:
        """ Applies the specified encodings on features and prev_actions, constructs scalar embedding and
        scalar context vectors.

        Args:
            features: A Dict with raw scalar features.
            prev_actions: A Dict containing the actions of the previous time step.

        Returns:
            A namedtuple with:
                - embedded_scalar: A scalar embedding vector.
                - scalar_context: A scalar context vector.
        """
        pass


@gin.register
class ConcatScalarEncoder(ScalarEncoder):
    """ Concat Encoder module for scalar features. Produces an encoding and a context vector through concatenation. """

    def __init__(self,
                 action_space: SC2ActionSpace = gin.REQUIRED,
                 feature_encoders: Dict[str, Type[FeatureEncoder]] = gin.REQUIRED,
                 prev_action_encoders: Dict[str, Type[ActionEncoder]] = gin.REQUIRED,
                 context_feature_names: Sequence[str] = ('home_race_requested', 'away_race_requested',
                                                         'available_actions')):
        """ Constructs the encoder module

        Args:
            action_space: The action space with the environment.
            feature_encoders: A Dict with feature encoders. Keys must correspond to inputs.
            prev_action_encoders: A Dict with action encoders. Keys must correspond to action names.
            context_feature_names: A List with feature names that should be included in the context vector.
        """
        super().__init__()
        self._action_space = action_space
        self._context_feature_names = context_feature_names
        self._embed_features = {key: enc() for key, enc in feature_encoders.items()}
        self._embed_actions = {key: enc(action_space.specs[key].n) for key, enc in prev_action_encoders.items()}

    def __call__(self, features: Dict[str, tf.Tensor], prev_actions: Dict[str, tf.Tensor]) -> ScalarEncoderOutputs:
        embedded_features = {key: embed(features[key]) for key, embed in self._embed_features.items()}
        embedded_actions = {key: embed(prev_actions[key]) for key, embed in self._embed_actions.items()}
        scalar_context = tf.concat([embedded_features[key] for key in self._context_feature_names], axis=-1)
        embedded_scalar = tf.concat(tf.nest.flatten(embedded_features) + tf.nest.flatten(embedded_actions), axis=-1)
        return ScalarEncoderOutputs(embedded_scalar, scalar_context)
