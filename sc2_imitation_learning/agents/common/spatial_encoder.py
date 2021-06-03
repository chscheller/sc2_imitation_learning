from abc import ABC, abstractmethod
from typing import Sequence, Union, Text, Optional, List, Dict, NamedTuple, Type

import gin
import sonnet as snt
import tensorflow as tf

from sc2_imitation_learning.agents.common.feature_encoder import FeatureEncoder
from sc2_imitation_learning.common.conv import ResBlock, ImpalaResBlock, ConvNet2D


class SpatialEncoderOutputs(NamedTuple):
    embedded_spatial: tf.Tensor
    map_skip: List[tf.Tensor]


class SpatialEncoder(snt.Module, ABC):
    @abstractmethod
    def __call__(self, features: Dict[str, tf.Tensor]) -> SpatialEncoderOutputs:
        """ Applies the spatial encoder transformation.

        Args:
            features: A Dict of spatial feature layers (tf.Tensor).

        Returns:
            A NamedTuple:
                - embedded_spatial: A 1D tensor that represents the spatial embedding.
                - map_skip: A List of 2D tensors that represent intermediate spatial representations before the 1D
                    bottleneck (can be used by spatial policies that rely on spatial information).
        """
        pass


@gin.register
class ImpalaCNNSpatialEncoder(SpatialEncoder):
    """ Spatial encoder based on the residual CNN architecture described in `IMPALA: Scalable Distributed Deep-RL
    with Importance Weighted Actor-Learner Architectures` (https://arxiv.org/abs/1802.01561).

    This architecture consists of a number of `ImpalaResModule` that transform spatial feature layers into a spatial
    representation. The output tensor of the last `ImpalaResModule` is stored in `map_skip` and embedded into a 1D
    tensor by a linear layer followed by ReLU activations."""

    def __init__(self,
                 feature_layer_encoders: Dict[str, Type[FeatureEncoder]] = gin.REQUIRED,
                 input_projection_dim: int = gin.REQUIRED,
                 num_blocks: Sequence[int] = gin.REQUIRED,
                 output_channels: Sequence[int] = gin.REQUIRED,
                 max_pool_padding: str = 'SAME',
                 spatial_embedding_size: int = gin.REQUIRED,
                 name: Optional[Text] = None):
        """ Constructs the Impala CNN module

        Args:
            feature_layer_encoders: A Dict of `FeatureEncoder`s that specify the feature layers and their encoding.
            input_projection_dim: A scalar that defines the channel dim of the input projection.
            num_blocks: A sequence of scalars that define the number residual conv blocks in each `ImpalaResModule`.
            output_channels: A sequence of scalars that defines the channel dimension in each `ImpalaResModule`.
            max_pool_padding: The padding applied to the inputs of the max-pooling layer (either 'SAME' or 'Valid').
            spatial_embedding_size: The size of the 1D embedding.
            name: An optional module name.
        """
        super().__init__(name)
        self._feature_layer_encoders = {key: enc() for key, enc in feature_layer_encoders.items()}
        self._input_projection = snt.Conv2D(
            output_channels=input_projection_dim, kernel_shape=1, stride=1, padding='SAME')
        self._cnn = snt.Sequential([
            ImpalaResBlock(num_blocks=b, out_channels=c, max_pool_padding=max_pool_padding, name=f'impala_block_{i}')
            for i, (b, c) in enumerate(zip(num_blocks, output_channels))])
        self._flatten = snt.Flatten()
        self._final_linear = snt.Linear(output_size=spatial_embedding_size)

    def __call__(self, features: Dict[str, tf.Tensor]) -> SpatialEncoderOutputs:
        embedded_feature_layers = {key: enc(features[key]) for key, enc in self._feature_layer_encoders.items()}
        embedded_feature_layers = tf.concat(tf.nest.flatten(embedded_feature_layers), axis=-1)
        embedded_feature_layers = self._input_projection(embedded_feature_layers)

        conv_out = self._cnn(embedded_feature_layers)
        conv_out = tf.nn.relu(conv_out)

        embedded_spatial = self._flatten(conv_out)
        embedded_spatial = self._final_linear(embedded_spatial)
        embedded_spatial = tf.nn.relu(embedded_spatial)

        return SpatialEncoderOutputs(embedded_spatial, [conv_out])


@gin.register
class AlphaStarSpatialEncoder(SpatialEncoder):
    """ Spatial encoder based on the spatial encoder architecture described in `Grandmaster level in StarCraft II using
    multi-agent reinforcement learning` (https://www.nature.com/articles/s41586-019-1724-z)."""

    def __init__(self,
                 feature_layer_encoders: Dict[str, Type[FeatureEncoder]] = gin.REQUIRED,
                 input_projection_dim: int = gin.REQUIRED,
                 downscale_conv_net: ConvNet2D = gin.REQUIRED,
                 res_out_channels: int = gin.REQUIRED,
                 res_num_blocks: int = gin.REQUIRED,
                 res_stride: Union[int, Sequence[int]] = gin.REQUIRED,
                 spatial_embedding_size: int = gin.REQUIRED,
                 name: Optional[Text] = None):
        """ Constructs the AlphaStar spatial encoder module.

        Args:
            feature_layer_encoders: A Dict of `FeatureEncoder`s that specify the feature layers and their encoding.
            input_projection_dim: A scalar that defines the channel dim of the input projection.
            downscale_conv_net: A ConvNet2D that initially downscales spatial inputs.
            res_out_channels: A scalar that defines the channel dimension of the `ResBlock`s that are applied after the
                downscale convolutions.
            res_num_blocks: A scalar that defines the number of the `ResBlock`s that are applied after the downscale
                convolutions.
            res_stride:  A kernel stride (either scalar or sequence of scalars) that define the stride of the
                `ResBlock`s that are applied after the downscale convolutions.
            spatial_embedding_size: The size of the 1D embedding.
            name: An optional module name.
        """
        super().__init__(name)
        self._feature_layer_encoders = {key: enc() for key, enc in feature_layer_encoders.items()}
        self._input_projection = snt.Conv2D(
            output_channels=input_projection_dim, kernel_shape=1, stride=1, padding='SAME')
        self._downscale = downscale_conv_net
        self._res_blocks = [
            ResBlock(out_channels=res_out_channels, stride=res_stride) for _ in range(res_num_blocks)]
        self._spatial_embed = snt.Sequential([
            snt.Flatten(),
            snt.Linear(output_size=spatial_embedding_size),
            tf.nn.relu
        ])

    def __call__(self, features: Dict[str, tf.Tensor]) -> SpatialEncoderOutputs:
        embedded_feature_layers = {key: enc(features[key]) for key, enc in self._feature_layer_encoders.items()}
        embedded_feature_layers = tf.concat(tf.nest.flatten(embedded_feature_layers), axis=-1)
        embedded_feature_layers = self._input_projection(embedded_feature_layers)

        conv_out = self._downscale(embedded_feature_layers)
        map_skip = [conv_out]

        for layer in self._res_blocks:
            conv_out = layer(conv_out)
            map_skip.append(conv_out)
        conv_out = tf.nn.relu(conv_out)

        embedded_spatial = self._spatial_embed(conv_out)

        return SpatialEncoderOutputs(embedded_spatial, map_skip)
