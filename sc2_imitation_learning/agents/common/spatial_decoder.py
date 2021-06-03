from abc import ABC, abstractmethod
from typing import Sequence, Text, Optional

import gin
import sonnet as snt
import tensorflow as tf

from sc2_imitation_learning.common.conv import FiLMedResBlock, ResBlock, broadcast_nc_to_nhwc


class SpatialDecoder(snt.Module, ABC):
    @abstractmethod
    def __call__(self, autoregressive_embedding: tf.Tensor, map_skip: Sequence[tf.Tensor]) -> tf.Tensor:
        pass


@gin.register
class ResSpatialDecoder(SpatialDecoder):
    def __init__(self,
                 out_channels: int = gin.REQUIRED,
                 num_blocks: int = gin.REQUIRED,
                 name: Optional[Text] = None):
        super().__init__(name)
        self._out_channels = out_channels
        self._num_blocks = num_blocks
        self._input_transform = snt.Conv2D(output_channels=self._out_channels, kernel_shape=1, stride=1)
        self._layers = []
        for i in range(self._num_blocks):
            self._layers.append(ResBlock(out_channels=self._out_channels, stride=1))

    def __call__(self, autoregressive_embedding: tf.Tensor, map_skip: Sequence[tf.Tensor]) -> tf.Tensor:
        map_skip = list(reversed(map_skip))
        inputs, map_skip = map_skip[0], map_skip[1:]

        broadcast_embedding = broadcast_nc_to_nhwc(autoregressive_embedding, inputs.shape[1], inputs.shape[2])
        inputs = tf.concat([broadcast_embedding, inputs], axis=-1)
        inputs = tf.nn.relu(inputs)
        inputs = self._input_transform(inputs)
        inputs = tf.nn.relu(inputs)

        conv_out = inputs
        if len(map_skip) == 0:
            for layer in self._layers:
                conv_out = layer(conv_out)
        else:
            assert self._num_blocks == len(map_skip), \
                f"'num_blocks' must be equal to the lengths of 'map_skip' but got: {self._num_blocks}, {len(map_skip)}"
            for layer, skip_connection in zip(self._layers, reversed(map_skip)):
                conv_out = layer(conv_out)
                conv_out += skip_connection

        conv_out = tf.nn.relu(conv_out)

        return conv_out


@gin.register
class FiLMedSpatialDecoder(SpatialDecoder):
    def __init__(self,
                 out_channels: int = gin.REQUIRED,
                 num_blocks: int = gin.REQUIRED,
                 name: Optional[Text] = None):
        super().__init__(name)
        self._out_channels = out_channels
        self._num_blocks = num_blocks
        self._input_transform = snt.Conv2D(output_channels=self._out_channels, kernel_shape=1, stride=1)
        self._layers = []
        for i in range(self._num_blocks):
            self._layers.append(FiLMedResBlock(out_channels=self._out_channels, stride=1))

    @snt.once
    def _initialize(self, inputs: tf.Tensor, autoregressive_embedding: tf.Tensor):
        assert self._out_channels * self._num_blocks * 2 == autoregressive_embedding.shape[-1], \
            f"output_channels={self._out_channels} and num_blocks={self._num_blocks} are not " \
            f"compatible with autoregressive_embedding of size {autoregressive_embedding.shape[-1]}"

        self._reshape_non_spatial = snt.Reshape(output_shape=(inputs.shape[1], inputs.shape[2], -1))

    def __call__(self, autoregressive_embedding: tf.Tensor, map_skip: Sequence[tf.Tensor]) -> tf.Tensor:
        map_skip = list(reversed(map_skip))
        inputs, map_skip = map_skip[0], map_skip[1:]

        self._initialize(inputs, autoregressive_embedding)

        inputs = tf.concat([self._reshape_non_spatial(autoregressive_embedding), inputs], axis=-1)
        inputs = tf.nn.relu(inputs)
        inputs = self._input_transform(inputs)
        inputs = tf.nn.relu(inputs)

        gammas, betas = [], []
        for i in range(0, autoregressive_embedding.shape[-1], self._out_channels * 2):
            gammas.append(autoregressive_embedding[:, i:i+self._out_channels])
            betas.append(autoregressive_embedding[:, i+self._out_channels:i+2*self._out_channels])

        assert self._num_blocks == len(map_skip) == len(gammas) == len(betas),\
            f"'num_blocks' must be equal to the lengths of 'map_skip', 'gammas' and 'betas' but got: " \
            f"{self._num_blocks}, {len(map_skip)}, {len(gammas)} and {len(betas)}"

        conv_out = inputs
        for layer, gamma, beta, skip_connection in zip(self._layers, gammas, betas, map_skip):
            conv_out = layer(conv_out, gamma, beta)
            conv_out += skip_connection
        conv_out = tf.nn.relu(conv_out)

        return conv_out
