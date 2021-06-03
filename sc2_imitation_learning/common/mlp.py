from typing import Sequence, Union, Text, Callable, Optional

import gin
import sonnet as snt
import tensorflow as tf


@gin.register()
class MLP(snt.Module):
    def __init__(self,
                 output_sizes: Sequence[int],
                 activation: Union[Callable[[tf.Tensor], tf.Tensor], Text] = tf.nn.relu,
                 with_layer_norm: bool = False,
                 activate_final: bool = False,
                 name: Optional[Text] = None):
        super().__init__(name)
        self._output_sizes = output_sizes
        self._activate_final = activate_final
        self._with_layer_norm = with_layer_norm
        if isinstance(activation, str):
            self._activation = tf.keras.activations.deserialize(activation)
        else:
            self._activation = activation
        self._layers = []
        if self._with_layer_norm:
            self._layer_norms = []
        for output_size in self._output_sizes:
            self._layers.append(snt.Linear(output_size=output_size))
            if self._with_layer_norm:
                self._layer_norms.append(snt.LayerNorm(axis=-1, create_scale=True, create_offset=True))

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        num_layers = len(self._layers)
        mlp_out = inputs
        for i, layer in enumerate(self._layers):
            mlp_out = layer(mlp_out)
            if i < (num_layers - 1) or self._activate_final:
                if self._with_layer_norm:
                    mlp_out = self._layer_norms[i](mlp_out)
                mlp_out = self._activation(mlp_out)
        return mlp_out


class ResMLPBlock(snt.Module):
    def __init__(self,
                 output_size: int,
                 with_projection: bool = False,
                 with_layer_norm: bool = False,
                 name: Optional[Text] = None):
        super().__init__(name)
        if with_projection:
            self._linear_proj = snt.Linear(output_size=output_size)
        else:
            self._linear_proj = None
        self._linear = snt.Linear(output_size=output_size)
        if with_layer_norm:
            self._layer_norm = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        block_out = inputs
        if self._layer_norm is not None:
            block_out = self._layer_norm(block_out)
        block_out = tf.nn.relu(block_out)
        if self._linear_proj is not None:
            shortcut = self._linear_proj(block_out)
        else:
            shortcut = inputs
        block_out = self._linear(block_out)
        block_out = block_out + shortcut
        return block_out


@gin.register()
class ResMLP(snt.Module):
    def __init__(self,
                 output_size: int,
                 num_blocks: int,
                 with_projection: bool = False,
                 with_layer_norm: bool = False,
                 activate_final: bool = False,
                 name: Optional[Text] = None):
        super().__init__(name)
        self._output_size = output_size
        self._num_blocks = num_blocks
        self._with_projection = with_projection
        self._with_layer_norm = with_layer_norm
        self._activate_final = activate_final
        layers = []
        for i in range(self._num_blocks):
            layers.append(ResMLPBlock(
                output_size=self._output_size, with_projection=self._with_projection and i == 0,
                with_layer_norm=self._with_layer_norm))
        if self._activate_final:
            if self._with_layer_norm:
                layers.append(snt.LayerNorm(axis=-1, create_scale=True, create_offset=True))
            layers.append(tf.nn.relu)
        self._net = snt.Sequential(layers)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        mlp_out = self._net(inputs)
        return mlp_out
