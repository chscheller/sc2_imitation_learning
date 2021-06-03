from typing import Optional, Union, Sequence, Callable

import gin
import sonnet as snt
import tensorflow as tf
from sonnet.src.conv import ConvND
from sonnet.src.conv_transpose import ConvNDTranspose


def broadcast_nc_to_nhwc(inputs: tf.Tensor, height: int, width: int) -> tf.Tensor:
    """ Broadcasts a vector along height and width dimensions.

    Args:
        inputs: A tf.Tensor of shape (batch_dim, vector_dim).
        height: An integer value indicating the broadcast height.
        width: An integer value indicating the broadcast width.

    Returns: A tf.Tensor of shape (batch_dim, height, width, vector_dim).

    """
    assert inputs.shape.rank == 2, \
        f"Expected input rank of 2, got rank of {inputs.get_shape().rank}"
    return tf.tile(tf.expand_dims(tf.expand_dims(inputs, 1), 2), [1, height, width, 1])


def film_layer(inputs: tf.Tensor, gamma: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    """ FiLM operation as defined in https://arxiv.org/abs/1709.07871.

    Args:
        inputs: A tf.Tensor of shape (batch_dim, height, width, channels).
        gamma: A tf.Tensor of shape (batch_dim, channels).
        beta: A tf.Tensor of shape (batch_dim, channels).

    Returns: A tf.Tensor of shape (batch_dim, height, width, channels).
    """
    assert inputs.shape.rank == 4, \
        f"Expected rank of 4, got rank of {inputs.get_shape().rank}"
    assert inputs.shape[0] == gamma.shape[0] == beta.shape[0], \
        f"Batch dimensions must match but are: {inputs.shape[0]}, {gamma.shape[0]}, {beta.shape[0]}"
    assert inputs.shape[3] == gamma.shape[1] == beta.shape[1], \
        f"Channel dimensions must match but are: {inputs.shape[3]}, {gamma.shape[1]}, {beta.shape[1]}"

    height, width = inputs.shape[1:3]

    gamma = broadcast_nc_to_nhwc(gamma, height, width)
    beta = broadcast_nc_to_nhwc(beta, height, width)

    return (gamma * inputs) + beta


@gin.register()
class ResBlock(snt.Module):
    def __init__(self,
                 out_channels: int,
                 stride: Union[int, Sequence[int]] = 1,
                 use_projection: bool = False,
                 name: Optional[str] = None):
        super().__init__(name)
        self._use_projection = use_projection
        if self._use_projection:
            self._proj_layer = snt.Conv2D(output_channels=out_channels, kernel_shape=1, stride=1, name="proj_layer")
        self._conv0 = snt.Conv2D(output_channels=out_channels, kernel_shape=3, stride=stride, name="conv_0")
        self._conv1 = snt.Conv2D(output_channels=out_channels, kernel_shape=3, stride=stride, name="conv_1")

    def __call__(self, inputs):
        block_out = inputs
        block_out = tf.nn.relu(block_out)
        if self._use_projection:
            shortcut = self._proj_layer(block_out)
        else:
            shortcut = inputs
        block_out = self._conv0(block_out)
        block_out = tf.nn.relu(block_out)
        block_out = self._conv1(block_out)
        block_out = block_out + shortcut
        return block_out


@gin.register()
class FiLMedResBlock(snt.Module):
    def __init__(self,
                 out_channels: int,
                 stride: Union[int, Sequence[int]],
                 use_projection: bool = False,
                 name: Optional[str] = None):
        super().__init__(name)
        self._out_channels = out_channels
        self._stride = stride
        self._use_projection = use_projection
        if self._use_projection:
            self._proj_layer = snt.Conv2D(output_channels=out_channels, kernel_shape=1, stride=1, name="proj_layer")
        self._conv0 = snt.Conv2D(output_channels=out_channels, kernel_shape=3, stride=stride, name="conv_0")
        self._conv1 = snt.Conv2D(output_channels=out_channels, kernel_shape=3, stride=stride, name="conv_1")

    def __call__(self, inputs: tf.Tensor, gamma: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
        block_out = inputs
        block_out = tf.nn.relu(block_out)
        if self._use_projection:
            shortcut = self._proj_layer(block_out)
        else:
            shortcut = inputs
        block_out = self._conv0(block_out)
        block_out = tf.nn.relu(block_out)
        block_out = self._conv1(block_out)
        block_out = film_layer(block_out, gamma, beta)
        block_out = block_out + shortcut
        return block_out


@gin.register()
class ImpalaResBlock(snt.Module):
    def __init__(self,
                 num_blocks: int,
                 out_channels: int,
                 max_pool_padding: str = 'SAME',
                 name: Optional[str] = None):
        super().__init__(name)
        layers = [
            snt.Conv2D(output_channels=out_channels, kernel_shape=3, stride=1, name='downscale'),
            lambda x: tf.nn.pool(x, window_shape=[3, 3], pooling_type='MAX', padding=max_pool_padding, strides=[2, 2])
        ]
        layers.extend([ResBlock(out_channels) for _ in range(num_blocks)])
        self._layers = snt.Sequential(layers)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._layers(inputs)


@gin.register()
class ImpalaCNN(snt.Module):
    def __init__(self,
                 num_blocks: Sequence[int] = (2, 2, 2),
                 num_channels: Sequence[int] = (16, 32, 64),
                 mlp_output_hidden: Sequence[int] = (256,),
                 max_pool_padding: str = 'SAME',
                 name: Optional[str] = None):
        super().__init__(name=name)
        assert len(num_blocks) == len(num_channels), "the number of elements in num_blocks and num_channels must match."
        self._resnet = snt.Sequential([
            ImpalaResBlock(num_blocks=b, out_channels=c, max_pool_padding=max_pool_padding, name=f'impala_block_{i}')
            for i, (b, c) in enumerate(zip(num_blocks, num_channels))
        ] + [
            tf.nn.relu,
            snt.Flatten(),
            snt.nets.MLP(mlp_output_hidden, activation=tf.nn.relu, activate_final=True),
        ])

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._resnet(inputs)


@gin.register()
class ConvNet(snt.Module):
    def __init__(self,
                 layers: Sequence[Union[ConvND, ConvNDTranspose]],
                 activation: Union[Callable[[tf.Tensor], tf.Tensor], str] = tf.nn.relu,
                 activate_final: bool = False,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._layers = layers
        if isinstance(activation, str):
            self._activation = tf.keras.activations.deserialize(activation)
        else:
            self._activation = activation
        self._activate_final = activate_final

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        num_layers = len(self._layers)
        conv_out = inputs
        for i, layer in enumerate(self._layers):
            conv_out = layer(conv_out)
            if i < (num_layers - 1) or self._activate_final:
                conv_out = self._activation(conv_out)
        return conv_out


@gin.register()
class ConvNet1D(ConvNet):
    def __init__(self,
                 output_channels: Sequence[int],
                 kernel_shapes: Sequence[Union[int, Sequence[int]]],
                 strides: Sequence[Union[int, Sequence[int]]],
                 paddings: Sequence[Union[str, int, Sequence[int]]],
                 w_init: Optional[snt.initializers.Initializer] = None,
                 b_init: Optional[snt.initializers.Initializer] = None,
                 activation: Union[Callable[[tf.Tensor], tf.Tensor], str] = tf.nn.relu,
                 activate_final: bool = False,
                 name: Optional[str] = None):
        assert len(output_channels) == len(kernel_shapes) == len(strides) == len(paddings), \
            "output_channels, kernel_shapes, strides and paddings must have the same length."

        layers = [
            snt.Conv1D(
                output_channels=c,
                kernel_shape=k,
                stride=s,
                padding=p,
                w_init=w_init,
                b_init=b_init)
            for c, k, s, p in zip(output_channels, kernel_shapes, strides, paddings)]

        super().__init__(
            layers=layers,
            activation=activation,
            activate_final=activate_final,
            name=name)


@gin.register()
class ConvNet2D(ConvNet):
    def __init__(self,
                 output_channels: Sequence[int],
                 kernel_shapes: Sequence[Union[int, Sequence[int]]],
                 strides: Sequence[Union[int, Sequence[int]]],
                 paddings: Sequence[Union[str, int, Sequence[int]]],
                 w_init: Optional[snt.initializers.Initializer] = None,
                 b_init: Optional[snt.initializers.Initializer] = None,
                 activation: Union[Callable[[tf.Tensor], tf.Tensor], str] = tf.nn.relu,
                 activate_final: bool = False,
                 name: Optional[str] = None):
        assert len(output_channels) == len(kernel_shapes) == len(strides) == len(paddings), \
            "output_channels, kernel_shapes, strides and paddings must have the same length."

        layers = [
            snt.Conv2D(
                output_channels=c,
                kernel_shape=k,
                stride=s,
                padding=p,
                w_init=w_init,
                b_init=b_init)
            for c, k, s, p in zip(output_channels, kernel_shapes, strides, paddings)]

        super().__init__(
            layers=layers,
            activation=activation,
            activate_final=activate_final,
            name=name)


@gin.register()
class ConvNet1DTranspose(ConvNet):
    def __init__(self,
                 output_channels: Sequence[int],
                 kernel_shapes: Sequence[Union[int, Sequence[int]]],
                 strides: Sequence[Union[int, Sequence[int]]],
                 paddings: Sequence[Union[str, int, Sequence[int]]],
                 w_init: Optional[snt.initializers.Initializer] = None,
                 b_init: Optional[snt.initializers.Initializer] = None,
                 activation: Union[Callable[[tf.Tensor], tf.Tensor], str] = tf.nn.relu,
                 activate_final: bool = False,
                 name: Optional[str] = None):
        assert len(output_channels) == len(kernel_shapes) == len(strides) == len(paddings), \
            "output_channels, kernel_shapes, strides and paddings must have the same length."

        layers = [
            snt.Conv1DTranspose(
                output_channels=c,
                kernel_shape=k,
                stride=s,
                padding=p,
                w_init=w_init,
                b_init=b_init)
            for c, k, s, p in zip(output_channels, kernel_shapes, strides, paddings)]

        super().__init__(
            layers=layers,
            activation=activation,
            activate_final=activate_final,
            name=name)


@gin.register()
class ConvNet2DTranspose(ConvNet):
    def __init__(self,
                 output_channels: Sequence[int],
                 kernel_shapes: Sequence[Union[int, Sequence[int]]],
                 strides: Sequence[Union[int, Sequence[int]]],
                 paddings: Sequence[Union[str, int, Sequence[int]]],
                 w_init: Optional[snt.initializers.Initializer] = None,
                 b_init: Optional[snt.initializers.Initializer] = None,
                 activation: Union[Callable[[tf.Tensor], tf.Tensor], str] = tf.nn.relu,
                 activate_final: bool = False,
                 name: Optional[str] = None):
        assert len(output_channels) == len(kernel_shapes) == len(strides) == len(paddings), \
            "output_channels, kernel_shapes, strides and paddings must have the same length."

        layers = [
            snt.Conv2DTranspose(
                output_channels=c,
                kernel_shape=k,
                stride=s,
                padding=p,
                w_init=w_init,
                b_init=b_init)
            for c, k, s, p in zip(output_channels, kernel_shapes, strides, paddings)]

        super().__init__(
            layers=layers,
            activation=activation,
            activate_final=activate_final,
            name=name)
