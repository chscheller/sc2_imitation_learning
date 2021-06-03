from typing import Optional, Text, Union, Sequence

import numpy as np
import sonnet as snt
import tensorflow as tf


class MaskedGlobalAveragePooling1D(snt.Module):
    """ Global average pooling operation for masked temporal inputs. """

    def __init__(self, mask_value=0, name: Optional[Text] = None):
        """ Initializes the sparse average pooling module

        Args:
            mask_value: input value that will be masked.
            name: An optional string name for the module.
        """
        super().__init__(name)
        self._mask_value = mask_value

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Applies the defined pooling operation on the inputs.

        Args:
            inputs: A tf.Tensor with shape (batch_dim, sequence_length, channels)

        Returns: A tf.Tensor with shape (batch_dim, channels)
        """
        mask = tf.reduce_any(tf.not_equal(inputs, self._mask_value), axis=-1)  # B x T
        mask = tf.cast(mask, inputs.dtype)
        mask = tf.expand_dims(mask, axis=2)
        inputs *= mask
        return tf.math.divide_no_nan(tf.reduce_sum(inputs, axis=1), tf.reduce_sum(mask, axis=1))


class SparseOneHot(snt.Module):
    """ Embedding module for sparse vocabulary. Supports unknown tokens that lay between 0 and max(vocab). """

    def __init__(self,
                 vocab: Sequence[int],
                 dtype: tf.DType = tf.float32,
                 name: Optional[Text] = None):
        """ Initializes the sparse one hot module.

        Args:
            vocab: A list of non-negative integer vocabulary tokens.
            embed_dim: Embedding dimension.
            name: An optional string name for the module.
        """
        super().__init__(name)
        if not all(i >= 0 for i in vocab):
            raise ValueError("Negative vocabulary tokens are not supported.")
        self._dtype = dtype
        self._vocab_size = len(vocab) + 1
        vocab_lookup = np.zeros((np.max(vocab) + 1,), dtype=np.int32)
        # start lookup range from one to keep zeros for unknowns
        vocab_lookup[vocab] = np.arange(1, len(vocab) + 1, dtype=np.int32)
        self._vocab_lookup = tf.constant(vocab_lookup, dtype=tf.int32)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Embeds the given inputs.

        Args:
            inputs: Input tensor with vocabulary indices.

        Returns:
            The resulting embedding tensor.
        """
        inputs = tf.expand_dims(inputs, axis=-1)
        indices = tf.gather_nd(params=self._vocab_lookup, indices=tf.cast(inputs, dtype=tf.int32))
        return tf.one_hot(indices=indices, depth=self._vocab_size, dtype=self._dtype)


class SparseEmbed(snt.Module):
    """ Embedding module for sparse vocabulary. Supports unknown tokens that lay between 0 and max(vocab). """

    def __init__(self, vocab: Union[Sequence[int], tf.Tensor], embed_dim: int, densify_gradients: bool = False,
                 name: Optional[Text] = None):
        """ Initializes the sparse embedding module

        Args:
            vocab: A list of non-negative vocabulary integer tokens.
            embed_dim: Embedding dimension.
            name: An optional string name for the module.
        """
        super().__init__(name)
        if not all(i >= 0 for i in vocab):
            raise ValueError("Negative vocabulary tokens are not supported.")
        vocab_lookup = np.zeros((np.max(vocab) + 1,), dtype=np.int32)
        # start lookup range from one to keep zeros for unknowns
        vocab_lookup[vocab] = np.arange(1, len(vocab) + 1, dtype=np.int32)
        self._vocab_lookup = tf.constant(vocab_lookup, dtype=tf.int32)
        self._embed = snt.Embed(vocab_size=len(vocab) + 1, embed_dim=embed_dim,
                                densify_gradients=densify_gradients, dtype=tf.float32)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Embeds the given inputs.

        Args:
            inputs: Input tensor with vocabulary indices.

        Returns:
            The resulting embedding tensor.
        """
        inputs = tf.expand_dims(inputs, axis=-1)
        indices = tf.gather_nd(params=self._vocab_lookup, indices=tf.cast(inputs, dtype=tf.int32))
        return self._embed(indices)
