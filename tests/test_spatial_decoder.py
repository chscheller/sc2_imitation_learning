from unittest import TestCase
import numpy as np
import tensorflow as tf

from sc2_imitation_learning.agents.common.spatial_decoder import ResSpatialDecoder, FiLMedSpatialDecoder


class Test(tf.test.TestCase):
    def test_res_spatial_decoder(self):
        dec = ResSpatialDecoder(out_channels=64, num_blocks=4)
        autoregressive_embedding = tf.constant(np.random.randn(1, 64), dtype=tf.float32)
        map_skip = [tf.constant(np.random.randn(1, 8, 8, 64), dtype=tf.float32)]
        conv_out = dec(autoregressive_embedding=autoregressive_embedding, map_skip=map_skip)

        self.assertEqual(conv_out.dtype, tf.float32)
        self.assertEqual(conv_out.shape.as_list(), [1, 8, 8, 64])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(conv_out)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(conv_out)), False)

    def test_filmed_spatial_decoder(self):
        dec = FiLMedSpatialDecoder(out_channels=64, num_blocks=4)
        autoregressive_embedding = tf.constant(np.random.randn(1, 512), dtype=tf.float32)
        map_skip = [
            tf.constant(np.random.randn(1, 8, 8, 64), dtype=tf.float32),
            tf.constant(np.random.randn(1, 8, 8, 64), dtype=tf.float32),
            tf.constant(np.random.randn(1, 8, 8, 64), dtype=tf.float32),
            tf.constant(np.random.randn(1, 8, 8, 64), dtype=tf.float32),
            tf.constant(np.random.randn(1, 8, 8, 64), dtype=tf.float32)
        ]
        conv_out = dec(autoregressive_embedding=autoregressive_embedding, map_skip=map_skip)

        self.assertEqual(conv_out.dtype, tf.float32)
        self.assertEqual(conv_out.shape.as_list(), [1, 8, 8, 64])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(conv_out)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(conv_out)), False)
