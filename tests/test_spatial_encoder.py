import numpy as np
import tensorflow as tf

from sc2_imitation_learning.agents.common.feature_encoder import OneHotEncoder
from sc2_imitation_learning.agents.common.spatial_encoder import ImpalaCNNSpatialEncoder, AlphaStarSpatialEncoder
from sc2_imitation_learning.common.conv import ConvNet2D


class Test(tf.test.TestCase):
    def test_impala_cnnspatial_encoder(self):
        enc = ImpalaCNNSpatialEncoder(
            feature_layer_encoders={
                'player_relative': lambda: OneHotEncoder(depth=5)
            },
            input_projection_dim=32,
            num_blocks=[2, 2, 2],
            output_channels=[32, 64, 64],
            max_pool_padding='SAME',
            spatial_embedding_size=256
        )

        raw_features = {
            'player_relative': tf.constant(np.random.randint(0, 5, (1, 64, 64), dtype=np.uint16), dtype=tf.uint16)
        }

        embedded_spatial, map_skip = enc(raw_features)

        self.assertEqual(embedded_spatial.dtype, tf.float32)
        self.assertEqual(embedded_spatial.shape.as_list(), [1, 256])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(embedded_spatial)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(embedded_spatial)), False)

        self.assertEqual(len(map_skip), 1)
        self.assertEqual(map_skip[0].dtype, tf.float32)
        self.assertEqual(map_skip[0].shape.as_list(), [1, 8, 8, 64])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(map_skip[0])), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(map_skip[0])), False)


    def test_alpha_star_spatial_encoder(self):

        enc = AlphaStarSpatialEncoder(
            feature_layer_encoders={
                'player_relative': lambda: OneHotEncoder(depth=5)
            },
            input_projection_dim=16,
            downscale_conv_net=ConvNet2D(
                output_channels=[16, 32],
                kernel_shapes=[4, 4],
                strides=[2, 2],
                paddings=['SAME', 'SAME'],
                activate_final=True,
            ),
            res_out_channels=32,
            res_num_blocks=4,
            res_stride=1,
            spatial_embedding_size=256
        )

        raw_features = {
            'player_relative': tf.constant(np.random.randint(0, 5, (1, 64, 64), dtype=np.uint16), dtype=tf.uint16)
        }

        embedded_spatial, map_skip = enc(raw_features)

        self.assertEqual(embedded_spatial.dtype, tf.float32)
        self.assertEqual(embedded_spatial.shape.as_list(), [1, 256])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(embedded_spatial)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(embedded_spatial)), False)

        self.assertEqual(len(map_skip), 5)
        for i in range(len(map_skip)):
            self.assertEqual(map_skip[i].dtype, tf.float32)
            self.assertEqual(map_skip[i].shape.as_list(), [1, 16, 16, 32])
            self.assertEqual(tf.reduce_any(tf.math.is_inf(map_skip[i])), False)
            self.assertEqual(tf.reduce_any(tf.math.is_nan(map_skip[i])), False)
