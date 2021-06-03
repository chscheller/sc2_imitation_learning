import tensorflow as tf
from pysc2.lib.static_data import UNIT_TYPES

from sc2_imitation_learning.agents.common.feature_encoder import UnitSelectionEncoder
from sc2_imitation_learning.agents.common.unit_group_encoder import mask_unit_group, ConcatAverageUnitGroupEncoder
from sc2_imitation_learning.common.conv import ConvNet1D


class UnitGroupEncoderTest(tf.test.TestCase):
    def test_mask_unit_group(self):
        unit_group = tf.constant([[[0], [1], [2]]])
        unit_group_length = tf.constant([2])
        unit_group_masked = mask_unit_group(unit_group, unit_group_length)
        self.assertAllClose(unit_group_masked, [[[0], [1], [0]]])

    def test_concat_average_unit_group_encoder(self):
        enc = ConcatAverageUnitGroupEncoder(embedding_size=16, feature_encoders={
            'multi_select': lambda: UnitSelectionEncoder(encoder=ConvNet1D(
                output_channels=[16], kernel_shapes=[1], strides=[1], paddings=['SAME'], activate_final=True))
        })

        raw_multi_select = tf.constant([[[
            UNIT_TYPES[i],  # unit_type
            0,  # player_relative
            100,  # health
            0,  # shields
            0,  # energy
            0,  # transport_slots_taken
            0,  # build_progress
        ] for i in range(3)]], dtype=tf.uint16)

        raw_multi_select_length = tf.constant([2], dtype=tf.uint16)

        embedded_unit_group, unit_group_embeddings = enc({
            'multi_select': raw_multi_select,
            'multi_select_length': raw_multi_select_length,
        })

        self.assertEqual(embedded_unit_group.dtype, tf.float32)
        self.assertEqual(embedded_unit_group.shape.as_list(), [1, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(embedded_unit_group)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(embedded_unit_group)), False)

        self.assertEqual(unit_group_embeddings['multi_select'].dtype, tf.float32)
        self.assertEqual(unit_group_embeddings['multi_select'].shape.as_list(), [1, 3, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(unit_group_embeddings['multi_select'])), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(unit_group_embeddings['multi_select'])), False)

        raw_multi_select = tf.constant([[[
            UNIT_TYPES[i],  # unit_type
            0,  # player_relative
            100,  # health
            0,  # shields
            0,  # energy
            0,  # transport_slots_taken
            0,  # build_progress
        ] for i in range(3)]], dtype=tf.uint16)

        raw_multi_select_length = tf.constant([0], dtype=tf.uint16)

        embedded_unit_group, unit_group_embeddings = enc({
            'multi_select': raw_multi_select,
            'multi_select_length': raw_multi_select_length,
        })

        self.assertEqual(embedded_unit_group.dtype, tf.float32)
        self.assertEqual(embedded_unit_group.shape.as_list(), [1, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(embedded_unit_group)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(embedded_unit_group)), False)
        self.assertAllClose(embedded_unit_group, tf.zeros_like(embedded_unit_group))

        self.assertEqual(unit_group_embeddings['multi_select'].dtype, tf.float32)
        self.assertEqual(unit_group_embeddings['multi_select'].shape.as_list(), [1, 3, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(unit_group_embeddings['multi_select'])), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(unit_group_embeddings['multi_select'])), False)