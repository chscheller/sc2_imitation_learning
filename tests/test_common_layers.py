import tensorflow as tf

from sc2_imitation_learning.common.layers import SparseEmbed, MaskedGlobalAveragePooling1D


class TestMaskedGlobalAveragePooling2D(tf.test.TestCase):
    def test_masked_global_average_pooling_2d(self):
        pooling = MaskedGlobalAveragePooling1D(mask_value=0)

        inputs = tf.constant([[
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ], [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ], [
            [0, 0, 0],
            [1, 1, 1],
            [-1, -1, 0],
            [0, 0, 0],
        ]], dtype=tf.float32)

        output = pooling(inputs)

        self.assertAllClose(output, [[2, 2, 2], [1, 1, 1], [0, 0, 0.5]])


class TestSparseEmbed(tf.test.TestCase):
    def test_sparse_embed(self):
        sparse_embed = SparseEmbed([0, 2, 5], embed_dim=3)

        output_1 = sparse_embed([0, 0, 2])  # valid ids
        output_2 = sparse_embed([5, 5, 2])  # valid ids
        output_3 = sparse_embed([1, 3, 4])  # invalid ids (within bounds)
        output_4 = sparse_embed([6, 7, 8])  # invalid ids (out of bounds)

        self.assertEqual(output_1.shape, (3, 3))
        self.assertNotAllEqual(output_1, output_2)
        self.assertNotAllEqual(output_1, output_3)
        self.assertNotAllEqual(output_2, output_3)
        self.assertAllEqual(output_3, output_4)

        with self.assertRaises(ValueError):
            sparse_embed2 = SparseEmbed([-1, 0, 1], embed_dim=3)
