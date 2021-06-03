import numpy as np
import tensorflow as tf

from sc2_imitation_learning.common.transformer import SC2EntityTransformerEncoder


class TestSC2EntityTransformerEncoder(tf.test.TestCase):
    def test_forward(self):
        transformer = SC2EntityTransformerEncoder(num_layers=2, model_dim=2, num_heads=2, dff=4)

        entities = tf.constant(np.random.randn(2, 3, 4), dtype=tf.float32)

        embedded_entities = transformer(entities)

        self.assertEqual(embedded_entities.dtype, tf.float32)
        self.assertEqual(embedded_entities.shape.as_list(), [2, 3, 2])
        self.assertFalse(tf.reduce_any(tf.math.is_inf(embedded_entities)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(embedded_entities)))
        self.assertNotAllClose(embedded_entities, tf.zeros_like(embedded_entities))

    def test_mask(self):
        transformer = SC2EntityTransformerEncoder(num_layers=2, model_dim=2, num_heads=2, dff=4, mask_value=0)

        # no entity masked
        entities = tf.constant(np.random.randn(2, 3, 4), dtype=tf.float32)
        embedded_entities = transformer(entities)

        self.assertEqual(embedded_entities.dtype, tf.float32)
        self.assertEqual(embedded_entities.shape.as_list(), [2, 3, 2])
        self.assertFalse(tf.reduce_any(tf.math.is_inf(embedded_entities)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(embedded_entities)))
        self.assertFalse(tf.reduce_any(embedded_entities == 0.))

        # some entities masked
        entities = tf.constant(np.concatenate([np.random.randn(2, 3, 2), np.zeros((2, 3, 2))], axis=-1), dtype=tf.float32)
        embedded_entities = transformer(entities)

        self.assertEqual(embedded_entities.dtype, tf.float32)
        self.assertEqual(embedded_entities.shape.as_list(), [2, 3, 2])
        self.assertFalse(tf.reduce_any(tf.math.is_inf(embedded_entities)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(embedded_entities)))
        self.assertFalse(tf.reduce_any(embedded_entities[:, :, :2] == 0.))
        self.assertTrue(tf.reduce_all(embedded_entities[:, :, 2:] == 0.))

        # all entities masked
        entities = tf.constant(np.zeros((2, 3, 4)), dtype=tf.float32)
        embedded_entities = transformer(entities)

        self.assertEqual(embedded_entities.dtype, tf.float32)
        self.assertEqual(embedded_entities.shape.as_list(), [2, 3, 2])
        self.assertFalse(tf.reduce_any(tf.math.is_inf(embedded_entities)))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(embedded_entities)))
        self.assertTrue(tf.reduce_all(embedded_entities == 0.))
