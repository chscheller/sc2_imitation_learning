import numpy as np
import tensorflow as tf

from sc2_imitation_learning.agents.common.policy_head import ActionTypePolicyHead, PolicyContextFeatures, ScalarPolicyHead, \
    SpatialPolicyHead, UnitGroupPointerPolicyHead, ActionEmbedding, ActionArgumentMask, AutoregressivePolicyHead
from sc2_imitation_learning.agents.common.spatial_decoder import ResSpatialDecoder
from sc2_imitation_learning.common.conv import ConvNet2DTranspose
from sc2_imitation_learning.common.mlp import ResMLP, MLP
from sc2_imitation_learning.environment.sc2_environment import SC2ActionSpace, SC2InterfaceConfig


class Test(tf.test.TestCase):
    def test_action_type_policy_head(self):
        policy = ActionTypePolicyHead(
            num_actions=16,
            decoder=ResMLP(
                output_size=256, num_blocks=16, with_projection=True, with_layer_norm=True, activate_final=True))
        inputs = tf.constant(np.random.rand(1, 256))
        context = PolicyContextFeatures(
            scalar_context=tf.constant(np.random.rand(1, 256)),
            unit_groups={},  # unused
            available_actions=tf.constant([([False]*15) + [True]], dtype=tf.bool),  # only action index 15 available
            map_skip={}  # unused
        )
        action, logits = policy(inputs, context)

        self.assertTrue(tf.squeeze(action) == 15)
        self.assertEqual(action.shape.as_list(), [1])

        self.assertIn(logits.dtype, [tf.float32, tf.float64])
        self.assertEqual(logits.shape.as_list(), [1, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(logits)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(logits)), False)

    def test_scalar_policy_head(self):
        policy = ScalarPolicyHead(
            num_actions=16,
            decoder=MLP(output_sizes=[256, 256], with_layer_norm=False, activate_final=True))
        inputs = tf.constant(np.random.rand(1, 256))
        context = PolicyContextFeatures(
            scalar_context=tf.constant(np.random.rand(1, 256)),
            unit_groups={},  # unused
            available_actions=tf.constant([[False]*16], dtype=tf.bool),  # unused
            map_skip={}  # unused
        )
        action, logits = policy(inputs, context)

        self.assertIn(action.dtype, [tf.int32, tf.int64])
        self.assertEqual(action.shape.as_list(), [1])
        self.assertTrue(0 <= tf.squeeze(action) < 16)

        self.assertIn(logits.dtype, [tf.float32, tf.float64])
        self.assertEqual(logits.shape.as_list(), [1, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(logits)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(logits)), False)

    def test_spatial_policy_head(self):
        policy = SpatialPolicyHead(
            num_actions=64*64,
            decoder=ResSpatialDecoder(out_channels=64, num_blocks=4),
            upsample_conv_net=ConvNet2DTranspose(
                output_channels=[32, 16, 16],
                kernel_shapes=[4, 4, 4],
                strides=[2, 2, 2],
                paddings=['SAME', 'SAME', 'SAME']
            ),
            map_skip='screen')
        inputs = tf.constant(np.random.rand(1, 64))
        context = PolicyContextFeatures(
            scalar_context=tf.constant(np.random.rand(1, 256)),
            unit_groups={},  # unused
            available_actions=tf.constant([[True]], dtype=tf.bool),  # unused
            map_skip={
                'screen': [tf.constant(np.random.rand(1, 8, 8, 64))]
            }
        )
        action, logits = policy(inputs, context)

        self.assertIn(action.dtype, [tf.int32, tf.int64])
        self.assertEqual(action.shape.as_list(), [1])
        self.assertTrue(0 <= tf.squeeze(action) < 64*64)

        self.assertIn(logits.dtype, [tf.float32, tf.float64])
        self.assertEqual(logits.shape.as_list(), [1, 64*64])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(logits)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(logits)), False)

    def test_unit_group_pointer_policy_head(self):
        policy = UnitGroupPointerPolicyHead(
            num_actions=64*64,
            query_embedding_output_sizes=[256, 16],
            key_embedding_output_sizes=[16],
            target_group='multi_select',
            mask_zeros=True)
        inputs = tf.constant(np.random.randn(1, 64))
        context = PolicyContextFeatures(
            scalar_context=tf.constant(np.random.randn(1, 256)),
            unit_groups={
                'multi_select':  tf.constant(np.concatenate([
                    np.random.randn(1, 3, 32),
                    np.zeros(shape=(1, 1, 32), dtype=np.float32),
                ], axis=1))
            },
            available_actions=tf.constant([[True]], dtype=tf.bool),  # unused
            map_skip={}  # unused
        )
        action, logits = policy(inputs, context)

        self.assertIn(action.dtype, [tf.int32, tf.int64])
        self.assertEqual(action.shape.as_list(), [1])
        self.assertTrue(0 <= tf.squeeze(action) < 3)

        self.assertIn(logits.dtype, [tf.float32, tf.float64])
        self.assertEqual(logits.shape.as_list(), [1, 4])
        self.assertAllClose(logits[:, -1], [logits.dtype.min])  # all-zero entities should have masked logits
        self.assertEqual(tf.reduce_any(tf.math.is_inf(logits)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(logits)), False)

    def test_action_embedding(self):
        embed = ActionEmbedding(num_actions=16, output_sizes=[256, 64], with_layer_norm=False)
        embedded_action = embed(tf.constant([-1, 0, 12, 15]))

        self.assertIn(embedded_action.dtype, [tf.float32, tf.float64])
        self.assertEqual(embedded_action.shape.as_list(), [4, 64])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(embedded_action)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(embedded_action)), False)

    def test_action_argument_mask(self):
        mask = ActionArgumentMask(argument_name='minimap', action_mask_value=-1)
        masked_action = mask({'action_type': tf.constant([0, 13])}, tf.constant([1, 2]))

        self.assertIn(masked_action.dtype, [tf.int32, tf.int64])
        self.assertEqual(masked_action.shape.as_list(), [2])
        self.assertAllEqual(masked_action, [-1, 2])


    def test_autoregressive_policy_head(self):
        ar_policy = AutoregressivePolicyHead(
            action_space=SC2ActionSpace(SC2InterfaceConfig()),
            action_name='select_add',
            policy_head=lambda a: ScalarPolicyHead(
                num_actions=a, decoder=MLP(output_sizes=[256, 256], with_layer_norm=False, activate_final=True)),
            action_embed=lambda a: ActionEmbedding(
                num_actions=a, output_sizes=[256, 64], with_layer_norm=False),
            action_mask=lambda a, b: ActionArgumentMask(argument_name=a, action_mask_value=b),
            action_mask_value=-1
        )
        ar_embedding = tf.constant(np.random.randn(2, 64), dtype=tf.float32)
        (action, logits), updated_ar_embedding = ar_policy(
            autoregressive_embedding=ar_embedding,
            context=PolicyContextFeatures(
                scalar_context=tf.constant(np.random.randn(2, 256), dtype=tf.float32),
                unit_groups={},  # unused
                available_actions=tf.constant([[True], [True]], dtype=tf.bool),  # unused
                map_skip={}  # unused
            ),
            partial_action={
                'action_type': tf.constant([0, 3])
            }
        )

        self.assertIn(action.dtype, [tf.int32, tf.int64])
        self.assertEqual(action.shape.as_list(), [2])
        self.assertEqual(action[0], -1)  # action_type = 0 (no_op) does NOT require a select_add argument
        self.assertNotEqual(action[1], -1)  # action_type = 3 (select_rect) does require a select_add argument

        self.assertIn(logits.dtype, [tf.float32, tf.float64])
        self.assertEqual(logits.shape.as_list(), [2, 2])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(logits)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(logits)), False)

        self.assertIn(updated_ar_embedding.dtype, [tf.float32, tf.float64])
        self.assertEqual(updated_ar_embedding.shape.as_list(), [2, 64])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(updated_ar_embedding)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(updated_ar_embedding)), False)
        self.assertAllClose(ar_embedding[0], updated_ar_embedding[0])  # masked actions should not update the embedding
        self.assertNotAllClose(ar_embedding[1], updated_ar_embedding[1])  # unmasked actions should update the embedding


