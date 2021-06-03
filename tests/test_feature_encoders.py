import numpy as np
import pysc2.lib.actions
import tensorflow as tf
from pysc2.env.sc2_env import Race
from pysc2.lib.features import Player
from pysc2.lib.static_data import UNIT_TYPES, ABILITIES
from pysc2.lib.upgrades import Upgrades

from sc2_imitation_learning.agents.common.feature_encoder import IdentityEncoder, PlayerEncoder, RaceEncoder, \
    UpgradesEncoder, GameLoopEncoder, AvailableActionsEncoder, UnitCountsEncoder, ActionEncoder, ControlGroupEncoder, \
    ProductionQueueEncoder, UnitSelectionEncoder, OneHotEncoder, ScaleEncoder, LogScaleEncoder, UnitTypeEncoder
from sc2_imitation_learning.agents.common.unit_group_encoder import mask_unit_group
from sc2_imitation_learning.common.conv import ConvNet2D, ConvNet1D
from sc2_imitation_learning.common.transformer import SC2EntityTransformerEncoder


class FeatureEncoderTest(tf.test.TestCase):
    def test_identity_encoder(self):
        enc = IdentityEncoder()

        # test float tensor
        t = tf.constant([[0]], dtype=tf.float32)
        self.assertEqual(t, enc(t))

        # test uint8 tensor
        t = tf.constant([[0]], dtype=tf.uint8)
        self.assertEqual(t, enc(t))

    def test_player_encoder(self):
        enc = PlayerEncoder(embedding_size=16)

        player_features = tf.constant([list(range(len(Player)))], dtype=tf.uint8)
        encoded_player_features = enc(player_features)

        self.assertEqual(encoded_player_features.dtype, tf.float32)
        self.assertEqual(encoded_player_features.shape.as_list(), [1, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_player_features)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_player_features)), False)


    def test_race_encoder(self):
        enc = RaceEncoder(embedding_size=16)

        for race in range(len(Race)):
            race_features = tf.constant([[race]], dtype=tf.uint16)
            encoded_race_features = enc(race_features)

            self.assertEqual(encoded_race_features.dtype, tf.float32)
            self.assertEqual(encoded_race_features.shape.as_list(), [1, 16])
            self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_race_features)), False)
            self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_race_features)), False)

    def test_upgrades_encoder(self):
        enc = UpgradesEncoder(embedding_size=16)

        raw_upgrades = np.full(shape=(1, len(Upgrades)), fill_value=False, dtype=np.bool)
        raw_upgrades[:, ::2] = True
        raw_upgrades = tf.constant(raw_upgrades)

        encoded_upgrades = enc(raw_upgrades)

        self.assertEqual(encoded_upgrades.dtype, tf.float32)
        self.assertEqual(encoded_upgrades.shape.as_list(), [1, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_upgrades)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_upgrades)), False)


    def test_game_loop_encoder(self):
        enc = GameLoopEncoder(embedding_size=16)

        raw_game_loop = tf.constant([[4], [8], [9]], dtype=tf.uint16)

        encoded_game_loop = enc(raw_game_loop)

        self.assertEqual(encoded_game_loop.dtype, tf.float32)
        self.assertEqual(encoded_game_loop.shape.as_list(), [3, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_game_loop)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_game_loop)), False)

    def test_available_actions_encoder(self):
        enc = AvailableActionsEncoder(embedding_size=16)

        raw_available_actions = np.full(shape=(1, len(pysc2.lib.actions.FUNCTIONS)), fill_value=0, dtype=np.uint16)
        raw_available_actions[:, [0, 1, 3, 5]] = 1
        raw_available_actions = tf.constant(raw_available_actions, dtype=tf.uint16)

        encoded_available_actions = enc(raw_available_actions)

        self.assertEqual(encoded_available_actions.dtype, tf.float32)
        self.assertEqual(encoded_available_actions.shape.as_list(), [1, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_available_actions)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_available_actions)), False)

    def test_unit_counts_encoder(self):
        enc = UnitCountsEncoder(embedding_size=16)

        raw_unit_counts = np.full(shape=(1, len(UNIT_TYPES)), fill_value=0, dtype=np.uint16)
        raw_unit_counts[:, [0, 1, 3, 5]] = [1, 2, 3, 4]
        raw_unit_counts = tf.constant(raw_unit_counts, dtype=tf.uint16)

        encoded_raw_unit_counts = enc(raw_unit_counts)

        self.assertEqual(encoded_raw_unit_counts.dtype, tf.float32)
        self.assertEqual(encoded_raw_unit_counts.shape.as_list(), [1, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_raw_unit_counts)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_raw_unit_counts)), False)

    def test_action_encoder(self):
        enc = ActionEncoder(num_actions=16, embedding_size=16)

        raw_action = tf.constant([-1, 15, 20], dtype=tf.int64)

        encoded_raw_action = enc(raw_action)

        self.assertEqual(encoded_raw_action.dtype, tf.float32)
        self.assertEqual(encoded_raw_action.shape.as_list(), [3, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_raw_action)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_raw_action)), False)

    def test_control_group_encoder(self):
        enc = ControlGroupEncoder(encoder=ConvNet1D(
            output_channels=[16], kernel_shapes=[1], strides=[1], paddings=['SAME'], activate_final=True))

        raw_control_group = tf.constant([np.stack([
            np.asarray([UNIT_TYPES[i] for i in range(10)], dtype=np.uint16),
            np.arange(10, dtype=np.uint16)
        ], axis=-1)], dtype=tf.uint16)

        encoded_control_group = enc(raw_control_group)

        self.assertEqual(encoded_control_group.dtype, tf.float32)
        self.assertEqual(encoded_control_group.shape.as_list(), [1, 10, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_control_group)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_control_group)), False)

    def test_production_queue_encoder(self):
        raw_control_group = tf.constant([np.stack([
            np.asarray([ABILITIES[i] for i in range(16)], dtype=np.uint16),
            np.arange(16, dtype=np.uint16)
        ], axis=-1)], dtype=tf.uint16)

        # test with full production queue
        enc = ProductionQueueEncoder(max_position=16, encoder=ConvNet1D(
            output_channels=[16], kernel_shapes=[1], strides=[1], paddings=['SAME'], activate_final=True))

        encoded_control_group = enc(raw_control_group)

        self.assertEqual(encoded_control_group.dtype, tf.float32)
        self.assertEqual(encoded_control_group.shape.as_list(), [1, 16, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_control_group)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_control_group)), False)

        # test with production queue of length 10
        enc = ProductionQueueEncoder(max_position=16, encoder=SC2EntityTransformerEncoder(
            num_layers=2, model_dim=32, num_heads=2, dff=64, mask_value=0))

        encoded_control_group = enc(mask_unit_group(raw_control_group, tf.constant([10], dtype=tf.int32), -1))

        self.assertEqual(encoded_control_group.dtype, tf.float32)
        self.assertEqual(encoded_control_group.shape.as_list(), [1, 16, 32])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_control_group)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_control_group)), False)
        self.assertNotEqual(tf.reduce_sum(encoded_control_group[:, 0:]), 0)
        self.assertEqual(tf.reduce_sum(encoded_control_group[:, 10:]), 0)

        # test with empty production queue
        encoded_control_group = enc(mask_unit_group(raw_control_group, tf.constant([0], dtype=tf.int32), -1))

        self.assertEqual(encoded_control_group.dtype, tf.float32)
        self.assertEqual(encoded_control_group.shape.as_list(), [1, 16, 32])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_control_group)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_control_group)), False)
        self.assertEqual(tf.reduce_sum(encoded_control_group[:, 0:]), 0)

    def test_unit_selection_encoder(self):
        enc = UnitSelectionEncoder(encoder=ConvNet1D(
            output_channels=[16], kernel_shapes=[1], strides=[1], paddings=['SAME'], activate_final=True))

        raw_control_group = tf.constant([[[
            UNIT_TYPES[0],  # unit_type
            0,  # player_relative
            100,  # health
            0,  # shields
            0,  # energy
            0,  # transport_slots_taken
            0,  # build_progress
         ]]], dtype=tf.uint16)

        encoded_control_group = enc(raw_control_group)

        self.assertEqual(encoded_control_group.dtype, tf.float32)
        self.assertEqual(encoded_control_group.shape.as_list(), [1, 1, 16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_control_group)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_control_group)), False)

    def test_one_hot_encoder(self):
        enc = OneHotEncoder(depth=16)

        raw_feature = tf.constant([-1, 15, 20], dtype=tf.int64)

        encoded_feature = enc(raw_feature)

        self.assertEqual(encoded_feature.dtype, tf.float32)
        self.assertEqual(encoded_feature.shape.as_list(), [3, 16])
        self.assertAllClose(encoded_feature, [[0]*16, ([0]*15) + [1], [0]*16])
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_feature)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_feature)), False)

    def test_scale_encoder(self):
        enc = ScaleEncoder(factor=0.5)

        raw_feature = tf.constant([-1, 0, 20], dtype=tf.int64)

        encoded_feature = enc(raw_feature)

        self.assertEqual(encoded_feature.dtype, tf.float32)
        self.assertEqual(encoded_feature.shape.as_list(), [3, 1])
        self.assertAllClose(encoded_feature, [[-0.5], [0.0], [10.0]])

    def test_log_scale_encoder(self):
        enc = LogScaleEncoder()

        raw_feature = tf.constant([0, 1, 20], dtype=tf.int64)

        encoded_feature = enc(raw_feature)

        self.assertEqual(encoded_feature.dtype, tf.float32)
        self.assertEqual(encoded_feature.shape.as_list(), [3, 1])
        self.assertAllClose(encoded_feature, [[0], [np.log(2)], [np.log(21)]])

    def test_unit_type_encoder(self):
        enc = UnitTypeEncoder(max_unit_count=2, encoder=ConvNet1D(
            output_channels=[16], kernel_shapes=[1], strides=[1], paddings=['SAME'], activate_final=True))

        raw_feature = tf.constant([
            [
                [UNIT_TYPES[0], 0],
                [0, UNIT_TYPES[1]]
            ],
            [
                [UNIT_TYPES[0], 0],
                [UNIT_TYPES[1], 0]
            ]
        ], dtype=tf.uint16)

        encoded_feature = enc(raw_feature)

        self.assertEqual(encoded_feature.dtype, tf.float32)
        self.assertEqual(encoded_feature.shape.as_list(), [2, 2, 2, 16])
        self.assertAllClose(encoded_feature[0, 0, 1], np.zeros((16,)))
        self.assertAllClose(encoded_feature[0, 1, 0], np.zeros((16,)))
        self.assertAllClose(encoded_feature[1, 0, 1], np.zeros((16,)))
        self.assertAllClose(encoded_feature[1, 1, 1], np.zeros((16,)))
        self.assertNotAllClose(encoded_feature[0, 0, 0], np.zeros((16,)))
        self.assertNotAllClose(encoded_feature[0, 1, 1], np.zeros((16,)))
        self.assertNotAllClose(encoded_feature[1, 0, 0], np.zeros((16,)))
        self.assertNotAllClose(encoded_feature[1, 1, 0], np.zeros((16,)))
        self.assertEqual(tf.reduce_any(tf.math.is_inf(encoded_feature)), False)
        self.assertEqual(tf.reduce_any(tf.math.is_nan(encoded_feature)), False)

