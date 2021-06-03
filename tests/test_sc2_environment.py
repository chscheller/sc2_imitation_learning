from unittest import TestCase

import gym
import numpy as np
from pysc2.lib.static_data import UNIT_TYPES, UPGRADES

from sc2_imitation_learning.environment.sc2_environment import get_scalar_feature, SC2InterfaceConfig, Feature, \
    get_screen_feature, get_minimap_feature, LookupFeature, AvailableActionsFeature, PaddedSequenceFeature, \
    SequenceLengthFeature, UnitCountsFeature, UpgradesFeature


class SC2EnvironmentTest(TestCase):
    def _sc2_interface_config(self):
        return SC2InterfaceConfig(
            dimension_screen=(64, 64),
            dimension_minimap=(64, 64),
            screen_features=('visibility_map', 'player_relative', 'unit_type', 'selected', 'unit_hit_points_ratio',
                             'unit_energy_ratio', 'unit_density_aa'),
            minimap_features=('camera', 'player_relative', 'alerts'),
            scalar_features=('game_loop', 'available_actions', 'player'),
            available_actions=None,
            upgrade_set=None,
            max_step_mul=16,
            max_multi_select=64,
            max_cargo=8,
            max_build_queue=8,
            max_production_queue=16,
        )

    def test_get_scalar_feature(self):
        interface_config = self._sc2_interface_config()
        player_feature = get_scalar_feature('player', interface_config)
        self.assertIsInstance(player_feature, Feature)
        with self.assertRaises(Exception):
            _ = get_scalar_feature('non_existing_feature', interface_config)

    def test_get_screen_feature(self):
        interface_config = self._sc2_interface_config()
        player_feature = get_screen_feature('visibility_map', interface_config)
        self.assertIsInstance(player_feature, Feature)
        with self.assertRaises(Exception):
            _ = get_screen_feature('non_existing_feature', interface_config)

    def test_get_minimap_feature(self):
        interface_config = self._sc2_interface_config()
        player_feature = get_minimap_feature('camera', interface_config)
        self.assertIsInstance(player_feature, Feature)
        with self.assertRaises(Exception):
            _ = get_minimap_feature('non_existing_feature', interface_config)

    def test_lookup_feature(self):
        obs = {
            'a': np.random.uniform(1.0, 2.0, (2, 2)).astype(np.float64),
            'b': np.random.uniform(0.0, 1.0, (2, 2)).astype(np.float64)
        }
        feature = LookupFeature(obs_key='b', low=0.0, high=1.0, shape=(2, 2), dtype=np.float32)

        self.assertEqual(feature.spec(), gym.spaces.Box(low=0.0, high=1.0, shape=(2, 2), dtype=np.float32))

        extracted = feature.extract(obs)

        self.assertTrue(extracted.dtype == np.float32)

    def test_available_actions_feature(self):
        feature = AvailableActionsFeature(max_num_actions=3)
        self.assertEqual(feature.spec(), gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.uint16))

        obs = {'available_actions': np.asarray([1, 2])}
        extracted = feature.extract(obs)
        self.assertTrue(extracted.dtype == np.uint16)
        self.assertTrue(np.allclose(extracted, [0, 1, 1]))

        obs = {'available_actions': np.asarray([])}
        with self.assertRaises(IndexError):
            feature.extract(obs)

        obs = {'available_actions': np.asarray([0, 1, 2])}
        extracted = feature.extract(obs)
        self.assertTrue(extracted.dtype == np.uint16)
        self.assertTrue(np.allclose(extracted, [1, 1, 1]))

        obs = {'available_actions': np.asarray([4])}
        with self.assertRaises(IndexError):
            feature.extract(obs)

    def test_padded_sequence_feature(self):
        feature = PaddedSequenceFeature(
            obs_key='a', max_length=2, feature_shape=(), low=0, high=np.iinfo(np.uint16).max, dtype=np.uint16)
        self.assertEqual(feature.spec(), gym.spaces.Box(
            low=0, high=np.iinfo(np.uint16).max, shape=(2,), dtype=np.uint16))

        obs = {'a': np.arange(2)}
        extracted = feature.extract(obs)
        self.assertEqual(extracted.dtype, np.uint16)
        self.assertTrue(np.allclose(extracted, np.arange(2)))

        obs = {'a': np.arange(1)}
        extracted = feature.extract(obs)
        self.assertEqual(extracted.dtype, np.uint16)
        self.assertTrue(np.allclose(extracted, np.zeros((2,))))

        obs = {'a': np.arange(3)}
        extracted = feature.extract(obs)
        self.assertEqual(extracted.dtype, np.uint16)
        self.assertTrue(np.allclose(extracted, np.arange(2)))

    def test_sequence_length_feature(self):
        feature = SequenceLengthFeature(obs_key='a', max_length=2)
        self.assertEqual(feature.spec(), gym.spaces.Box(low=0, high=2, shape=(1,), dtype=np.uint16))

        obs = {'a': np.arange(0)}
        extracted = feature.extract(obs)
        self.assertEqual(extracted.dtype, np.uint16)
        self.assertEqual(extracted.shape, (1,))
        self.assertEqual(extracted.squeeze(), 0)

        obs = {'a': np.arange(1)}
        extracted = feature.extract(obs)
        self.assertEqual(extracted.dtype, np.uint16)
        self.assertEqual(extracted.shape, (1,))
        self.assertEqual(extracted.squeeze(), 1)

        obs = {'a': np.arange(2)}
        extracted = feature.extract(obs)
        self.assertEqual(extracted.dtype, np.uint16)
        self.assertEqual(extracted.shape, (1,))
        self.assertEqual(extracted.squeeze(), 2)

        obs = {'a': np.arange(3)}
        extracted = feature.extract(obs)
        self.assertEqual(extracted.dtype, np.uint16)
        self.assertEqual(extracted.shape, (1,))
        self.assertEqual(extracted.squeeze(), 2)

    def test_unit_counts_feature(self):
        feature = UnitCountsFeature()
        self.assertEqual(
            feature.spec(),
            gym.spaces.Box(low=0, high=np.iinfo(np.uint16).max, shape=(len(UNIT_TYPES) + 1,), dtype=np.uint16))

        # default case
        obs = {'unit_counts': np.stack([
            np.asarray([UNIT_TYPES[0], UNIT_TYPES[3]]),
            np.asarray([1, 1])
        ], axis=-1)}
        extracted = feature.extract(obs)
        expected = np.zeros((len(UNIT_TYPES) + 1,), dtype=np.uint16)
        expected[1] = 1
        expected[4] = 1
        self.assertEqual(extracted.dtype, np.uint16)
        self.assertEqual(extracted.shape, (len(UNIT_TYPES) + 1,))
        self.assertTrue(np.allclose(extracted, expected))

        # unknown unit id, < max(UNIT_TYPES)
        obs = {'unit_counts': np.stack([
            np.asarray([UNIT_TYPES[0], next(i for i in range(len(UNIT_TYPES)) if i not in UNIT_TYPES)]),
            np.asarray([1, 1])
        ], axis=-1)}
        extracted = feature.extract(obs)
        expected = np.zeros((len(UNIT_TYPES) + 1,), dtype=np.uint16)
        expected[1] = 1
        expected[0] = 1
        self.assertEqual(extracted.dtype, np.uint16)
        self.assertEqual(extracted.shape, (len(UNIT_TYPES) + 1,))
        self.assertTrue(np.allclose(extracted, expected))

        # unknown unit id, > max(UNIT_TYPES)
        obs = {'unit_counts': np.stack([
            np.asarray([UNIT_TYPES[0], max(UNIT_TYPES) + 1]),
            np.asarray([1, 1])
        ], axis=-1)}
        with self.assertRaises(IndexError):
            feature.extract(obs)

    def test_upgrades_feature(self):
        feature = UpgradesFeature()
        self.assertEqual(feature.spec(), gym.spaces.Box(low=False, high=True, shape=(len(UPGRADES),), dtype=np.bool))

        # default case
        obs = {'upgrades': np.asarray([feature._upgrade_set[0], feature._upgrade_set[4]])}
        extracted = feature.extract(obs)
        expected = np.zeros((len(UPGRADES),), dtype=np.bool)
        expected[0] = True
        expected[4] = True
        self.assertEqual(extracted.dtype, np.bool)
        self.assertEqual(extracted.shape, (len(UPGRADES),))
        self.assertTrue(np.allclose(extracted, expected))

        # unknown upgrade
        obs = {'upgrades': np.asarray([feature._upgrade_set[0], max(UPGRADES) + 1])}
        extracted = feature.extract(obs)
        expected = np.zeros((len(UPGRADES),), dtype=np.bool)
        expected[0] = True
        self.assertEqual(extracted.dtype, np.bool)
        self.assertEqual(extracted.shape, (len(UPGRADES),))
        self.assertTrue(np.allclose(extracted, expected))

        feature = UpgradesFeature([UPGRADES[0], UPGRADES[4]])

        # custom upgrade set, unknown upgrade
        obs = {'upgrades': np.asarray([UPGRADES[0], UPGRADES[1]])}
        extracted = feature.extract(obs)
        expected = np.zeros((2,), dtype=np.bool)
        expected[0] = True
        self.assertEqual(extracted.dtype, np.bool)
        self.assertEqual(extracted.shape, (2,))
        self.assertTrue(np.allclose(extracted, expected))
