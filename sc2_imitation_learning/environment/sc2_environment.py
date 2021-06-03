""" StarCraft II environment """

import collections
import glob
import logging
import os
import traceback
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List, Type, Sequence, Set

import gin
import gym
import numpy as np
import pysc2.lib.actions
import pysc2.maps
from dataclasses import dataclass

import sc2reader
import tree
from pysc2.env import sc2_env, lan_sc2_env
from pysc2.env.environment import StepType
from pysc2.env.sc2_env import Race
from pysc2.lib.features import UnitLayer, Player, ProductionQueue, ScoreCumulative, ScoreByCategory, ScoreByVital, \
    ScoreVitals, ScoreCategories
from pysc2.lib.static_data import UNIT_TYPES
from pysc2.lib.upgrades import Upgrades
from s2clientprotocol.error_pb2 import ActionResult
from s2clientprotocol.sc2api_pb2 import Alert

from sc2_imitation_learning.environment.environment import EnvMeta, StepOutput, ActionSpace, ObservationSpace

logger = logging.getLogger(__name__)


HUMAN_SCORES = collections.defaultdict(lambda: 1, {
    'MoveToBeacon': 26,
    'CollectMineralShards': 133,
    'FindAndDefeatZerglings': 46,
    'DefeatRoaches': 41,
    'DefeatZerglingsAndBanelings': 729,
    'CollectMineralsAndGas': 6880,
    'BuildMarines': 138,
})

RANDOM_SCORES = collections.defaultdict(lambda: -1, {
    'MoveToBeacon': 1,
    'CollectMineralShards': 17,
    'FindAndDefeatZerglings': 4,
    'DefeatRoaches': 1,
    'DefeatZerglingsAndBanelings': 23,
    'CollectMineralsAndGas': 12,
    'BuildMarines': 0,
})


class Feature(ABC):
    """ A environment observation feature. """

    @abstractmethod
    def spec(self) -> gym.spaces.Space:
        """ The observation space of this feature  """
        pass

    @abstractmethod
    def extract(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """ Extracts the feature from a raw observation dictionary.

        Args:
            obs (Dict[str, np.ndarray]): raw observation dictionary as returned from the pysc2 environment.

        Returns:
            A numpy ndarray containing the extracted feature value.
        """
        pass


class BoxFeature(Feature):
    """ A box environment observation feature. """

    def __init__(self, low, high, shape: tuple, dtype: Type = np.uint16) -> None:
        super().__init__()
        self._spec = gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

    def spec(self) -> gym.spaces.Space:
        return self._spec

    @abstractmethod
    def extract(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        pass


class LookupFeature(BoxFeature):
    """ An environment observation feature that can simply be looked up from the raw observations. """

    def __init__(self, obs_key: str, low, high, shape: tuple, dtype=np.uint16, sub_space: str = None) -> None:
        super().__init__(low=low, high=high, shape=shape, dtype=dtype)
        self._obs_key = obs_key
        self._sub_space = sub_space

    def extract(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        if self._sub_space is not None:
            obs = obs[self._sub_space]
        extracted_feature = obs[self._obs_key]
        if extracted_feature.dtype != self._spec.dtype.type:
            extracted_feature = extracted_feature.astype(self._spec.dtype)
        return extracted_feature


class AvailableActionsFeature(BoxFeature):
    """ An environment observation feature that indicates available actions as a boolean vector. """

    def __init__(self, max_num_actions: int, action_set: Optional[List[int]] = None) -> None:
        super().__init__(low=0, high=1, shape=(max_num_actions,), dtype=np.uint16)
        self._max_num_actions = max_num_actions
        self._action_set = action_set
        if self._action_set is not None:
            if min(self._action_set) < 0:
                raise IndexError("All indices in action_set must be >= 0")
            self._action_mask = np.zeros(self._max_num_actions, dtype=np.uint16)
            self._action_mask[self._action_set] = np.uint16(1)
            assert self._action_mask[0] == 1, "no_op must always be available."
        else:
            self._action_mask = None

    def extract(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        available_actions = np.zeros(self._max_num_actions, dtype=self._spec.dtype)
        available_actions[obs['available_actions']] = self._spec.dtype.type(1)
        if self._action_mask is not None:
            available_actions *= self._action_mask
        return available_actions


class PaddedSequenceFeature(LookupFeature):
    """ An environment observation feature that pads a raw observation to a fixed length. """

    def __init__(self, obs_key: str, max_length: int, feature_shape: Tuple, low=0, high=np.iinfo(np.uint16).max,
                 dtype=np.uint16, sub_space: str = None) -> None:
        super().__init__(obs_key=obs_key, low=low, high=high, shape=(max_length,) + feature_shape, dtype=dtype,
                         sub_space=sub_space)
        self._max_length = max_length
        self._feature_shape = feature_shape

    def extract(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        sequence = super().extract(obs)[:self._max_length]
        assert len(sequence.shape) == len(self._feature_shape) + 1, \
            f"{self._obs_key}: expected sequence with rank {len(self._feature_shape) + 1} " \
            f"but got {len(sequence.shape)}."
        sequence_length = sequence.shape[0]
        if sequence_length < self._max_length:
            pad_width = [(0, self._max_length - sequence_length)] + [(0, 0) for _ in self._feature_shape]
            sequence = np.pad(sequence, pad_width)
        return sequence


class SequenceLengthFeature(LookupFeature):
    """ An environment observation feature that indicates the length of a raw sequence feature. """

    def __init__(self, obs_key: str, max_length: int, sub_space: str = None) -> None:
        super().__init__(obs_key=obs_key, low=0, high=max_length, shape=(1,), dtype=np.uint16, sub_space=sub_space)
        self._max_length = max_length

    def extract(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray([min(super().extract(obs).shape[0], self._max_length)], dtype=self._spec.dtype)


class UnitCountsFeature(BoxFeature):
    """ An environment observation feature that contains units/buildings counts in the form of a BOW vector. """

    def __init__(self) -> None:
        super().__init__(low=0, high=np.iinfo(np.uint16).max, shape=(len(UNIT_TYPES) + 1,), dtype=np.uint16)
        self._unit_type_lookup = np.zeros((np.max(UNIT_TYPES) + 1,), dtype=np.uint16)
        # + 1 to account for unknowns:
        self._unit_type_lookup[UNIT_TYPES] = np.arange(1, len(UNIT_TYPES) + 1, dtype=np.uint16)

    def extract(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        # todo: create test with empty unit_counts obs
        # + 1 to account for unknowns:
        unit_counts = np.zeros((len(UNIT_TYPES) + 1,), dtype=np.uint16)
        if obs['unit_counts'].size > 0:
            unit_types = self._unit_type_lookup[obs['unit_counts'][:, 0]]
            unit_counts[unit_types] = obs['unit_counts'][:, 1]
        return unit_counts


class UpgradesFeature(BoxFeature):
    """ A feature represented by a boolean vector that indicates whether certain upgrades are present or not. """

    def __init__(self, upgrade_set: Optional[List[int]] = None) -> None:
        if upgrade_set is None:
            self._upgrade_set = np.asarray([int(u) for u in Upgrades])
        else:
            self._upgrade_set = np.asarray(upgrade_set)
        super().__init__(low=False, high=True, shape=self._upgrade_set.shape, dtype=np.bool)

    def extract(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.isin(self._upgrade_set, obs['upgrades'], assume_unique=True)


SCALAR_FEATURES = {
    "action_result": lambda env_config:  PaddedSequenceFeature('action_result', 1, (), high=max(ActionResult.values())),
    "action_result_length": lambda env_config:  SequenceLengthFeature('action_result', 1),
    "alerts": lambda env_config:  PaddedSequenceFeature('alerts', 1, (), high=max(Alert.values())),
    "alerts_length": lambda env_config:  SequenceLengthFeature('alerts', 1),
    "build_queue": lambda env_config: PaddedSequenceFeature(
        'build_queue', env_config.max_build_queue, (len(UnitLayer),)),
    "build_queue_length": lambda env_config: SequenceLengthFeature('build_queue', env_config.max_build_queue),
    "cargo": lambda env_config: PaddedSequenceFeature('cargo', env_config.max_cargo, (len(UnitLayer),)),
    "cargo_length": lambda env_config: SequenceLengthFeature('cargo', env_config.max_cargo),
    "cargo_slots_available": lambda env_config: LookupFeature(
        'cargo_slots_available', 0, np.iinfo(np.uint16).max, (1,)),
    "control_groups": lambda env_config: LookupFeature('control_groups', 0, np.iinfo(np.uint16).max, (10, 2)),
    "game_loop": lambda env_config: LookupFeature('game_loop', 0, np.iinfo(np.uint16).max, (1,)),
    "multi_select": lambda env_config: PaddedSequenceFeature(
        'multi_select', env_config.max_multi_select, (len(UnitLayer),)),
    "multi_select_length": lambda env_config: SequenceLengthFeature('multi_select', env_config.max_multi_select),
    "player": lambda env_config: LookupFeature('player', 0, np.iinfo(np.uint16).max, (len(Player),)),
    "production_queue": lambda env_config: PaddedSequenceFeature(
        'production_queue', env_config.max_production_queue, (len(ProductionQueue),)),
    "production_queue_length": lambda env_config: SequenceLengthFeature(
        'production_queue', env_config.max_production_queue),
    "score_cumulative": lambda env_config: LookupFeature(
        'score_cumulative', 0, np.iinfo(np.uint16).max, (len(ScoreCumulative),)),
    "score_by_category": lambda env_config: LookupFeature(
        'score_by_category', 0, np.iinfo(np.uint16).max, (len(ScoreByCategory), len(ScoreCategories))),
    "score_by_vital": lambda env_config: LookupFeature(
        'score_by_vital', 0, np.iinfo(np.uint16).max, (len(ScoreByVital), len(ScoreVitals))),
    "single_select": lambda env_config: PaddedSequenceFeature('single_select', 1, (len(UnitLayer),)),
    "single_select_length": lambda env_config: SequenceLengthFeature('single_select', 1),
    "available_actions": lambda env_config: AvailableActionsFeature(
        len(pysc2.lib.actions.FUNCTIONS), env_config.available_actions),
    "upgrades": lambda env_config: UpgradesFeature(env_config.upgrade_set),
    "unit_counts": lambda env_config: UnitCountsFeature(),
    "home_race_requested": lambda env_config: LookupFeature('home_race_requested', 0, len(Race), (1,)),
    "away_race_requested": lambda env_config: LookupFeature('away_race_requested', 0, len(Race), (1,)),
}


def get_scalar_feature(name: str, config: 'SC2InterfaceConfig') -> Feature:
    return SCALAR_FEATURES[name](config)


def get_screen_feature(name: str, config: 'SC2InterfaceConfig') -> Feature:
    feature = getattr(pysc2.lib.features.SCREEN_FEATURES, name)
    return LookupFeature(feature.name, low=0, high=feature.scale - 1, shape=config.dimension_screen,
                         dtype=(np.uint8 if feature.scale <= 256 else np.uint16), sub_space='feature_screen')


def get_minimap_feature(name: str, config: 'SC2InterfaceConfig') -> Feature:
    feature = getattr(pysc2.lib.features.MINIMAP_FEATURES, name)
    return LookupFeature(feature.name, low=0, high=feature.scale, shape=config.dimension_screen,
                         dtype=(np.uint8 if feature.scale <= 256 else np.uint16), sub_space='feature_minimap')


@gin.register
@dataclass(frozen=True)
class SC2InterfaceConfig:
    dimension_screen: Tuple[int, int] = (64, 64)
    dimension_minimap: Tuple[int, int] = (64, 64)
    screen_features: Sequence[str] = ('visibility_map', 'player_relative', 'unit_type', 'selected',
                                      'unit_hit_points_ratio', 'unit_energy_ratio', 'unit_density_aa')
    minimap_features: Sequence[str] = ('camera', 'player_relative', 'alerts')
    scalar_features: Sequence[str] = ('game_loop', 'available_actions', 'player')
    available_actions: Optional[Sequence[int]] = None
    upgrade_set: Optional[Sequence[int]] = None
    max_step_mul: int = 16
    max_multi_select: int = 64
    max_cargo: int = 8
    max_build_queue: int = 8
    max_production_queue: int = 16


@gin.register
class SC2ObservationSpace(ObservationSpace):
    """ A StarCraft II observation space. """

    def __init__(self, config: SC2InterfaceConfig) -> None:
        super().__init__()
        self._dimension_screen = config.dimension_screen
        self._dimension_minimap = config.dimension_minimap
        self._features = {
            'scalar_features': {name: get_scalar_feature(name, config) for name in config.scalar_features},
            'screen_features': {name: get_screen_feature(name, config) for name in config.screen_features},
            'minimap_features': {name: get_minimap_feature(name, config) for name in config.minimap_features},
        }
        self._specs = tree.map_structure(lambda feature: feature.spec(), self._features)

    @property
    def dimension_screen(self) -> Tuple[int, int]:
        return self._dimension_screen

    @property
    def dimension_minimap(self) -> Tuple[int, int]:
        return self._dimension_minimap

    @property
    def specs(self) -> Dict:
        return self._specs

    def transform(self, observation: Dict) -> Dict:
        raise NotImplementedError

    def transform_back(self, observation: Dict) -> Dict:
        return tree.map_structure(lambda feature: feature.extract(observation), self._features)


@gin.register
class SC2ActionSpace(ActionSpace):
    """ A StarCraft II action space. """

    def __init__(self, config: SC2InterfaceConfig) -> None:
        super().__init__()
        self._dimension_screen = config.dimension_screen
        self._dimension_minimap = config.dimension_minimap
        self._max_step_mul = config.max_step_mul

        def _arg_size(arg_type: pysc2.lib.actions.ArgumentType):
            if arg_type.name in ['screen', 'screen2']:
                return np.prod(self._dimension_screen)
            elif arg_type.name == 'minimap':
                return np.prod(self._dimension_minimap)
            elif arg_type.name == 'select_unit_id':
                return config.max_multi_select
            elif arg_type.name == 'build_queue_id':
                return config.max_build_queue
            elif arg_type.name == 'unload_id':
                return config.max_cargo
            else:
                return np.prod(arg_type.sizes)

        self._specs = dict(
            {'action_type': gym.spaces.Discrete(len(pysc2.lib.actions.FUNCTIONS)),
             'step_mul': gym.spaces.Discrete(n=self._max_step_mul)},
            **({arg_type.name: gym.spaces.Discrete(n=_arg_size(arg_type)) for arg_type in pysc2.lib.actions.TYPES}),
        )

    @property
    def specs(self) -> Dict[str, gym.spaces.Space]:
        return self._specs

    def no_op(self) -> Dict:
        no_op = {action_name: -1 for action_name in self._specs.keys()}
        no_op['action_type'] = 0
        no_op['step_mul'] = 0
        return no_op

    def transform(self, action: Dict) -> Tuple[pysc2.lib.actions.FunctionCall, int]:
        def _transform_arg(arg, arg_type):
            if arg_type.name in ['screen', 'screen2']:
                y = arg // self._dimension_screen[1]
                x = arg - y * self._dimension_screen[0]
                return x, y
            elif arg_type.name in ['minimap']:
                y = arg // self._dimension_minimap[1]
                x = arg - y * self._dimension_minimap[0]
                return x, y
            return (arg,)

        function = pysc2.lib.actions.FUNCTIONS[action['action_type']]
        function_call = pysc2.lib.actions.FunctionCall(
            function.id, [_transform_arg(action[t.name], t) for t in function.args])
        return function_call, action['step_mul'] + 1

    def transform_back(self, action: pysc2.lib.actions.FunctionCall, step_mul: int) -> Dict:
        def _transform_arg(_arg, _arg_type):
            if _arg_type.name in ['screen', 'screen2']:
                return _arg[1] * self._dimension_screen[1] + _arg[0]
            elif _arg_type.name in ['minimap']:
                return _arg[1] * self._dimension_minimap[1] + _arg[0]
            else:
                if isinstance(_arg, list):
                    assert len(_arg) == 1
                    _arg = _arg[0]
                # clip actions that exceed their max value
                # TODO: maybe there is a better way to handle such actions?
                return min(self._specs[_arg_type.name].n - 1, _arg)

        transformed_action = {action_name: -1 for action_name in self._specs.keys()}
        transformed_action['action_type'] = action.function
        for arg, arg_type in zip(action.arguments, pysc2.lib.actions.FUNCTIONS[action.function].args):
            transformed_action[arg_type.name] = _transform_arg(arg, arg_type)
        transformed_action['step_mul'] = step_mul - 1
        return transformed_action


def matchup_identifier(map_name: str, agent_race: str, bot_race: str, bot_difficulty: str, bot_build: str):
    return f'{map_name}_{agent_race[0]}_{bot_race[0]}_{bot_difficulty}_{bot_build}'


@gin.register
class SC2SingleAgentEnv(gym.Env, EnvMeta):
    """ A StarCraft II environment where the opponent is controlled by a built-in AI. """
    def __init__(self,
                 interface_config: SC2InterfaceConfig = gin.REQUIRED,
                 observation_space: SC2ObservationSpace = gin.REQUIRED,
                 action_space: SC2ActionSpace = gin.REQUIRED,
                 map_name: str = gin.REQUIRED,
                 battle_net_map: bool = False,
                 agent_race: str = 'terran',
                 agent_name: str = 'Hambbe',
                 bot_race: str = 'zerg',
                 bot_difficulty: str = 'easy',
                 bot_build: str = 'random',
                 visualize: bool = False,
                 realtime: bool = False,
                 save_replay_episodes: int = 0,
                 replay_dir: Optional[str] = None,
                 replay_prefix: Optional[str] = None,
                 game_steps_per_episode: Optional[int] = None,
                 score_index: Optional[int] = None,
                 score_multiplier: Optional[float] = None,
                 disable_fog: bool = False,
                 ensure_available_actions: bool = True,
                 version: Optional[str] = None,
                 random_seed: Optional[int] = None):
        super().__init__()
        self._interface_config = interface_config
        self._observation_space = observation_space
        self._action_space = action_space
        self._map_name = map_name
        self._battle_net_map = battle_net_map
        self._agent_race = agent_race
        self._agent_name = agent_name
        self._bot_race = bot_race
        self._bot_difficulty = bot_difficulty
        self._bot_build = bot_build
        self._visualize = visualize
        self._realtime = realtime
        self._save_replay_episodes = save_replay_episodes
        self._replay_dir = replay_dir
        self._replay_prefix = replay_prefix or self.level_name
        self._game_steps_per_episode = game_steps_per_episode
        self._score_index = score_index
        self._score_multiplier = score_multiplier
        self._disable_fog = disable_fog
        self._ensure_available_actions = ensure_available_actions
        self._version = version
        self._random_seed = random_seed
        self._env: Optional[sc2_env.SC2Env] = None

    def launch(self, random_seed: Optional[int] = None):
        self._launch(random_seed=random_seed)

    def _launch(self, random_seed: Optional[int] = None):
        assert self._env is None, "Cannot start environment twice."
        sc2_map = pysc2.maps.get(self._map_name)
        players = [sc2_env.Agent(sc2_env.Race[self._agent_race], self._agent_name)]
        if sc2_map.players >= 2:
            players.append(sc2_env.Bot(
                race=sc2_env.Race[self._bot_race],
                difficulty=sc2_env.Difficulty[self._bot_difficulty],
                build=sc2_env.BotBuild[self._bot_build]))

        self._env = sc2_env.SC2Env(
            map_name=self._map_name,
            battle_net_map=self._battle_net_map,
            players=players,
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=self._interface_config.dimension_screen,
                    minimap=self._interface_config.dimension_minimap
                ),
                use_unit_counts=True,  #
                hide_specific_actions=True,  #
            ),
            visualize=self._visualize,
            realtime=self._realtime,
            save_replay_episodes=self._save_replay_episodes,
            replay_dir=self._replay_dir,
            replay_prefix=self._replay_prefix,
            game_steps_per_episode=self._game_steps_per_episode,
            score_index=self._score_index,
            score_multiplier=self._score_multiplier,
            random_seed=random_seed,
            disable_fog=self._disable_fog,
            ensure_available_actions=self._ensure_available_actions,
            version=self._version,
        )

    def reset(self):
        time_steps = self._env.reset()
        return self._observation_space.transform_back(time_steps[0].observation)

    def restart(self, random_seed: Optional[int] = None):
        self.close()
        self._launch(random_seed)

    def step(self, action: Dict) -> StepOutput:
        raw_action, step_mul = self._action_space.transform({k: int(np.squeeze(a)) for k, a in action.items()})
        try:
            time_steps = self._env.step(actions=[raw_action], step_mul=step_mul)
            return StepOutput(reward=np.float32(time_steps[0].reward),
                              info={'game_loop': time_steps[0].observation['game_loop'][0]},
                              done=time_steps[0].step_type == StepType.FIRST,
                              observation=self._observation_space.transform_back(time_steps[0].observation))
        except Exception as e:
            logger.error(f"Failed to take step '{raw_action}' (stacktrace below). Restart env.")
            traceback.print_exc()
            self.restart()
            return StepOutput(reward=np.float32(0.0), info={'game_loop': np.int32(0)}, done=np.bool(True),
                              observation=self.reset())

    def close(self) -> None:
        try:
            if self._env is not None:
                self._env.close()
        except Exception as e:
            print(f'Failed to close environment: {e}')
            print(traceback.format_exc())
        self._env = None

    def render(self, mode='human'):
        raise NotImplementedError

    @property
    def level_name(self) -> str:
        return f"{self._map_name}_{self._agent_race[0]}_{self._bot_race[0]}_{self._bot_difficulty}_{self._bot_build}"

    @property
    def action_space(self) -> SC2ActionSpace:
        return self._action_space

    @property
    def observation_space(self) -> SC2ObservationSpace:
        return self._observation_space


@gin.register
class SC2LanEnv(gym.Env, EnvMeta):
    """ A StarCraft II environment that connects to an already launched LAN game. """
    def __init__(self,
                 host: str = gin.REQUIRED,
                 config_port: int = gin.REQUIRED,
                 interface_config: SC2InterfaceConfig = gin.REQUIRED,
                 observation_space: SC2ObservationSpace = gin.REQUIRED,
                 action_space: SC2ActionSpace = gin.REQUIRED,
                 agent_race: str = 'terran',
                 agent_name: str = 'Hambbe',
                 visualize: bool = False,
                 realtime: bool = False,
                 replay_dir: Optional[str] = None,
                 replay_prefix: Optional[str] = None):
        super().__init__()
        self._host = host
        self._config_port = config_port
        self._interface_config = interface_config
        self._observation_space = observation_space
        self._action_space = action_space
        self._agent_race = agent_race
        self._agent_name = agent_name
        self._visualize = visualize
        self._realtime = realtime
        self._replay_dir = replay_dir
        self._replay_prefix = replay_prefix
        self._env: Optional[lan_sc2_env.LanSC2Env] = None

    def launch(self):
        self._launch()

    def _launch(self):
        assert self._env is None, "Cannot start environment twice."
        self._env = lan_sc2_env.LanSC2Env(
            host=self._host,
            config_port=self._config_port,
            race=sc2_env.Race[self._agent_race],
            name=self._agent_name,
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=self._interface_config.dimension_screen,
                    minimap=self._interface_config.dimension_minimap
                ),
                use_unit_counts=True,
                hide_specific_actions=True,
            ),
            visualize=self._visualize,
            realtime=self._realtime,
            replay_dir=self._replay_dir,
            replay_prefix=self._replay_prefix
        )

    def reset(self):
        time_steps = self._env.reset()
        return self._observation_space.transform_back(time_steps[0].observation)

    def restart(self):
        self.close()
        self._launch()

    def step(self, action: Dict) -> StepOutput:
        raw_action, step_mul = self._action_space.transform({k: int(np.squeeze(a)) for k, a in action.items()})
        try:
            time_steps = self._env.step(actions=[raw_action], step_mul=step_mul)
            return StepOutput(reward=np.float32(time_steps[0].reward),
                              info={'game_loop': time_steps[0].observation['game_loop'][0]},
                              done=time_steps[0].step_type == StepType.FIRST,
                              observation=self._observation_space.transform_back(time_steps[0].observation))
        except Exception as e:
            logger.error(f"Failed to take step '{raw_action}' (stacktrace below). Restart env.")
            traceback.print_exc()
            self.restart()
            return StepOutput(reward=np.float32(0.0), info={'game_loop': np.int32(0)}, done=np.bool(True),
                              observation=self.reset())

    def close(self) -> None:
        try:
            if self._env is not None:
                self._env.close()
        except Exception as e:
            print(f'Failed to close environment: {e}')
            print(traceback.format_exc())
        self._env = None

    def render(self, mode='human'):
        raise NotImplementedError

    @property
    def level_name(self) -> str:
        return f"{self._map_name}_{self._agent_race[0]}_{self._bot_race[0]}_{self._bot_difficulty}_{self._bot_build}"

    @property
    def action_space(self) -> SC2ActionSpace:
        return self._action_space

    @property
    def observation_space(self) -> SC2ObservationSpace:
        return self._observation_space


class SC2Maps(object):
    def __init__(self, sc2_path: str) -> None:
        super().__init__()
        self._sc2_path = sc2_path
        self._map_name_to_aliases = dict()
        self._alias_to_map_name = dict()
        for map_path in glob.glob(os.path.join(self._sc2_path, 'Maps', '**', '*.SC2Map'), recursive=True):
            map = sc2reader.load_map(map_path)
            aliases = []
            for filename in map.archive.files:
                filename_dec = filename.decode()
                if 'SC2Data\\LocalizedData\\GameStrings.txt' in filename_dec:
                    locale, _ = filename_dec.split('.', 1)
                    game_strings_file = map.archive.read_file(filename_dec)
                    if game_strings_file:
                        for line in game_strings_file.decode('utf8').split("\r\n"):
                            if len(line) == 0:
                                continue
                            key, value = line.split('=', 1)
                            if key == 'DocInfo/Name':
                                name = value.split('//', 1)[0].strip()
                                aliases.append(name)
                                self._alias_to_map_name[name] = map.name
                                break
            self._map_name_to_aliases[map.name] = set(aliases)

    def aliases(self, map_name: str) -> Set[str]:
        return self._map_name_to_aliases[map_name]

    def normalize_map_name(self, map_name: str) -> str:
        return self._alias_to_map_name[map_name]
