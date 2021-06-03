import json
import re
from typing import Tuple, Iterator, Dict, Text, Optional

import mpyq
import six
from absl import logging
from pysc2 import run_configs, maps
from pysc2.env import sc2_env, environment
from pysc2.env.sc2_env import possible_results
from pysc2.lib import features
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from s2clientprotocol import sc2api_pb2 as sc_pb

from sc2_imitation_learning.environment.sc2_environment import SC2ObservationSpace, SC2ActionSpace, SC2InterfaceConfig


def get_replay_info(replay_path: Text) -> Dict:
    with open(replay_path, 'rb') as f:
        archive = mpyq.MPQArchive(f).extract()
        metadata = json.loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))
        return metadata


def get_game_version(replay_data: bytes) -> str:
    replay_io = six.BytesIO()
    replay_io.write(replay_data)
    replay_io.seek(0)
    archive = mpyq.MPQArchive(replay_io).extract()
    metadata = json.loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))
    version = metadata["GameVersion"]
    return ".".join(version.split(".")[:-1])


class ReplayProcessor(object):

    def __init__(
            self,
            replay_path: str,
            interface_config: SC2InterfaceConfig,
            observation_space: SC2ObservationSpace,
            action_space: SC2ActionSpace,
            discount: float = 1.,
            score_index: Optional[int] = None,
            score_multiplier: Optional[float] = None,
            disable_fog: bool = False,
            map_path: str = None,
            observed_player_id: int = 1,
            version: Optional[str] = None) -> None:
        super().__init__()
        self._replay_path = replay_path

        self.observation_space = observation_space
        self.action_space = action_space

        self._discount = discount
        self._disable_fog = disable_fog

        self._default_score_index = score_index or 0
        self._default_score_multiplier = score_multiplier
        self._default_episode_length = None

        self._run_config = run_configs.get(version=version)

        agent_interface_format = sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(
                screen=interface_config.dimension_screen,
                minimap=interface_config.dimension_minimap
            ),
            use_unit_counts=True,
            hide_specific_actions=True,
        )
        interface = sc2_env.SC2Env._get_interface(agent_interface_format, False)

        self._launch(replay_path, interface, map_path, observed_player_id)

        self._finalize(agent_interface_format)

    def _launch(self, replay_path, interface, map_path, observed_player_id):
        replay_data = self._run_config.replay_data(replay_path)

        version = get_game_version(replay_data)
        logging.info(f"Start SC2 process (game version={version})...")

        self._sc2_proc = self._run_config.start(
            # version=version,
            full_screen=False
        )
        self._controller = self._sc2_proc.controller
        self.replay_info = self._controller.replay_info(replay_data)

        map_name = re.sub(r"[ '-]|[LTRS]E$", "", self.replay_info.map_name)
        if map_name == 'MacroEconomy':
            map_name = 'CollectMineralsAndGas'
        map_inst = maps.get(map_name)

        start_replay = sc_pb.RequestStartReplay(
            replay_data=replay_data,
            options=interface,
            disable_fog=False,
            observed_player_id=observed_player_id
        )

        def _default_if_none(value, default):
            return default if value is None else value

        self._score_index = _default_if_none(self._default_score_index, map_inst.score_index)
        self._score_multiplier = _default_if_none(self._default_score_multiplier, map_inst.score_multiplier)
        self._episode_length = _default_if_none(self._default_episode_length, map_inst.game_steps_per_episode)

        map_path = map_path or self.replay_info.local_map_path
        if map_path:
            start_replay.map_data = self._run_config.map_data(map_path)

        self._controller.start_replay(start_replay)

    def _finalize(
            self,
            agent_interface_format: sc2_env.AgentInterfaceFormat,
    ) -> None:
        self._features = features.features_from_game_info(
            game_info=self._controller.game_info(),
            agent_interface_format=agent_interface_format
        )

        self._state = environment.StepType.FIRST

        self._episode_steps = 0

        logging.info('Replay environment is ready for replay: %s', self._replay_path)

    def iterator(self) -> Iterator[Tuple[environment.TimeStep, dict]]:
        # returns ((s, r, d, \gamma), a) samples
        curr_time_step, _ = self.observe()
        while curr_time_step.step_type != environment.StepType.LAST:
            next_time_step, action = self.next(1)
            yield curr_time_step, action
            curr_time_step = next_time_step
        yield curr_time_step, self.action_space.no_op()

    def next(self, step_mul: int) -> Tuple[environment.TimeStep, dict]:
        if step_mul <= 0:
            raise ValueError(f"expect step_mul > 0, got {step_mul}")

        if self._state == environment.StepType.LAST:
            raise RuntimeError("Replay already ended.")

        self._state = environment.StepType.MID

        self._controller.step(step_mul)

        observation = self._observe(step_mul)

        return observation

    def observe(self) -> Tuple[environment.TimeStep, dict]:
        return self._observe()

    def _observe(self, step_mul=1) -> Tuple[environment.TimeStep, dict]:
        raw_observation = self._controller.observe()

        actions = []
        try:
            actions = [self._features.reverse_action(action) for action in raw_observation.actions]
        except ValueError as e:
            logging.warning(f"Failed to reverse_action: {e}")
        actions = actions if len(actions) > 0 else [FunctionCall(FUNCTIONS["no_op"].id, [])]

        observation = self._features.transform_obs(raw_observation)

        self._episode_steps = observation['game_loop'][0]

        outcome = 0
        discount = self._discount
        episode_complete = raw_observation.player_result
        if episode_complete:
            self._state = environment.StepType.LAST
            discount = 0
            player_id = raw_observation.observation.player_common.player_id
            for result in raw_observation.player_result:
                if result.player_id == player_id:
                    outcome = possible_results.get(result.result, 0)

        if self._score_index >= 0:  # Game score, not win/loss reward.
            cur_score = observation["score_cumulative"][self._score_index]
            if self._episode_steps == 0:  # First reward is always 0.
                reward = 0
            else:
                reward = max(0, cur_score - self._last_score)
            self._last_score = cur_score
        else:
            reward = outcome

        observation = self.observation_space.transform_back(observation)
        action = self.action_space.transform_back(actions[0], step_mul)

        time_step = environment.TimeStep(
            step_type=self._state, reward=reward * self._score_multiplier, discount=discount, observation=observation
        )

        return time_step, action

    @property
    def state(self):
        return self._state

    def close(self) -> None:
        if self._controller is not None:
            self._controller.quit()
            self._controller = None

        if self._sc2_proc is not None:
            self._sc2_proc.close()
            self._sc2_proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
