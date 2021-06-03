import logging
import os
import platform
import sys
import time
import traceback
from typing import Type, Tuple, Optional

import gin
import portpicker
import tensorflow as tf
from absl import app
from absl import flags
from pysc2 import run_configs, maps
from pysc2.env import sc2_env, lan_sc2_env
from pysc2.lib import renderer_human
from pysc2.lib.protocol import Status
from s2clientprotocol import sc2api_pb2 as sc_pb

from sc2_imitation_learning.common.utils import gin_register_external_configurables, make_dummy_action
from sc2_imitation_learning.environment.sc2_environment import SC2LanEnv

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

flags.DEFINE_string('agent_dir', default=None, help='Path to the directory where the agent is stored.')
flags.DEFINE_multi_string('gin_file', ['configs/1v1/play_agent_vs_human.gin'], 'List of paths to Gin config files.')
flags.DEFINE_multi_string('gin_param', None, 'List of Gin parameter bindings.')
flags.DEFINE_bool('human', False, 'Human.')

FLAGS = flags.FLAGS

gin_register_external_configurables()


@gin.configurable
def human(map_name: str = gin.REQUIRED,
          render: Optional[bool] = None,
          host: str = '127.0.0.1',
          config_port: int = 14380,
          remote: Optional[str] = None,
          realtime: bool = False,
          fps: float = 22.4,
          rgb_screen_size: Optional[Tuple[int, int]] = None,
          rgb_minimap_size: Optional[Tuple[int, int]] = None,
          feature_screen_size: Optional[Tuple[int, int]] = None,
          feature_minimap_size: Optional[Tuple[int, int]] = None,
          race: str = 'zerg',
          player_name: str = '<unknown>'):
    if render is None:
        render = platform.system() == "Linux"

    run_config = run_configs.get()
    map_inst = maps.get(map_name)

    ports = [config_port + p for p in range(5)]  # tcp + 2 * num_players
    if not all(portpicker.is_port_free(p) for p in ports):
        sys.exit("Need 5 free ports after the config port.")

    proc = None
    ssh_proc = None
    tcp_conn = None
    udp_sock = None
    try:
        proc = run_config.start(extra_ports=ports[1:], timeout_seconds=300, host=host, window_loc=(50, 50))

        tcp_port = ports[0]
        settings = {
            "remote": remote,
            "game_version": proc.version.game_version,
            "realtime": realtime,
            "map_name": map_inst.name,
            "map_path": map_inst.path,
            "map_data": map_inst.data(run_config),
            "ports": {
                "server": {"game": ports[1], "base": ports[2]},
                "client": {"game": ports[3], "base": ports[4]},
            }
        }

        create = sc_pb.RequestCreateGame(
            realtime=settings["realtime"],
            local_map=sc_pb.LocalMap(map_path=settings["map_path"]))
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(type=sc_pb.Participant)

        controller = proc.controller
        controller.save_map(settings["map_path"], settings["map_data"])
        controller.create_game(create)

        if remote is not None:
            ssh_proc = lan_sc2_env.forward_ports(
                remote, proc.host, [settings["ports"]["client"]["base"]],
                [tcp_port, settings["ports"]["server"]["base"]])

        tcp_conn = lan_sc2_env.tcp_server(lan_sc2_env.Addr(proc.host, tcp_port), settings)

        if remote is not None:
            udp_sock = lan_sc2_env.udp_server(
                lan_sc2_env.Addr(proc.host, settings["ports"]["client"]["game"]))

            lan_sc2_env.daemon_thread(
                lan_sc2_env.tcp_to_udp,
                (tcp_conn, udp_sock, lan_sc2_env.Addr(proc.host, settings["ports"]["server"]["game"])))

            lan_sc2_env.daemon_thread(lan_sc2_env.udp_to_tcp, (udp_sock, tcp_conn))

        join = sc_pb.RequestJoinGame()
        join.shared_port = 0  # unused
        join.server_ports.game_port = settings["ports"]["server"]["game"]
        join.server_ports.base_port = settings["ports"]["server"]["base"]
        join.client_ports.add(game_port=settings["ports"]["client"]["game"],
                              base_port=settings["ports"]["client"]["base"])

        # join.observed_player_id = 2
        join.race = sc2_env.Race[race]
        join.player_name = player_name
        if render:
            join.options.raw = True
            join.options.score = True
            join.options.raw_affects_selection = True
            join.options.raw_crop_to_playable_area = True
            join.options.show_cloaked = True
            join.options.show_burrowed_shadows = True
            join.options.show_placeholders = True
            if feature_screen_size and feature_minimap_size:
                fl = join.options.feature_layer
                fl.width = 24
                fl.resolution.x = feature_screen_size[0]
                fl.resolution.y = feature_screen_size[1]
                fl.minimap_resolution.x = feature_minimap_size[0]
                fl.minimap_resolution.y = feature_minimap_size[1]
            if rgb_screen_size and rgb_minimap_size:
                join.options.render.resolution.x = rgb_screen_size[0]
                join.options.render.resolution.y = rgb_screen_size[1]
                join.options.render.minimap_resolution.x = rgb_minimap_size[0]
                join.options.render.minimap_resolution.y = rgb_minimap_size[1]
        controller.join_game(join)

        if render:
            renderer = renderer_human.RendererHuman(fps=fps, render_feature_grid=False)
            while controller.status == Status.init_game:
                print("Waiting in status = Status.init_game...")
                time.sleep(1)
            renderer.run(run_configs.get(), controller, max_episodes=1)
        else:  # Still step forward so the Mac/Windows renderer works.
            while True:
                frame_start_time = time.time()
                if not realtime:
                    controller.step()
                obs = controller.observe()
                if obs.player_result:
                    break
                time.sleep(max(0, frame_start_time - time.time() + 1 / fps))
    except KeyboardInterrupt:
        pass
    finally:
        if tcp_conn:
            tcp_conn.close()
        if proc:
            proc.close()
        if udp_sock:
            udp_sock.close()
        if ssh_proc:
            ssh_proc.terminate()
            for _ in range(5):
                if ssh_proc.poll() is not None:
                    break
                time.sleep(1)
            if ssh_proc.poll() is None:
                ssh_proc.kill()
                ssh_proc.wait()


@gin.configurable
def agent(env_fn: Type[SC2LanEnv] = gin.REQUIRED) -> None:
    agent = tf.saved_model.load(FLAGS.agent_dir)
    agent_state = agent.initial_state(1)

    env = env_fn()
    env.launch()
    episode_reward = 0.
    episode_frames = 0
    episode_steps = 0
    try:
        reward, done, observation = 0., False, env.reset()
        action = make_dummy_action(env.action_space, num_batch_dims=1)
        while not done:
            env_outputs = (
                tf.constant([reward], dtype=tf.float32),
                tf.constant([episode_steps == 0], dtype=tf.bool),
                tf.nest.map_structure(lambda o: tf.constant([o], dtype=tf.dtypes.as_dtype(o.dtype)), observation))
            agent_output, agent_state = agent(action, env_outputs, agent_state)
            action = tf.nest.map_structure(lambda t: t.numpy(), agent_output.actions)
            reward, _, done, observation = env.step(action)
            episode_reward += reward
            episode_frames += action['step_mul'] + 1
            episode_steps += 1
    except Exception as e:
        logger.error(f"Failed to play episode(stacktrace below).")
        traceback.print_exc()
    finally:
        logger.info(f"Episode completed: total reward={episode_reward}, frames={episode_frames}, "
                    f"steps={episode_steps}")
        env.close()


def main(_):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    if FLAGS.human:
        human()
    else:
        agent()


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    app.run(main)
