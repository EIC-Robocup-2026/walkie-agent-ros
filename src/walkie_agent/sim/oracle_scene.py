"""Stand in for walkie-vision in a simulation: draw a scene file for every camera image.

    pixi run oracle-chula
    python -m walkie_agent.sim.oracle_scene --use-sim-time config/scenes/chula_living_room.toml

It reads a fixed scene (walkie_agent.sim.scene_file) and runs
OraclePerception on a node of its own, which publishes walkie-vision's
people and object arrays on /walkie_vision/people and /walkie_vision/objects
for every camera_info of the robot's head camera. Run it in the
simulation's ROS graph, after the simulation is up: until TF has the map
frame, images are skipped with a warning.

It gives the oracle no clock epoch: the scene is the same at every stamp,
so a pose cannot be paired with another clock's scene, and the oracle's TF
buffer, on the node's clock, clears itself when sim time goes back.

A bad scene file stops it before rclpy starts, with exit code 2.
"""

import argparse
import logging
import sys
from collections.abc import Sequence

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.parameter import Parameter

from walkie_agent.sim.oracle_perception import (
    HEAD_CAMERA_INFO_TOPIC,
    OracleConfig,
    OraclePerception,
)
from walkie_agent.sim.scene_file import SceneFile, load_scene

logger = logging.getLogger(__name__)

NODE_NAME = "oracle_perception"
STARTUP_ERROR_EXIT_CODE = 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="oracle_scene",
        description="Publish walkie-vision's arrays for a fixed scene, one pair per camera image.",
    )
    parser.add_argument("scene", help="The scene file, TOML (walkie_agent.sim.scene_file)")
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        help="Run the node on the simulation's /clock, as the agent does in a simulation",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="The seed of the detection noise (default 0)"
    )
    parser.add_argument(
        "--camera-info-topic",
        default=HEAD_CAMERA_INFO_TOPIC,
        help=f"The camera_info topic to draw for (default {HEAD_CAMERA_INFO_TOPIC})",
    )
    return parser.parse_args(argv)


def oracle_config(loaded: SceneFile, args: argparse.Namespace) -> OracleConfig:
    """The OracleConfig for a scene file and the command line."""
    fields = {"seed": args.seed, "camera_info_topic": args.camera_info_topic}
    if loaded.classes is not None:
        fields["classes"] = loaded.classes
    return OracleConfig(**fields)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        loaded = load_scene(args.scene)
        config = oracle_config(loaded, args)
    except (OSError, ValueError) as error:
        print(f"oracle_scene: {error}", file=sys.stderr)
        return STARTUP_ERROR_EXIT_CODE

    rclpy.init()
    node = rclpy.create_node(
        NODE_NAME,
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, args.use_sim_time)],
    )
    scene = loaded.scene
    oracle = OraclePerception(node, lambda _stamp: scene, config)
    logger.info(
        "Drawing %s (world %s): %d people and %d objects, classes %s, for %s on %s and %s, "
        "stream_id %s. This is a simulated scene, not perception.",
        args.scene,
        loaded.world or "not named",
        len(scene.people),
        len(scene.objects),
        list(config.classes),
        config.camera_info_topic,
        config.people_topic,
        config.objects_topic,
        oracle.stream_id,
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        oracle.destroy()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
