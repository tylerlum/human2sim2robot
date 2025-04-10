from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as pb
import tyro
import yaml
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)

from human2sim2robot.human_demo.utils.pybullet_utils import (
    draw_collision_spheres,
    get_num_actuatable_joints,
    set_robot_state,
)


@dataclass
class Args:
    robot_file: str
    visualize_collision_spheres: bool = False


def main(args: Args):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80)

    # Read in config
    robot_yml_path = join_path(get_robot_configs_path(), args.robot_file)
    assert Path(robot_yml_path).exists(), (
        f"Robot YAML path {robot_yml_path} does not exist"
    )
    robot_yml = load_yaml(robot_yml_path)

    # Get urdf path
    urdf_path = robot_yml["robot_cfg"]["kinematics"]["urdf_path"]
    urdf_path = Path(join_path(get_assets_path(), urdf_path))
    assert urdf_path.exists(), f"URDF path {urdf_path} does not exist"

    # Start visualizer
    pb.connect(pb.GUI)
    robot = pb.loadURDF(
        str(urdf_path),
        useFixedBase=True,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
    )
    num_actuatable_joints = get_num_actuatable_joints(robot)
    print(f"num_actuatable_joints = {num_actuatable_joints}")

    # Set robot state
    ROBOT_STATE_DEFAULT = np.zeros(num_actuatable_joints)
    set_robot_state(robot, ROBOT_STATE_DEFAULT)

    if args.visualize_collision_spheres:
        # Get collision spheres path
        collision_spheres_yml_path = robot_yml["robot_cfg"]["kinematics"][
            "collision_spheres"
        ]
        collision_spheres_yml_path = Path(
            join_path(get_robot_configs_path(), collision_spheres_yml_path)
        )
        assert collision_spheres_yml_path.exists(), (
            f"Collision spheres yaml path {collision_spheres_yml_path} does not exist"
        )

        # Read in collision spheres
        collision_config = yaml.safe_load(
            open(
                collision_spheres_yml_path,
                "r",
            )
        )
        draw_collision_spheres(robot, collision_config)

    print("=" * 80)
    print("Setting breakpoint to allow user to visualize robot")
    print("=" * 80 + "\n")
    breakpoint()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
