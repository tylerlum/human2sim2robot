from __future__ import annotations

import pathlib
import time
from typing import List, Optional, Tuple

import numpy as np
import yaml
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from tqdm import tqdm

from human2sim2robot.human_demo.utils.collision_sphere_utils import (
    draw_collision_spheres,
    remove_collision_spheres,
)
from human2sim2robot.human_demo.utils.utils import (
    create_transform,
)


def get_link_name_to_idx(robot: int) -> dict:
    """
    Get link name to index mapping for robot
    """
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    link_name_to_idx = {}
    for i in range(pb.getNumJoints(robot)):
        joint_info = pb.getJointInfo(robot, i)
        link_name_to_idx[joint_info[12].decode("utf-8")] = i
    return link_name_to_idx


def get_joint_limits(
    robot: int,
) -> Tuple[List[float], List[float], List[float], List[str]]:
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    joint_lower_limits = []
    joint_upper_limits = []
    joint_ranges = []
    joint_names = []
    for i in range(pb.getNumJoints(robot)):
        joint_info = pb.getJointInfo(robot, i)
        if joint_info[2] == pb.JOINT_FIXED:
            continue
        joint_lower_limits.append(joint_info[8])
        joint_upper_limits.append(joint_info[9])
        joint_ranges.append(joint_info[9] - joint_info[8])
        joint_names.append(joint_info[1])
    return joint_lower_limits, joint_upper_limits, joint_ranges, joint_names


def visualize_transform(
    xyz: np.ndarray,
    rotation_matrix: np.ndarray,
    length: float = 0.5,
    lines: Optional[list] = None,
) -> list:
    """
    Visualize transform axes (mostly for debugging). x = red, y = green, z = blue
    """
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    T = create_transform(pos=xyz, rot=rotation_matrix)
    assert T.shape == (4, 4), T.shape

    origin = np.array([0, 0, 0])
    x_pos = np.array([length, 0, 0])
    y_pos = np.array([0, length, 0])
    z_pos = np.array([0, 0, length])

    tranformed_origin = T[:3, :3] @ origin + T[:3, 3]
    tranformed_x_pos = T[:3, :3] @ x_pos + T[:3, 3]
    tranformed_y_pos = T[:3, :3] @ y_pos + T[:3, 3]
    tranformed_z_pos = T[:3, :3] @ z_pos + T[:3, 3]

    LINE_WIDTH = 3
    RED_RGB = [1, 0, 0]
    GREEN_RGB = [0, 1, 0]
    BLUE_RGB = [0, 0, 1]

    if lines is None:
        lines = []

        lines.append(
            pb.addUserDebugLine(
                tranformed_origin,
                tranformed_x_pos,
                lineColorRGB=RED_RGB,
                lineWidth=LINE_WIDTH,
            )
        )
        lines.append(
            pb.addUserDebugLine(
                tranformed_origin,
                tranformed_y_pos,
                lineColorRGB=GREEN_RGB,
                lineWidth=LINE_WIDTH,
            )
        )
        lines.append(
            pb.addUserDebugLine(
                tranformed_origin,
                tranformed_z_pos,
                lineColorRGB=BLUE_RGB,
                lineWidth=LINE_WIDTH,
            )
        )
    else:
        pb.addUserDebugLine(
            tranformed_origin,
            tranformed_x_pos,
            replaceItemUniqueId=lines[0],
            lineColorRGB=RED_RGB,
            lineWidth=LINE_WIDTH,
        )
        pb.addUserDebugLine(
            tranformed_origin,
            tranformed_y_pos,
            replaceItemUniqueId=lines[1],
            lineColorRGB=GREEN_RGB,
            lineWidth=LINE_WIDTH,
        )
        pb.addUserDebugLine(
            tranformed_origin,
            tranformed_z_pos,
            replaceItemUniqueId=lines[2],
            lineColorRGB=BLUE_RGB,
            lineWidth=LINE_WIDTH,
        )
    return lines


FAR_AWAY_POSITION = [100, 100, 100]

BLUE_RGB = [0, 0, 1]
RED_RGB = [1, 0, 0]
GREEN_RGB = [0, 1, 0]
YELLOW_RGB = [1, 1, 0]
CYAN_RGB = [0, 1, 1]
MAGENTA_RGB = [1, 0, 1]
WHITE_RGB = [1, 1, 1]
BLACK_RGB = [0, 0, 0]

BLUE_RGBA = [*BLUE_RGB, 1]
RED_RGBA = [*RED_RGB, 1]
GREEN_RGBA = [*GREEN_RGB, 1]
YELLOW_RGBA = [*YELLOW_RGB, 1]
CYAN_RGBA = [*CYAN_RGB, 1]
MAGENTA_RGBA = [*MAGENTA_RGB, 1]
WHITE_RGBA = [*WHITE_RGB, 1]
BLACK_RGBA = [*BLACK_RGB, 1]

BLUE_TRANSLUCENT_RGBA = [*BLUE_RGB, 0.5]
RED_TRANSLUCENT_RGBA = [*RED_RGB, 0.5]
GREEN_TRANSLUCENT_RGBA = [*GREEN_RGB, 0.5]
YELLOW_TRANSLUCENT_RGBA = [*YELLOW_RGB, 0.5]
CYAN_TRANSLUCENT_RGBA = [*CYAN_RGB, 0.5]
MAGENTA_TRANSLUCENT_RGBA = [*MAGENTA_RGB, 0.5]
WHITE_TRANSLUCENT_RGBA = [*WHITE_RGB, 0.5]
BLACK_TRANSLUCENT_RGBA = [*BLACK_RGB, 0.5]


def get_joint_names(robot) -> List[str]:
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    num_total_joints = pb.getNumJoints(robot)
    return [
        pb.getJointInfo(robot, i)[1].decode("utf-8")
        for i in range(num_total_joints)
        if pb.getJointInfo(robot, i)[2] != pb.JOINT_FIXED
    ]


def get_link_names(robot) -> List[str]:
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    num_total_joints = pb.getNumJoints(robot)
    return [
        pb.getJointInfo(robot, i)[12].decode("utf-8")
        for i in range(num_total_joints)
        if pb.getJointInfo(robot, i)[2] != pb.JOINT_FIXED
    ]


def draw_collision_spheres_default_config(robot, robot_file: str) -> None:
    COLLISION_SPHERES_YAML_PATH = load_yaml(
        join_path(get_robot_configs_path(), robot_file)
    )["robot_cfg"]["kinematics"]["collision_spheres"]
    COLLISION_SPHERES_YAML_PATH = pathlib.Path(
        join_path(get_robot_configs_path(), COLLISION_SPHERES_YAML_PATH)
    )
    assert COLLISION_SPHERES_YAML_PATH.exists()

    collision_config = yaml.safe_load(
        open(
            COLLISION_SPHERES_YAML_PATH,
            "r",
        )
    )
    draw_collision_spheres(
        robot=robot,
        config=collision_config,
    )


def remove_collision_spheres_default_config() -> None:
    remove_collision_spheres()


def set_robot_state(robot, q: np.ndarray) -> None:
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    actuatable_joint_idxs = get_actuatable_joint_idxs(robot)
    num_actuatable_joints = len(actuatable_joint_idxs)

    assert len(q.shape) == 1, f"q.shape: {q.shape}"
    assert q.shape[0] <= num_actuatable_joints, (
        f"q.shape: {q.shape}, num_actuatable_joints: {num_actuatable_joints}"
    )

    for i, joint_idx in enumerate(actuatable_joint_idxs):
        # q may not contain all the actuatable joints, so we assume that the joints not in q are all 0
        if i < len(q):
            pb.resetJointState(robot, joint_idx, q[i])
        else:
            pb.resetJointState(robot, joint_idx, 0)


def get_actuatable_joint_idxs(robot) -> List[int]:
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    num_total_joints = pb.getNumJoints(robot)
    actuatable_joint_idxs = [
        i
        for i in range(num_total_joints)
        if pb.getJointInfo(robot, i)[2] != pb.JOINT_FIXED
    ]
    return actuatable_joint_idxs


def get_num_actuatable_joints(robot) -> int:
    actuatable_joint_idxs = get_actuatable_joint_idxs(robot)
    return len(actuatable_joint_idxs)


def animate_robot(robot, qs: np.ndarray, dt: float) -> None:
    N_pts = qs.shape[0]
    last_update_time = time.time()
    for i in tqdm(range(N_pts), desc="Animating robot"):
        q = qs[i]
        set_robot_state(robot, q)

        time_since_last_update = time.time() - last_update_time
        if time_since_last_update <= dt:
            time.sleep(dt - time_since_last_update)
        else:
            print(f"WARNING: Time since last update {time_since_last_update} > dt {dt}")
        last_update_time = time.time()


def add_cuboid(halfExtents, position, orientation, rgbaColor=RED_RGBA):
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    # Create a visual shape for the cuboid
    visualShapeId = pb.createVisualShape(
        shapeType=pb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=rgbaColor
    )  # Red color

    # Create a collision shape for the cuboid
    collisionShapeId = pb.createCollisionShape(
        shapeType=pb.GEOM_BOX, halfExtents=halfExtents
    )

    # Create the cuboid as a rigid body
    cuboidId = pb.createMultiBody(
        baseMass=1,  # Mass of the cuboid
        baseCollisionShapeIndex=collisionShapeId,
        baseVisualShapeIndex=visualShapeId,
        basePosition=position,
        baseOrientation=orientation,
    )
    return cuboidId


def add_sphere(radius, position, rgbaColor=BLUE_RGBA):
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    # Create a visual shape for the sphere
    visualShapeId = pb.createVisualShape(
        shapeType=pb.GEOM_SPHERE, radius=radius, rgbaColor=rgbaColor
    )  # Blue color

    # Create a collision shape for the sphere
    collisionShapeId = pb.createCollisionShape(shapeType=pb.GEOM_SPHERE, radius=radius)

    # Create the sphere as a rigid body
    sphereId = pb.createMultiBody(
        baseMass=1,  # Mass of the sphere
        baseCollisionShapeIndex=collisionShapeId,
        baseVisualShapeIndex=visualShapeId,
        basePosition=position,
    )  # Initial position (x, y, z)
    return sphereId


def move_sphere(sphereId, position):
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    pb.resetBasePositionAndOrientation(sphereId, position, [0, 0, 0, 1])


def change_color(id, rgbaColor):
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    pb.changeVisualShape(id, -1, rgbaColor=rgbaColor)


def hide_sphere(sphereId) -> None:
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    pb.resetBasePositionAndOrientation(sphereId, FAR_AWAY_POSITION, [0, 0, 0, 1])


def add_line(start, end, rgbColor=WHITE_RGB, lineWidth=3):
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    return pb.addUserDebugLine(start, end, lineColorRGB=rgbColor, lineWidth=lineWidth)


def move_line(lineId, start, end, rgbColor=WHITE_RGB, lineWidth=3):
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    pb.addUserDebugLine(
        start,
        end,
        replaceItemUniqueId=lineId,
        lineColorRGB=rgbColor,
        lineWidth=lineWidth,
    )
