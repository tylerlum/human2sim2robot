import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
import yaml
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from human2sim2robot.human_demo.utils.pybullet_utils import (
    BLUE_RGBA,
    BLUE_TRANSLUCENT_RGBA,
    CYAN_RGBA,
    FAR_AWAY_POSITION,
    GREEN_RGBA,
    GREEN_TRANSLUCENT_RGBA,
    MAGENTA_RGBA,
    RED_RGBA,
    RED_TRANSLUCENT_RGBA,
    YELLOW_RGBA,
    YELLOW_TRANSLUCENT_RGBA,
    add_sphere,
    draw_collision_spheres,
    get_link_name_to_idx,
    get_num_actuatable_joints,
    move_sphere,
    set_robot_state,
    visualize_transform,
)
from human2sim2robot.human_demo.utils.utils import (
    create_urdf,
    normalize,
    transform_point,
)
from human2sim2robot.sim_training import get_asset_root
from human2sim2robot.sim_training.utils.cross_embodiment.camera_extrinsics import (
    REALSENSE_CAMERA_T_R_C,
    ZED_CAMERA_T_R_C,
)

# TODO: Set the T_R_C appropriately
CAMERA = "zed"
if CAMERA == "zed":
    T_R_C = ZED_CAMERA_T_R_C
elif CAMERA == "realsense":
    T_R_C = REALSENSE_CAMERA_T_R_C
else:
    raise ValueError(f"Invalid camera: {CAMERA}")
assert T_R_C.shape == (4, 4), T_R_C.shape


@dataclass
class Args:
    obj_path: Path
    object_poses_dir: Path
    hand_poses_dir: Path
    robot_file: Optional[str] = None
    visualize_hand_meshes: bool = False
    retargeted_robot_file: Optional[Path] = None
    visualize_collision_spheres: bool = False
    visualize_transforms: bool = False
    visualize_table: bool = False
    dt: float = 1.0 / 30
    start_idx: int = 0


def set_keypoint_sphere_positions(hand_keypoint_to_xyz: dict) -> None:
    keypoint_to_color = {
        "wrist_back": RED_TRANSLUCENT_RGBA,
        "wrist_front": RED_RGBA,
        "index_0_back": GREEN_TRANSLUCENT_RGBA,
        "index_0_front": GREEN_RGBA,
        "middle_0_back": BLUE_TRANSLUCENT_RGBA,
        "middle_0_front": BLUE_RGBA,
        "ring_0_back": YELLOW_TRANSLUCENT_RGBA,
        "ring_0_front": YELLOW_RGBA,
        "index_3": GREEN_RGBA,
        "middle_3": BLUE_RGBA,
        "ring_3": YELLOW_RGBA,
        "thumb_3": MAGENTA_RGBA,
        "PALM_TARGET": CYAN_RGBA,
    }
    keypoints = keypoint_to_color.keys()

    if not hasattr(set_keypoint_sphere_positions, "sphere_ids"):
        set_keypoint_sphere_positions.sphere_ids = [
            add_sphere(
                radius=0.02,
                position=hand_keypoint_to_xyz[keypoint],
                rgbaColor=keypoint_to_color[keypoint],
            )
            for keypoint in keypoints
        ]
    else:
        for i, keypoint in enumerate(keypoints):
            move_sphere(
                set_keypoint_sphere_positions.sphere_ids[i],
                hand_keypoint_to_xyz[keypoint],
            )


def create_transformed_keypoint_to_xyz(hand_json: dict, T_R_C: np.ndarray) -> dict:
    keypoint_to_xyz = hand_json

    keypoints = [
        "wrist_back",
        "wrist_front",
        "index_0_back",
        "index_0_front",
        "middle_0_back",
        "middle_0_front",
        "ring_0_back",
        "ring_0_front",
        "index_3",
        "middle_3",
        "ring_3",
        "thumb_3",
    ]
    for keypoint in keypoints:
        assert keypoint in keypoint_to_xyz, (
            f"{keypoint} not in {keypoint_to_xyz.keys()}"
        )
        keypoint_to_xyz[keypoint] = np.array(keypoint_to_xyz[keypoint])

    # Shorthand for next computations
    kpt_map = keypoint_to_xyz

    # Palm target
    mean_middle_0 = np.mean(
        [
            kpt_map["middle_0_back"],
            kpt_map["middle_0_front"],
        ],
        axis=0,
    )
    palm_normal = normalize(
        np.cross(
            normalize(kpt_map["index_0_front"] - kpt_map["ring_0_front"]),
            normalize(kpt_map["middle_0_front"] - kpt_map["wrist_front"]),
        )
    )
    kpt_map["PALM_TARGET"] = (
        mean_middle_0
        # VERSION 1
        - normalize(kpt_map["middle_0_front"] - kpt_map["wrist_front"]) * 0.03
        - palm_normal * 0.03
        #
        # VERSION 2
        # - palm_normal * 0.03 * np.sqrt(2)
        #
        # VERSION 3
        # - normalize(kpt_map["middle_0_front"] - kpt_map["wrist_front"]) * 0.03 * np.sqrt(2)
    )

    transformed_keypoint_to_xyz = {
        keypoint: transform_point(T=T_R_C, point=kpt_map[keypoint])
        for keypoint in keypoints + ["PALM_TARGET"]
    }

    # WARNING: After extensive testing, we find that the Allegro hand robot in the real world
    #          is about 1.2cm lower than the simulated Allegro hand for most joint angles.
    #          This difference is severe enough to cause low-profile manipulation tasks to fail
    #          Thus, we manually offset the robot base by 1.2cm in the z-direction.
    MANUAL_OFFSET_ROBOT_Z = 0.012
    NEW_transformed_keypoint_to_xyz = {
        keypoint: transformed_keypoint_to_xyz[keypoint]
        + np.array([0, 0, MANUAL_OFFSET_ROBOT_Z])
        for keypoint in keypoints + ["PALM_TARGET"]
    }
    transformed_keypoint_to_xyz = NEW_transformed_keypoint_to_xyz

    # HACK: add global_orient
    transformed_keypoint_to_xyz["global_orient"] = kpt_map["global_orient"]
    return transformed_keypoint_to_xyz


def compute_r_R_P(keypoint_to_xyz: dict) -> np.ndarray:
    # Z = palm to middle finger
    # Y = palm to thumb
    # X = palm normal
    kpt_map = keypoint_to_xyz
    palm_to_middle_finger = normalize(
        kpt_map["middle_0_front"] - kpt_map["wrist_front"]
    )
    palm_to_thumb = normalize(kpt_map["index_0_front"] - kpt_map["ring_0_front"])
    _palm_normal = normalize(np.cross(palm_to_middle_finger, palm_to_thumb))

    Z = palm_to_middle_finger
    Y_not_orthogonal = palm_to_thumb
    Y = normalize(Y_not_orthogonal - np.dot(Y_not_orthogonal, Z) * Z)
    X = normalize(
        np.cross(
            Y,
            Z,
        )
    )
    r_R_P = np.stack(
        [X, Y, Z],
        axis=1,
    )
    return r_R_P


def main():
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    args = tyro.cli(Args)
    print("=" * 80)
    print(args)
    print("=" * 80)

    # Start visualizer
    pb.connect(pb.GUI)
    pb.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraYaw=90,
        cameraPitch=-15,
        cameraTargetPosition=[0, 0, 0],
    )

    # Load object poses
    assert args.object_poses_dir.exists(), (
        f"Object poses dir {args.object_poses_dir} does not exist"
    )
    object_pose_files = sorted(list(args.object_poses_dir.glob("*.txt")))
    assert len(object_pose_files) > 0, (
        f"No object poses found in {args.object_poses_dir}"
    )
    T_C_Os = [
        np.loadtxt(filename)
        for filename in tqdm(object_pose_files, desc="Loading object poses")
    ]
    for i, T_C_O in enumerate(T_C_Os):
        assert T_C_O.shape == (
            4,
            4,
        ), f"Object pose {T_C_O} at index {i} has shape {T_C_O.shape}"
    T_R_Os = [T_R_C @ T_C_O for T_C_O in T_C_Os]

    # Load object
    assert args.obj_path.exists(), f"Object path {args.obj_path} does not exist"
    object_urdf_path = create_urdf(args.obj_path)
    print(f"Loading object from {object_urdf_path}")
    obj = pb.loadURDF(
        str(object_urdf_path),
        useFixedBase=True,
        basePosition=T_R_Os[0][:3, 3],
        baseOrientation=R.from_matrix(T_R_Os[0][:3, :3]).as_quat(),
    )

    # Load hand poses
    assert args.hand_poses_dir.exists(), (
        f"Hand poses dir {args.hand_poses_dir} does not exist"
    )

    hand_json_files = sorted(list(args.hand_poses_dir.glob("*.json")))
    assert len(hand_json_files) > 0, f"No hand poses found in {args.hand_poses_dir}"
    hand_jsons = []
    for filename in tqdm(hand_json_files, desc="Loading hand poses"):
        with open(filename, "r") as f:
            hand_jsons.append(json.load(f))
    hand_keypoint_to_xyzs = [
        create_transformed_keypoint_to_xyz(hand_json, T_R_C) for hand_json in hand_jsons
    ]

    # Load hand meshes
    if args.visualize_hand_meshes:
        # Each timestep has a different hand mesh because they can change shape
        # So this is slow to load
        hand_urdf_files = [
            create_urdf(hand_json_file.with_suffix(".obj"))
            for hand_json_file in hand_json_files
        ]

        SPAWN_HANDS_AT_FAR_AWAY_POSITION = False
        if SPAWN_HANDS_AT_FAR_AWAY_POSITION:
            hand_xyz, hand_quat_xyzw = FAR_AWAY_POSITION, [0, 0, 0, 1]
        else:
            hand_xyz, hand_quat_xyzw = (
                T_R_C[:3, 3],
                R.from_matrix(T_R_C[:3, :3]).as_quat(),
            )

        hand_ids = []
        for hand_urdf_file in tqdm(hand_urdf_files, desc="Loading hands"):
            hand_id = pb.loadURDF(
                str(hand_urdf_file),
                useFixedBase=True,
                basePosition=hand_xyz,
                baseOrientation=hand_quat_xyzw,
            )
            if not SPAWN_HANDS_AT_FAR_AWAY_POSITION:
                # Move hand to far away position after spawning
                pb.resetBasePositionAndOrientation(
                    hand_id, FAR_AWAY_POSITION, [0, 0, 0, 1]
                )

            hand_ids.append(hand_id)
            pb.changeVisualShape(hand_id, -1, rgbaColor=[1, 1, 1, 1])

    # Load robot if given
    if args.robot_file is not None:
        robot_yml_path = join_path(get_robot_configs_path(), args.robot_file)
        assert Path(robot_yml_path).exists(), (
            f"Robot YAML path {robot_yml_path} does not exist"
        )
        robot_yml = load_yaml(robot_yml_path)

        # Get urdf path
        urdf_path = robot_yml["robot_cfg"]["kinematics"]["urdf_path"]
        urdf_path = Path(join_path(get_assets_path(), urdf_path))
        assert urdf_path.exists(), f"URDF path {urdf_path} does not exist"

        # Load robot
        print(f"Loading robot from {urdf_path}")
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

        if args.retargeted_robot_file is not None:
            assert args.retargeted_robot_file.exists(), (
                f"Retargeted hand poses file {args.retargeted_robot_file} does not exist"
            )
            retargeted_robot_file = np.load(args.retargeted_robot_file)
            qs = retargeted_robot_file["qs"]
            idxs = retargeted_robot_file["idxs"]
            assert qs.ndim == 2, f"qs has shape {qs.shape}"
            assert qs.shape[1] == num_actuatable_joints, (
                f"qs has shape {qs.shape} but expected shape to have {num_actuatable_joints} in the second dimension"
            )
            assert idxs.ndim == 1, f"idxs has shape {idxs.shape}"
            assert idxs.shape[0] == qs.shape[0], (
                f"idxs has shape {idxs.shape} but expected shape to have {qs.shape[0]} in the first dimension"
            )
            idxs = list(idxs)
            idxs_set = set(idxs)

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

        # Load transforms if given
        if args.visualize_transforms:
            target_palm_link_lines = visualize_transform(
                xyz=np.zeros(3), rotation_matrix=np.eye(3)
            )
            actual_palm_link_lines = visualize_transform(
                xyz=np.zeros(3), rotation_matrix=np.eye(3)
            )

    # Load table if given
    if args.visualize_table:
        from human2sim2robot.sim_training.utils.cross_embodiment.table_constants import (
            TABLE_QW,
            TABLE_QX,
            TABLE_QY,
            TABLE_QZ,
            TABLE_X,
            TABLE_Y,
            TABLE_Z,
        )

        table_urdf = Path(get_asset_root()) / "table/table.urdf"
        print(f"Loading table from {table_urdf}")
        _table = pb.loadURDF(
            str(table_urdf),
            useFixedBase=True,
            basePosition=[TABLE_X, TABLE_Y, TABLE_Z],
            baseOrientation=[TABLE_QX, TABLE_QY, TABLE_QZ, TABLE_QW],
        )

    assert len(T_R_Os) == len(hand_keypoint_to_xyzs), (
        f"Number of object poses ({len(T_R_Os)}) does not match number of hand poses ({len(hand_keypoint_to_xyzs)})"
    )
    N_TIMESTEPS = len(T_R_Os)

    # Visualization loop
    while True:
        for i, (T_R_O, hand_keypoint_to_xyz) in tqdm(
            enumerate(zip(T_R_Os, hand_keypoint_to_xyzs)),
            total=N_TIMESTEPS,
            desc="Visualizing trajectory",
        ):
            if i < args.start_idx:
                continue

            start_time = time.time()

            # Object
            obj_xyz, obj_quat_xyzw = (
                T_R_O[:3, 3],
                R.from_matrix(T_R_O[:3, :3]).as_quat(),
            )
            pb.resetBasePositionAndOrientation(
                obj,
                obj_xyz,
                obj_quat_xyzw,
            )

            # Hand keypoints
            set_keypoint_sphere_positions(hand_keypoint_to_xyz)

            # Hand meshes
            if args.visualize_hand_meshes:
                # Move previous hand to far away position
                # Works when i = 0 because it just moves the last one
                prev_hand_id = hand_ids[i - 1]
                pb.resetBasePositionAndOrientation(
                    prev_hand_id,
                    FAR_AWAY_POSITION,
                    [0, 0, 0, 1],
                )

                hand_id = hand_ids[i]
                hand_xyz, hand_quat_xyzw = (
                    T_R_C[:3, 3],
                    R.from_matrix(T_R_C[:3, :3]).as_quat(),
                )
                pb.resetBasePositionAndOrientation(
                    hand_id,
                    hand_xyz,
                    hand_quat_xyzw,
                )

            # Robot
            if args.robot_file is not None:
                if args.retargeted_robot_file is not None:
                    if i in idxs_set:
                        idx = idxs.index(i)
                        set_robot_state(robot, qs[idx])

                if args.visualize_collision_spheres:
                    draw_collision_spheres(robot, collision_config)
                if args.visualize_transforms:
                    robot_link_name_to_id = get_link_name_to_idx(robot)
                    robot_palm_com, robot_palm_quat, *_ = pb.getLinkState(
                        robot,
                        robot_link_name_to_id["palm_link"],
                        computeForwardKinematics=1,
                    )

                    visualize_transform(
                        xyz=np.array(robot_palm_com),
                        rotation_matrix=R.from_quat(robot_palm_quat).as_matrix(),
                        lines=actual_palm_link_lines,
                    )

                    r_R_P = compute_r_R_P(keypoint_to_xyz=hand_keypoint_to_xyz)
                    T_R_P = np.eye(4)
                    T_R_P[:3, :3] = r_R_P
                    T_R_P[:3, 3] = hand_keypoint_to_xyz["PALM_TARGET"]

                    visualize_transform(
                        xyz=T_R_P[:3, 3],
                        rotation_matrix=T_R_P[:3, :3],
                        lines=target_palm_link_lines,
                    )

            end_time = time.time()
            extra_dt = args.dt - (end_time - start_time)
            if extra_dt > 0:
                time.sleep(extra_dt)
            else:
                print(
                    f"Visualization is running slow, late by {-extra_dt * 1000:.2f} ms"
                )

        print("=" * 80)
        print("Setting breakpoint. Continue to start over")
        print("=" * 80 + "\n")
        breakpoint()


if __name__ == "__main__":
    main()
