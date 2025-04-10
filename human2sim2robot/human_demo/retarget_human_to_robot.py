import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tyro
import yaml
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from human2sim2robot.human_demo.utils.curobo_utils import (
    get_world_cfg,
)
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
from human2sim2robot.sim_training.utils.torch_jit_utils import (
    matrix_to_quat_xyzw,
    quat_xyzw_to_matrix,
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
    robot_file: str
    visualize_hand_meshes: bool = False
    visualize_collision_spheres: bool = False
    visualize_transforms: bool = False
    visualize_table: bool = False
    dt: float = 1.0 / 30
    num_arm_joints: int = 7
    num_hand_joints: int = 16
    num_fingers: int = 4
    headless: bool = False
    output_filepath: Optional[Path] = None
    start_idx: int = 0

    def __post_init__(self):
        if self.output_filepath is not None:
            assert self.output_filepath.suffix == ".npz", (
                f"Invalid output_filepath: {self.output_filepath}"
            )


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


def calculate_hand_dist(
    hand_keypoint_to_xyz: dict, hand_keypoint_to_xyz_prev: dict
) -> float:
    keypoints = [
        "wrist_front",
        "index_0_front",
        "middle_0_front",
        "ring_0_front",
        "thumb_3",
        "index_3",
        "middle_3",
        "ring_3",
    ]
    N_KEYPOINTS = len(keypoints)

    points = np.array([hand_keypoint_to_xyz[keypoint] for keypoint in keypoints])
    points_prev = np.array(
        [hand_keypoint_to_xyz_prev[keypoint] for keypoint in keypoints]
    )
    assert points.shape == (
        N_KEYPOINTS,
        3,
    ), f"points.shape: {points.shape}, N_KEYPOINTS: {N_KEYPOINTS}"
    assert points_prev.shape == (
        N_KEYPOINTS,
        3,
    ), f"points_prev.shape: {points_prev.shape}, N_KEYPOINTS: {N_KEYPOINTS}"

    # return np.mean(np.linalg.norm(points - points_prev, axis=1))
    return np.max(np.linalg.norm(points - points_prev, axis=1))


def solve_fingertip_ik(
    allegro_id: int,
    fingertip_pos_list: List[np.ndarray],
    fingertip_idx_list: List[int],
    num_hand_joints: int,
    num_fingers: int,
) -> np.ndarray:
    # HACK: Hide pybullet import in functions that use it to avoid
    import pybullet as pb

    # HACK: For numerous reasons, it was hard to do the hand IK while constraining the arm joints
    # Thus, we simply load the allegro hand robot and use it for the hand IK
    # This code should change per hand
    target_q = []

    N_FINGERS = 4
    assert num_fingers == N_FINGERS, (
        f"num_fingers: {num_fingers} != N_FINGERS: {N_FINGERS}"
    )
    assert len(fingertip_pos_list) == N_FINGERS, (
        f"len(fingertip_pos_list): {len(fingertip_pos_list)} != N_FINGERS: {N_FINGERS}"
    )
    assert len(fingertip_idx_list) == N_FINGERS, (
        f"len(fingertip_idx_list): {len(fingertip_idx_list)} != N_FINGERS: {N_FINGERS}"
    )

    for i in range(N_FINGERS):
        result = pb.calculateInverseKinematics(
            allegro_id,
            fingertip_idx_list[i],
            fingertip_pos_list[i],
            maxNumIterations=2000,
            residualThreshold=0.001,
        )
        result = np.array(result)
        assert result.shape == (num_hand_joints,), f"result.shape: {result.shape}"
        target_q += result[4 * i : 4 * (i + 1)].tolist()

    target_q = np.array(target_q)
    assert target_q.shape == (num_hand_joints,), f"target_q.shape: {target_q.shape}"
    return target_q


def interpolate(t: np.ndarray, x: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, f"x.shape: {x.shape}"
    assert t.ndim == 1, f"t.shape: {t.shape}"
    N, D = x.shape

    new_x = np.zeros((len(new_t), D))
    for i in range(D):
        new_x[:, i] = np.interp(
            x=new_t,
            xp=t,
            fp=x[:, i],
        )
    return new_x


def interpolate_poses(t: np.ndarray, x: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    assert t.ndim == 1, f"t.shape: {t.shape}"
    assert x.ndim == 3, f"x.shape: {x.shape}"
    assert new_t.ndim == 1, f"new_t.shape: {new_t.shape}"
    N = t.shape[0]
    assert x.shape == (N, 4, 4), f"x.shape: {x.shape}"
    new_N = new_t.shape[0]

    new_x = np.eye(4)[None, ...].repeat(new_N, axis=0)

    # Translation
    new_x[:, :3, 3] = interpolate(
        t=t,
        x=x[:, :3, 3],
        new_t=new_t,
    )

    # Rotation
    new_x[:, :3, :3] = interpolate_rotation(
        t=t,
        x=x[:, :3, :3],
        new_t=new_t,
    )

    return new_x


def interpolate_rotation(t: np.ndarray, x: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    assert t.ndim == 1, f"t.shape: {t.shape}"
    assert x.ndim == 3, f"x.shape: {x.shape}"
    assert new_t.ndim == 1, f"new_t.shape: {new_t.shape}"
    N = t.shape[0]
    assert x.shape == (N, 3, 3), f"x.shape: {x.shape}"
    new_N = new_t.shape[0]

    quats = matrix_to_quat_xyzw(torch.from_numpy(x).float().cuda()).cpu().numpy()

    new_quats = interpolate_quats(
        t=t,
        x=quats,
        new_t=new_t,
    )

    new_x = (
        quat_xyzw_to_matrix(torch.from_numpy(new_quats).float().cuda()).cpu().numpy()
    )
    assert new_x.shape == (
        new_N,
        3,
        3,
    ), f"new_x.shape: {new_x.shape}, new_N: {new_N}"

    return new_x


def interpolate_quats(
    t: np.ndarray,  # shape: (N,)
    x: np.ndarray,  # shape: (N, 4), quaternions in (w, x, y, z) or (x, y, z, w) - just be consistent
    new_t: np.ndarray,  # shape: (new_N,)
) -> np.ndarray:
    """
    Interpolates quaternions x (shape: (N,4)) at times t (shape: (N,))
    onto new times new_t (shape: (new_N,)) using spherical linear interpolation (slerp).

    Returns new_x of shape (new_N, 4).
    """
    assert t.ndim == 1, f"t.shape: {t.shape}"
    assert x.ndim == 2, f"x.shape: {x.shape}"
    assert new_t.ndim == 1, f"new_t.shape: {new_t.shape}"
    N = t.shape[0]
    assert x.shape == (N, 4), f"x.shape: {x.shape}"
    new_N = new_t.shape[0]

    new_x = np.zeros((new_N, 4), dtype=np.float64)

    # 1) Clamp new_t to be within [t[0], t[-1]] so we don't go out of range.
    #    (Alternatively, you could extrapolate if desired.)
    new_t_clamped = np.clip(new_t, t[0], t[-1])

    # 2) For each new time, find the interval [t[idx-1], t[idx]] it belongs to.
    #    np.searchsorted with side='right' gives an index such that
    #    t[idx-1] <= new_t[i] < t[idx].
    #    We'll just keep them in [1, N-1] range to avoid going out of bounds.
    idx = np.searchsorted(t, new_t_clamped, side="right")
    idx = np.clip(idx, 1, N - 1)

    # 3) Compute interpolation factor alpha
    t0 = t[idx - 1]
    t1 = t[idx]
    alpha = (new_t_clamped - t0) / (t1 - t0)  # shape: (new_N,)

    # 4) Gather the corresponding quaternions
    q0 = x[idx - 1]  # shape: (new_N, 4)
    q1 = x[idx]  # shape: (new_N, 4)

    # 5) Vectorized SLERP
    #    We'll implement a helper function that operates on batches of quaternions.
    new_x = _vectorized_slerp(q0, q1, alpha)

    return new_x


def _vectorized_slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Vectorized slerp between two sets of quaternions q0 and q1, each shape (B,4),
    with interpolation fractions alpha of shape (B,).
    Returns interpolated quaternions of shape (B,4).

    Assumes q0, q1 are in the same (w,x,y,z) or (x,y,z,w) convention and are non-zero.
    """
    # Normalize input quaternions to avoid accumulating numerical errors
    q0 = q0 / np.linalg.norm(q0, axis=1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)

    # Compute dot products for each pair
    dots = np.sum(q0 * q1, axis=1)  # shape: (B,)

    # If dot < 0, negate q1 so we take the shorter arc
    neg_mask = dots < 0.0
    q1[neg_mask] = -q1[neg_mask]
    dots[neg_mask] = -dots[neg_mask]

    # Allocate output
    out = np.zeros_like(q0)

    # Handle "very close" quaternions with linear interpolation (to avoid numerical instability)
    close_mask = dots > 0.9995  # shape: (B,)
    not_close_mask = ~close_mask

    # --- Linear interpolation branch ---
    # out_close = q0 * (1 - alpha) + q1 * alpha, then normalize
    if np.any(close_mask):
        alpha_close = alpha[close_mask, np.newaxis]
        lerp = q0[close_mask] * (1.0 - alpha_close) + q1[close_mask] * alpha_close
        lerp /= np.linalg.norm(lerp, axis=1, keepdims=True)
        out[close_mask] = lerp

    # --- Slerp branch ---
    if np.any(not_close_mask):
        dot_not_close = dots[not_close_mask]
        alpha_nc = alpha[not_close_mask]
        q0_nc = q0[not_close_mask]
        q1_nc = q1[not_close_mask]

        # theta is the angle between the quaternions
        theta = np.arccos(np.clip(dot_not_close, -1.0, 1.0))
        sin_theta = np.sin(theta)

        # s0 = sin((1 - alpha)*theta) / sin(theta)
        # s1 = sin(alpha*theta) / sin(theta)
        s0 = np.sin((1.0 - alpha_nc) * theta) / sin_theta
        s1 = np.sin(alpha_nc * theta) / sin_theta

        # out_not_close = s0 * q0 + s1 * q1
        out_nc = (q0_nc * s0[:, None]) + (q1_nc * s1[:, None])
        out_nc /= np.linalg.norm(out_nc, axis=1, keepdims=True)
        out[not_close_mask] = out_nc

    return out


def slow_down_high_speed_motion(
    t: np.ndarray, x: np.ndarray, max_v: float
) -> Tuple[np.ndarray, np.ndarray]:
    assert t.ndim == 1, f"t.shape: {t.shape}"
    assert x.ndim == 2, f"x.shape: {x.shape}"
    N, D = x.shape
    assert t.shape[0] == N, f"t.shape: {t.shape}, x.shape: {x.shape}"
    assert D == 23, f"D: {D}"

    # Compute velocities
    dts = np.diff(t, axis=0)
    x_diffs = np.diff(x, axis=0)
    v = np.abs(x_diffs / dts[:, None])
    assert dts.shape == (N - 1,), f"dts.shape: {dts.shape}, N: {N}"
    assert x_diffs.shape == (N - 1, D), f"x_diffs.shape: {x_diffs.shape}, N: {N}"
    assert v.shape == (N - 1, D), f"v.shape: {v.shape}, N: {N}"

    # We only care about the arm joints
    # Rescale each dt to ensure no velocity exceeds max_v
    over_limit_factors = np.max(v[:, :7] / max_v, axis=1)
    assert over_limit_factors.shape == (N - 1,), (
        f"over_limit_factors.shape: {over_limit_factors.shape}, N: {N}"
    )
    print(f"Over limit factors: {over_limit_factors}")

    # Rescale factors >1 means that the velocity exceeds max_v, so we need to increase the dt
    # Rescale factors <1 means that the velocity does not exceed max_v, so we can leave the dt unchanged
    rescale_factors = np.clip(over_limit_factors, a_min=1.0, a_max=None)
    rescaled_dts = dts * rescale_factors
    rescaled_t = t[0] + np.concatenate([[0], np.cumsum(rescaled_dts)])
    assert rescaled_t.shape == (N,), f"rescaled_t.shape: {rescaled_t.shape}, N: {N}"
    return rescaled_t, x


def main():
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as pb

    args = tyro.cli(Args)
    print("=" * 80)
    print(args)
    print("=" * 80)

    # Start visualizer
    if args.headless:
        pb.connect(pb.DIRECT)
    else:
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

        # if args.retargeted_hand_poses_file is not None:
        #     assert args.retargeted_hand_poses_file.exists(), (
        #         f"Retargeted hand poses file {args.retargeted_hand_poses_file} does not exist"
        #     )
        #     retargeted_hand_poses = np.loadtxt(args.retargeted_hand_poses_file)
        #     assert retargeted_hand_poses.ndim == 2, (
        #         f"Retargeted hand poses {retargeted_hand_poses} has shape {retargeted_hand_poses.shape}"
        #     )
        #     assert retargeted_hand_poses.shape[1] == num_actuatable_joints, (
        #         f"Retargeted hand poses {retargeted_hand_poses} has shape {retargeted_hand_poses.shape} but expected shape to have {num_actuatable_joints} in the second dimension"
        #     )

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

    # Setup IK solver
    DEFAULT_KUKA_JOINTS = np.deg2rad([0, 0, 0, -90, 0, 90, 0]).tolist()
    DEFAULT_ALLEGRO_JOINTS = [
        0.0,
        0.3,
        0.3,
        0.3,
        0.0,
        0.3,
        0.3,
        0.3,
        0.0,
        0.3,
        0.3,
        0.3,
        0.72383858,
        0.60147215,
        0.33795027,
        0.60845138,
    ]

    ALLEGRO_URDF = get_asset_root() / "kuka_allegro/allegro.urdf"
    assert ALLEGRO_URDF.exists(), f"Allegro URDF {ALLEGRO_URDF} does not exist"

    # HACK: For numerous reasons, it was hard to do the hand IK while constraining the arm joints
    # Thus, we simply load the allegro hand robot and use it for the hand IK
    allegro = pb.loadURDF(
        str(ALLEGRO_URDF),
        useFixedBase=True,
        basePosition=FAR_AWAY_POSITION,
        baseOrientation=[0, 0, 0, 1],
    )

    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot_file))[
        "robot_cfg"
    ]
    robot_cfg["kinematics"]["ee_link"] = "palm_link"
    robot_cfg = RobotConfig.from_dict(robot_cfg)

    # Adjust arm joint limits to be within 10 degrees of actual joint limits as buffer
    print("ADDING BUFFER TO JOINT LIMITS")
    ARM_BUFFER_DEG = 10
    for i in range(args.num_arm_joints):
        robot_cfg.kinematics.kinematics_config.joint_limits.position[0, i] += (
            np.deg2rad(ARM_BUFFER_DEG)
        )
        robot_cfg.kinematics.kinematics_config.joint_limits.position[1, i] -= (
            np.deg2rad(ARM_BUFFER_DEG)
        )

    # Adjust hand joint limits to be within 10 degrees of the default hand joint positions
    HAND_BUFFER_DEG = 10
    for i in range(args.num_hand_joints):
        robot_cfg.kinematics.kinematics_config.joint_limits.position[
            0, i + args.num_arm_joints
        ] = DEFAULT_ALLEGRO_JOINTS[i] - np.deg2rad(HAND_BUFFER_DEG)
        robot_cfg.kinematics.kinematics_config.joint_limits.position[
            1, i + args.num_arm_joints
        ] = DEFAULT_ALLEGRO_JOINTS[i] + np.deg2rad(HAND_BUFFER_DEG)

    # Ignore environment collision checking for now
    world_cfg = get_world_cfg()
    N_SEEDS = 20
    arm_ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.05,
        num_seeds=N_SEEDS,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    arm_ik_solver = IKSolver(arm_ik_config)

    num_joints = arm_ik_solver.robot_config.kinematics.kinematics_config.joint_limits.position.shape[
        -1
    ]
    assert num_joints == args.num_arm_joints + args.num_hand_joints, (
        f"num_joints: {num_joints} != args.num_arm_joints + args.num_hand_joints: {args.num_arm_joints + args.num_hand_joints}"
    )
    MAX_HAND_DIST_M = 0.02
    MAX_ARM_JOINT_DIST_DEG = 15

    # Storage
    ARM_Q_SOLUTION_LIST: List[
        np.ndarray
    ] = []  # This will be N_TIMESTEPS list of (args.num_arm_joints,) np.ndarrays
    HAND_Q_SOLUTION_LIST: List[
        np.ndarray
    ] = []  # This will be N_TIMESTEPS list of (args.num_hand_joints,) np.ndarrays
    Q_IDX_LIST: List[int] = []  # This will be N_TIMESTEPS list of ints
    OBJECT_POSE_LIST: List[np.ndarray] = []
    HAND_DIST_LIST: List[float] = []
    ARM_DIST_LIST: List[float] = []
    GOOD_HAND_KEYPOINT_TO_XYZ_LIST: List[Dict[str, np.ndarray]] = []
    GOOD_HAND_IDX_LIST: List[int] = []
    RELATIVE_PREMANIP_POSE_LIST: List[np.ndarray] = []

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

            if len(GOOD_HAND_KEYPOINT_TO_XYZ_LIST) > 0 and len(GOOD_HAND_IDX_LIST) > 0:
                # Logic to filter out bad hand keypoints
                hand_dist = calculate_hand_dist(
                    hand_keypoint_to_xyz, GOOD_HAND_KEYPOINT_TO_XYZ_LIST[-1]
                )
                HAND_DIST_LIST.append(hand_dist)
                idx_dist = i - GOOD_HAND_IDX_LIST[-1]
                if hand_dist > MAX_HAND_DIST_M * idx_dist:
                    print(
                        f"WARNING: Hand distance too large at timestep {i}, hand_dist = {hand_dist:.4f} m, idx_dist = {idx_dist}, MAX_HAND_DIST_M = {MAX_HAND_DIST_M:.4f} m"
                    )
                    continue

            GOOD_HAND_KEYPOINT_TO_XYZ_LIST.append(hand_keypoint_to_xyz)
            GOOD_HAND_IDX_LIST.append(i)

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

            # Arm IK
            N_SOLUTIONS = 100

            r_R_P = compute_r_R_P(keypoint_to_xyz=hand_keypoint_to_xyz)
            T_R_P = np.eye(4)
            T_R_P[:3, :3] = r_R_P
            T_R_P[:3, 3] = hand_keypoint_to_xyz["PALM_TARGET"]

            T_O_R = np.linalg.inv(T_R_O)
            T_O_P = T_O_R @ T_R_P

            position = (
                torch.from_numpy(T_R_P[:3, 3])
                .float()
                .cuda()
                .unsqueeze(dim=0)
                .repeat_interleave(N_SOLUTIONS, dim=0)
            )
            rotation = (
                torch.from_numpy(T_R_P[:3, :3])
                .float()
                .cuda()
                .unsqueeze(dim=0)
                .repeat_interleave(N_SOLUTIONS, dim=0)
            )
            target_pose = Pose(
                position=position,
                rotation=rotation,
            )

            # Init from previous solution if available, otherwise use default
            if len(ARM_Q_SOLUTION_LIST) == 0:
                init_q = DEFAULT_KUKA_JOINTS + DEFAULT_ALLEGRO_JOINTS
            else:
                init_q = ARM_Q_SOLUTION_LIST[-1].tolist() + DEFAULT_ALLEGRO_JOINTS

            retract_config = (
                torch.tensor(
                    init_q,
                    device=tensor_args.device,
                    dtype=tensor_args.dtype,
                )[None, ...]
                .repeat_interleave(N_SOLUTIONS, dim=0)
                .to(tensor_args.device)
            )
            seed_config = retract_config.clone()[:, None, ...].repeat_interleave(
                N_SEEDS, dim=1
            )

            SEED_NOISE_DEG = 15  # Tune
            seed_config += np.deg2rad(SEED_NOISE_DEG) * torch.randn_like(seed_config)

            assert retract_config.shape == (
                N_SOLUTIONS,
                23,
            ), f"retract_config.shape: {retract_config.shape}"
            assert seed_config.shape == (
                N_SOLUTIONS,
                N_SEEDS,
                23,
            ), f"seed_config.shape: {seed_config.shape}"

            result = arm_ik_solver.solve_batch(
                target_pose,
                retract_config=retract_config,
                seed_config=seed_config,
            )

            N = target_pose.position.shape[0]
            assert result.solution.shape == (
                N,
                1,
                num_joints,
            ), f"result.solution.shape: {result.solution.shape}"
            assert result.success.shape == (
                N,
                1,
            ), f"result.success.shape: {result.success.shape}"

            Q_SOLUTIONS = result.solution.squeeze(dim=1).detach().cpu().numpy()
            successes = result.success.squeeze(dim=1).detach().cpu().numpy()
            N_SUCCESS = np.sum(successes)
            print(f"N_SUCCESS: {N_SUCCESS} / {N_SOLUTIONS}")
            if N_SUCCESS == 0:
                print(f"WARNING: No solutions found for timestep {i}")
                continue

            valid_q_solutions = Q_SOLUTIONS[successes > 0]
            valid_q_arm_solutions = valid_q_solutions[:, : args.num_arm_joints]

            # Pick the solution with the least distance to the previous solution
            # Otherwise, pick the first solution
            if len(ARM_Q_SOLUTION_LIST) == 0:
                ARM_Q_SOLUTION_LIST.append(valid_q_arm_solutions[0])
                Q_IDX_LIST.append(i)
            else:
                q_arm = valid_q_arm_solutions
                prev_q_arm = ARM_Q_SOLUTION_LIST[-1]
                assert q_arm.shape == (
                    N_SUCCESS,
                    args.num_arm_joints,
                ), f"q_arm.shape: {q_arm.shape}"
                assert prev_q_arm.shape == (args.num_arm_joints,), (
                    f"prev_q_arm.shape: {prev_q_arm.shape}"
                )

                # Distance metrics
                l2_diffs = np.linalg.norm(q_arm - prev_q_arm[None], axis=1, ord=2)
                l1_diffs = np.linalg.norm(q_arm - prev_q_arm[None], axis=1, ord=1)
                linf_diffs = np.linalg.norm(
                    q_arm - prev_q_arm[None], axis=1, ord=np.inf
                )

                METRIC = "inf"
                if METRIC == "l2":
                    diffs = l2_diffs
                elif METRIC == "l1":
                    diffs = l1_diffs
                elif METRIC == "inf":
                    diffs = linf_diffs

                selected_idx = np.argmin(diffs)

                diffs = np.linalg.norm(q_arm - prev_q_arm[None], axis=1, ord=np.inf)
                idx_diff = i - Q_IDX_LIST[-1]
                best_solution_linf_diff = np.min(np.rad2deg(linf_diffs))
                if best_solution_linf_diff > MAX_ARM_JOINT_DIST_DEG * idx_diff:
                    print(
                        f"WARNING: Arm distance too large at timestep {i}, best_solution_linf_diff = {best_solution_linf_diff:.4f} deg, MAX_ARM_JOINT_DIST_DEG = {MAX_ARM_JOINT_DIST_DEG:.4f} deg"
                    )
                    continue

                ARM_DIST_LIST.append(best_solution_linf_diff)
                selected_idx = np.argmin(diffs)
                ARM_Q_SOLUTION_LIST.append(valid_q_arm_solutions[selected_idx])
                Q_IDX_LIST.append(i)

            prev_hand_q = (
                HAND_Q_SOLUTION_LIST[-1]
                if len(HAND_Q_SOLUTION_LIST) > 0
                else np.array(DEFAULT_ALLEGRO_JOINTS)
            )
            set_robot_state(
                robot,
                np.concatenate([ARM_Q_SOLUTION_LIST[-1], prev_hand_q]),
            )

            """
            HAND IK
            """
            robot_link_name_to_id = get_link_name_to_idx(robot)
            allegro_link_name_to_id = get_link_name_to_idx(allegro)

            robot_palm_com, robot_palm_quat, *_ = pb.getLinkState(
                robot,
                robot_link_name_to_id["palm_link"],
                computeForwardKinematics=1,
            )
            pb.resetBasePositionAndOrientation(allegro, robot_palm_com, robot_palm_quat)

            hand_q = solve_fingertip_ik(
                allegro_id=allegro,
                fingertip_pos_list=[
                    hand_keypoint_to_xyz["index_3"],
                    hand_keypoint_to_xyz["middle_3"],
                    hand_keypoint_to_xyz["ring_3"],
                    hand_keypoint_to_xyz["thumb_3"],
                ],
                fingertip_idx_list=[
                    allegro_link_name_to_id["index_biotac_tip"],
                    allegro_link_name_to_id["middle_biotac_tip"],
                    allegro_link_name_to_id["ring_biotac_tip"],
                    allegro_link_name_to_id["thumb_biotac_tip"],
                ],
                num_hand_joints=args.num_hand_joints,
                num_fingers=args.num_fingers,
            )

            OVERWRITE_WITH_DEFAULT_HAND = False
            if OVERWRITE_WITH_DEFAULT_HAND:
                HAND_Q_SOLUTION_LIST.append(np.array(DEFAULT_ALLEGRO_JOINTS))
            else:
                HAND_Q_SOLUTION_LIST.append(hand_q)
            OBJECT_POSE_LIST.append(T_R_Os[i])
            RELATIVE_PREMANIP_POSE_LIST.append(T_O_P)

            set_robot_state(allegro, HAND_Q_SOLUTION_LIST[-1])
            set_robot_state(
                robot,
                np.concatenate([ARM_Q_SOLUTION_LIST[-1], HAND_Q_SOLUTION_LIST[-1]]),
            )

            PRINT_SOLUTION = True
            if PRINT_SOLUTION:
                print(f"i: {i}")
                print(
                    f"RELATIVE_PREMANIP_POSE_LIST[-1]: {RELATIVE_PREMANIP_POSE_LIST[-1]}"
                )
                print(f"ARM_Q_SOLUTION_LIST[-1]: {ARM_Q_SOLUTION_LIST[-1]}")
                print(f"HAND_Q_SOLUTION_LIST[-1]: {HAND_Q_SOLUTION_LIST[-1]}")
                print()

            # Robot
            if args.robot_file is not None:
                # if args.retargeted_hand_poses_file is not None:
                #     set_robot_state(robot, retargeted_hand_poses[i])
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

        break  # Keep this for minimal diff

    """
    Visualize
    """
    pb.resetBasePositionAndOrientation(allegro, FAR_AWAY_POSITION, [0, 0, 0, 1])

    # Raw
    ARM_Q_SOLUTIONS = np.array(ARM_Q_SOLUTION_LIST)
    HAND_Q_SOLUTIONS = np.array(HAND_Q_SOLUTION_LIST)
    Q_IDXS = np.array(Q_IDX_LIST)
    Q_TIMES = Q_IDXS * args.dt
    OBJECT_POSES = np.array(OBJECT_POSE_LIST)
    print(f"ARM_Q_SOLUTIONS.shape: {ARM_Q_SOLUTIONS.shape}")
    print(f"HAND_Q_SOLUTIONS.shape: {HAND_Q_SOLUTIONS.shape}")
    print(f"Q_IDXS.shape: {Q_IDXS.shape}")
    print(f"OBJECT_POSES.shape: {OBJECT_POSES.shape}")
    Q_SOLUTIONS = np.concatenate([ARM_Q_SOLUTIONS, HAND_Q_SOLUTIONS], axis=1)
    print(f"Q_SOLUTIONS.shape: {Q_SOLUTIONS.shape}")

    """
    Clean up trajectory
    We will:
    - Subsample to a lower frequency
    - Identify high speed motion and rescale to a reasonable speed
    - Interpolate to a higher frequency
    """
    # NOTE: We subtract the first one so that time starts at 0
    Q_TIMES -= Q_TIMES[0]

    # Subsample
    SUBSAMPLE_FACTOR = 5
    SUBSAMPLED_Q_TIMES = Q_TIMES[::SUBSAMPLE_FACTOR]
    SUBSAMPLED_Q_SOLUTIONS = Q_SOLUTIONS[::SUBSAMPLE_FACTOR]
    SUBSAMPLED_OBJECT_POSES = OBJECT_POSES[::SUBSAMPLE_FACTOR]

    # Identify high speeds and rescale
    MAX_V_DEG_PER_SEC = 45
    SLOWED_DOWN_Q_TIMES, SLOWED_DOWN_Q_SOLUTIONS = slow_down_high_speed_motion(
        t=SUBSAMPLED_Q_TIMES,
        x=SUBSAMPLED_Q_SOLUTIONS,
        max_v=np.deg2rad(MAX_V_DEG_PER_SEC),
    )
    SLOWED_DOWN_OBJECT_POSES = SUBSAMPLED_OBJECT_POSES.copy()

    # Interpolate
    INTERPOLATED_Q_TIMES = np.arange(
        SLOWED_DOWN_Q_TIMES[0], SLOWED_DOWN_Q_TIMES[-1], args.dt
    )
    INTERPOLATED_Q_SOLUTIONS = interpolate(
        t=SLOWED_DOWN_Q_TIMES, x=SLOWED_DOWN_Q_SOLUTIONS, new_t=INTERPOLATED_Q_TIMES
    )
    INTERPOLATED_OBJECT_POSES = interpolate_poses(
        t=SLOWED_DOWN_Q_TIMES, x=SLOWED_DOWN_OBJECT_POSES, new_t=INTERPOLATED_Q_TIMES
    )

    if not args.headless:
        print("SHOWING ANIMATION OF CLEANED UP")
        last_update_time = time.time()
        N_pts = INTERPOLATED_Q_SOLUTIONS.shape[0]
        for i in tqdm(range(N_pts), desc="Animating robot"):
            q = INTERPOLATED_Q_SOLUTIONS[i]
            object_pose = INTERPOLATED_OBJECT_POSES[i]

            set_robot_state(robot, q)

            obj_xyz, obj_quat_xyzw = (
                object_pose[:3, 3],
                R.from_matrix(object_pose[:3, :3]).as_quat(),
            )
            pb.resetBasePositionAndOrientation(
                obj,
                obj_xyz,
                obj_quat_xyzw,
            )

            time_since_last_update = time.time() - last_update_time
            if time_since_last_update <= args.dt:
                time.sleep(args.dt - time_since_last_update)
            else:
                print(
                    f"WARNING: Time since last update {time_since_last_update} > dt {args.dt}"
                )
            last_update_time = time.time()

    if args.output_filepath is not None:
        args.output_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_filepath,
            interpolated_qs=INTERPOLATED_Q_SOLUTIONS,
            interpolated_ts=INTERPOLATED_Q_TIMES,
            interpolated_object_poses=INTERPOLATED_OBJECT_POSES,
            qs=Q_SOLUTIONS,
            ts=Q_TIMES,
            idxs=Q_IDXS,
            object_poses=OBJECT_POSES,
            start_idx=args.start_idx,
            relative_premanip_poses=np.array(RELATIVE_PREMANIP_POSE_LIST),
        )
        print(f"Saved to {args.output_filepath}")


if __name__ == "__main__":
    main()
