#!/usr/bin/env python
from human2sim2robot.sim_training.utils.cross_embodiment.create_env import create_env  # isort:skip
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from fabrics_sim.fabrics.kuka_allegro_pose_allhand_fabric import (
    KukaAllegroPoseAllHandFabric,
)

# Import from the fabrics_sim package
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import capture_fabric, initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from rl_player import RlPlayer

from human2sim2robot.sim_training.utils.cross_embodiment.constants import (
    NUM_XYZ,
)
from human2sim2robot.sim_training.utils.cross_embodiment.fabric_world import (
    world_dict_robot_frame,
)
from human2sim2robot.sim_training.utils.cross_embodiment.kuka_allegro_constants import (
    ALLEGRO_FINGERTIP_LINK_NAMES,
    KUKA_ALLEGRO_ASSET_ROOT,
    KUKA_ALLEGRO_FILENAME,
    NUM_FINGERS,
    PALM_LINK_NAME,
    PALM_LINK_NAMES,
    PALM_X_LINK_NAME,
    PALM_Y_LINK_NAME,
    PALM_Z_LINK_NAME,
)
from human2sim2robot.sim_training.utils.cross_embodiment.kuka_allegro_constants import (
    NUM_HAND_ARM_DOFS as KUKA_ALLEGRO_NUM_DOFS,
)
from human2sim2robot.sim_training.utils.cross_embodiment.object_constants import (
    NUM_OBJECT_KEYPOINTS,
    OBJECT_KEYPOINT_OFFSETS,
)
from human2sim2robot.sim_training.utils.cross_embodiment.utils import (
    assert_equals,
    rescale,
)
from human2sim2robot.sim_training.utils.torch_utils import to_torch
from human2sim2robot.sim_training.utils.wandb_utils import restore_file_from_wandb

FABRIC_MODE: Literal["PCA", "ALL"] = "PCA"

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16


def taskmap_helper(
    q: torch.Tensor, qd: torch.Tensor, taskmap, taskmap_link_names: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = q.shape[0]
    assert_equals(q.shape, (N, KUKA_ALLEGRO_NUM_DOFS))
    assert_equals(qd.shape, (N, KUKA_ALLEGRO_NUM_DOFS))

    x, jac = taskmap(q, None)
    n_points = len(taskmap_link_names)
    assert_equals(x.shape, (N, NUM_XYZ * n_points))
    assert_equals(jac.shape, (N, NUM_XYZ * n_points, KUKA_ALLEGRO_NUM_DOFS))

    # Calculate the velocity in the task space
    xd = torch.bmm(jac, qd.unsqueeze(2)).squeeze(2)
    assert_equals(xd.shape, (N, NUM_XYZ * n_points))

    return (
        x.reshape(N, n_points, NUM_XYZ),
        xd.reshape(N, n_points, NUM_XYZ),
        jac.reshape(N, n_points, NUM_XYZ, KUKA_ALLEGRO_NUM_DOFS),
    )


def create_observation(
    iiwa_position: np.ndarray,
    iiwa_velocity: np.ndarray,
    allegro_position: np.ndarray,
    allegro_velocity: np.ndarray,
    fabric_q: np.ndarray,
    fabric_qd: np.ndarray,
    object_position_R: np.ndarray,
    object_quat_xyzw_R: np.ndarray,
    goal_object_pos_R: np.ndarray,
    goal_object_quat_xyzw_R: np.ndarray,
    object_pos_R_prev: np.ndarray,
    object_quat_xyzw_R_prev: np.ndarray,
    object_pos_R_prev_prev: np.ndarray,
    object_quat_xyzw_R_prev_prev: np.ndarray,
    device: torch.device,
    taskmap,
    taskmap_link_names: List[str],
    num_observations: int,
) -> Optional[torch.Tensor]:
    keypoint_offsets = to_torch(
        OBJECT_KEYPOINT_OFFSETS,
        device=device,
        dtype=torch.float,
    )
    assert_equals(keypoint_offsets.shape, (NUM_OBJECT_KEYPOINTS, NUM_XYZ))

    q = np.concatenate([iiwa_position, allegro_position])
    qd = np.concatenate([iiwa_velocity, allegro_velocity])

    taskmap_positions, _, _ = taskmap_helper(
        q=torch.from_numpy(q).float().unsqueeze(0).to(device),
        qd=torch.from_numpy(qd).float().unsqueeze(0).to(device),
        taskmap=taskmap,
        taskmap_link_names=taskmap_link_names,
    )
    taskmap_positions = taskmap_positions.squeeze(0).cpu().numpy()
    palm_pos = taskmap_positions[taskmap_link_names.index(PALM_LINK_NAME)]
    palm_x_pos = taskmap_positions[taskmap_link_names.index(PALM_X_LINK_NAME)]
    palm_y_pos = taskmap_positions[taskmap_link_names.index(PALM_Y_LINK_NAME)]
    palm_z_pos = taskmap_positions[taskmap_link_names.index(PALM_Z_LINK_NAME)]
    fingertip_positions = np.stack(
        [
            taskmap_positions[taskmap_link_names.index(link_name)]
            for link_name in ALLEGRO_FINGERTIP_LINK_NAMES
        ],
        axis=0,
    )

    obs_dict = {}
    obs_dict["q"] = np.concatenate([iiwa_position, allegro_position])
    obs_dict["qd"] = np.concatenate([iiwa_velocity, allegro_velocity])
    obs_dict["fingertip_positions"] = fingertip_positions.reshape(NUM_FINGERS * NUM_XYZ)
    obs_dict["palm_pos"] = palm_pos
    obs_dict["palm_x_pos"] = palm_x_pos
    obs_dict["palm_y_pos"] = palm_y_pos
    obs_dict["palm_z_pos"] = palm_z_pos
    obs_dict["object_pos"] = object_position_R
    obs_dict["object_quat_xyzw"] = object_quat_xyzw_R
    obs_dict["goal_pos"] = goal_object_pos_R
    obs_dict["goal_quat_xyzw"] = goal_object_quat_xyzw_R

    obs_dict["prev_object_pos"] = object_pos_R_prev
    obs_dict["prev_object_quat_xyzw"] = object_quat_xyzw_R_prev
    obs_dict["prev_prev_object_pos"] = object_pos_R_prev_prev
    obs_dict["prev_prev_object_quat_xyzw"] = object_quat_xyzw_R_prev_prev

    # object_keypoint_positions = (
    #     compute_keypoint_positions(
    #         pos=torch.tensor(object_position_R, device=self.device)
    #         .unsqueeze(0)
    #         .float(),
    #         quat_xyzw=torch.tensor(object_quat_xyzw_R, device=self.device)
    #         .unsqueeze(0)
    #         .float(),
    #         keypoint_offsets=keypoint_offsets.unsqueeze(0).float(),
    #     )
    #     .squeeze(0)
    #     .cpu()
    #     .numpy()
    # )
    # goal_object_keypoint_positions = (
    #     compute_keypoint_positions(
    #         pos=torch.tensor(goal_object_pos_R, device=self.device)
    #         .unsqueeze(0)
    #         .float(),
    #         quat_xyzw=torch.tensor(goal_object_quat_xyzw_R, device=self.device)
    #         .unsqueeze(0)
    #         .float(),
    #         keypoint_offsets=keypoint_offsets.unsqueeze(0).float(),
    #     )
    #     .squeeze(0)
    #     .cpu()
    #     .numpy()
    # )
    # object_vel = np.zeros(3)
    # object_angvel = np.zeros(3)
    # obs_dict["object_keypoint_positions"] = (
    #     object_keypoint_positions.reshape(
    #         NUM_OBJECT_KEYPOINTS * NUM_XYZ
    #     )
    # )
    # obs_dict["goal_object_keypoint_positions"] = (
    #     goal_object_keypoint_positions.reshape(
    #         NUM_OBJECT_KEYPOINTS * NUM_XYZ
    #     )
    # )
    # obs_dict["object_vel"] = object_vel
    # obs_dict["object_angvel"] = object_angvel
    # obs_dict["t"] = np.array([self.t]).reshape(1)
    # self.t += 1

    obs_dict["fabric_q"] = fabric_q
    obs_dict["fabric_qd"] = fabric_qd

    for k, v in obs_dict.items():
        assert len(v.shape) == 1, f"Shape of {k} is {v.shape}, expected 1D tensor"

    # DEBUG
    # for k, v in obs_dict.items():
    #     print(f"{k}: {v}")
    # print()

    # Concatenate all observations into a 1D tensor
    observation = np.concatenate(
        [obs for obs in obs_dict.values()],
        axis=-1,
    )
    assert_equals(observation.shape, (num_observations,))

    return torch.from_numpy(observation).float().unsqueeze(0).to(device)


def rescale_action(
    action: torch.Tensor,
    palm_mins: torch.Tensor,
    palm_maxs: torch.Tensor,
    hand_mins: torch.Tensor,
    hand_maxs: torch.Tensor,
    num_actions: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = action.shape[0]
    assert_equals(action.shape, (N, num_actions))

    # Rescale the normalized actions from [-1, 1] to the actual target ranges
    palm_target = rescale(
        values=action[:, :6],
        old_mins=torch.ones_like(palm_mins) * -1,
        old_maxs=torch.ones_like(palm_maxs) * 1,
        new_mins=palm_mins,
        new_maxs=palm_maxs,
    )
    hand_target = rescale(
        values=action[:, 6:],
        old_mins=torch.ones_like(hand_mins) * -1,
        old_maxs=torch.ones_like(hand_maxs) * 1,
        new_mins=hand_mins,
        new_maxs=hand_maxs,
    )
    return palm_target, hand_target


def main():
    """
    Env
    """
    # State
    iiwa_joint_q = None
    allegro_joint_q = None
    iiwa_joint_qd = None
    allegro_joint_qd = None
    object_pos_R = None
    object_quat_xyzw_R = None

    _, CONFIG_PATH = restore_file_from_wandb(
        "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-16_experiments/files/runs/plate_hard_65damp_move1_2025-01-16_00-53-04-618360/config_resolved.yaml?runName=plate_hard_65damp_move1_2025-01-16_00-53-04-618360_2amo0e8y"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = create_env(
        config_path=CONFIG_PATH,
        device=device,
        # headless=True,
        headless=False,
        # enable_viewer_sync_at_start=False,
        enable_viewer_sync_at_start=True,
    )

    # Set control rate
    control_dt = env.control_dt
    _sim_dt = env.sim_dt

    """
    Fabric
    """
    num_envs = 1  # Single environment for this example

    # Initialize warp
    initialize_warp(warp_cache_name="")

    # Set up the world model
    fabric_world_dict = world_dict_robot_frame
    # self.fabric_world_dict = None
    fabric_world_model = WorldMeshesModel(
        batch_size=num_envs,
        max_objects_per_env=20,
        device=device,
        world_dict=fabric_world_dict,
    )
    fabric_object_ids, fabric_object_indicator = fabric_world_model.get_object_ids()

    # Create Kuka-Allegro Pose Fabric
    if FABRIC_MODE == "PCA":
        fabric_class = KukaAllegroPoseFabric
    elif FABRIC_MODE == "ALL":
        fabric_class = KukaAllegroPoseAllHandFabric
    else:
        raise ValueError(f"Invalid FABRIC_MODE = {FABRIC_MODE}")
    fabric = fabric_class(
        batch_size=num_envs,
        device=device,
        timestep=control_dt,
        graph_capturable=True,
    )
    fabric_integrator = DisplacementIntegrator(fabric)

    # Initialize random targets for palm and hand
    if FABRIC_MODE == "PCA":
        num_hand_target = 5
    elif FABRIC_MODE == "ALL":
        num_hand_target = 16
    else:
        raise ValueError(f"Invalid FABRIC_MODE = {FABRIC_MODE}")
    fabric_hand_target = torch.zeros(
        num_envs, num_hand_target, device="cuda", dtype=torch.float
    )
    fabric_palm_target = torch.zeros(num_envs, 6, device="cuda", dtype=torch.float)

    # Joint states
    fabric_q = torch.zeros(num_envs, fabric.num_joints, device=device)
    fabric_qd = torch.zeros_like(fabric_q)
    fabric_qdd = torch.zeros_like(fabric_q)

    # Capture the fabric graph for CUDA optimization
    fabric_inputs = [
        fabric_hand_target,
        fabric_palm_target,
        "euler_zyx",
        fabric_q.detach(),
        fabric_qd.detach(),
        fabric_object_ids,
        fabric_object_indicator,
    ]
    (
        fabric_cuda_graph,
        fabric_q_new,
        fabric_qd_new,
        fabric_qdd_new,
    ) = capture_fabric(
        fabric=fabric,
        q=fabric_q,
        qd=fabric_qd,
        qdd=fabric_qdd,
        timestep=control_dt,
        fabric_integrator=fabric_integrator,
        inputs=fabric_inputs,
        device=device,
    )

    """
    RL Player
    """
    # RL Player setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_observations = 144  # Update this number based on actual dimensions
    if FABRIC_MODE == "PCA":
        num_actions = 11  # First 6 for palm, last 5 for hand
    elif FABRIC_MODE == "ALL":
        num_actions = 22  # First 6 for palm, last 16 for hand
    else:
        raise ValueError(f"Invalid FABRIC_MODE: {FABRIC_MODE}")

    _, config_path = restore_file_from_wandb(
        "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-16_experiments/files/runs/plate_hard_65damp_move1_2025-01-16_00-53-04-618360/config_resolved.yaml?runName=plate_hard_65damp_move1_2025-01-16_00-53-04-618360_2amo0e8y"
    )
    _, checkpoint_path = restore_file_from_wandb(
        "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-16_experiments/files/runs/plate_hard_65damp_move1_2025-01-16_00-53-04-618360/nn/best.pth?runName=plate_hard_65damp_move1_2025-01-16_00-53-04-618360_2amo0e8y"
    )

    # Create the RL player
    player = RlPlayer(
        num_observations=num_observations,
        num_actions=num_actions,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # ROS rate

    # Define limits for palm and hand targets
    palm_mins = torch.tensor([0.0, -0.7, 0, -3.1416, -3.1416, -3.1416], device=device)
    palm_maxs = torch.tensor([1.0, 0.7, 1.0, 3.1416, 3.1416, 3.1416], device=device)

    hand_action_space = player.cfg["task"]["env"]["custom"]["FABRIC_HAND_ACTION_SPACE"]
    assert hand_action_space == FABRIC_MODE, (
        f"Invalid hand action space: {hand_action_space} != {FABRIC_MODE}"
    )
    if FABRIC_MODE == "PCA":
        hand_mins = torch.tensor(
            [0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=device
        )
        hand_maxs = torch.tensor(
            [3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=device
        )
    elif FABRIC_MODE == "ALL":
        hand_mins = torch.tensor(
            [
                -0.4700,
                -0.1960,
                -0.1740,
                -0.2270,
                -0.4700,
                -0.1960,
                -0.1740,
                -0.2270,
                -0.4700,
                -0.1960,
                -0.1740,
                -0.2270,
                0.2630,
                -0.1050,
                -0.1890,
                -0.1620,
            ],
            device=device,
        )
        hand_maxs = torch.tensor(
            [
                0.4700,
                1.6100,
                1.7090,
                1.6180,
                0.4700,
                1.6100,
                1.7090,
                1.6180,
                0.4700,
                1.6100,
                1.7090,
                1.6180,
                1.3960,
                1.1630,
                1.6440,
                1.7190,
            ],
            device=device,
        )
    else:
        raise ValueError(f"Invalid FABRIC_MODE = {FABRIC_MODE}")

    import warp as wp

    wp.init()
    from fabrics_sim.taskmaps.robot_frame_origins_taskmap import (
        RobotFrameOriginsTaskMap,
    )

    # Create task map that consists of the origins of the following frames stacked together.
    taskmap_link_names = PALM_LINK_NAMES + ALLEGRO_FINGERTIP_LINK_NAMES
    taskmap = RobotFrameOriginsTaskMap(
        urdf_path=str(Path(KUKA_ALLEGRO_ASSET_ROOT) / KUKA_ALLEGRO_FILENAME),
        link_names=taskmap_link_names,
        batch_size=1,
        device=device,
    )

    """
    Loop
    """
    right_robot_dof_pos = env.right_robot_dof_pos[0]
    right_robot_dof_vel = env.right_robot_dof_vel[0]

    iiwa_joint_q = right_robot_dof_pos[:NUM_ARM_JOINTS]
    allegro_joint_q = right_robot_dof_pos[
        NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS
    ]
    iiwa_joint_qd = right_robot_dof_vel[:NUM_ARM_JOINTS]
    allegro_joint_qd = right_robot_dof_vel[
        NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS
    ]

    object_pos_R = env.object_pos[0]
    object_quat_xyzw_R = env.object_quat_xyzw[0]
    goal_object_pos_R = env.goal_object_pos[0]
    goal_object_quat_xyzw_R = env.goal_object_quat_xyzw[0]

    prev_object_pos_R = object_pos_R.clone()
    prev_object_quat_xyzw_R = object_quat_xyzw_R.clone()
    prev_prev_object_pos_R = prev_object_pos_R.clone()
    prev_prev_object_quat_xyzw_R = prev_object_quat_xyzw_R.clone()

    fabric_q.copy_(
        torch.from_numpy(
            np.concatenate(
                [
                    iiwa_joint_q.detach().cpu().numpy(),
                    allegro_joint_q.detach().cpu().numpy(),
                ],
                axis=0,
            )
        )
        .float()
        .to(device)
    )

    while True:
        """
        Compute action from RL player
        """
        # Create observation from the latest messages
        obs = create_observation(
            iiwa_position=iiwa_joint_q.detach().cpu().numpy(),
            iiwa_velocity=iiwa_joint_qd.detach().cpu().numpy(),
            allegro_position=allegro_joint_q.detach().cpu().numpy(),
            allegro_velocity=allegro_joint_qd.detach().cpu().numpy(),
            fabric_q=fabric_q.detach().cpu().numpy()[0],
            fabric_qd=fabric_qd.detach().cpu().numpy()[0],
            object_position_R=object_pos_R.detach().cpu().numpy(),
            object_quat_xyzw_R=object_quat_xyzw_R.detach().cpu().numpy(),
            goal_object_pos_R=goal_object_pos_R.detach().cpu().numpy(),
            goal_object_quat_xyzw_R=goal_object_quat_xyzw_R.detach().cpu().numpy(),
            object_pos_R_prev=prev_object_pos_R.detach().cpu().numpy(),
            object_quat_xyzw_R_prev=prev_object_quat_xyzw_R.detach().cpu().numpy(),
            object_pos_R_prev_prev=prev_prev_object_pos_R.detach().cpu().numpy(),
            object_quat_xyzw_R_prev_prev=prev_prev_object_quat_xyzw_R.detach()
            .cpu()
            .numpy(),
            device=device,
            taskmap=taskmap,
            taskmap_link_names=taskmap_link_names,
            num_observations=num_observations,
        )
        assert_equals(obs.shape, (1, num_observations))

        # Get the normalized action from the RL player
        normalized_action = player.get_normalized_action(
            obs=obs, deterministic_actions=False
        )
        assert_equals(normalized_action.shape, (1, num_actions))

        # Rescale the action to get palm and hand targets
        palm_target, hand_target = rescale_action(
            action=normalized_action,
            palm_mins=palm_mins,
            palm_maxs=palm_maxs,
            hand_mins=hand_mins,
            hand_maxs=hand_maxs,
            num_actions=num_actions,
        )
        assert_equals(palm_target.shape, (1, 6))

        """
        Step fabric
        """
        # Update fabric targets for palm and hand
        fabric_palm_target.copy_(
            torch.from_numpy(palm_target.detach().cpu().numpy()).float().to(device)
        )
        fabric_hand_target.copy_(
            torch.from_numpy(hand_target.detach().cpu().numpy()).float().to(device)
        )

        # Step the fabric using the captured CUDA graph
        fabric_cuda_graph.replay()
        fabric_q.copy_(fabric_q_new)
        fabric_qd.copy_(fabric_qd_new)
        fabric_qdd.copy_(fabric_qdd_new)

        """
        Step env
        """
        action = fabric_q.clone()

        env.step_no_fabric(action, set_dof_pos_targets=True)
        # real_control_freq_inv = env.control_freq_inv
        # for i in range(real_control_freq_inv):
        #     on_last_step = (i == real_control_freq_inv - 1)
        #     run_post_physics_step = on_last_step
        #     env.step_no_fabric(action, set_dof_pos_targets=True, control_freq_inv=1, run_post_physics_step=run_post_physics_step)
        #     if not on_last_step:
        #         fabric_cuda_graph.replay()
        #         fabric_q.copy_(fabric_q_new)
        #         fabric_qd.copy_(fabric_qd_new)
        #         fabric_qdd.copy_(fabric_qdd_new)
        #         action = fabric_q.clone()

        right_robot_dof_pos = env.right_robot_dof_pos[0]
        right_robot_dof_vel = env.right_robot_dof_vel[0]

        iiwa_joint_q = right_robot_dof_pos[:NUM_ARM_JOINTS]
        allegro_joint_q = right_robot_dof_pos[
            NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS
        ]
        iiwa_joint_qd = right_robot_dof_vel[:NUM_ARM_JOINTS]
        allegro_joint_qd = right_robot_dof_vel[
            NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS
        ]

        prev_prev_object_pos_R = prev_object_pos_R.clone()
        prev_prev_object_quat_xyzw_R = prev_object_quat_xyzw_R.clone()
        prev_object_pos_R = object_pos_R.clone()
        prev_object_quat_xyzw_R = object_quat_xyzw_R.clone()

        object_pos_R = env.object_pos[0]
        object_quat_xyzw_R = env.object_quat_xyzw[0]
        goal_object_pos_R = env.goal_object_pos[0]
        goal_object_quat_xyzw_R = env.goal_object_quat_xyzw[0]


if __name__ == "__main__":
    main()
