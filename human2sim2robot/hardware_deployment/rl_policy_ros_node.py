#!/usr/bin/env python

import copy
import functools
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Pose
from print_utils import get_ros_loop_rate_str
from rl_player import RlPlayer
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

from human2sim2robot.sim_training.utils.cross_embodiment.camera_extrinsics import (
    REALSENSE_CAMERA_T_R_C,
    ZED_CAMERA_T_R_C,
)
from human2sim2robot.sim_training.utils.cross_embodiment.constants import (
    NUM_XYZ,
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


def var_to_is_none_str(var) -> str:
    if var is None:
        return "None"
    return "Not None"


def pose_msg_to_T(msg: Pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
    T[:3, :3] = R.from_quat(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    ).as_matrix()
    return T


def T_to_pos_quat_xyzw(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = T[:3, 3]
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat_xyzw


FABRIC_MODE: Literal["PCA", "ALL"] = "PCA"


class RLPolicyNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("rl_policy_node")

        # Publisher for palm and hand targets
        self.palm_target_pub = rospy.Publisher(
            "/palm_target", Float64MultiArray, queue_size=10
        )
        self.hand_target_pub = rospy.Publisher(
            "/hand_target", Float64MultiArray, queue_size=10
        )

        # Variables to store the latest messages
        self.object_pose_msg = None
        self.goal_object_pose_msg = None
        self.iiwa_joint_state_msg = None
        self.allegro_joint_state_msg = None
        self.fabric_state_msg = None
        self.received_fabric_state_time = None

        self.prev_object_pose_msg = None
        self.prev_prev_object_pose_msg = None

        # Subscribers
        self.object_pose_sub = rospy.Subscriber(
            "/object_pose", Pose, self.object_pose_callback
        )
        self.goal_object_pose_sub = rospy.Subscriber(
            "/goal_object_pose", Pose, self.goal_object_pose_callback
        )
        self.iiwa_joint_state_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_joint_state_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback
        )
        self.fabric_sub = rospy.Subscriber(
            "/fabric_state", JointState, self.fabric_state_callback
        )

        # RL Player setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_observations = 144  # Update this number based on actual dimensions
        if FABRIC_MODE == "PCA":
            self.num_actions = 11  # First 6 for palm, last 5 for hand
        elif FABRIC_MODE == "ALL":
            self.num_actions = 22  # First 6 for palm, last 16 for hand
        else:
            raise ValueError(f"Invalid FABRIC_MODE: {FABRIC_MODE}")

        _, self.config_path = restore_file_from_wandb(
            "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-31_FINAL_TASKS/files/runs/snackbox_lift_wall_CUROBO_RANDPOS_25Hand_gcloud_2025-01-31_18-48-46-127371/config_resolved.yaml?runName=snackbox_lift_wall_CUROBO_RANDPOS_25Hand_gcloud_2025-01-31_18-48-46-127371_uw8i7n9x"
        )
        _, self.checkpoint_path = restore_file_from_wandb(
            "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-31_FINAL_TASKS/files/runs/snackbox_lift_wall_CUROBO_RANDPOS_25Hand_gcloud_2025-01-31_18-48-46-127371/nn/best.pth?runName=snackbox_lift_wall_CUROBO_RANDPOS_25Hand_gcloud_2025-01-31_18-48-46-127371_uw8i7n9x"
        )

        # Create the RL player
        self.player = RlPlayer(
            num_observations=self.num_observations,
            num_actions=self.num_actions,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        # ROS rate
        # self.rate_hz = 15
        # self.rate_hz = 60
        control_dt = (
            self.player.cfg["task"]["env"]["controlFrequencyInv"]
            * self.player.cfg["task"]["sim"]["dt"]
        )
        self.rate_hz = 1.0 / control_dt
        self.rate = rospy.Rate(self.rate_hz)

        # Define limits for palm and hand targets
        self.palm_mins = torch.tensor(
            [0.0, -0.7, 0, -3.1416, -3.1416, -3.1416], device=self.device
        )
        self.palm_maxs = torch.tensor(
            [1.0, 0.7, 1.0, 3.1416, 3.1416, 3.1416], device=self.device
        )

        hand_action_space = self.player.cfg["task"]["env"]["custom"][
            "FABRIC_HAND_ACTION_SPACE"
        ]
        assert hand_action_space == FABRIC_MODE, (
            f"Invalid hand action space: {hand_action_space} != {FABRIC_MODE}"
        )
        if FABRIC_MODE == "PCA":
            self.hand_mins = torch.tensor(
                [0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=self.device
            )
            self.hand_maxs = torch.tensor(
                [3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=self.device
            )
        elif FABRIC_MODE == "ALL":
            self.hand_mins = torch.tensor(
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
                device=self.device,
            )
            self.hand_maxs = torch.tensor(
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
                device=self.device,
            )
        else:
            raise ValueError(f"Invalid FABRIC_MODE = {FABRIC_MODE}")

        self._setup_taskmap()
        self.t = 0

    def object_pose_callback(self, msg: Pose):
        self.object_pose_msg = msg

    def goal_object_pose_callback(self, msg: Pose):
        self.goal_object_pose_msg = msg

    def iiwa_joint_state_callback(self, msg: JointState):
        self.iiwa_joint_state_msg = msg

    def allegro_joint_state_callback(self, msg: JointState):
        self.allegro_joint_state_msg = msg

    def fabric_state_callback(self, msg: JointState):
        self.fabric_state_msg = msg
        self.received_fabric_state_time = rospy.Time.now()

    def create_observation(self) -> Optional[torch.Tensor]:
        # Ensure all messages are received before processing
        if (
            self.iiwa_joint_state_msg is None
            or self.allegro_joint_state_msg is None
            or self.object_pose_msg is None
            or self.goal_object_pose_msg is None
            or self.fabric_state_msg is None
        ):
            rospy.logwarn(
                f"Waiting for all messages to be received... iiwa_joint_state_msg: {var_to_is_none_str(self.iiwa_joint_state_msg)}, allegro_joint_state_msg: {var_to_is_none_str(self.allegro_joint_state_msg)}, object_pose_msg: {var_to_is_none_str(self.object_pose_msg)}, goal_object_pose_msg: {var_to_is_none_str(self.goal_object_pose_msg)}, fabric_state_msg: {var_to_is_none_str(self.fabric_state_msg)}"
            )
            return None

        # Stop if the fabric states are not received for a long time
        assert self.received_fabric_state_time is not None
        MAX_DT_FABRIC_STATE_SEC = 0.5
        time_since_fabric_state = (
            rospy.Time.now() - self.received_fabric_state_time
        ).to_sec()
        if time_since_fabric_state > MAX_DT_FABRIC_STATE_SEC:
            log_msg = (
                f"Did not receive fabric states for {time_since_fabric_state} seconds"
            )
            rospy.logerr(log_msg)
            raise ValueError(log_msg)

        iiwa_joint_state_msg = copy.copy(self.iiwa_joint_state_msg)
        allegro_joint_state_msg = copy.copy(self.allegro_joint_state_msg)
        object_pose_msg = copy.copy(self.object_pose_msg)
        goal_object_pose_msg = copy.copy(self.goal_object_pose_msg)
        fabric_state_msg = copy.copy(self.fabric_state_msg)

        # Concatenate the data from joint states and object pose
        iiwa_position = np.array(iiwa_joint_state_msg.position)
        iiwa_velocity = np.array(iiwa_joint_state_msg.velocity)

        allegro_position = np.array(allegro_joint_state_msg.position)
        allegro_velocity = np.array(allegro_joint_state_msg.velocity)

        T_C_O = pose_msg_to_T(object_pose_msg)
        T_C_G = pose_msg_to_T(goal_object_pose_msg)

        if self.prev_object_pose_msg is not None:
            T_C_O_prev = pose_msg_to_T(self.prev_object_pose_msg)
        else:
            T_C_O_prev = T_C_O

        if self.prev_prev_object_pose_msg is not None:
            T_C_O_prev_prev = pose_msg_to_T(self.prev_prev_object_pose_msg)
        else:
            T_C_O_prev_prev = T_C_O_prev

        self.prev_prev_object_pose_msg = self.prev_object_pose_msg
        self.prev_object_pose_msg = object_pose_msg

        T_R_O = self.T_R_C @ T_C_O
        object_position_R, object_quat_xyzw_R = T_to_pos_quat_xyzw(T_R_O)

        T_R_G = self.goal_T_R_C @ T_C_G
        goal_object_pos_R, goal_object_quat_xyzw_R = T_to_pos_quat_xyzw(T_R_G)

        T_R_O_prev = self.T_R_C @ T_C_O_prev
        object_position_R_prev, object_quat_xyzw_R_prev = T_to_pos_quat_xyzw(T_R_O_prev)

        T_R_O_prev_prev = self.T_R_C @ T_C_O_prev_prev
        object_position_R_prev_prev, object_quat_xyzw_R_prev_prev = T_to_pos_quat_xyzw(
            T_R_O_prev_prev
        )

        keypoint_offsets = to_torch(
            OBJECT_KEYPOINT_OFFSETS,
            device=self.device,
            dtype=torch.float,
        )
        assert_equals(keypoint_offsets.shape, (NUM_OBJECT_KEYPOINTS, NUM_XYZ))

        q = np.concatenate([iiwa_position, allegro_position])
        qd = np.concatenate([iiwa_velocity, allegro_velocity])

        fabric_q = np.array(fabric_state_msg.position)
        fabric_qd = np.array(fabric_state_msg.velocity)

        taskmap_positions, _, _ = self.taskmap_helper(
            q=torch.from_numpy(q).float().unsqueeze(0).to(self.device),
            qd=torch.from_numpy(qd).float().unsqueeze(0).to(self.device),
        )
        taskmap_positions = taskmap_positions.squeeze(0).cpu().numpy()
        palm_pos = taskmap_positions[self.taskmap_link_names.index(PALM_LINK_NAME)]
        palm_x_pos = taskmap_positions[self.taskmap_link_names.index(PALM_X_LINK_NAME)]
        palm_y_pos = taskmap_positions[self.taskmap_link_names.index(PALM_Y_LINK_NAME)]
        palm_z_pos = taskmap_positions[self.taskmap_link_names.index(PALM_Z_LINK_NAME)]
        fingertip_positions = np.stack(
            [
                taskmap_positions[self.taskmap_link_names.index(link_name)]
                for link_name in ALLEGRO_FINGERTIP_LINK_NAMES
            ],
            axis=0,
        )

        obs_dict = {}
        obs_dict["q"] = np.concatenate([iiwa_position, allegro_position])
        obs_dict["qd"] = np.concatenate([iiwa_velocity, allegro_velocity])
        obs_dict["fingertip_positions"] = fingertip_positions.reshape(
            NUM_FINGERS * NUM_XYZ
        )
        obs_dict["palm_pos"] = palm_pos
        obs_dict["palm_x_pos"] = palm_x_pos
        obs_dict["palm_y_pos"] = palm_y_pos
        obs_dict["palm_z_pos"] = palm_z_pos
        obs_dict["object_pos"] = object_position_R
        obs_dict["object_quat_xyzw"] = object_quat_xyzw_R
        obs_dict["goal_pos"] = goal_object_pos_R
        obs_dict["goal_quat_xyzw"] = goal_object_quat_xyzw_R

        obs_dict["prev_object_pos"] = object_position_R_prev
        obs_dict["prev_object_quat_xyzw"] = object_quat_xyzw_R_prev
        obs_dict["prev_prev_object_pos"] = object_position_R_prev_prev
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
        assert_equals(observation.shape, (self.num_observations,))

        return torch.from_numpy(observation).float().unsqueeze(0).to(self.device)

    def _setup_taskmap(self) -> None:
        import warp as wp

        wp.init()
        from fabrics_sim.taskmaps.robot_frame_origins_taskmap import (
            RobotFrameOriginsTaskMap,
        )

        # Create task map that consists of the origins of the following frames stacked together.
        self.taskmap_link_names = PALM_LINK_NAMES + ALLEGRO_FINGERTIP_LINK_NAMES
        self.taskmap = RobotFrameOriginsTaskMap(
            urdf_path=str(Path(KUKA_ALLEGRO_ASSET_ROOT) / KUKA_ALLEGRO_FILENAME),
            link_names=self.taskmap_link_names,
            batch_size=1,
            device=self.device,
        )

    def taskmap_helper(
        self, q: torch.Tensor, qd: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = q.shape[0]
        assert_equals(q.shape, (N, KUKA_ALLEGRO_NUM_DOFS))
        assert_equals(qd.shape, (N, KUKA_ALLEGRO_NUM_DOFS))

        x, jac = self.taskmap(q, None)
        n_points = len(self.taskmap_link_names)
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

    def rescale_action(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = action.shape[0]
        assert_equals(action.shape, (N, self.num_actions))

        # Rescale the normalized actions from [-1, 1] to the actual target ranges
        palm_target = rescale(
            values=action[:, :6],
            old_mins=torch.ones_like(self.palm_mins) * -1,
            old_maxs=torch.ones_like(self.palm_maxs) * 1,
            new_mins=self.palm_mins,
            new_maxs=self.palm_maxs,
        )
        hand_target = rescale(
            values=action[:, 6:],
            old_mins=torch.ones_like(self.hand_mins) * -1,
            old_maxs=torch.ones_like(self.hand_maxs) * 1,
            new_mins=self.hand_mins,
            new_maxs=self.hand_maxs,
        )
        return palm_target, hand_target

    def publish_targets(self, palm_target: torch.Tensor, hand_target: torch.Tensor):
        # Convert palm_target to Float64MultiArray and publish
        palm_msg = Float64MultiArray()
        palm_msg.layout = MultiArrayLayout(
            dim=[MultiArrayDimension(label="palm_target", size=6, stride=6)],
            data_offset=0,
        )
        palm_msg.data = palm_target.cpu().numpy().flatten().tolist()
        self.palm_target_pub.publish(palm_msg)

        # Convert hand_target to Float64MultiArray and publish
        if FABRIC_MODE == "PCA":
            num_hand_actions = 5
        elif FABRIC_MODE == "ALL":
            num_hand_actions = 16
        else:
            raise ValueError(f"Invalid FABRIC_MODE = {FABRIC_MODE}")

        hand_msg = Float64MultiArray()
        hand_msg.layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(
                    label="hand_target", size=num_hand_actions, stride=num_hand_actions
                )
            ],
            data_offset=0,
        )
        hand_msg.data = hand_target.cpu().numpy().flatten().tolist()
        self.hand_target_pub.publish(hand_msg)

    def run(self):
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Create observation from the latest messages
            obs = self.create_observation()

            if obs is not None:
                assert_equals(obs.shape, (1, self.num_observations))

                # Get the normalized action from the RL player
                normalized_action = self.player.get_normalized_action(
                    obs=obs,
                    deterministic_actions=False,
                    # obs=obs, deterministic_actions=True
                )
                # normalized_action = torch.zeros(1, self.num_actions, device=self.device)
                assert_equals(normalized_action.shape, (1, self.num_actions))

                # Rescale the action to get palm and hand targets
                palm_target, hand_target = self.rescale_action(normalized_action)
                assert_equals(palm_target.shape, (1, 6))

                if FABRIC_MODE == "PCA":
                    num_hand_actions = 5
                elif FABRIC_MODE == "ALL":
                    num_hand_actions = 16
                else:
                    raise ValueError(f"Invalid FABRIC_MODE = {FABRIC_MODE}")
                assert_equals(hand_target.shape, (1, num_hand_actions))
                palm_target = palm_target.squeeze(0)
                hand_target = hand_target.squeeze(0)

                # DEBUG
                # print(f"normalized_action: {normalized_action}")
                # print(f"palm_target: {palm_target}")
                # print(f"hand_target: {hand_target}")
                # print()

                # Publish the targets
                self.publish_targets(palm_target, hand_target)

            # Sleep to maintain 15 loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()

            rospy.loginfo(
                get_ros_loop_rate_str(
                    start_time=start_time,
                    before_sleep_time=before_sleep_time,
                    after_sleep_time=after_sleep_time,
                    node_name=rospy.get_name(),
                )
            )

    @property
    @functools.lru_cache()
    def camera(self) -> Literal["zed", "realsense"]:
        # Check camera parameter
        camera = rospy.get_param("/camera", None)
        if camera is None:
            DEFAULT_CAMERA = "zed"
            rospy.logwarn(
                f"No /camera parameter found, using default camera {DEFAULT_CAMERA}"
            )
            camera = DEFAULT_CAMERA
        rospy.loginfo(f"Using camera: {camera}")
        assert camera in ["zed", "realsense"], f"camera: {camera}"
        return camera

    @property
    @functools.lru_cache()
    def goal_camera(self) -> Literal["zed", "realsense"]:
        # Check goal_camera parameter
        goal_camera = rospy.get_param("/goal_camera", None)
        if goal_camera is None:
            DEFAULT_CAMERA = "zed"
            rospy.logwarn(
                f"No /goal_camera parameter found, using default camera {DEFAULT_CAMERA}"
            )
            goal_camera = DEFAULT_CAMERA
        rospy.loginfo(f"Using goal_camera: {goal_camera}")
        assert goal_camera in ["zed", "realsense"], f"goal_camera: {goal_camera}"
        return goal_camera

    @property
    @functools.lru_cache()
    def T_R_C(self) -> np.ndarray:
        if self.camera == "zed":
            return ZED_CAMERA_T_R_C
        elif self.camera == "realsense":
            return REALSENSE_CAMERA_T_R_C
        else:
            raise ValueError(f"Unknown camera: {self.camera}")

    @property
    @functools.lru_cache()
    def goal_T_R_C(self) -> np.ndarray:
        if self.goal_camera == "zed":
            return ZED_CAMERA_T_R_C
        elif self.goal_camera == "realsense":
            return REALSENSE_CAMERA_T_R_C
        else:
            raise ValueError(f"Unknown goal_camera: {self.goal_camera}")


if __name__ == "__main__":
    try:
        rl_policy_node = RLPolicyNode()
        rl_policy_node.run()
    except rospy.ROSInterruptException:
        pass
