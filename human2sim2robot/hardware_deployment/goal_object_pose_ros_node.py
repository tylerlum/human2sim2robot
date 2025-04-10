#!/usr/bin/env python

import functools
from typing import Literal

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R

from human2sim2robot.sim_training import get_data_dir
from human2sim2robot.sim_training.utils.cross_embodiment.camera_extrinsics import (
    REALSENSE_CAMERA_T_R_C,
    ZED_CAMERA_T_R_C,
)
from human2sim2robot.sim_training.utils.cross_embodiment.utils import (
    clip_T_list,
    read_in_T_list,
)


class GoalObjectPosePublisher:
    def __init__(self):
        rospy.init_node("goal_object_pose_publisher")

        self.pose_pub = rospy.Publisher("/goal_object_pose", Pose, queue_size=1)
        self.rate_hz = 30
        self.rate = rospy.Rate(self.rate_hz)
        self.current_index = 0

        MODE: Literal["trajectory", "position"] = "trajectory"
        if MODE == "trajectory":
            # Set up VecOliviaReferenceMotion
            # TASK_NAME = "snackbox_pivot_hard_onestep"
            # TASK_NAME = "ladel_hard_scoop"
            # TASK_NAME = "plate_hard"
            # TASK_NAME = "watering_can"

            # TASK_NAME = "snackbox_pivot"
            # TASK_NAME = "snackbox_pivotmove"
            # TASK_NAME = "plate_pivotrack"

            # TASK_NAME = "plate_push"
            # TASK_NAME = "plate_pivotrack_redo"
            # TASK_NAME = "wateringcan_pour_redo"

            # TASK_NAME = "snackbox_pour_arc"
            # TASK_NAME = "snackbox_pour"
            # TASK_NAME = "snackbox_pivot"
            # TASK_NAME = "snackbox_push"
            TASK_NAME = "snackbox_lift_wall"
            # TASK_NAME = "snackbox_pushpivot"
            # TASK_NAME = "snackbox_pushpivotmove"
            # TASK_NAME = "snackbox_push"
            # TASK_NAME = "plate_push"
            # TASK_NAME = "plate_pivotrack_redo"
            # TASK_NAME = "plate_rack_wall"
            # TASK_NAME = "plate_pushpivotrack"
            # TASK_NAME = "wateringcan_pour_redo"
            # TASK_NAME = "pot_pour_sidegrasp_norm"

            TRAJECTORY_FOLDERPATH = (
                get_data_dir()
                / "human_demo_processed_data"
                / TASK_NAME
                / "object_pose_trajectory"
                / "ob_in_cam"
            )

            self.data_hz = 30
            data_dt = 1 / self.data_hz

            raw_T_list = (
                torch.from_numpy(read_in_T_list(TRAJECTORY_FOLDERPATH))
                .float()
                .to("cuda")
            )
            T_C_Os = clip_T_list(raw_T_list, data_dt).cpu().numpy()
            self.T_C_O_list = [T_C_Os[i] for i in range(T_C_Os.shape[0])]

            # Control speed of the replay
            # Data rate is 30Hz
            # If we publish at 30Hz, we will publish 1x speed
            # If we publish at 60Hz, we will publish 2x speed
            # If we publish at 15Hz, we will publish 0.5x speed
            SPEEDUP_FACTOR = 0.5
            # SPEEDUP_FACTOR = 1.0
            self.rate_hz = self.data_hz * SPEEDUP_FACTOR
            self.rate = rospy.Rate(self.rate_hz)

            self.N = len(self.T_C_O_list)

        elif MODE == "position":
            # goal_object_pos = np.array([0.4637, -0.2200, 0.5199])
            goal_object_pos = np.array([0.5735, -0.1633, 0.2038]) + np.array(
                [0.0, -0.12, 0.35]
            )
            goal_object_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
            T_R_O = np.eye(4)
            T_R_O[:3, :3] = R.from_quat(goal_object_quat_xyzw).as_matrix()
            T_R_O[:3, 3] = goal_object_pos

            T_C_R = np.linalg.inv(self.goal_T_R_C)
            T_C_O = T_C_R @ T_R_O

            self.T_C_O_list = [T_C_O]
            self.N = len(self.T_C_O_list)
        else:
            raise ValueError(f"Invalid mode {MODE}")

    def _get_goal_object_pose_olivia_helper(self, reference_motion, device, num_envs):
        from human2sim2robot.sim_training.utils.torch_jit_utils import (
            matrix_to_quat_xyzw,
            quat_xyzw_to_matrix,
        )

        T_C_Os = (
            torch.eye(4, device=device)
            .unsqueeze(dim=0)
            .repeat_interleave(num_envs, dim=0)
        )
        T_C_Os[:, :3, 3] = reference_motion.object_pos
        T_C_Os[:, :3, :3] = quat_xyzw_to_matrix(reference_motion.object_quat_xyzw)

        new_goal_object_pos = T_C_Os[:, :3, 3]
        new_goal_object_quat_xyzw = matrix_to_quat_xyzw(T_C_Os[:, :3, :3])

        return new_goal_object_pos, new_goal_object_quat_xyzw, None, None

    @staticmethod
    def create_transform(pos, rot):
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    @staticmethod
    def quat_xyzw_to_matrix(quat_xyzw):
        return R.from_quat(quat_xyzw).as_matrix()

    def publish_pose(self):
        if self.current_index >= self.N:
            self.current_index = self.N - 1

        T = self.T_C_O_list[self.current_index]
        trans = T[:3, 3]
        quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()

        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = trans
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (
            quat_xyzw
        )

        self.pose_pub.publish(msg)
        rospy.logdebug(f"Pose {self.current_index} published to /goal_object_pose")

        self.current_index += 1

    def run(self):
        while not rospy.is_shutdown():
            rospy.loginfo(
                f"Publishing goal object poses at {self.rate_hz}Hz to /goal_object_pose"
            )
            self.publish_pose()
            self.rate.sleep()

    @property
    @functools.lru_cache()
    def goal_T_R_C(self) -> np.ndarray:
        # Check camera parameter
        camera = rospy.get_param("/goal_camera", None)
        if camera is None:
            DEFAULT_CAMERA = "zed"
            rospy.logwarn(
                f"No /camera parameter found, using default camera {DEFAULT_CAMERA}"
            )
            camera = DEFAULT_CAMERA
        rospy.loginfo(f"Using camera: {camera}")
        if camera == "zed":
            return ZED_CAMERA_T_R_C
        elif camera == "realsense":
            return REALSENSE_CAMERA_T_R_C
        else:
            raise ValueError(f"Unknown camera: {camera}")


if __name__ == "__main__":
    try:
        node = GoalObjectPosePublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
