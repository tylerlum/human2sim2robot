#!/usr/bin/env python
from human2sim2robot.sim_training.utils.cross_embodiment.create_env import create_env  # isort:skip
import numpy as np
import rospy
import torch
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState

from human2sim2robot.hardware_deployment.utils.print_utils import get_ros_loop_rate_str
from human2sim2robot.sim_training.utils.cross_embodiment.camera_extrinsics import (
    ZED_CAMERA_T_R_C,
)
from human2sim2robot.sim_training.utils.wandb_utils import restore_file_from_wandb

# Hardcode use ZED camera
T_R_C = ZED_CAMERA_T_R_C

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16

DEFAULT_ARM_Q = np.array(
    [
        # LEFT
        # -0.0095571,
        # 0.87742555,
        # 0.28864127,
        # -2.0917962,
        # -1.434597,
        # 1.8186541,
        # 1.414263,
        # TOP
        -0.8689132,
        0.4176688,
        0.5549343,
        -2.0467792,
        -0.3155458,
        0.7586144,
        -0.12089629,
    ]
)
DEFAULT_HAND_Q = np.array(
    [
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
        1.2,
        0.6,
        0.3,
        0.6,
    ]
)

USE_CONTROL_DT = True


class IsaacFakeRobotNode:
    def __init__(self):
        # ROS setup
        rospy.init_node("isaac_fake_robot_ros_node")

        # ROS msgs
        self.iiwa_joint_cmd = None
        self.allegro_joint_cmd = None

        # Publisher and subscriber
        self.iiwa_pub = rospy.Publisher("/iiwa/joint_states", JointState, queue_size=10)
        self.allegro_pub = rospy.Publisher(
            "/allegroHand_0/joint_states", JointState, queue_size=10
        )
        self.pose_pub = rospy.Publisher("/object_pose", Pose, queue_size=1)
        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.iiwa_joint_cmd_callback
        )
        self.allegro_cmd_sub = rospy.Subscriber(
            "/allegroHand_0/joint_cmd", JointState, self.allegro_joint_cmd_callback
        )

        # State
        self.iiwa_joint_q = None
        self.allegro_joint_q = None
        self.iiwa_joint_qd = None
        self.allegro_joint_qd = None
        self.object_pos_R = None
        self.object_quat_xyzw_R = None

        # When only testing the arm, set this to False to ignore the Allegro hand
        self.WAIT_FOR_ALLEGRO_CMD = True
        if not self.WAIT_FOR_ALLEGRO_CMD:
            rospy.logwarn("NOT WAITING FOR ALLEGRO CMD")
            self.allegro_joint_cmd = np.zeros(NUM_HAND_JOINTS)

        _, CONFIG_PATH = restore_file_from_wandb(
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-10-05_cup_fabric_reset-early_multigpu/files/runs/TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674/config_resolved.yaml?runName=TOP_4-freq_coll-on_juno1_2_2024-10-07_23-27-58-967674"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-16_experiments/files/runs/plate_hard_65damp_move1_2025-01-16_00-53-04-618360/config_resolved.yaml?runName=plate_hard_65damp_move1_2025-01-16_00-53-04-618360_2amo0e8y"
            # "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-26_new_scene/files/runs/snackbox_pivot_move2_2025-01-26_04-33-44-941499/config_resolved.yaml?runName=snackbox_pivot_move2_2025-01-26_04-33-44-941499_v7l0sxlj"
            "https://wandb.ai/tylerlum/cross_embodiment/groups/2025-01-26_new_scene/files/runs/snackbox_pivotmove_FORCES_NOISE_LOWROBOT_gcloud_2025-01-27_09-20-08-282947/config_resolved.yaml?runName=snackbox_pivotmove_FORCES_NOISE_LOWROBOT_gcloud_2025-01-27_09-20-08-282947_g0w3baca"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = create_env(
            config_path=CONFIG_PATH,
            device=self.device,
            # headless=False,
            headless=True,
            enable_viewer_sync_at_start=True,
            # enable_viewer_sync_at_start=False,
        )

        # Set control rate
        if USE_CONTROL_DT:
            self.dt = self.env.control_dt
        else:
            self.dt = self.env.sim_dt
        self.rate_hz = 1 / self.dt
        self.rate = rospy.Rate(self.rate_hz)

    def iiwa_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.iiwa_joint_cmd = np.array(msg.position)

    def allegro_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.allegro_joint_cmd = np.array(msg.position)

    def update_states(self):
        """Update the Isaac simulation with the commanded joint positions."""
        if self.iiwa_joint_cmd is None or self.allegro_joint_cmd is None:
            rospy.logwarn(
                f"Waiting: iiwa_joint_cmd: {self.iiwa_joint_cmd}, allegro_joint_cmd: {self.allegro_joint_cmd}"
            )
            self.env.step_no_fabric(
                torch.zeros(
                    (self.env.num_envs, self.env.num_actions),
                    device=self.device,
                    dtype=torch.float,
                ),
                set_dof_pos_targets=False,
                control_freq_inv=None if USE_CONTROL_DT else 1,
            )
        else:
            action = (
                torch.from_numpy(
                    np.concatenate([self.iiwa_joint_cmd, self.allegro_joint_cmd])
                )
                .to(device=self.device, dtype=torch.float)
                .unsqueeze(dim=0)
                .repeat_interleave(self.env.num_envs, dim=0)
            )

            self.env.step_no_fabric(
                action,
                set_dof_pos_targets=True,
                control_freq_inv=None if USE_CONTROL_DT else 1,
            )

        right_robot_dof_pos = self.env.right_robot_dof_pos[0].detach().cpu().numpy()
        right_robot_dof_vel = self.env.right_robot_dof_vel[0].detach().cpu().numpy()

        self.iiwa_joint_q = right_robot_dof_pos[:NUM_ARM_JOINTS]
        self.allegro_joint_q = right_robot_dof_pos[
            NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS
        ]
        self.iiwa_joint_qd = right_robot_dof_vel[:NUM_ARM_JOINTS]
        self.allegro_joint_qd = right_robot_dof_vel[
            NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS
        ]

        self.object_pos_R = self.env.object_pos[0].detach().cpu().numpy()
        self.object_quat_xyzw_R = self.env.object_quat_xyzw[0].detach().cpu().numpy()

    def publish_joint_states(self):
        """Publish the current joint states from Isaac."""
        if (
            self.iiwa_joint_q is None
            or self.allegro_joint_q is None
            or self.iiwa_joint_qd is None
            or self.allegro_joint_qd is None
        ):
            rospy.logwarn(
                f"Can't publish: iiwa_joint_q: {self.iiwa_joint_q}, allegro_joint_q: {self.allegro_joint_q}, iiwa_joint_qd: {self.iiwa_joint_qd}, allegro_joint_qd: {self.allegro_joint_qd}"
            )
            return

        iiwa_msg = JointState()
        iiwa_msg.header.stamp = rospy.Time.now()
        iiwa_msg.name = ["iiwa_joint_" + str(i) for i in range(NUM_ARM_JOINTS)]
        iiwa_msg.position = self.iiwa_joint_q.tolist()
        iiwa_msg.velocity = self.iiwa_joint_qd.tolist()
        self.iiwa_pub.publish(iiwa_msg)

        allegro_msg = JointState()
        allegro_msg.header.stamp = rospy.Time.now()
        allegro_msg.name = ["allegro_joint_" + str(i) for i in range(NUM_HAND_JOINTS)]
        allegro_msg.position = self.allegro_joint_q.tolist()
        allegro_msg.velocity = self.allegro_joint_qd.tolist()
        self.allegro_pub.publish(allegro_msg)

    def publish_pose(self):
        if self.object_pos_R is None or self.object_quat_xyzw_R is None:
            rospy.logwarn(
                f"Can't publish pose: object_pos_R: {self.object_pos_R}, object_quat_xyzw_R: {self.object_quat_xyzw_R}"
            )
            return

        T_R_O = np.eye(4)
        T_R_O[:3, 3] = self.object_pos_R
        T_R_O[:3, :3] = R.from_quat(self.object_quat_xyzw_R).as_matrix()

        # Extract translation and quaternion from the transformation matrix
        T_C_R = np.linalg.inv(T_R_C)

        T_C_O = T_C_R @ T_R_O
        trans = T_C_O[:3, 3]
        quat_xyzw = R.from_matrix(T_C_O[:3, :3]).as_quat()

        # Create Pose message
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = trans
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (
            quat_xyzw
        )

        # Publish the pose message
        self.pose_pub.publish(msg)
        rospy.logdebug("Pose published to /object_pose")

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the states
            self.update_states()

            # Publish the current joint states to ROS
            self.publish_joint_states()
            self.publish_pose()

            # Sleep to maintain the loop rate
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


if __name__ == "__main__":
    try:
        # Create and run the IsaacFakeRobotNode
        node = IsaacFakeRobotNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
