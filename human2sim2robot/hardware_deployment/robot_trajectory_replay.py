#!/usr/bin/env python


import numpy as np
import rospy
from sensor_msgs.msg import JointState

from human2sim2robot.sim_training import get_data_dir

IIWA_NAMES = [
    "iiwa_joint_1",
    "iiwa_joint_2",
    "iiwa_joint_3",
    "iiwa_joint_4",
    "iiwa_joint_5",
    "iiwa_joint_6",
    "iiwa_joint_7",
]
ALLEGRO_NAMES = [
    "allegro_joint_0",
    "allegro_joint_1",
    "allegro_joint_2",
    "allegro_joint_3",
    "allegro_joint_4",
    "allegro_joint_5",
    "allegro_joint_6",
    "allegro_joint_7",
    "allegro_joint_8",
    "allegro_joint_9",
    "allegro_joint_10",
    "allegro_joint_11",
    "allegro_joint_12",
    "allegro_joint_13",
    "allegro_joint_14",
    "allegro_joint_15",
]


class RobotTrajectoryPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("trajectory_publisher")

        # Get trajectory filepath from ROS param
        # TASK_NAME = "snackbox_wallslide_easy_bottom"
        # TASK_NAME = "snackbox_pivot_easy"

        # TASK_NAME = "snackbox_pour"
        # TASK_NAME = "snackbox_pivot"
        # TASK_NAME = "snackbox_push"
        # TASK_NAME = "snackbox_lift_wall"
        # TASK_NAME = "snackbox_pushpivot"
        # TASK_NAME = "snackbox_pushpivotmove"
        # TASK_NAME = "plate_push"
        # TASK_NAME = "plate_pivotrack_redo"
        # TASK_NAME = "plate_rack_wall"
        # TASK_NAME = "plate_pushpivotrack"
        TASK_NAME = "wateringcan_pour_redo"
        # TASK_NAME = "pot_pour_sidegrasp_norm"

        self.filepath = (
            get_data_dir()
            / "human_demo_processed_data"
            / TASK_NAME
            / "retargeted_trajectory.npz"
        )
        assert self.filepath.exists(), f"Trajectory file not found: {self.filepath}"

        # Load trajectory data
        data = np.load(self.filepath)
        self.ts = data["interpolated_ts"]
        self.qs = data["interpolated_qs"]

        SLOW_DOWN_FACTOR = 2
        rospy.loginfo(f"Slowing down trajectory by a factor of {SLOW_DOWN_FACTOR}")
        self.ts *= SLOW_DOWN_FACTOR

        # Verify shapes
        assert self.ts.shape[0] == self.qs.shape[0], (
            f"Timestamp and joint trajectory lengths don't match: {self.ts.shape} vs {self.qs.shape}"
        )
        assert self.qs.shape[1] == 23, f"Expected 23 joints, got {self.qs.shape[1]}"

        # Publisher
        self.iiwa_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=10
        )
        self.allegro_cmd_pub = rospy.Publisher(
            "/allegroHand_0/joint_cmd", JointState, queue_size=10
        )

        # ROS rate
        self.rate = rospy.Rate(60)  # 60 Hz

        rospy.loginfo(f"Loaded trajectory with {len(self.ts)} points")
        rospy.loginfo(f"Trajectory duration: {self.ts[-1]:.2f} seconds")

    def run(self):
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            dt_from_start = (current_time - start_time).to_sec()

            # Handle before start and after end cases
            if dt_from_start < self.ts[0]:
                rospy.loginfo_throttle(
                    1.0,
                    f"Waiting to start trajectory, dt_from_start: {dt_from_start:.2f}",
                )
                self.rate.sleep()
                continue

            if dt_from_start > self.ts[-1]:
                rospy.loginfo_throttle(
                    1.0, f"Trajectory completed, dt_from_start: {dt_from_start:.2f}"
                )
                self.rate.sleep()
                continue

            # Find the appropriate segment
            idx = np.searchsorted(self.ts, dt_from_start) - 1
            idx = np.clip(idx, 0, len(self.ts) - 2)  # Ensure valid index

            # Get the current joint positions
            current_q = self.qs[idx]
            assert current_q.shape == (23,), (
                f"Expected 23 joints, got {current_q.shape}"
            )

            # Create and publish iiwa command message
            iiwa_msg = JointState()
            iiwa_msg.header.stamp = current_time
            iiwa_msg.name = IIWA_NAMES
            iiwa_msg.position = current_q.tolist()[:7]
            iiwa_msg.velocity = []
            iiwa_msg.effort = []
            self.iiwa_cmd_pub.publish(iiwa_msg)

            # Create and publish allegro command message
            allegro_msg = JointState()
            allegro_msg.header.stamp = current_time
            allegro_msg.name = ALLEGRO_NAMES
            allegro_msg.position = current_q.tolist()[7:]
            allegro_msg.velocity = []
            allegro_msg.effort = []
            self.allegro_cmd_pub.publish(allegro_msg)

            # Maintain loop rate
            self.rate.sleep()


if __name__ == "__main__":
    try:
        robot_trajectory_publisher = RobotTrajectoryPublisher()
        robot_trajectory_publisher.run()
    except rospy.ROSInterruptException:
        pass
