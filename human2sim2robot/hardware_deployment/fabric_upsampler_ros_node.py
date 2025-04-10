#!/usr/bin/env python

import copy
from typing import Optional

import numpy as np
import rospy
from print_utils import get_ros_loop_rate_str
from sensor_msgs.msg import JointState

# Constants
NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16
PUBLISH_RATE = 200  # Publish at 200 Hz
INTERPOLATION_DT = 1.0 / 120.0  # Time to reach the new target (seconds)
MAX_DT_COMMAND_SEC = 0.5  # Maximum time to wait for a new command (seconds)


class FabricUpsamplerNode:
    def __init__(self):
        rospy.init_node("fabric_upsampler_node", anonymous=True)

        # Subscriptions:
        # 1) Fabric state
        self.fabric_sub = rospy.Subscriber(
            "/fabric_state", JointState, self.fabric_state_callback, queue_size=1
        )

        # 2) Current robot states
        self.iiwa_state_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_state_callback, queue_size=1
        )
        self.allegro_state_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states",
            JointState,
            self.allegro_state_callback,
            queue_size=1,
        )

        # Publishers:
        self.iiwa_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=10
        )
        self.allegro_cmd_pub = rospy.Publisher(
            "/allegroHand_0/joint_cmd", JointState, queue_size=10
        )

        # Store the latest states from the robot
        self.current_iiwa_pos: Optional[np.ndarray] = (
            None  # Latest /iiwa/joint_states positions
        )
        self.current_allegro_pos: Optional[np.ndarray] = (
            None  # Latest /allegroHand_0/joint_states positions
        )
        self.current_joint_pos: Optional[np.ndarray] = (
            None  # Combined positions (arm+hand) of length (7+16)
        )

        # Interpolation variables
        self.interp_joint_pos_start: Optional[np.ndarray] = None
        self.interp_joint_pos_end: Optional[np.ndarray] = None
        self.interp_joint_pos_curr: Optional[np.ndarray] = None
        self.latest_fabric_msg_time: Optional[rospy.Time] = None

        # Set up the publish rate
        self.loop_rate = rospy.Rate(PUBLISH_RATE)

        # Wait until we have some initial robot state before proceeding
        rospy.loginfo(
            "FabricUpsamplerNode: Waiting for initial /iiwa/joint_states and /allegroHand_0/joint_states..."
        )
        while not rospy.is_shutdown():
            if self.current_joint_pos is not None:
                rospy.loginfo("FabricUpsamplerNode: Received initial robot states.")
                break
            rospy.sleep(0.1)

    def iiwa_state_callback(self, msg: JointState):
        """Callback for /iiwa/joint_states."""
        if len(msg.position) < NUM_ARM_JOINTS:
            rospy.logwarn(
                "FabricUpsamplerNode: /iiwa/joint_states has fewer positions than expected."
            )
            return
        self.current_iiwa_pos = np.array(msg.position[:NUM_ARM_JOINTS])

        self.update_latest_robot_pos()

    def allegro_state_callback(self, msg: JointState):
        """Callback for /allegroHand_0/joint_states."""
        if len(msg.position) < NUM_HAND_JOINTS:
            rospy.logwarn(
                "FabricUpsamplerNode: /allegroHand_0/joint_states has fewer positions than expected."
            )
            return
        self.current_allegro_pos = np.array(msg.position[:NUM_HAND_JOINTS])

        self.update_latest_robot_pos()

    def update_latest_robot_pos(self):
        """
        If we have both current_iiwa_pos and current_allegro_pos,
        combine them into one array for convenience.
        """
        if self.current_iiwa_pos is not None and self.current_allegro_pos is not None:
            self.current_joint_pos = np.concatenate(
                (self.current_iiwa_pos, self.current_allegro_pos)
            )

    def fabric_state_callback(self, msg: JointState):
        """
        Callback for /fabric_state.
        """
        if len(msg.position) < (NUM_ARM_JOINTS + NUM_HAND_JOINTS):
            rospy.logwarn(
                "FabricUpsamplerNode: /fabric_state has fewer positions than expected."
            )
            return

        # The new "end" position for interpolation
        fabric_positions = np.array(msg.position[: NUM_ARM_JOINTS + NUM_HAND_JOINTS])

        # If we haven't interpolated before, our 'start' is the robot's actual latest position
        if self.interp_joint_pos_curr is None:
            # This also implies we were waiting for a fabric target
            # and want to begin from the robot's current state
            if self.current_joint_pos is None:
                # If we still don't have a robot state, can't do much
                rospy.logwarn(
                    "FabricUpsamplerNode: No robot state available yet; ignoring fabric target."
                )
                return
            self.interp_joint_pos_start = copy.deepcopy(self.current_joint_pos)
            self.interp_joint_pos_curr = copy.deepcopy(self.current_joint_pos)
        else:
            # Otherwise, we shift our start to our current interpolation value
            self.interp_joint_pos_start = copy.deepcopy(self.interp_joint_pos_curr)

        self.interp_joint_pos_end = fabric_positions
        self.latest_fabric_msg_time = rospy.Time.now()

    def run(self):
        """Main run loop of the interpolator."""

        while (
            self.interp_joint_pos_start is None
            or self.interp_joint_pos_end is None
            or self.interp_joint_pos_curr is None
            or self.latest_fabric_msg_time is None
        ):
            rospy.loginfo(
                f"FabricUpsamplerNode: Waiting: self.interp_joint_pos_start = {self.interp_joint_pos_start}, self.interp_joint_pos_end = {self.interp_joint_pos_end}, self.interp_joint_pos_curr = {self.interp_joint_pos_curr}, self.latest_fabric_msg_time = {self.latest_fabric_msg_time}"
            )
            self.loop_rate.sleep()

        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Check how long it has been since we got the last /fabric_state command
            dt = (start_time - self.latest_fabric_msg_time).to_sec()
            if dt > MAX_DT_COMMAND_SEC:
                rospy.logerr(
                    f"FabricUpsamplerNode: No new /fabric_state message for {dt:.2f} seconds. Shutting down."
                )
                return

            # Compute interpolation factor
            alpha = np.clip(dt / INTERPOLATION_DT, 0.0, 1.0)

            self.interp_joint_pos_curr = (
                1.0 - alpha
            ) * self.interp_joint_pos_start + alpha * self.interp_joint_pos_end

            self.publish_commands(self.interp_joint_pos_curr)

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.loop_rate.sleep()
            after_sleep_time = rospy.Time.now()
            rospy.loginfo(
                get_ros_loop_rate_str(
                    start_time=start_time,
                    before_sleep_time=before_sleep_time,
                    after_sleep_time=after_sleep_time,
                    node_name=rospy.get_name(),
                )
            )

    def publish_commands(self, full_pos_array: np.ndarray):
        """
        Publish to /iiwa/joint_cmd (first 7) and /allegroHand_0/joint_cmd (next 16).
        We leave velocity and effort empty, though you could populate if needed.
        """
        now = rospy.Time.now()

        # Arm command
        iiwa_msg = JointState()
        iiwa_msg.header.stamp = now
        iiwa_msg.name = [f"iiwa_joint_{i}" for i in range(NUM_ARM_JOINTS)]
        iiwa_msg.position = full_pos_array[:NUM_ARM_JOINTS].tolist()
        iiwa_msg.velocity = []
        iiwa_msg.effort = []
        self.iiwa_cmd_pub.publish(iiwa_msg)

        # Hand command
        allegro_msg = JointState()
        allegro_msg.header.stamp = now
        allegro_msg.name = [f"allegro_joint_{i}" for i in range(NUM_HAND_JOINTS)]
        allegro_msg.position = full_pos_array[
            NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS
        ].tolist()
        allegro_msg.velocity = []
        allegro_msg.effort = []
        self.allegro_cmd_pub.publish(allegro_msg)


def main():
    try:
        node = FabricUpsamplerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
