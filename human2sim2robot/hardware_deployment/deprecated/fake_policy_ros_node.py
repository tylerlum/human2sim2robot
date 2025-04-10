#!/usr/bin/env python

import numpy as np
import rospy
import torch
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout


class FakePolicyNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("fake_policy_node")

        # Publisher for palm and hand targets
        self.palm_target_pub = rospy.Publisher(
            "/palm_target", Float64MultiArray, queue_size=10
        )
        self.hand_target_pub = rospy.Publisher(
            "/hand_target", Float64MultiArray, queue_size=10
        )

        # ROS rate (60Hz)
        self.rate_hz = 60
        self.rate = rospy.Rate(self.rate_hz)

        # Number of environments (batch size)
        self.num_envs = 1

        # Device setup for torch tensors
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Step counter for switching targets every 120 steps (i.e., every 2 seconds)
        self.loop_count = 0
        self.SWITCH_TARGET_FREQ_SECONDS = 5
        self.SWITCH_TARGET_FREQ = self.SWITCH_TARGET_FREQ_SECONDS * self.rate_hz

    def sample_palm_target(self) -> torch.Tensor:
        # x forward, y left, z up
        # roll pitch yaw
        palm_target = torch.zeros(
            self.num_envs, 6, device=self.device, dtype=torch.float
        )
        palm_target[:, 0] = (
            torch.FloatTensor(self.num_envs)
            .uniform_(0.1, 0.6)
            .to(device=palm_target.device)
        )
        palm_target[:, 1] = (
            torch.FloatTensor(self.num_envs)
            .uniform_(-0.4, 0.0)
            .to(device=palm_target.device)
        )
        palm_target[:, 2] = (
            torch.FloatTensor(self.num_envs)
            .uniform_(0.1, 0.3)
            .to(device=palm_target.device)
        )
        DEFAULT_EULER_Z = np.deg2rad(0)
        DEFAULT_EULER_Y = np.deg2rad(90)
        DEFAULT_EULER_X = np.deg2rad(0)

        DELTA = np.deg2rad(25)
        palm_target[:, 3] = (
            torch.FloatTensor(self.num_envs)
            .uniform_(DEFAULT_EULER_Z - DELTA, DEFAULT_EULER_Z + DELTA)
            .to(device=palm_target.device)
        )
        palm_target[:, 4] = (
            torch.FloatTensor(self.num_envs)
            .uniform_(DEFAULT_EULER_Y - DELTA, DEFAULT_EULER_Y + DELTA)
            .to(device=palm_target.device)
        )
        palm_target[:, 5] = (
            torch.FloatTensor(self.num_envs)
            .uniform_(DEFAULT_EULER_X - DELTA, DEFAULT_EULER_X + DELTA)
            .to(device=palm_target.device)
        )
        return palm_target

    def sample_hand_target(self) -> torch.Tensor:
        # Joint limits for the hand and palm targets
        fabric_hand_mins = torch.tensor(
            [0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=self.device
        )
        fabric_hand_maxs = torch.tensor(
            [3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=self.device
        )

        # Initialize random targets for palm and hand
        fabric_hand_target = (fabric_hand_maxs - fabric_hand_mins) * torch.rand(
            self.num_envs, fabric_hand_maxs.numel(), device=self.device
        ) + fabric_hand_mins
        return fabric_hand_target

    def run(self):
        while not rospy.is_shutdown():
            # Update palm and hand targets every 120 steps (2 seconds)
            if self.loop_count % self.SWITCH_TARGET_FREQ == 0:
                palm_target = self.sample_palm_target()
                hand_target = self.sample_hand_target()

                # Convert palm_target to Float64MultiArray and publish
                palm_msg = Float64MultiArray()
                palm_msg.layout = MultiArrayLayout(
                    dim=[MultiArrayDimension(label="palm_target", size=6, stride=6)],
                    data_offset=0,
                )
                palm_msg.data = (
                    palm_target.cpu().numpy().flatten().tolist()
                )  # Convert to list
                self.palm_target_pub.publish(palm_msg)

                # Convert hand_target to Float64MultiArray and publish
                hand_msg = Float64MultiArray()
                hand_msg.layout = MultiArrayLayout(
                    dim=[MultiArrayDimension(label="hand_target", size=5, stride=5)],
                    data_offset=0,
                )
                hand_msg.data = (
                    hand_target.cpu().numpy().flatten().tolist()
                )  # Convert to list
                self.hand_target_pub.publish(hand_msg)

                rospy.loginfo(
                    f"Published new palm and hand targets at step {self.loop_count}"
                )

            # Sleep to maintain 60Hz loop rate
            self.rate.sleep()
            self.loop_count += 1


if __name__ == "__main__":
    try:
        fake_policy_node = FakePolicyNode()
        fake_policy_node.run()
    except rospy.ROSInterruptException:
        pass
