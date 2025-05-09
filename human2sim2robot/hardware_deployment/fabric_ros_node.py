#!/usr/bin/env python

from typing import Literal

import numpy as np
import rospy
import torch
from fabrics_sim.fabrics.kuka_allegro_pose_allhand_fabric import (
    KukaAllegroPoseAllHandFabric,
)

# Import from the fabrics_sim package
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import capture_fabric, initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from human2sim2robot.hardware_deployment.utils.print_utils import get_ros_loop_rate_str
from human2sim2robot.sim_training.utils.cross_embodiment.fabric_world import (
    world_dict_robot_frame,
)

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16
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

FABRIC_MODE: Literal["PCA", "ALL"] = "PCA"


class IiwaAllegroFabricPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("fabric_ros_node")

        # ROS msgs
        self.iiwa_joint_state = None
        self.allegro_joint_state = None
        self.palm_target = None
        self.hand_target = None
        self.received_iiwa_joint_state_time = None
        self.received_allegro_joint_state_time = None

        # Publisher and subscriber
        self.iiwa_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=10
        )
        self.iiwa_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_cmd_pub = rospy.Publisher(
            "/allegroHand_0/joint_cmd", JointState, queue_size=10
        )
        self.allegro_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback
        )
        self.palm_target_sub = rospy.Subscriber(
            "/palm_target", Float64MultiArray, self.palm_target_callback
        )
        self.hand_target_sub = rospy.Subscriber(
            "/hand_target", Float64MultiArray, self.hand_target_callback
        )
        self.fabric_pub = rospy.Publisher("/fabric_state", JointState, queue_size=10)

        # ROS rate
        self.rate = rospy.Rate(60)  # 60 Hz
        self.device = "cuda:0"

        # Time step
        self.control_dt = 1.0 / 60.0

        # Setup the Fabric
        self._setup_fabric_action_space()

        # When only testing the arm, set this to False to ignore the Allegro hand
        self.WAIT_FOR_ALLEGRO_STATE = True
        if not self.WAIT_FOR_ALLEGRO_STATE:
            rospy.logwarn("NOT WAITING FOR ALLEGRO STATE")
            self.allegro_joint_state = np.zeros(NUM_HAND_JOINTS)

        # Wait for the initial joint states
        while not rospy.is_shutdown():
            if (
                self.iiwa_joint_state is not None
                and self.allegro_joint_state is not None
            ):
                rospy.loginfo("Got iiwa and allegro joint states")
                break

            rospy.loginfo(
                f"Waiting: iiwa_joint_state: {self.iiwa_joint_state}, allegro_joint_state: {self.allegro_joint_state}, palm_target: {self.palm_target}, hand_target: {self.hand_target}"
            )
            rospy.sleep(0.1)

        # VERY IMPORTANT: Set the initial fabric_q to match the initial joint states
        assert self.iiwa_joint_state is not None
        assert self.allegro_joint_state is not None
        self.fabric_q.copy_(
            torch.from_numpy(
                np.concatenate(
                    [self.iiwa_joint_state, self.allegro_joint_state], axis=0
                ),
            )
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

    def iiwa_joint_state_callback(self, msg: JointState) -> None:
        self.iiwa_joint_state = np.array(msg.position)
        self.received_iiwa_joint_state_time = rospy.Time.now()

    def allegro_joint_state_callback(self, msg: JointState) -> None:
        self.allegro_joint_state = np.array(msg.position)
        self.received_allegro_joint_state_time = rospy.Time.now()

    def palm_target_callback(self, msg: Float64MultiArray) -> None:
        self.palm_target = np.array(msg.data)

    def hand_target_callback(self, msg: Float64MultiArray) -> None:
        self.hand_target = np.array(msg.data)

    def _setup_fabric_action_space(self):
        self.num_envs = 1  # Single environment for this example

        # Initialize warp
        initialize_warp(warp_cache_name="")

        # Set up the world model
        self.fabric_world_dict = world_dict_robot_frame
        # self.fabric_world_dict = None
        self.fabric_world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=20,
            device=self.device,
            world_dict=self.fabric_world_dict,
        )
        self.fabric_object_ids, self.fabric_object_indicator = (
            self.fabric_world_model.get_object_ids()
        )

        # Create Kuka-Allegro Pose Fabric
        if FABRIC_MODE == "PCA":
            fabric_class = KukaAllegroPoseFabric
        elif FABRIC_MODE == "ALL":
            fabric_class = KukaAllegroPoseAllHandFabric
        else:
            raise ValueError(f"Invalid FABRIC_MODE = {FABRIC_MODE}")
        self.fabric = fabric_class(
            batch_size=self.num_envs,
            device=self.device,
            timestep=self.control_dt,
            graph_capturable=True,
        )
        self.fabric_integrator = DisplacementIntegrator(self.fabric)

        # Initialize random targets for palm and hand
        if FABRIC_MODE == "PCA":
            num_hand_target = 5
        elif FABRIC_MODE == "ALL":
            num_hand_target = 16
        else:
            raise ValueError(f"Invalid FABRIC_MODE = {FABRIC_MODE}")
        self.fabric_hand_target = torch.zeros(
            self.num_envs, num_hand_target, device="cuda", dtype=torch.float
        )
        self.fabric_palm_target = torch.zeros(
            self.num_envs, 6, device="cuda", dtype=torch.float
        )

        # Joint states
        self.fabric_q = torch.zeros(
            self.num_envs, self.fabric.num_joints, device=self.device
        )
        self.fabric_qd = torch.zeros_like(self.fabric_q)
        self.fabric_qdd = torch.zeros_like(self.fabric_q)

        # Capture the fabric graph for CUDA optimization
        fabric_inputs = [
            self.fabric_hand_target,
            self.fabric_palm_target,
            "euler_zyx",
            self.fabric_q.detach(),
            self.fabric_qd.detach(),
            self.fabric_object_ids,
            self.fabric_object_indicator,
        ]
        (
            self.fabric_cuda_graph,
            self.fabric_q_new,
            self.fabric_qd_new,
            self.fabric_qdd_new,
        ) = capture_fabric(
            fabric=self.fabric,
            q=self.fabric_q,
            qd=self.fabric_qd,
            qdd=self.fabric_qdd,
            timestep=self.control_dt,
            fabric_integrator=self.fabric_integrator,
            inputs=fabric_inputs,
            device=self.device,
        )

    def run(self):
        # Must have initial joint states before starting
        # Do not need to have targets yet
        assert self.iiwa_joint_state is not None
        assert self.allegro_joint_state is not None
        assert self.received_iiwa_joint_state_time is not None
        assert self.received_allegro_joint_state_time is not None

        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            if self.palm_target is not None and self.hand_target is not None:
                # Update fabric targets for palm and hand
                self.fabric_palm_target.copy_(
                    torch.from_numpy(self.palm_target)
                    .unsqueeze(0)
                    .float()
                    .to(self.device)
                )
                self.fabric_hand_target.copy_(
                    torch.from_numpy(self.hand_target)
                    .unsqueeze(0)
                    .float()
                    .to(self.device)
                )

                # Step the fabric using the captured CUDA graph
                self.fabric_cuda_graph.replay()
                self.fabric_q.copy_(self.fabric_q_new)
                self.fabric_qd.copy_(self.fabric_qd_new)
                self.fabric_qdd.copy_(self.fabric_qdd_new)
            else:
                rospy.logwarn(
                    f"Waiting for targets... palm_target: {self.palm_target}, hand_target: {self.hand_target}"
                )

            # Stop if the joint states are not received for a long time
            MAX_DT_JOINT_STATE_SEC = 0.5
            time_since_iiwa_joint_state = (
                rospy.Time.now() - self.received_iiwa_joint_state_time
            ).to_sec()
            time_since_allegro_joint_state = (
                rospy.Time.now() - self.received_allegro_joint_state_time
            ).to_sec()
            if time_since_iiwa_joint_state > MAX_DT_JOINT_STATE_SEC:
                rospy.logerr(
                    f"Did not receive iiwa joint states for {time_since_iiwa_joint_state} seconds"
                )
                return
            if time_since_allegro_joint_state > MAX_DT_JOINT_STATE_SEC:
                rospy.logerr(
                    f"Did not receive allegro joint states for {time_since_allegro_joint_state} seconds"
                )
                return

            # Still publish the joint states even if the targets are not received
            fabric_msg = JointState()
            fabric_msg.header.stamp = rospy.Time.now()

            # Set joint values
            fabric_msg.name = IIWA_NAMES + ALLEGRO_NAMES
            fabric_msg.position = self.fabric_q.cpu().numpy()[0].tolist()
            fabric_msg.velocity = self.fabric_qd.cpu().numpy()[0].tolist()
            fabric_msg.effort = self.fabric_qdd.cpu().numpy()[0].tolist()

            # Publish the joint states
            self.fabric_pub.publish(fabric_msg)

            PUBLISH_CMD = False
            if PUBLISH_CMD:
                iiwa_msg = JointState()
                iiwa_msg.header.stamp = fabric_msg.header.stamp
                iiwa_msg.name = IIWA_NAMES
                iiwa_msg.position = fabric_msg.position[:NUM_ARM_JOINTS]
                iiwa_msg.velocity = fabric_msg.velocity[:NUM_ARM_JOINTS]
                iiwa_msg.effort = []  # No effort information
                self.iiwa_cmd_pub.publish(iiwa_msg)

                allegro_msg = JointState()
                allegro_msg.header.stamp = fabric_msg.header.stamp
                allegro_msg.name = ALLEGRO_NAMES
                allegro_msg.position = fabric_msg.position[
                    NUM_ARM_JOINTS : NUM_ARM_JOINTS + NUM_HAND_JOINTS
                ]
                allegro_msg.velocity = []  # Leave velocity as empty
                allegro_msg.effort = []
                self.allegro_cmd_pub.publish(allegro_msg)

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
        iiwa_fabric_publisher = IiwaAllegroFabricPublisher()
        iiwa_fabric_publisher.run()
    except rospy.ROSInterruptException:
        pass
