#!/usr/bin/env python

from typing import Literal

import numpy as np
import rospy
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R

from human2sim2robot.sim_training.utils.cross_embodiment.camera_extrinsics import T_R_C


class FakeObjectPose:
    def __init__(self):
        # Publisher for the object pose
        self.pose_pub = rospy.Publisher("/object_pose", Pose, queue_size=1)
        self.rate = rospy.Rate(60)

    def publish_pose(self):
        MODE: Literal["T_R_O", "T_C_O"] = "T_R_O"

        if MODE == "T_R_O":
            # Set a fixed transformation matrix T (4x4)
            T_R_O = np.eye(4)
            T_R_O[:3, 3] = np.array([0.5735, -0.1633, 0.2038])

            # Publish rate of 60Hz
            # Extract translation and quaternion from the transformation matrix
            T_C_R = np.linalg.inv(T_R_C)

            T_C_O = T_C_R @ T_R_O
        elif MODE == "T_C_O":
            T_C_O = np.array(
                [
                    [
                        -3.762190341949462891e-01,
                        6.942993402481079102e-02,
                        -9.239256381988525391e-01,
                        1.114022210240364075e-01,
                    ],
                    [
                        4.116456806659698486e-01,
                        -8.808401823043823242e-01,
                        -2.338127344846725464e-01,
                        2.898024022579193115e-01,
                    ],
                    [
                        -8.300643563270568848e-01,
                        -4.682947397232055664e-01,
                        3.028083145618438721e-01,
                        8.317363858222961426e-01,
                    ],
                    [
                        0.000000000000000000e00,
                        0.000000000000000000e00,
                        0.000000000000000000e00,
                        1.000000000000000000e00,
                    ],
                ]
            )

        else:
            raise ValueError(f"Invalid MODE: {MODE}")

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
        while not rospy.is_shutdown():
            self.publish_pose()
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("fake_object_pose")
    node = FakeObjectPose()
    rospy.loginfo("Publishing fixed object pose at 60Hz to /object_pose")
    node.run()
