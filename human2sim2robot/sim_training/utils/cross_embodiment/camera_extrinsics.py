import numpy as np

# C is camera frame with Z forward and Y down
# From Claire's extrinsics
ZED_CAMERA_T_R_C = np.eye(4)
ZED_CAMERA_T_R_C[:3, :3] = np.array(
    [
        [0.9543812680846684, 0.08746057618774912, -0.2854943830305726],
        [0.29537672607257903, -0.41644924520026877, 0.8598387150313551],
        [-0.043691930876822334, -0.904942359371598, -0.42328517738189414],
    ]
)
ZED_CAMERA_T_R_C[:3, 3] = np.array(
    [0.5947949577333569, -0.9635715691360609, 0.6851893282998003]
)

# C is camera frame with Z forward and Y down
# Realsense extrinsics
REALSENSE_CAMERA_T_R_C = np.eye(4)
REALSENSE_CAMERA_T_R_C[:3, :3] = np.array(
    [
        [-0.3598, 0.052, -0.9316],
        [0.9328, -0.0021, -0.3604],
        [-0.0207, -0.9986, -0.0478],
    ]
)
REALSENSE_CAMERA_T_R_C[:3, 3] = np.array(
    [1.444818179743563, 0.1269503174990328, 0.3810442085428029]
)

# ZED_CAMERA_T_C_Cptcloud
# For zed, point cloud frame is camera frame with X forward and Y left
# https://community.stereolabs.com/t/coordinate-system-of-pointcloud/908/2
ZED_CAMERA_T_C_Cptcloud = np.eye(4)
ZED_CAMERA_T_C_Cptcloud[:3, :3] = np.array(
    [
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0],
    ]
)
ZED_CAMERA_T_R_Cptcloud = ZED_CAMERA_T_R_C @ ZED_CAMERA_T_C_Cptcloud

# REALSENSE_CAMERA_T_C_Cptcloud
# For realsense, it seems that point cloud frame is same as camera frame
REALSENSE_CAMERA_T_R_Cptcloud = REALSENSE_CAMERA_T_R_C
