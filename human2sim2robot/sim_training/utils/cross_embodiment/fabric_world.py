import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_str_to_T(transform_str: str) -> np.ndarray:
    # Go from string like f"0 0 {MAX_HEIGHT} 0 0 0 1"  # x y z qx qy qz qw
    # To 4x4 matrix
    xyz_qxyzw = np.array([float(x) for x in transform_str.split()])
    from scipy.spatial.transform import Rotation as R

    rotation_matrix = R.from_quat(xyz_qxyzw[3:]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = xyz_qxyzw[:3]
    return T


def T_to_transform_str(T: np.ndarray) -> str:
    # Go from 4x4 matrix to string like f"0 0 {MAX_HEIGHT} 0 0 0 1"  # x y z qx qy qz qw
    r = R.from_matrix(T[:3, :3])
    xyz_qxyzw = np.zeros(7)
    xyz_qxyzw[:3] = T[:3, 3]
    xyz_qxyzw[3:] = r.as_quat()
    return " ".join([str(x) for x in xyz_qxyzw])


THICKNESS = 0.05
MAX_HEIGHT = 1

TABLE_X_LEN = 0.7
TABLE_Y_LEN = 1.6

SIDE_TABLE_X_LEN = 0.4
SIDE_TABLE_Y_LEN = TABLE_Y_LEN / 2

world_dict_table_frame = {
    "right_wall": {
        "env_index": "all",
        "type": "box",
        "scaling": f"{2 * TABLE_X_LEN} {THICKNESS} {MAX_HEIGHT}",
        "transform": f"0 {0.5 * TABLE_Y_LEN} {0.5 * MAX_HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
    },
    "left_wall": {
        "env_index": "all",
        "type": "box",
        "scaling": f"{2 * TABLE_X_LEN} {THICKNESS} {MAX_HEIGHT}",
        "transform": f"0 {-0.5 * TABLE_Y_LEN} {0.5 * MAX_HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
    },
    "back_wall": {
        "env_index": "all",
        "type": "box",
        "scaling": f"{THICKNESS} {TABLE_Y_LEN} {MAX_HEIGHT}",
        "transform": f"{-SIDE_TABLE_X_LEN} 0 {0.5 * MAX_HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
    },
    "front_wall": {
        "env_index": "all",
        "type": "box",
        "scaling": f"{THICKNESS} {TABLE_Y_LEN} {MAX_HEIGHT}",
        "transform": f"{TABLE_X_LEN} 0 {0.5 * MAX_HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
    },
    "table": {
        "env_index": "all",
        "type": "box",
        "scaling": f"{TABLE_X_LEN} {TABLE_Y_LEN} {THICKNESS}",
        "transform": f"{0.5 * TABLE_X_LEN} 0 0 0 0 0 1",  # x y z qx qy qz qw
    },
    "side_table": {
        "env_index": "all",
        "type": "box",
        "scaling": f"{SIDE_TABLE_X_LEN} {SIDE_TABLE_Y_LEN} {THICKNESS}",
        "transform": f"{-0.5 * SIDE_TABLE_X_LEN} {0.5 * SIDE_TABLE_Y_LEN} 0 0 0 0 1",  # x y z qx qy qz qw
    },
    "ceiling": {
        "env_index": "all",
        "type": "box",
        "scaling": f"{2 * TABLE_X_LEN} {TABLE_Y_LEN} {THICKNESS}",
        "transform": f"0 0 {MAX_HEIGHT} 0 0 0 1",  # x y z qx qy qz qw
    },
}

SAFETY_BUFFER = -0.02  # ADJUST THE TABLE HEIGHT AWAY FROM THE ACTUAL

T_O_R = np.eye(4)
T_O_R[:3, 3] = [-0.15, -0.2, -0.13 - SAFETY_BUFFER]
T_O_R[:3, :3] = R.from_euler("z", 42, degrees=True).as_matrix()

T_R_O = np.linalg.inv(T_O_R)

world_dict_robot_frame = {
    k: {
        "env_index": v["env_index"],
        "type": v["type"],
        "scaling": v["scaling"],
        "transform": T_to_transform_str(T_R_O @ transform_str_to_T(v["transform"])),
    }
    for k, v in world_dict_table_frame.items()
}
