import numpy as np
from scipy.spatial.transform import Rotation as R

TABLE_LENGTH_X, TABLE_LENGTH_Y, TABLE_LENGTH_Z = (
    0.76,
    1.53,
    0.05,
)  # BRITTLE: Must match table.urdf

# TC is table center
# TE is table end
X_TE_TC = np.eye(4)
X_TE_TC[:3, 3] = np.array(
    [
        TABLE_LENGTH_X / 2,
        0,
        0,
    ]
)

X_TE_R = np.eye(4)
ANGLE_DEG = 42
X_TE_R[:3, :3] = R.from_euler("z", ANGLE_DEG, degrees=True).as_matrix()
X_TE_R[:3, 3] = np.array(
    [
        # WARNING: The z should be -0.135 based on our measurements in the real world
        #          However, we are using the -0.13 value because for some reason -0.135
        #          is causing the objects to be inside of the table (maybe FP issue or camera calibration issue)
        -0.18,
        -0.205,
        -0.13,
    ]
)

X_R_TE = np.linalg.inv(X_TE_R)
X_R_TC = X_R_TE @ X_TE_TC

TABLE_X, TABLE_Y, TABLE_Z = X_R_TC[:3, 3]
TABLE_QX, TABLE_QY, TABLE_QZ, TABLE_QW = R.from_matrix(X_R_TC[:3, :3]).as_quat()
