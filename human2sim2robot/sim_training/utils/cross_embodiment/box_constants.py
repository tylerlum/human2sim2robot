import numpy as np
from scipy.spatial.transform import Rotation as R

from human2sim2robot.sim_training.utils.cross_embodiment.table_constants import (
    TABLE_LENGTH_Y,
    TABLE_LENGTH_Z,
    X_R_TE,
)

BOX_LENGTH_X, BOX_LENGTH_Y, BOX_LENGTH_Z = (
    0.415,
    0.19,
    0.12,
)  # BRITTLE: Must match box.urdf

# FBC is flat box center
# TE is table end on table surface (z)
X_TE_FBC = np.eye(4)
X_TE_FBC[:3, 3] = np.array(
    [
        0.065 + BOX_LENGTH_X / 2,
        TABLE_LENGTH_Y / 2 - BOX_LENGTH_Y / 2 - 0.295,
        TABLE_LENGTH_Z / 2 + BOX_LENGTH_Z / 2,
    ]
)

X_R_FBC = X_R_TE @ X_TE_FBC
FLAT_BOX_X, FLAT_BOX_Y, FLAT_BOX_Z = X_R_FBC[:3, 3]
FLAT_BOX_QX, FLAT_BOX_QY, FLAT_BOX_QZ, FLAT_BOX_QW = R.from_matrix(
    X_R_FBC[:3, :3]
).as_quat()
