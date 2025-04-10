import numpy as np

from human2sim2robot.sim_training import get_asset_root

# Kuka arm and hand
KUKA_ALLEGRO_ASSET_ROOT = str(get_asset_root() / "kuka_allegro")
KUKA_ALLEGRO_FILENAME = "kuka_allegro.urdf"

NUM_ARM_DOFS = 7
NUM_DOFS_PER_FINGER = 4
NUM_FINGERS = 4
NUM_HAND_DOFS = NUM_DOFS_PER_FINGER * NUM_FINGERS
NUM_HAND_ARM_DOFS = NUM_ARM_DOFS + NUM_HAND_DOFS

ARM_JOINT_NAMES = [
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
]
assert len(ARM_JOINT_NAMES) == NUM_ARM_DOFS

PALM_LINK_NAME = "palm_link"
PALM_X_LINK_NAME = "palm_x"
PALM_Y_LINK_NAME = "palm_y"
PALM_Z_LINK_NAME = "palm_z"
PALM_LINK_NAMES = [PALM_LINK_NAME, PALM_X_LINK_NAME, PALM_Y_LINK_NAME, PALM_Z_LINK_NAME]
ALLEGRO_FINGERTIP_LINK_NAMES = [
    "index_biotac_tip",
    "middle_biotac_tip",
    "ring_biotac_tip",
    "thumb_biotac_tip",
]
assert len(ALLEGRO_FINGERTIP_LINK_NAMES) == NUM_FINGERS

HAND_JOINT_NAMES = [
    "index_joint_0",
    "index_joint_1",
    "index_joint_2",
    "index_joint_3",
    "middle_joint_0",
    "middle_joint_1",
    "middle_joint_2",
    "middle_joint_3",
    "ring_joint_0",
    "ring_joint_1",
    "ring_joint_2",
    "ring_joint_3",
    "thumb_joint_0",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
]
assert len(HAND_JOINT_NAMES) == NUM_HAND_DOFS

INDEX_FINGER_IDX, MIDDLE_FINGER_IDX, RING_FINGER_IDX, THUMB_IDX = 0, 1, 2, 3

# Default config used in IPRL
DEFAULT_KUKA_DOF_POS = np.deg2rad([0, 0, 0, -90, 0, 90, 0]).tolist()

LEFT_KUKA_DOF_POS = [
    # Manually designed from cup lift
    # -0.0095571,
    # 0.87742555,
    # 0.28864127,
    # -2.0917962,
    # -1.434597,
    # 1.8186541,
    # 1.414263,
    # Manually designed from cracker lift
    1.4336165,
    1.8368409,
    -1.4400831,
    -1.9198623,
    -2.7925267,
    -1.1355215,
    -2.8797934,
]
RIGHT_KUKA_DOF_POS = [
    # Manually designed from cup lift
    # -1.0825572,
    # 1.049477,
    # 0.3596481,
    # -1.5662788,
    # 0.6610003,
    # 1.8230815,
    # -0.9897525,
    # Manually designed from cracker lift
    # -1.3408781,
    # 1.2292578,
    # 0.7270317,
    # -1.2897283,
    # 0.42491052,
    # 1.9198623,
    # -0.66980076,
    # Manually designed from cracker lift trajectory
    -1.794322,
    1.4900819,
    1.1861304,
    -1.125223,
    0.03699509,
    1.9198623,
    -0.36774716,
]
TOP_KUKA_DOF_POS = [
    # Manually designed from cup lift
    # -0.8689132,
    # 0.4176688,
    # 0.5549343,
    # -2.0467792,
    # -0.3155458,
    # 0.7586144,
    # -0.12089629,
    # Manually designed from cracker lift
    -0.24951012,
    0.11640371,
    -0.16731983,
    -1.8449856,
    0.02123778,
    1.1811602,
    -0.42378142,
]
RIGHT_DIAGONAL_PUSH_KUKA_DOF_POS = [
    -2.0590215,
    1.0980108,
    0.7432827,
    -1.5465801,
    -0.01284043,
    1.9198623,
    -0.96888727,
]
RIGHT_DIAGONAL_PULL_KUKA_DOF_POS = [
    -0.88342065,
    1.2187221,
    0.40126476,
    -1.0748798,
    0.82238686,
    1.9198623,
    -0.686731,
]
assert len(DEFAULT_KUKA_DOF_POS) == NUM_ARM_DOFS
assert len(LEFT_KUKA_DOF_POS) == NUM_ARM_DOFS
assert len(RIGHT_KUKA_DOF_POS) == NUM_ARM_DOFS
assert len(TOP_KUKA_DOF_POS) == NUM_ARM_DOFS
assert len(RIGHT_DIAGONAL_PUSH_KUKA_DOF_POS) == NUM_ARM_DOFS
assert len(RIGHT_DIAGONAL_PULL_KUKA_DOF_POS) == NUM_ARM_DOFS
DEMO_KUKA_ALLEGRO_DOF_POS = [
    # Latest Cracker
    1.5181026 - 0.1,
    -1.5155725 + 0.05,
    -1.9369793,
    -1.3040557,
    0.19394092,
    2.0393198 - 0.2,
    -0.43677106,
    -0.08274121,
    0.24389888,
    0.25241292,
    -0.05667633,
    -0.04693202,
    0.4236268,
    0.31045106,
    0.18039484,
    0.22096293,
    0.3781656,
    -0.01176546,
    0.8622075,
    0.62304556,
    0.28457075,
    0.3383089,
    0.7622851,
]
assert len(DEMO_KUKA_ALLEGRO_DOF_POS) == NUM_HAND_ARM_DOFS

DEFAULT_ALLEGRO_DOF_POS = [
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
assert len(DEFAULT_ALLEGRO_DOF_POS) == NUM_HAND_DOFS

DEFAULT_DOF_POS = DEFAULT_KUKA_DOF_POS + DEFAULT_ALLEGRO_DOF_POS
assert len(DEFAULT_DOF_POS) == NUM_HAND_ARM_DOFS

# Stiffness from iiwa_hardware/iiwa_java
KUKA_STIFFNESS = [600, 600, 500, 400, 200, 200, 200]

# Damping from iiwa_hardware/iiwa_java (Lehr's damping = 0.3, not sure how to convert...)
KUKA_DAMPING = [70, 70, 70, 70, 40, 30, 30]

# Efforts from DextrAH-G
KUKA_EFFORT = [176, 176, 110, 110, 110, 40, 40]

ALLEGRO_EFFORT = 0.5
ALLEGRO_STIFFNESS = 0.5
ALLEGRO_DAMPING = 0.1

KUKA_ARMATURE = 0
ALLEGRO_ARMATURE = 0

DOF_FRICTION = -1.0
