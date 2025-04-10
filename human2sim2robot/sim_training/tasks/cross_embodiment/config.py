from dataclasses import dataclass, field
from typing import List, Literal, Optional

from omegaconf import MISSING

from human2sim2robot.sim_training.utils.cross_embodiment.curriculum import (
    CurriculumConfig,
    CurriculumUpdate,
)


@dataclass
class LogConfig:
    populateWandbDictEveryNSteps: int = MISSING
    captureVideo: bool = MISSING
    numVideoFrames: int = MISSING
    captureVideoEveryNSteps: int = MISSING
    saveBestModelToWandbEveryNSteps: int = MISSING
    printMetricsEveryNSteps: int = MISSING


@dataclass
class RandomForcesConfig:
    forceProb: float = MISSING
    forceScale: float = MISSING
    torqueProb: float = MISSING
    torqueScale: float = MISSING


@dataclass
class CustomEnvConfig:
    log: LogConfig = field(default_factory=LogConfig)
    randomForces: RandomForcesConfig = field(default_factory=RandomForcesConfig)
    FORCE_REFERENCE_TRAJECTORY_TRACKING: bool = MISSING
    OBJECT_ORIENTATION_MATTERS: bool = MISSING
    RANDOMIZE_GOAL_OBJECT_ORIENTATION_DEG: float = MISSING
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    curriculum_updates: List[CurriculumUpdate] = field(default_factory=list)
    USE_FABRIC_ACTION_SPACE: bool = MISSING
    FABRIC_HAND_ACTION_SPACE: Literal["PCA", "ALL"] = MISSING
    ENABLE_FABRIC_COLLISION_AVOIDANCE: bool = MISSING
    FABRIC_CSPACE_DAMPING: Optional[float] = MISSING
    FABRIC_CSPACE_DAMPING_HAND: Optional[float] = MISSING
    SUCCESS_REGION_RADIUS: float = MISSING
    enableDebugViz: bool = MISSING
    object_friction: float = MISSING
    object_mass_scale: float = MISSING
    object_inertia_scale: float = MISSING
    right_robot_friction: float = MISSING
    table_friction: float = MISSING
    OBSERVED_OBJECT_UNCORR_POS_NOISE: float = MISSING
    OBSERVED_OBJECT_UNCORR_RPY_DEG_NOISE: float = MISSING
    OBSERVED_OBJECT_CORR_POS_NOISE: float = MISSING
    OBSERVED_OBJECT_CORR_RPY_DEG_NOISE: float = MISSING
    OBSERVED_OBJECT_RANDOM_POSE_INJECTION_PROB: float = MISSING

    reset_object_sample_noise_x: float = MISSING
    reset_object_sample_noise_y: float = MISSING
    reset_object_sample_noise_z: float = MISSING
    reset_object_sample_noise_roll_deg: float = MISSING
    reset_object_sample_noise_pitch_deg: float = MISSING
    reset_object_sample_noise_yaw_deg: float = MISSING
    reset_right_robot_sample_noise_arm_deg: float = MISSING
    reset_right_robot_sample_noise_hand_deg: float = MISSING

    action_smoothing_penalty_weight: float = MISSING
    USE_CUROBO: bool = MISSING

    object_urdf_path: str = MISSING
    retargeted_robot_file: str = MISSING
    object_poses_dir: str = MISSING


@dataclass
class EnvConfig:
    numEnvs: int = MISSING
    envSpacing: float = MISSING
    enableCameraSensors: bool = MISSING
    clipObservations: float = MISSING
    clipActions: float = MISSING
    controlFrequencyInv: int = MISSING
    numStates: int = MISSING
    numObservations: int = MISSING
    numActions: int = MISSING
    numAgents: int = MISSING
    maxEpisodeLength: int = MISSING

    custom: CustomEnvConfig = field(default_factory=CustomEnvConfig)
