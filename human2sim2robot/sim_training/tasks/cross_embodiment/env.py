# gymtorch must be imported before torch
from isaacgym import gymapi, gymtorch, gymutil  # isort:skip

import functools
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import wandb
import yaml
from cached_property_with_invalidation import cached_property_with_invalidation
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from human2sim2robot.ppo.utils.dict_to_dataclass import dict_to_dataclass
from human2sim2robot.sim_training import (
    datetime_str,
    get_asset_root,
    get_repo_root_dir,
    get_sim_training_dir,
)
from human2sim2robot.sim_training.tasks.base.vec_task import VecTask
from human2sim2robot.sim_training.tasks.cross_embodiment.config import (
    CustomEnvConfig,
    EnvConfig,
    LogConfig,
    RandomForcesConfig,
)
from human2sim2robot.sim_training.utils.cross_embodiment.box_constants import (
    BOX_LENGTH_X,
    BOX_LENGTH_Y,
    BOX_LENGTH_Z,
    FLAT_BOX_QW,
    FLAT_BOX_QX,
    FLAT_BOX_QY,
    FLAT_BOX_QZ,
    FLAT_BOX_X,
    FLAT_BOX_Y,
    FLAT_BOX_Z,
    X_R_FBC,
)
from human2sim2robot.sim_training.utils.cross_embodiment.camera_extrinsics import (
    REALSENSE_CAMERA_T_R_C as REALSENSE_CAMERA_T_R_C_np,
)
from human2sim2robot.sim_training.utils.cross_embodiment.camera_extrinsics import (
    ZED_CAMERA_T_R_C as ZED_CAMERA_T_R_C_np,
)
from human2sim2robot.sim_training.utils.cross_embodiment.constants import (
    BLUE,
    CYAN,
    DEBUG_NUM_LATS,
    DEBUG_NUM_LONS,
    DEBUG_SPHERE_RADIUS,
    END_ANG_VEL_IDX,
    END_POS_IDX,
    END_QUAT_IDX,
    END_VEL_IDX,
    GREEN,
    MAGENTA,
    NUM_QUAT,
    NUM_RGBA,
    NUM_STATES,
    NUM_XYZ,
    RED,
    START_ANG_VEL_IDX,
    START_POS_IDX,
    START_QUAT_IDX,
    START_VEL_IDX,
    WHITE,
)
from human2sim2robot.sim_training.utils.cross_embodiment.kuka_allegro_constants import (
    ALLEGRO_ARMATURE,
    ALLEGRO_DAMPING,
    ALLEGRO_EFFORT,
    ALLEGRO_FINGERTIP_LINK_NAMES,
    ALLEGRO_STIFFNESS,
    DOF_FRICTION,
    INDEX_FINGER_IDX,
    KUKA_ALLEGRO_ASSET_ROOT,
    KUKA_ALLEGRO_FILENAME,
    KUKA_ARMATURE,
    KUKA_DAMPING,
    KUKA_EFFORT,
    KUKA_STIFFNESS,
    NUM_FINGERS,
    PALM_LINK_NAME,
    PALM_LINK_NAMES,
    PALM_X_LINK_NAME,
    PALM_Y_LINK_NAME,
    PALM_Z_LINK_NAME,
)
from human2sim2robot.sim_training.utils.cross_embodiment.kuka_allegro_constants import (
    NUM_ARM_DOFS as KUKA_ALLEGRO_NUM_ARM_DOFS,
)
from human2sim2robot.sim_training.utils.cross_embodiment.kuka_allegro_constants import (
    NUM_HAND_ARM_DOFS as KUKA_ALLEGRO_NUM_DOFS,
)
from human2sim2robot.sim_training.utils.cross_embodiment.kuka_allegro_constants import (
    NUM_HAND_DOFS as KUKA_ALLEGRO_NUM_HAND_DOFS,
)
from human2sim2robot.sim_training.utils.cross_embodiment.object_constants import (
    NUM_OBJECT_KEYPOINTS,
    OBJECT_KEYPOINT_OFFSETS,
    OBJECT_KEYPOINT_OFFSETS_ROT_INVARIANT,
    OBJECT_KEYPOINT_OFFSETS_YAW_INVARIANT,
    OBJECT_NUM_RIGID_BODIES,
)
from human2sim2robot.sim_training.utils.cross_embodiment.record_types import (
    Frame,
    Trajectory,
)
from human2sim2robot.sim_training.utils.cross_embodiment.table_constants import (
    TABLE_LENGTH_X,
    TABLE_LENGTH_Y,
    TABLE_LENGTH_Z,
    TABLE_QW,
    TABLE_QX,
    TABLE_QY,
    TABLE_QZ,
    TABLE_X,
    TABLE_Y,
    TABLE_Z,
)
from human2sim2robot.sim_training.utils.cross_embodiment.utils import (
    AverageMeter,
    add_rpy_noise_to_quat_xyzw,
    assert_equals,
    clamp_magnitude,
    clip_T_list,
    compute_keypoint_positions,
    read_in_T_list,
    rescale,
    wandb_started,
)
from human2sim2robot.sim_training.utils.reformat import omegaconf_to_dict
from human2sim2robot.sim_training.utils.torch_jit_utils import (
    matrix_to_quat_wxyz,
    matrix_to_quat_xyzw,
    quat_xyzw_from_euler_xyz,
    quat_xyzw_to_matrix,
)
from human2sim2robot.sim_training.utils.torch_utils import (
    quat_mul,
    scale,
    to_torch,
    torch_rand_float,
)

CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME = "num_steps_taken"  # BRITTLE: Must match variable name in the environment class that is incremented for each env step

# Trajectory tracking
REFERENCE_MOTION_DT = 1.0 / 30

# Trajectory warping
REFERENCE_MIN_SPEED_FACTOR, REFERENCE_MAX_SPEED_FACTOR = 0.5, 1.0
REFERENCE_MOTION_OFFSET_X = 0.02
REFERENCE_MOTION_OFFSET_Y = 0.02
REFERENCE_MOTION_OFFSET_YAW_DEG = 1

# Early reset based on object-goal distance
EARLY_RESET_OBJECT_GOAL_DISTANCE_THRESHOLD = 100

# Fingertips-object distance used for early reset and reward
FINGERTIPS_OBJECT_CLOSE_THRESHOLD = 0.3

# Early reset based on fingertips-object distance
EARLY_RESET_BASED_ON_FINGERTIPS_OBJECT_DISTANCE = True

# Stop reference motion based on object-goal distance
STOP_REFERENCE_MOTION_BASED_ON_OBJECT_GOAL_DISTANCE = True
STOP_REFERENCE_MOTION_OBJECT_GOAL_DISTANCE_THRESHOLD = 0.2

# Save trajectory to file for blender visualization
SAVE_BLENDER_TRAJECTORY = False

USE_REAL_TABLE_MESH = False  # Use real table mesh (or a simple box)

INCLUDE_METAL_CYLINDER = False
INCLUDE_LARGE_SAUCEPAN = False

USE_STATE_AS_OBSERVATION = False

REPLAY_OPEN_LOOP_TRAJECTORY = (
    False  # Read in a pre-recorded trajectory from human demonstrations
)


def compute_num_actions(
    USE_FABRIC_ACTION_SPACE: bool, FABRIC_HAND_ACTION_SPACE: Literal["PCA", "ALL"]
) -> int:
    # Make sure this aligns with pre_physics_step
    if USE_FABRIC_ACTION_SPACE:
        NUM_PALM_ACTIONS = 6
        if FABRIC_HAND_ACTION_SPACE == "PCA":
            NUM_HAND_ACTIONS = 5
        elif FABRIC_HAND_ACTION_SPACE == "ALL":
            NUM_HAND_ACTIONS = KUKA_ALLEGRO_NUM_HAND_DOFS
        else:
            raise ValueError(
                f"Invalid fabric_hand_action_space: {FABRIC_HAND_ACTION_SPACE}"
            )

        return NUM_PALM_ACTIONS + NUM_HAND_ACTIONS
    else:
        return KUKA_ALLEGRO_NUM_DOFS


def compute_num_observations(USE_FABRIC_ACTION_SPACE: bool) -> int:
    # Make sure this aligns with compute_observations
    num_q = KUKA_ALLEGRO_NUM_DOFS
    num_qd = KUKA_ALLEGRO_NUM_DOFS
    num_fingertip_positions = NUM_FINGERS * NUM_XYZ
    num_palm_pos = NUM_XYZ
    num_palm_x_pos = NUM_XYZ
    num_palm_y_pos = NUM_XYZ
    num_palm_z_pos = NUM_XYZ
    num_object_pos = NUM_XYZ
    num_object_quat = NUM_QUAT
    num_goal_object_pos = NUM_XYZ
    num_goal_object_quat = NUM_QUAT

    num_prev_object_pos = NUM_XYZ
    num_prev_object_quat = NUM_QUAT
    num_prev_prev_object_pos = NUM_XYZ
    num_prev_prev_object_quat = NUM_QUAT

    num_fabric_q = KUKA_ALLEGRO_NUM_DOFS
    num_fabric_qd = KUKA_ALLEGRO_NUM_DOFS

    return (
        num_q
        + num_qd
        + num_fingertip_positions
        + num_palm_pos
        + num_palm_x_pos
        + num_palm_y_pos
        + num_palm_z_pos
        + num_object_pos
        + num_object_quat
        + num_goal_object_pos
        + num_goal_object_quat
        + num_prev_object_pos
        + num_prev_object_quat
        + num_prev_prev_object_pos
        + num_prev_prev_object_quat
        + (num_fabric_q + num_fabric_qd) * USE_FABRIC_ACTION_SPACE
    )


def compute_num_states(USE_FABRIC_ACTION_SPACE: bool, INCLUDE_DISH_RACK: bool) -> int:
    # Make sure this aligns with compute_states
    num_obs = compute_num_observations(USE_FABRIC_ACTION_SPACE)

    num_object_keypoint_positions = NUM_OBJECT_KEYPOINTS * NUM_XYZ
    num_goal_object_keypoint_positions = NUM_OBJECT_KEYPOINTS * NUM_XYZ
    num_object_vel = NUM_XYZ
    num_object_angvel = NUM_XYZ
    num_t = 1

    num_dof_force = KUKA_ALLEGRO_NUM_DOFS

    NUM_RIGID_BODIES = 28  # BRITTLE: Depends on the number of objects
    if INCLUDE_METAL_CYLINDER:
        NUM_RIGID_BODIES += 1
    if INCLUDE_LARGE_SAUCEPAN:
        NUM_RIGID_BODIES += 1
    if INCLUDE_DISH_RACK:
        NUM_RIGID_BODIES += 1

    num_contact_forces = NUM_RIGID_BODIES * NUM_XYZ

    num_force_sensors = NUM_RIGID_BODIES * (NUM_XYZ + NUM_XYZ)

    return (
        num_obs
        + num_object_keypoint_positions
        + num_goal_object_keypoint_positions
        + num_object_vel
        + num_object_angvel
        + num_t
        + num_dof_force
        + num_contact_forces
        + num_force_sensors
    )


def get_object_mesh(object_urdf_path: Path) -> trimesh.Trimesh:
    assert object_urdf_path.exists(), object_urdf_path
    urdf = URDF.load(str(object_urdf_path))

    mesh_path_and_scale_list = []
    for link in urdf.links:
        if len(link.collisions) == 0:
            continue

        for i, collision_link in enumerate(link.collisions):
            mesh_path = object_urdf_path.parent / collision_link.geometry.mesh.filename
            assert mesh_path.exists(), mesh_path

            mesh_scale = (
                np.array([1, 1, 1])
                if collision_link.geometry.mesh.scale is None
                else np.array(collision_link.geometry.mesh.scale)
            )
            mesh_path_and_scale_list.append((mesh_path, mesh_scale))

    # Assume urdf has only 1 link with only 1 collision mesh
    assert len(mesh_path_and_scale_list) == 1, (
        f"{mesh_path_and_scale_list} has len {len(mesh_path_and_scale_list)}"
    )

    mesh_path, mesh_scale = mesh_path_and_scale_list[0]
    mesh = trimesh.load_mesh(str(mesh_path))
    mesh.apply_scale(mesh_scale)

    return mesh


class CrossEmbodiment(VecTask):
    """From human hand to robot hand with RL

    Action Space (pre_physics_step):

    Observation Space (compute_observations):

    Reward Function (compute_reward_jit):

    Initial State Distribution (reset_idx):

    Reset Condition (compute_reset_jit):
    """

    ##### INITIALIZATION START #####
    def __init__(
        self,
        cfg: dict,
        rl_device: Union[str, torch.device],
        sim_device: Union[str, torch.device],
        graphics_device_id: int,
        headless: bool,
        virtual_screen_capture: bool,
        force_render: bool,
    ) -> None:
        # Store cfg and read parameters
        self.cfg = OmegaConf.create(cfg)  # Given dict, needs OmegaConf DictConfig
        self.validate_cfg()

        # Do not use fabric if using residual policy (direct PD target deltas)
        # or if doing open-loop trajectory replay
        # Important to properly do env stepping
        if REPLAY_OPEN_LOOP_TRAJECTORY:
            self.custom_env_cfg.USE_FABRIC_ACTION_SPACE = False

        self.num_steps_taken = 0
        self.start_time = time.time()

        self._subscribe_to_keyboard_events()
        self._update_num_observations_and_actions_if_needed()  # Must be before super().__init__

        # Load retargeted robot
        self._load_retargeted_robot()  # Must be before super().__init__

        # This calls create_sim
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # self.index_to_view = int(
        #     0.1 * self.num_envs
        # )  # 0 at corner, may want to view in middle
        self.index_to_view = 0

        self._initialize_state_tensors()
        self._sanity_checks()

        self._setup_reward_weights()

        from human2sim2robot.sim_training.utils.cross_embodiment.fabric_world import (
            world_dict_robot_frame,
        )

        if self.custom_env_cfg.ENABLE_FABRIC_COLLISION_AVOIDANCE:
            self.fabric_world_dict = world_dict_robot_frame.copy()
        else:
            self.fabric_world_dict = {}

        # TODO: Use the actual scene mesh for fabric world
        # self.fabric_world_dict = {
        #     "robot_scene": {
        #         "env_index": "all",
        #         "type": "robot_scene",
        #         "scaling": "1 1 1",
        #         "transform": "0.171316 -0.575692 0.363355 -0.056238 0.703445 0.706355 -0.055363",  #  x y z qx qy qz qw
        #     }
        # }

        # Set up curobo
        if self.custom_env_cfg.USE_CUROBO:
            self._setup_curobo()

        if REPLAY_OPEN_LOOP_TRAJECTORY:
            OPEN_LOOP_TRAJECTORY_PATH = (
                get_repo_root_dir() / self.custom_env_cfg.retargeted_robot_file
            )
            assert OPEN_LOOP_TRAJECTORY_PATH.exists(), OPEN_LOOP_TRAJECTORY_PATH
            open_loop_trajectory_dict = np.load(OPEN_LOOP_TRAJECTORY_PATH)
            self.open_loop_qs = (
                torch.from_numpy(open_loop_trajectory_dict["interpolated_qs"])
                .to(self.device)
                .float()
            )
            self.open_loop_ts = (
                torch.from_numpy(open_loop_trajectory_dict["interpolated_ts"])
                .to(self.device)
                .float()
            )
            SLOW_DOWN_FACTOR = 1
            self.open_loop_ts *= SLOW_DOWN_FACTOR
            N_TIMESTEPS = self.open_loop_qs.shape[0]
            assert_equals(self.open_loop_qs.shape, (N_TIMESTEPS, KUKA_ALLEGRO_NUM_DOFS))
            assert_equals(self.open_loop_ts.shape, (N_TIMESTEPS,))

        # Metrics
        self.reward_metric = AverageMeter().to(self.device)
        self.individual_reward_metrics = {
            reward_name: AverageMeter().to(self.device)
            for reward_name in self.reward_names
        }
        self.individual_weighted_reward_metrics = {
            reward_name: AverageMeter().to(self.device)
            for reward_name in self.reward_names
        }
        self.steps_metric = AverageMeter().to(self.device)
        self.largest_this_episode_num_consecutive_successes_metric = AverageMeter().to(
            self.device
        )
        self.smallest_this_episode_object_goal_distance_metric = AverageMeter().to(
            self.device
        )
        self.has_enough_consecutive_successes_to_end_episode_metric = AverageMeter().to(
            self.device
        )

        # Must run first reset once to init things
        self.reset_all_idxs()

        # Need refresh to get correct table pos to set camera
        self._refresh_state_tensors()

        # Set camera to view in front of the right robot
        right_robot_pos_to_view = self.right_robot_pos[self.index_to_view].cpu().numpy()
        cam_target = gymapi.Vec3(
            right_robot_pos_to_view[0] + 0.5,
            right_robot_pos_to_view[1],
            right_robot_pos_to_view[2] + TABLE_LENGTH_Z / 2 + 0.5,
        )
        cam_pos = cam_target + gymapi.Vec3(1.5, 0.0, 0.0)
        if self.viewer is not None:
            self.gym.viewer_camera_look_at(
                self.viewer,
                self.envs[self.index_to_view],
                cam_pos,
                cam_target,
            )

        # Init camera for wandb logging
        self._initialize_camera_sensor(cam_pos=cam_pos, cam_target=cam_target)

        self._save_config_file_to_wandb()

        # Log
        self.wandb_dict = {}

        from fabrics_sim.utils.utils import initialize_warp

        warp_cache_name = f"{self.device}"
        initialize_warp(warp_cache_name=warp_cache_name)
        if self.custom_env_cfg.USE_FABRIC_ACTION_SPACE:
            self._setup_fabric_action_space()
        self._setup_taskmap()

        if np.max(np.abs(self.object_mesh.centroid)) > 0.02:
            # We require that the object origin is at or near its centroid
            # Use its longest bound
            # We don't want flipping the object over to be enough to count as lifted
            raise ValueError(
                f"Object mesh centroid is {self.object_mesh.centroid}, which is too far from the origin"
            )

        # Set up curriculum
        self._setup_curriculum()

    def _setup_fabric_action_space(self) -> None:
        # Hide imports so that the code still runs without fabrics if unused
        from fabrics_sim.fabrics.kuka_allegro_pose_allhand_fabric import (
            KukaAllegroPoseAllHandFabric,
        )
        from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
        from fabrics_sim.integrator.integrators import DisplacementIntegrator
        from fabrics_sim.utils.path_utils import get_params_path
        from fabrics_sim.utils.utils import capture_fabric
        from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

        # Load fabric params and potentially modify
        path = Path(get_params_path())
        fabric_params_filename = "kuka_allegro_pose_params.yaml"
        config_path = path / fabric_params_filename
        with open(config_path, "r") as file:
            fabric_params = yaml.safe_load(file)
        fabric_params = fabric_params["fabric_params"]

        if self.custom_env_cfg.FABRIC_CSPACE_DAMPING is not None:
            fabric_params["cspace_damping"]["gain"] = (
                self.custom_env_cfg.FABRIC_CSPACE_DAMPING
            )

        if self.custom_env_cfg.FABRIC_CSPACE_DAMPING_HAND is not None:
            fabric_params["cspace_damping"]["hand_gain"] = (
                self.custom_env_cfg.FABRIC_CSPACE_DAMPING_HAND
            )

        # Declare device for fabric
        self.fabric_world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=20,
            device=self.device,
            world_dict=self.fabric_world_dict,
        )
        self.fabric_object_ids, self.fabric_object_indicator = (
            self.fabric_world_model.get_object_ids()
        )
        if self.custom_env_cfg.FABRIC_HAND_ACTION_SPACE == "PCA":
            self.fabric = KukaAllegroPoseFabric(
                batch_size=self.num_envs,
                device=self.device,
                timestep=self.sim_dt,
                graph_capturable=True,
                fabric_params=fabric_params,
            )
            self.fabric_hand_mins = torch.tensor(
                [0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=self.device
            )
            self.fabric_hand_maxs = torch.tensor(
                [3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=self.device
            )
        elif self.custom_env_cfg.FABRIC_HAND_ACTION_SPACE == "ALL":
            self.fabric = KukaAllegroPoseAllHandFabric(
                batch_size=self.num_envs,
                device=self.device,
                timestep=self.sim_dt,
                graph_capturable=True,
                fabric_params=fabric_params,
            )
            self.fabric_hand_mins = torch.tensor(
                self.right_robot_dof_lower_limits[
                    KUKA_ALLEGRO_NUM_ARM_DOFS : KUKA_ALLEGRO_NUM_ARM_DOFS
                    + KUKA_ALLEGRO_NUM_HAND_DOFS
                ],
                device=self.device,
            )
            self.fabric_hand_maxs = torch.tensor(
                self.right_robot_dof_upper_limits[
                    KUKA_ALLEGRO_NUM_ARM_DOFS : KUKA_ALLEGRO_NUM_ARM_DOFS
                    + KUKA_ALLEGRO_NUM_HAND_DOFS
                ],
                device=self.device,
            )
        else:
            raise ValueError(
                f"Invalid fabric_hand_action_space: {self.custom_env_cfg.FABRIC_HAND_ACTION_SPACE}"
            )

        self.fabric_palm_mins = torch.tensor(
            [0.0, -0.7, 0, -3.1416, -3.1416, -3.1416], device=self.device
        )
        self.fabric_palm_maxs = torch.tensor(
            [1.0, 0.7, 1.0, 3.1416, 3.1416, 3.1416], device=self.device
        )
        assert (self.fabric_hand_maxs > self.fabric_hand_mins).all(), (
            f"{self.fabric_hand_maxs} <= {self.fabric_hand_mins}"
        )
        assert (self.fabric_palm_maxs > self.fabric_palm_mins).all(), (
            f"{self.fabric_palm_maxs} <= {self.fabric_palm_mins}"
        )

        # Targets (stored as variables to enable CUDA graph computation)
        # Palm target is (origin, Euler ZYX)
        self.fabric_hand_target = rescale(
            torch.rand(
                self.num_envs, self.fabric_hand_maxs.numel(), device=self.device
            ),
            old_mins=torch.zeros_like(self.fabric_hand_mins),
            old_maxs=torch.ones_like(self.fabric_hand_mins),
            new_mins=self.fabric_hand_mins,
            new_maxs=self.fabric_hand_maxs,
        )

        default_palm_target = np.array(
            [-0.6868, 0.0320, 0.6685, -2.3873, -0.0824, 3.1301]
        )
        self.fabric_palm_target = (
            torch.from_numpy(default_palm_target)
            .float()
            .to(self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0)
        )

        self.fabric_integrator = DisplacementIntegrator(self.fabric)

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
            timestep=self.sim_dt,
            fabric_integrator=self.fabric_integrator,
            inputs=fabric_inputs,
            device=self.device,
        )

    def _setup_curriculum(self) -> None:
        from human2sim2robot.sim_training.utils.cross_embodiment.curriculum import (
            Curriculum,
            CurriculumUpdate,
            CurriculumUpdater,
        )

        curriculum_updates = [
            CurriculumUpdate(
                variable_name=curriculum_update.variable_name,
                update_amount=curriculum_update.update_amount,
                min=curriculum_update.min,
                max=curriculum_update.max,
            )
            for curriculum_update in self.custom_env_cfg.curriculum_updates
        ]
        self.curriculum = Curriculum(
            curriculum_cfg=self.custom_env_cfg.curriculum,
            curriculum_updater=CurriculumUpdater(
                curriculum_updates=curriculum_updates,
                context=self,
            ),
        )

    def _setup_curobo(self) -> None:
        # https://curobo.org/get_started/2c_world_collision.html#batched-environments

        # Set up worlds
        from curobo.geom.sdf.world import (
            CollisionQueryBuffer,
            WorldCollisionConfig,
            WorldPrimitiveCollision,
        )
        from curobo.geom.types import (
            Cuboid,
            WorldConfig,
        )
        from curobo.types.base import TensorDeviceType
        from curobo.util_file import join_path

        from human2sim2robot.sim_training.utils.cross_embodiment.fabric_world import (
            transform_str_to_T,
        )

        def T_to_pose(T: np.ndarray) -> List[float]:
            return (
                T[:3, 3].tolist()
                + R.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()
            )

        tensor_args = TensorDeviceType(device=torch.device(self.device))
        object_dims = self.object_mesh.extents
        world_configs = [
            WorldConfig(
                cuboid=[
                    Cuboid(
                        name="object",
                        pose=self.object_pos[i].tolist()
                        + self.object_quat_xyzw[i, [3, 0, 1, 2]].tolist(),
                        dims=object_dims.tolist(),
                        color=[0.8, 0.0, 0.0, 1.0],
                        tensor_args=tensor_args,
                    ),
                    Cuboid(
                        name="table",
                        pose=[
                            TABLE_X,
                            TABLE_Y,
                            TABLE_Z,
                            TABLE_QW,
                            TABLE_QX,
                            TABLE_QY,
                            TABLE_QZ,
                        ],
                        dims=[TABLE_LENGTH_X, TABLE_LENGTH_Y, TABLE_LENGTH_Z],
                        color=[0.0, 0.8, 0.0, 1.0],
                        tensor_args=tensor_args,
                    ),
                ]
                + [
                    Cuboid(
                        name=f"fabric_{k}",
                        pose=T_to_pose(transform_str_to_T(v["transform"])),
                        dims=[float(x) for x in v["scaling"].split(" ")],
                        color=[0.0, 0.0, 0.8, 1.0],
                        tensor_args=tensor_args,
                    )
                    for k, v in self.fabric_world_dict.items()
                ]
            )
            for i in range(self.num_envs)
        ]
        world_coll_config = WorldCollisionConfig(
            tensor_args=tensor_args, world_model=world_configs
        )
        self.world_ccheck = WorldPrimitiveCollision(world_coll_config)

        from curobo.cuda_robot_model.cuda_robot_model import (
            CudaRobotModel,
        )
        from curobo.types.robot import RobotConfig
        from curobo.util_file import (
            get_robot_configs_path,
            load_yaml,
        )

        ROBOT_FILE = "iiwa_allegro.yml"
        robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), ROBOT_FILE))[
            "robot_cfg"
        ]
        robot_cfg = RobotConfig.from_dict(robot_cfg_dict, tensor_args=tensor_args)
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

        self.NUM_SAMPLES_PER_ENV = 1
        q_repeated = (
            self.right_robot_dof_pos.detach()
            .clone()
            .unsqueeze(dim=0)
            .repeat_interleave(self.NUM_SAMPLES_PER_ENV, dim=0)
            .reshape(-1, KUKA_ALLEGRO_NUM_DOFS)
        )
        assert_equals(
            q_repeated.shape,
            (self.num_envs * self.NUM_SAMPLES_PER_ENV, KUKA_ALLEGRO_NUM_DOFS),
        )
        x_sph = self._get_collision_spheres(q_repeated)
        x_sph = x_sph.view(self.num_envs, self.NUM_SAMPLES_PER_ENV, -1, 4)
        self.query_buffer = CollisionQueryBuffer.initialize_from_shape(
            shape=x_sph.shape,
            tensor_args=tensor_args,
            collision_types=self.world_ccheck.collision_types,
        )
        self.act_distance = tensor_args.to_device([0.0])
        self.weight = tensor_args.to_device([1])

        # IK Solver
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        # Note: Larger batch size to parallelize more. But at a certain size:
        # - CUDA out of memory
        # - Parallelization is fully loaded, so no further speedup
        # - Wasteful because must use the full batch size for IK solver, but often only need to reset a few
        self.IK_BATCH_SIZE = 50

        # If the number of environments is less than the batch size, we don't need to use such a large batch size
        if self.num_envs < self.IK_BATCH_SIZE:
            self.IK_BATCH_SIZE = self.num_envs

        self.IK_N_SEEDS = 20
        ik_world_configs = [
            WorldConfig(
                cuboid=[
                    Cuboid(
                        name="object",
                        pose=self.object_pos[i].tolist()
                        + self.object_quat_xyzw[i, [3, 0, 1, 2]].tolist(),
                        dims=object_dims.tolist(),
                        color=[0.8, 0.0, 0.0, 1.0],
                        tensor_args=tensor_args,
                    ),
                    Cuboid(
                        name="table",
                        pose=[
                            TABLE_X,
                            TABLE_Y,
                            TABLE_Z,
                            TABLE_QW,
                            TABLE_QX,
                            TABLE_QY,
                            TABLE_QZ,
                        ],
                        dims=[TABLE_LENGTH_X, TABLE_LENGTH_Y, TABLE_LENGTH_Z],
                        color=[0.0, 0.8, 0.0, 1.0],
                        tensor_args=tensor_args,
                    ),
                ]
                + [
                    Cuboid(
                        name=f"fabric_{k}",
                        pose=T_to_pose(transform_str_to_T(v["transform"])),
                        dims=[float(x) for x in v["scaling"].split(" ")],
                        color=[0.0, 0.0, 0.8, 1.0],
                        tensor_args=tensor_args,
                    )
                    for k, v in self.fabric_world_dict.items()
                ]
            )
            for i in range(self.IK_BATCH_SIZE)
        ]
        ik_world_coll_config = WorldCollisionConfig(
            tensor_args=tensor_args, world_model=ik_world_configs
        )
        self.ik_world_ccheck = WorldPrimitiveCollision(ik_world_coll_config)

        ik_q_repeated = (
            self.right_robot_dof_pos.detach()[: self.IK_BATCH_SIZE]
            .clone()
            .unsqueeze(dim=0)
            .repeat_interleave(self.NUM_SAMPLES_PER_ENV, dim=0)
            .reshape(-1, KUKA_ALLEGRO_NUM_DOFS)
        )
        assert_equals(
            ik_q_repeated.shape,
            (self.IK_BATCH_SIZE * self.NUM_SAMPLES_PER_ENV, KUKA_ALLEGRO_NUM_DOFS),
        )
        ik_x_sph = self._get_collision_spheres(ik_q_repeated)
        ik_x_sph = ik_x_sph.view(self.IK_BATCH_SIZE, self.NUM_SAMPLES_PER_ENV, -1, 4)
        self.ik_query_buffer = CollisionQueryBuffer.initialize_from_shape(
            shape=ik_x_sph.shape,
            tensor_args=tensor_args,
            collision_types=self.ik_world_ccheck.collision_types,
        )
        self.ik_act_distance = tensor_args.to_device([0.0])
        self.ik_weight = tensor_args.to_device([1])

        ik_robot_cfg_dict = robot_cfg_dict.copy()
        ik_robot_cfg_dict["kinematics"]["ee_link"] = "palm_link"
        ik_robot_cfg_dict["kinematics"]["link_names"] = []
        ik_robot_cfg = RobotConfig.from_dict(ik_robot_cfg_dict, tensor_args=tensor_args)
        ik_config = IKSolverConfig.load_from_robot_config(
            ik_robot_cfg,
            world_coll_checker=self.world_ccheck,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=self.IK_N_SEEDS,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(ik_config)

    def _load_retargeted_robot(self) -> None:
        retargeted_robot_file = (
            get_repo_root_dir() / self.custom_env_cfg.retargeted_robot_file
        )
        assert retargeted_robot_file.exists(), (
            f"Retargeted robot file {retargeted_robot_file} does not exist"
        )
        self.retargeted_robot_data = np.load(retargeted_robot_file)

        qs = self.retargeted_robot_data["qs"]
        idxs = self.retargeted_robot_data["idxs"]
        T_O_Ps = self.retargeted_robot_data["relative_premanip_poses"]

        N = qs.shape[0]
        assert qs.ndim == 2, f"qs has shape {qs.shape}"
        assert qs.shape == (N, KUKA_ALLEGRO_NUM_DOFS), (
            f"qs has shape {qs.shape} but expected shape to be (N, {KUKA_ALLEGRO_NUM_DOFS})"
        )
        assert idxs.shape == (N,), (
            f"idxs has shape {idxs.shape} but expected shape to be (N,)"
        )
        assert T_O_Ps.shape == (N, 4, 4), (
            f"T_O_Ps has shape {T_O_Ps.shape} but expected shape to be (N, 4, 4)"
        )

        self.absolute_premanipulation_pose = qs[0]
        self.relative_premanipulation_pose = T_O_Ps[0]
        self.demo_start_idx = idxs[0]

    def solve_iks_mf_one_step(
        self,
        X_W_Hs: torch.Tensor,
        default_qs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from curobo.types.math import Pose

        N = X_W_Hs.shape[0]
        assert N == self.num_envs, f"N: {N}, self.num_envs: {self.num_envs}"
        assert X_W_Hs.shape == (N, 4, 4), f"X_W_Hs.shape: {X_W_Hs.shape}"
        if default_qs is not None:
            assert default_qs.shape == (N, 23), f"default_q.shape: {default_qs.shape}"

        trans = X_W_Hs[:, :3, 3]
        rot_matrix = X_W_Hs[:, :3, :3]
        quat_wxyz = matrix_to_quat_wxyz(rot_matrix)

        target_pose = Pose(
            trans,
            quaternion=quat_wxyz,
        )

        num_joints = self.ik_solver.robot_config.kinematics.kinematics_config.joint_limits.position.shape[
            -1
        ]

        result = self.ik_solver.solve_batch(
            target_pose,
            retract_config=(default_qs if default_qs is not None else None),
            seed_config=(
                default_qs[:, None].repeat_interleave(self.IK_N_SEEDS, dim=1)
                if default_qs is not None
                else None
            ),
        )
        assert result.solution.shape == (N, 1, num_joints)
        assert result.success.shape == (N, 1)

        return (
            result.solution.squeeze(dim=1),
            result.success.squeeze(dim=1),
        )

    def solve_iks_mf_multiple_steps(
        self,
        X_W_Hs: torch.Tensor,
        object_state: torch.Tensor,
        default_qs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from curobo.types.math import Pose

        N = X_W_Hs.shape[0]
        assert X_W_Hs.shape == (N, 4, 4), f"X_W_Hs.shape: {X_W_Hs.shape}"
        assert object_state.shape == (
            N,
            NUM_STATES,
        ), f"object_state.shape: {object_state.shape}"
        if default_qs is not None:
            assert default_qs.shape == (N, 23), f"default_q.shape: {default_qs.shape}"

        trans = X_W_Hs[:, :3, 3]
        rot_matrix = X_W_Hs[:, :3, :3]
        quat_wxyz = matrix_to_quat_wxyz(rot_matrix)

        num_joints = self.ik_solver.robot_config.kinematics.kinematics_config.joint_limits.position.shape[
            -1
        ]

        new_solution, new_success = (
            torch.zeros((N, num_joints), device=self.device),
            torch.zeros((N,), device=self.device),
        )
        n_full_batches = N // self.IK_BATCH_SIZE
        last_batch_size = N % self.IK_BATCH_SIZE
        if n_full_batches == 0:
            # Edge case: Only 1 batch
            # We need to use the full batch size for IK solver, so we do repeating to make it a full batch
            assert last_batch_size > 0
            n_full_batches = 1
            last_batch_size = 0
            trans = torch.cat(
                [
                    trans,
                    trans.clone()[0:1].repeat_interleave(self.IK_BATCH_SIZE - N, dim=0),
                ]
            )
            quat_wxyz = torch.cat(
                [
                    quat_wxyz,
                    quat_wxyz.clone()[0:1].repeat_interleave(
                        self.IK_BATCH_SIZE - N, dim=0
                    ),
                ]
            )
            if default_qs is not None:
                default_qs = torch.cat(
                    [
                        default_qs,
                        default_qs.clone()[0:1].repeat_interleave(
                            self.IK_BATCH_SIZE - N, dim=0
                        ),
                    ]
                )
            object_state = torch.cat(
                [
                    object_state,
                    object_state.clone()[0:1].repeat_interleave(
                        self.IK_BATCH_SIZE - N, dim=0
                    ),
                ]
            )
            target_pose = Pose(
                trans,
                quaternion=quat_wxyz,
            )
            self.ik_world_ccheck.update_obstacle_poses(
                name="object",
                w_obj_pose=Pose(
                    position=object_state[:, START_POS_IDX:END_POS_IDX].clone(),
                    quaternion=object_state[:, START_QUAT_IDX:END_QUAT_IDX][
                        :, [3, 0, 1, 2]
                    ].clone(),
                ),
                env_idxs=torch.ones(self.IK_BATCH_SIZE, device=self.device)
                .nonzero(as_tuple=False)
                .squeeze(dim=-1),
            )
            result = self.ik_solver.solve_batch(
                target_pose,
                retract_config=(default_qs if default_qs is not None else None),
                seed_config=(
                    default_qs[:, None].repeat_interleave(self.IK_N_SEEDS, dim=1)
                    if default_qs is not None
                    else None
                ),
            )
            assert result.solution.shape == (self.IK_BATCH_SIZE, 1, num_joints)
            assert result.success.shape == (self.IK_BATCH_SIZE, 1)
            new_solution[:N] = result.solution[:N].squeeze(dim=1)
            new_success[:N] = result.success[:N].squeeze(dim=1)

            return (
                new_solution,
                new_success,
            )

        # Normal case: Multiple batches
        target_pose = Pose(
            trans,
            quaternion=quat_wxyz,
        )

        for i in range(n_full_batches):
            self.ik_world_ccheck.update_obstacle_poses(
                name="object",
                w_obj_pose=Pose(
                    position=object_state[
                        i * self.IK_BATCH_SIZE : (i + 1) * self.IK_BATCH_SIZE,
                        START_POS_IDX:END_POS_IDX,
                    ].clone(),
                    quaternion=object_state[
                        i * self.IK_BATCH_SIZE : (i + 1) * self.IK_BATCH_SIZE,
                        START_QUAT_IDX:END_QUAT_IDX,
                    ][:, [3, 0, 1, 2]].clone(),
                ),
                env_idxs=torch.ones(self.IK_BATCH_SIZE, device=self.device)
                .nonzero(as_tuple=False)
                .squeeze(dim=-1),
            )

            result = self.ik_solver.solve_batch(
                target_pose[i * self.IK_BATCH_SIZE : (i + 1) * self.IK_BATCH_SIZE],
                retract_config=(
                    default_qs[i * self.IK_BATCH_SIZE : (i + 1) * self.IK_BATCH_SIZE]
                    if default_qs is not None
                    else None
                ),
                seed_config=(
                    default_qs[i * self.IK_BATCH_SIZE : (i + 1) * self.IK_BATCH_SIZE][
                        :, None
                    ].repeat_interleave(self.IK_N_SEEDS, dim=1)
                    if default_qs is not None
                    else None
                ),
            )
            assert result.solution.shape == (self.IK_BATCH_SIZE, 1, num_joints)
            assert result.success.shape == (self.IK_BATCH_SIZE, 1)
            new_solution[i * self.IK_BATCH_SIZE : (i + 1) * self.IK_BATCH_SIZE] = (
                result.solution.squeeze(dim=1)
            )
            new_success[i * self.IK_BATCH_SIZE : (i + 1) * self.IK_BATCH_SIZE] = (
                result.success.squeeze(dim=1)
            )

        if last_batch_size > 0:
            # Must still use full batch size for IK solver
            # Then we only extract the ones we need
            self.ik_world_ccheck.update_obstacle_poses(
                name="object",
                w_obj_pose=Pose(
                    position=object_state[
                        -self.IK_BATCH_SIZE :, START_POS_IDX:END_POS_IDX
                    ].clone(),
                    quaternion=object_state[
                        -self.IK_BATCH_SIZE :, START_QUAT_IDX:END_QUAT_IDX
                    ][:, [3, 0, 1, 2]].clone(),
                ),
                env_idxs=torch.ones(self.IK_BATCH_SIZE, device=self.device)
                .nonzero(as_tuple=False)
                .squeeze(dim=-1),
            )
            result = self.ik_solver.solve_batch(
                target_pose[-self.IK_BATCH_SIZE :],
                retract_config=(
                    default_qs[-self.IK_BATCH_SIZE :]
                    if default_qs is not None
                    else None
                ),
                seed_config=(
                    default_qs[-self.IK_BATCH_SIZE :][:, None].repeat_interleave(
                        self.IK_N_SEEDS, dim=1
                    )
                    if default_qs is not None
                    else None
                ),
            )
            assert result.solution.shape == (self.IK_BATCH_SIZE, 1, num_joints)
            assert result.success.shape == (self.IK_BATCH_SIZE, 1)
            new_solution[-last_batch_size:] = result.solution[
                -last_batch_size:
            ].squeeze(dim=1)
            new_success[-last_batch_size:] = result.success[-last_batch_size:].squeeze(
                dim=1
            )

        return (
            new_solution,
            new_success,
        )

    def _get_collision_spheres(self, q: torch.Tensor) -> torch.Tensor:
        N = q.shape[0]
        assert_equals(q.shape, (N, KUKA_ALLEGRO_NUM_DOFS))

        out = self.kin_model.get_state(q)
        link_spheres_tensor = out.link_spheres_tensor
        assert link_spheres_tensor is not None
        N_SPHERES = link_spheres_tensor.shape[1]
        assert link_spheres_tensor.shape == (
            N,
            N_SPHERES,
            4,
        ), link_spheres_tensor.shape

        x_sph = link_spheres_tensor.unsqueeze(dim=1)
        assert x_sph.shape == (N, 1, N_SPHERES, 4), x_sph.shape
        return x_sph

    def _setup_taskmap(self) -> None:
        from fabrics_sim.taskmaps.robot_frame_origins_taskmap import (
            RobotFrameOriginsTaskMap,
        )

        # Create task map that consists of the origins of the following frames stacked together.
        self.taskmap_link_names = PALM_LINK_NAMES + ALLEGRO_FINGERTIP_LINK_NAMES
        self.taskmap = RobotFrameOriginsTaskMap(
            urdf_path=str(Path(KUKA_ALLEGRO_ASSET_ROOT) / KUKA_ALLEGRO_FILENAME),
            link_names=self.taskmap_link_names,
            batch_size=self.num_envs,
            device=self.device,
        )

    def _load_reference_motion(self) -> None:
        reference_object_trajectory_folder = (
            get_repo_root_dir() / self.custom_env_cfg.object_poses_dir
        )
        assert reference_object_trajectory_folder.exists(), (
            f"Reference object trajectory folder {reference_object_trajectory_folder} does not exist"
        )
        raw_T_list = (
            torch.from_numpy(
                read_in_T_list(
                    object_trajectory_folder=reference_object_trajectory_folder
                )
            )
            .float()
            .to(self.device)
        )

        START_IDX_MODE = "CLIP"
        if START_IDX_MODE == "DEMO":
            self.reference_T_C_O_list = raw_T_list[self.demo_start_idx :]
        elif START_IDX_MODE == "CLIP":
            self.reference_T_C_O_list = clip_T_list(
                raw_T_list=raw_T_list,
                data_dt=REFERENCE_MOTION_DT,
            )
        else:
            raise ValueError(f"Invalid START_IDX_MODE: {START_IDX_MODE}")

    def _update_num_observations_and_actions_if_needed(self) -> None:
        need_wandb_update = False
        if self.env_cfg.numObservations < 0:
            self.env_cfg.numObservations = compute_num_observations(
                USE_FABRIC_ACTION_SPACE=self.custom_env_cfg.USE_FABRIC_ACTION_SPACE
            )
            if USE_STATE_AS_OBSERVATION:
                self.env_cfg.numObservations = compute_num_states(
                    USE_FABRIC_ACTION_SPACE=self.custom_env_cfg.USE_FABRIC_ACTION_SPACE,
                    INCLUDE_DISH_RACK=self.INCLUDE_DISH_RACK,
                )

            self.logger.info(
                f"Setting numObservations to {self.env_cfg.numObservations}"
            )
            need_wandb_update = True
        if self.env_cfg.numActions < 0:
            self.env_cfg.numActions = compute_num_actions(
                USE_FABRIC_ACTION_SPACE=self.custom_env_cfg.USE_FABRIC_ACTION_SPACE,
                FABRIC_HAND_ACTION_SPACE=self.custom_env_cfg.FABRIC_HAND_ACTION_SPACE,
            )
            self.logger.info(f"Setting numActions to {self.env_cfg.numActions}")
            need_wandb_update = True
        if self.env_cfg.numStates < 0:
            self.env_cfg.numStates = compute_num_states(
                USE_FABRIC_ACTION_SPACE=self.custom_env_cfg.USE_FABRIC_ACTION_SPACE,
                INCLUDE_DISH_RACK=self.INCLUDE_DISH_RACK,
            )
            self.logger.info(f"Setting numStates to {self.env_cfg.numStates}")
            need_wandb_update = True
        if need_wandb_update and wandb_started():
            wandb.config.update(
                {"task": omegaconf_to_dict(self.cfg)}, allow_val_change=True
            )

    def _initialize_camera_sensor(self, cam_pos, cam_target) -> None:
        self.camera_properties = gymapi.CameraProperties()
        RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE = 4
        self.camera_properties.width = int(
            self.camera_properties.width / RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE
        )
        self.camera_properties.height = int(
            self.camera_properties.height / RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE
        )
        self.camera_handle = self.gym.create_camera_sensor(
            self.envs[self.index_to_view],
            self.camera_properties,
        )

        # self.video_frames is important for understanding the state of video recording
        #   Case 1: self.video_frames is None:
        #     * This means that we are not recording video
        #   Case 2: self.video_frames = []
        #     * This means that we should start recording video
        #     * BUT, we want our videos to start at the first frame of an episode
        #     * So, we are waiting for this
        #   Case 3: self.video_frames = [np.array(frame) for frame in ...]
        #     * These are image frames that will be assembled into a video when enough frames are capture
        self.video_frames: Optional[List[np.ndarray]] = None
        self.gym.set_camera_location(
            self.camera_handle, self.envs[self.index_to_view], cam_pos, cam_target
        )

    def _setup_reward_weights(self) -> None:
        # Make sure this aligns with compute_reward_jit
        self.reward_weight_dict = {
            "Indexfingertip-Goal Distance Reward": 0.0,
            "Fingertips-Object Distance Reward": 1.0,
            "Object-Goal Distance Reward": 10.0,
            "Success Reward": 2.0,
            "Consecutive Success Reward": 4.0,
            "Object Tracking Reward": 0.0,
            "Action Smoothing Penalty": self.custom_env_cfg.action_smoothing_penalty_weight,
            "Hand Tracking Reward": 0.0,
        }

        if self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING:
            self.reward_weight_dict = {
                "Indexfingertip-Goal Distance Reward": 0.0,
                "Fingertips-Object Distance Reward": 0.0,
                "Object-Goal Distance Reward": 0.0,
                "Success Reward": 0.0,
                "Consecutive Success Reward": 0.0,
                "Object Tracking Reward": 1.0,
                "Action Smoothing Penalty": self.custom_env_cfg.action_smoothing_penalty_weight,
                "Hand Tracking Reward": 0.0,
            }

        self.logger.info("*" * 80)
        self.logger.info("reward_weight_dict:")
        for name, weight in self.reward_weight_dict.items():
            self.logger.info(f"  {name}: {weight}")
        self.logger.info("*" * 80)

        self.reward_names = [
            reward_name for reward_name in self.reward_weight_dict.keys()
        ]

        self.reward_weights = torch.tensor(
            [self.reward_weight_dict[name] for name in self.reward_names],
            device=self.device,
        ).reshape(1, -1)

    def create_sim(self) -> None:
        self.up_axis = self.cfg["sim"]["up_axis"]
        assert self.up_axis in ["y", "z"], f"Invalid up_axis: {self.up_axis}"
        self.up_axis_index = 1 if self.up_axis == "y" else 2

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.env_cfg.envSpacing, int(np.sqrt(self.num_envs))
        )

    def _create_ground_plane(self) -> None:
        plane_params = gymapi.PlaneParams()
        plane_params.normal = (
            gymapi.Vec3(0, 1, 0) if self.up_axis == "y" else gymapi.Vec3(0, 0, 1)
        )
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs: int, spacing: float, num_per_row: int) -> None:
        # Must be done here because it is before self.init_object_pose is used
        # But after self.num_envs and self.device are set
        if self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING:
            self._load_reference_motion()

        lower = (
            gymapi.Vec3(-spacing, -spacing, 0.0)
            if self.up_axis == "z"
            else gymapi.Vec3(-spacing, 0.0, -spacing)
        )
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # BRITTLE: Add assets here
        ALL_ACTOR_ASSETS = [
            self.right_robot_asset,
            self.table_asset,
            self.object_asset,
            self.goal_object_asset,
            self.flat_box_asset,
        ]
        if INCLUDE_METAL_CYLINDER:
            ALL_ACTOR_ASSETS.append(self.metal_cylinder_asset)
        if INCLUDE_LARGE_SAUCEPAN:
            ALL_ACTOR_ASSETS.append(self.large_saucepan_asset)
        if self.INCLUDE_DISH_RACK:
            ALL_ACTOR_ASSETS.append(self.dishrack_asset)

        max_agg_bodies = sum(
            [self.gym.get_asset_rigid_body_count(asset) for asset in ALL_ACTOR_ASSETS]
        )
        max_agg_shapes = sum(
            [self.gym.get_asset_rigid_shape_count(asset) for asset in ALL_ACTOR_ASSETS]
        )

        # Add force sensors
        rb_idxs = [
            self.gym.find_asset_rigid_body_index(asset, name)
            for asset in ALL_ACTOR_ASSETS
            for name in self.gym.get_asset_rigid_body_names(asset)
        ]
        for rb_idx in rb_idxs:
            self.gym.create_asset_force_sensor(
                self.right_robot_asset,
                rb_idx,
                self.force_sensor_pose,
                self.force_sensor_properties,
            )

        # Set asset rigid shape properties
        self.gym.set_asset_rigid_shape_properties(
            self.object_asset, self.desired_object_rigid_shape_props
        )
        self.gym.set_asset_rigid_shape_properties(
            self.right_robot_asset, self.desired_right_robot_rigid_shape_props
        )
        self.gym.set_asset_rigid_shape_properties(
            self.table_asset, self.desired_table_rigid_shape_props
        )
        self.gym.set_asset_rigid_shape_properties(
            self.flat_box_asset, self.desired_flat_box_rigid_shape_props
        )

        # Store
        self.envs = []
        self.right_robots = []
        self.goal_objects = []
        self.objects = []
        self.tables = []
        self.flat_boxes = []
        self.metal_cylinders = []
        self.large_saucepans = []
        self.dishracks = []

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)

            # Different collision groups to different envs don't interact
            # collision_filter = -1 for default
            # collision_filter = 0 for enabled self-collision
            # collision_filter > 0 for disabled self-collision
            # Note: for some reason, need to set collision_filter to 0 or else the two hands pass through each other
            collision_group, collision_filter, segmentation_id = i, 0, 0

            ENABLE_AGGREGATE = False  # Note: For some reason, aggregate can sometimes cause issues, set to False in those cases
            if ENABLE_AGGREGATE:
                self.gym.begin_aggregate(env, max_agg_bodies, max_agg_shapes, True)

            # Right robot
            right_robot_actor = self.gym.create_actor(
                env,
                self.right_robot_asset,
                self.init_right_robot_pose,
                "right_robot",
                collision_group,
                collision_filter,
                segmentation_id,
            )
            self.gym.enable_actor_dof_force_sensors(env, right_robot_actor)
            self.gym.set_actor_dof_properties(
                env, right_robot_actor, self.desired_right_robot_dof_props
            )
            self.right_robots.append(right_robot_actor)

            # Table
            table_actor = self.gym.create_actor(
                env,
                self.table_asset,
                self.init_table_pose,
                "table",
                collision_group,
                collision_filter,
                segmentation_id + 1,
            )
            if not USE_REAL_TABLE_MESH:
                self._set_actor_texture(env, table_actor, self.table_texture)
            self.tables.append(table_actor)

            # Goal object
            goal_object_actor = self.gym.create_actor(
                env,
                self.goal_object_asset,
                self.init_goal_object_pose,
                "goal_object",
                self._get_next_dummy_collision_group(
                    num_envs=num_envs, collision_group=collision_group
                ),
                collision_filter,
                segmentation_id + 2,
            )
            self._set_actor_color(env, goal_object_actor, GREEN)
            self.goal_objects.append(goal_object_actor)

            # Object
            object_actor = self.gym.create_actor(
                env,
                self.object_asset,
                self.init_object_pose,
                "object",
                collision_group,
                collision_filter,
                segmentation_id + 2,
            )

            self.objects.append(object_actor)

            # Flat box
            flat_box_actor = self.gym.create_actor(
                env,
                self.flat_box_asset,
                self.init_flat_box_pose,
                "flat_box",
                collision_group,
                collision_filter,
                segmentation_id + 6,
            )
            self._set_actor_texture(env, flat_box_actor, self.box_texture)
            self.flat_boxes.append(flat_box_actor)

            if INCLUDE_METAL_CYLINDER:
                # Metal cylinder
                metal_cylinder_actor = self.gym.create_actor(
                    env,
                    self.metal_cylinder_asset,
                    self.init_metal_cylinder_pose,
                    "metal_cylinder",
                    collision_group,
                    collision_filter,
                    segmentation_id + 7,
                )
                self.metal_cylinders.append(metal_cylinder_actor)

            if INCLUDE_LARGE_SAUCEPAN:
                # Large saucepan
                large_saucepan_actor = self.gym.create_actor(
                    env,
                    self.large_saucepan_asset,
                    self.init_large_saucepan_pose,
                    "large_saucepan",
                    collision_group,
                    collision_filter,
                    segmentation_id + 5,
                )
                self.large_saucepans.append(large_saucepan_actor)

            if self.INCLUDE_DISH_RACK:
                # Dishrack
                dishrack_actor = self.gym.create_actor(
                    env,
                    self.dishrack_asset,
                    self.init_dishrack_pose,
                    "dishrack",
                    collision_group,
                    collision_filter,
                    segmentation_id + 3,
                )
                self.dishracks.append(dishrack_actor)

            if ENABLE_AGGREGATE:
                self.gym.end_aggregate(env)

        self.right_robot_indices = self._get_actor_indices(
            envs=self.envs, actors=self.right_robots
        )
        self.table_indices = self._get_actor_indices(envs=self.envs, actors=self.tables)
        self.goal_object_indices = self._get_actor_indices(
            envs=self.envs, actors=self.goal_objects
        )
        self.object_indices = self._get_actor_indices(
            envs=self.envs, actors=self.objects
        )

        self.flat_box_indices = self._get_actor_indices(
            envs=self.envs, actors=self.flat_boxes
        )
        if INCLUDE_METAL_CYLINDER:
            self.metal_cylinder_indices = self._get_actor_indices(
                envs=self.envs, actors=self.metal_cylinders
            )
        if INCLUDE_LARGE_SAUCEPAN:
            self.large_saucepan_indices = self._get_actor_indices(
                envs=self.envs, actors=self.large_saucepans
            )
        if self.INCLUDE_DISH_RACK:
            self.dishrack_indices = self._get_actor_indices(
                envs=self.envs, actors=self.dishracks
            )

        # Get original mass and inertia
        original_object_mass = []
        for env, object in zip(self.envs, self.objects):
            object_rb_props = self.gym.get_actor_rigid_body_properties(env, object)
            assert_equals(len(object_rb_props), OBJECT_NUM_RIGID_BODIES)
            original_object_mass.append(
                np.sum(
                    [object_rb_props[i].mass for i in range(OBJECT_NUM_RIGID_BODIES)]
                )
            )
        self.original_object_mass = to_torch(
            original_object_mass, dtype=torch.float, device=self.device
        )

        original_object_inertia = []
        for env, object in zip(self.envs, self.objects):
            object_rb_props = self.gym.get_actor_rigid_body_properties(env, object)
            assert_equals(len(object_rb_props), OBJECT_NUM_RIGID_BODIES)
            original_object_inertia.append(
                np.sum(
                    [
                        np.mean(
                            [
                                object_rb_props[i].inertia.x.x,
                                object_rb_props[i].inertia.y.y,
                                object_rb_props[i].inertia.z.z,
                            ]
                        )
                        for i in range(OBJECT_NUM_RIGID_BODIES)
                    ]
                )
            )
        self.original_object_inertia = to_torch(
            original_object_inertia, dtype=torch.float, device=self.device
        )

        # Set mass and inertia
        self.object_mass_scale = self.custom_env_cfg.object_mass_scale
        self.object_inertia_scale = self.custom_env_cfg.object_inertia_scale

    def _get_next_dummy_collision_group(
        self, num_envs: int, collision_group: int
    ) -> int:
        if not hasattr(self, "_next_dummy_collision_group_dict"):
            self._next_dummy_collision_group_dict = {}

        if collision_group not in self._next_dummy_collision_group_dict:
            self._next_dummy_collision_group_dict[collision_group] = (
                collision_group + num_envs
            )
        else:
            self._next_dummy_collision_group_dict[collision_group] += num_envs
        return self._next_dummy_collision_group_dict[collision_group]

    def _get_actor_indices(self, envs, actors) -> torch.Tensor:
        assert_equals(len(envs), len(actors))
        actor_indices = to_torch(
            [
                self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)
                for env, actor in zip(envs, actors)
            ],
            dtype=torch.long,
            device=self.device,
        )
        return actor_indices

    def _set_actor_color(self, env, actor, color: Tuple[float, float, float]) -> None:
        for rigid_body_idx in range(self.gym.get_actor_rigid_body_count(env, actor)):
            self.gym.set_rigid_body_color(
                env,
                actor,
                rigid_body_idx,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(*color),
            )

    def _set_actor_texture(self, env, actor, texture) -> None:
        if texture is None:
            return

        for rigid_body_idx in range(self.gym.get_actor_rigid_body_count(env, actor)):
            self.gym.set_rigid_body_texture(
                env,
                actor,
                rigid_body_idx,
                gymapi.MESH_VISUAL_AND_COLLISION,
                texture,
            )

    def _initialize_state_tensors(self) -> None:
        # Make sure this aligns with _refresh_state_tensors

        # Dof state
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., 1]
        print(f"Found {self.dof_state.shape[1]} dofs")

        # Root state
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.actor_root_state = gymtorch.wrap_tensor(actor_root_state_tensor)

        # Rigid body state
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor)
        self.rigid_body_state_by_env = self.rigid_body_state.view(
            self.num_envs, -1, NUM_STATES
        )
        print(f"Found {self.rigid_body_state_by_env.shape[1]} rigid bodies per env")

        # Force sensor state
        try:
            force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            NUM_SENSORS = -1
            self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(
                self.num_envs, NUM_SENSORS, (NUM_XYZ + NUM_XYZ)
            )
            print(f"Found {self.force_sensor_tensor.shape[1]} force sensors per env")
        except AttributeError:
            print("No force sensors found")
            self.force_sensor_tensor = None

        # Dof force state
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs,
            -1,
        )
        print(f"Found {self.dof_force_tensor.shape[1]} dofs per env")

        # Net contact force
        net_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_contact_force_tensor = gymtorch.wrap_tensor(
            net_contact_force_tensor
        ).view(self.num_envs, -1, NUM_XYZ)
        print(f"Found {self.net_contact_force_tensor.shape[1]} contacts per env")

        # Log
        self.logger.info("#" * 80)
        self.logger.info(f"self.dof_state.shape = {self.dof_state.shape}")
        self.logger.info(f"self.dof_pos.shape = {self.dof_pos.shape}")
        self.logger.info(f"self.dof_vel.shape = {self.dof_vel.shape}")
        self.logger.info(f"self.actor_root_state.shape = {self.actor_root_state.shape}")
        self.logger.info(f"self.rigid_body_state.shape = {self.rigid_body_state.shape}")
        self.logger.info(
            f"self.rigid_body_state_by_env.shape = {self.rigid_body_state_by_env.shape}"
        )
        if self.force_sensor_tensor is not None:
            self.logger.info(
                f"self.force_sensor_tensor.shape = {self.force_sensor_tensor.shape}"
            )
        self.logger.info(f"self.dof_force_tensor.shape = {self.dof_force_tensor.shape}")
        self.logger.info(
            f"self.net_contact_force_tensor.shape = {self.net_contact_force_tensor.shape}"
        )
        self.logger.info("#" * 80 + "\n")

        # For deferred setting to avoid calling set_* more than once per step
        self.set_dof_state_object_indices = []
        self.set_actor_root_state_object_indices = []
        self.set_dof_position_target_tensor_indices = []
        self.set_dof_velocity_target_tensor_indices = []

    def _refresh_state_tensors(self) -> None:
        # Make sure this aligns with _initialize_state_tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _sanity_checks(self) -> None:
        # Check number of dofs
        sim_dof_count = self.gym.get_sim_dof_count(self.sim)
        env_dof_count = sim_dof_count // self.num_envs
        assert_equals(sim_dof_count, env_dof_count * self.num_envs)

    ##### INITIALIZATION END #####

    ##### WANDB START #####
    def log_wandb_dict(self) -> None:
        if not wandb_started():
            return

        # Skip if empty
        if len(self.wandb_dict) == 0:
            return

        wandb.log(self.wandb_dict)
        self.wandb_dict = {}

    def _save_config_file_to_wandb(self) -> None:
        if not wandb_started():
            return

        # BRITTLE: should match train.py
        config_filepath_1 = self.log_dir / "config.yaml"
        config_filepath_2 = self.log_dir / "config_resolved.yaml"
        self._save_file_to_wandb_safe(config_filepath_1)
        self._save_file_to_wandb_safe(config_filepath_2)

    def save_model_to_wandb(self) -> None:
        if not wandb_started():
            return

        should_save_model = (
            self.num_steps_taken % self.log_cfg.saveBestModelToWandbEveryNSteps == 0
            and self.num_steps_taken // self.log_cfg.saveBestModelToWandbEveryNSteps > 0
        )
        if not should_save_model:
            return

        nn_dir = self.log_dir / "nn"
        if not nn_dir.exists():
            self.logger.warning(f"nn_dir {nn_dir} does not exist")
            return

        pth_filepaths = sorted(list(nn_dir.glob("*.pth")))

        if len(pth_filepaths) == 0:
            self.logger.warning(f"no pth files in {pth_filepaths}")
            return

        # Latest pth
        latest_pth_filepath = max(pth_filepaths, key=os.path.getctime)
        SAVE_LATEST_PTH = False
        if SAVE_LATEST_PTH:
            self._save_file_to_wandb_safe(latest_pth_filepath)

        # Best pth
        best_pth_filepath = nn_dir / "best.pth"
        self._save_file_to_wandb_safe(best_pth_filepath)

    def _save_file_to_wandb_safe(self, filepath: Path) -> None:
        if not wandb_started():
            return

        self.logger.info(f"Saving file to wandb: {filepath}")

        try:
            SAVE_FILES_IN_ROOT_DIR = False
            if SAVE_FILES_IN_ROOT_DIR:
                base_path = str(filepath.parent)
                wandb.save(
                    filepath,
                    base_path=base_path,
                )
            else:
                wandb.save(filepath)
        except Exception as e:
            self.logger.warning(f"Failed to save file to wandb: {filepath} {e}")
            self.logger.warning("Continuing...")

    ##### WANDB END #####

    ##### PRE PHYSICS STEP START #####
    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # Make sure this aligns with compute_num_actions

        # Since this is done before resetting, it we reset, the prev_raw_actions will also be reset appropriately
        if hasattr(self, "raw_actions"):
            self.prev_raw_actions = self.raw_actions.clone()
        self.raw_actions = actions.clone().to(self.device)

        # Reset
        # reset is best called in post_physics_step
        # but both work, and putting it in pre_physics_step helps
        # with the forward kinematics workaround (reset_idx_after_physics)
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(done_env_ids) > 0:
            self.update_metrics(done_env_ids)
            self.reset_idx(done_env_ids)

        # Teleport object to random position to simulate object being dropped or knocked over
        RANDOMLY_TELEPORT_OBJECT = False
        if RANDOMLY_TELEPORT_OBJECT:
            TELEPORT_PROB = 0.01

            teleport_env_ids = (
                (torch.rand(self.num_envs, device=self.device) < TELEPORT_PROB)
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )
            # NOTE: Must use self.actor_root_state instead of self.object_pos and self.object_quat_xyzw because they are not continuous tensors
            self.actor_root_state[self.object_indices[teleport_env_ids], :] = (
                self._sample_teleported_object_state(len(teleport_env_ids)).clone()
            )
            self.deferred_set_actor_root_state_tensor_indexed(
                self.object_indices[teleport_env_ids]
            )

        # Compute and set dof pos targets
        if self.custom_env_cfg.USE_FABRIC_ACTION_SPACE:
            # HACK: Manually set actions for debugging
            # if not hasattr(self, "HACK_raw_actions") or self.num_steps_taken % 50 == 0:
            #     self.HACK_raw_actions = torch_rand_float(
            #         lower=-1,
            #         upper=1,
            #         shape=(self.num_envs, self.num_actions),
            #         device=self.device,
            #     )
            #     self.HACK_raw_actions[:, 2] = -1
            # self.raw_actions = self.HACK_raw_actions

            # Actions are in robot frame, xyz, euler_ZYX
            # World: X = forward, Y = left, Z = up
            # Palm: x = palm normal, y = palm-to_thumb, z= palm-to-finger
            # 0 = forward, 1 = left, 2 = up
            # 3 = euler_Z, 4 = euler_Y, 5 = euler_X
            # self.raw_actions[:, 0] = 0.5
            # self.raw_actions[:, 1] = 0.5
            # self.raw_actions[:, 2] = -1

            # if self.progress_buf[0] > 20:
            #     self.raw_actions[:, 0] = 0.5
            #     self.raw_actions[:, 1] = 0.5
            #     self.raw_actions[:, 2] = -0.2

            # self.raw_actions[self.progress_buf > 100, 0] = -0.3
            # self.raw_actions[self.progress_buf > 100, 1] = -0.3
            # self.raw_actions[self.progress_buf > 100, 2] = 0
            # self.raw_actions[self.progress_buf > 60, :3] = rescale(
            #     values=self.goal_object_pos[self.progress_buf > 60],
            #     old_mins=self.fabric_palm_mins[:3],
            #     old_maxs=self.fabric_palm_maxs[:3],
            #     new_mins=torch.ones_like(self.fabric_palm_mins[:3]) * -1,
            #     new_maxs=torch.ones_like(self.fabric_palm_maxs[:3]) * 1,
            # )
            # self.raw_actions[self.progress_buf > 60, 3] = -0.5
            # self.raw_actions[self.progress_buf > 60, 4] = 0
            # self.raw_actions[self.progress_buf > 60, 5] = -0.25

            # Update fabric targets
            # Action is in [-1, 1] => [min, max]
            self.fabric_palm_target.copy_(
                rescale(
                    values=self.raw_actions[:, :6],
                    old_mins=torch.ones_like(self.fabric_palm_mins) * -1,
                    old_maxs=torch.ones_like(self.fabric_palm_maxs) * 1,
                    new_mins=self.fabric_palm_mins,
                    new_maxs=self.fabric_palm_maxs,
                )
            )
            self.fabric_hand_target.copy_(
                rescale(
                    values=self.raw_actions[:, 6:],
                    old_mins=torch.ones_like(self.fabric_hand_mins) * -1,
                    old_maxs=torch.ones_like(self.fabric_hand_maxs) * 1,
                    new_mins=self.fabric_hand_mins,
                    new_maxs=self.fabric_hand_maxs,
                )
            )

            # Step fabric
            self.fabric_cuda_graph.replay()
            self.fabric_q.copy_(self.fabric_q_new)
            self.fabric_qd.copy_(self.fabric_qd_new)
            self.fabric_qdd.copy_(self.fabric_qdd_new)

            right_robot_dof_pos_targets = self.fabric_q.clone()
        else:
            MODE = "relative"
            ALPHA = 0.5
            if MODE == "absolute":
                new_right_robot_dof_pos_targets = (
                    self._compute_dof_pos_targets_absolute(
                        raw_actions=self.raw_actions,
                        dof_lower_limits=self.right_robot_dof_lower_limits,
                        dof_upper_limits=self.right_robot_dof_upper_limits,
                    )
                )
            elif MODE == "relative":
                new_right_robot_dof_pos_targets = (
                    self._compute_dof_pos_targets_relative(
                        raw_actions=self.raw_actions,
                        dof_lower_limits=self.right_robot_dof_lower_limits,
                        dof_upper_limits=self.right_robot_dof_upper_limits,
                        current_dof_pos=self.right_robot_dof_pos,
                        arm_scale_fraction=0.1,
                        hand_scale_fraction=0.5,
                    )
                )
            else:
                raise ValueError(f"Invalid MODE: {MODE}")

            right_robot_dof_pos_targets = (
                ALPHA * new_right_robot_dof_pos_targets
                + (1 - ALPHA) * self.right_robot_dof_pos_targets
            )

        if REPLAY_OPEN_LOOP_TRAJECTORY:
            open_loop_qs = self.current_open_loop_qs

            if REPLAY_OPEN_LOOP_TRAJECTORY:
                right_robot_dof_pos_targets = open_loop_qs
            else:
                raise ValueError(
                    "Either REPLAY_OPEN_LOOP_TRAJECTORY or USE_RESIDUAL_POLICY must be True"
                )

        SET_DOF_POS_TARGETS = True  # Set to False to debug
        if SET_DOF_POS_TARGETS:
            self._set_dof_pos_targets(
                right_robot_dof_pos_targets=right_robot_dof_pos_targets
            )

        if self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING:
            # Update goal object pose
            reference_idxs = self.reference_motion_float_idx.long()
            reference_idxs = torch.clamp(
                reference_idxs, max=self.reference_T_C_O_list.shape[0] - 1
            )

            new_T_C_Os = self.reference_T_C_O_list[reference_idxs]

            new_T_R_Os = self._T_C_Os_to_T_R_Os(new_T_C_Os)
            new_goal_object_pos, new_goal_object_quat_xyzw = self._T_to_pos_quat_xyzw(
                new_T_R_Os
            )

            # Apply offset
            new_goal_object_pos[:, 0] += self.reference_motion_offset_x
            new_goal_object_pos[:, 1] += self.reference_motion_offset_y
            new_goal_object_quat_xyzw = quat_mul(
                new_goal_object_quat_xyzw,
                quat_xyzw_from_euler_xyz(
                    torch.zeros_like(self.reference_motion_offset_yaw),
                    torch.zeros_like(self.reference_motion_offset_yaw),
                    self.reference_motion_offset_yaw,
                ),
            )

            # NOTE: Must use self.actor_root_state instead of self.goal_object_pos and self.goal_object_quat_xyzw because they are not continuous tensors
            self.actor_root_state[
                self.goal_object_indices, START_POS_IDX:END_POS_IDX
            ] = new_goal_object_pos
            self.actor_root_state[
                self.goal_object_indices, START_QUAT_IDX:END_QUAT_IDX
            ] = new_goal_object_quat_xyzw
            self.deferred_set_actor_root_state_tensor_indexed(
                self.goal_object_indices[self.all_env_ids]
            )

        self.set_dof_state_tensor_indexed()
        self.set_actor_root_state_tensor_indexed()
        self.set_dof_position_target_tensor_indexed()
        self.set_dof_velocity_target_tensor_indexed()
        self.add_random_forces_to_force_tensor()
        self.apply_forces()

        USE_LIVE_PLOTTER = False
        if USE_LIVE_PLOTTER:
            self._update_live_plotter()

    def pre_physics_step_no_fabric(
        self,
        actions: torch.Tensor,
        set_dof_pos_targets: bool = True,  # Set to False to debug
    ) -> None:
        # Should be same as pre_physics_step, but without fabric
        right_robot_dof_pos_targets = actions

        if set_dof_pos_targets:
            self._set_dof_pos_targets(
                right_robot_dof_pos_targets=right_robot_dof_pos_targets
            )

        if self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING:
            # Update goal object pose
            reference_idxs = self.reference_motion_float_idx.long()
            reference_idxs = torch.clamp(
                reference_idxs, max=self.reference_T_C_O_list.shape[0] - 1
            )

            new_T_C_Os = self.reference_T_C_O_list[reference_idxs]

            new_T_R_Os = self._T_C_Os_to_T_R_Os(new_T_C_Os)
            new_goal_object_pos, new_goal_object_quat_xyzw = self._T_to_pos_quat_xyzw(
                new_T_R_Os
            )

            # Apply offset
            new_goal_object_pos[:, 0] += self.reference_motion_offset_x
            new_goal_object_pos[:, 1] += self.reference_motion_offset_y
            new_goal_object_quat_xyzw = quat_mul(
                new_goal_object_quat_xyzw,
                quat_xyzw_from_euler_xyz(
                    torch.zeros_like(self.reference_motion_offset_yaw),
                    torch.zeros_like(self.reference_motion_offset_yaw),
                    self.reference_motion_offset_yaw,
                ),
            )

            # NOTE: Must use self.actor_root_state instead of self.goal_object_pos and self.goal_object_quat_xyzw because they are not continuous tensors
            self.actor_root_state[
                self.goal_object_indices, START_POS_IDX:END_POS_IDX
            ] = new_goal_object_pos
            self.actor_root_state[
                self.goal_object_indices, START_QUAT_IDX:END_QUAT_IDX
            ] = new_goal_object_quat_xyzw
            self.deferred_set_actor_root_state_tensor_indexed(
                self.goal_object_indices[self.all_env_ids]
            )

        self.set_dof_state_tensor_indexed()
        self.set_actor_root_state_tensor_indexed()
        self.set_dof_position_target_tensor_indexed()
        self.set_dof_velocity_target_tensor_indexed()
        self.add_random_forces_to_force_tensor()
        self.apply_forces()

        USE_LIVE_PLOTTER = False
        if USE_LIVE_PLOTTER:
            self._update_live_plotter()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if not hasattr(self, "last_time"):
            self.last_time = time.time()
        STEP_WINDOW_SIZE = 1
        if self.num_steps_taken % STEP_WINDOW_SIZE == 0:
            print(f"Time for {STEP_WINDOW_SIZE} steps: {time.time() - self.last_time}")
            self.last_time = time.time()

        """Copy of vec_task step, but with small modifications for fabric"""
        # randomize actions
        if self.dr_randomizations.get("actions", None):
            actions = self.dr_randomizations["actions"]["noise_lambda"](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)

            # We already do this once in pre_physics_step
            # So we don't do this on the last step or else it will do it 1 too many times
            # Say control_freq_inv is 1, then we do it 1 time in pre_physics_step and 0 times here
            # Say control_freq_inv is 4, then we do it 1 time in pre_physics_step and 3 times here
            if (
                i < self.control_freq_inv - 1
                and self.custom_env_cfg.USE_FABRIC_ACTION_SPACE
            ):
                # Step fabric
                self.fabric_cuda_graph.replay()
                self.fabric_q.copy_(self.fabric_q_new)
                self.fabric_qd.copy_(self.fabric_qd_new)
                self.fabric_qdd.copy_(self.fabric_qdd_new)

                # Update right robot dof pos targets
                right_robot_dof_pos_targets = self.fabric_q.clone()
                self._set_dof_pos_targets(
                    right_robot_dof_pos_targets=right_robot_dof_pos_targets
                )

                self.set_dof_position_target_tensor_indexed()
                self.set_dof_velocity_target_tensor_indexed()

        # to fix!
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (
            self.reset_buf != 0
        )

        # randomize observations
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](
                self.obs_buf
            )

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(
            self.obs_buf, -self.clip_obs, self.clip_obs
        ).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return (
            self.obs_dict,
            self.rew_buf.to(self.rl_device),
            self.reset_buf.to(self.rl_device),
            self.extras,
        )

    def step_no_fabric(
        self,
        actions: torch.Tensor,
        set_dof_pos_targets: bool = True,  # Set to False to debug
        control_freq_inv: Optional[int] = None,
        run_post_physics_step: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Copy of vec_task step, but with small modifications for no fabric"""
        # randomize actions
        # if self.dr_randomizations.get("actions", None):
        # actions = self.dr_randomizations["actions"]["noise_lambda"](actions)

        # action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step_no_fabric(
            actions, set_dof_pos_targets=set_dof_pos_targets
        )

        # step physics and render each frame
        actual_control_freq_inv = (
            self.control_freq_inv if control_freq_inv is None else control_freq_inv
        )
        for i in range(actual_control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        if run_post_physics_step:
            self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (
            self.reset_buf != 0
        )

        # randomize observations
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](
                self.obs_buf
            )

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(
            self.obs_buf, -self.clip_obs, self.clip_obs
        ).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return (
            self.obs_dict,
            self.rew_buf.to(self.rl_device),
            self.reset_buf.to(self.rl_device),
            self.extras,
        )

    def _update_live_plotter(self) -> None:
        WEIGHTED = True
        weighted_text = "Weighted" if WEIGHTED else "Unweighted"

        if not hasattr(self, "live_plotter"):
            self.live_plotter_data = {}

            self.live_plotter_data.update(
                {
                    f"Agg {weighted_text} {reward_name}": []
                    for reward_name in self.reward_names
                }
            )
            self.live_plotter_data.update(
                {
                    f"{weighted_text} {reward_name}": []
                    for reward_name in self.reward_names
                }
            )

            from live_plotter import FastLivePlotter

            self.live_plotter = FastLivePlotter(
                n_plots=2,
                titles=[f"Agg {weighted_text} Rewards", f"{weighted_text} Rewards"],
                legends=[self.reward_names, self.reward_names],
            )

        if self.num_steps_taken == 0:
            return

        for (
            reward_name,
            individual_aggregated_rew_buf,
        ) in self.individual_aggregated_rew_bufs.items():
            self.live_plotter_data[f"Agg {weighted_text} {reward_name}"].append(
                individual_aggregated_rew_buf[self.index_to_view].item()
                * (self.reward_weight_dict[reward_name] if WEIGHTED else 1.0)
            )
        for reward_name, rew_buf in self.reward_dict.items():
            self.live_plotter_data[f"{weighted_text} {reward_name}"].append(
                rew_buf[self.index_to_view].item()
                * (self.reward_weight_dict[reward_name] if WEIGHTED else 1.0)
            )

        PLOT_EVERY_N_STEPS = 1
        if self.num_steps_taken == 0 or self.num_steps_taken % PLOT_EVERY_N_STEPS != 0:
            return

        # Store latest elements then clear
        agg_y_data = np.array(
            [
                self.live_plotter_data[f"Agg {weighted_text} {reward_name}"]
                for reward_name in self.reward_names
            ]
        ).T
        non_agg_y_data = np.array(
            [
                self.live_plotter_data[f"{weighted_text} {reward_name}"]
                for reward_name in self.reward_names
            ]
        ).T
        self.live_plotter.plot(
            y_data_list=[
                agg_y_data,
                non_agg_y_data,
            ]
        )

    def _T_C_Os_to_T_R_Os(self, T_C_Os: torch.Tensor) -> torch.Tensor:
        # Let R = robot frame = isaacgym world frame
        #     C = camera frame
        #     O = object frame

        # T_C_O, T_C_O_init from reference motion
        # T_R_C from camera extrinsics

        N = T_C_Os.shape[0]
        assert_equals(T_C_Os.shape, (N, 4, 4))
        T_R_C = torch.from_numpy(self.T_R_C_np).float().to(self.device)

        T_R_Os = torch.bmm(
            T_R_C.unsqueeze(dim=0).repeat_interleave(N, dim=0),
            T_C_Os,
        )
        return T_R_Os

    def _T_to_pos_quat_xyzw(self, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = T.shape[0]
        assert_equals(T.shape, (N, 4, 4))

        pos = T[:, :3, 3]
        quat_xyzw = matrix_to_quat_xyzw(T[:, :3, :3])
        return pos, quat_xyzw

    def _compute_dof_pos_targets_absolute(
        self,
        raw_actions: torch.Tensor,
        dof_lower_limits: torch.Tensor,
        dof_upper_limits: torch.Tensor,
    ) -> torch.Tensor:
        num_envs = raw_actions.shape[0]
        assert_equals(raw_actions.shape, (num_envs, KUKA_ALLEGRO_NUM_DOFS))
        assert_equals(dof_lower_limits.shape, (KUKA_ALLEGRO_NUM_DOFS,))
        assert_equals(dof_upper_limits.shape, (KUKA_ALLEGRO_NUM_DOFS,))
        assert torch.le(raw_actions, 1).all()
        assert torch.ge(raw_actions, -1).all()

        dof_pos_targets = scale(
            x=raw_actions,
            lower=dof_lower_limits,
            upper=dof_upper_limits,
        )

        EPS = 1e-4
        if (dof_pos_targets < dof_lower_limits - EPS).any():
            print(
                f"WARNING: dof_pos_targets < dof_lower_limits: {dof_pos_targets} < {dof_lower_limits}"
            )
        if (dof_pos_targets > dof_upper_limits + EPS).any():
            breakpoint()
            print(
                f"WARNING: dof_pos_targets > dof_upper_limits: {dof_pos_targets} > {dof_upper_limits}"
            )
        dof_pos_targets = torch.clamp(
            dof_pos_targets, min=dof_lower_limits, max=dof_upper_limits
        )
        return dof_pos_targets

    def _compute_dof_pos_targets_relative(
        self,
        raw_actions: torch.Tensor,
        dof_lower_limits: torch.Tensor,
        dof_upper_limits: torch.Tensor,
        current_dof_pos: torch.Tensor,
        arm_scale_fraction: float = 0.1,
        hand_scale_fraction: float = 0.3,
    ) -> torch.Tensor:
        num_envs = raw_actions.shape[0]
        assert_equals(raw_actions.shape, (num_envs, KUKA_ALLEGRO_NUM_DOFS))
        assert_equals(current_dof_pos.shape, (num_envs, KUKA_ALLEGRO_NUM_DOFS))
        assert_equals(dof_lower_limits.shape, (KUKA_ALLEGRO_NUM_DOFS,))
        assert_equals(dof_upper_limits.shape, (KUKA_ALLEGRO_NUM_DOFS,))
        assert torch.le(raw_actions, 1).all()
        assert torch.ge(raw_actions, -1).all()

        arm_dof_lower_limits, arm_dof_upper_limits = (
            dof_lower_limits[:KUKA_ALLEGRO_NUM_ARM_DOFS],
            dof_upper_limits[:KUKA_ALLEGRO_NUM_ARM_DOFS],
        )
        hand_dof_lower_limits, hand_dof_upper_limits = (
            dof_lower_limits[KUKA_ALLEGRO_NUM_ARM_DOFS:],
            dof_upper_limits[KUKA_ALLEGRO_NUM_ARM_DOFS:],
        )

        arm_raw_actions = raw_actions[:, :KUKA_ALLEGRO_NUM_ARM_DOFS]
        hand_raw_actions = raw_actions[:, KUKA_ALLEGRO_NUM_ARM_DOFS:]

        arm_scale = arm_scale_fraction * (arm_dof_upper_limits - arm_dof_lower_limits)
        hand_scale = hand_scale_fraction * (
            hand_dof_upper_limits - hand_dof_lower_limits
        )

        relative_arm_dof_pos_targets = arm_raw_actions * arm_scale
        relative_hand_dof_pos_targets = hand_raw_actions * hand_scale

        relative_dof_pos_targets = torch.cat(
            [relative_arm_dof_pos_targets, relative_hand_dof_pos_targets], dim=-1
        )
        assert_equals(relative_dof_pos_targets.shape, (num_envs, KUKA_ALLEGRO_NUM_DOFS))

        dof_pos_targets = current_dof_pos + relative_dof_pos_targets
        assert_equals(dof_pos_targets.shape, (num_envs, KUKA_ALLEGRO_NUM_DOFS))

        dof_pos_targets = torch.clamp(
            dof_pos_targets, min=dof_lower_limits, max=dof_upper_limits
        )

        return dof_pos_targets

    def _set_dof_pos_targets(
        self,
        right_robot_dof_pos_targets: torch.Tensor,
    ) -> None:
        self.right_robot_dof_pos_targets[:] = right_robot_dof_pos_targets
        self.right_robot_dof_vel_targets[:] = 0.0

        self.deferred_set_dof_position_target_tensor_indexed(
            self.right_robot_indices[self.all_env_ids]
        )
        self.deferred_set_dof_velocity_target_tensor_indexed(
            self.right_robot_indices[self.all_env_ids]
        )

    def add_random_forces_to_force_tensor(self) -> None:
        self.random_force = (
            self._sample_random_vector(
                prob=self.random_force_prob,
                scale=self.random_force_scale,
            )
            * self.original_object_mass[:, None]
        )
        self.applied_rb_forces[:, self.object_base_rigid_body_index, :] += (
            self.random_force
        )

        self.random_torque = (
            self._sample_random_vector(
                prob=self.random_torque_prob,
                scale=self.random_torque_scale,
            )
            * self.original_object_inertia[:, None]
        )
        self.applied_rb_torques[:, self.object_base_rigid_body_index, :] += (
            self.random_torque
        )

    def _sample_random_vector(self, prob: float, scale: float) -> torch.Tensor:
        envs_with_random_forces = torch.rand(self.num_envs, device=self.device) < prob
        random_force_direction = (
            F.normalize(
                torch_rand_float(
                    lower=-1.0,
                    upper=1.0,
                    shape=(self.num_envs, NUM_XYZ),
                    device=self.device,
                ),
                dim=-1,
            )
            * envs_with_random_forces[:, None]
        )
        assert_equals(random_force_direction.shape, (self.num_envs, NUM_XYZ))
        random_force = random_force_direction * scale
        assert_equals(random_force.shape, (self.num_envs, NUM_XYZ))
        return random_force

    def apply_forces(self) -> None:
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.applied_rb_forces),
            gymtorch.unwrap_tensor(self.applied_rb_torques),
            gymapi.GLOBAL_SPACE,
        )

        # Need to reset forces so they don't accumulate
        # But to visualize them, we need to keep them stored in prev_applied_rb_forces and prev_applied_rb_torques
        self.prev_applied_rb_forces = self.applied_rb_forces.clone()
        self.prev_applied_rb_torques = self.applied_rb_torques.clone()
        self.applied_rb_forces[:] = 0.0
        self.applied_rb_torques[:] = 0.0

    ##### PRE PHYSICS STEP END #####

    ##### POST PHYSICS STEP START #####
    def post_physics_step(self) -> None:
        self._refresh_state_tensors()

        # Important + Brittle: MUST increment this AFTER _refresh_state_tensors so that the state tensors are updated (not cached)
        # This is used by cached_with_counter_check, so it needs to be incremented after the sim step and _refresh_state_tensors
        self.num_steps_taken += 1

        # Reset state variables that could not be reset in reset_idx
        env_ids_just_reset_before_physics = (
            (self.progress_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        )
        if len(env_ids_just_reset_before_physics) > 0:
            self.reset_idx_after_physics(env_ids_just_reset_before_physics)

        self.progress_buf += 1
        self.randomize_buf += 1

        # Update RL state
        self._start_of_step_update_stored_variables()
        if USE_STATE_AS_OBSERVATION:
            self.obs_buf[:] = self.compute_states()
        else:
            self.obs_buf[:] = self.compute_observations()
        self.states_buf[:] = self.compute_states()
        (
            self.rew_buf[:],
            self.reward_matrix,
            self.weighted_reward_matrix,
            self.reward_dict,
        ) = self.compute_reward()
        self.reset_buf[:] = self.compute_reset()

        # Debug
        if self.viewer and self.enable_viewer_sync:
            self._draw_debug_info()

        self._capture_video_if_needed()

        self.populate_wandb_dict()
        self.log_wandb_dict()

        self.save_model_to_wandb()

        self._end_of_step_update_stored_variables()

        self.curriculum.update(
            success_metric=self.largest_this_episode_num_consecutive_successes_metric.get_mean().item()
        )

        if SAVE_BLENDER_TRAJECTORY:
            self._save_blender_trajectory()

    def _save_blender_trajectory(self, global_scaling: float = 1) -> None:
        """
        saved_trajectory is a dict:
        <visual_link_name>: Trajectory

        We have one <visual_link_name> for each visual component

        Each urdf has a number of links (also called rigid bodies)
        Each link has a number of visual components
        Each visual has a mesh and an associated local transform (visual frame relative to the link frame)

        On first pass, we store the mesh_path, mesh_scale, and local transform
        On subsequent passes, we store the global position and orientation of the link
        """
        if not hasattr(self, "_prepared_to_save_trajectory"):
            print("\n" + "@" * 80)
            print("Preparing to save trajectory")
            self._prepare_save_trajectory(global_scaling=global_scaling)
            print("Done preparing to save trajectory")
            print("@" * 80 + "\n")
            self._prepared_to_save_trajectory = True

        N_SECONDS = 5
        N_CONTROL_STEPS = int(N_SECONDS / self.control_dt)
        if self.num_steps_taken <= N_CONTROL_STEPS:
            self._save_trajectory_helper()
        elif self.num_steps_taken == N_CONTROL_STEPS + 1:
            output_filepath = self.log_dir / "trajectories.pkl"
            print(f"Saving trajectories to {output_filepath}")
            pickle.dump(
                {
                    link_name: trajectory.to_dict()
                    for link_name, trajectory in self.visual_link_name_to_trajectory.items()
                },
                open(output_filepath, "wb"),
            )

    def _prepare_save_trajectory(self, global_scaling: float = 1) -> None:
        # BRITTLE: Depends on having all these kept up to date with _create_envs
        actor_list = ["right_robot", "table", "goal_object", "object"]
        actor_to_handles = {
            "right_robot": self.right_robots,
            "table": self.tables,
            "goal_object": self.goal_objects,
            "object": self.objects,
        }
        actor_to_rigid_body_names = {
            "right_robot": self.gym.get_asset_rigid_body_names(self.right_robot_asset),
            "table": self.gym.get_asset_rigid_body_names(self.table_asset),
            "goal_object": self.gym.get_asset_rigid_body_names(self.goal_object_asset),
            "object": self.gym.get_asset_rigid_body_names(self.object_asset),
            "flat_box": self.gym.get_asset_rigid_body_names(self.flat_box_asset),
        }
        if INCLUDE_METAL_CYLINDER:
            actor_to_rigid_body_names["metal_cylinder"] = (
                self.gym.get_asset_rigid_body_names(self.metal_cylinder_asset)
            )
        if INCLUDE_LARGE_SAUCEPAN:
            actor_to_rigid_body_names["large_saucepan"] = (
                self.gym.get_asset_rigid_body_names(self.large_saucepan_asset)
            )
        if self.INCLUDE_DISH_RACK:
            actor_to_rigid_body_names["dishrack"] = self.gym.get_asset_rigid_body_names(
                self.dishrack_asset
            )

        actor_to_urdf_paths = {
            "right_robot": Path(KUKA_ALLEGRO_ASSET_ROOT) / KUKA_ALLEGRO_FILENAME,
            "table": Path(self.asset_root) / "table/table.urdf",
            "goal_object": self.object_urdf_path,
            "object": self.object_urdf_path,
        }

        # Sanity checks
        assert set(actor_list) == set(actor_to_rigid_body_names.keys()), (
            f"{actor_list} {actor_to_rigid_body_names.keys()}"
        )
        assert set(actor_list) == set(actor_to_urdf_paths.keys()), (
            f"{actor_list} {actor_to_urdf_paths.keys()}"
        )

        num_rigid_body_names = sum(
            len(names) for names in actor_to_rigid_body_names.values()
        )
        assert num_rigid_body_names == self.num_rigid_bodies, (
            f"{num_rigid_body_names} {self.num_rigid_bodies}"
        )

        for urdf_path in actor_to_urdf_paths.values():
            assert urdf_path.exists(), urdf_path

        # Map rigid body names to indices (to index from rigid_body_state_by_env)
        rigid_body_name_to_idx = {}
        for actor in actor_list:
            env = self.envs[self.index_to_view]
            handle = actor_to_handles[actor][self.index_to_view]

            names = actor_to_rigid_body_names[actor]
            for name in names:
                rigid_body_idx = self.gym.find_actor_rigid_body_index(
                    env, handle, name, gymapi.DOMAIN_ENV
                )
                rigid_body_name = f"{actor}/{name}"
                rigid_body_name_to_idx[rigid_body_name] = rigid_body_idx

        print(f"rigid_body_name_to_idx: {rigid_body_name_to_idx}")

        # Store mappings from visual link name to stored information
        self.visual_link_name_to_trajectory = {}
        self.visual_link_name_to_local_transform = {}
        self.visual_link_name_to_rigid_body_idx = {}
        for actor in actor_list:
            urdf_path = actor_to_urdf_paths[actor]
            urdf = URDF.load(str(urdf_path))

            for link in urdf.links:
                rigid_body_name = f"{actor}/{link.name}"
                assert rigid_body_name in rigid_body_name_to_idx, (
                    f"{rigid_body_name} {rigid_body_name_to_idx}"
                )

                if len(link.visuals) == 0:
                    continue

                for i, visual_link in enumerate(link.visuals):
                    mesh_path = urdf_path.parent / visual_link.geometry.mesh.filename
                    assert mesh_path.exists(), mesh_path

                    mesh_scale = (
                        np.array([global_scaling, global_scaling, global_scaling])
                        if visual_link.geometry.mesh.scale is None
                        else visual_link.geometry.mesh.scale * global_scaling
                    )

                    T_L_V = visual_link.origin * global_scaling

                    visual_link_name = f"{actor}/{link.name}/{i}"
                    self.visual_link_name_to_trajectory[visual_link_name] = Trajectory(
                        type="mesh",
                        mesh_path=mesh_path,
                        mesh_scale=mesh_scale,
                        frames=[],
                    )
                    self.visual_link_name_to_local_transform[visual_link_name] = T_L_V
                    self.visual_link_name_to_rigid_body_idx[visual_link_name] = (
                        rigid_body_name_to_idx[rigid_body_name]
                    )

        print(
            f"self.visual_link_name_to_rigid_body_idx = {self.visual_link_name_to_rigid_body_idx}"
        )

    def _save_trajectory_helper(self) -> None:
        for visual_link_name in self.visual_link_name_to_trajectory.keys():
            rigid_body_idx = self.visual_link_name_to_rigid_body_idx[visual_link_name]

            T_W_L = np.eye(4)
            T_W_L[:3, :3] = (
                quat_xyzw_to_matrix(
                    self.rigid_body_state_by_env[
                        self.index_to_view, rigid_body_idx, 3:7
                    ].unsqueeze(dim=0)
                )
                .squeeze(dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            T_W_L[:3, 3] = (
                self.rigid_body_state_by_env[self.index_to_view, rigid_body_idx, :3]
                .detach()
                .cpu()
                .numpy()
            )

            T_L_V = self.visual_link_name_to_local_transform[visual_link_name]
            T_W_V = T_W_L @ T_L_V

            self.visual_link_name_to_trajectory[visual_link_name].frames.append(
                Frame(
                    position=T_W_V[:3, 3].tolist(),
                    orientation=matrix_to_quat_xyzw(
                        torch.from_numpy(T_W_V[:3, :3]).float().unsqueeze(dim=0)
                    )
                    .squeeze(dim=0)
                    .tolist(),
                )
            )

    def _start_of_step_update_stored_variables(self) -> None:
        self.num_consecutive_successes = torch.where(
            self.is_in_success_region, self.num_consecutive_successes + 1, 0
        )

        self.largest_this_episode_num_consecutive_successes = torch.where(
            self.largest_this_episode_num_consecutive_successes
            > self.num_consecutive_successes,
            self.largest_this_episode_num_consecutive_successes,
            self.num_consecutive_successes,
        )

    def _end_of_step_update_stored_variables(self) -> None:
        self.aggregated_rew_buf += self.rew_buf
        for reward_name in self.reward_names:
            self.individual_aggregated_rew_bufs[reward_name] += self.reward_dict[
                reward_name
            ]
            self.individual_weighted_aggregated_rew_bufs[reward_name] += (
                self.reward_dict[reward_name] * self.reward_weight_dict[reward_name]
            )

        if self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING:
            # Update reference motion
            if STOP_REFERENCE_MOTION_BASED_ON_OBJECT_GOAL_DISTANCE:
                # Only update reference motion if the object is close to the goal
                small_object_goal_distance_ids = (
                    torch.logical_not(
                        self.object_and_goal_far_apart_need_stop_reference_motion
                    )
                    .nonzero(as_tuple=False)
                    .squeeze(-1)
                )
                self.reference_motion_float_idx[small_object_goal_distance_ids] += (
                    self.reference_motion_step_size[small_object_goal_distance_ids]
                )
            else:
                self.reference_motion_float_idx += self.reference_motion_step_size

        self.smallest_this_episode_indexfingertip_goal_distance = torch.where(
            self.smallest_this_episode_indexfingertip_goal_distance
            < self.indexfingertip_goal_distance,
            self.smallest_this_episode_indexfingertip_goal_distance,
            self.indexfingertip_goal_distance,
        )
        self.smallest_this_episode_fingertips_object_distance = torch.where(
            self.smallest_this_episode_fingertips_object_distance
            < self.fingertips_object_distance,
            self.smallest_this_episode_fingertips_object_distance,
            self.fingertips_object_distance,
        )
        self.smallest_this_episode_object_goal_distance = torch.where(
            self.smallest_this_episode_object_goal_distance < self.object_goal_distance,
            self.smallest_this_episode_object_goal_distance,
            self.object_goal_distance,
        )

        self.prev_prev_observed_object_pos = self.prev_observed_object_pos.clone()
        self.prev_prev_observed_object_quat_xyzw = (
            self.prev_observed_object_quat_xyzw.clone()
        )
        self.prev_observed_object_pos = self.observed_object_pos.clone()
        self.prev_observed_object_quat_xyzw = self.observed_object_quat_xyzw.clone()

    def _sample_observed_object_pose(
        self, object_pos: torch.Tensor, object_quat_xyzw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Uncorrelated random noise
        uncorr_pos_noise = torch_rand_float(
            lower=-self.observed_object_uncorr_pos_noise,
            upper=self.observed_object_uncorr_pos_noise,
            shape=object_pos.shape,
            device=self.device,
        )
        uncorr_rpy_noise = torch_rand_float(
            lower=-np.deg2rad(self.observed_object_uncorr_rpy_deg_noise),
            upper=np.deg2rad(self.observed_object_uncorr_rpy_deg_noise),
            shape=(self.num_envs, NUM_XYZ),
            device=self.device,
        )

        # Correlated random noise
        corr_pos_noise = self.corr_observed_object_pos_noise
        corr_rpy_noise = self.corr_observed_object_rpy_noise

        # Random pose injection noise
        random_pose_injection_ids = (
            torch.rand(self.num_envs, device=self.device)
            < self.observed_object_random_pose_injection_prob
        )
        injection_pos_noise = torch.where(
            random_pose_injection_ids[:, None],
            torch_rand_float(
                lower=-0.5,
                upper=0.5,
                shape=object_pos.shape,
                device=self.device,
            ),
            torch.zeros_like(object_pos),
        )
        injection_rpy_noise = torch.where(
            random_pose_injection_ids[:, None],
            torch_rand_float(
                lower=-np.deg2rad(90),
                upper=np.deg2rad(90),
                shape=(self.num_envs, NUM_XYZ),
                device=self.device,
            ),
            torch.zeros_like(object_pos),
        )

        observed_object_pos = (
            object_pos + uncorr_pos_noise + corr_pos_noise + injection_pos_noise
        )
        observed_object_quat_xyzw = add_rpy_noise_to_quat_xyzw(
            quat_xyzw=object_quat_xyzw,
            rpy_noise=uncorr_rpy_noise + corr_rpy_noise + injection_rpy_noise,
        )

        return observed_object_pos, observed_object_quat_xyzw

    def compute_observations(self) -> torch.Tensor:
        # Make sure this aligns with compute_num_observations
        self.observed_object_pos, self.observed_object_quat_xyzw = (
            self._sample_observed_object_pose(
                object_pos=self.object_pos, object_quat_xyzw=self.object_quat_xyzw
            )
        )

        # Must all be of shape (num_envs, ...)
        self.obs_dict = {}
        self.obs_dict["q"] = self.right_robot_dof_pos
        self.obs_dict["qd"] = self.right_robot_dof_vel
        self.obs_dict["fingertip_positions"] = (
            self.right_robot_fingertip_positions.reshape(
                self.num_envs, NUM_FINGERS * NUM_XYZ
            )
        )
        self.obs_dict["palm_pos"] = self.right_robot_palm_pos
        self.obs_dict["palm_x_pos"] = self.right_robot_palm_x_pos
        self.obs_dict["palm_y_pos"] = self.right_robot_palm_y_pos
        self.obs_dict["palm_z_pos"] = self.right_robot_palm_z_pos

        self.obs_dict["object_pos"] = self.observed_object_pos
        self.obs_dict["object_quat_xyzw"] = self.observed_object_quat_xyzw
        self.obs_dict["goal_pos"] = self.goal_object_pos
        self.obs_dict["goal_quat_xyzw"] = self.goal_object_quat_xyzw

        self.obs_dict["prev_object_pos"] = self.prev_observed_object_pos
        self.obs_dict["prev_object_quat_xyzw"] = self.prev_observed_object_quat_xyzw
        self.obs_dict["prev_prev_object_pos"] = self.prev_prev_observed_object_pos
        self.obs_dict["prev_prev_object_quat_xyzw"] = (
            self.prev_prev_observed_object_quat_xyzw
        )

        if self.custom_env_cfg.USE_FABRIC_ACTION_SPACE:
            self.obs_dict["fabric_q"] = self.fabric_q
            self.obs_dict["fabric_qd"] = self.fabric_qd

        any_infs_or_nans = False
        for key, obs in self.obs_dict.items():
            if torch.isinf(obs).any() or torch.isnan(obs).any():
                any_infs_or_nans = True
                self.logger.error(
                    f"{key}: {torch.isinf(obs).any()} or {torch.isnan(obs).any()}"
                )
                self.logger.error(f"{obs}")
        if any_infs_or_nans:
            breakpoint()

        for key, obs in self.obs_dict.items():
            assert_equals(obs.shape[0], self.num_envs)
            assert_equals(len(obs.shape), 2)

        observation = torch.cat([obs for obs in self.obs_dict.values()], dim=-1)

        if not USE_STATE_AS_OBSERVATION:
            assert_equals(observation.shape, (self.num_envs, self.num_obs))
        return observation

    def compute_states(self) -> torch.Tensor:
        # Make sure this aligns with compute_num_states
        self.state_dict = {}
        self.state_dict["obs"] = self.compute_observations()

        self.state_dict["object_keypoint_positions"] = (
            self.object_keypoint_positions.reshape(
                self.num_envs, NUM_OBJECT_KEYPOINTS * NUM_XYZ
            )
        )
        self.state_dict["goal_object_keypoint_positions"] = (
            self.goal_object_keypoint_positions.reshape(
                self.num_envs, NUM_OBJECT_KEYPOINTS * NUM_XYZ
            )
        )
        self.state_dict["object_vel"] = self.object_vel
        self.state_dict["object_angvel"] = self.object_angvel
        self.state_dict["t"] = self.progress_buf.reshape(self.num_envs, 1)
        self.state_dict["dof_force"] = self.dof_force_tensor
        self.state_dict["contact_force"] = self.net_contact_force_tensor.reshape(
            self.num_envs, self.num_rigid_bodies * NUM_XYZ
        )
        if self.force_sensor_tensor is not None:
            self.state_dict["force_sensor"] = self.force_sensor_tensor.reshape(
                self.num_envs, self.num_rigid_bodies * (NUM_XYZ + NUM_XYZ)
            )
        else:
            print("WARNING: force_sensor_tensor is None")

        any_infs_or_nans = False
        for key, state in self.state_dict.items():
            if torch.isinf(state).any() or torch.isnan(state).any():
                any_infs_or_nans = True
                self.logger.error(
                    f"{key}: {torch.isinf(state).any()} or {torch.isnan(state).any()}"
                )
                self.logger.error(f"{state}")
        if any_infs_or_nans:
            breakpoint()

        full_state = torch.cat([state for state in self.state_dict.values()], dim=-1)
        assert_equals(full_state.shape, (self.num_envs, self.num_states))
        return full_state

    def compute_reward(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        raw_actions = (
            self.raw_actions if hasattr(self, "raw_actions") else self.prev_raw_actions
        )
        reward_dict = compute_reward_jit(
            rew_buf=self.rew_buf,
            is_object_lifted=self.is_object_lifted,
            is_goal_object_lifted=self.is_goal_object_lifted,
            smallest_this_episode_indexfingertip_goal_distance=self.smallest_this_episode_indexfingertip_goal_distance,
            indexfingertip_goal_distance=self.indexfingertip_goal_distance,
            smallest_this_episode_fingertips_object_distance=self.smallest_this_episode_fingertips_object_distance,
            fingertips_object_distance=self.fingertips_object_distance,
            smallest_this_episode_object_goal_distance=self.smallest_this_episode_object_goal_distance,
            object_goal_distance=self.object_goal_distance,
            is_fingertips_object_close=self.is_fingertips_object_close,
            is_in_success_region=self.is_in_success_region,
            has_enough_consecutive_successes_to_end_episode=self.has_enough_consecutive_successes_to_end_episode,
            max_episode_length=self.max_episode_length,
            progress_buf=self.progress_buf,
            raw_actions=raw_actions,
            prev_raw_actions=self.prev_raw_actions,
            object_and_goal_far_apart_need_stop_reference_motion=self.object_and_goal_far_apart_need_stop_reference_motion,
        )

        assert set(reward_dict.keys()) == set(self.reward_names), "\n".join(
            [
                f"Only in reward_dict: {set(reward_dict.keys()) - set(self.reward_names)}",
                f"Only in self.reward_names: {set(self.reward_names) - set(reward_dict.keys())}",
            ]
        )

        any_infs_or_nans = False
        for key, reward in reward_dict.items():
            if torch.isinf(reward).any() or torch.isnan(reward).any():
                any_infs_or_nans = True
                self.logger.error(
                    f"{key}: {torch.isinf(reward).any()} or {torch.isnan(reward).any()}"
                )
                self.logger.error(f"{reward}")
        if any_infs_or_nans:
            breakpoint()

        reward_matrix = torch.stack(
            [reward_dict[name] for name in self.reward_names], dim=1
        )
        assert_equals(reward_matrix.shape, (self.num_envs, len(self.reward_names)))

        weighted_reward_matrix = reward_matrix * self.reward_weights
        total_reward = weighted_reward_matrix.sum(dim=1)
        return total_reward, reward_matrix, weighted_reward_matrix, reward_dict

    def compute_reset(self) -> torch.Tensor:
        if self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING:
            reference_motion_done = (
                self.reference_motion_float_idx.long()
                >= self.reference_T_C_O_list.shape[0]
            )
        else:
            reference_motion_done = torch.zeros_like(self.reset_buf, device=self.device)

        reset = compute_reset_jit(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            max_episode_length=self.max_episode_length,
            object_fallen_off_table=self.object_fallen_off_table,
            reference_motion_done=reference_motion_done,
            object_and_goal_far_apart_need_reset=self.object_and_goal_far_apart_need_reset,
            has_enough_consecutive_successes_to_end_episode=self.has_enough_consecutive_successes_to_end_episode,
            is_fingertips_object_close=self.is_fingertips_object_close,
            FORCE_REFERENCE_TRAJECTORY_TRACKING=self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING,
            EARLY_RESET_BASED_ON_FINGERTIPS_OBJECT_DISTANCE=EARLY_RESET_BASED_ON_FINGERTIPS_OBJECT_DISTANCE,
        )

        if REPLAY_OPEN_LOOP_TRAJECTORY:
            current_times = (
                self.progress_buf * self.control_dt * self.reference_motion_speed_factor
            )
            max_time = self.open_loop_ts[-1]
            reset = torch.zeros_like(reset)
            reset[current_times >= max_time] = True

        DEBUG_RESET = False
        if DEBUG_RESET and reset.item():
            print(f"reset: {reset}")
            print(f"progress_buf: {self.progress_buf}")
            print(f"object_fallen_off_table: {self.object_fallen_off_table}")
            print(f"reference_motion_done: {reference_motion_done}")
            print(
                f"object_and_goal_far_apart_need_reset: {self.object_and_goal_far_apart_need_reset}"
            )
            print(
                f"has_enough_consecutive_successes_to_end_episode: {self.has_enough_consecutive_successes_to_end_episode}"
            )
            print(
                f"FORCE_REFERENCE_TRAJECTORY_TRACKING: {self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING}"
            )
        return reset

    ##### POST PHYSICS STEP END #####

    ##### DEFERRED SET FUNCTIONS START #####
    def deferred_set_dof_state_tensor_indexed(
        self, object_indices: torch.Tensor
    ) -> None:
        self.set_dof_state_object_indices.append(object_indices.reshape(-1))

    def deferred_set_actor_root_state_tensor_indexed(
        self, object_indices: torch.Tensor
    ) -> None:
        self.set_actor_root_state_object_indices.append(object_indices.reshape(-1))

    def deferred_set_dof_position_target_tensor_indexed(
        self, object_indices: torch.Tensor
    ) -> None:
        self.set_dof_position_target_tensor_indices.append(object_indices.reshape(-1))

    def deferred_set_dof_velocity_target_tensor_indexed(
        self, object_indices: torch.Tensor
    ) -> None:
        self.set_dof_velocity_target_tensor_indices.append(object_indices.reshape(-1))

    def set_dof_state_tensor_indexed(self) -> None:
        object_indices = self.set_dof_state_object_indices
        if len(object_indices) == 0:
            return

        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )
        self.set_dof_state_object_indices = []

    def set_actor_root_state_tensor_indexed(self) -> None:
        object_indices = self.set_actor_root_state_object_indices
        if len(object_indices) == 0:
            return

        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.actor_root_state),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )
        self.set_actor_root_state_object_indices = []

    def set_dof_position_target_tensor_indexed(self) -> None:
        object_indices = self.set_dof_position_target_tensor_indices
        if len(object_indices) == 0:
            return

        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_pos_targets),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )
        self.set_dof_position_target_tensor_indices = []

    def set_dof_velocity_target_tensor_indexed(self) -> None:
        object_indices = self.set_dof_velocity_target_tensor_indices
        if len(object_indices) == 0:
            return

        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))
        self.gym.set_dof_velocity_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_vel_targets),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )
        self.set_dof_velocity_target_tensor_indices = []

    ##### DEFERRED SET FUNCTIONS END #####

    ##### DEBUG AND LOG START #####
    def _draw_debug_info(self) -> None:
        if self.viewer is not None:
            self.gym.clear_lines(self.viewer)

        if not self._enable_debug_viz:
            return

        for keypoint_i in range(NUM_OBJECT_KEYPOINTS):
            self._draw_debug_sphere(
                env=self.envs[self.index_to_view],
                position=gymapi.Vec3(
                    *self.object_keypoint_positions[self.index_to_view, keypoint_i, :]
                ),
                color=BLUE,
                radius=0.01,
            )
            self._draw_debug_sphere(
                env=self.envs[self.index_to_view],
                position=gymapi.Vec3(
                    *self.goal_object_keypoint_positions[
                        self.index_to_view, keypoint_i, :
                    ]
                ),
                color=GREEN,
                radius=0.01,
            )
            self._draw_debug_sphere(
                env=self.envs[self.index_to_view],
                position=gymapi.Vec3(
                    *self.observed_object_keypoint_positions[
                        self.index_to_view, keypoint_i, :
                    ]
                ),
                color=RED,
                radius=0.01,
            )

        DRAW_FINGERTIPS = False
        if DRAW_FINGERTIPS:
            for fingertip_i in range(NUM_FINGERS):
                self._draw_debug_sphere(
                    env=self.envs[self.index_to_view],
                    position=gymapi.Vec3(
                        *self.right_robot_fingertip_positions[
                            self.index_to_view, fingertip_i, :
                        ]
                    ),
                    color=BLUE,
                    radius=0.02,
                )

        DRAW_PALM_POINTS = False
        if DRAW_PALM_POINTS:
            for pos in [
                self.right_robot_palm_pos,
                self.right_robot_palm_x_pos,
                self.right_robot_palm_y_pos,
                self.right_robot_palm_z_pos,
            ]:
                self._draw_debug_sphere(
                    env=self.envs[self.index_to_view],
                    position=gymapi.Vec3(*pos[self.index_to_view].cpu().numpy()),
                    color=RED,
                    radius=0.04,
                )

        self._draw_debug_sphere(
            env=self.envs[self.index_to_view],
            position=gymapi.Vec3(
                *self.mean_right_robot_fingertip_position[self.index_to_view]
                .cpu()
                .numpy()
            ),
            color=GREEN if self.is_fingertips_object_close[self.index_to_view] else RED,
            radius=FINGERTIPS_OBJECT_CLOSE_THRESHOLD,
        )

        self._draw_horizontal_progress_bar(
            progress=self.progress_buf,
            total=self.max_episode_length,
            color=WHITE,
            above_table_m=0.6,
        )
        self._draw_horizontal_progress_bar(
            progress=self.num_consecutive_successes,
            total=self.NUM_CONSECUTIVE_SUCCESSES_TO_END_EPISODE,
            color=GREEN,
            above_table_m=0.8,
        )
        self._draw_horizontal_progress_bar(
            progress=1 - self.fingertips_object_distance,
            total=1,
            color=BLUE,
            above_table_m=1.0,
        )
        self._draw_horizontal_progress_bar(
            progress=1 - self.object_goal_distance,
            total=1,
            color=CYAN,
            above_table_m=1.2,
        )
        self._draw_horizontal_progress_bar(
            progress=1 - self.indexfingertip_goal_distance,
            total=1,
            color=RED,
            above_table_m=1.4,
        )

        if self.custom_env_cfg.USE_FABRIC_ACTION_SPACE:
            self._draw_box(
                env=self.envs[self.index_to_view],
                transform=gymapi.Transform(),
                mins=self.fabric_palm_mins[:3],
                maxs=self.fabric_palm_maxs[:3],
                color=BLUE,
            )
            self._draw_debug_sphere(
                env=self.envs[self.index_to_view],
                position=gymapi.Vec3(
                    *self.fabric_palm_target[self.index_to_view, :3].cpu().numpy()
                ),
                color=BLUE,
                radius=0.1,
            )
            palm_target_pos = (
                self.fabric_palm_target[self.index_to_view, :3].cpu().numpy()
            )
            palm_target_euler_zyx = (
                self.fabric_palm_target[self.index_to_view, 3:].cpu().numpy()
            )
            palm_target_quat_xyzw = R.from_euler(
                "ZYX", palm_target_euler_zyx, degrees=False
            ).as_quat()
            palm_target_transform = gymapi.Transform(
                p=gymapi.Vec3(*palm_target_pos),
                r=gymapi.Quat(*palm_target_quat_xyzw),
            )
            self._draw_transform(transform=palm_target_transform)

            # Draw fabric boxes
            for object_name, object_dict in self.fabric_world_dict.items():
                if object_dict["type"] != "box":
                    self.logger.info(
                        f"Skipping {object_name} of type {object_dict['type']}, only drawing boxes"
                    )
                    continue

                pos = [
                    float(x)
                    for i, x in enumerate(object_dict["transform"].split())
                    if i < 3
                ]
                quat_xyzw = [
                    float(x)
                    for i, x in enumerate(object_dict["transform"].split())
                    if i >= 3
                ]
                scaling = [float(x) for x in object_dict["scaling"].split()]
                self._draw_box(
                    env=self.envs[self.index_to_view],
                    transform=gymapi.Transform(
                        p=gymapi.Vec3(*pos),
                        r=gymapi.Quat(*quat_xyzw),
                    ),
                    mins=torch.tensor([-x / 2 for x in scaling], device=self.device),
                    maxs=torch.tensor([x / 2 for x in scaling], device=self.device),
                    color=GREEN,
                )

        # Verbose force vis
        self._draw_applied_forces()

        # Draw camera transform
        camera_transform = gymapi.Transform(
            p=gymapi.Vec3(*self.T_R_C_np[:3, 3]),
            r=gymapi.Quat(*R.from_matrix(self.T_R_C_np[:3, :3]).as_quat()),
        )

        self._draw_transform(transform=camera_transform)

        # Draw goal object transform
        goal_object_transform = gymapi.Transform(
            p=gymapi.Vec3(*self.goal_object_pos[self.index_to_view].cpu().numpy()),
            r=gymapi.Quat(
                *self.goal_object_quat_xyzw[self.index_to_view].cpu().numpy()
            ),
        )
        self._draw_transform(transform=goal_object_transform)

        # Draw palm transform
        palm_pos = self.right_robot_palm_pos[self.index_to_view].cpu().numpy()
        palm_x_pos = self.right_robot_palm_x_pos[self.index_to_view].cpu().numpy()
        palm_y_pos = self.right_robot_palm_y_pos[self.index_to_view].cpu().numpy()
        palm_z_pos = self.right_robot_palm_z_pos[self.index_to_view].cpu().numpy()
        palm_x_dir = (palm_x_pos - palm_pos) / np.linalg.norm(palm_x_pos - palm_pos)
        palm_y_dir = (palm_y_pos - palm_pos) / np.linalg.norm(palm_y_pos - palm_pos)
        palm_z_dir = (palm_z_pos - palm_pos) / np.linalg.norm(palm_z_pos - palm_pos)
        palm_R = np.stack([palm_x_dir, palm_y_dir, palm_z_dir], axis=1)
        palm_quat_xyzw = R.from_matrix(palm_R).as_quat()
        palm_transform = gymapi.Transform(
            p=gymapi.Vec3(*palm_pos),
            r=gymapi.Quat(
                *palm_quat_xyzw,
            ),
        )
        self._draw_transform(transform=palm_transform)

        # Draw lifted threshold
        self._draw_box(
            env=self.envs[self.index_to_view],
            transform=self.table_surface_pose,
            mins=torch.tensor(
                [-TABLE_LENGTH_X / 2, -TABLE_LENGTH_Y / 2, self.LIFTED_THRESHOLD],
                device=self.device,
            ),
            maxs=torch.tensor(
                [TABLE_LENGTH_X / 2, TABLE_LENGTH_Y / 2, self.LIFTED_THRESHOLD + 0.001],
                device=self.device,
            ),
            color=GREEN if self.is_object_lifted[self.index_to_view] else RED,
        )

        # Collision spheres
        DRAW_COLLISION_SPHERES = False
        if DRAW_COLLISION_SPHERES:
            collision_spheres = (
                self.right_robot_collision_spheres[self.index_to_view].cpu().numpy()
            )
            collision_sphere_radii = self.right_robot_collision_sphere_radii
            n_spheres = collision_spheres.shape[0]
            assert_equals(collision_spheres.shape, (n_spheres, NUM_XYZ))
            assert_equals(len(collision_sphere_radii), n_spheres)
            for i in range(n_spheres):
                self._draw_debug_sphere(
                    env=self.envs[self.index_to_view],
                    position=gymapi.Vec3(*collision_spheres[i, :]),
                    color=BLUE,
                    radius=collision_sphere_radii[i],
                )

    def _draw_applied_forces(self) -> None:
        object_base_pos = self.object_base_pos[self.index_to_view]
        env = self.envs[self.index_to_view]

        force = self.prev_applied_rb_forces[
            self.index_to_view, self.object_base_rigid_body_index
        ]
        clamped_force = clamp_magnitude(force, max_magnitude=1.0)
        if torch.norm(clamped_force) > 0:
            self._draw_debug_line_of_spheres(
                env=env,
                start_pos=gymapi.Vec3(*object_base_pos.cpu().numpy()),
                end_pos=gymapi.Vec3(*(object_base_pos + clamped_force).cpu().numpy()),
                color=MAGENTA,
            )

        torque = self.prev_applied_rb_torques[
            self.index_to_view, self.object_base_rigid_body_index
        ]
        EXTRA_SCALE_TO_SEE_TORQUES = 10
        clamped_torque = clamp_magnitude(
            torque * EXTRA_SCALE_TO_SEE_TORQUES, max_magnitude=1.0
        )
        if torch.norm(clamped_torque) > 0:
            self._draw_debug_line_of_spheres(
                env=env,
                start_pos=gymapi.Vec3(*object_base_pos.cpu().numpy()),
                end_pos=gymapi.Vec3(*(object_base_pos + clamped_torque).cpu().numpy()),
                color=CYAN,
            )

    def _get_color_scaled_helper(
        self, color: Tuple[float, float, float], index: float, total: float
    ) -> Tuple[float, float, float]:
        fraction = (index + 1) / (
            total + 1
        )  # avoid black and white, total=2 => (1/3, 2/3)
        scaled_color = fraction * np.array(color)
        return (scaled_color[0], scaled_color[1], scaled_color[2])

    def _draw_horizontal_progress_bar(
        self,
        progress: torch.Tensor,
        total: float,
        color: Tuple[float, float, float],
        above_table_m: float,
    ) -> None:
        env = self.envs[self.index_to_view]
        table_pos = self.table_pos[self.index_to_view]
        start_pos = table_pos + torch.tensor(
            [0.0, -TABLE_LENGTH_Y / 2, TABLE_LENGTH_Z / 2 + above_table_m],
            dtype=torch.float,
            device=self.device,
        )
        end_pos = table_pos + torch.tensor(
            [0.0, TABLE_LENGTH_Y / 2, TABLE_LENGTH_Z / 2 + above_table_m],
            dtype=torch.float,
            device=self.device,
        )
        fraction = progress[self.index_to_view] / total
        progress_pos = start_pos + fraction * (end_pos - start_pos)

        gymutil.draw_line(
            p1=gymapi.Vec3(*start_pos.cpu().numpy()),
            p2=gymapi.Vec3(*progress_pos.cpu().numpy()),
            color=gymapi.Vec3(*color),
            gym=self.gym,
            viewer=self.viewer,
            env=env,
        )

        # Vertical lines start and end
        DELTA_VERTICAL = 0.02
        delta_vertical_up = torch.tensor(
            [0.0, 0.0, DELTA_VERTICAL], dtype=torch.float, device=self.device
        )
        delta_vertical_down = torch.tensor(
            [0.0, 0.0, -DELTA_VERTICAL], dtype=torch.float, device=self.device
        )
        gymutil.draw_line(
            p1=gymapi.Vec3(*(start_pos + delta_vertical_up).cpu().numpy()),
            p2=gymapi.Vec3(*(start_pos + delta_vertical_down).cpu().numpy()),
            color=gymapi.Vec3(*color),
            gym=self.gym,
            viewer=self.viewer,
            env=env,
        )
        gymutil.draw_line(
            p1=gymapi.Vec3(*(end_pos + delta_vertical_up).cpu().numpy()),
            p2=gymapi.Vec3(*(end_pos + delta_vertical_down).cpu().numpy()),
            color=gymapi.Vec3(*color),
            gym=self.gym,
            viewer=self.viewer,
            env=env,
        )

    def _draw_vertical_progress_bar(
        self,
        progress: torch.Tensor,
        total: float,
        color: Tuple[float, float, float],
        left_table_m: float,
    ) -> None:
        env = self.envs[self.index_to_view]
        start_pos = torch.tensor(
            [0.0, left_table_m, 0.1],
            dtype=torch.float,
            device=self.device,
        )
        end_pos = torch.tensor(
            [0.0, left_table_m, 0.6],
            dtype=torch.float,
            device=self.device,
        )
        fraction = progress[self.index_to_view] / total
        progress_pos = start_pos + fraction * (end_pos - start_pos)

        gymutil.draw_line(
            p1=gymapi.Vec3(*start_pos.cpu().numpy()),
            p2=gymapi.Vec3(*progress_pos.cpu().numpy()),
            color=gymapi.Vec3(*color),
            gym=self.gym,
            viewer=self.viewer,
            env=env,
        )

        # Horizontal lines start and end
        DELTA_HORIZONTAL = 0.02
        delta_horizontal_up = torch.tensor(
            [0.0, DELTA_HORIZONTAL, 0.0], dtype=torch.float, device=self.device
        )
        delta_horizontal_down = torch.tensor(
            [0.0, -DELTA_HORIZONTAL, 0.0], dtype=torch.float, device=self.device
        )
        gymutil.draw_line(
            p1=gymapi.Vec3(*(start_pos + delta_horizontal_up).cpu().numpy()),
            p2=gymapi.Vec3(*(start_pos + delta_horizontal_down).cpu().numpy()),
            color=gymapi.Vec3(*color),
            gym=self.gym,
            viewer=self.viewer,
            env=env,
        )
        gymutil.draw_line(
            p1=gymapi.Vec3(*(end_pos + delta_horizontal_up).cpu().numpy()),
            p2=gymapi.Vec3(*(end_pos + delta_horizontal_down).cpu().numpy()),
            color=gymapi.Vec3(*color),
            gym=self.gym,
            viewer=self.viewer,
            env=env,
        )

    def _draw_box(
        self,
        env,
        transform: gymapi.Transform,
        mins: torch.Tensor,
        maxs: torch.Tensor,
        color: Tuple[float, float, float],
    ) -> None:
        assert_equals(mins.shape, (NUM_XYZ,))
        assert_equals(maxs.shape, (NUM_XYZ,))

        mins_list = mins.cpu().numpy().tolist()
        maxs_list = maxs.cpu().numpy().tolist()

        # Get positions of corners of rectangular prism
        corners = []
        for x in [mins_list[0], maxs_list[0]]:
            for y in [mins_list[1], maxs_list[1]]:
                for z in [mins_list[2], maxs_list[2]]:
                    corner = transform.transform_point(gymapi.Vec3(x, y, z))
                    corners.append(corner)

        # Draw rectangular prism using the above corners using this api
        edge_indices = [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for start_idx, end_idx in edge_indices:
            start_pos = corners[start_idx]
            end_pos = corners[end_idx]
            gymutil.draw_line(
                p1=start_pos,
                p2=end_pos,
                color=gymapi.Vec3(*color),
                gym=self.gym,
                viewer=self.viewer,
                env=env,
            )

    def _draw_debug_line_of_spheres(
        self,
        env,
        start_pos: gymapi.Vec3,
        end_pos: gymapi.Vec3,
        color: Tuple[float, float, float],
        radius: float = DEBUG_SPHERE_RADIUS,
        num_lats: int = DEBUG_NUM_LATS,
        num_lons: int = DEBUG_NUM_LONS,
        num_spheres: int = 10,
    ) -> None:
        for i in range(num_spheres):
            fraction = (i + 1) / (num_spheres + 1)
            pos = start_pos + self._scale_vec3((end_pos - start_pos), fraction)
            self._draw_debug_sphere(
                env=env,
                position=pos,
                color=color,
                radius=radius,
                num_lats=num_lats,
                num_lons=num_lons,
            )

    def _scale_vec3(self, vec: gymapi.Vec3, scale: float) -> gymapi.Vec3:
        return gymapi.Vec3(vec.x * scale, vec.y * scale, vec.z * scale)

    def _draw_debug_sphere(
        self,
        env,
        position: gymapi.Vec3,
        color: Tuple[float, float, float],
        radius: float = DEBUG_SPHERE_RADIUS,
        num_lats: int = DEBUG_NUM_LATS,
        num_lons: int = DEBUG_NUM_LONS,
    ) -> None:
        pose = gymapi.Transform(
            p=position,
        )
        sphere = gymutil.WireframeSphereGeometry(
            radius=radius,
            num_lats=num_lats,
            num_lons=num_lons,
            color=color,
        )
        gymutil.draw_lines(
            sphere,
            self.gym,
            self.viewer,
            env,
            pose,
        )

    def _draw_transform(
        self, transform: gymapi.Transform, line_length: float = 0.2
    ) -> None:
        env = self.envs[self.index_to_view]

        origin = transform.transform_point(gymapi.Vec3(0, 0, 0))
        x_dir = transform.transform_point(gymapi.Vec3(line_length, 0, 0))
        y_dir = transform.transform_point(gymapi.Vec3(0, line_length, 0))
        z_dir = transform.transform_point(gymapi.Vec3(0, 0, line_length))

        for color, dir in zip([RED, GREEN, BLUE], [x_dir, y_dir, z_dir]):
            # gymutil.draw_line(
            #     p1=origin,
            #     p2=dir,
            #     color=gymapi.Vec3(*color),  # type: ignore
            #     gym=self.gym,
            #     viewer=self.viewer,
            #     env=env,
            # )
            self._draw_debug_line_of_spheres(
                env=env,
                start_pos=origin,
                end_pos=dir,
                color=color,
            )

    def _capture_video_if_needed(self) -> None:
        if not self.log_cfg.captureVideo:
            return

        should_start_video_capture_at_start_of_next_episode = (
            self.video_frames is None
            and self.num_steps_taken % self.log_cfg.captureVideoEveryNSteps == 0
            and (self.num_steps_taken // self.log_cfg.captureVideoEveryNSteps > 0)
        )
        if should_start_video_capture_at_start_of_next_episode:
            self.logger.info("-" * 80)
            self.logger.info(
                f"At self.num_steps_taken = {self.num_steps_taken}, should start video capture at start of next episode"
            )
            self.logger.info("-" * 80)
            self.video_frames = []
            return

        should_start_video_capture_now = (
            self.video_frames is not None
            and len(self.video_frames) == 0
            and self.progress_buf[self.index_to_view].item() <= 1
        )
        video_capture_in_progress = (
            self.video_frames is not None and len(self.video_frames) > 0
        )
        if should_start_video_capture_now or video_capture_in_progress:
            self._capture_video(video_capture_in_progress)

    def _capture_video(self, video_capture_in_progress: bool) -> None:
        assert self.video_frames is not None
        if not video_capture_in_progress:
            self.logger.info("-" * 80)
            self.logger.info("Starting to capture video frames...")
            self.logger.info("-" * 80)
            self.enable_viewer_sync_before = self.enable_viewer_sync

        # Store image
        self.enable_viewer_sync = True
        self.gym.render_all_camera_sensors(self.sim)
        color_image = self.gym.get_camera_image(
            self.sim,
            self.envs[self.index_to_view],
            self.camera_handle,
            gymapi.IMAGE_COLOR,
        )
        color_image = color_image.reshape(
            self.camera_properties.height, self.camera_properties.width, NUM_RGBA
        )
        self.video_frames.append(color_image)

        if len(self.video_frames) == self.log_cfg.numVideoFrames:
            video_filename = f"{datetime_str()}_video_{self.num_steps_taken}.mp4"
            video_path = self.log_dir / video_filename
            self.logger.info("-" * 80)
            self.logger.info(f"Saving video to {video_path} ...")

            if not self.enable_viewer_sync_before:
                self.video_frames.pop(0)  # Remove first frame because it was not synced

            import imageio

            imageio.mimsave(video_path, self.video_frames)
            self.wandb_dict["video"] = wandb.Video(
                str(video_path), fps=int(1.0 / self.control_dt)
            )
            self.logger.info("DONE")
            self.logger.info("-" * 80)

            # Reset variables
            self.video_frames = None
            self.enable_viewer_sync = self.enable_viewer_sync_before

    def populate_wandb_dict(self) -> None:
        should_populate_wandb_dict = (
            self.num_steps_taken % self.log_cfg.populateWandbDictEveryNSteps == 0
        )
        if not should_populate_wandb_dict:
            return

        self.wandb_dict.update(
            {
                "num_steps_taken": self.num_steps_taken,
                "progress_buf (mean)": self.progress_buf.float().mean().item(),
                "time_elapsed (s)": time.time() - self.start_time,
                "time_elapsed (min)": (time.time() - self.start_time) / 60,
                "time_elapsed (hr)": (time.time() - self.start_time) / 3600,
            }
        )
        for reward_idx, reward_name in enumerate(self.reward_names):
            self.wandb_dict.update(
                {
                    f"Unweighted/Mean/{reward_name}": self.reward_matrix[:, reward_idx]
                    .float()
                    .mean()
                    .item(),
                    f"Weighted/Mean/{reward_name}": self.weighted_reward_matrix[
                        :, reward_idx
                    ]
                    .float()
                    .mean()
                    .item(),
                    f"Unweighted/Index_To_View/{reward_name}": self.reward_matrix[
                        self.index_to_view, reward_idx
                    ]
                    .float()
                    .item(),
                    f"Weighted/Index_To_View/{reward_name}": self.weighted_reward_matrix[
                        self.index_to_view, reward_idx
                    ]
                    .float()
                    .item(),
                }
            )
        self.wandb_dict.update(
            {
                "Weighted/Mean/Total Reward": self.rew_buf[:].float().mean().item(),
                "Weighted/Index_To_View/Total Reward": self.rew_buf[self.index_to_view]
                .float()
                .item(),
            }
        )

        self.wandb_dict.update(
            {
                "metrics/mean/reward": self.reward_metric.get_mean().item(),
                "metrics/mean/steps": self.steps_metric.get_mean().item(),
                "metrics/mean/largest_this_episode_num_consecutive_successes": self.largest_this_episode_num_consecutive_successes_metric.get_mean().item(),
                "metrics/mean/smallest_this_episode_object_goal_distance": self.smallest_this_episode_object_goal_distance_metric.get_mean().item(),
                "metrics/mean/has_enough_consecutive_successes_to_end_episode": self.has_enough_consecutive_successes_to_end_episode_metric.get_mean().item(),
            }
        )
        self.wandb_dict.update(
            {
                f"metrics/mean/{reward_name}": metric.get_mean().item()
                for reward_name, metric in self.individual_reward_metrics.items()
            }
        )
        self.wandb_dict.update(
            {
                f"metrics/mean/weighted_{reward_name}": metric.get_mean().item()
                for reward_name, metric in self.individual_weighted_reward_metrics.items()
            }
        )

        self.wandb_dict.update(
            {
                f"curriculum/{curriculum_update.variable_name}": getattr(
                    self, curriculum_update.variable_name
                )
                for curriculum_update in self.curriculum.curriculum_updater.curriculum_updates
            }
        )
        self.wandb_dict.update(
            {
                f"curriculum/{curriculum_update.variable_name}_min": curriculum_update.min
                for curriculum_update in self.curriculum.curriculum_updater.curriculum_updates
                if curriculum_update.min is not None
            }
        )
        self.wandb_dict.update(
            {
                f"curriculum/{curriculum_update.variable_name}_max": curriculum_update.max
                for curriculum_update in self.curriculum.curriculum_updater.curriculum_updates
                if curriculum_update.max is not None
            }
        )

    def update_metrics(self, done_env_ids: torch.Tensor) -> None:
        self.reward_metric.update(self.aggregated_rew_buf[done_env_ids])
        for reward_name, metric in self.individual_reward_metrics.items():
            metric.update(
                self.individual_aggregated_rew_bufs[reward_name][done_env_ids]
            )
        for reward_name, metric in self.individual_weighted_reward_metrics.items():
            metric.update(
                self.individual_weighted_aggregated_rew_bufs[reward_name][done_env_ids]
            )
        self.steps_metric.update(self.progress_buf[done_env_ids])
        self.largest_this_episode_num_consecutive_successes_metric.update(
            self.largest_this_episode_num_consecutive_successes[done_env_ids]
        )
        self.smallest_this_episode_object_goal_distance_metric.update(
            self.smallest_this_episode_object_goal_distance[done_env_ids]
        )
        self.has_enough_consecutive_successes_to_end_episode_metric.update(
            self.has_enough_consecutive_successes_to_end_episode[done_env_ids]
        )

        if self.num_steps_taken % self.log_cfg.printMetricsEveryNSteps == 0:
            self.logger.info("-" * 80)
            self.logger.info("Metrics")
            self.logger.info(f"Mean reward: {self.reward_metric.get_mean()}")
            self.logger.info(f"Mean steps: {self.steps_metric.get_mean()}")
            self.logger.info(
                f"Mean largest_this_episode_num_consecutive_successes: {self.largest_this_episode_num_consecutive_successes_metric.get_mean()}"
            )
            self.logger.info(
                f"Mean smallest_this_episode_object_goal_distance: {self.smallest_this_episode_object_goal_distance_metric.get_mean()}"
            )
            self.logger.info(
                f"Mean has_enough_consecutive_successes_to_end_episode: {self.has_enough_consecutive_successes_to_end_episode_metric.get_mean()}"
            )
            for reward_name, metric in self.individual_reward_metrics.items():
                self.logger.info(f"Mean {reward_name}: {metric.get_mean()}")
            for reward_name, metric in self.individual_weighted_reward_metrics.items():
                self.logger.info(f"Mean weighted {reward_name}: {metric.get_mean()}")
            self.logger.info("-" * 80 + "\n")

    ##### DEBUG AND LOG END #####

    ##### RESET START #####
    def reset_idx(self, reset_env_ids: torch.Tensor) -> None:
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # Reset buffers
        self.reset_buf[reset_env_ids] = 0
        self.progress_buf[reset_env_ids] = 0
        self.rew_buf[reset_env_ids] = 0

        if self.custom_env_cfg.USE_CUROBO:
            self._reset_object_and_robot_curobo(reset_env_ids)
        else:
            # Reset object
            # NOTE: Must use self.actor_root_state instead of self.object_pos and self.object_quat_xyzw because they are not continuous tensors
            self.actor_root_state[self.object_indices[reset_env_ids], :] = (
                self._sample_reset_object_state(len(reset_env_ids)).clone()
            )
            self.deferred_set_actor_root_state_tensor_indexed(
                self.object_indices[reset_env_ids]
            )

            # Reset robot
            self.right_robot_dof_pos[reset_env_ids, :] = (
                self._sample_right_robot_dof_pos(len(reset_env_ids)).clone()
            )

            self.right_robot_dof_vel[reset_env_ids, :] = torch.zeros(
                len(reset_env_ids), KUKA_ALLEGRO_NUM_DOFS, device=self.device
            )
            self.deferred_set_dof_state_tensor_indexed(
                self.right_robot_indices[reset_env_ids]
            )

        if REPLAY_OPEN_LOOP_TRAJECTORY:
            first_open_loop_q = self.open_loop_qs[0]
            assert_equals(first_open_loop_q.shape, (KUKA_ALLEGRO_NUM_DOFS,))
            self.right_robot_dof_pos[reset_env_ids, :] = first_open_loop_q[None, :]
            self.deferred_set_dof_state_tensor_indexed(
                self.right_robot_indices[reset_env_ids]
            )

        if not self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING:
            # Reset goal object
            # NOTE: Must use self.actor_root_state instead of self.goal_object_pos and self.goal_object_quat_xyzw because they are not continuous tensors
            new_goal_object_pos, new_goal_object_quat_xyzw = (
                self._sample_goal_object_pose(len(reset_env_ids))
            )
            self.actor_root_state[
                self.goal_object_indices[reset_env_ids], START_POS_IDX:END_POS_IDX
            ] = new_goal_object_pos
            self.actor_root_state[
                self.goal_object_indices[reset_env_ids], START_QUAT_IDX:END_QUAT_IDX
            ] = new_goal_object_quat_xyzw

            self.deferred_set_actor_root_state_tensor_indexed(
                self.goal_object_indices[self.all_env_ids]
            )

        # Either create or reset state variables
        if not hasattr(self, "_reset_idx_has_been_called_before"):
            self._reset_idx_has_been_called_before = True
            assert len(reset_env_ids) == self.num_envs, (
                "Must reset all envs on first reset"
            )

            self.dof_pos_targets = self.dof_pos.clone()
            self.dof_vel_targets = torch.zeros_like(self.dof_vel, device=self.device)

            self.aggregated_rew_buf = torch.zeros_like(
                self.rew_buf, device=self.device, dtype=self.rew_buf.dtype
            )
            self.individual_aggregated_rew_bufs = {
                reward_name: torch.zeros_like(
                    self.rew_buf, device=self.device, dtype=self.rew_buf.dtype
                )
                for reward_name in self.reward_names
            }
            self.individual_weighted_aggregated_rew_bufs = {
                reward_name: torch.zeros_like(
                    self.rew_buf, device=self.device, dtype=self.rew_buf.dtype
                )
                for reward_name in self.reward_names
            }
            self.applied_rb_forces = torch.zeros(
                (self.num_envs, self.num_rigid_bodies, NUM_XYZ),
                dtype=torch.float,
                device=self.device,
            )
            self.applied_rb_torques = torch.zeros(
                (self.num_envs, self.num_rigid_bodies, NUM_XYZ),
                dtype=torch.float,
                device=self.device,
            )
            self.prev_applied_rb_forces = torch.zeros_like(
                self.applied_rb_forces, device=self.device
            )
            self.prev_applied_rb_torques = torch.zeros_like(
                self.applied_rb_torques, device=self.device
            )

            self.num_consecutive_successes = torch.zeros_like(
                self.progress_buf, dtype=torch.int, device=self.device
            )
            self.largest_this_episode_num_consecutive_successes = torch.zeros_like(
                self.progress_buf, dtype=torch.int, device=self.device
            )

            if self.custom_env_cfg.USE_FABRIC_ACTION_SPACE:
                self.fabric_q = self.right_robot_dof_pos.clone()
                self.fabric_qd = torch.zeros_like(self.fabric_q, device=self.device)
                self.fabric_qdd = torch.zeros_like(self.fabric_q, device=self.device)

            self.prev_raw_actions = torch.zeros(
                (self.num_envs, self.num_actions), dtype=torch.float, device=self.device
            )

            self.reference_motion_float_idx = torch.zeros(
                (self.num_envs,),
                dtype=torch.float,
                device=self.device,
            )
            self.reference_motion_speed_factor = torch_rand_float(
                lower=REFERENCE_MIN_SPEED_FACTOR,
                upper=REFERENCE_MAX_SPEED_FACTOR,
                shape=(self.num_envs, 1),
                device=self.device,
            ).squeeze(dim=-1)
            self.reference_motion_offset_x = torch_rand_float(
                lower=-REFERENCE_MOTION_OFFSET_X,
                upper=REFERENCE_MOTION_OFFSET_X,
                shape=(self.num_envs, 1),
                device=self.device,
            ).squeeze(dim=-1)
            self.reference_motion_offset_y = torch_rand_float(
                lower=-REFERENCE_MOTION_OFFSET_Y,
                upper=REFERENCE_MOTION_OFFSET_Y,
                shape=(self.num_envs, 1),
                device=self.device,
            ).squeeze(dim=-1)
            self.reference_motion_offset_yaw = torch_rand_float(
                lower=-np.deg2rad(REFERENCE_MOTION_OFFSET_YAW_DEG),
                upper=np.deg2rad(REFERENCE_MOTION_OFFSET_YAW_DEG),
                shape=(self.num_envs, 1),
                device=self.device,
            ).squeeze(dim=-1)
        else:
            self.dof_pos_targets[reset_env_ids] = self.dof_pos[reset_env_ids].clone()
            self.dof_vel_targets[reset_env_ids] = 0

            self.aggregated_rew_buf[reset_env_ids] = 0
            for reward_name in self.reward_names:
                self.individual_aggregated_rew_bufs[reward_name][reset_env_ids] = 0
                self.individual_weighted_aggregated_rew_bufs[reward_name][
                    reset_env_ids
                ] = 0
            self.applied_rb_forces[reset_env_ids] = 0
            self.applied_rb_torques[reset_env_ids] = 0
            self.prev_applied_rb_forces[reset_env_ids] = 0
            self.prev_applied_rb_torques[reset_env_ids] = 0

            self.num_consecutive_successes[reset_env_ids] = 0
            self.largest_this_episode_num_consecutive_successes[reset_env_ids] = 0

            if self.custom_env_cfg.USE_FABRIC_ACTION_SPACE:
                self.fabric_q[reset_env_ids] = self.right_robot_dof_pos[
                    reset_env_ids
                ].clone()
                self.fabric_qd[reset_env_ids] = 0
                self.fabric_qdd[reset_env_ids] = 0

            self.prev_raw_actions[reset_env_ids] = 0
            self.reference_motion_float_idx[reset_env_ids] = 0
            self.reference_motion_speed_factor[reset_env_ids] = torch_rand_float(
                lower=REFERENCE_MIN_SPEED_FACTOR,
                upper=REFERENCE_MAX_SPEED_FACTOR,
                shape=(len(reset_env_ids), 1),
                device=self.device,
            ).squeeze(dim=-1)
            self.reference_motion_offset_x[reset_env_ids] = torch_rand_float(
                lower=-REFERENCE_MOTION_OFFSET_X,
                upper=REFERENCE_MOTION_OFFSET_X,
                shape=(len(reset_env_ids), 1),
                device=self.device,
            ).squeeze(dim=-1)
            self.reference_motion_offset_y[reset_env_ids] = torch_rand_float(
                lower=-REFERENCE_MOTION_OFFSET_Y,
                upper=REFERENCE_MOTION_OFFSET_Y,
                shape=(len(reset_env_ids), 1),
                device=self.device,
            ).squeeze(dim=-1)
            self.reference_motion_offset_yaw[reset_env_ids] = torch_rand_float(
                lower=-np.deg2rad(REFERENCE_MOTION_OFFSET_YAW_DEG),
                upper=np.deg2rad(REFERENCE_MOTION_OFFSET_YAW_DEG),
                shape=(len(reset_env_ids), 1),
                device=self.device,
            ).squeeze(dim=-1)

        self.deferred_set_dof_position_target_tensor_indexed(
            self.right_robot_indices[reset_env_ids]
        )

    def _reset_object_and_robot_curobo(self, reset_env_ids: torch.Tensor) -> None:
        # OLD VERSION: For all reset_env_ids, sample a new robot dof pos and a new object pose and check for collisions
        #              If in collision, resample
        # NEW VERSION: For all reset_env_ids, sample a new object pose then IK a new robot dof pos with a pre-manipulation pose
        start_time = time.time()
        problematic_env_ids = reset_env_ids.clone()
        right_robot_dof_pos_copy = self.right_robot_dof_pos.detach().clone()
        object_state_copy = self.object_state.detach().clone()

        # Sample object
        object_state_copy[problematic_env_ids, :] = self._sample_reset_object_state(
            len(problematic_env_ids)
        )

        ABSOLUTE_PREMANIP = self.absolute_premanipulation_pose.copy()
        T_O_P = self.relative_premanipulation_pose.copy()

        # Compute pre-manipulation poses given these object poses
        T_O_Ps = (
            torch.tensor(
                T_O_P.tolist(),
                device=self.device,
            )
            .reshape(1, 4, 4)
            .repeat_interleave(self.num_envs, dim=0)
        )
        T_R_Os = (
            torch.eye(4, device=self.device)
            .reshape(1, 4, 4)
            .repeat_interleave(self.num_envs, dim=0)
        )
        T_R_Os[:, :3, 3] = object_state_copy[:, START_POS_IDX:END_POS_IDX]
        T_R_Os[:, :3, :3] = quat_xyzw_to_matrix(
            object_state_copy[:, START_QUAT_IDX:END_QUAT_IDX]
        )
        T_R_Ps = torch.bmm(T_R_Os, T_O_Ps)

        # Update obstacle poses
        from curobo.types.math import Pose

        self.world_ccheck.update_obstacle_poses(
            name="object",
            w_obj_pose=Pose(
                position=object_state_copy[:, START_POS_IDX:END_POS_IDX],
                quaternion=object_state_copy[:, START_QUAT_IDX:END_QUAT_IDX][
                    :, [3, 0, 1, 2]
                ],
            ),
            env_idxs=self.all_env_ids,
        )

        right_robot_dof_pos_new, successes = self.solve_iks_mf_multiple_steps(
            X_W_Hs=T_R_Ps[problematic_env_ids],
            object_state=object_state_copy[problematic_env_ids],
            default_qs=torch.tensor(
                ABSOLUTE_PREMANIP.tolist(),
                device=self.device,
            )
            .reshape(1, -1)
            .repeat_interleave(len(problematic_env_ids), dim=0),
        )
        right_robot_dof_pos_copy[problematic_env_ids, :] = right_robot_dof_pos_new

        # Overwrite the hand dofs to be from the absolute pre-manipulation pose
        right_robot_dof_pos_copy[
            problematic_env_ids,
            KUKA_ALLEGRO_NUM_ARM_DOFS : KUKA_ALLEGRO_NUM_ARM_DOFS
            + KUKA_ALLEGRO_NUM_HAND_DOFS,
        ] = torch.tensor(
            ABSOLUTE_PREMANIP[
                KUKA_ALLEGRO_NUM_ARM_DOFS : KUKA_ALLEGRO_NUM_ARM_DOFS
                + KUKA_ALLEGRO_NUM_HAND_DOFS
            ].tolist(),
            device=self.device,
        )[None, ...].repeat_interleave(len(problematic_env_ids), dim=0)

        # Add noise to the IK solution
        dof_pos_noise = self._sample_right_robot_dof_pos_noise(len(problematic_env_ids))
        right_robot_dof_pos_copy[problematic_env_ids, :] += dof_pos_noise

        num_successes, num_total = (
            successes.sum(),
            successes.numel(),
        )
        self.logger.info(
            f"Success rate of IK: {num_successes} / {num_total} = {num_successes / num_total}"
        )

        CHECK_COLLISIONS = True
        if CHECK_COLLISIONS:
            # Check for collisions
            x_sph = self._get_collision_spheres(
                right_robot_dof_pos_copy.detach().clone()
            ).view(self.num_envs, self.NUM_SAMPLES_PER_ENV, -1, 4)
            d = self.world_ccheck.get_sphere_distance(
                x_sph,
                self.query_buffer,
                self.weight,
                self.act_distance,
            )
            assert d is not None
            N_SPHERES = d.shape[2]
            assert d.shape == (
                self.num_envs,
                self.NUM_SAMPLES_PER_ENV,
                N_SPHERES,
            ), d.shape
            d = d.max(dim=2).values
            assert d.shape == (self.num_envs, self.NUM_SAMPLES_PER_ENV), d.shape

            # d == 0 if not in collision, d > 0 if in collision
            num_out_of_collision, num_total = (
                (d[problematic_env_ids] == 0).sum(),
                len(problematic_env_ids),
            )
            self.logger.info(
                f"num_out_of_collision: {num_out_of_collision} / {num_total} = {num_out_of_collision / num_total}"
            )
        self.logger.info(
            f"Took {time.time() - start_time} seconds to do curobo stuff here"
        )
        # Note: We are not actually doing anything with the success or collision checking here
        # Ideally, we should resample until we get a non-colliding sample

        # Reset object
        # NOTE: Must use self.actor_root_state instead of self.object_pos and self.object_quat_xyzw because they are not continuous tensors
        self.actor_root_state[self.object_indices[reset_env_ids], :] = (
            object_state_copy[reset_env_ids, :].clone()
        )
        self.deferred_set_actor_root_state_tensor_indexed(
            self.object_indices[reset_env_ids]
        )

        # Reset robot
        self.right_robot_dof_pos[reset_env_ids, :] = right_robot_dof_pos_copy[
            reset_env_ids, :
        ].clone()
        self.right_robot_dof_vel[reset_env_ids, :] = torch.zeros(
            len(reset_env_ids), KUKA_ALLEGRO_NUM_DOFS, device=self.device
        )
        self.deferred_set_dof_state_tensor_indexed(
            self.right_robot_indices[reset_env_ids]
        )

    def reset_idx_after_physics(self, env_ids: torch.Tensor) -> None:
        # Either create or reset state variables, meant for those best stored after physics (eg. forward kinematics)
        if not hasattr(self, "_reset_idx_after_physics_has_been_called_before"):
            self._reset_idx_after_physics_has_been_called_before = True
            self.smallest_this_episode_indexfingertip_goal_distance = (
                self.fingertips_object_distance.clone()
            )
            self.smallest_this_episode_fingertips_object_distance = (
                self.fingertips_object_distance.clone()
            )
            self.smallest_this_episode_object_goal_distance = (
                self.object_goal_distance.clone()
            )

            self.observed_object_pos = self.object_pos.clone()
            self.observed_object_quat_xyzw = self.object_quat_xyzw.clone()
            self.prev_observed_object_pos = self.observed_object_pos.clone()
            self.prev_observed_object_quat_xyzw = self.observed_object_quat_xyzw.clone()
            self.prev_prev_observed_object_pos = self.prev_observed_object_pos.clone()
            self.prev_prev_observed_object_quat_xyzw = (
                self.prev_observed_object_quat_xyzw.clone()
            )
            self.corr_observed_object_pos_noise = torch_rand_float(
                lower=-self.custom_env_cfg.OBSERVED_OBJECT_CORR_POS_NOISE,
                upper=self.custom_env_cfg.OBSERVED_OBJECT_CORR_POS_NOISE,
                shape=self.observed_object_pos.shape,
                device=self.device,
            )
            self.corr_observed_object_rpy_noise = torch_rand_float(
                lower=-np.deg2rad(
                    self.custom_env_cfg.OBSERVED_OBJECT_CORR_RPY_DEG_NOISE
                ),
                upper=np.deg2rad(
                    self.custom_env_cfg.OBSERVED_OBJECT_CORR_RPY_DEG_NOISE
                ),
                shape=(self.num_envs, NUM_XYZ),
                device=self.device,
            )

        else:
            self.smallest_this_episode_indexfingertip_goal_distance[env_ids] = (
                self.fingertips_object_distance[env_ids].clone()
            )
            self.smallest_this_episode_fingertips_object_distance[env_ids] = (
                self.fingertips_object_distance[env_ids].clone()
            )
            self.smallest_this_episode_object_goal_distance[env_ids] = (
                self.object_goal_distance[env_ids].clone()
            )

            self.observed_object_pos[env_ids] = self.object_pos[env_ids].clone()
            self.observed_object_quat_xyzw[env_ids] = self.object_quat_xyzw[
                env_ids
            ].clone()
            self.prev_observed_object_pos[env_ids] = self.observed_object_pos[
                env_ids
            ].clone()
            self.prev_observed_object_quat_xyzw[env_ids] = (
                self.observed_object_quat_xyzw[env_ids].clone()
            )
            self.prev_prev_observed_object_pos[env_ids] = self.prev_observed_object_pos[
                env_ids
            ].clone()
            self.prev_prev_observed_object_quat_xyzw[env_ids] = (
                self.prev_observed_object_quat_xyzw[env_ids].clone()
            )
            self.corr_observed_object_pos_noise[env_ids] = torch_rand_float(
                lower=-self.custom_env_cfg.OBSERVED_OBJECT_CORR_POS_NOISE,
                upper=self.custom_env_cfg.OBSERVED_OBJECT_CORR_POS_NOISE,
                shape=self.observed_object_pos[env_ids].shape,
                device=self.device,
            )
            self.corr_observed_object_rpy_noise[env_ids] = torch_rand_float(
                lower=-np.deg2rad(
                    self.custom_env_cfg.OBSERVED_OBJECT_CORR_RPY_DEG_NOISE
                ),
                upper=np.deg2rad(
                    self.custom_env_cfg.OBSERVED_OBJECT_CORR_RPY_DEG_NOISE
                ),
                shape=(len(env_ids), NUM_XYZ),
                device=self.device,
            )

    def reset_all_idxs(self) -> None:
        self.reset_idx(reset_env_ids=self.all_env_ids)

        # WARNING: May need to skip this self.set_* if self.num_steps_taken > 0
        #          because on first reset, the updating may not work properly, so make sure they are init in a good spot
        #          avoid weird bugs with deferred set on first
        # if self.num_steps_taken > 0:
        if True:
            self.set_dof_state_tensor_indexed()
            self.set_actor_root_state_tensor_indexed()
            self.set_dof_position_target_tensor_indexed()
            self.set_dof_velocity_target_tensor_indexed()

    def _sample_goal_object_pose(
        self, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        goal_object_x = (
            torch_rand_float(
                lower=-0.2,
                upper=0.2,
                shape=(num_samples, 1),
                device=self.device,
            )
            + self.table_surface_pose.p.x
        )
        goal_object_y = (
            torch_rand_float(
                lower=-0.2,
                upper=0.2,
                shape=(num_samples, 1),
                device=self.device,
            )
            + self.table_surface_pose.p.y
        )
        goal_object_z = (
            torch_rand_float(
                lower=0.2,
                upper=0.5,
                shape=(num_samples, 1),
                device=self.device,
            )
            + self.table_surface_pose.p.z
        )
        new_goal_object_pos = torch.cat(
            [goal_object_x, goal_object_y, goal_object_z], dim=-1
        )

        # Fixed rotation for now
        default_quat_xyzw = (
            torch.tensor(
                [
                    self.init_object_pose.r.x,
                    self.init_object_pose.r.y,
                    self.init_object_pose.r.z,
                    self.init_object_pose.r.w,
                ],
                device=self.device,
                dtype=torch.float,
            )
            .unsqueeze(dim=0)
            .repeat_interleave(num_samples, dim=0)
        )
        rpy_noise = torch_rand_float(
            lower=-np.deg2rad(
                self.custom_env_cfg.RANDOMIZE_GOAL_OBJECT_ORIENTATION_DEG
            ),
            upper=np.deg2rad(self.custom_env_cfg.RANDOMIZE_GOAL_OBJECT_ORIENTATION_DEG),
            shape=(num_samples, NUM_XYZ),
            device=self.device,
        )
        new_goal_object_quat_xyzw = add_rpy_noise_to_quat_xyzw(
            quat_xyzw=default_quat_xyzw,
            rpy_noise=rpy_noise,
        )
        return new_goal_object_pos, new_goal_object_quat_xyzw

    def _sample_reset_object_state(self, num_samples: int) -> torch.Tensor:
        NOISE_X = self.reset_object_sample_noise_x
        NOISE_Y = self.reset_object_sample_noise_y
        NOISE_Z = self.reset_object_sample_noise_z
        NOISE_ROLL_DEG = self.reset_object_sample_noise_roll_deg
        NOISE_PITCH_DEG = self.reset_object_sample_noise_pitch_deg
        NOISE_YAW_DEG = self.reset_object_sample_noise_yaw_deg

        object_noise_x = torch_rand_float(
            lower=-NOISE_X,
            upper=NOISE_X,
            shape=(num_samples, 1),
            device=self.device,
        )
        object_noise_y = torch_rand_float(
            lower=-NOISE_Y,
            upper=NOISE_Y,
            shape=(num_samples, 1),
            device=self.device,
        )
        object_noise_z = torch_rand_float(
            lower=-NOISE_Z,
            upper=NOISE_Z,
            shape=(num_samples, 1),
            device=self.device,
        )
        object_noise_roll = torch_rand_float(
            lower=-np.deg2rad(NOISE_ROLL_DEG),
            upper=np.deg2rad(NOISE_ROLL_DEG),
            shape=(num_samples, 1),
            device=self.device,
        ).squeeze(dim=-1)
        object_noise_pitch = torch_rand_float(
            lower=-np.deg2rad(NOISE_PITCH_DEG),
            upper=np.deg2rad(NOISE_PITCH_DEG),
            shape=(num_samples, 1),
            device=self.device,
        ).squeeze(dim=-1)
        object_noise_yaw = torch_rand_float(
            lower=-np.deg2rad(NOISE_YAW_DEG),
            upper=np.deg2rad(NOISE_YAW_DEG),
            shape=(num_samples, 1),
            device=self.device,
        ).squeeze(dim=-1)

        object_noise_quat_xyzw = quat_xyzw_from_euler_xyz(
            object_noise_roll, object_noise_pitch, object_noise_yaw
        )

        object_pos = self.init_object_state[START_POS_IDX:END_POS_IDX].unsqueeze(
            dim=0
        ).repeat_interleave(num_samples, dim=0) + torch.cat(
            [object_noise_x, object_noise_y, object_noise_z], dim=-1
        )

        object_quat_xyzw = quat_mul(
            self.init_object_state[START_QUAT_IDX:END_QUAT_IDX]
            .unsqueeze(dim=0)
            .repeat_interleave(num_samples, dim=0),
            object_noise_quat_xyzw,
        )

        object_vel = (
            self.init_object_state[START_VEL_IDX:END_VEL_IDX]
            .unsqueeze(dim=0)
            .repeat_interleave(num_samples, dim=0)
        )
        object_angvel = (
            self.init_object_state[START_ANG_VEL_IDX:END_ANG_VEL_IDX]
            .unsqueeze(dim=0)
            .repeat_interleave(num_samples, dim=0)
        )

        object_state = torch.cat(
            [
                object_pos,
                object_quat_xyzw,
                object_vel,
                object_angvel,
            ],
            dim=-1,
        )
        assert_equals(object_state.shape, (num_samples, NUM_STATES))
        return object_state

    def _sample_teleported_object_state(self, num_samples: int) -> torch.Tensor:
        centered_pose = self.centered_on_table_init_object_pose
        centered_pos = torch.tensor(
            [centered_pose.p.x, centered_pose.p.y, centered_pose.p.z],
            device=self.device,
        )
        centered_quat_xyzw = torch.tensor(
            [
                centered_pose.r.x,
                centered_pose.r.y,
                centered_pose.r.z,
                centered_pose.r.w,
            ],
            device=self.device,
        )

        NOISE_X = 0.15
        NOISE_Y = 0.15
        NOISE_Z = 0
        NOISE_ROLL_DEG = 45
        NOISE_PITCH_DEG = 45
        NOISE_YAW_DEG = 45

        object_noise_x = torch_rand_float(
            lower=-NOISE_X,
            upper=NOISE_X,
            shape=(num_samples, 1),
            device=self.device,
        )
        object_noise_y = torch_rand_float(
            lower=-NOISE_Y,
            upper=NOISE_Y,
            shape=(num_samples, 1),
            device=self.device,
        )
        object_noise_z = torch_rand_float(
            lower=-NOISE_Z,
            upper=NOISE_Z,
            shape=(num_samples, 1),
            device=self.device,
        )
        object_noise_roll = torch_rand_float(
            lower=-np.deg2rad(NOISE_ROLL_DEG),
            upper=np.deg2rad(NOISE_ROLL_DEG),
            shape=(num_samples, 1),
            device=self.device,
        ).squeeze(dim=-1)
        object_noise_pitch = torch_rand_float(
            lower=-np.deg2rad(NOISE_PITCH_DEG),
            upper=np.deg2rad(NOISE_PITCH_DEG),
            shape=(num_samples, 1),
            device=self.device,
        ).squeeze(dim=-1)
        object_noise_yaw = torch_rand_float(
            lower=-np.deg2rad(NOISE_YAW_DEG),
            upper=np.deg2rad(NOISE_YAW_DEG),
            shape=(num_samples, 1),
            device=self.device,
        ).squeeze(dim=-1)

        object_noise_quat_xyzw = quat_xyzw_from_euler_xyz(
            object_noise_roll, object_noise_pitch, object_noise_yaw
        )

        object_pos = centered_pos.unsqueeze(dim=0).repeat_interleave(
            num_samples, dim=0
        ) + torch.cat([object_noise_x, object_noise_y, object_noise_z], dim=-1)

        object_quat_xyzw = quat_mul(
            centered_quat_xyzw.unsqueeze(dim=0).repeat_interleave(num_samples, dim=0),
            object_noise_quat_xyzw,
        )

        object_vel = torch.zeros((num_samples, NUM_XYZ), device=self.device)
        object_angvel = torch.zeros((num_samples, NUM_XYZ), device=self.device)

        object_state = torch.cat(
            [
                object_pos,
                object_quat_xyzw,
                object_vel,
                object_angvel,
            ],
            dim=-1,
        )
        assert_equals(object_state.shape, (num_samples, NUM_STATES))
        return object_state

    def _sample_right_robot_dof_pos(self, num_samples: int) -> torch.Tensor:
        KUKA_ALLEGRO_DOF_POS = self.absolute_premanipulation_pose.tolist()

        dof_pos = (
            torch.tensor(KUKA_ALLEGRO_DOF_POS, device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(num_samples, dim=0)
        )
        dof_pos_noise = self._sample_right_robot_dof_pos_noise(num_samples)
        dof_pos += dof_pos_noise
        return dof_pos

    def _sample_right_robot_dof_pos_noise(self, num_samples: int) -> torch.Tensor:
        NOISE_ARM_DEG = self.reset_right_robot_sample_noise_arm_deg
        NOISE_HAND_DEG = self.reset_right_robot_sample_noise_hand_deg
        arm_pos_noise = torch_rand_float(
            lower=-np.deg2rad(NOISE_ARM_DEG),
            upper=np.deg2rad(NOISE_ARM_DEG),
            shape=(num_samples, KUKA_ALLEGRO_NUM_ARM_DOFS),
            device=self.device,
        )
        hand_pos_noise = torch_rand_float(
            lower=-np.deg2rad(NOISE_HAND_DEG),
            upper=np.deg2rad(NOISE_HAND_DEG),
            shape=(num_samples, KUKA_ALLEGRO_NUM_HAND_DOFS),
            device=self.device,
        )
        return torch.cat([arm_pos_noise, hand_pos_noise], dim=-1)

    ##### RESET END #####

    ##### KEYBOARD START #####
    def _subscribe_to_keyboard_events(self) -> None:
        from human2sim2robot.sim_training.tasks.base.keyboard_shortcut import (
            KeyboardShortcut,
        )

        keyboard_shortcuts = [
            KeyboardShortcut(
                name="breakpoint",
                key=gymapi.KEY_B,
                function=self._breakpoint_callback,
            ),
            KeyboardShortcut(
                name="reset",
                key=gymapi.KEY_R,
                function=self._reset_callback,
            ),
            KeyboardShortcut(
                name="enable_debug_viz",
                key=gymapi.KEY_E,
                function=self._enable_debug_viz_callback,
            ),
            KeyboardShortcut(
                name="increase_index_to_view",
                key=gymapi.KEY_Z,
                function=self._increase_index_to_view_callback,
            ),
            KeyboardShortcut(
                name="decrease_index_to_view",
                key=gymapi.KEY_X,
                function=self._decrease_index_to_view_callback,
            ),
            KeyboardShortcut(
                name="apply_force_x_positive_to_object",
                key=gymapi.KEY_DOWN,
                function=self._apply_force_x_positive_to_object_callback,
            ),
            KeyboardShortcut(
                name="apply_force_x_negative_to_object",
                key=gymapi.KEY_UP,
                function=self._apply_force_x_negative_to_object_callback,
            ),
            KeyboardShortcut(
                name="apply_force_y_positive_to_object",
                key=gymapi.KEY_RIGHT,
                function=self._apply_force_y_positive_to_object_callback,
            ),
            KeyboardShortcut(
                name="apply_force_y_negative_to_object",
                key=gymapi.KEY_LEFT,
                function=self._apply_force_y_negative_to_object_callback,
            ),
            KeyboardShortcut(
                name="apply_force_z_positive_to_object",
                key=gymapi.KEY_PAGE_UP,
                function=self._apply_force_z_positive_to_object_callback,
            ),
            KeyboardShortcut(
                name="apply_force_z_negative_to_object",
                key=gymapi.KEY_PAGE_DOWN,
                function=self._apply_force_z_negative_to_object_callback,
            ),
        ]
        self.name_to_keyboard_shortcut_dict = {
            keyboard_shortcut.name: keyboard_shortcut
            for keyboard_shortcut in keyboard_shortcuts
        }

        self.JOINT_IDX_TO_CONTROL = 0
        self.VISUALIZE_UPPER_LIMIT = True
        self._enable_debug_viz = self.custom_env_cfg.enableDebugViz

    def _breakpoint_callback(self) -> None:
        self.logger.info("Breakpoint")
        breakpoint()

    def _reset_callback(self) -> None:
        self.logger.info("Resetting...")
        self.reset_all_idxs()

    def _enable_debug_viz_callback(self) -> None:
        self._enable_debug_viz = not self._enable_debug_viz
        self.logger.info(f"Debug viz is now {self._enable_debug_viz}")

    def _increase_index_to_view_callback(self) -> None:
        self.index_to_view += 1
        if self.index_to_view >= self.num_envs:
            self.index_to_view = 0

        self.logger.info(f"index_to_view: {self.index_to_view}")

    def _decrease_index_to_view_callback(self) -> None:
        self.index_to_view -= 1
        if self.index_to_view < 0:
            self.index_to_view = self.num_envs - 1

        self.logger.info(f"index_to_view: {self.index_to_view}")

    def _apply_force_x_positive_to_object_callback(self) -> None:
        self.logger.info("Applying force x positive to object")
        self.applied_rb_forces[:, self.object_base_rigid_body_index, 0] += (
            self.random_force_scale * self.original_object_mass
        )

    def _apply_force_x_negative_to_object_callback(self) -> None:
        self.logger.info("Applying force x negative to object")
        self.applied_rb_forces[:, self.object_base_rigid_body_index, 0] -= (
            self.random_force_scale * self.original_object_mass
        )

    def _apply_force_y_positive_to_object_callback(self) -> None:
        self.logger.info("Applying force y positive to object")
        self.applied_rb_forces[:, self.object_base_rigid_body_index, 1] += (
            self.random_force_scale * self.original_object_mass
        )

    def _apply_force_y_negative_to_object_callback(self) -> None:
        self.logger.info("Applying force y negative to object")
        self.applied_rb_forces[:, self.object_base_rigid_body_index, 1] -= (
            self.random_force_scale * self.original_object_mass
        )

    def _apply_force_z_positive_to_object_callback(self) -> None:
        self.logger.info("Applying force z positive to object")
        self.applied_rb_forces[:, self.object_base_rigid_body_index, 2] += (
            self.random_force_scale * self.original_object_mass
        )

    def _apply_force_z_negative_to_object_callback(self) -> None:
        self.logger.info("Applying force z negative to object")
        self.applied_rb_forces[:, self.object_base_rigid_body_index, 2] -= (
            self.random_force_scale * self.original_object_mass
        )

    ##### KEYBOARD END #####

    def validate_cfg(self) -> None:
        # Will cause an error if the cfg is invalid
        dict_to_dataclass(omegaconf_to_dict(self.cfg["env"]), EnvConfig)
        return

    ##### CFG PROPERTIES START #####
    @property
    def env_cfg(self) -> EnvConfig:
        return self.cfg["env"]

    @property
    def custom_env_cfg(self) -> CustomEnvConfig:
        return self.env_cfg.custom

    @property
    def log_cfg(self) -> LogConfig:
        return self.custom_env_cfg.log

    @property
    def random_forces_cfg(self) -> RandomForcesConfig:
        return self.custom_env_cfg.randomForces

    ##### CFG PROPERTIES END #####

    ##### SIMPLE PROPERTIES FROM CFG START #####
    @property
    def max_episode_length(self) -> int:
        length = self.env_cfg.maxEpisodeLength
        if self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING:
            length = int(
                self.reference_T_C_O_list.shape[0] / REFERENCE_MIN_SPEED_FACTOR
            )
        return length

    @property
    def sim_dt(self) -> float:
        return self.cfg["sim"]["dt"]

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.control_freq_inv

    @property
    def randomize(self) -> bool:
        return self.cfg["randomize"]

    @property
    def randomization_params(self) -> dict:
        return self.cfg["randomization_params"]

    @property
    @functools.lru_cache()
    def rand_freq(self) -> int:
        return (
            self.randomization_params["frequency"]
            if "frequency" in self.randomization_params
            else 1
        )

    ##### SIMPLE PROPERTIES FROM CFG END #####

    ##### TENSOR SLICE PROPERTIES START #####
    @property
    def right_robot_state(self) -> torch.Tensor:
        return self.actor_root_state[self.right_robot_indices]

    @property
    def right_robot_pos(self) -> torch.Tensor:
        return self.right_robot_state[:, START_POS_IDX:END_POS_IDX]

    @property
    def table_state(self) -> torch.Tensor:
        return self.actor_root_state[self.table_indices]

    @property
    def table_pos(self) -> torch.Tensor:
        return self.table_state[:, START_POS_IDX:END_POS_IDX]

    @property
    def right_robot_dof_pos(self) -> torch.Tensor:
        return self.dof_pos[:, :KUKA_ALLEGRO_NUM_DOFS]

    @property
    def right_robot_dof_vel(self) -> torch.Tensor:
        return self.dof_vel[:, :KUKA_ALLEGRO_NUM_DOFS]

    @property
    def right_robot_actuated_dof_pos(self) -> torch.Tensor:
        return self.right_robot_dof_pos[:, self.right_robot_actuated_dof_indices]

    @property
    def right_robot_dof_pos_targets(self) -> torch.Tensor:
        return self.dof_pos_targets[:, :KUKA_ALLEGRO_NUM_DOFS]

    @property
    def right_robot_dof_vel_targets(self) -> torch.Tensor:
        return self.dof_vel_targets[:, :KUKA_ALLEGRO_NUM_DOFS]

    def right_robot_taskmap_helper(
        self, q: torch.Tensor, qd: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = q.shape[0]
        assert_equals(q.shape, (N, KUKA_ALLEGRO_NUM_DOFS))
        assert_equals(qd.shape, (N, KUKA_ALLEGRO_NUM_DOFS))

        x, jac = self.taskmap(q, None)
        n_points = len(self.taskmap_link_names)
        assert_equals(x.shape, (N, NUM_XYZ * n_points))
        assert_equals(jac.shape, (N, NUM_XYZ * n_points, KUKA_ALLEGRO_NUM_DOFS))

        # Calculate the velocity in the task space
        xd = torch.bmm(jac, qd.unsqueeze(2)).squeeze(2)
        assert_equals(xd.shape, (N, NUM_XYZ * n_points))

        return (
            x.reshape(N, n_points, NUM_XYZ),
            xd.reshape(N, n_points, NUM_XYZ),
            jac.reshape(N, n_points, NUM_XYZ, KUKA_ALLEGRO_NUM_DOFS),
        )

    def right_robot_collision_spheres_helper(self, q: torch.Tensor) -> torch.Tensor:
        N = q.shape[0]
        assert_equals(q.shape, (N, KUKA_ALLEGRO_NUM_DOFS))
        sphere_positions, _ = self.fabric.get_taskmap("body_points")(q.detach(), None)
        sphere_positions = sphere_positions.reshape(N, -1, NUM_XYZ)
        return sphere_positions

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def right_robot_collision_spheres(self) -> torch.Tensor:
        return self.right_robot_collision_spheres_helper(q=self.right_robot_dof_pos)

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def right_robot_collision_sphere_radii(self) -> List[float]:
        body_sphere_radii = self.fabric.get_sphere_radii()
        return body_sphere_radii

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def right_robot_taskmap(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.right_robot_taskmap_helper(
            q=self.right_robot_dof_pos, qd=self.right_robot_dof_vel
        )

    @property
    def right_robot_palm_pos(self) -> torch.Tensor:
        x, _, _ = self.right_robot_taskmap
        idx = self.taskmap_link_names.index(PALM_LINK_NAME)
        return x[:, idx, :]

    @property
    def right_robot_palm_x_pos(self) -> torch.Tensor:
        x, _, _ = self.right_robot_taskmap
        idx = self.taskmap_link_names.index(PALM_X_LINK_NAME)
        return x[:, idx, :]

    @property
    def right_robot_palm_y_pos(self) -> torch.Tensor:
        x, _, _ = self.right_robot_taskmap
        idx = self.taskmap_link_names.index(PALM_Y_LINK_NAME)
        return x[:, idx, :]

    @property
    def right_robot_palm_z_pos(self) -> torch.Tensor:
        x, _, _ = self.right_robot_taskmap
        idx = self.taskmap_link_names.index(PALM_Z_LINK_NAME)
        return x[:, idx, :]

    @property
    def right_robot_fingertip_positions(self) -> torch.Tensor:
        x, _, _ = self.right_robot_taskmap
        start_idx, end_idx = self.taskmap_fingertip_link_idxs()
        return x[:, start_idx:end_idx, :]

    def taskmap_fingertip_link_idxs(self) -> Tuple[int, int]:
        idxs = [
            self.taskmap_link_names.index(fingertip_link_name)
            for fingertip_link_name in ALLEGRO_FINGERTIP_LINK_NAMES
        ]
        for i, j in zip(idxs[:-1], idxs[1:]):
            assert i + 1 == j, f"Expected monotonic increasing idxs: {idxs}"

        start_idx = idxs[0]
        end_idx = idxs[-1] + 1
        return start_idx, end_idx

    @property
    def object_state(self) -> torch.Tensor:
        return self.actor_root_state[self.object_indices]

    @property
    def object_pose(self) -> torch.Tensor:
        return self.object_state[:, START_POS_IDX:END_QUAT_IDX]

    @property
    def object_pos(self) -> torch.Tensor:
        return self.object_pose[:, START_POS_IDX:END_POS_IDX]

    @property
    def object_quat_xyzw(self) -> torch.Tensor:
        return self.object_pose[:, START_QUAT_IDX:END_QUAT_IDX]

    @property
    def object_vel(self) -> torch.Tensor:
        return self.object_state[:, START_VEL_IDX:END_VEL_IDX]

    @property
    def object_angvel(self) -> torch.Tensor:
        return self.object_state[:, START_ANG_VEL_IDX:END_ANG_VEL_IDX]

    @property
    def goal_object_state(self) -> torch.Tensor:
        return self.actor_root_state[self.goal_object_indices]

    @property
    def goal_object_pos(self) -> torch.Tensor:
        return self.goal_object_state[:, START_POS_IDX:END_POS_IDX]

    @property
    def goal_object_quat_xyzw(self) -> torch.Tensor:
        return self.goal_object_state[:, START_QUAT_IDX:END_QUAT_IDX]

    @property
    def object_fallen_off_table(self) -> torch.Tensor:
        FALLEN_THRESHOLD = -0.1
        return self.object_z_above_table < FALLEN_THRESHOLD

    @property
    def object_z_above_table(self) -> torch.Tensor:
        object_z = self.object_pos[:, 2]
        table_z = self.table_surface_pose.p.z
        return object_z - table_z

    @property
    def goal_object_z_above_table(self) -> torch.Tensor:
        goal_object_z = self.goal_object_pos[:, 2]
        table_z = self.table_surface_pose.p.z
        return goal_object_z - table_z

    @property
    def object_and_goal_far_apart_need_reset(self) -> torch.Tensor:
        return self.object_goal_distance > EARLY_RESET_OBJECT_GOAL_DISTANCE_THRESHOLD

    @property
    def object_and_goal_far_apart_need_stop_reference_motion(self) -> torch.Tensor:
        return (
            self.object_goal_distance
            > STOP_REFERENCE_MOTION_OBJECT_GOAL_DISTANCE_THRESHOLD
        )

    @property
    def reference_motion_step_size(self) -> torch.Tensor:
        return self.reference_motion_speed_factor * (
            self.control_dt / REFERENCE_MOTION_DT
        )

    ##### TENSOR SLICE PROPERTIES END #####

    ##### USEFUL CONSTANT PROPERTIES START #####
    @property
    @functools.lru_cache()
    def all_env_ids(self) -> torch.Tensor:
        return torch.ones_like(self.reset_buf).nonzero(as_tuple=False).squeeze(dim=-1)

    @property
    @functools.lru_cache()
    def init_right_robot_pose(self) -> gymapi.Transform:
        pose = gymapi.Transform()

        # WARNING: After extensive testing, we find that the Allegro hand robot in the real world
        #          is about 1.2cm lower than the simulated Allegro hand for most joint angles.
        #          This difference is severe enough to cause low-profile manipulation tasks to fail
        #          Thus, we manually offset the robot base by 1.2cm in the z-direction.
        MANUAL_OFFSET_ROBOT_Z = -0.012
        pose.p.z += MANUAL_OFFSET_ROBOT_Z
        return pose

    @property
    @functools.lru_cache()
    def init_table_pose(self) -> gymapi.Transform:
        if USE_REAL_TABLE_MESH:
            # Define the transformation matrix T as a numpy array hardcoded
            # Assign the values to the pose
            pose = gymapi.Transform()
            T = np.linalg.inv(
                np.array(
                    [
                        [
                            -9.87544368e-01,
                            -1.57333070e-01,
                            -1.55753395e-03,
                            7.91730212e-02,
                        ],
                        [
                            -9.08047145e-04,
                            -4.19989728e-03,
                            9.99990768e-01,
                            -3.65614006e-01,
                        ],
                        [
                            -1.57338159e-01,
                            9.87536666e-01,
                            4.00471907e-03,
                            5.94016453e-01,
                        ],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                )
            )
            x, y, z = T[:3, 3]
            qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()
            pose.p = gymapi.Vec3(x, y, z)
            pose.r = gymapi.Quat(qx, qy, qz, qw)
            return pose
        else:
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(TABLE_X, TABLE_Y, TABLE_Z)
            pose.r = gymapi.Quat(TABLE_QX, TABLE_QY, TABLE_QZ, TABLE_QW)
            return pose

    @property
    @functools.lru_cache()
    def table_surface_pose(self) -> gymapi.Transform:
        # Center of table surface
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(TABLE_X, TABLE_Y, TABLE_Z)
        pose.p = pose.p + gymapi.Vec3(0, 0, TABLE_LENGTH_Z / 2)

        pose.r = gymapi.Quat(TABLE_QX, TABLE_QY, TABLE_QZ, TABLE_QW)
        return pose

    @property
    @functools.lru_cache()
    def init_metal_cylinder_pose(self) -> gymapi.Transform:
        # FBC is flat box center
        # MC is metal cylinder
        # R is right robot
        _MC_LENGTH_X, MC_LENGTH_Y, MC_LENGTH_Z = 0.135, 0.135, 0.01

        X_FBC_MC = np.eye(4)
        X_FBC_MC[:3, 3] = np.array(
            [
                -BOX_LENGTH_X / 2 + 0.1535,
                -BOX_LENGTH_Y / 2 - MC_LENGTH_Y / 2,
                -BOX_LENGTH_Z / 2 + MC_LENGTH_Z / 2,
            ]
        )
        X_R_MC = X_R_FBC @ X_FBC_MC
        T = X_R_MC

        pos = T[:3, 3]
        rot = R.from_matrix(T[:3, :3])
        quat_xyzw = rot.as_quat()

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*pos)
        pose.r = gymapi.Quat(*quat_xyzw)
        return pose

    @property
    @functools.lru_cache()
    def init_dishrack_pose(self) -> gymapi.Transform:
        # From running FoundationPose on the dishrack mesh
        T = np.array(
            [
                [0.63538144, -0.77183964, -0.02353686, 0.77921412],
                [0.77214927, 0.63538973, 0.00808669, -0.27293226],
                [0.00871346, -0.0233121, 0.99969026, 0.19244207],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # MANUAL_OFFSET = np.array([0.0, 0.0, -0.03])
        MANUAL_OFFSET = np.array([0.0, 0.0, 0.0])

        pos = T[:3, 3] + MANUAL_OFFSET
        rot = R.from_matrix(T[:3, :3])
        quat_xyzw = rot.as_quat()

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*pos)
        pose.r = gymapi.Quat(*quat_xyzw)
        return pose

    @property
    @functools.lru_cache()
    def init_large_saucepan_pose(self) -> gymapi.Transform:
        # FBC is flat box center
        # LSP is large saucepan
        # R is right robot
        LSP_LENGTH_Z = 0.1

        X_FBC_LSP = np.eye(4)
        X_FBC_LSP[:3, 3] = np.array([0.0, 0.0, BOX_LENGTH_Z / 2 + LSP_LENGTH_Z / 2])
        X_FBC_LSP[:3, :3] = R.from_euler("xyz", [0, 0, 180], degrees=True).as_matrix()

        X_R_LSP = X_R_FBC @ X_FBC_LSP
        T = X_R_LSP

        pos = T[:3, 3]
        rot = R.from_matrix(T[:3, :3])
        quat_xyzw = rot.as_quat()

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*pos)
        pose.r = gymapi.Quat(*quat_xyzw)
        return pose

    @property
    @functools.lru_cache()
    def init_flat_box_pose(self) -> gymapi.Transform:
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(FLAT_BOX_X, FLAT_BOX_Y, FLAT_BOX_Z)
        pose.r = gymapi.Quat(FLAT_BOX_QX, FLAT_BOX_QY, FLAT_BOX_QZ, FLAT_BOX_QW)
        return pose

    @property
    @functools.lru_cache()
    def object_mesh(self) -> trimesh.Trimesh:
        return get_object_mesh(self.object_urdf_path)

    @property
    @functools.lru_cache()
    def init_object_pose(self) -> gymapi.Transform:
        pose = self.centered_on_table_init_object_pose
        if self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING:
            pose = self.reference_motion_init_object_pose

            # Add vertical offset to avoid table collision
            pose.p.z += 0.01

        return pose

    @property
    @functools.lru_cache()
    def centered_on_table_init_object_pose(self) -> gymapi.Transform:
        USE_HARDCODED_OFFSET = False
        if USE_HARDCODED_OFFSET:
            OBJECT_TABLE_VERTICAL_OFFSET = 0.05  # How much to init the object above the table, depends on the object and where its center is (bottom or center)
        else:
            bounds = self.object_mesh.bounds
            assert_equals(bounds.shape, (2, 3))

            if self.OBJECT_UP_DIR == "Y":
                OBJECT_TABLE_VERTICAL_OFFSET = -bounds[0, 1] + 0.01
            elif self.OBJECT_UP_DIR == "Z":
                OBJECT_TABLE_VERTICAL_OFFSET = -bounds[0, 2] + 0.01
            else:
                raise ValueError(f"Invalid OBJECT_UP_DIR: {self.OBJECT_UP_DIR}")

        OBJECT_X_OFFSET = 0
        OBJECT_Y_OFFSET = 0
        pose = gymapi.Transform()
        pose.p = self.table_surface_pose.p + gymapi.Vec3(
            OBJECT_X_OFFSET, OBJECT_Y_OFFSET, OBJECT_TABLE_VERTICAL_OFFSET
        )
        pose.r = gymapi.Quat(0, 0, 0, 1)
        return pose

    @property
    @functools.lru_cache()
    def OBJECT_UP_DIR(self) -> Literal["Y", "Z"]:
        # TODO: Modify this per object manually if it is not z up
        return "Z"

    @property
    @functools.lru_cache()
    def T_R_C_np(self) -> np.ndarray:
        # TODO: Modify this manually
        CAMERA: Literal["zed", "realsense"] = "zed"
        if CAMERA == "zed":
            return ZED_CAMERA_T_R_C_np
        elif CAMERA == "realsense":
            return REALSENSE_CAMERA_T_R_C_np
        else:
            raise ValueError(f"Invalid CAMERA: {CAMERA}")

    @property
    @functools.lru_cache()
    def LIFTED_THRESHOLD(self) -> float:
        USE_HARDCODED = False
        if USE_HARDCODED:
            LIFTED_THRESHOLD = 0.15
        else:
            extents = self.object_mesh.extents
            assert_equals(extents.shape, (3,))

            diagonal = np.linalg.norm(extents) / 2
            LIFTED_THRESHOLD = diagonal + 0.02

        return LIFTED_THRESHOLD

    @property
    @functools.lru_cache()
    def reference_motion_init_object_pose(self) -> gymapi.Transform:
        assert self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING

        init_T_C_O = self.reference_T_C_O_list[0]
        init_T_R_O = self._T_C_Os_to_T_R_Os(init_T_C_O.unsqueeze(dim=0)).squeeze(dim=0)
        init_object_pos, init_object_quat_xyzw = self._T_to_pos_quat_xyzw(
            init_T_R_O.unsqueeze(dim=0)
        )
        init_object_pos = init_object_pos.squeeze(dim=0)
        init_object_quat_xyzw = init_object_quat_xyzw.squeeze(dim=0)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*init_object_pos.cpu().numpy())
        pose.r = gymapi.Quat(*init_object_quat_xyzw.cpu().numpy())
        return pose

    @property
    @functools.lru_cache()
    def init_object_state(self) -> torch.Tensor:
        object_pose = self.init_object_pose

        state = torch.tensor(
            [
                object_pose.p.x,
                object_pose.p.y,
                object_pose.p.z,
            ]
            + [
                object_pose.r.x,
                object_pose.r.y,
                object_pose.r.z,
                object_pose.r.w,
            ]
            + [0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0],
            dtype=torch.float,
            device=self.device,
        )
        assert_equals(state.shape, (NUM_STATES,))
        return state

    @property
    @functools.lru_cache()
    def init_goal_object_pose(self) -> gymapi.Transform:
        pos, quat_xyzw = self._sample_goal_object_pose(num_samples=1)
        pos = pos.squeeze(dim=0)
        quat_xyzw = quat_xyzw.squeeze(dim=0)

        pose = gymapi.Transform(
            p=gymapi.Vec3(*pos.cpu().numpy()),
            r=gymapi.Quat(*quat_xyzw.cpu().numpy()),
        )
        return pose

    @property
    @functools.lru_cache()
    def num_rigid_bodies(self) -> int:
        _, num_rigid_bodies, _ = self.rigid_body_state_by_env.shape
        return num_rigid_bodies

    ##### USEFUL CONSTANT PROPERTIES END #####

    ##### ASSET PROPERTIES START #####
    @property
    @functools.lru_cache()
    def asset_root(self) -> str:
        asset_root = get_asset_root()
        assert asset_root.exists(), f"Asset root {asset_root} does not exist"
        return str(asset_root)

    @property
    @functools.lru_cache()
    def right_robot_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False

        asset_options.collapse_fixed_joints = (
            True  # True for simplicity, False for saving trajectories to have all links
        )
        if SAVE_BLENDER_TRAJECTORY:
            asset_options.collapse_fixed_joints = False

        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0
        asset_options.linear_damping = 0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        asset = self.gym.load_asset(
            self.sim,
            KUKA_ALLEGRO_ASSET_ROOT,
            KUKA_ALLEGRO_FILENAME,
            asset_options,
        )
        return asset

    @property
    def right_robot_num_rigid_bodies(self) -> int:
        return self.gym.get_asset_rigid_body_count(self.right_robot_asset)

    @property
    def right_robot_num_dofs(self) -> int:
        return self.gym.get_asset_dof_count(self.right_robot_asset)

    @property
    def right_robot_num_joints(self) -> int:
        return self.gym.get_asset_joint_count(self.right_robot_asset)

    @property
    def right_robot_num_actuated_dofs(self) -> int:
        return self.gym.get_asset_actuator_count(self.right_robot_asset)

    @property
    def right_robot_num_tendons(self) -> int:
        return self.gym.get_asset_tendon_count(self.right_robot_asset)

    @property
    def right_robot_num_shapes(self) -> int:
        return self.gym.get_asset_rigid_shape_count(self.right_robot_asset)

    @property
    def right_robot_rigid_body_names(self) -> List[str]:
        return self.gym.get_asset_rigid_body_names(self.right_robot_asset)

    @property
    def right_robot_dof_names(self) -> List[str]:
        return self.gym.get_asset_dof_names(self.right_robot_asset)

    @property
    def right_robot_joint_names(self) -> List[str]:
        return self.gym.get_asset_joint_names(self.right_robot_asset)

    @property
    def right_robot_actuated_dof_names(self) -> List[str]:
        return [
            self.gym.get_asset_actuator_joint_name(self.right_robot_asset, i)
            for i in range(self.right_robot_num_actuated_dofs)
        ]

    @property
    def right_robot_tendon_names(self) -> List[str]:
        return [
            self.gym.get_asset_tendon_name(self.right_robot_asset, i)
            for i in range(self.right_robot_num_tendons)
        ]

    @property
    def right_robot_joint_type_names(self) -> List[str]:
        return [
            self.gym.get_joint_type_string(
                self.gym.get_asset_joint_type(self.right_robot_asset, i)
            )
            for i in range(self.right_robot_num_joints)
        ]

    @property
    def right_robot_actuated_dof_indices(self) -> torch.Tensor:
        return to_torch(
            [
                self.gym.find_asset_dof_index(self.right_robot_asset, name)
                for name in self.right_robot_actuated_dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )

    @property
    def right_robot_dof_indices(self) -> torch.Tensor:
        return to_torch(
            [
                self.gym.find_asset_dof_index(self.right_robot_asset, name)
                for name in self.right_robot_dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )

    @property
    @functools.lru_cache()
    def desired_right_robot_dof_props(self):
        dof_props = self.gym.get_asset_dof_properties(self.right_robot_asset)

        # Sanity check
        for key in ["effort", "stiffness", "damping", "armature"]:
            assert len(dof_props[key]) == KUKA_ALLEGRO_NUM_DOFS, (
                f"For {key}, {len(dof_props[key])} != {KUKA_ALLEGRO_NUM_DOFS}"
            )

        NUM_ARM_DOFS = KUKA_ALLEGRO_NUM_ARM_DOFS
        NUM_HAND_ARM_DOFS = KUKA_ALLEGRO_NUM_DOFS
        dof_props["effort"][:NUM_ARM_DOFS] = KUKA_EFFORT
        dof_props["stiffness"][:NUM_ARM_DOFS] = KUKA_STIFFNESS
        dof_props["damping"][:NUM_ARM_DOFS] = KUKA_DAMPING

        dof_props["effort"][NUM_ARM_DOFS:NUM_HAND_ARM_DOFS].fill(ALLEGRO_EFFORT)
        dof_props["stiffness"][NUM_ARM_DOFS:NUM_HAND_ARM_DOFS].fill(ALLEGRO_STIFFNESS)
        dof_props["damping"][NUM_ARM_DOFS:NUM_HAND_ARM_DOFS].fill(ALLEGRO_DAMPING)

        dof_props["armature"][:NUM_ARM_DOFS].fill(KUKA_ARMATURE)
        dof_props["armature"][NUM_ARM_DOFS:NUM_HAND_ARM_DOFS].fill(ALLEGRO_ARMATURE)

        if DOF_FRICTION >= 0:
            dof_props["friction"].fill(DOF_FRICTION)
        return dof_props

    @property
    @functools.lru_cache()
    def table_asset(self):
        if USE_REAL_TABLE_MESH:
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.vhacd_enabled = True  # convex decomposition
            asset = self.gym.load_asset(
                self.sim,
                self.asset_root,
                "scene_mesh_cropped/model.urdf",
                asset_options,
            )
        else:
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset = self.gym.load_asset(
                self.sim, self.asset_root, "table/table.urdf", asset_options
            )
        return asset

    @property
    @functools.lru_cache()
    def table_texture(self):
        path = Path(self.asset_root) / "table/wood.png"
        texture = self.gym.create_texture_from_file(self.sim, str(path))
        return texture

    @property
    @functools.lru_cache()
    def box_texture(self):
        path = Path(self.asset_root) / "box/cardboard.png"
        texture = self.gym.create_texture_from_file(self.sim, str(path))
        return texture

    @property
    @functools.lru_cache()
    def right_robot_dof_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dof_props = self.desired_right_robot_dof_props
        dof_lower_limits = []
        dof_upper_limits = []
        for i in range(self.right_robot_num_dofs):
            dof_lower_limits.append(dof_props["lower"][i])
            dof_upper_limits.append(dof_props["upper"][i])
        dof_lower_limits = torch.tensor(dof_lower_limits, device=self.device)
        dof_upper_limits = torch.tensor(dof_upper_limits, device=self.device)
        return dof_lower_limits, dof_upper_limits

    @property
    @functools.lru_cache()
    def right_robot_dof_lower_limits(self) -> torch.Tensor:
        dof_lower_limits, _ = self.right_robot_dof_limits
        return dof_lower_limits

    @property
    @functools.lru_cache()
    def right_robot_dof_upper_limits(self) -> torch.Tensor:
        _, dof_upper_limits = self.right_robot_dof_limits
        return dof_upper_limits

    @property
    @functools.lru_cache()
    def right_robot_actuated_dof_lower_limits(self) -> torch.Tensor:
        return self.right_robot_dof_lower_limits[self.right_robot_actuated_dof_indices]

    @property
    @functools.lru_cache()
    def right_robot_actuated_dof_upper_limits(self) -> torch.Tensor:
        return self.right_robot_dof_upper_limits[self.right_robot_actuated_dof_indices]

    @property
    @functools.lru_cache()
    def object_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.vhacd_enabled = True  # convex decomposition

        # Increase resolution for objects that need it
        if "pitcher" in self.custom_env_cfg.object_urdf_path:
            asset_options.vhacd_params.resolution = 1_000_000

        asset = self.gym.load_asset(
            self.sim,
            self.asset_root,
            str(self.object_urdf_path.relative_to(self.asset_root)),
            asset_options,
        )
        return asset

    @property
    @functools.lru_cache()
    def goal_object_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset = self.gym.load_asset(
            self.sim,
            self.asset_root,
            str(self.object_urdf_path.relative_to(self.asset_root)),
            asset_options,
        )
        return asset

    @property
    @functools.lru_cache()
    def dishrack_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.vhacd_enabled = True  # convex decomposition

        # TODO: See if we need to increase resolution to make dishrack work
        # asset_options.vhacd_params.resolution = 500_000
        # asset_options.vhacd_params.max_convex_hulls = 100
        # asset_options.vhacd_params.max_num_vertices_per_ch = 100

        asset = self.gym.load_asset(
            self.sim,
            self.asset_root,
            "ikea/dishrack/dishrack.urdf",
            asset_options,
        )
        return asset

    @property
    @functools.lru_cache()
    def metal_cylinder_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset = self.gym.load_asset(
            self.sim,
            self.asset_root,
            "metal_cylinder/metal_cylinder.urdf",
            asset_options,
        )
        return asset

    @property
    @functools.lru_cache()
    def large_saucepan_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.vhacd_enabled = True  # convex decomposition

        asset = self.gym.load_asset(
            self.sim,
            self.asset_root,
            "ikea/saucepan_large/saucepan_large.urdf",
            asset_options,
        )
        return asset

    @property
    @functools.lru_cache()
    def flat_box_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset = self.gym.load_asset(
            self.sim,
            self.asset_root,
            "box/box.urdf",
            asset_options,
        )
        return asset

    @property
    @functools.lru_cache()
    def desired_object_rigid_shape_props(self):
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self.object_asset)
        assert_equals(
            len(rigid_shape_props),
            self.gym.get_asset_rigid_shape_count(self.object_asset),
        )
        for i in range(len(rigid_shape_props)):
            rigid_shape_props[i].friction = self.custom_env_cfg.object_friction

        return rigid_shape_props

    @property
    @functools.lru_cache()
    def desired_right_robot_rigid_shape_props(self):
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(
            self.right_robot_asset
        )
        assert_equals(
            len(rigid_shape_props),
            self.gym.get_asset_rigid_shape_count(self.right_robot_asset),
        )

        # Different friction for right robot normal links (low friction) and fingertips (high friction)
        for i in range(len(rigid_shape_props)):
            rigid_shape_props[i].friction = self.custom_env_cfg.right_robot_friction

        # Rigid bodies (links) are not the same as rigid shapes (collision geometries)
        # Each rigid body can have >=1 rigid shapes
        rb_names = self.gym.get_asset_rigid_body_names(self.right_robot_asset)
        rb_shape_indices = self.gym.get_asset_rigid_body_shape_indices(
            self.right_robot_asset
        )
        assert_equals(len(rb_names), len(rb_shape_indices))
        rb_name_to_shape_indices = {
            name: (x.start, x.count) for name, x in zip(rb_names, rb_shape_indices)
        }

        fingertip_names = [
            "index_link_3",
            "middle_link_3",
            "ring_link_3",
            "thumb_link_3",
        ]
        for name in fingertip_names:
            start, count = rb_name_to_shape_indices[name]
            for i in range(start, start + count):
                rigid_shape_props[i].friction = 1.5

        return rigid_shape_props

    @property
    @functools.lru_cache()
    def desired_table_rigid_shape_props(self):
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self.table_asset)
        assert_equals(
            len(rigid_shape_props),
            self.gym.get_asset_rigid_shape_count(self.table_asset),
        )
        for i in range(len(rigid_shape_props)):
            rigid_shape_props[i].friction = self.custom_env_cfg.table_friction

        return rigid_shape_props

    @property
    @functools.lru_cache()
    def desired_flat_box_rigid_shape_props(self):
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(
            self.flat_box_asset
        )
        assert_equals(
            len(rigid_shape_props),
            self.gym.get_asset_rigid_shape_count(self.flat_box_asset),
        )
        for i in range(len(rigid_shape_props)):
            rigid_shape_props[i].friction = 0.3
        return rigid_shape_props

    @property
    @functools.lru_cache()
    def INCLUDE_DISH_RACK(self) -> bool:
        if "plate" in self.custom_env_cfg.object_urdf_path:
            return True
        return False

    @property
    @functools.lru_cache()
    def object_urdf_path(self) -> Path:
        # HACK: When specifying the object_urdf_path, it feels "natural" to specify it with respect to the repo root dir
        # However, we actually ned to specify it with respect to the asset root dir sometimes (e.g., when loading the object asset)
        # Thus, we do some simple checks to make sure we know where the object urdf path is
        path1 = get_asset_root() / self.custom_env_cfg.object_urdf_path
        path2 = get_repo_root_dir() / self.custom_env_cfg.object_urdf_path
        assert path1.exists() or path2.exists(), (
            f"Object urdf path {self.custom_env_cfg.object_urdf_path} does not exist"
        )
        if path1.exists():
            return path1
        return path2

    ##### ASSET PROPERTIES END #####

    ##### ACTOR RIGID BODY INDEX PROPERTIES START #####
    @property
    @functools.lru_cache()
    def object_base_rigid_body_index(self) -> int:
        object_base_rb_name = self.gym.get_asset_rigid_body_names(self.object_asset)[0]

        arbitrary_idx = 0
        env = self.envs[arbitrary_idx]
        actor = self.objects[arbitrary_idx]
        return self.gym.find_actor_rigid_body_index(
            env,
            actor,
            object_base_rb_name,
            gymapi.DOMAIN_ENV,
        )

    @property
    def object_base_state(self) -> torch.Tensor:
        return self.rigid_body_state_by_env[:, self.object_base_rigid_body_index]

    @property
    def object_base_pos(self) -> torch.Tensor:
        return self.object_base_state[:, START_POS_IDX:END_POS_IDX]

    ##### ACTOR RIGID BODY INDEX PROPERTIES END #####

    ##### FORCE SENSOR PROPERTIES START #####
    @property
    def force_sensor_properties(self) -> gymapi.ForceSensorProperties:
        # Use default force sensor properties for now
        force_sensor_properties = gymapi.ForceSensorProperties()
        force_sensor_properties.enable_forward_dynamics_forces = True
        force_sensor_properties.enable_constraint_solver_forces = True
        force_sensor_properties.use_world_frame = True
        return force_sensor_properties

    @property
    def force_sensor_pose(self) -> gymapi.Transform:
        # Use default force sensor pose for now
        force_sensor_pose = gymapi.Transform()
        return force_sensor_pose

    ##### FORCE SENSOR PROPERTIES END #####

    ##### KEYPOINT POSITIONS START #####
    @property
    @functools.lru_cache()
    def object_keypoint_offsets(self) -> torch.Tensor:
        offsets = OBJECT_KEYPOINT_OFFSETS

        # Set keypoints to be yaw invariant for the ones that need it
        if "plate" in self.custom_env_cfg.object_urdf_path:
            offsets = OBJECT_KEYPOINT_OFFSETS_YAW_INVARIANT

        keypoint_offsets = to_torch(
            offsets,
            device=self.device,
            dtype=torch.float,
        )
        assert_equals(keypoint_offsets.shape, (NUM_OBJECT_KEYPOINTS, NUM_XYZ))

        N = self.num_envs
        repeated = keypoint_offsets.unsqueeze(0).repeat(N, 1, 1)
        assert_equals(repeated.shape, (N, NUM_OBJECT_KEYPOINTS, NUM_XYZ))
        return repeated

    @property
    @functools.lru_cache()
    def object_keypoint_offsets_rot_invariant(self) -> torch.Tensor:
        keypoint_offsets = to_torch(
            OBJECT_KEYPOINT_OFFSETS_ROT_INVARIANT,
            device=self.device,
            dtype=torch.float,
        )
        assert_equals(keypoint_offsets.shape, (NUM_OBJECT_KEYPOINTS, NUM_XYZ))

        N = self.num_envs
        repeated = keypoint_offsets.unsqueeze(0).repeat(N, 1, 1)
        assert_equals(repeated.shape, (N, NUM_OBJECT_KEYPOINTS, NUM_XYZ))
        return repeated

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def object_keypoint_positions(self) -> torch.Tensor:
        keypoint_positions = compute_keypoint_positions(
            pos=self.object_pos,
            quat_xyzw=self.object_quat_xyzw,
            keypoint_offsets=self.object_keypoint_offsets,
        )
        return keypoint_positions

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def goal_object_keypoint_positions(self) -> torch.Tensor:
        keypoint_positions = compute_keypoint_positions(
            pos=self.goal_object_pos,
            quat_xyzw=self.goal_object_quat_xyzw,
            keypoint_offsets=self.object_keypoint_offsets,
        )
        return keypoint_positions

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def observed_object_keypoint_positions(self) -> torch.Tensor:
        keypoint_positions = compute_keypoint_positions(
            pos=self.observed_object_pos,
            quat_xyzw=self.observed_object_quat_xyzw,
            keypoint_offsets=self.object_keypoint_offsets,
        )
        return keypoint_positions

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def object_keypoint_positions_rot_invariant(self) -> torch.Tensor:
        keypoint_positions = compute_keypoint_positions(
            pos=self.object_pos,
            quat_xyzw=self.object_quat_xyzw,
            keypoint_offsets=self.object_keypoint_offsets_rot_invariant,
        )
        return keypoint_positions

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def goal_object_keypoint_positions_rot_invariant(self) -> torch.Tensor:
        keypoint_positions = compute_keypoint_positions(
            pos=self.goal_object_pos,
            quat_xyzw=self.goal_object_quat_xyzw,
            keypoint_offsets=self.object_keypoint_offsets_rot_invariant,
        )
        return keypoint_positions

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def observed_object_keypoint_positions_rot_invariant(self) -> torch.Tensor:
        keypoint_positions = compute_keypoint_positions(
            pos=self.observed_object_pos,
            quat_xyzw=self.observed_object_quat_xyzw,
            keypoint_offsets=self.object_keypoint_offsets_rot_invariant,
        )
        return keypoint_positions

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def object_goal_distance(self) -> torch.Tensor:
        if self.custom_env_cfg.OBJECT_ORIENTATION_MATTERS:
            object_keypoint_positions = self.object_keypoint_positions
            goal_object_keypoint_positions = self.goal_object_keypoint_positions
        else:
            object_keypoint_positions = self.object_keypoint_positions_rot_invariant
            goal_object_keypoint_positions = (
                self.goal_object_keypoint_positions_rot_invariant
            )

        return torch.mean(
            torch.norm(
                object_keypoint_positions - goal_object_keypoint_positions,
                dim=-1,
            ),
            dim=-1,
        )

    ##### KEYPOINT POSITIONS END #####

    ##### CACHED PROPERTIES WITH INVALIDATION START #####
    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def right_robot_indexfingertip_pos(self) -> torch.Tensor:
        return self.right_robot_fingertip_positions[:, INDEX_FINGER_IDX]

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def indexfingertip_goal_distance(self) -> torch.Tensor:
        return torch.norm(
            self.right_robot_indexfingertip_pos - self.goal_object_pos, dim=-1
        )

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def mean_right_robot_fingertip_position(self) -> torch.Tensor:
        return self.right_robot_fingertip_positions.mean(dim=1)

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def fingertips_object_distance(self) -> torch.Tensor:
        return (
            (self.right_robot_fingertip_positions - self.object_pos.unsqueeze(dim=1))
            .norm(dim=-1)
            .mean(dim=-1)
        )

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def is_fingertips_object_close(self) -> torch.Tensor:
        return self.fingertips_object_distance < FINGERTIPS_OBJECT_CLOSE_THRESHOLD

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def is_object_lifted(self) -> torch.Tensor:
        return self.object_z_above_table > self.LIFTED_THRESHOLD

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def is_goal_object_lifted(self) -> torch.Tensor:
        return self.goal_object_z_above_table > self.LIFTED_THRESHOLD

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def is_in_success_region(self) -> torch.Tensor:
        return self.object_goal_distance < self.custom_env_cfg.SUCCESS_REGION_RADIUS

    @property
    def NUM_CONSECUTIVE_SUCCESSES_TO_END_EPISODE(self) -> int:
        CONSECUTIVE_SECONDS = 3
        TIMESTEPS_PER_SECOND = 1 / self.control_dt
        return int(CONSECUTIVE_SECONDS * TIMESTEPS_PER_SECOND)

    @cached_property_with_invalidation(CACHED_PROPERTY_INVALIDATION_VARIABLE_NAME)
    def has_enough_consecutive_successes_to_end_episode(self) -> torch.Tensor:
        has_enough_consecutive_successes_to_end_episode = (
            self.num_consecutive_successes
            >= self.NUM_CONSECUTIVE_SUCCESSES_TO_END_EPISODE
        )
        if self.custom_env_cfg.FORCE_REFERENCE_TRAJECTORY_TRACKING:
            has_enough_consecutive_successes_to_end_episode = torch.zeros_like(
                has_enough_consecutive_successes_to_end_episode,
                dtype=torch.bool,
                device=self.device,
            )
        return has_enough_consecutive_successes_to_end_episode

    @property
    def current_open_loop_qs(self) -> torch.Tensor:
        assert hasattr(self, "open_loop_qs"), "Open loop trajectory not loaded"

        # Only works if we read in the open loop trajectory filepath
        current_times = (
            self.progress_buf * self.control_dt * self.reference_motion_speed_factor
        )
        current_idxs = torch.searchsorted(self.open_loop_ts, current_times)
        current_idxs = torch.clamp(current_idxs, max=self.open_loop_ts.shape[0] - 1)
        return self.open_loop_qs[current_idxs]

    @property
    def current_open_loop_fingertip_positions(self) -> torch.Tensor:
        assert hasattr(self, "open_loop_qs"), "Open loop trajectory not loaded"

        open_loop_qs = self.current_open_loop_qs
        assert_equals(open_loop_qs.shape, (self.num_envs, KUKA_ALLEGRO_NUM_DOFS))
        current_open_loop_taskmap, _, _ = self.right_robot_taskmap_helper(
            q=open_loop_qs, qd=torch.zeros_like(open_loop_qs)
        )
        start_idx, end_idx = self.taskmap_fingertip_link_idxs()
        current_open_loop_fingertip_positions = current_open_loop_taskmap[
            :, start_idx:end_idx, :
        ]
        assert_equals(
            current_open_loop_fingertip_positions.shape,
            (self.num_envs, NUM_FINGERS, NUM_XYZ),
        )
        return current_open_loop_fingertip_positions

    ##### CACHED PROPERTIES WITH INVALIDATION END #####

    ##### CURRICULUM PROPERTIES START #####
    @property
    def object_friction(self) -> float:
        if not hasattr(self, "_object_friction"):
            self._object_friction = self.custom_env_cfg.object_friction
        return self._object_friction

    @object_friction.setter
    def object_friction(self, value: float) -> None:
        self._object_friction = value

        for env, object in zip(self.envs, self.objects):
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env, object)
            assert_equals(len(rigid_shape_props), OBJECT_NUM_RIGID_BODIES)
            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self._object_friction
            self.gym.set_actor_rigid_shape_properties(env, object, rigid_shape_props)

    @property
    def object_mass_scale(self) -> float:
        if not hasattr(self, "_object_mass_scale"):
            self._object_mass_scale = self.custom_env_cfg.object_mass_scale
        return self._object_mass_scale

    @object_mass_scale.setter
    def object_mass_scale(self, value: float) -> None:
        orig_value = self.object_mass_scale
        self._object_mass_scale = value

        for env, object in zip(self.envs, self.objects):
            object_rb_props = self.gym.get_actor_rigid_body_properties(env, object)
            assert_equals(len(object_rb_props), OBJECT_NUM_RIGID_BODIES)
            for i in range(OBJECT_NUM_RIGID_BODIES):
                object_rb_props[i].mass *= value / orig_value
            self.gym.set_actor_rigid_body_properties(env, object, object_rb_props)

    @property
    def object_inertia_scale(self) -> float:
        if not hasattr(self, "_object_inertia_scale"):
            self._object_inertia_scale = self.custom_env_cfg.object_inertia_scale
        return self._object_inertia_scale

    @object_inertia_scale.setter
    def object_inertia_scale(self, value: float) -> None:
        orig_value = self.object_inertia_scale
        self._object_inertia_scale = value

        for env, object in zip(self.envs, self.objects):
            object_rb_props = self.gym.get_actor_rigid_body_properties(env, object)
            assert_equals(len(object_rb_props), OBJECT_NUM_RIGID_BODIES)
            for i in range(OBJECT_NUM_RIGID_BODIES):
                object_rb_props[i].inertia.x *= value / orig_value
                object_rb_props[i].inertia.y *= value / orig_value
                object_rb_props[i].inertia.z *= value / orig_value
            self.gym.set_actor_rigid_body_properties(env, object, object_rb_props)

    @property
    def random_force_scale(self) -> float:
        if not hasattr(self, "_random_force_scale"):
            self._random_force_scale = self.custom_env_cfg.randomForces.forceScale
        return self._random_force_scale

    @random_force_scale.setter
    def random_force_scale(self, value: float) -> None:
        self._random_force_scale = value

    @property
    def random_torque_scale(self) -> float:
        if not hasattr(self, "_random_torque_scale"):
            self._random_torque_scale = self.custom_env_cfg.randomForces.torqueScale
        return self._random_torque_scale

    @random_torque_scale.setter
    def random_torque_scale(self, value: float) -> None:
        self._random_torque_scale = value

    @property
    def random_force_prob(self) -> float:
        if not hasattr(self, "_random_force_prob"):
            self._random_force_prob = self.custom_env_cfg.randomForces.forceProb
        return self._random_force_prob

    @random_force_prob.setter
    def random_force_prob(self, value: float) -> None:
        self._random_force_prob = value

    @property
    def random_torque_prob(self) -> float:
        if not hasattr(self, "_random_torque_prob"):
            self._random_torque_prob = self.custom_env_cfg.randomForces.torqueProb
        return self._random_torque_prob

    @random_torque_prob.setter
    def random_torque_prob(self, value: float) -> None:
        self._random_torque_prob = value

    @property
    def observed_object_uncorr_pos_noise(self) -> float:
        if not hasattr(self, "_observed_object_uncorr_pos_noise"):
            self._observed_object_uncorr_pos_noise = (
                self.custom_env_cfg.OBSERVED_OBJECT_UNCORR_POS_NOISE
            )
        return self._observed_object_uncorr_pos_noise

    @observed_object_uncorr_pos_noise.setter
    def observed_object_uncorr_pos_noise(self, value: float) -> None:
        self._observed_object_uncorr_pos_noise = value

    @property
    def observed_object_uncorr_rpy_deg_noise(self) -> float:
        if not hasattr(self, "_observed_object_uncorr_rpy_deg_noise"):
            self._observed_object_uncorr_rpy_deg_noise = (
                self.custom_env_cfg.OBSERVED_OBJECT_UNCORR_RPY_DEG_NOISE
            )
        return self._observed_object_uncorr_rpy_deg_noise

    @observed_object_uncorr_rpy_deg_noise.setter
    def observed_object_uncorr_rpy_deg_noise(self, value: float) -> None:
        self._observed_object_uncorr_rpy_deg_noise = value

    @property
    def observed_object_corr_pos_noise(self) -> float:
        if not hasattr(self, "_observed_object_corr_pos_noise"):
            self._observed_object_corr_pos_noise = (
                self.custom_env_cfg.OBSERVED_OBJECT_CORR_POS_NOISE
            )
        return self._observed_object_corr_pos_noise

    @observed_object_corr_pos_noise.setter
    def observed_object_corr_pos_noise(self, value: float) -> None:
        self._observed_object_corr_pos_noise = value

    @property
    def observed_object_corr_rpy_deg_noise(self) -> float:
        if not hasattr(self, "_observed_object_corr_rpy_deg_noise"):
            self._observed_object_corr_rpy_deg_noise = (
                self.custom_env_cfg.OBSERVED_OBJECT_CORR_RPY_DEG_NOISE
            )
        return self._observed_object_corr_rpy_deg_noise

    @observed_object_corr_rpy_deg_noise.setter
    def observed_object_corr_rpy_deg_noise(self, value: float) -> None:
        self._observed_object_corr_rpy_deg_noise = value

    @property
    def observed_object_random_pose_injection_prob(self) -> float:
        if not hasattr(self, "_observed_object_random_pose_injection_prob"):
            self._observed_object_random_pose_injection_prob = (
                self.custom_env_cfg.OBSERVED_OBJECT_RANDOM_POSE_INJECTION_PROB
            )
        return self._observed_object_random_pose_injection_prob

    @observed_object_random_pose_injection_prob.setter
    def observed_object_random_pose_injection_prob(self, value: float) -> None:
        self._observed_object_random_pose_injection_prob = value

    @property
    def reset_object_sample_noise_x(self) -> float:
        if not hasattr(self, "_reset_object_sample_noise_x"):
            self._reset_object_sample_noise_x = (
                self.custom_env_cfg.reset_object_sample_noise_x
            )
        return self._reset_object_sample_noise_x

    @reset_object_sample_noise_x.setter
    def reset_object_sample_noise_x(self, value: float) -> None:
        self._reset_object_sample_noise_x = value

    @property
    def reset_object_sample_noise_y(self) -> float:
        if not hasattr(self, "_reset_object_sample_noise_y"):
            self._reset_object_sample_noise_y = (
                self.custom_env_cfg.reset_object_sample_noise_y
            )
        return self._reset_object_sample_noise_y

    @reset_object_sample_noise_y.setter
    def reset_object_sample_noise_y(self, value: float) -> None:
        self._reset_object_sample_noise_y = value

    @property
    def reset_object_sample_noise_z(self) -> float:
        if not hasattr(self, "_reset_object_sample_noise_z"):
            self._reset_object_sample_noise_z = (
                self.custom_env_cfg.reset_object_sample_noise_z
            )
        return self._reset_object_sample_noise_z

    @reset_object_sample_noise_z.setter
    def reset_object_sample_noise_z(self, value: float) -> None:
        self._reset_object_sample_noise_z = value

    @property
    def reset_object_sample_noise_roll_deg(self) -> float:
        if not hasattr(self, "_reset_object_sample_noise_roll_deg"):
            self._reset_object_sample_noise_roll_deg = (
                self.custom_env_cfg.reset_object_sample_noise_roll_deg
            )
        return self._reset_object_sample_noise_roll_deg

    @reset_object_sample_noise_roll_deg.setter
    def reset_object_sample_noise_roll_deg(self, value: float) -> None:
        self._reset_object_sample_noise_roll_deg = value

    @property
    def reset_object_sample_noise_pitch_deg(self) -> float:
        if not hasattr(self, "_reset_object_sample_noise_pitch_deg"):
            self._reset_object_sample_noise_pitch_deg = (
                self.custom_env_cfg.reset_object_sample_noise_pitch_deg
            )
        return self._reset_object_sample_noise_pitch_deg

    @reset_object_sample_noise_pitch_deg.setter
    def reset_object_sample_noise_pitch_deg(self, value: float) -> None:
        self._reset_object_sample_noise_pitch_deg = value

    @property
    def reset_object_sample_noise_yaw_deg(self) -> float:
        if not hasattr(self, "_reset_object_sample_noise_yaw_deg"):
            self._reset_object_sample_noise_yaw_deg = (
                self.custom_env_cfg.reset_object_sample_noise_yaw_deg
            )
        return self._reset_object_sample_noise_yaw_deg

    @reset_object_sample_noise_yaw_deg.setter
    def reset_object_sample_noise_yaw_deg(self, value: float) -> None:
        self._reset_object_sample_noise_yaw_deg = value

    @property
    def reset_right_robot_sample_noise_arm_deg(self) -> float:
        if not hasattr(self, "_reset_right_robot_sample_noise_arm_deg"):
            self._reset_right_robot_sample_noise_arm_deg = (
                self.custom_env_cfg.reset_right_robot_sample_noise_arm_deg
            )
        return self._reset_right_robot_sample_noise_arm_deg

    @reset_right_robot_sample_noise_arm_deg.setter
    def reset_right_robot_sample_noise_arm_deg(self, value: float) -> None:
        self._reset_right_robot_sample_noise_arm_deg = value

    @property
    def reset_right_robot_sample_noise_hand_deg(self) -> float:
        if not hasattr(self, "_reset_right_robot_sample_noise_hand_deg"):
            self._reset_right_robot_sample_noise_hand_deg = (
                self.custom_env_cfg.reset_right_robot_sample_noise_hand_deg
            )
        return self._reset_right_robot_sample_noise_hand_deg

    @reset_right_robot_sample_noise_hand_deg.setter
    def reset_right_robot_sample_noise_hand_deg(self, value: float) -> None:
        self._reset_right_robot_sample_noise_hand_deg = value

    ##### CURRICULUM PROPERTIES END #####

    ##### LOG PROPERTIES START #####
    @property
    @functools.lru_cache()
    def log_dir(self) -> Path:
        log_dir = Path("runs") / self.cfg.full_experiment_name
        if not log_dir.exists():
            print("~" * 80)
            print(f"WARNING: Log dir {log_dir} does not exist")
            print(f"Trying with {get_sim_training_dir()}")
            print("~" * 80)
            log_dir = get_sim_training_dir() / log_dir

        assert log_dir.exists(), f"Log dir {log_dir} does not exist"
        return log_dir

    @property
    @functools.lru_cache()
    def logger(self) -> logging.Logger:
        log_filepath = self.log_dir / "log.txt"
        print("-" * 80)
        print(f"Logging to {log_filepath}")
        print("-" * 80 + "\n")
        logging.basicConfig(
            format="%(asctime)s,%(msecs)03d %(levelname)-9s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_filepath, mode="a"),
                logging.StreamHandler(),
            ],
        )
        logger = logging.getLogger(__name__)
        return logger

    ##### LOG PROPERTIES END #####


@torch.jit.script
def compute_reward_jit(
    rew_buf: torch.Tensor,
    is_object_lifted: torch.Tensor,
    is_goal_object_lifted: torch.Tensor,
    smallest_this_episode_indexfingertip_goal_distance: torch.Tensor,
    indexfingertip_goal_distance: torch.Tensor,
    smallest_this_episode_fingertips_object_distance: torch.Tensor,
    fingertips_object_distance: torch.Tensor,
    smallest_this_episode_object_goal_distance: torch.Tensor,
    object_goal_distance: torch.Tensor,
    is_fingertips_object_close: torch.Tensor,
    is_in_success_region: torch.Tensor,
    has_enough_consecutive_successes_to_end_episode: torch.Tensor,
    max_episode_length: int,
    progress_buf: torch.Tensor,
    raw_actions: torch.Tensor,
    prev_raw_actions: torch.Tensor,
    object_and_goal_far_apart_need_stop_reference_motion: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    # Make sure this aligns with _setup_reward_weights
    reward_dict = {}

    reward_dict["Indexfingertip-Goal Distance Reward"] = torch.where(
        smallest_this_episode_indexfingertip_goal_distance
        < indexfingertip_goal_distance,
        torch.zeros_like(rew_buf),
        smallest_this_episode_indexfingertip_goal_distance
        - indexfingertip_goal_distance,
    )

    reward_dict["Fingertips-Object Distance Reward"] = torch.where(
        smallest_this_episode_fingertips_object_distance < fingertips_object_distance,
        torch.zeros_like(rew_buf),
        smallest_this_episode_fingertips_object_distance - fingertips_object_distance,
    )

    robot_is_lifting_object_and_should = torch.logical_and(
        torch.logical_and(is_object_lifted, is_goal_object_lifted),
        is_fingertips_object_close,
    )

    reward_dict["Object-Goal Distance Reward"] = torch.where(
        smallest_this_episode_object_goal_distance < object_goal_distance,
        torch.zeros_like(rew_buf),
        smallest_this_episode_object_goal_distance - object_goal_distance,
    )
    reward_dict["Success Reward"] = torch.where(
        is_in_success_region,
        torch.ones_like(rew_buf),
        torch.zeros_like(rew_buf),
    )
    reward_dict["Success Reward"] = torch.where(
        robot_is_lifting_object_and_should,
        reward_dict["Success Reward"],
        torch.zeros_like(rew_buf),
    )

    reward_dict["Consecutive Success Reward"] = torch.where(
        has_enough_consecutive_successes_to_end_episode,
        torch.ones_like(rew_buf) * (max_episode_length - progress_buf),
        torch.zeros_like(rew_buf),
    )
    reward_dict["Consecutive Success Reward"] = torch.where(
        robot_is_lifting_object_and_should,
        reward_dict["Consecutive Success Reward"],
        torch.zeros_like(rew_buf),
    )

    # Keypoint errors range from 0 to 0.5 most of the time
    #   * Roughly 0.1 keypoint error is still not great, but starting to get good
    #   * Roughly 0.01 keypoint error is very good
    # Thus we want the following approximate values
    #   * 0.01 keypoint error -> ~1.0 reward
    #   * 0.1 keypoint error -> ~0.3 reward
    # r = exp(-10 * e) roughly satisfies this
    TRACKING_ERROR_SCALE = 10.0
    reward_dict["Object Tracking Reward"] = torch.where(
        is_fingertips_object_close,
        torch.exp(-object_goal_distance * TRACKING_ERROR_SCALE),
        torch.zeros_like(rew_buf),
    )

    GIVE_LIFT_REWARD = False
    if GIVE_LIFT_REWARD:
        # Give a boost to the object tracking reward if the robot is lifting the object and should
        reward_dict["Object Tracking Reward"] = torch.where(
            robot_is_lifting_object_and_should,
            5 * reward_dict["Object Tracking Reward"],
            reward_dict["Object Tracking Reward"],
        )

    # Downscale the reward if the object tracking is bad
    reward_dict["Object Tracking Reward"] = torch.where(
        object_and_goal_far_apart_need_stop_reference_motion,
        0.1 * reward_dict["Object Tracking Reward"],
        reward_dict["Object Tracking Reward"],
    )

    # Action smoothing penalty
    reward_dict["Action Smoothing Penalty"] = (raw_actions - prev_raw_actions).norm(
        p=2, dim=-1
    ) ** 2

    reward_dict["Hand Tracking Reward"] = torch.zeros_like(rew_buf)

    return reward_dict


@torch.jit.script
def compute_reset_jit(
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: int,
    object_fallen_off_table: torch.Tensor,
    reference_motion_done: torch.Tensor,
    object_and_goal_far_apart_need_reset: torch.Tensor,
    has_enough_consecutive_successes_to_end_episode: torch.Tensor,
    is_fingertips_object_close: torch.Tensor,
    FORCE_REFERENCE_TRAJECTORY_TRACKING: bool,
    EARLY_RESET_BASED_ON_FINGERTIPS_OBJECT_DISTANCE: bool,
) -> torch.Tensor:
    reset = torch.zeros_like(reset_buf)
    reset[progress_buf >= max_episode_length] = 1
    reset[object_fallen_off_table] = 1
    if FORCE_REFERENCE_TRAJECTORY_TRACKING:
        reset[reference_motion_done] = 1
        reset[object_and_goal_far_apart_need_reset] = 1
    reset[has_enough_consecutive_successes_to_end_episode] = 1
    if (
        FORCE_REFERENCE_TRAJECTORY_TRACKING
        and EARLY_RESET_BASED_ON_FINGERTIPS_OBJECT_DISTANCE
    ):
        reset[~is_fingertips_object_close] = 1  # If the object is not close, reset
    return reset
