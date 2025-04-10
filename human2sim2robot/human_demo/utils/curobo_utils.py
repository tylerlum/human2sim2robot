from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from curobo.geom.types import WorldConfig
from scipy.spatial.transform import Rotation as R

from human2sim2robot.sim_training.utils.cross_embodiment.fabric_world import (
    transform_str_to_T,
)
from human2sim2robot.sim_training.utils.cross_embodiment.fabric_world import (
    world_dict_robot_frame as fabric_world_dict_robot_frame,
)


def get_table_collision_dict() -> dict:
    return {
        "cuboid": {
            "table": {
                "dims": [1.3208, 1.8288, 0.02],
                "pose": [0.50165, 0.0, -0.01, 1.0, 0.0, 0.0, 0.0],
            },
        }
    }


def get_object_collision_dict(
    file_paths: List[Path],
    xyzs: List[Tuple[float, float, float]],
    quat_wxyzs: List[Tuple[float, float, float, float]],
    obj_names: List[str] = ["object"],
) -> dict:
    def get_single_object_collision_dict(obj_name, xyz, quat_wxyz, file_path):
        return {
            obj_name: {
                "pose": [*xyz, *quat_wxyz],
                "file_path": str(file_path),
            }
        }

    object_collision_dict = {}
    for obj_name, xyz, quat_wxyz, file_path in zip(
        obj_names, xyzs, quat_wxyzs, file_paths
    ):
        object_collision_dict.update(
            get_single_object_collision_dict(obj_name, xyz, quat_wxyz, file_path)
        )
    return {"mesh": object_collision_dict}


def get_dummy_collision_dict() -> dict:
    FAR_AWAY_POS = 10.0
    return {
        "cuboid": {
            "dummy": {
                "dims": [0.1, 0.1, 0.1],
                "pose": [FAR_AWAY_POS, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            },
        }
    }


def get_world_cfg() -> WorldConfig:
    world_dict = {}

    # Add fabrics
    world_dict["cuboid"] = {}
    for k, v in fabric_world_dict_robot_frame.items():
        dims = [float(x) for x in v["scaling"].split()]
        T = transform_str_to_T(v["transform"])
        xyz = T[:3, 3]
        quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        xyz_qwxyz = np.zeros(7)
        xyz_qwxyz[:3] = xyz
        xyz_qwxyz[3:] = quat_wxyz
        world_dict["cuboid"][k] = {
            "dims": dims,
            "pose": xyz_qwxyz.tolist(),
        }
    world_cfg = WorldConfig.from_dict(world_dict)
    return world_cfg
