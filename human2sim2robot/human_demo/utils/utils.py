from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize 1D vector
    """
    assert v.ndim == 1, f"v.shape: {v.shape}"
    norm = np.linalg.norm(v)
    assert norm > 0, f"norm: {norm}"
    return v / norm


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Transform point by transform T
    """
    assert point.shape == (3,)
    assert T.shape == (4, 4)
    point = np.concatenate([point, [1]])
    transformed_point = T @ point
    return transformed_point[:3]


def create_transform(
    pos: np.ndarray,
    rot: np.ndarray,
) -> np.ndarray:
    """
    Create transform T from position and rotation
    """
    assert pos.shape == (3,)
    assert rot.shape == (3, 3)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def create_urdf(
    obj_filepath: Path,
    mass: float = 0.066,
    ixx: float = 1e-3,
    iyy: float = 1e-3,
    izz: float = 1e-3,
    color: Optional[Literal["white"]] = None,
) -> Path:
    """
    Create URDF file for new object from path to object mesh
    """
    if color == "white":
        color_material = (
            """<material name="white"> <color rgba="1. 1. 1. 1."/> </material>"""
        )
    elif color is None:
        color_material = ""
    else:
        raise ValueError(f"Invalid color {color}")

    assert obj_filepath.suffix == ".obj"
    urdf_filepath = obj_filepath.with_suffix(".urdf")
    urdf_text = f"""<?xml version="1.0" ?>
        <robot name="model.urdf">
        <link name="baseLink">
            <contact>
                <lateral_friction value="0.8"/>
                <rolling_friction value="0.001"/>g
                <contact_cfm value="0.0"/>
                <contact_erp value="1.0"/>
            </contact>
            <inertial>
                <mass value="{mass}"/>
                <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
            </inertial>
            <visual>
            <geometry>
                <mesh filename="{obj_filepath.name}" scale="1 1 1"/>
            </geometry>
            {color_material}
            </visual>
            <collision>
            <geometry>
                <mesh filename="{obj_filepath.name}" scale="1 1 1"/>
            </geometry>
            </collision>
        </link>
        </robot>"""
    with urdf_filepath.open("w") as f:
        f.write(urdf_text)
    return urdf_filepath
