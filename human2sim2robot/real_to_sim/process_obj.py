from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
import tyro

from human2sim2robot.human_demo.utils.utils import create_urdf


@dataclass
class Args:
    obj_path: Path
    output_dir: Path
    create_urdf: bool = False
    center_origin: bool = False
    current_up_dir: Literal["x", "y", "z", "neg_x", "neg_y", "neg_z", None] = None
    new_up_dir: Literal["x", "y", "z", "neg_x", "neg_y", "neg_z", None] = None


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    assert args.obj_path.exists(), f"OBJ path {args.obj_path} does not exist"
    assert args.obj_path.suffix == ".obj", (
        f"OBJ path {args.obj_path} must have .obj extension"
    )

    mesh = trimesh.load_mesh(args.obj_path)

    if args.center_origin:
        T_translate = np.eye(4)
        T_translate[:3, 3] = -mesh.centroid
        print(f"Centering origin to {mesh.centroid}")
        mesh.apply_transform(T_translate)

    if args.current_up_dir is not None and args.new_up_dir is not None:
        assert args.current_up_dir in [
            "x",
            "y",
            "z",
            "neg_x",
            "neg_y",
            "neg_z",
        ], f"Invalid current_up_dir {args.current_up_dir}"
        assert args.new_up_dir in [
            "x",
            "y",
            "z",
            "neg_x",
            "neg_y",
            "neg_z",
        ], f"Invalid new_up_dir {args.new_up_dir}"
        print(f"Rotating from up-dir {args.current_up_dir} to up-dir {args.new_up_dir}")

        # Map string directions to actual vectors
        direction_map = {
            "x": np.array([1, 0, 0]),
            "neg_x": np.array([-1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "neg_y": np.array([0, -1, 0]),
            "z": np.array([0, 0, 1]),
            "neg_z": np.array([0, 0, -1]),
        }

        old_up = direction_map[args.current_up_dir]
        new_up = direction_map[args.new_up_dir]

        # Create a rotation that aligns old_up with new_up
        T_rotate = trimesh.geometry.align_vectors(old_up, new_up)

        # Apply this transform to the mesh
        mesh.apply_transform(T_rotate)
    elif args.current_up_dir is not None or args.new_up_dir is not None:
        raise ValueError(
            f"Must specify both or neither of current_up_dir ({args.current_up_dir}) and new_up_dir ({args.new_up_dir})"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_obj_path = args.output_dir / args.obj_path.name
    mesh.export(output_obj_path)
    print(f"Saved to {output_obj_path}")

    if args.create_urdf:
        output_urdf_path = create_urdf(output_obj_path)
        print(f"Created URDF at {output_urdf_path}")


if __name__ == "__main__":
    main()
