import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import tyro
from scipy.spatial.transform import Rotation as R


@dataclass
class Args:
    source_path: Path = Path(__file__).parent / "robot_base.stl"
    target_path: Path = Path(__file__).parent / "scene_with_box_cropped.obj"
    output_path: Path = Path(__file__).parent / "transform.txt"
    threshold: float = 0.02
    number_of_points: int = 20_000
    bounding_box_x: float = -0.3
    bounding_box_y: float = -0.2
    bounding_box_z: float = 0.33
    bounding_box_len_x: float = 0.5
    bounding_box_len_y: float = 0.5
    bounding_box_len_z: float = 0.3
    init_roll_deg: float = -90.0
    init_pitch_deg: float = 0.0
    init_yaw_deg: float = 0.0
    max_iterations: int = 10_000

    @property
    def bounding_box_center(self) -> np.ndarray:
        return np.array([self.bounding_box_x, self.bounding_box_y, self.bounding_box_z])

    @property
    def bounding_box_size(self) -> np.ndarray:
        return np.array(
            [self.bounding_box_len_x, self.bounding_box_len_y, self.bounding_box_len_z]
        )


def draw_registration_result(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    bounding_box: Optional[o3d.geometry.AxisAlignedBoundingBox] = None,
) -> None:
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    AMBER = [1, 0.706, 0]
    TEAL = [0, 0.651, 0.929]
    source_temp.paint_uniform_color(AMBER)
    target_temp.paint_uniform_color(TEAL)
    source_temp.transform(transformation)

    geometries = [source_temp, target_temp]
    if bounding_box is not None:
        geometries.append(bounding_box)

    o3d.visualization.draw_geometries(geometries)


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    assert args.source_path.exists(), f"Source path not found at {args.source_path}"
    assert args.target_path.exists(), f"Target path not found at {args.target_path}"
    print(f"Source path: {args.source_path}")
    print(f"Target path: {args.target_path}")
    print("Loading source and target")

    if args.source_path.suffix in [".obj", ".stl"]:
        source_mesh = o3d.io.read_triangle_mesh(str(args.source_path))
        source = source_mesh.sample_points_poisson_disk(
            number_of_points=args.number_of_points
        )
    elif args.source_path.suffix == ".ply":
        source = o3d.io.read_point_cloud(str(args.source_path))
    else:
        raise ValueError(f"Unsupported source file format: {args.source_path.suffix}")

    if args.target_path.suffix in [".obj", ".stl"]:
        target_mesh = o3d.io.read_triangle_mesh(str(args.target_path))
        target = target_mesh.sample_points_poisson_disk(
            number_of_points=args.number_of_points
        )
    elif args.target_path.suffix == ".ply":
        target = o3d.io.read_point_cloud(str(args.target_path))
    else:
        raise ValueError(f"Unsupported target file format: {args.target_path.suffix}")

    # Bounding box (TUNE)
    print("Creating bounding box")
    center = target.get_center() + args.bounding_box_center
    size = args.bounding_box_size
    print(f"Center: {center}")
    print(f"Size: {size}")

    min_bound = center - size / 2
    max_bound = center + size / 2
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=min_bound, max_bound=max_bound
    )
    bounding_box.color = (1, 0, 0)

    print("Visualizing source and target with identity transformation and bounding box")
    draw_registration_result(source, target, np.eye(4), bounding_box)

    # Crop
    cropped_target = target.crop(bounding_box)

    # Create a rotation that aligns source_up with target_up and source_forward with target_forward
    T_init = np.eye(4)
    T_init[:3, :3] = R.from_euler(
        "xyz",
        [args.init_roll_deg, args.init_pitch_deg, args.init_yaw_deg],
        degrees=True,
    ).as_matrix()
    T_init[:3, 3] = center
    print(f"T_init: {T_init}")

    determinant = np.linalg.det(T_init[:3, :3])
    assert np.isclose(determinant, 1), (
        f"Determinant is not 1: {determinant} for {T_init[:3, :3]}"
    )

    print("~" * 50)
    print(
        f"Visualizing source and cropped target with initial transformation: {T_init}"
    )
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, cropped_target, args.threshold, T_init
    )
    print(evaluation)
    print("~" * 50 + "\n")
    draw_registration_result(source, cropped_target, T_init)

    print("~" * 50)
    print("Apply point-to-point ICP with more iterations")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        cropped_target,
        args.threshold,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=args.max_iterations
        ),
    )
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("~" * 50 + "\n")
    draw_registration_result(source, cropped_target, reg_p2p.transformation)

    T: np.ndarray = reg_p2p.transformation
    print("~" * 50)
    print("Visualizing this transformation to the full target")
    print("Transformation is:")
    print(T)
    print("~" * 50 + "\n")
    draw_registration_result(source, target, T)

    # Save the transform
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving transform to {args.output_path}")
    np.savetxt(args.output_path, T)


if __name__ == "__main__":
    main()
