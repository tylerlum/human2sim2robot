from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
import tyro
from PIL import Image


@dataclass
class Args:
    rgb_dir: Path
    depth_dir: Path
    cam_intrinsics_path: Path
    idx: int
    mask_dir: Optional[Path] = None


def visualize_geometries(
    width: int,
    height: int,
    cam_intrinsics: dict,
    geometries: List[o3d.geometry.Geometry],
    rescale_factor: float = 2.0,
):
    rescaled_width, rescaled_height = (
        int(width * rescale_factor),
        int(height * rescale_factor),
    )

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=rescaled_width, height=rescaled_height)

    # Add point clouds to visualizer
    for geom in geometries:
        vis.add_geometry(geom)

    # Get ViewControl and current camera parameters
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    # Update intrinsic matrix
    camera_params.intrinsic.set_intrinsics(
        width=rescaled_width,
        height=rescaled_height,
        fx=cam_intrinsics["fx"],
        fy=cam_intrinsics["fy"],
        cx=cam_intrinsics["cx"],
        cy=cam_intrinsics["cy"],
    )

    # Set up camera extrinsics (camera at origin with Z forward and Y down)
    extrinsics = np.eye(4)
    extrinsics[:3, 3] = np.array([0, 0, 0])  # origin
    extrinsics[:3, 0] = np.array([1, 0, 0])  # X-right
    extrinsics[:3, 1] = np.array([0, 1, 0])  # Y-down
    extrinsics[:3, 2] = np.array([0, 0, 1])  # Z-forward
    camera_params.extrinsic = extrinsics

    # Apply updated parameters
    view_control.convert_from_pinhole_camera_parameters(
        camera_params, allow_arbitrary=True
    )

    # Render and show
    vis.run()
    vis.destroy_window()


def get_point_cloud_of_segmask(
    mask: np.ndarray,
    depth_img: np.ndarray,
    img: np.ndarray,
    intrinsics: dict,
) -> o3d.geometry.PointCloud:
    """
    Return the point cloud that corresponds to the segmentation mask in the depth image.
    """
    idxs_y, idxs_x, _ = mask.nonzero()
    depth_masked = depth_img[idxs_y, idxs_x]
    seg_points = get_3D_point_from_pixel(
        px=idxs_x, py=idxs_y, depth=depth_masked, intrinsics=intrinsics
    )
    seg_colors = img[idxs_y, idxs_x, :] / 255.0  # Normalize to [0,1] for cv2

    pcd = get_pcd_from_points(seg_points, colors=seg_colors)

    return pcd


def get_pcd_from_points(
    points: np.ndarray, colors: Optional[np.ndarray] = None
) -> o3d.geometry.PointCloud:
    """
    Convert a list of points to an Open3D point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.remove_non_finite_points()
    return pcd


def get_3D_point_from_pixel(
    px: int, py: int, depth: float, intrinsics: dict
) -> np.ndarray:
    """
    Convert pixel coordinates and depth to 3D point.
    """
    x = (px - intrinsics["cx"]) / intrinsics["fx"]
    y = (py - intrinsics["cy"]) / intrinsics["fy"]

    depth = depth / 1000

    X = x * depth
    Y = y * depth
    if len(X.shape) == 0:
        p = np.array([X, Y, depth])
    else:
        p = np.stack((X, Y, depth), axis=1)

    return p


def main():
    args = tyro.cli(Args)
    assert args.rgb_dir.exists(), f"RGB directory not found: {args.rgb_dir}"
    assert args.depth_dir.exists(), f"Depth directory not found: {args.depth_dir}"
    assert args.cam_intrinsics_path.exists(), (
        f"Camera intrinsics file not found: {args.cam_intrinsics_path}"
    )
    if args.mask_dir is not None:
        assert args.mask_dir.exists(), f"Mask directory not found: {args.mask_dir}"

    rgb_filepath = args.rgb_dir / f"{args.idx:05d}.png"
    depth_filepath = args.depth_dir / f"{args.idx:05d}.png"
    assert rgb_filepath.exists(), f"RGB file not found: {rgb_filepath}"
    assert depth_filepath.exists(), f"Depth file not found: {depth_filepath}"
    if args.mask_dir is not None:
        mask_filepath = args.mask_dir / f"{args.idx:05d}.png"
        assert mask_filepath.exists(), f"Mask file not found: {mask_filepath}"

    rgb = np.array(Image.open(rgb_filepath))
    depth = np.array(Image.open(depth_filepath))
    cam_K = np.loadtxt(args.cam_intrinsics_path)
    if args.mask_dir is not None:
        mask = np.array(Image.open(mask_filepath))
        print(f"mask.shape: {mask.shape}")

    assert cam_K.shape == (3, 3), (
        f"Camera intrinsics matrix must be 3x3, got {cam_K.shape}"
    )
    height, width, channels = rgb.shape
    print(
        f"rgb.shape: {rgb.shape}, depth.shape: {depth.shape}, cam_K.shape: {cam_K.shape}"
    )
    assert depth.shape == (height, width), (
        f"Depth image must have the same height and width as RGB image, got {depth.shape} and {rgb.shape}"
    )
    assert channels == 3, f"RGB image must have 3 channels, got {channels}"

    cam_intrinsics = {
        "fx": cam_K[0, 0],
        "fy": cam_K[1, 1],
        "cx": cam_K[0, 2],
        "cy": cam_K[1, 2],
    }
    if args.mask_dir is not None:
        pcd = get_point_cloud_of_segmask(
            mask=mask, depth_img=depth, img=rgb, intrinsics=cam_intrinsics
        )
    else:
        pcd = get_point_cloud_of_segmask(
            mask=np.ones((height, width, channels), dtype=np.uint8),
            depth_img=depth,
            img=rgb,
            intrinsics=cam_intrinsics,
        )
    visualize_geometries(
        width=width, height=height, cam_intrinsics=cam_intrinsics, geometries=[pcd]
    )


if __name__ == "__main__":
    main()
