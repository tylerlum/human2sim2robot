from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import tyro
from tqdm import tqdm

from human2sim2robot.human_demo.utils.utils import create_urdf


@dataclass
class Args:
    scene_dir: Path
    transform_path: Path
    output_dir: Path
    inverse_transform: bool = False
    create_urdf: bool = False


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(args)
    print("=" * 80 + "\n")

    assert args.scene_dir.exists(), f"Scene directory {args.scene_dir} does not exist"
    assert args.transform_path.exists(), (
        f"Transform file {args.transform_path} does not exist"
    )
    assert args.output_dir.exists(), (
        f"Output directory {args.output_dir} does not exist"
    )

    T = np.loadtxt(args.transform_path)
    assert T.shape == (4, 4), f"Transform matrix must be 4x4, got shape {T.shape}"
    print(f"Loaded the following transform matrix:\n{T}")

    if args.inverse_transform:
        T = np.linalg.inv(T)
        print(f"Inverting the transform matrix:\n{T}")

    input_objs_filepaths = sorted(list(args.scene_dir.glob("*.obj")))
    assert len(input_objs_filepaths) > 0, f"No .obj files found in {args.scene_dir}"

    for input_obj_filepath in tqdm(input_objs_filepaths, desc="Processing scenes"):
        print(f"Processing {input_obj_filepath}")

        # Load the scene
        scene = trimesh.load(input_obj_filepath)

        # Apply the transform
        scene.apply_transform(T)

        # Save the scene
        output_obj_filepath = args.output_dir / input_obj_filepath.name
        scene.export(output_obj_filepath)
        print(f"Saved processed scene to {output_obj_filepath}")

        if args.create_urdf:
            output_urdf_filepath = create_urdf(output_obj_filepath)
            print(f"Saved URDF to {output_urdf_filepath}")


if __name__ == "__main__":
    main()
