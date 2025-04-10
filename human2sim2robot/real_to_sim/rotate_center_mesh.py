from pathlib import Path

import numpy as np
import trimesh

INPUT_PATH = Path(
    "/afs/cs.stanford.edu/u/tylerlum/Downloads/plate_yellow_smiley/3DModel.obj"
)
OUTPUT_PATH = INPUT_PATH.parent / f"{INPUT_PATH.stem}_new.obj"
mesh = trimesh.load_mesh(INPUT_PATH)
print(f"INPUT_PATH = {INPUT_PATH}")

T = np.eye(4)
T[:3, 3] = -mesh.centroid
print(f"mesh.centroid = {mesh.centroid}")
mesh.apply_transform(T)

T2 = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
mesh.apply_transform(T2)

mesh.export(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
