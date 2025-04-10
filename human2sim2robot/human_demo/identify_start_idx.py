from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tyro
from tqdm import tqdm


@dataclass
class Args:
    object_poses_dir: Path
    speed_threshold: Optional[float] = None
    dt: float = 1.0 / 30
    t_offset: float = 1.0
    window: int = 30
    plot: bool = False

    def __post_init__(self):
        self.idx_offset = int(self.t_offset / self.dt)


def identify_start_idx(args: Args) -> int:
    # Load object poses
    assert args.object_poses_dir.exists(), (
        f"Object poses dir {args.object_poses_dir} does not exist"
    )
    object_pose_files = sorted(list(args.object_poses_dir.glob("*.txt")))
    assert len(object_pose_files) > 0, (
        f"No object poses found in {args.object_poses_dir}"
    )
    T_C_Os = np.stack(
        [
            np.loadtxt(filename)
            for filename in tqdm(object_pose_files, desc="Loading object poses")
        ],
        axis=0,
    )
    N = T_C_Os.shape[0]
    assert T_C_Os.shape == (N, 4, 4), f"T_C_Os has shape {T_C_Os.shape}"

    speeds = np.array(
        [
            np.linalg.norm(T_C_Os[i + args.window, :3, 3] - T_C_Os[i, :3, 3])
            / (args.dt * args.window)
            for i in range(T_C_Os.shape[0] - args.window)
        ]
    )

    if args.speed_threshold is None:
        speed_threshold = np.percentile(speeds, 50)
        print(f"Raw speed_threshold = {speed_threshold:.4f}")

        # HACK: If the speed threshold is too low, set it to 0.01
        if speed_threshold < 0.001:
            speed_threshold = 0.01
    else:
        speed_threshold = args.speed_threshold

    is_moving = speeds > speed_threshold
    is_moving_idxs = np.where(is_moving)[0]

    if is_moving_idxs.size > 0:
        raw_start_idx = is_moving_idxs[0]
        print(f"raw_start_idx = {raw_start_idx}")
        print(f"idx_offset = {args.idx_offset}")
        start_idx = np.clip(raw_start_idx - args.idx_offset, a_min=0, a_max=None)
        print(f"start_idx = {start_idx}")
    else:
        raw_start_idx, start_idx = None, None

    if args.plot:
        plt.plot(speeds, label="speeds")
        plt.plot(is_moving * np.max(speeds), label="is_moving")
        plt.plot(
            np.ones_like(speeds) * speed_threshold,
            label=f"MIN_SPEED = {speed_threshold:.4f}",
        )
        plt.legend()
        plt.xlabel("Time Index")
        plt.ylabel("Speed (m/s)")

        if raw_start_idx is not None and start_idx is not None:
            plt.plot(
                raw_start_idx,
                speeds[raw_start_idx],
                "ro",
                label=f"raw_start_idx = {raw_start_idx}",
            )
            plt.plot(
                start_idx, speeds[start_idx], "go", label=f"start_idx = {start_idx}"
            )

        plt.legend()
        plt.show()

    if raw_start_idx is None or start_idx is None:
        raise ValueError(
            f"No moving object found in given trajectory, speeds = {speeds}, moving = {is_moving}, np.max(speeds) = {np.max(speeds)}"
        )

    return start_idx


def main():
    args = tyro.cli(Args)
    print("=" * 100)
    print(f"Args: {args}")
    print("=" * 100)
    identify_start_idx(args)


if __name__ == "__main__":
    main()
