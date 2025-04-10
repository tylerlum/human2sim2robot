from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import wandb
from human2sim2robot.sim_training.utils.cross_embodiment.constants import (
    NUM_QUAT,
    NUM_XYZ,
)
from human2sim2robot.sim_training.utils.torch_jit_utils import (
    quat_from_euler_xyz,
    quat_rotate,
)
from human2sim2robot.sim_training.utils.torch_utils import (
    quat_mul,
)


def assert_equals(a, b) -> None:
    assert a == b, f"{a} != {b}"


def wandb_started() -> bool:
    return wandb.run is not None


def clamp_magnitude(x: torch.Tensor, max_magnitude: float) -> torch.Tensor:
    magnitude = torch.norm(x, dim=-1, keepdim=True)
    return torch.where(magnitude > max_magnitude, x * (max_magnitude / magnitude), x)


def compute_keypoint_positions(
    pos: torch.Tensor,
    quat_xyzw: torch.Tensor,
    keypoint_offsets: torch.Tensor,
) -> torch.Tensor:
    N, _ = pos.shape
    assert_equals(pos.shape, (N, NUM_XYZ))
    assert_equals(quat_xyzw.shape, (N, NUM_QUAT))
    n_keypoints = keypoint_offsets.shape[1]
    assert_equals(keypoint_offsets.shape, (N, n_keypoints, NUM_XYZ))

    # Rotate keypoint offsets by quat_xyzw
    keypoint_offsets_rotated = torch.zeros_like(
        keypoint_offsets, device=keypoint_offsets.device
    )
    for i in range(n_keypoints):
        keypoint_offsets_i = keypoint_offsets[:, i]
        assert_equals(keypoint_offsets_i.shape, (N, NUM_XYZ))
        keypoint_offsets_rotated_i = quat_rotate(q=quat_xyzw, v=keypoint_offsets_i)
        assert_equals(keypoint_offsets_rotated_i.shape, (N, NUM_XYZ))

        keypoint_offsets_rotated[:, i] = keypoint_offsets_rotated_i

    # Add to pos
    keypoint_positions = pos.unsqueeze(dim=1) + keypoint_offsets_rotated
    assert_equals(keypoint_positions.shape, (N, n_keypoints, NUM_XYZ))
    return keypoint_positions


class AverageMeter(nn.Module):
    def __init__(self, in_shape: int = 1, max_size: int = 1000) -> None:
        super().__init__()
        self.max_size = max_size

        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values: torch.Tensor) -> None:
        assert_equals(len(values.shape), 1)
        size = values.size()[0]
        if size == 0:
            return

        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self) -> None:
        self.current_size = 0
        self.mean.fill_(0.0)

    def __len__(self) -> int:
        return self.current_size

    def get_mean(self) -> np.ndarray:
        return self.mean.squeeze(0).cpu().numpy()


def add_rpy_noise_to_quat_xyzw(
    quat_xyzw: torch.Tensor, rpy_noise: torch.Tensor
) -> torch.Tensor:
    N: int = quat_xyzw.shape[0]
    assert_equals(quat_xyzw.shape, (N, NUM_QUAT))
    assert_equals(rpy_noise.shape, (N, NUM_XYZ))

    rand_quat_xyzw = quat_from_euler_xyz(
        roll=rpy_noise[:, 0],
        pitch=rpy_noise[:, 1],
        yaw=rpy_noise[:, 2],
    )
    quat_xyzw_rotated = quat_mul(quat_xyzw, rand_quat_xyzw)

    return quat_xyzw_rotated


def rescale(
    values: torch.Tensor,
    old_mins: torch.Tensor,
    old_maxs: torch.Tensor,
    new_mins: torch.Tensor,
    new_maxs: torch.Tensor,
):
    """
    Rescale the input tensor from the old range to the new range.

    Args:
    values (torch.Tensor): Input tensor to be rescaled, shape (N, M)
    old_mins (torch.Tensor): Minimum values of the old range, shape (M,)
    old_maxs (torch.Tensor): Maximum values of the old range, shape (M,)
    new_mins (torch.Tensor): Minimum values of the new range, shape (M,)
    new_maxs (torch.Tensor): Maximum values of the new range, shape (M,)

    Returns:
    torch.Tensor: Rescaled tensor, shape (N, M)
    """
    assert_equals(len(values.shape), 2)
    N, M = values.shape
    assert_equals(old_mins.shape, (M,))
    assert_equals(old_maxs.shape, (M,))
    assert_equals(new_mins.shape, (M,))
    assert_equals(new_maxs.shape, (M,))

    # Ensure all inputs are tensors and on the same device
    old_mins = torch.as_tensor(old_mins, dtype=values.dtype, device=values.device)
    old_maxs = torch.as_tensor(old_maxs, dtype=values.dtype, device=values.device)
    new_mins = torch.as_tensor(new_mins, dtype=values.dtype, device=values.device)
    new_maxs = torch.as_tensor(new_maxs, dtype=values.dtype, device=values.device)

    # Clip the input values to be within the old range
    values_clipped = torch.clamp(values, min=old_mins[None], max=old_maxs[None])

    # Perform the rescaling
    rescaled = (values_clipped - old_mins[None]) / (old_maxs[None] - old_mins[None]) * (
        new_maxs[None] - new_mins[None]
    ) + new_mins[None]

    return rescaled


def clip_T_list(raw_T_list: torch.Tensor, data_dt: float) -> torch.Tensor:
    N_raw = raw_T_list.shape[0]
    assert raw_T_list.shape == (
        N_raw,
        4,
        4,
    ), f"Expected shape {(N_raw, 4, 4)}, got {raw_T_list.shape}"

    all_xyz = raw_T_list[:, :3, 3].cpu().numpy()

    # Use a window for smoother speed calculation
    WINDOW = 30
    speeds = np.linalg.norm(all_xyz[WINDOW:] - all_xyz[:-WINDOW], axis=-1) / (
        WINDOW * data_dt
    )

    # MIN_SPEED = 0.05  # BRITTLE: May need to tune per task
    MIN_SPEED = np.percentile(speeds, 50)
    is_moving = speeds > MIN_SPEED

    N_CONSECUTIVE_MOVING = 10
    # Create a rolling sum over a window of size N_CONSECUTIVE_MOVING
    rolling_sum = np.convolve(
        is_moving, np.ones(N_CONSECUTIVE_MOVING, dtype=int), mode="valid"
    )
    is_moving_consecutively = rolling_sum == N_CONSECUTIVE_MOVING

    PLOT = False
    if PLOT:
        import matplotlib.pyplot as plt

        plt.plot(speeds, label="speeds")
        plt.plot([MIN_SPEED] * N_raw, label="MIN_SPEED")
        plt.plot(
            is_moving_consecutively.astype(float) * np.max(speeds),
            label="is_moving_consecutively",
        )
        plt.legend()
        plt.show()
        breakpoint()

    is_moving_idxs = np.where(is_moving_consecutively)[0]
    if is_moving_idxs.size == 0:
        raise ValueError(
            f"No moving object found in given trajectory, speeds = {speeds}, moving = {is_moving}, np.max(speeds) = {np.max(speeds)}"
        )

    start_step = is_moving_idxs[0]

    END_EARLY = False
    if END_EARLY:
        end_step = is_moving_idxs[-1] + WINDOW
    else:
        end_step = N_raw

    return raw_T_list[start_step:end_step]


def read_in_T_list(object_trajectory_folder: Path) -> np.ndarray:
    assert object_trajectory_folder.exists(), (
        f"Object trajectory folder {object_trajectory_folder} does not exist"
    )

    # Process reference motion
    assert object_trajectory_folder.is_dir(), (
        f"{object_trajectory_folder} is not a directory"
    )
    txt_files = sorted(list(object_trajectory_folder.glob("*.txt")))
    if len(txt_files) == 0:
        raise ValueError(f"No txt files found in {object_trajectory_folder}")

    raw_T_list = np.stack(
        [np.loadtxt(filename) for filename in txt_files],
        axis=0,
    )
    return raw_T_list
