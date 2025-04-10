import time
from pathlib import Path
from typing import Any, Callable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float64"): torch.float32,
    np.dtype("float32"): torch.float32,
    # np.dtype('float64')    : torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}

torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}


def policy_kl(
    p0_mu: torch.Tensor,
    p0_sigma: torch.Tensor,
    p1_mu: torch.Tensor,
    p1_sigma: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    if reduce:
        return kl.mean()
    else:
        return kl


def safe_filesystem_op(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    This is to prevent spurious crashes related to saving checkpoints or restoring from checkpoints in a Network
    Filesystem environment (i.e. NGC cloud or SLURM)
    """
    num_attempts = 5
    for attempt in range(num_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            print(
                f"Exception {exc} when trying to execute {func} with args:{args} and kwargs:{kwargs}..."
            )
            wait_sec = 2**attempt
            print(f"Waiting {wait_sec} before trying again...")
            time.sleep(wait_sec)

    raise RuntimeError(
        f"Could not execute {func}, give up after {num_attempts} attempts..."
    )


def safe_save(state: Any, filename: Path) -> Any:
    return safe_filesystem_op(torch.save, state, filename)


def safe_load(filename: Path) -> Any:
    # return safe_filesystem_op(torch.load, filename)

    # HACK: Load to device=0
    # This is a reasonable assumption, as we typically run inference on a machine with 1 GPU
    # Proper solution: Pass in the map_location into this function with default to None
    from functools import partial

    load_fn = partial(torch.load, map_location=torch.device("cuda"))
    return safe_filesystem_op(load_fn, filename)


def save_checkpoint(filename: Path, state: Any) -> None:
    print(f"=> saving checkpoint '{filename}'")
    safe_save(state, filename)


def load_checkpoint(filename: Path) -> Any:
    print(f"=> loading checkpoint '{filename}'")
    state = safe_load(filename)
    return state


class AverageMeter(nn.Module):
    def __init__(self, in_shape: Union[int, Tuple[int, ...]], max_size: int) -> None:
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values: torch.Tensor) -> None:
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
        self.mean.fill_(0)

    def __len__(self) -> int:
        return self.current_size

    def get_mean(self) -> np.ndarray:
        return self.mean.squeeze(0).cpu().numpy()
