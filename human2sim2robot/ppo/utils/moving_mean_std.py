from typing import Literal, Tuple, Union, get_args

import torch
import torch.nn as nn

# Define the Literal values once
GeneralizedMovingStatsImpl = Literal[
    "off",
    "mean_std",
    "mean_std_corr",
    "min_max",
    "perc_ema",
    "perc_ema_corr",
    "mean_mag",
    "max_mag",
]

# Extract the allowed values at runtime from the type alias
ALLOWED_IMPLS = get_args(GeneralizedMovingStatsImpl)


class GeneralizedMovingStats(nn.Module):
    def __init__(
        self,
        insize: int,
        impl: GeneralizedMovingStatsImpl = "mean_std",
        decay: float = 0.99,
        max: float = 1e5,
        eps: float = 0.0,
        perclo: float = 0.05,
        perchi: float = 0.95,
    ) -> None:
        super().__init__()
        self.impl = impl
        self.decay = decay
        self.max = max
        self.eps = eps
        self.perclo = perclo
        self.perchi = perchi

        assert self.impl in ALLOWED_IMPLS, (
            f"Invalid impl: {self.impl} not in {ALLOWED_IMPLS}"
        )

        if self.impl == "off":
            pass
        elif self.impl == "mean_std":
            self.step = torch.nn.Parameter(
                torch.ones((1), dtype=torch.int32), requires_grad=False
            )
            self.mean = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
            self.sqrs = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
        elif self.impl == "mean_std_corr":
            self.step = torch.nn.Parameter(
                torch.ones((1), dtype=torch.int32), requires_grad=False
            )
            self.mean = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
            self.sqrs = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
        elif self.impl == "min_max":
            self.low = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
            self.high = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
        elif self.impl == "perc_ema":
            self.low = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
            self.high = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
        elif self.impl == "perc_ema_corr":
            self.step = torch.nn.Parameter(
                torch.ones((1), dtype=torch.int32), requires_grad=False
            )
            self.low = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
            self.high = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
        elif self.impl == "mean_mag":
            self.mag = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
        elif self.impl == "max_mag":
            self.mag = torch.nn.Parameter(
                torch.zeros((insize), dtype=torch.float32), requires_grad=False
            )
        else:
            raise NotImplementedError(
                f"Invalid impl: {self.impl} not in {ALLOWED_IMPLS}"
            )

    def _get_stats(
        self,
    ) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
        if self.impl == "off":
            return 0.0, 1.0
        elif self.impl == "mean_std":
            mean = self.mean
            var = (self.sqrs) - self.mean**2
            std = torch.sqrt(torch.clamp_min(var, 1 / self.max**2) + self.eps)
            return mean, std
        elif self.impl == "mean_std_corr":
            corr = 1.0 - self.decay ** self.step.float()
            mean = self.mean / corr
            var = (self.sqrs / corr) - self.mean**2
            std = torch.sqrt(torch.clamp_min(var, 1 / self.max**2) + self.eps)
            return mean, std
        elif self.impl == "min_max":
            offset = self.low
            invscale = torch.clamp_min(self.high - self.low, 1 / self.max)
            return offset, invscale
        elif self.impl == "perc_ema":
            offset = self.low
            invscale = torch.clamp_min(self.high - self.low, 1 / self.max)
            return offset, invscale
        elif self.impl == "perc_ema_corr":
            corr = 1 - self.decay ** self.step.float()
            lo = self.low / corr
            hi = self.high / corr
            invscale = torch.clamp_min(hi - lo, 1 / self.max)
            return lo, invscale
        else:
            raise NotImplementedError(
                f"Invalid impl: {self.impl} not in {ALLOWED_IMPLS}"
            )

    def _update_stats(self, x: torch.Tensor) -> None:
        m = self.decay
        if self.impl == "off":
            pass
        elif self.impl == "mean_std":
            self.step.data += 1
            self.mean.data = m * self.mean.data + (1 - m) * torch.mean(x)
            self.sqrs.data = m * self.sqrs.data + (1 - m) * torch.mean(x**2)
        elif self.impl == "mean_std_corr":
            self.step.data += 1
            self.mean.data = m * self.mean.data + (1 - m) * torch.mean(x)
            self.sqrs.data = m * self.sqrs.data + (1 - m) * torch.mean(x**2)
        elif self.impl == "min_max":
            low, high = torch.min(x), torch.max(x)
            self.low.data = m * torch.minimum(self.low.data, low) + (1 - m) * low
            self.high.data = m * torch.maximum(self.high.data, high) + (1 - m) * high
        elif self.impl == "perc_ema":
            low, high = torch.quantile(x, self.perclo), torch.quantile(x, self.perchi)
            self.low.data = m * self.low.data + (1 - m) * low
            self.high.data = m * self.high.data + (1 - m) * high
        elif self.impl == "perc_ema_corr":
            self.step.data += 1
            low, high = torch.quantile(x, self.perclo), torch.quantile(x, self.perchi)
            self.low.data = m * self.low.data + (1 - m) * low
            self.high.data = m * self.high.data + (1 - m) * high
        else:
            raise NotImplementedError(
                f"Invalid impl: {self.impl} not in {ALLOWED_IMPLS}"
            )

    def forward(self, input: torch.Tensor, denorm: bool = False) -> torch.Tensor:
        if self.training:
            self._update_stats(input)

        offset, invscale = self._get_stats()

        if denorm:
            y = input * invscale + offset
        else:
            y = (input - offset) / invscale
            y = torch.clamp(y, min=-5.0, max=5.0)
        return y
