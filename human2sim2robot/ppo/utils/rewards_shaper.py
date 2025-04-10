from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RewardsShaperParams:
    scale_value: float = 1
    shift_value: float = 0
    min_val: float = -np.inf
    max_val: float = np.inf
    log_val: bool = False
    is_torch: bool = True


class DefaultRewardsShaper:
    def __init__(
        self,
        params: RewardsShaperParams = RewardsShaperParams(),
    ) -> None:
        self.scale_value = params.scale_value
        self.shift_value = params.shift_value
        self.min_val = params.min_val
        self.max_val = params.max_val
        self.log_val = params.log_val
        self.is_torch = params.is_torch

        if self.is_torch:
            import torch

            self.log = torch.log
            self.clip = torch.clamp
        else:
            self.log = np.log
            self.clip = np.clip

    def __call__(self, reward: Any) -> Any:
        reward = reward + self.shift_value
        reward = reward * self.scale_value

        reward = self.clip(reward, self.min_val, self.max_val)

        if self.log_val:
            reward = self.log(reward)
        return reward
