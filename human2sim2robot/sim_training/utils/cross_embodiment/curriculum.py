from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from omegaconf import MISSING


@dataclass
class CurriculumConfig:
    enabled: bool = MISSING
    log_freq: int = MISSING
    min_steps_between_updates: int = MISSING
    success_threshold_to_start_curriculum: float = MISSING
    success_threshold: float = MISSING
    consecutive_successes_required_to_update: int = MISSING


@dataclass
class CurriculumUpdate:
    variable_name: str
    update_amount: float
    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self):
        if self.update_amount > 0.0:
            assert self.max is not None and self.min is None, (
                f"max must be set if update_amount > 0.0, got {self.max} and {self.min}"
            )
        elif self.update_amount < 0.0:
            assert self.min is not None and self.max is None, (
                f"min must be set if update_amount < 0.0, got {self.min} and {self.max}"
            )
        else:
            raise ValueError(
                f"update_amount must be non-zero, got {self.update_amount}"
            )

    def validate(self, context) -> None:
        assert hasattr(context, self.variable_name), (
            f"Variable {self.variable_name} not found in context"
        )

    def done(self, context) -> bool:
        current_value = getattr(context, self.variable_name)
        return (self.max is not None and current_value >= self.max) or (
            self.min is not None and current_value <= self.min
        )

    def update(self, context) -> None:
        current_value = getattr(context, self.variable_name)
        new_value = np.clip(current_value + self.update_amount, self.min, self.max)
        print(f"Updating {self.variable_name} from {current_value} to {new_value}")
        setattr(context, self.variable_name, new_value)


class CurriculumUpdater:
    def __init__(self, context, curriculum_updates: List[CurriculumUpdate]) -> None:
        self.context = context
        self.curriculum_updates = curriculum_updates
        self.curriculum_idx = 0
        self._validate_curriculum_updates()

    def update(self) -> None:
        if self.curriculum_idx >= len(self.curriculum_updates):
            print("No more curriculum updates")
            return

        # Update variable
        curriculum_update = self.curriculum_updates[self.curriculum_idx]
        if not curriculum_update.done(self.context):
            curriculum_update.update(self.context)
            return

        # Move to next curriculum update
        self.curriculum_idx += 1
        self.update()

    def _validate_curriculum_updates(self) -> None:
        for curriculum_update in self.curriculum_updates:
            curriculum_update.validate(self.context)
        for curriculum_update in self.curriculum_updates:
            if curriculum_update.done(self.context):
                print(f"WARNING: curriculum_update ({curriculum_update}) already done")


class Curriculum:
    def __init__(
        self, curriculum_cfg: CurriculumConfig, curriculum_updater: CurriculumUpdater
    ) -> None:
        """
        curriculum_updates = [
            CurriculumUpdate(
                ...
            )
            for ...
        ]
        curriculum = Curriculum(
            curriculum_cfg=CurriculumConfig(...)
            curriculum_updater=CurriculumUpdater(
                curriculum_updates=curriculum_updates,
                context=self,
            )
        )
        """
        self.curriculum_cfg = curriculum_cfg
        self.curriculum_updater = curriculum_updater

        self.steps_since_last_curriculum_update = 0
        self.consecutive_steps_reached_curriculum_threshold = 0
        self.num_curriculum_updates = 0

    def update(self, success_metric: float) -> None:
        if not self.curriculum_cfg.enabled:
            return

        """
        1. Wait until success_threshold_to_start_curriculum reached before starting
        2. Wait at least min_steps_between_updates before updating
        3. Wait until consecutive_successes_required_to_update consecutive successes before updating
        4. Update
        """
        # Update state
        self.steps_since_last_curriculum_update += 1

        # Should print
        should_print = (
            self.steps_since_last_curriculum_update % self.curriculum_cfg.log_freq == 0
        )
        if should_print:
            print(f"success_metric: {success_metric}")
            print(
                f"steps_since_last_curriculum_update: {self.steps_since_last_curriculum_update}"
            )
            print(
                f"consecutive_steps_reached_curriculum_threshold: {self.consecutive_steps_reached_curriculum_threshold}"
            )

        # 1. Wait until success_threshold_to_start_curriculum reached before starting
        if (
            self.num_curriculum_updates == 0
            and success_metric
            < self.curriculum_cfg.success_threshold_to_start_curriculum
        ):
            if should_print:
                print(
                    f"success_metric ({success_metric}) < success_threshold_to_start_curriculum ({self.curriculum_cfg.success_threshold_to_start_curriculum}), so not starting curriculum"
                )
            return

        # 2. Wait at least min_steps_between_updates before updating
        passed_success_threshold = (
            success_metric >= self.curriculum_cfg.success_threshold
        )
        enough_steps_have_passed_since_last_update = (
            self.steps_since_last_curriculum_update
            >= self.curriculum_cfg.min_steps_between_updates
        )
        should_increment_consecutive_steps_reached = (
            passed_success_threshold and enough_steps_have_passed_since_last_update
        )
        if not should_increment_consecutive_steps_reached:
            self.consecutive_steps_reached_curriculum_threshold = 0
            return

        # 3. Wait until consecutive_successes_required_to_update consecutive successes before updating
        self.consecutive_steps_reached_curriculum_threshold += 1
        should_update = (
            self.consecutive_steps_reached_curriculum_threshold
            >= self.curriculum_cfg.consecutive_successes_required_to_update
        )
        if not should_update:
            return

        # 4. Update
        print("Updating curriculum")
        print("=" * 80)
        print(f"success_metric: {success_metric}")
        print(
            f"steps_since_last_curriculum_update: {self.steps_since_last_curriculum_update}"
        )
        self.curriculum_updater.update()
        print("=" * 80 + "\n")

        # Reset state
        self.steps_since_last_curriculum_update = 0
        self.consecutive_steps_reached_curriculum_threshold = 0
        self.num_curriculum_updates += 1
