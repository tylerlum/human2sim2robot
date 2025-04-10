# gymtorch must be imported before torch
from human2sim2robot.sim_training.tasks.cross_embodiment.env import CrossEmbodiment  # isort:skip

from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from human2sim2robot.sim_training.tasks import isaacgym_task_map
from human2sim2robot.sim_training.utils.cross_embodiment.rl_player_utils import (
    read_cfg_omegaconf,
)
from human2sim2robot.sim_training.utils.reformat import omegaconf_to_dict
from human2sim2robot.sim_training.utils.wandb_utils import restore_file_from_wandb


def create_env(
    config_path: str,
    device: str,
    headless: bool = False,
    enable_viewer_sync_at_start: bool = True,
    merge_with_default_config: bool = True,
) -> CrossEmbodiment:
    cfg = read_cfg_omegaconf(config_path=config_path, device=device)

    if merge_with_default_config:
        # Use this if the config from config path is missing fields
        # For example, say we recently added a new field "object_friction" to the config
        # If this wasn't in the config file, this would normally fail
        # Merging with the default config will add this field with the default value
        print("Merging with default config")

        with initialize(version_base="1.1", config_path="../../cfg"):
            init_cfg = compose(config_name="config")

        # Disable struct mode to allow merging
        OmegaConf.set_struct(init_cfg, False)
        OmegaConf.set_struct(cfg, False)

        # Put cfg second to override init_cfg
        merged_cfg = OmegaConf.merge(init_cfg, cfg)
        assert isinstance(merged_cfg, DictConfig), (
            f"Expected DictConfig, got {type(merged_cfg)}"
        )

        # Print the differences
        diff = recursive_diff(
            OmegaConf.to_container(cfg, resolve=True),
            OmegaConf.to_container(merged_cfg, resolve=True),
        )
        print("Changes:")
        print("-" * 80)
        for key, change in diff.items():
            print(f"{key}: {change}")

        cfg = merged_cfg

    return create_env_from_cfg(
        cfg=cfg,
        headless=headless,
        enable_viewer_sync_at_start=enable_viewer_sync_at_start,
    )


def create_env_from_cfg(
    cfg: DictConfig,
    headless: bool = False,
    enable_viewer_sync_at_start: bool = True,
) -> CrossEmbodiment:
    # Modify the config
    cfg.headless = headless
    cfg.task.sim.enable_viewer_sync_at_start = enable_viewer_sync_at_start
    cfg.task.env.numEnvs = 1

    # HACK: Assume that graphics_device_id should be 0
    # This is a pretty reasonable assumption because we are typically doing this testing on a workstation with 1 GPU
    cfg.graphics_device_id = 0

    # Modify the config for the task
    # cfg.task.env.custom.object_friction = 0.5
    # cfg.task.env.custom.object_mass_scale = 1.0
    # cfg.task.env.custom.object_inertia_scale = 1.0

    env = isaacgym_task_map[cfg.task_name](
        cfg=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        rl_device=cfg.rl_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,
        force_render=True,
    )
    return env


def recursive_diff(cfg1: dict, cfg2: dict, path: str = "") -> dict:
    """Recursively compare two DictConfigs and return differences."""
    differences = {}

    # Get the keys from both configs
    keys1 = set(cfg1.keys()) if isinstance(cfg1, dict) else set()
    keys2 = set(cfg2.keys()) if isinstance(cfg2, dict) else set()

    # Check for keys that are only in cfg1
    for key in keys1 - keys2:
        differences[f"{path}.{key}".lstrip(".")] = f"{cfg1[key]} -> None"

    # Check for keys that are only in cfg2
    for key in keys2 - keys1:
        differences[f"{path}.{key}".lstrip(".")] = f"None -> {cfg2[key]}"

    # Check for keys that are in both configs
    for key in keys1 & keys2:
        val1 = cfg1[key]
        val2 = cfg2[key]

        # Recursively compare dictionaries or lists
        if isinstance(val1, dict) and isinstance(val2, dict):
            diff = recursive_diff(val1, val2, path=f"{path}.{key}".lstrip("."))
            differences.update(diff)
        elif val1 != val2:
            # If values differ, record the difference
            differences[f"{path}.{key}".lstrip(".")] = f"{val1} -> {val2}"

    return differences


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, CONFIG_PATH = restore_file_from_wandb(
        "https://wandb.ai/tylerlum/cross_embodiment/groups/2024-11-01_cup_trajectory_harder/files/runs/7_friction-0-3-5-8_force-50-0-05_noise-0-05-30-0-2_move2_2024-11-02_11-48-16-723074/config_resolved.yaml?runName=7_friction-0-3-5-8_force-50-0-05_noise-0-05-30-0-2_move2_2024-11-02_11-48-16-723074"
    )
    assert Path(CONFIG_PATH).exists()

    env = create_env(
        config_path=str(CONFIG_PATH),
        device=device,
    )

    print(env)
    obs = env.reset()
    N_STEPS = 10
    for _ in range(N_STEPS):
        action = torch.rand(
            (env.num_envs, env.num_acts), device=device, dtype=torch.float
        )
        obs, reward, done, info = env.step(action)


if __name__ == "__main__":
    main()
