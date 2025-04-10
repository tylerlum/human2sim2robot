# run.py
# Script to train or test policies in Isaac Gym
#
# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def main(cfg: DictConfig):
    # gymtorch must be imported before torch
    from human2sim2robot.sim_training.tasks import isaacgym_task_map  # isort:skip
    import os
    from pathlib import Path

    import torch

    # noinspection PyUnresolvedReferences
    from hydra.utils import to_absolute_path

    import wandb
    from human2sim2robot.ppo.ppo_agent import PpoAgent, PpoConfig
    from human2sim2robot.ppo.ppo_player import PlayerConfig, PpoPlayer
    from human2sim2robot.ppo.utils.dict_to_dataclass import dict_to_dataclass
    from human2sim2robot.ppo.utils.network import NetworkConfig
    from human2sim2robot.sim_training import get_cfg_dir
    from human2sim2robot.sim_training.utils.reformat import (
        omegaconf_to_dict,
        print_dict,
    )
    from human2sim2robot.sim_training.utils.utils import set_np_formatting, set_seed
    from wandb.sdk.lib.runid import generate_id

    merge_with_default_config = True  # HACK
    if merge_with_default_config:
        # Use this if the config from config path is missing fields
        # For example, say we recently added a new field "object_friction" to the config
        # If this wasn't in the config file, this would normally fail
        # Merging with the default config will add this field with the default value
        print("Merging with default config")

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        from human2sim2robot.sim_training.utils.cross_embodiment.create_env import (
            recursive_diff,
        )

        # HACK: Do this a better way
        # Hydra complains if we load another config without clearing the global hydra instance
        GlobalHydra.instance().clear()

        with initialize_config_dir(version_base="1.1", config_dir=str(get_cfg_dir())):
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

    if cfg.checkpoint:
        ### WANDB CHECKPOINT START ###
        if cfg.checkpoint.startswith("https://wandb.ai"):
            from human2sim2robot.sim_training.utils.wandb_utils import (
                restore_model_file_from_wandb,
            )

            print("-" * 80)
            print(f"Attempting to restore model from {cfg.checkpoint}")
            cfg.checkpoint = restore_model_file_from_wandb(
                wandb_file_url=cfg.checkpoint,
            )
            print(f"Restored model from {cfg.checkpoint}")
            print("-" * 80)
            print()
        ### WANDB CHECKPOINT END ###

        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    experiment_name = f"{cfg.full_experiment_name}"
    print("-" * 80)
    print(f"Running experiment: {experiment_name}")
    print("-" * 80)
    print()
    print_dict(omegaconf_to_dict(cfg))
    print("-" * 80)
    print()

    # set numpy formatting for printing only
    set_np_formatting()

    # dump config dict
    ### DUMP CONFIG BOTH RESOLVED AND NOT START ###
    experiment_dir = Path("runs") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(experiment_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=False))
    with open(experiment_dir / "config_resolved.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    ### DUMP CONFIG BOTH RESOLVED AND NOT END ###

    assert cfg.train.ppo.multi_gpu == cfg.multi_gpu, (
        f"{cfg.train.ppo.multi_gpu} != {cfg.multi_gpu}"
    )
    rank = int(os.getenv("LOCAL_RANK", "0"))  # TODO: Maybe use global rank RANK
    if cfg.multi_gpu:
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f"cuda:{rank}"
        cfg.rl_device = f"cuda:{rank}"
        # sets seed. if seed is -1 will pick a random one
        cfg.seed = set_seed(
            cfg.seed + rank, torch_deterministic=cfg.torch_deterministic
        )
    else:
        # use the same device for sim and rl
        cfg.sim_device = f"cuda:{cfg.device_id}" if cfg.device_id >= 0 else "cpu"
        cfg.rl_device = f"cuda:{cfg.device_id}" if cfg.device_id >= 0 else "cpu"
        cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # connect to wandb, do it before creating the environment
    train = not cfg.test
    if train and cfg.wandb_activate:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            group=cfg.wandb_group,
            config=omegaconf_to_dict(cfg),
            sync_tensorboard=True,
            id=f"{cfg.wandb_name}_{generate_id()}",
        )

    print("Start Building the Environment")
    env = isaacgym_task_map[cfg.task_name](
        cfg=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        rl_device=cfg.rl_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,
        force_render=True,
    )

    # convert CLI arguments into dictionary
    # create runner and set the settings
    torch.backends.cudnn.benchmark = True
    train_params = omegaconf_to_dict(cfg.train)
    checkpoint = cfg.checkpoint
    sigma = cfg.sigma if cfg.sigma != "" else None

    if train:
        network_config = dict_to_dataclass(train_params["network"], NetworkConfig)
        ppo_config = dict_to_dataclass(train_params["ppo"], PpoConfig)
        ppo_config.device = cfg.rl_device

        agent = PpoAgent(
            experiment_dir=experiment_dir,
            ppo_config=ppo_config,
            network_config=network_config,
            env=env,
        )
        if checkpoint is not None and checkpoint != "":
            agent.restore(Path(checkpoint))
        if sigma is not None:
            agent.override_sigma(sigma)
        agent.train()

    else:
        network_config = dict_to_dataclass(train_params["network"], NetworkConfig)
        player_config = dict_to_dataclass(train_params["player"], PlayerConfig)
        ppo_player_config = dict_to_dataclass(
            train_params["ppo"], PpoConfig
        ).to_ppo_player_config()
        ppo_player_config.device = cfg.rl_device

        player = PpoPlayer(
            ppo_player_config=ppo_player_config,
            player_config=player_config,
            network_config=network_config,
            env=env,
        )
        if checkpoint is not None and checkpoint != "":
            player.restore(Path(checkpoint))
        if sigma is not None:
            player.override_sigma(sigma)
        player.run()


if __name__ == "__main__":
    main()
