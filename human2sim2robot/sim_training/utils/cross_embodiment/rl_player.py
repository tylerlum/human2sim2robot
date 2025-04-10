from typing import Optional

import numpy as np
import torch
from gym import spaces

from human2sim2robot.ppo.ppo_agent import PpoConfig
from human2sim2robot.ppo.ppo_player import PlayerConfig, PpoPlayer
from human2sim2robot.ppo.utils.dict_to_dataclass import dict_to_dataclass
from human2sim2robot.ppo.utils.network import NetworkConfig
from human2sim2robot.sim_training.utils.cross_embodiment.rl_player_utils import (
    DummyEnv,
    read_cfg,
)


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


class RlPlayer:
    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        config_path: str,
        checkpoint_path: Optional[str],
        device: str,
    ) -> None:
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.device = device

        # Must create observation and action space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_observations,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_actions,), dtype=np.float32
        )

        self.cfg = read_cfg(config_path=config_path, device=self.device)
        self._run_sanity_checks()
        self.player = self.create_rl_player(checkpoint_path=checkpoint_path)

    def create_rl_player(self, checkpoint_path: Optional[str]) -> PpoPlayer:
        train_params = self.cfg["train"]
        network_config = dict_to_dataclass(train_params["network"], NetworkConfig)
        player_config = dict_to_dataclass(train_params["player"], PlayerConfig)
        ppo_player_config = dict_to_dataclass(
            train_params["ppo"], PpoConfig
        ).to_ppo_player_config()

        player = PpoPlayer(
            ppo_player_config=ppo_player_config,
            player_config=player_config,
            network_config=network_config,
            env=DummyEnv(
                observation_space=self.observation_space, action_space=self.action_space
            ),
        )
        player.init_rnn()
        player.has_batch_dimension = True
        if checkpoint_path is not None:
            player.restore(str(checkpoint_path))
        return player

    def _run_sanity_checks(self) -> None:
        cfg_num_observations = self.cfg["task"]["env"]["numObservations"]
        cfg_num_actions = self.cfg["task"]["env"]["numActions"]

        if cfg_num_observations != self.num_observations and cfg_num_observations > 0:
            print(
                f"WARNING: num_observations in config ({cfg_num_observations}) does not match num_observations passed to RlPlayer ({self.num_observations})"
            )
        if cfg_num_actions != self.num_actions and cfg_num_actions > 0:
            print(
                f"WARNING: num_actions in config ({cfg_num_actions}) does not match num_actions passed to RlPlayer ({self.num_actions})"
            )

    def get_normalized_action(
        self, obs: torch.Tensor, deterministic_actions: bool = True
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        assert_equals(obs.shape, (batch_size, self.num_observations))

        normalized_action = self.player.get_action(
            obs=obs, is_deterministic=deterministic_actions
        )

        normalized_action = normalized_action.reshape(-1, self.num_actions)
        assert_equals(normalized_action.shape, (batch_size, self.num_actions))
        return normalized_action


def main() -> None:
    import pathlib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_to_this_dir = pathlib.Path(__file__).parent.absolute()

    CONFIG_PATH = path_to_this_dir / "config.yaml"
    CHECKPOINT_PATH = path_to_this_dir / "checkpoint.pt"
    NUM_OBSERVATIONS = 100
    NUM_ACTIONS = 100

    player = RlPlayer(
        num_observations=NUM_OBSERVATIONS,
        num_actions=NUM_ACTIONS,
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
    )

    batch_size = 2
    obs = torch.rand(batch_size, NUM_OBSERVATIONS).to(device)
    normalized_action = player.get_normalized_action(obs=obs)
    print(f"Using player with config: {CONFIG_PATH} and checkpoint: {CHECKPOINT_PATH}")
    print(f"And num_observations: {NUM_OBSERVATIONS} and num_actions: {NUM_ACTIONS}")
    print(f"Sampled obs: {obs} with shape: {obs.shape}")
    print(
        f"Got normalized_action: {normalized_action} with shape: {normalized_action.shape}"
    )
    print(f"player: {player.player.model}")


if __name__ == "__main__":
    main()
