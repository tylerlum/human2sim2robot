import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from gym import spaces

from human2sim2robot.ppo.ppo_agent import PpoAgent, PpoConfig
from human2sim2robot.ppo.utils.dict_to_dataclass import dict_to_dataclass
from human2sim2robot.ppo.utils.network import NetworkConfig
from human2sim2robot.sim_training.utils.cross_embodiment.rl_player_utils import (
    DummyEnv,
    read_cfg,
)


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


class RlAgent:
    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        num_states: int,
        config_path: str,
        checkpoint_path: Optional[str],
        device: str,
    ) -> None:
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_states = num_states
        self.device = device

        # Must create observation and action space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_observations,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_actions,), dtype=np.float32
        )

        # Must create state space
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_states,), dtype=np.float32
        )

        self.cfg = read_cfg(config_path=config_path, device=self.device)
        self._run_sanity_checks()
        self.agent = self.create_rl_agent(checkpoint_path=checkpoint_path)

    def create_rl_agent(self, checkpoint_path: Optional[str]) -> PpoAgent:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        train_params = self.cfg["train"]

        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ppo_config = dict_to_dataclass(train_params["ppo"], PpoConfig)
        network_config = dict_to_dataclass(train_params["network"], NetworkConfig)
        env = DummyEnv(
            observation_space=self.observation_space,
            action_space=self.action_space,
            state_space=self.state_space,
        )
        agent = PpoAgent(
            experiment_dir=f"runs/{datetime_str}",
            ppo_config=ppo_config,
            network_config=network_config,
            env=env,
        )

        if checkpoint_path is not None and checkpoint_path != "":
            agent.restore(checkpoint_path)

        agent.init_tensors()
        agent.is_tensor_obses = True
        return agent

    def _run_sanity_checks(self) -> None:
        cfg_num_observations = self.cfg["task"]["env"]["numObservations"]
        cfg_num_actions = self.cfg["task"]["env"]["numActions"]

        if cfg_num_observations != self.num_observations and cfg_num_observations > 0:
            print(
                f"WARNING: num_observations in config ({cfg_num_observations}) does not match num_observations passed to Agent ({self.num_observations})"
            )
        if cfg_num_actions != self.num_actions and cfg_num_actions > 0:
            print(
                f"WARNING: num_actions in config ({cfg_num_actions}) does not match num_actions passed to Agent ({self.num_actions})"
            )

    def get_res_dict(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        update_rnn_states: bool = True,
        no_grad: bool = True,
    ) -> dict:
        batch_size = obs.shape[0]
        assert_equals(obs.shape, (batch_size, self.num_observations))
        assert_equals(state.shape, (batch_size, self.num_states))

        if no_grad:
            res_dict = self.agent.get_action_values(
                {
                    "obs": obs,
                    "states": state,
                }
            )
        else:
            res_dict = self.get_action_values_with_grad(
                {
                    "obs": obs,
                    "states": state,
                }
            )

        if update_rnn_states:
            self.agent.rnn_states = res_dict["rnn_states"]
        return res_dict

    def get_action_values_with_grad(self, obs_dict: dict) -> dict:
        # Same as a2c_common.py get_action_values, but with grad
        processed_obs = self.agent._preproc_obs(obs_dict["obs"])
        self.agent.model.eval()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "rnn_states": self.agent.rnn_states,
        }

        res_dict = self.agent.model(input_dict)
        if self.agent.has_asymmetric_critic:
            states = obs_dict["states"]
            input_dict = {
                "is_train": False,
                "states": states,
            }
            value = self.agent.get_asymmetric_critic_value(input_dict)
            res_dict["values"] = value
        return res_dict

    def get_normalized_action(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        deterministic_actions: bool = True,
        no_grad: bool = True,
    ) -> torch.Tensor:
        batch_size = obs.shape[0]

        res_dict = self.get_res_dict(
            obs=obs, state=state, update_rnn_states=True, no_grad=no_grad
        )
        actions = res_dict["actions"]
        mus = res_dict["mus"]

        actions_processed = self.agent.preprocess_actions(actions)
        mus_processed = self.agent.preprocess_actions(mus)
        assert actions_processed.shape in [
            (batch_size, self.num_actions),
            (2 * batch_size, self.num_actions),
        ], f"actions_processed.shape={actions_processed.shape}"
        assert mus_processed.shape in [
            (batch_size, self.num_actions),
            (2 * batch_size, self.num_actions),
        ], f"mus_processed.shape={mus_processed.shape}"

        if deterministic_actions:
            return mus_processed
        else:
            return actions_processed

    def get_values(
        self, obs: torch.Tensor, state: torch.Tensor, no_grad: bool = True
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        res_dict = self.get_res_dict(
            obs=obs, state=state, update_rnn_states=False, no_grad=no_grad
        )
        values = res_dict["values"]
        assert values.shape in [
            (batch_size, 1),
            (2 * batch_size, 1),
        ], f"values.shape={values.shape}"
        return values


def main() -> None:
    import pathlib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_to_this_dir = pathlib.Path(__file__).parent.absolute()

    CONFIG_PATH = path_to_this_dir / "config.yaml"
    CHECKPOINT_PATH = path_to_this_dir / "checkpoint.pt"
    NUM_OBSERVATIONS = 100
    NUM_ACTIONS = 100
    NUM_STATES = 200

    agent = RlAgent(
        num_observations=NUM_OBSERVATIONS,
        num_actions=NUM_ACTIONS,
        num_states=NUM_STATES,
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
    )

    batch_size = 2
    obs = torch.rand(batch_size, NUM_OBSERVATIONS).to(device)
    state = torch.rand(batch_size, NUM_STATES).to(device)
    normalized_action = agent.get_normalized_action(obs=obs, state=state)
    print(f"Using agent with config: {CONFIG_PATH} and checkpoint: {CHECKPOINT_PATH}")
    print(f"And num_observations: {NUM_OBSERVATIONS} and num_actions: {NUM_ACTIONS}")
    print(f"Sampled obs: {obs} with shape: {obs.shape}")
    print(
        f"Got normalized_action: {normalized_action} with shape: {normalized_action.shape}"
    )
    print(f"agent: {agent.agent.model}")


if __name__ == "__main__":
    main()
