from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from human2sim2robot.ppo.utils.network import Network, NetworkConfig
from human2sim2robot.ppo.utils.running_mean_std import RunningMeanStd


class BaseModel(nn.Module):
    def __init__(
        self,
        network_config: NetworkConfig,
        actions_num: int,
        input_shape: Sequence[int],
        normalize_value: bool,
        normalize_input: bool,
        value_size: int,
        num_seqs: int = 1,
    ) -> None:
        nn.Module.__init__(self)
        self.a2c_network = Network(
            config=network_config,
            actions_num=actions_num,
            input_shape=input_shape,
            value_size=value_size,
            num_seqs=num_seqs,
        )
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input

        if normalize_value:
            self.value_mean_std = RunningMeanStd((value_size,))
        if normalize_input:
            self.running_mean_std = RunningMeanStd(input_shape)

    def is_rnn(self):
        return self.a2c_network.is_rnn()

    def get_value_layer(self):
        return self.a2c_network.get_value_layer()

    def get_default_rnn_state(self):
        return self.a2c_network.get_default_rnn_state()

    def norm_obs(self, observation):
        with torch.no_grad():
            return (
                self.running_mean_std(observation)
                if self.normalize_input
                else observation
            )

    def denorm_value(self, value):
        with torch.no_grad():
            return (
                self.value_mean_std(value, denorm=True)
                if self.normalize_value
                else value
            )


class ModelA2CContinuousLogStd(BaseModel):
    def forward(self, input_dict):
        is_train = input_dict.get("is_train", True)
        prev_actions = input_dict.get("prev_actions", None)
        input_dict["obs"] = self.norm_obs(input_dict["obs"])
        mu, logstd, value, states = self.a2c_network(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma, validate_args=False)
        if is_train:
            entropy = distr.entropy().sum(dim=-1)
            prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
            result = {
                "prev_neglogp": torch.squeeze(prev_neglogp),
                "values": value,
                "entropy": entropy,
                "rnn_states": states,
                "mus": mu,
                "sigmas": sigma,
            }
            return result
        else:
            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, sigma, logstd)
            result = {
                "neglogpacs": torch.squeeze(neglogp),
                "values": self.denorm_value(value),
                "actions": selected_action,
                "rnn_states": states,
                "mus": mu,
                "sigmas": sigma,
            }
            return result

    def neglogp(self, x, mean, std, logstd):
        return (
            0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
            + logstd.sum(dim=-1)
        )


class ModelAsymmetricCritic(BaseModel):
    def forward(self, input_dict):
        is_train = input_dict.get("is_train", True)
        _prev_actions = input_dict.get("prev_actions", None)
        input_dict["obs"] = self.norm_obs(input_dict["obs"])
        value, states = self.a2c_network(input_dict)
        if not is_train:
            value = self.denorm_value(value)

        result = {"values": value, "rnn_states": states}
        return result
