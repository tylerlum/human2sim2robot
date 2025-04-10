from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn as nn

from human2sim2robot.ppo.utils.layers.recurrent import RnnWithDones


@dataclass
class MlpConfig:
    units: Sequence[int]


@dataclass
class RnnConfig:
    units: int
    layers: int
    name: Literal["lstm", "gru"]
    layer_norm: bool = False
    before_mlp: bool = False
    concat_input: bool = False
    concat_output: bool = False


@dataclass
class NetworkConfig:
    mlp: MlpConfig
    rnn: Optional[RnnConfig] = None
    separate_value_mlp: bool = False
    asymmetric_critic: bool = False


class Network(nn.Module):
    def __init__(
        self,
        config: NetworkConfig,
        actions_num: int,
        input_shape: Sequence[int],
        value_size: int = 1,
        num_seqs: int = 1,
    ) -> None:
        super().__init__()
        self.config = config

        if self.asymmetric_critic:
            self.is_continuous = False
            self.fixed_sigma = False
        else:
            self.is_continuous = True
            self.fixed_sigma = True

        self.value_size = value_size
        self.num_seqs = num_seqs

        self.actor_mlp = nn.Sequential()
        self.critic_mlp = nn.Sequential()

        input_size = self._calc_input_size(input_shape)

        mlp_input_size = input_size
        if len(self.units) == 0:
            mlp_output_size = input_size
        else:
            mlp_output_size = self.units[-1]

        # rnn concat_input: rnn's new input will be its [original input, output from previous stage (if exists)]
        # rnn concat_output: rnn's new output will be its [original output, output from the previous stage (if exists)]
        if self.has_rnn:
            if not self.is_rnn_before_mlp:
                # mlp -> rnn

                # rnn(mlp(x))
                rnn_in_size = mlp_output_size

                if self.rnn_concat_input:
                    # rnn([mlp(x), x])
                    rnn_in_size += input_size

                # out = rnn(...)
                out_size = self.rnn_units

                if self.rnn_concat_output:
                    # out = rnn(...) + mlp(x)
                    out_size += mlp_output_size
            else:
                # rnn -> mlp

                # rnn(x)
                rnn_in_size = input_size

                # mlp(rnn(x))
                mlp_input_size = self.rnn_units

                if self.rnn_concat_output:
                    # mlp(rnn(x) + x)
                    mlp_input_size += input_size

                # out = mlp(...)
                out_size = mlp_output_size

            if self.separate_value_mlp:
                self.a_rnn = self._build_rnn(
                    self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                )
                self.c_rnn = self._build_rnn(
                    self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                )
                if self.rnn_ln:
                    self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                    self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
            else:
                self.rnn = self._build_rnn(
                    self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                )
                if self.rnn_ln:
                    self.layer_norm = torch.nn.LayerNorm(self.rnn_units)
        else:
            out_size = mlp_output_size

        self.actor_mlp = self._build_mlp(input_size=mlp_input_size, units=self.units)
        if self.separate_value_mlp:
            self.critic_mlp = self._build_mlp(
                input_size=mlp_input_size, units=self.units
            )

        self.value = torch.nn.Linear(out_size, self.value_size)
        self.value_act = nn.Identity()

        if self.is_continuous:
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = nn.Identity()
            self.sigma_act = nn.Identity()

            if self.fixed_sigma:
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            else:
                self.sigma = torch.nn.Linear(out_size, actions_num)

        mlp_init = nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        if self.is_continuous:
            mu_init = nn.Identity()
            sigma_init = partial(nn.init.constant_, val=0)
            mu_init(self.mu.weight)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

    def forward(self, obs_dict: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        obs = obs_dict["obs"]
        states = obs_dict.get("rnn_states", None)
        dones = obs_dict.get("dones", None)
        bptt_len = obs_dict.get("bptt_len", 0)

        if self.separate_value_mlp:
            a_out = c_out = obs
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            c_out = c_out.contiguous().view(c_out.size(0), -1)

            if self.has_rnn:
                seq_length = obs_dict.get("seq_length", 1)

                a_in = a_out
                c_in = c_out
                if not self.is_rnn_before_mlp:
                    # mlp -> rnn

                    # rnn(mlp(x))
                    a_out = self.actor_mlp(a_in)
                    c_out = self.critic_mlp(c_in)

                    a_mlp_out = a_out
                    c_mlp_out = c_out

                    if self.rnn_concat_input:
                        # rnn([mlp(x), x])
                        a_out = torch.cat([a_out, a_in], dim=1)
                        c_out = torch.cat([c_out, c_in], dim=1)

                batch_size = a_out.size()[0]
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1)
                c_out = c_out.reshape(num_seqs, seq_length, -1)

                a_out = a_out.transpose(0, 1)
                c_out = c_out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)

                if len(states) == 2:
                    a_states = states[0]
                    c_states = states[1]
                else:
                    a_states = states[:2]
                    c_states = states[2:]
                a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)

                a_out = a_out.transpose(0, 1)
                c_out = c_out.transpose(0, 1)
                a_out = a_out.contiguous().reshape(
                    a_out.size()[0] * a_out.size()[1], -1
                )
                c_out = c_out.contiguous().reshape(
                    c_out.size()[0] * c_out.size()[1], -1
                )

                if self.rnn_ln:
                    a_out = self.a_layer_norm(a_out)
                    c_out = self.c_layer_norm(c_out)

                if type(a_states) is not tuple:
                    a_states = (a_states,)
                    c_states = (c_states,)
                states = a_states + c_states

                if self.rnn_concat_output and not self.is_rnn_before_mlp:
                    # mlp -> rnn

                    # out = rnn(...) + mlp(x)
                    a_out = torch.cat([a_out, a_mlp_out], dim=1)
                    c_out = torch.cat([c_out, c_mlp_out], dim=1)
                elif self.rnn_concat_output and self.is_rnn_before_mlp:
                    # rnn -> mlp

                    # out = mlp(rnn(x) + x)
                    a_out = torch.cat([a_out, a_in], dim=1)
                    c_out = torch.cat([c_out, c_in], dim=1)
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)

            else:
                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)

            value = self.value_act(self.value(c_out))

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.fixed_sigma:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma, value, states
        else:
            out = obs
            out = out.flatten(1)

            if self.has_rnn:
                seq_length = obs_dict.get("seq_length", 1)

                in_ = out
                if not self.is_rnn_before_mlp:
                    # mlp -> rnn

                    # rnn(mlp(x))
                    out = self.actor_mlp(out)
                    mlp_out = out

                    if self.rnn_concat_input:
                        # rnn([mlp(x), x])
                        out = torch.cat([out, in_], dim=1)

                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)

                if len(states) == 1:
                    states = states[0]

                out = out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)
                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)
                if self.rnn_concat_output and not self.is_rnn_before_mlp:
                    # mlp -> rnn

                    # out = rnn(...) + mlp(x)
                    out = torch.cat([out, mlp_out], dim=1)
                elif self.rnn_concat_output and self.is_rnn_before_mlp:
                    # rnn -> mlp

                    # out = mlp(rnn(x) + x)
                    out = torch.cat([out, in_], dim=1)
                    out = self.actor_mlp(out)
                elif not self.rnn_concat_output and self.is_rnn_before_mlp:
                    # rnn -> mlp

                    # out = mlp(rnn(x))
                    out = self.actor_mlp(out)
                if type(states) is not tuple:
                    states = (states,)
            else:
                out = self.actor_mlp(out)
            value = self.value_act(self.value(out))

            if self.asymmetric_critic:
                return value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu * 0 + sigma, value, states

    def is_separate_critic(self) -> bool:
        return self.separate_value_mlp

    def is_rnn(self) -> bool:
        return self.has_rnn

    def get_default_rnn_state(self) -> Optional[Tuple[torch.Tensor, ...]]:
        if not self.has_rnn:
            return None
        num_layers = self.rnn_layers
        rnn_units = self.rnn_units
        if self.rnn_name == "lstm":
            if self.separate_value_mlp:
                return (
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                )
            else:
                return (
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                )
        else:
            if self.separate_value_mlp:
                return (
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                )
            else:
                return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)

    @property
    def separate_value_mlp(self) -> bool:
        return self.config.separate_value_mlp

    @property
    def units(self) -> Sequence[int]:
        return self.config.mlp.units

    @property
    def has_rnn(self) -> bool:
        return self.config.rnn is not None

    @property
    def asymmetric_critic(self) -> bool:
        return self.config.asymmetric_critic

    @property
    def rnn_units(self) -> int:
        return self.config.rnn.units

    @property
    def rnn_layers(self) -> int:
        return self.config.rnn.layers

    @property
    def rnn_name(self) -> Literal["lstm", "gru"]:
        return self.config.rnn.name

    @property
    def rnn_ln(self) -> bool:
        return self.config.rnn.layer_norm

    @property
    def is_rnn_before_mlp(self) -> bool:
        return self.config.rnn.before_mlp

    @property
    def rnn_concat_input(self) -> bool:
        return self.config.rnn.concat_input

    @property
    def rnn_concat_output(self) -> bool:
        return self.config.rnn.concat_output

    def get_value_layer(self) -> nn.Module:
        return self.value

    def _calc_input_size(self, input_shape: Sequence[int]) -> int:
        assert len(input_shape) == 1
        return input_shape[0]

    def _build_rnn(
        self, name: Literal["lstm", "gru"], input: int, units: int, layers: int
    ) -> RnnWithDones:
        if name == "lstm":
            return RnnWithDones(
                rnn_layer=torch.nn.LSTM(
                    input_size=input, hidden_size=units, num_layers=layers
                ),
            )
        if name == "gru":
            return RnnWithDones(
                rnn_layer=torch.nn.GRU(
                    input_size=input, hidden_size=units, num_layers=layers
                ),
            )
        raise ValueError(f"Unknown rnn type: {name}")

    def _build_mlp(self, input_size: int, units: Sequence[int]) -> nn.Sequential:
        in_size = input_size
        layers = []
        for unit in units:
            layers.append(torch.nn.Linear(in_size, unit))
            layers.append(torch.nn.ELU())
            in_size = unit

        return nn.Sequential(*layers)
