import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from human2sim2robot.ppo.utils import datasets, models, schedulers
from human2sim2robot.ppo.utils.network import NetworkConfig


@dataclass
class AsymmetricCriticConfig:
    name: str  # Required
    network: NetworkConfig  # Required
    normalize_input: bool  # Required (no default given in code)
    learning_rate: float  # Required
    mini_epochs: int  # Required

    # Optional fields with defaults based on get(..., default)
    truncate_grads: bool = False
    freeze_critic: bool = False
    lr_schedule: Optional[Literal["linear"]] = None
    minibatch_size_per_env: int = 0
    minibatch_size: Optional[int] = None
    weight_decay: float = 0.0
    grad_norm: float = 1.0
    e_clip: float = 0.2
    freeze_critic: bool = False


class AsymmetricCritic(nn.Module):
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        value_size: int,
        ppo_device: Union[str, torch.device],
        num_agents: int,
        horizon_length: int,
        num_actors: int,
        num_actions: int,
        seq_length: int,
        normalize_value: bool,
        config: AsymmetricCriticConfig,
        writer: Any,
        max_epochs: int,
        multi_gpu: bool,
        zero_rnn_on_done: bool,
    ) -> None:
        nn.Module.__init__(self)
        self.cfg = config

        self.ppo_device = ppo_device
        self.num_agents, self.horizon_length, self.num_actors, self.seq_length = (
            num_agents,
            horizon_length,
            num_actors,
            seq_length,
        )
        self.normalize_value = normalize_value
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.value_size = value_size
        self.max_epochs = max_epochs
        self.multi_gpu = multi_gpu
        self.zero_rnn_on_done = zero_rnn_on_done

        self.model = models.ModelAsymmetricCritic(
            network_config=self.cfg.network,
            actions_num=num_actions,
            input_shape=state_shape,
            normalize_value=self.normalize_value,
            normalize_input=self.cfg.normalize_input,
            value_size=value_size,
            num_seqs=num_actors,
        )
        self.lr = float(self.cfg.learning_rate)

        # todo: support max frames as well
        if self.cfg.lr_schedule == "linear":
            self.scheduler = schedulers.LinearScheduler(
                self.lr,
                max_steps=self.max_epochs,
                apply_to_entropy=False,
                start_entropy_coef=0,
            )
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.minibatch_size = (
            config.minibatch_size
            if config.minibatch_size is not None
            else self.num_actors * self.cfg.minibatch_size_per_env
        )
        assert self.minibatch_size > 0
        self.batch_size = self.horizon_length * self.num_actors
        self.num_minibatches = self.batch_size // self.minibatch_size

        self.writer = writer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            float(self.lr),
            eps=1e-08,
            weight_decay=self.cfg.weight_decay,
        )
        self.frame = 0
        self.epoch_num = 0
        self.running_mean_std = None

        self.is_rnn = self.model.is_rnn()
        self.rnn_states = None

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]
            total_agents = self.num_actors
            num_seqs = self.horizon_length // self.seq_length
            assert (
                self.horizon_length * total_agents // self.num_minibatches
            ) % self.seq_length == 0
            self.mb_rnn_states = [
                torch.zeros(
                    (num_seqs, s.size()[0], total_agents, s.size()[2]),
                    dtype=torch.float32,
                    device=self.ppo_device,
                )
                for s in self.rnn_states
            ]

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1
        if self.multi_gpu:
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            self.device_name = "cuda:" + str(self.local_rank)

        self.dataset = datasets.PPODataset(
            batch_size=self.batch_size,
            minibatch_size=self.minibatch_size,
            is_rnn=self.is_rnn,
            device=self.ppo_device,
            seq_length=self.seq_length,
        )

    def update_lr(self, lr: float) -> None:
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device_name)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_stats_weights(self, model_stats: bool = False) -> Dict[str, Any]:
        state = {}
        if model_stats:
            if self.cfg.normalize_input:
                state["running_mean_std"] = self.model.running_mean_std.state_dict()
            if self.normalize_value:
                state["reward_mean_std"] = self.model.value_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights: Dict[str, Any]) -> None:
        pass

    def update_dataset(self, batch_dict: Dict[str, torch.Tensor]) -> None:
        value_preds = batch_dict["old_values"]
        returns = batch_dict["returns"]
        actions = batch_dict["actions"]
        dones = batch_dict["dones"]
        if self.num_agents > 1:
            res = self.update_multiagent_tensors(value_preds, returns, actions, dones)
            batch_dict["old_values"] = res[0]
            batch_dict["returns"] = res[1]
            batch_dict["actions"] = res[2]
            batch_dict["dones"] = res[3]

        if self.is_rnn:
            states = []
            for mb_s in self.mb_rnn_states:
                t_size = mb_s.size()[0] * mb_s.size()[2]
                h_size = mb_s.size()[3]
                states.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))

            batch_dict["rnn_states"] = states
        self.dataset.update_values_dict(batch_dict)

    def _preproc_obs(self, obs_batch: Any) -> Any:
        if isinstance(obs_batch, dict):
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def pre_step_rnn(self, n: int) -> None:
        if not self.is_rnn:
            return
        if n % self.seq_length == 0:
            for s, mb_s in zip(self.rnn_states, self.mb_rnn_states):
                mb_s[n // self.seq_length, :, :, :] = s

    def post_step_rnn(
        self, all_done_indices: torch.Tensor, zero_rnn_on_done: bool = True
    ) -> None:
        if not self.is_rnn:
            return
        if not self.zero_rnn_on_done:
            return
        all_done_indices = all_done_indices[:: self.num_agents] // self.num_agents
        for s in self.rnn_states:
            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

    def forward(self, input_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.model(input_dict)

    def get_value(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        self.eval()
        obs_batch = input_dict["states"]
        actions = input_dict.get("actions", None)

        obs_batch = self._preproc_obs(obs_batch)
        res_dict = self.forward(
            {
                "obs": obs_batch,
                "actions": actions,
                "rnn_states": self.rnn_states,
                "is_train": False,
            }
        )
        value, self.rnn_states = res_dict["values"], res_dict["rnn_states"]
        if self.num_agents > 1:
            value = value.repeat(1, self.num_agents)
            value = value.view(value.size()[0] * self.num_agents, -1)

        return value

    def train_critic(self, input_dict: Dict[str, Any]) -> float:
        self.train()
        loss = self.calc_gradients(input_dict)
        return loss.item()

    def update_multiagent_tensors(
        self,
        value_preds: torch.Tensor,
        returns: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = self.batch_size
        ma_batch_size = self.num_actors * self.num_agents * self.horizon_length
        value_preds = value_preds.view(
            self.num_actors, self.num_agents, self.horizon_length, self.value_size
        ).transpose(0, 1)
        returns = returns.view(
            self.num_actors, self.num_agents, self.horizon_length, self.value_size
        ).transpose(0, 1)
        value_preds = value_preds.contiguous().view(ma_batch_size, self.value_size)[
            :batch_size
        ]
        returns = returns.contiguous().view(ma_batch_size, self.value_size)[:batch_size]
        dones = dones.contiguous().view(ma_batch_size, self.value_size)[:batch_size]

        return value_preds, returns, actions, dones

    def train_net(self) -> float:
        self.train()
        loss = 0
        for _ in range(self.cfg.mini_epochs):
            if self.cfg.freeze_critic:
                break
            for idx in range(len(self.dataset)):
                loss += self.train_critic(self.dataset[idx])
            if self.cfg.normalize_input:
                self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch

        avg_loss = loss / (self.cfg.mini_epochs * self.num_minibatches)

        self.epoch_num += 1
        self.lr, _ = self.scheduler.update(self.lr, 0, self.epoch_num, 0, 0)
        self.update_lr(self.lr)
        self.frame += self.batch_size
        if self.writer is not None:
            self.writer.add_scalar("losses/cval_loss", avg_loss, self.frame)
            self.writer.add_scalar("info/cval_lr", self.lr, self.frame)
        return avg_loss

    def calc_gradients(self, batch: Dict[str, Any]) -> torch.Tensor:
        obs_batch = self._preproc_obs(batch["obs"])
        value_preds_batch = batch["old_values"]
        returns_batch = batch["returns"]
        actions_batch = batch["actions"]
        dones_batch = batch["dones"]

        batch_dict = {
            "obs": obs_batch,
            "actions": actions_batch,
            "seq_length": self.seq_length,
            "dones": dones_batch,
        }
        if self.is_rnn:
            batch_dict["rnn_states"] = batch["rnn_states"]

        res_dict = self.model(batch_dict)
        values = res_dict["values"]

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.cfg.e_clip, self.cfg.e_clip
        )
        value_losses = (values - returns_batch) ** 2
        value_losses_clipped = (value_pred_clipped - returns_batch) ** 2
        c_loss = torch.max(value_losses, value_losses_clipped)
        c_loss = c_loss.squeeze(dim=1)

        loss = c_loss.mean()
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None
        loss.backward()

        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))
            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(
                            param.grad.data
                        )
                        / self.world_size
                    )
                    offset += param.numel()

        if self.cfg.truncate_grads:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm)

        self.optimizer.step()

        return loss
