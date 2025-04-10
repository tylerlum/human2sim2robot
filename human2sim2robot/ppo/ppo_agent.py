import copy
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch import nn, optim

from human2sim2robot.ppo.ppo_player import PpoPlayerConfig
from human2sim2robot.ppo.utils import (
    asymmetric_critic,
    datasets,
    models,
    schedulers,
    torch_ext,
)
from human2sim2robot.ppo.utils.asymmetric_critic import AsymmetricCriticConfig
from human2sim2robot.ppo.utils.experience import ExperienceBuffer
from human2sim2robot.ppo.utils.moving_mean_std import GeneralizedMovingStats
from human2sim2robot.ppo.utils.network import NetworkConfig
from human2sim2robot.ppo.utils.rewards_shaper import (
    DefaultRewardsShaper,
    RewardsShaperParams,
)


@dataclass
class PpoConfig:
    # Required fields (accessed via config["key"])
    num_actors: int
    learning_rate: float
    entropy_coef: float
    horizon_length: int
    normalize_advantage: bool
    normalize_input: bool
    grad_norm: float
    critic_coef: float
    gamma: float
    tau: float
    reward_shaper: RewardsShaperParams
    mini_epochs: int
    e_clip: float

    # Fields with defaults (accessed via config.get(..., default))
    multi_gpu: bool = False
    device: str = "cuda:0"
    weight_decay: float = 0.0
    asymmetric_critic: Optional[AsymmetricCriticConfig] = None
    truncate_grads: bool = False
    save_frequency: int = 0
    save_best_after: int = 100
    print_stats: bool = True
    max_epochs: int = -1
    max_frames: int = -1
    lr_schedule: Optional[str] = None
    schedule_type: Literal["legacy", "standard"] = "legacy"
    kl_threshold: Optional[float] = None
    seq_length: int = 4
    bptt_length: Optional[int] = None  # If not provided, use seq_length at runtime
    zero_rnn_on_done: bool = True
    normalize_rms_advantage: bool = False
    normalize_value: bool = False
    games_to_track: int = 100
    minibatch_size_per_env: int = 0
    minibatch_size: Optional[int] = None
    mixed_precision: bool = False
    bounds_loss_coef: Optional[float] = None
    bound_loss_type: Literal["bound", "regularisation"] = "bound"
    value_bootstrap: bool = False
    adv_rms_momentum: float = 0.5
    clip_actions: bool = True
    schedule_entropy: bool = False
    freeze_critic: bool = False

    def to_ppo_player_config(self) -> PpoPlayerConfig:
        return PpoPlayerConfig(
            normalize_input=self.normalize_input,
            clip_actions=self.clip_actions,
            device=self.device,
            normalize_value=self.normalize_value,
        )


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


def print_statistics(
    print_stats: bool,
    curr_frames: float,
    step_time: float,
    step_inference_time: float,
    total_time: float,
    epoch_num: int,
    max_epochs: int,
    frame: float,
    max_frames: int,
) -> None:
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        if max_epochs == -1 and max_frames == -1:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}"
            )
        elif max_epochs == -1:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{max_frames:.0f}"
            )
        elif max_frames == -1:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}"
            )
        else:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}/{max_frames:.0f}"
            )


class PpoAgent:
    def __init__(
        self,
        experiment_dir: Path,
        ppo_config: PpoConfig,
        network_config: NetworkConfig,
        env: Any,
    ):
        self.cfg = ppo_config

        ## A2CBase ##
        # multi-gpu/multi-node data
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        if self.cfg.multi_gpu:
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            dist.init_process_group(
                "nccl", rank=self.global_rank, world_size=self.world_size
            )

            self.device_name = "cuda:" + str(self.local_rank)
            self.cfg.device = self.device_name
            if self.global_rank != 0:
                self.cfg.print_stats = False
                self.cfg.lr_schedule = None

        self.env = env
        self.env_info = self.env.get_env_info()
        self.value_size = self.env_info.get("value_size", 1)
        self.observation_space = self.env_info["observation_space"]
        self.num_agents = self.env_info.get("agents", 1)

        self.has_asymmetric_critic = self.cfg.asymmetric_critic is not None
        if self.has_asymmetric_critic:
            self.state_space = self.env_info.get("state_space", None)
            if isinstance(self.state_space, gym.spaces.Dict):
                self.state_shape = {}
                for k, v in self.state_space.spaces.items():
                    self.state_shape[k] = v.shape
            else:
                self.state_shape = self.state_space.shape

        self.rnn_states = None

        # Setting learning rate scheduler
        if self.cfg.lr_schedule == "adaptive":
            assert self.cfg.kl_threshold is not None
            self.scheduler = schedulers.AdaptiveScheduler(
                kl_threshold=self.cfg.kl_threshold
            )

        elif self.cfg.lr_schedule == "linear":
            if self.cfg.max_epochs == -1 and self.cfg.max_frames == -1:
                print(
                    "Max epochs and max frames are not set. Linear learning rate schedule can't be used, switching to the contstant (identity) one."
                )
                self.scheduler = schedulers.IdentityScheduler()
            else:
                use_epochs = True
                max_steps = self.cfg.max_epochs

                if self.cfg.max_epochs == -1:
                    use_epochs = False
                    max_steps = self.cfg.max_frames

                self.scheduler = schedulers.LinearScheduler(
                    float(self.cfg.learning_rate),
                    max_steps=max_steps,
                    use_epochs=use_epochs,
                    apply_to_entropy=self.cfg.schedule_entropy,
                    start_entropy_coef=self.cfg.entropy_coef,
                )
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.rewards_shaper = DefaultRewardsShaper(self.cfg.reward_shaper)

        self.bptt_len = (
            self.cfg.bptt_length
            if self.cfg.bptt_length is not None
            else self.cfg.seq_length
        )  # not used right now. Didn't show that it is usefull

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape

        print("current training device:", self.device)
        self.game_rewards = torch_ext.AverageMeter(
            in_shape=self.value_size, max_size=self.cfg.games_to_track
        ).to(self.device)
        self.game_shaped_rewards = torch_ext.AverageMeter(
            in_shape=self.value_size, max_size=self.cfg.games_to_track
        ).to(self.device)
        self.game_lengths = torch_ext.AverageMeter(
            in_shape=1, max_size=self.cfg.games_to_track
        ).to(self.device)
        self.obs = None

        self.batch_size_envs = self.cfg.horizon_length * self.cfg.num_actors
        self.batch_size = self.batch_size_envs * self.num_agents

        # either minibatch_size_per_env or minibatch_size should be present in a config
        # if both are present, minibatch_size is used
        # otherwise minibatch_size_per_env is used minibatch_size_per_env is used to calculate minibatch_size
        self.minibatch_size = (
            self.cfg.minibatch_size
            if self.cfg.minibatch_size is not None
            else self.cfg.num_actors * self.cfg.minibatch_size_per_env
        )
        assert self.minibatch_size > 0

        self.num_minibatches = self.batch_size // self.minibatch_size
        assert self.batch_size % self.minibatch_size == 0, (
            f"{self.batch_size}, {self.minibatch_size}"
        )
        assert self.num_minibatches > 0, f"{self.batch_size}, {self.minibatch_size}"

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        self.current_lr = float(self.cfg.learning_rate)
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0

        self.experiment_dir = experiment_dir
        self.nn_dir = self.experiment_dir / "nn"
        self.summaries_dir = self.experiment_dir / "summaries"

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.nn_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        self.current_entropy_coef = self.cfg.entropy_coef

        if self.global_rank == 0:
            writer = SummaryWriter(str(self.summaries_dir))
            self.writer = writer
        else:
            self.writer = None

        if self.cfg.normalize_advantage and self.cfg.normalize_rms_advantage:
            momentum = self.cfg.adv_rms_momentum
            self.advantage_mean_std = GeneralizedMovingStats(
                insize=1, decay=momentum
            ).to(self.device)

        self.is_tensor_obses = False

        self.last_rnn_indices = None
        self.last_state_indices = None

        ## ContinuousA2CBase ##
        action_space = self.env_info["action_space"]
        self.actions_num = action_space.shape[0]

        self.actions_low = (
            torch.from_numpy(action_space.low.copy()).float().to(self.device)
        )
        self.actions_high = (
            torch.from_numpy(action_space.high.copy()).float().to(self.device)
        )

        ## A2CAgent ##
        self.model = models.ModelA2CContinuousLogStd(
            network_config=network_config,
            actions_num=self.actions_num,
            input_shape=self.obs_shape,
            normalize_value=self.cfg.normalize_value,
            normalize_input=self.cfg.normalize_input,
            value_size=self.env_info.get("value_size", 1),
            num_seqs=self.cfg.num_actors * self.num_agents,
        )

        self.model.to(self.device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            float(self.current_lr),
            eps=1e-08,
            weight_decay=self.cfg.weight_decay,
        )

        if self.has_asymmetric_critic:
            print("Adding Asymmetric Critic Network")
            assert self.cfg.asymmetric_critic is not None
            self.asymmetric_critic_net = asymmetric_critic.AsymmetricCritic(
                state_shape=self.state_shape,
                value_size=self.value_size,
                ppo_device=self.device,
                num_agents=self.num_agents,
                horizon_length=self.cfg.horizon_length,
                num_actors=self.cfg.num_actors,
                num_actions=self.actions_num,
                seq_length=self.cfg.seq_length,
                normalize_value=self.cfg.normalize_value,
                config=self.cfg.asymmetric_critic,
                writer=self.writer,
                max_epochs=self.cfg.max_epochs,
                multi_gpu=self.cfg.multi_gpu,
                zero_rnn_on_done=self.cfg.zero_rnn_on_done,
            ).to(self.device)

        self.dataset = datasets.PPODataset(
            batch_size=self.batch_size,
            minibatch_size=self.minibatch_size,
            is_rnn=self.is_rnn,
            device=self.device,
            seq_length=self.cfg.seq_length,
        )
        if self.cfg.normalize_value:
            self.value_mean_std = (
                self.asymmetric_critic_net.model.value_mean_std
                if self.has_asymmetric_critic
                else self.model.value_mean_std
            )

    def truncate_gradients_and_step(self) -> None:
        if self.cfg.multi_gpu:
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
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def write_stats(
        self,
        total_time: float,
        epoch_num: int,
        step_time: float,
        play_time: float,
        update_time: float,
        a_losses: List[torch.Tensor],
        c_losses: List[torch.Tensor],
        entropies: List[torch.Tensor],
        kls: List[torch.Tensor],
        current_lr: float,
        lr_mul: float,
        frame: int,
        scaled_time: float,
        scaled_play_time: float,
        curr_frames: float,
    ) -> None:
        if self.writer is None:
            print("writer is None, skipping writing stats")
            return

        # do we need scaled time?
        self.writer.add_scalar(
            tag="performance/step_inference_rl_update_fps",
            scalar_value=curr_frames / scaled_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_inference_fps",
            scalar_value=curr_frames / scaled_play_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_fps",
            scalar_value=curr_frames / step_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/rl_update_time",
            scalar_value=update_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_inference_time",
            scalar_value=play_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_time", scalar_value=step_time, global_step=frame
        )
        self.writer.add_scalar(
            tag="losses/a_loss",
            scalar_value=torch.mean(torch.stack(a_losses)).item(),
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="losses/c_loss",
            scalar_value=torch.mean(torch.stack(c_losses)).item(),
            global_step=frame,
        )

        self.writer.add_scalar(
            tag="losses/entropy",
            scalar_value=torch.mean(torch.stack(entropies)).item(),
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="info/current_lr", scalar_value=current_lr * lr_mul, global_step=frame
        )
        self.writer.add_scalar(
            tag="info/lr_mul", scalar_value=lr_mul, global_step=frame
        )
        self.writer.add_scalar(
            tag="info/e_clip", scalar_value=self.cfg.e_clip * lr_mul, global_step=frame
        )
        self.writer.add_scalar(
            tag="info/kl",
            scalar_value=torch.mean(torch.stack(kls)).item(),
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="info/epochs", scalar_value=epoch_num, global_step=frame
        )

    def set_eval(self) -> None:
        self.model.eval()
        if self.cfg.normalize_rms_advantage:
            self.advantage_mean_std.eval()

    def set_train(self) -> None:
        self.model.train()
        if self.cfg.normalize_rms_advantage:
            self.advantage_mean_std.train()

    def update_lr(self, lr: float) -> None:
        if self.cfg.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # if self.has_asymmetric_critic:
        #    self.asymmetric_critic_net.update_lr(lr)

    def get_action_values(self, obs) -> dict:
        processed_obs = self._preproc_obs(obs["obs"])
        self.model.eval()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "rnn_states": self.rnn_states,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_asymmetric_critic:
                states = obs["states"]
                input_dict = {
                    "is_train": False,
                    "states": states,
                }
                value = self.get_asymmetric_critic_value(input_dict)
                res_dict["values"] = value
        return res_dict

    def get_values(self, obs) -> torch.Tensor:
        with torch.no_grad():
            if self.has_asymmetric_critic:
                states = obs["states"]
                self.asymmetric_critic_net.eval()
                input_dict = {
                    "is_train": False,
                    "states": states,
                    "actions": None,
                    "is_done": self.dones,
                }
                value = self.get_asymmetric_critic_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs["obs"])
                input_dict = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": processed_obs,
                    "rnn_states": self.rnn_states,
                }
                result = self.model(input_dict)
                value = result["values"]
            return value

    @property
    def device(self) -> Union[str, torch.device]:
        return self.cfg.device

    def reset_envs(self) -> None:
        self.obs = self.env_reset()

    def init_tensors(self) -> None:
        ## A2CBase ##
        batch_size = self.num_agents * self.cfg.num_actors
        self.experience_buffer = ExperienceBuffer(
            env_info=self.env_info,
            num_actors=self.cfg.num_actors,
            horizon_length=self.cfg.horizon_length,
            has_asymmetric_critic=self.has_asymmetric_critic,
            device=self.device,
        )

        _val_shape = (self.cfg.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(
            current_rewards_shape, dtype=torch.float32, device=self.device
        )
        self.current_shaped_rewards = torch.zeros(
            current_rewards_shape, dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.device) for s in self.rnn_states]

            total_agents = self.num_agents * self.cfg.num_actors
            num_seqs = self.cfg.horizon_length // self.cfg.seq_length
            assert (
                self.cfg.horizon_length * total_agents // self.num_minibatches
            ) % self.cfg.seq_length == 0
            self.mb_rnn_states = [
                torch.zeros(
                    (num_seqs, s.size()[0], total_agents, s.size()[2]),
                    dtype=torch.float32,
                    device=self.device,
                )
                for s in self.rnn_states
            ]

        ## ContinuousA2CBase ##
        self.update_list = ["actions", "neglogpacs", "values", "mus", "sigmas"]
        self.tensor_list = self.update_list + ["obses", "states", "dones"]

    def init_rnn_from_model(self, model) -> None:
        self.is_rnn = self.model.is_rnn()

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert obs.dtype != np.int8
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        return obs

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or "obs" not in obs:
            upd_obs = {"obs": upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(
                self.actions_low, self.actions_high, clamped_actions
            )
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def env_step(self, actions: torch.Tensor) -> Tuple:
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return (
                self.obs_to_tensors(obs),
                rewards.to(self.device),
                dones.to(self.device),
                infos,
            )
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return (
                self.obs_to_tensors(obs),
                torch.from_numpy(rewards).to(self.device).float(),
                torch.from_numpy(dones).to(self.device),
                infos,
            )

    def env_reset(self):
        obs = self.env.reset()
        obs = self.obs_to_tensors(obs)
        return obs

    def discount_values(
        self,
        fdones: torch.Tensor,
        last_extrinsic_values: torch.Tensor,
        mb_fdones: torch.Tensor,
        mb_extrinsic_values: torch.Tensor,
        mb_rewards: torch.Tensor,
    ) -> torch.Tensor:
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.cfg.horizon_length)):
            if t == self.cfg.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t + 1]
                nextvalues = mb_extrinsic_values[t + 1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = (
                mb_rewards[t]
                + self.cfg.gamma * nextvalues * nextnonterminal
                - mb_extrinsic_values[t]
            )
            mb_advs[t] = lastgaelam = (
                delta + self.cfg.gamma * self.cfg.tau * nextnonterminal * lastgaelam
            )
        return mb_advs

    def clear_stats(self) -> None:
        self.game_rewards.clear()
        self.game_shaped_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -100500

    def update_epoch(self) -> int:
        self.epoch_num += 1
        return self.epoch_num

    def train(self) -> Tuple[float, int]:
        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.cfg.multi_gpu:
            torch.cuda.set_device(self.local_rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            if self.has_asymmetric_critic:
                model_params.append(self.asymmetric_critic_net.state_dict())
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])
            if self.has_asymmetric_critic:
                self.asymmetric_critic_net.load_state_dict(model_params[1])

        while True:
            epoch_num = self.update_epoch()
            (
                step_time,
                play_time,
                update_time,
                sum_time,
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                current_lr,
                lr_mul,
            ) = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = (
                    self.curr_frames * self.world_size
                    if self.cfg.multi_gpu
                    else self.curr_frames
                )
                self.frame += curr_frames

                print_statistics(
                    print_stats=self.cfg.print_stats,
                    curr_frames=curr_frames,
                    step_time=step_time,
                    step_inference_time=scaled_play_time,
                    total_time=scaled_time,
                    epoch_num=epoch_num,
                    max_epochs=self.cfg.max_epochs,
                    frame=frame,
                    max_frames=self.cfg.max_frames,
                )

                self.write_stats(
                    total_time=total_time,
                    epoch_num=epoch_num,
                    step_time=step_time,
                    play_time=play_time,
                    update_time=update_time,
                    a_losses=a_losses,
                    c_losses=c_losses,
                    entropies=entropies,
                    kls=kls,
                    current_lr=current_lr,
                    lr_mul=lr_mul,
                    frame=frame,
                    scaled_time=scaled_time,
                    scaled_play_time=scaled_play_time,
                    curr_frames=curr_frames,
                )

                if len(b_losses) > 0:
                    self.writer.add_scalar(
                        tag="losses/bounds_loss",
                        scalar_value=torch.mean(torch.stack(b_losses)).item(),
                        global_step=frame,
                    )

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = "rewards" if i == 0 else "rewards{0}".format(i)
                        self.writer.add_scalar(
                            tag=rewards_name + "/step".format(),
                            scalar_value=mean_rewards[i],
                            global_step=frame,
                        )
                        self.writer.add_scalar(
                            tag=rewards_name + "/iter".format(),
                            scalar_value=mean_rewards[i],
                            global_step=epoch_num,
                        )
                        self.writer.add_scalar(
                            tag=rewards_name + "/time".format(),
                            scalar_value=mean_rewards[i],
                            global_step=total_time,
                        )
                        self.writer.add_scalar(
                            tag="shaped_" + rewards_name + "/step".format(),
                            scalar_value=mean_shaped_rewards[i],
                            global_step=frame,
                        )
                        self.writer.add_scalar(
                            tag="shaped_" + rewards_name + "/iter".format(),
                            scalar_value=mean_shaped_rewards[i],
                            global_step=epoch_num,
                        )
                        self.writer.add_scalar(
                            tag="shaped_" + rewards_name + "/time".format(),
                            scalar_value=mean_shaped_rewards[i],
                            global_step=total_time,
                        )

                    self.writer.add_scalar(
                        tag="episode_lengths/step",
                        scalar_value=mean_lengths,
                        global_step=frame,
                    )
                    self.writer.add_scalar(
                        tag="episode_lengths/iter",
                        scalar_value=mean_lengths,
                        global_step=epoch_num,
                    )
                    self.writer.add_scalar(
                        tag="episode_lengths/time",
                        scalar_value=mean_lengths,
                        global_step=total_time,
                    )

                    if self.cfg.save_frequency > 0:
                        if epoch_num % self.cfg.save_frequency == 0:
                            self.save(
                                self.nn_dir
                                / f"ep_{epoch_num}_rew_{mean_rewards[0]}.pth"
                            )

                    if (
                        mean_rewards[0] > self.last_mean_rewards
                        and epoch_num >= self.cfg.save_best_after
                    ):
                        print("saving next best rewards: ", mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(self.nn_dir / "best.pth")

                if epoch_num >= self.cfg.max_epochs and self.cfg.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print(
                            "WARNING: Max epochs reached before any env terminated at least once"
                        )
                        mean_rewards = -np.inf

                    self.save(
                        self.nn_dir
                        / f"last_ep_{epoch_num}_rew_{str(mean_rewards).replace('[', '_').replace(']', '_')}.pth"
                    )
                    print("MAX EPOCHS NUM!")
                    should_exit = True

                if self.frame >= self.cfg.max_frames and self.cfg.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print(
                            "WARNING: Max frames reached before any env terminated at least once"
                        )
                        mean_rewards = -np.inf

                    self.save(
                        self.nn_dir
                        / f"last_frame_{self.frame}_rew_{str(mean_rewards).replace('[', '_').replace(']', '_')}.pth"
                    )
                    print("MAX FRAMES NUM!")
                    should_exit = True

                update_time = 0

            if self.cfg.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num

    def prepare_dataset(self, batch_dict) -> None:
        obses = batch_dict["obses"]
        returns = batch_dict["returns"]
        dones = batch_dict["dones"]
        values = batch_dict["values"]
        actions = batch_dict["actions"]
        neglogpacs = batch_dict["neglogpacs"]
        mus = batch_dict["mus"]
        sigmas = batch_dict["sigmas"]
        rnn_states = batch_dict.get("rnn_states", None)

        advantages = returns - values

        if self.cfg.normalize_value:
            if self.cfg.freeze_critic:
                self.value_mean_std.eval()
            else:
                self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, dim=1)

        if self.cfg.normalize_advantage:
            if self.is_rnn:
                if self.cfg.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
            else:
                if self.cfg.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

        dataset_dict = {}
        dataset_dict["old_values"] = values
        dataset_dict["old_logp_actions"] = neglogpacs
        dataset_dict["advantages"] = advantages
        dataset_dict["returns"] = returns
        dataset_dict["actions"] = actions
        dataset_dict["obs"] = obses
        dataset_dict["dones"] = dones
        dataset_dict["rnn_states"] = rnn_states
        dataset_dict["mu"] = mus
        dataset_dict["sigma"] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_asymmetric_critic:
            dataset_dict = {}
            dataset_dict["old_values"] = values
            dataset_dict["advantages"] = advantages
            dataset_dict["returns"] = returns
            dataset_dict["actions"] = actions
            dataset_dict["obs"] = batch_dict["states"]
            dataset_dict["dones"] = dones
            self.asymmetric_critic_net.update_dataset(dataset_dict)

    def train_epoch(
        self,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        float,
        float,
    ]:
        ## A2CBase ##
        self.env.set_train_info(self.frame, self)

        ## ContinuousA2CBase ##
        self.set_eval()
        play_time_start = time.perf_counter()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()

        self.set_train()
        self.curr_frames = batch_dict.pop("played_frames")
        self.prepare_dataset(batch_dict)
        if self.has_asymmetric_critic:
            self.train_asymmetric_critic()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.cfg.mini_epochs):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, current_lr, lr_mul, cmu, csigma, b_loss = (
                    self.train_actor_critic(self.dataset[i])
                )
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.cfg.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.cfg.schedule_type == "legacy":
                    av_kls = kl
                    if self.cfg.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.world_size
                    self.current_lr, self.current_entropy_coef = self.scheduler.update(
                        self.current_lr,
                        self.current_entropy_coef,
                        self.epoch_num,
                        0,
                        av_kls.item(),
                    )
                    self.update_lr(self.current_lr)

            av_kls = torch.mean(torch.stack(ep_kls))
            if self.cfg.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size
            if self.cfg.schedule_type == "standard":
                self.current_lr, self.current_entropy_coef = self.scheduler.update(
                    self.current_lr,
                    self.current_entropy_coef,
                    self.epoch_num,
                    0,
                    av_kls.item(),
                )
                self.update_lr(self.current_lr)

            kls.append(av_kls)
            if self.cfg.normalize_input:
                self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch

        update_time_end = time.perf_counter()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return (
            batch_dict["step_time"],
            play_time,
            update_time,
            total_time,
            a_losses,
            c_losses,
            b_losses,
            entropies,
            kls,
            current_lr,
            lr_mul,
        )

    def train_actor_critic(
        self, input_dict
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        float,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute gradients needed to step the networks of the algorithm.

        Core algo logic is defined here

        Args:
            input_dict (:obj:`dict`): Algo inputs as a dict.

        """
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        returns_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.cfg.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        if self.is_rnn:
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.cfg.seq_length

            if self.cfg.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropies = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
            a_losses = torch.max(-surr1, -surr2)

            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -curr_e_clip, curr_e_clip
            )
            value_losses = (values - returns_batch) ** 2
            value_losses_clipped = (value_pred_clipped - returns_batch) ** 2
            c_losses = torch.max(value_losses, value_losses_clipped)
            c_losses = c_losses.squeeze(dim=1)

            if self.cfg.bounds_loss_coef is not None:
                if self.cfg.bound_loss_type == "regularisation":
                    b_losses = (mu * mu).sum(dim=-1)
                elif self.cfg.bound_loss_type == "bound":
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_losses = (mu_loss_low + mu_loss_high).sum(dim=-1)
                else:
                    raise ValueError(
                        f"Unknown bound loss type {self.cfg.bound_loss_type}"
                    )
            else:
                b_losses = torch.zeros(1, device=self.device)

            a_loss, c_loss, entropy, b_loss = (
                a_losses.mean(),
                c_losses.mean(),
                entropies.mean(),
                b_losses.mean(),
            )
            loss = (
                a_loss
                + 0.5 * c_loss * self.cfg.critic_coef
                - entropy * self.current_entropy_coef
                + b_loss * self.cfg.bounds_loss_coef
            )
            if self.cfg.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        # TODO: Refactor this ugliest code of the year
        self.truncate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = True
            kl_dist = torch_ext.policy_kl(
                mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl
            )

        return (
            a_loss,
            c_loss,
            entropy,
            kl_dist,
            self.current_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
        )

    def get_asymmetric_critic_value(self, obs_dict) -> torch.Tensor:
        return self.asymmetric_critic_net.get_value(obs_dict)

    def train_asymmetric_critic(self) -> float:
        return self.asymmetric_critic_net.train_net()

    def get_full_state_weights(self) -> Dict[str, Any]:
        state = self.get_weights()
        state["epoch"] = self.epoch_num
        state["frame"] = self.frame
        state["optimizer"] = self.optimizer.state_dict()

        if self.has_asymmetric_critic:
            state["assymetric_vf_nets"] = self.asymmetric_critic_net.state_dict()

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state["last_mean_rewards"] = self.last_mean_rewards

        if self.env is not None:
            env_state = self.env.get_env_state()
            state["env_state"] = env_state

        return state

    def set_full_state_weights(self, weights, set_epoch=True) -> None:
        self.set_weights(weights)
        if set_epoch:
            self.epoch_num = weights["epoch"]
            self.frame = weights["frame"]

        if self.has_asymmetric_critic:
            self.asymmetric_critic_net.load_state_dict(weights["assymetric_vf_nets"])

        self.optimizer.load_state_dict(weights["optimizer"])

        self.last_mean_rewards = weights.get("last_mean_rewards", -1000000000)

        if self.env is not None:
            env_state = weights.get("env_state", None)
            self.env.set_env_state(env_state)

    def get_weights(self) -> Dict[str, Any]:
        state = self.get_stats_weights()
        state["model"] = self.model.state_dict()
        return state

    def get_stats_weights(self, model_stats=False) -> Dict[str, Any]:
        state = {}
        if self.cfg.mixed_precision:
            state["scaler"] = self.scaler.state_dict()
        if self.has_asymmetric_critic:
            state["central_val_stats"] = self.asymmetric_critic_net.get_stats_weights(
                model_stats
            )
        if model_stats:
            if self.cfg.normalize_input:
                state["running_mean_std"] = self.model.running_mean_std.state_dict()
            if self.cfg.normalize_value:
                state["reward_mean_std"] = self.model.value_mean_std.state_dict()

        return state

    def set_stats_weights(self, weights) -> None:
        if self.cfg.normalize_rms_advantage:
            self.advantage_mean_std.load_state_dict(weights["advantage_mean_std"])
        if self.cfg.normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])
        if self.cfg.normalize_value and "normalize_value" in weights:
            self.model.value_mean_std.load_state_dict(weights["reward_mean_std"])
        if self.cfg.mixed_precision and "scaler" in weights:
            self.scaler.load_state_dict(weights["scaler"])

    def set_weights(self, weights) -> None:
        self.model.load_state_dict(weights["model"])
        self.set_stats_weights(weights)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
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

    def play_steps(self) -> Dict[str, Any]:
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.cfg.horizon_length):
            res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_asymmetric_critic:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            step_time_start = time.perf_counter()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            step_time_end = time.perf_counter()

            step_time += step_time_end - step_time_start

            shaped_rewards = self.rewards_shaper(rewards)
            if self.cfg.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.cfg.gamma
                    * res_dict["values"]
                    * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
                )

            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(
                self.current_shaped_rewards[env_done_indices]
            )
            self.game_lengths.update(self.current_lengths[env_done_indices])

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = (
                self.current_shaped_rewards * not_dones.unsqueeze(1)
            )
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_advs = self.discount_values(
            fdones=fdones,
            last_extrinsic_values=last_values,
            mb_fdones=mb_fdones,
            mb_extrinsic_values=mb_values,
            mb_rewards=mb_rewards,
        )
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(
            swap_and_flatten01, self.tensor_list
        )
        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size
        batch_dict["step_time"] = step_time

        return batch_dict

    def play_steps_rnn(self) -> Dict[str, Any]:
        update_list = self.update_list
        mb_rnn_states = self.mb_rnn_states
        step_time = 0.0

        for n in range(self.cfg.horizon_length):
            if n % self.cfg.seq_length == 0:
                for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                    mb_s[n // self.cfg.seq_length, :, :, :] = s

            if self.has_asymmetric_critic:
                self.asymmetric_critic_net.pre_step_rnn(n)

            res_dict = self.get_action_values(self.obs)

            self.rnn_states = res_dict["rnn_states"]
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones.byte())

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_asymmetric_critic:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            step_time_start = time.perf_counter()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            step_time_end = time.perf_counter()

            step_time += step_time_end - step_time_start

            shaped_rewards = self.rewards_shaper(rewards)

            if self.cfg.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.cfg.gamma
                    * res_dict["values"]
                    * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
                )

            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[:: self.num_agents]

            if len(all_done_indices) > 0:
                if self.cfg.zero_rnn_on_done:
                    for s in self.rnn_states:
                        s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_asymmetric_critic:
                    self.asymmetric_critic_net.post_step_rnn(all_done_indices)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(
                self.current_shaped_rewards[env_done_indices]
            )
            self.game_lengths.update(self.current_lengths[env_done_indices])

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = (
                self.current_shaped_rewards * not_dones.unsqueeze(1)
            )
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()

        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_advs = self.discount_values(
            fdones, last_values, mb_fdones, mb_values, mb_rewards
        )
        mb_returns = mb_advs + mb_values
        batch_dict = self.experience_buffer.get_transformed_list(
            transform_op=swap_and_flatten01, tensor_list=self.tensor_list
        )

        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size
        states = []
        for mb_s in mb_rnn_states:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))

        batch_dict["rnn_states"] = states
        batch_dict["step_time"] = step_time

        return batch_dict

    def save(self, filename: Path) -> None:
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(filename=filename, state=state)

    def restore(self, filename: Path, set_epoch: bool = True) -> None:
        checkpoint = torch_ext.load_checkpoint(filename)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def override_sigma(self, sigma) -> None:
        net = self.model.network
        if hasattr(net, "sigma") and hasattr(net, "fixed_sigma"):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(sigma))
            else:
                print("Print cannot set new sigma because fixed_sigma is False")
        else:
            print("Print cannot set new sigma because sigma is not present")
