import copy
import shutil
import threading
import time
from dataclasses import dataclass
from os.path import basename
from pathlib import Path
from typing import Any, Optional, Tuple

import gym
import numpy as np
import torch

from human2sim2robot.ppo.utils import models, torch_ext
from human2sim2robot.ppo.utils.helpers import unsqueeze_obs
from human2sim2robot.ppo.utils.network import NetworkConfig


@dataclass
class PpoPlayerConfig:
    normalize_input: bool
    clip_actions: bool = True
    device: str = "cuda:0"
    normalize_value: bool = False


@dataclass
class PlayerConfig:
    render: bool = False
    games_num: int = 2000
    deterministic: bool = True
    n_game_life: int = 1
    print_stats: bool = True
    render_sleep: float = 0.002
    evaluation: bool = False
    update_checkpoint_freq: int = 100
    dir_to_monitor: Optional[str] = None


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class PpoPlayer:
    def __init__(
        self,
        ppo_player_config: PpoPlayerConfig,
        player_config: PlayerConfig,
        network_config: NetworkConfig,
        env: Any,
    ):
        # Base part
        self.ppo_player_config = ppo_player_config
        self.player_config = player_config
        self.env = env
        self.env_info = env.get_env_info()

        self.num_agents = self.env_info.get("agents", 1)
        self.value_size = self.env_info.get("value_size", 1)
        self.action_space = self.env_info["action_space"]

        self.observation_space = self.env_info["observation_space"]
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        self.is_tensor_obses = False

        self.states = None
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False

        self.max_steps = 108000 // 4

        # if we run player as evaluation worker this will take care of loading new checkpoints
        # path to the newest checkpoint
        self.checkpoint_to_load: Optional[Path] = None
        if (
            self.player_config.evaluation
            and self.player_config.dir_to_monitor is not None
        ):
            self.checkpoint_mutex = threading.Lock()
            self.eval_checkpoint_dir = (
                Path(self.player_config.dir_to_monitor) / "eval_checkpoints"
            )
            self.eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)

            patterns = ["*.pth"]
            from watchdog.events import PatternMatchingEventHandler
            from watchdog.observers import Observer

            self.file_events = PatternMatchingEventHandler(patterns)
            self.file_events.on_created = self.on_file_created
            self.file_events.on_modified = self.on_file_modified

            self.file_observer = Observer()
            self.file_observer.schedule(
                self.file_events, self.player_config.dir_to_monitor, recursive=False
            )
            self.file_observer.start()

        # Custom part
        self.actions_num = self.action_space.shape[0]
        self.actions_low = (
            torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        )
        self.actions_high = (
            torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        )

        self.normalize_input = self.ppo_player_config.normalize_input
        self.normalize_value = self.ppo_player_config.normalize_value

        self.model = models.ModelA2CContinuousLogStd(
            network_config=network_config,
            actions_num=self.actions_num,
            input_shape=self.obs_shape,
            normalize_value=self.normalize_value,
            normalize_input=self.normalize_input,
            value_size=self.env_info.get("value_size", 1),
            num_seqs=self.num_agents,
        )

        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_deterministic=False) -> torch.Tensor:
        if not self.has_batch_dimension:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if not self.has_batch_dimension:
            current_action = torch.squeeze(current_action.detach())

        if self.ppo_player_config.clip_actions:
            return rescale_actions(
                low=self.actions_low,
                high=self.actions_high,
                action=torch.clamp(current_action, -1.0, 1.0),
            )
        else:
            return current_action

    def restore(self, filename: Path) -> None:
        checkpoint = torch_ext.load_checkpoint(filename)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

        env_state = checkpoint.get("env_state", None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

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

    def reset(self) -> None:
        self.init_rnn()

    def wait_for_checkpoint(self) -> None:
        if self.player_config.dir_to_monitor is None:
            return

        attempt = 0
        while True:
            attempt += 1
            with self.checkpoint_mutex:
                if self.checkpoint_to_load is not None:
                    if attempt % 10 == 0:
                        print(
                            f"Evaluation: waiting for new checkpoint in {self.player_config.dir_to_monitor}..."
                        )
                    break
            time.sleep(1.0)

        print(f"Checkpoint {self.checkpoint_to_load} is available!")

    def maybe_load_new_checkpoint(self) -> None:
        # lock mutex while loading new checkpoint
        with self.checkpoint_mutex:
            if self.checkpoint_to_load is not None:
                print(
                    f"Evaluation: loading new checkpoint {self.checkpoint_to_load}..."
                )
                # try if we can load anything from the pth file, this will quickly fail if the file is corrupted
                # without triggering the retry loop in "safe_filesystem_op()"
                load_error = False
                try:
                    torch.load(self.checkpoint_to_load)
                except Exception as e:
                    print(
                        f"Evaluation: checkpoint file is likely corrupted {self.checkpoint_to_load}: {e}"
                    )
                    load_error = True

                if not load_error:
                    try:
                        self.restore(self.checkpoint_to_load)
                    except Exception as e:
                        print(
                            f"Evaluation: failed to load new checkpoint {self.checkpoint_to_load}: {e}"
                        )

                # whether we succeeded or not, forget about this checkpoint
                self.checkpoint_to_load = None

    def process_new_eval_checkpoint(self, path: str) -> None:
        with self.checkpoint_mutex:
            # print(f"New checkpoint {path} available for evaluation")
            # copy file to eval_checkpoints dir using shutil
            # since we're running the evaluation worker in a separate process,
            # there is a chance that the file is changed/corrupted while we're copying it
            # not sure what we can do about this. In practice it never happened so far though
            try:
                eval_checkpoint_path = self.eval_checkpoint_dir / basename(path)
                shutil.copyfile(path, eval_checkpoint_path)
            except Exception as e:
                print(f"Failed to copy {path} to {eval_checkpoint_path}: {e}")
                return

            self.checkpoint_to_load = eval_checkpoint_path

    def on_file_created(self, event) -> None:
        self.process_new_eval_checkpoint(event.src_path)

    def on_file_modified(self, event) -> None:
        self.process_new_eval_checkpoint(event.src_path)

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

    def env_step(self, env, actions: torch.Tensor) -> Tuple:
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)
        if hasattr(obs, "dtype") and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return (
                self.obs_to_torch(obs),
                torch.from_numpy(rewards),
                torch.from_numpy(dones),
                infos,
            )

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if "obs" in obs:
                obs = obs["obs"]
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert obs.dtype != np.int8
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(obs)

    def get_weights(self) -> dict:
        weights = {}
        weights["model"] = self.model.state_dict()
        return weights

    def set_weights(self, weights) -> None:
        self.model.load_state_dict(weights["model"])
        if self.normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def init_rnn(self) -> None:
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            assert rnn_states is not None
            self.states = [
                torch.zeros(
                    (s.size()[0], self.batch_size, s.size()[2]), dtype=torch.float32
                ).to(self.device)
                for s in rnn_states
            ]

    def run(self) -> None:
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = self.player_config.games_num * self.player_config.n_game_life
        games_played = 0

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses=obses, batch_size=batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if (
                    self.player_config.evaluation
                    and n % self.player_config.update_checkpoint_freq == 0
                ):
                    self.maybe_load_new_checkpoint()

                action = self.get_action(
                    obs=obses, is_deterministic=self.player_config.deterministic
                )

                obses, r, done, info = self.env_step(env=self.env, actions=action)
                cr += r
                steps += 1

                if self.player_config.render:
                    self.env.render(mode="human")
                    time.sleep(self.player_config.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[:: self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if "battle_won" in info:
                            print_game_res = True
                            game_res = info.get("battle_won", 0.5)
                        if "scores" in info:
                            print_game_res = True
                            game_res = info.get("scores", 0.5)

                    if self.player_config.print_stats:
                        cur_rewards_done = cur_rewards / done_count
                        cur_steps_done = cur_steps / done_count
                        if print_game_res:
                            print(
                                f"reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}"
                            )
                        else:
                            print(
                                f"reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}"
                            )

                    sum_game_res += game_res
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print(
                "av reward:",
                sum_rewards / games_played * self.player_config.n_game_life,
                "av steps:",
                sum_steps / games_played * self.player_config.n_game_life,
                "winrate:",
                sum_game_res / games_played * self.player_config.n_game_life,
            )
        else:
            print(
                "av reward:",
                sum_rewards / games_played * self.player_config.n_game_life,
                "av steps:",
                sum_steps / games_played * self.player_config.n_game_life,
            )

    def get_batch_size(self, obses, batch_size: int) -> int:
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if "obs" in obses:
                obses = obses["obs"]
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if "observation" in obses:
                first_key = "observation"
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size

    @property
    def device(self) -> torch.device:
        return torch.device(self.ppo_player_config.device)

    @property
    def is_deterministic(self) -> bool:
        return self.player_config.deterministic
