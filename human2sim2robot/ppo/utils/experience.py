from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch

from human2sim2robot.ppo.utils.torch_ext import numpy_to_torch_dtype_dict


class ExperienceBuffer:
    """
    More generalized than replay buffers.
    Implemented for on-policy algorithms.
    """

    def __init__(
        self,
        env_info: Dict[str, Any],
        num_actors: int,
        horizon_length: int,
        has_asymmetric_critic: bool,
        device: Union[str, torch.device],
        aux_tensor_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.env_info = env_info
        self.device = device

        self.num_agents = env_info.get("agents", 1)
        self.action_space = env_info["action_space"]

        self.num_actors = num_actors
        self.horizon_length = horizon_length
        self.has_asymmetric_critic = has_asymmetric_critic
        batch_size = self.num_actors * self.num_agents
        self.is_discrete = False
        self.is_multi_discrete = False
        self.is_continuous = False
        self.obs_base_shape = (self.horizon_length, batch_size)
        self.state_base_shape = (self.horizon_length, self.num_actors)
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.actions_shape = ()
            self.actions_num = self.action_space.n
            self.is_discrete = True
        if isinstance(self.action_space, gym.spaces.Tuple):
            self.actions_shape = (len(self.action_space),)
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        if isinstance(self.action_space, gym.spaces.Box):
            self.actions_shape = (self.action_space.shape[0],)
            self.actions_num = self.action_space.shape[0]
            self.is_continuous = True
        self.tensor_dict = {}
        self._init_from_env_info(self.env_info)

        self.aux_tensor_dict = aux_tensor_dict
        if self.aux_tensor_dict is not None:
            self._init_from_aux_dict(self.aux_tensor_dict)

    def _init_from_env_info(self, env_info: Dict[str, Any]) -> None:
        obs_base_shape = self.obs_base_shape
        state_base_shape = self.state_base_shape

        self.tensor_dict["obses"] = self._create_tensor_from_space(
            env_info["observation_space"], obs_base_shape
        )
        if self.has_asymmetric_critic:
            self.tensor_dict["states"] = self._create_tensor_from_space(
                env_info["state_space"], state_base_shape
            )

        val_space = gym.spaces.Box(
            low=0, high=1, shape=(env_info.get("value_size", 1),)
        )
        self.tensor_dict["rewards"] = self._create_tensor_from_space(
            val_space, obs_base_shape
        )
        self.tensor_dict["values"] = self._create_tensor_from_space(
            val_space, obs_base_shape
        )
        self.tensor_dict["neglogpacs"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=(), dtype=np.float32), obs_base_shape
        )
        self.tensor_dict["dones"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=(), dtype=np.uint8), obs_base_shape
        )

        if self.is_discrete or self.is_multi_discrete:
            self.tensor_dict["actions"] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=self.actions_shape, dtype=int),
                obs_base_shape,
            )
        if self.is_continuous:
            self.tensor_dict["actions"] = self._create_tensor_from_space(
                gym.spaces.Box(
                    low=0, high=1, shape=self.actions_shape, dtype=np.float32
                ),
                obs_base_shape,
            )
            self.tensor_dict["mus"] = self._create_tensor_from_space(
                gym.spaces.Box(
                    low=0, high=1, shape=self.actions_shape, dtype=np.float32
                ),
                obs_base_shape,
            )
            self.tensor_dict["sigmas"] = self._create_tensor_from_space(
                gym.spaces.Box(
                    low=0, high=1, shape=self.actions_shape, dtype=np.float32
                ),
                obs_base_shape,
            )

    def _init_from_aux_dict(self, tensor_dict: Dict[str, Any]) -> None:
        obs_base_shape = self.obs_base_shape
        for k, v in tensor_dict.items():
            self.tensor_dict[k] = self._create_tensor_from_space(
                gym.spaces.Box(low=0, high=1, shape=(v), dtype=np.float32),
                obs_base_shape,
            )

    def _create_tensor_from_space(
        self, space: gym.Space, base_shape: Tuple[int, ...]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(space, gym.spaces.Box):
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(
                base_shape + space.shape, dtype=dtype, device=self.device
            )
        if isinstance(space, gym.spaces.Discrete):
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape, dtype=dtype, device=self.device)
        if isinstance(space, gym.spaces.Tuple):
            # Assuming that tuple is only a Discrete tuple
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            tuple_len = len(space)
            return torch.zeros(
                base_shape + (tuple_len,), dtype=dtype, device=self.device
            )
        if isinstance(space, gym.spaces.Dict):
            t_dict = {}
            for k, v in space.spaces.items():
                t_dict[k] = self._create_tensor_from_space(v, base_shape)
            return t_dict

    def update_data(self, name: str, index: int, val: Any) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                self.tensor_dict[name][k][index, :] = v
        else:
            self.tensor_dict[name][index, :] = val

    def get_transformed(
        self, transform_op: Callable[[torch.Tensor], torch.Tensor]
    ) -> Dict[str, Any]:
        res_dict = {}
        for k, v in self.tensor_dict.items():
            if isinstance(v, dict):
                transformed_dict = {}
                for kd, vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)
        return res_dict

    def get_transformed_list(
        self,
        transform_op: Callable[[torch.Tensor], torch.Tensor],
        tensor_list: List[str],
    ) -> Dict[str, Any]:
        res_dict = {}
        for k in tensor_list:
            v = self.tensor_dict.get(k)
            if v is None:
                continue
            if isinstance(v, dict):
                transformed_dict = {}
                for kd, vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)
        return res_dict
