from typing import Any, Dict, Union

import torch
from torch.utils.data import Dataset


class PPODataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        minibatch_size: int,
        is_rnn: bool,
        device: Union[str, torch.device],
        seq_length: int,
    ) -> None:
        self.is_rnn = is_rnn
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = self.batch_size // self.minibatch_size
        total_games = self.batch_size // self.seq_length
        self.num_games_batch = self.minibatch_size // self.seq_length
        self.game_indexes = torch.arange(
            total_games, dtype=torch.long, device=self.device
        )
        self.flat_indexes = torch.arange(
            total_games * self.seq_length, dtype=torch.long, device=self.device
        ).reshape(total_games, self.seq_length)

    def update_values_dict(self, values_dict: Dict[str, Any]) -> None:
        self.values_dict = values_dict

    def update_mu_sigma(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        start = self.last_range[0]
        end = self.last_range[1]
        self.values_dict["mu"][start:end] = mu
        self.values_dict["sigma"][start:end] = sigma

    def __len__(self) -> int:
        return self.length

    def _get_item_rnn(self, idx: int) -> Dict[str, Any]:
        gstart = idx * self.num_games_batch
        gend = (idx + 1) * self.num_games_batch
        start = gstart * self.seq_length
        end = gend * self.seq_length
        self.last_range = (start, end)

        input_dict = {}
        for k, v in self.values_dict.items():
            if v is None:
                input_dict[k] = None
                continue

            if k == "rnn_states":
                rnn_states = v
                input_dict["rnn_states"] = [
                    s[:, gstart:gend, :].contiguous() for s in rnn_states
                ]
                continue

            if isinstance(v, dict):
                v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[start:end]

        return input_dict

    def _get_item(self, idx: int) -> Dict[str, Any]:
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.values_dict.items():
            if v is None:
                input_dict[k] = None
                continue

            if k == "rnn_states":
                continue

            if isinstance(v, dict):
                v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[start:end]

        return input_dict

    def __getitem__(self, idx: int) -> Any:
        if self.is_rnn:
            sample = self._get_item_rnn(idx)
        else:
            sample = self._get_item(idx)
        return sample
