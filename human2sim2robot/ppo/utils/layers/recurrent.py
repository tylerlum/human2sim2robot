from typing import Any, Optional, Tuple, Union

import torch
from torch import nn


def multiply_hidden(
    h: Union[torch.Tensor, Tuple[Any, ...]], mask: torch.Tensor
) -> Union[torch.Tensor, Tuple[Any, ...]]:
    if isinstance(h, torch.Tensor):
        return h * mask
    else:
        return tuple(multiply_hidden(v, mask) for v in h)


class RnnWithDones(nn.Module):
    def __init__(self, rnn_layer: nn.Module) -> None:
        nn.Module.__init__(self)
        self.rnn = rnn_layer

    # got idea from ikostrikov :)
    def forward(
        self,
        input: torch.Tensor,
        states: Any,
        done_masks: Optional[torch.Tensor] = None,
        bptt_len: int = 0,
    ) -> Tuple[torch.Tensor, Any]:
        # ignoring bptt_len for now
        if done_masks is None:
            return self.rnn(input, states)

        max_steps = input.size()[0]
        _batch_size = input.size()[1]
        out_batch = []
        not_dones = 1.0 - done_masks
        has_zeros = (
            (not_dones.squeeze()[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
        )
        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [max_steps]
        out_batch = []

        for i in range(len(has_zeros) - 1):
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]
            not_done = not_dones[start_idx].float().unsqueeze(0)
            states = multiply_hidden(states, not_done)
            out, states = self.rnn(input[start_idx:end_idx], states)
            out_batch.append(out)
        return torch.cat(out_batch, dim=0), states
