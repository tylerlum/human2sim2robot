from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np


def dicts_to_dict_with_arrays(
    dicts: List[Dict[Any, Any]], add_batch_dim: bool = True
) -> Union[List[Dict[Any, Any]], Dict[Any, Any]]:
    def stack(v: List[Any]) -> np.ndarray:
        if len(np.shape(v)) == 1:
            return np.array(v)
        else:
            return np.stack(v)

    def concatenate(v: List[Any]) -> np.ndarray:
        if len(np.shape(v)) == 1:
            return np.array(v)
        else:
            return np.concatenate(v)

    dicts_len = len(dicts)
    if dicts_len <= 1:
        return dicts
    res = defaultdict(list)
    {res[key].append(sub[key]) for sub in dicts for key in sub}
    if add_batch_dim:
        concat_func = stack
    else:
        concat_func = concatenate

    res = {k: concat_func(v) for k, v in res.items()}
    return res


def unsqueeze_obs(obs: Any) -> Any:
    if type(obs) is dict:
        for k, v in obs.items():
            obs[k] = unsqueeze_obs(v)
    else:
        if len(obs.size()) > 1 or obs.size()[0] > 1:
            obs = obs.unsqueeze(0)
    return obs
