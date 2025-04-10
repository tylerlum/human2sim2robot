import os
import random
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionory."""
    if type(val) is dict:
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        print(val)


def set_seed(seed, rank=0):
    """set seed across modules"""
    if seed == -1:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def match_value(key: str, *cases) -> Any:
    """
    Return a corresponding result for the given `key` by matching it
    against pairs in `cases` (each pair is (possible_key, result_value)) and a default_value.

    If no match is found in `(possible_key, result_value)`, return `default_value`.

    Args:
        key: The key to match on.
        *cases: A sequence of alternating (match_key, result_value) pairs and a default_value.

    Returns:
        The first matching result_value in `cases`, or `default_value` otherwise.

    Raises:
        AssertionError: If the `cases` argument doesn't have an even number of items.
    """
    # We expect pairs, so `cases` must have even length
    assert len(cases) % 2 == 1, (
        f"Expected an odd number of arguments for pairs + default_value, got {len(cases)}: {cases}. key: {key}"
    )

    n_pairs = len(cases) // 2
    possible_keys = [cases[2 * i] for i in range(n_pairs)]
    result_values = [cases[2 * i + 1] for i in range(n_pairs)]
    default_value = cases[-1]

    # Go through each pair: (possible_key, result_value)
    for possible_key, result_value in zip(possible_keys, result_values):
        if key == possible_key:
            return result_value

    # If no match, return the default
    return default_value


def add_omegaconf_resolvers() -> None:
    OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver("match_value", match_value)
    OmegaConf.register_new_resolver("eval", eval)
