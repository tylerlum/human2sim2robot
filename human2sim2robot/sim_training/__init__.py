import functools
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


@functools.lru_cache()
def get_sim_training_dir() -> Path:
    return Path(__file__).parent.absolute()


@functools.lru_cache()
def get_cfg_dir() -> Path:
    return get_sim_training_dir() / "cfg"


@functools.lru_cache()
def get_package_root_dir() -> Path:
    return get_sim_training_dir().parent


@functools.lru_cache()
def get_repo_root_dir() -> Path:
    return get_package_root_dir().parent


@functools.lru_cache()
def get_asset_root() -> Path:
    return get_repo_root_dir() / "assets"


@functools.lru_cache()
def get_data_dir() -> Path:
    return get_repo_root_dir() / "data"


@functools.lru_cache()
def datetime_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("datetime_str", lambda: datetime_str())
