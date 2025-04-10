import dataclasses
from typing import (
    Any,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from omegaconf import MISSING

T = TypeVar("T")


def is_optional_type(t: Any) -> bool:
    """Check if a type is Optional[...]"""
    return get_origin(t) is Union and type(None) in get_args(t)


def get_optional_inner_type(t: Any) -> Any:
    """If t is Optional[InnerType], return InnerType."""
    if is_optional_type(t):
        args = get_args(t)
        # args looks like (InnerType, NoneType), find the non-None one
        return args[0] if args[1] is type(None) else args[1]
    return t


def dict_to_dataclass(d: dict, d_class: Type[T]) -> T:
    if not dataclasses.is_dataclass(d_class):
        raise ValueError(f"{d_class} is not a dataclass type")

    fields = dataclasses.fields(d_class)
    field_names = {f.name for f in fields}

    extra_keys = set(d.keys()) - field_names
    if extra_keys:
        raise ValueError(
            f"Extra keys in dictionary not in {d_class.__name__} fields: {extra_keys}"
        )

    kwargs = {}
    for f in fields:
        if f.name not in d:
            # Field not in dict, check for defaults
            if (
                f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
            ) or f.default == MISSING:
                raise ValueError(
                    f"Missing required field '{f.name}' for {d_class.__name__} and no default is defined"
                )
            if f.default_factory is not dataclasses.MISSING:
                kwargs[f.name] = f.default_factory()  # type: ignore
            else:
                kwargs[f.name] = f.default
        else:
            value = d[f.name]
            f_type = f.type

            # Handle optional fields
            if is_optional_type(f_type):
                inner_type = get_optional_inner_type(f_type)
                if value is None or (isinstance(value, dict) and not value):
                    # None or empty dict -> None
                    kwargs[f.name] = None
                else:
                    # If inner_type is a dataclass, recurse
                    if dataclasses.is_dataclass(inner_type):
                        if not isinstance(value, dict):
                            raise ValueError(
                                f"Field '{f.name}' of {d_class.__name__} expects a dict or None to create {inner_type.__name__}, got {type(value)}"
                            )
                        kwargs[f.name] = dict_to_dataclass(value, inner_type)
                    else:
                        # Just assign directly for non-dataclass optionals
                        kwargs[f.name] = value
            else:
                # Non-optional field
                if dataclasses.is_dataclass(f_type):
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"Field '{f.name}' of {d_class.__name__} expects a dict to create {f_type.__name__}, got {type(value)}"
                        )
                    kwargs[f.name] = dict_to_dataclass(value, f_type)
                else:
                    kwargs[f.name] = value

    return d_class(**kwargs)
