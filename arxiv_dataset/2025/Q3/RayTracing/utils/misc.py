import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterable, Iterator, Tuple, TypeVar

import equinox as eqx
import jax
import yaml
from jaxtyping import PyTree

T = TypeVar("T")


def triplewise(iterable: Iterable[T]) -> Iterator[Tuple[T, T, T]]:
    """Return successive overlapping triplets from iterable.

    Similar to pairwise() but returns 3-element tuples.

    Examples:
        >>> list(triplewise('ABCDEF'))
        [('A', 'B', 'C'), ('B', 'C', 'D'), ('C', 'D', 'E'), ('D', 'E', 'F')]

        >>> list(triplewise([1, 2, 3, 4, 5]))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    """
    iterator = iter(iterable)
    a = next(iterator, None)
    b = next(iterator, None)
    if a is None or b is None:
        return
    for c in iterator:
        yield a, b, c
        a = b
        b = c


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def load_config_from_yaml(path: str) -> dict:
    """Load a config file from a yaml."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resulting_dim_conv(
    kernel_size: int, stride: int, padding: int, dilation: int, input_dim: int
) -> int:
    """Calculate the resulting dimension of a convolutional layer."""
    return (input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def save_model(filename: str | Path, hyperparams: dict, model: PyTree):
    """Save an equinox model (weights and hyperparams/structure) to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(EasyDict(hyperparams))
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(filename: str | Path, get_model: Callable) -> eqx.Module:
    """Load an equinox model whose weights/structure are saved in `filename`,
    and whose instantiation is given by the function `get_model`."""
    with open(filename, "rb") as f:
        rng = jax.random.key(
            0
        )  # required to initialize params that are overwritten by deserialise_leaves
        hyperparams = json.loads(f.readline().decode())
        model = get_model(EasyDict(hyperparams), rng)
        return eqx.tree_deserialise_leaves(f, model), hyperparams
