from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, TypeVar

from ..types import Example, Batch

Datum = TypeVar("Datum", Example, Batch)


class Transform(ABC):
    """Abstract transform interface."""

    @abstractmethod
    def __call__(self, data: Datum) -> Datum:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        return self.__class__.__name__ + "({!s})"


class ExampleTransform(Transform):
    """Transforms an example."""

    @abstractmethod
    def __call__(self, example: Example) -> Example:
        pass

    def __repr__(self) -> str:
        _base = super().__repr__()
        return "[example-wise]" + _base

    def batchify(self, collate_fn: callable = list) -> Map:
        """Converts an example transform to a batch transform."""
        return Map(self, collate_fn)


class Map(Transform):
    """Maps a transform to a batch of examples."""

    def __init__(self, transform: ExampleTransform, collate_fn: callable = list):
        self.transform = transform
        self.collate_fn = collate_fn

    def __call__(self, batch: Iterator[Example]) -> Batch:
        return self.collate_fn(list(map(self.transform, batch)))

    def __repr__(self) -> str:
        _base = super().__repr__()
        # strip the [example-wise] prefix from the self.transform repr
        _instance_repr = repr(self.transform)
        if _instance_repr.startswith("[example-wise]"):
            _instance_repr = _instance_repr[len("[example-wise]") :]
        args_repr = f"{_instance_repr}, collate_fn={self.collate_fn.__name__}"
        return "[batch-wise]" + _base.format(args_repr)


class Dispatch(dict, Transform):
    """Dispatches a transform to an example based on a key field.

    Attributes:
        self: A map of key to transform.
    """

    def __call__(self, data: Datum) -> Datum:
        """Apply each transform to the field of an example matching its key."""
        result = {}
        try:
            for key, transform in self.items():
                result[key] = transform(getattr(data, key))
        except KeyError as exc:
            raise TypeError(
                f"Invalid {key=} in transforms. All keys need to match the "
                f"fields of an example."
            ) from exc

        return data._replace(**result)

    def __repr__(self) -> str:
        _base = Transform.__repr__(self)
        transforms_repr = ", ".join(
            f"{key}: {repr(transform)}" for key, transform in self.items()
        )
        return _base.format(transforms_repr)


class Compose(list, Transform):
    """Creates a transform from a sequence of transforms."""

    def __call__(self, data: Datum) -> Datum:
        for transform in self:
            data = transform(data)
        return data

    def __repr__(self) -> str:
        transforms_repr = " \u2192 ".join(repr(transform) for transform in self)
        return f"[{transforms_repr}]"
