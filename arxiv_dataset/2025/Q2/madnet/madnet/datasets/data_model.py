from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import jax
import numpy as np
from jax import Array


class DataSet(NamedTuple):
    a: Array  # treatment (n,)
    x: Array  # covariates (n, p)
    y: Array  # factual outcome (n,)
    fold: Array  # training fold (n,)
    mu: Array  # noiseless mean outcome (n,)
    moment: Array  # noiseless outcome moment (n,)

    @property
    def n(self) -> int:
        """Number of samples."""
        return self.x.shape[0]

    @property
    def p(self) -> int:
        """Covariate dimension."""
        return self.x.shape[1]

    @property
    def z(self) -> tuple[Array, Array]:
        """Inputs."""
        return (self.a, self.x)

    @property
    def w(self) -> tuple[Array, Array, Array]:
        """Inputs and labels."""
        return (self.a, self.x, self.y)

    def split(
        self, test_fold: int = 1, in_test: np.ndarray | None = None
    ) -> "DataSplit":
        if in_test is None:
            in_test = self.fold == test_fold

        data = (self.a, self.x, self.y, self.fold, self.mu, self.moment)
        train = jax.tree.map(lambda ds: ds[~in_test,], data)
        test = jax.tree.map(lambda ds: ds[in_test,], data)
        return DataSplit(train=DataSet(*train), test=DataSet(*test))


class DataSplit(NamedTuple):
    train: DataSet
    test: DataSet


@dataclass(frozen=True)
class SyntheticDataGenerator(ABC):
    max_seed: int | None

    def __call__(self, seed: int, **kwargs) -> DataSet:
        if self.max_seed is not None and seed > self.max_seed:
            raise ValueError(f"Only {self.max_seed} datasets are available")
        else:
            return self._generate(seed, **kwargs)

    @abstractmethod
    def _generate(self, seed: int) -> DataSet:
        raise NotImplementedError("Data generation method not set.")
