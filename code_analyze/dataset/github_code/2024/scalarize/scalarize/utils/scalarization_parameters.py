#!/usr/bin/env python3

r"""
Transformations for the scalarization parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module


class ScalarizationParameterTransform(Module, ABC):
    r"""Abstract base class for scalarization parameter transformations."""

    parameter_dim: int
    latent_dim: int
    bounds: List[Tuple[float, float]]

    def __init__(self, num_objectives: int, **kwargs) -> None:
        r"""Base constructor for scalarization parameter transformations.

        Args:
            num_objectives: The number of objectives.
        """
        super().__init__()
        self.num_objectives = num_objectives

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Transform the latent scalarization parameter to the scalarization
        parameter.

        Args:
            X: A `batch_shape x latent_dim`-dim Tensor containing the latent
                scalarization parameters.

        Returns:
            transformed_X: A `batch_shape x parameter_dim`-dim Tensor containing the
                scalarization parameters.
        """
        pass  # pragma: no cover


class SimplexWeight(ScalarizationParameterTransform):
    r"""Weights lying in the probability simplex."""

    def __init__(
        self,
        num_objectives: int,
        transform_label: str = "normalize",
    ) -> None:
        r"""Weights lying in the probability simplex.

        Args:
            num_objectives: The number of objectives.
            transform_label: The label of the transformation.
        """
        super().__init__(num_objectives=num_objectives)
        self.parameter_dim = num_objectives

        transformations = {
            "normalize": [self.normalize, num_objectives],
            "scale": [self.scale, num_objectives - 1],
            "log_normalize": [self.log_normalize, num_objectives],
        }
        self.transform_label = transform_label
        self.transform = transformations[transform_label][0]
        self.latent_dim = transformations[transform_label][1]
        self.bounds = [(0, 1) for _ in range(self.latent_dim)]

    @staticmethod
    def normalize(X: Tensor):
        r"""Transform a batch of vectors in the hypercube to the probability simplex
        by normalizing the vectors using their sums:

        `transform(X) = X / sum(X)`.

        Args:
            X: A `batch_shape x num_objectives`-dim Tensor containing the latent
                scalarization parameters.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        return X / torch.sum(X, dim=-1, keepdim=True)

    @staticmethod
    def scale(X: Tensor) -> Tensor:
        r"""Transform a batch of vectors in the hypercube to the probability simplex
        via the scaling strategy:

        `M = num_objectives`

        `transform(X)[0] = X[0]`,
        `transform(X)[1] = X[1] * (1 - transform(X)[0])`,
        `transform(X)[2] = X[2] * (1 - transform(X)[0] - transform(X)[1])`,
        ...
        `transform(X)[M-1] = (1 - transform(X)[0] - ... - transform(X)[M-2])`.

        Note this could be slow when the number of objectives is increased as the
        computations are performed iteratively.

        Args:
            X: A `batch_shape x (num_objectives-1)`-dim Tensor containing the latent
                scalarization parameters.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        tkwargs = {"dtype": X.dtype, "device": X.device}
        num_objectives = X.shape[-1] + 1
        transformed_shape = torch.Size([*X.shape[:-1], torch.tensor([num_objectives])])

        # `batch_shape x (latent_dim + 1)`
        transformed_X = torch.zeros(transformed_shape, **tkwargs)

        for m in range(0, num_objectives - 1):
            transformed_X[..., m] = X[..., m]
            if m > 0:
                remainder = 1.0 - torch.sum(transformed_X[..., 0:m], dim=-1)
                transformed_X[..., m] = remainder * X[..., m]

        transformed_X[..., -1] = 1 - torch.sum(transformed_X[..., 0:-1], dim=-1)

        return transformed_X

    @staticmethod
    def log_normalize(X: Tensor) -> Tensor:
        r"""Transform a batch of vectors in the hypercube to the probability simplex
        by normalizing the negative logarithm vectors using their sums:

        `transform(X) = -log(X) / sum(-log(X))`.

        Args:
            X: A `batch_shape x num_objectives`-dim Tensor containing the latent
                scalarization parameters.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        # add eps to avoid torch.log(0)
        eps = torch.finfo(X.dtype).eps
        transformed_X = -torch.log(eps + (1 - eps) * X)
        return SimplexWeight.normalize(transformed_X)

    def forward(self, X: Tensor) -> Tensor:
        r"""Transform a batch of vectors in the hypercube to the probability simplex.

        Args:
            X: A `batch_shape x latent_dim`-dim Tensor containing the latent
                scalarization parameters.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        return self.transform(X=X)


class UnitVector(ScalarizationParameterTransform):
    r"""Non-negative unit vectors."""

    def __init__(
        self,
        num_objectives: int,
        transform_label: str = "erf_normalize",
    ) -> None:
        r"""Non-negative unit vectors.

        Args:
            num_objectives: The number of objectives.
            transform_label: The label of the transformation.
        """
        super().__init__(num_objectives=num_objectives)

        self.parameter_dim = num_objectives

        transformations = {
            "normalize": [self.normalize, num_objectives],
            "scale": [self.scale, num_objectives - 1],
            "polar": [self.polar, num_objectives - 1],
            "erf_normalize": [self.erf_normalize, num_objectives],
        }
        self.transform_label = transform_label
        self.transform = transformations[transform_label][0]
        self.latent_dim = transformations[transform_label][1]
        self.bounds = [(0, 1) for _ in range(self.latent_dim)]

    @staticmethod
    def normalize(X: Tensor) -> Tensor:
        r"""Transform vectors on the hypercube into a non-negative unit vector by
        normalizing the vectors using the L2 norm:

        `transform(X) = X / sqrt(sum(X^2))`.

        Args:
            X: A `batch_shape x num_objectives`-dim Tensor containing the latent
                scalarization parameters.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        return X / torch.norm(X, p=2, dim=-1, keepdim=True)

    @staticmethod
    def scale(X: Tensor) -> Tensor:
        r"""Transform vectors on the hypercube into a non-negative unit vector via
        the scaling strategy:

        `M = num_objectives`

        `transform(X)[0] = X[0]`,
        `transform(X)[1] = X[1](1 - transform(X)[0]^2)^(1/2)`,
        `transform(X)[2] = X[2](1 - transform(X)[0]^2 - transform(X)[1]^2)`,
        ...
        `transform(X)[M-1] = (1 - transform(X)[0]^2 - ... - transform(X)[M-1]^2)^(1/2)`.

        Note this could be slow when the number of objectives is increased as the
        computations are performed iteratively.

        Args:
            X: A `batch_shape x (num_objectives - 1)`-dim Tensor containing the
                latent scalarization parameters.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        tkwargs = {"dtype": X.dtype, "device": X.device}
        num_objectives = X.shape[-1] + 1
        transformed_shape = torch.Size([*X.shape[:-1], torch.tensor([num_objectives])])

        # `batch_shape x (latent_dim + 1)`
        transformed_X = torch.zeros(transformed_shape, **tkwargs)

        for m in range(0, num_objectives - 1):
            transformed_X[..., m] = X[..., m]
            if m > 0:
                remainder = torch.sqrt(
                    1.0 - torch.sum(torch.pow(transformed_X[..., 0:m], 2), dim=-1)
                )
                transformed_X[..., m] = remainder * X[..., m]

        transformed_X[..., -1] = torch.sqrt(
            1 - torch.sum(torch.pow(transformed_X[..., 0:-1], 2), dim=-1)
        )

        return transformed_X

    @staticmethod
    def polar(X: Tensor) -> Tensor:
        r"""Transform vectors on the hypercube into a non-negative unit vector by
        applying the standard spherical polar coordinates transformation:

        `M = num_objectives`

        `transform(X) = (
            cos(X[0] * pi / 2),
            sin(X[0] * pi / 2)cos(X[1] * pi / 2),
            sin(X[0] * pi / 2)sin(X[1] * pi / 2)cos(X[2] * pi / 2),
            ...,
            sin(X[0] * pi / 2)...sin(X[M-1] * pi / 2)
        )`.

        Note this could be slow when the number of objectives is increased as the
        computations are performed iteratively.

        Args:
            X: A `batch_shape x (num_objectives - 1)`-dim Tensor containing the
                latent scalarization parameters.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        tkwargs = {"dtype": X.dtype, "device": X.device}
        num_objectives = X.shape[-1] + 1
        transformed_shape = torch.Size([*X.shape[:-1], torch.tensor([num_objectives])])
        # `batch_shape x (latent_dim + 1)`
        transformed_X = torch.zeros(transformed_shape, **tkwargs)
        Y = torch.pi * X / 2

        for m in range(0, num_objectives - 1):
            transformed_X[..., m] = torch.cos(Y[..., m])
            if m > 0:
                sin_product = torch.prod(torch.sin(Y[..., 0:m]), dim=-1)
                transformed_X[..., m] = sin_product * transformed_X[..., m]

        transformed_X[..., -1] = torch.prod(torch.sin(Y), dim=-1)
        # we need to clip values otherwise we might end up with negative weights
        return UnitVector.normalize(transformed_X.clip(torch.finfo(X.dtype).eps))

    @staticmethod
    def erf_normalize(X: Tensor) -> Tensor:
        r"""Transform vectors on the hypercube into a non-negative unit vector by
        normalizing the error function inverse vectors using the L2 norm:

        `transform(X) = erfinv(X) / sqrt(erfinv(X)^2)`.

        Args:
            X: A `batch_shape x num_objectives`-dim Tensor containing the latent
                scalarization parameters.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        # Note that torch.erfinv(1) = inf, therefore we minus epsilon.
        transformed_X = torch.erfinv(X - torch.finfo(X.dtype).eps).clamp_min(0)
        return UnitVector.normalize(transformed_X.clip(torch.finfo(X.dtype).eps))

    def forward(self, X: Tensor) -> Tensor:
        r"""Transform vectors on the hypercube into a non-negative unit vector.

        Args:
            X: A `batch_shape x latent_dim`-dim Tensor containing the latent
                scalarization parameters.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        return self.transform(X=X)


class OrderedUniform(ScalarizationParameterTransform):
    r"""Ordered uniform vectors."""

    def __init__(
        self,
        num_objectives: int,
        descending: bool = False,
        transform_label: str = "exponential_spacing",
    ) -> None:
        r"""Ordered uniform vectors.

        Args:
            num_objectives: The number of objectives.
            descending: If True, sort in descending order, else sort in ascending
                order.
            transform_label: The label of the transformation.
        """
        super().__init__(num_objectives=num_objectives)
        self.descending = descending
        transformations = {
            "exponential_spacing": [self.exponential_spacing, num_objectives + 1],
            "scale": [self.scale, num_objectives],
        }
        self.transform_label = transform_label
        self.transform = transformations[transform_label][0]
        self.latent_dim = transformations[transform_label][1]
        self.bounds = [(0, 1) for _ in range(self.latent_dim)]

    @staticmethod
    def exponential_spacing(X: Tensor, descending: bool = False) -> Tensor:
        r"""Transform vectors on the hypercube into an ordered vector using the
        exponential spacing strategy:

        `M = num_objectives`
        `Y = - log(X)`

        If `descending=False`:
            `transform(X) =  (
                sum(Y[0]) / sum(Y),
                sum(Y[0], Y[1]) / sum(Y),
                ...,
                sum(Y[0], ..., Y[M-1]) / sum(Y)
            )`.

        Else,
            flip the result.

        Args:
            X: A `batch_shape x (num_objectives + 1)`-dim Tensor containing the
                latent scalarization parameters.
            descending: If True, sort in descending order, else sort in ascending
                order.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        # compute exponential random variable
        # add eps to avoid torch.log(0)
        eps = torch.finfo(X.dtype).eps
        eX = -torch.log(eps + (1 - eps) * X)

        csum = torch.cumsum(eX, dim=-1)
        transformed_X = csum[..., :-1] / csum[..., -1].unsqueeze(dim=-1)

        if descending:
            return torch.flip(transformed_X, (-1,))
        else:
            return transformed_X

    @staticmethod
    def scale(X: Tensor, descending: bool = False) -> Tensor:
        r"""Transform vectors on the hypercube into an ordered vector using the
        inverse transform strategy:

        `M = num_objectives`

        If `descending=True`:
            `transform(X)[0] = X[0]^(1/M)`,
            `transform(X)[1] = transform(X)[0] * X[1]^(1/(M-1))`,
            `transform(X)[2] = transform(X)[1] * X[2]^(1/(M-2))`,
            ...
            `transform(X)[M-1] = transform(X)[M-2] * X[M-1]`.
        Else,
            flip the result.

        Args:
            X: A `batch_shape x num_objectives`-dim Tensor containing the latent
                scalarization parameters.
            descending: If True, sort in descending order, else sort in ascending
                order.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        M = X.shape[-1]
        powers = torch.tensor([1 / (M - i) for i in range(M)])
        transformed_X = torch.cumprod(X**powers, dim=-1)

        if descending:
            return transformed_X
        else:
            return torch.flip(transformed_X, (-1,))

    def forward(self, X: Tensor) -> Tensor:
        r"""Transform vectors on the hypercube into an ordered vector using the
        exponential spacing strategy.

        Args:
            X: A `batch_shape x latent_dim`-dim Tensor containing the
                latent scalarization parameters.

        Returns:
            transformed_X: A `batch_shape x num_objectives`-dim Tensor containing
                the scalarization parameters.
        """
        return self.transform(X=X, descending=self.descending)
