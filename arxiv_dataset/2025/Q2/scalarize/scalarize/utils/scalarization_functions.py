#!/usr/bin/env python3

r"""
Scalarization functions for multi-objective optimization. The scalarized values are
designed such that larger values imply greater quality. The default settings are
designed for a maximization problem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import pi

import torch
from botorch.exceptions.errors import UnsupportedError
from scipy.special import gamma
from torch import Tensor
from torch.nn import Module


class ScalarizationFunction(Module, ABC):
    r"""Abstract base class for scalarization functions."""
    num_params: int

    def __init__(self) -> None:
        r"""Base constructor for scalarization functions."""
        super().__init__()

    @staticmethod
    @abstractmethod
    def evaluate(Y: Tensor, **kwargs) -> Tensor:
        r"""Evaluate the scalarization functions on a set of points.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x param_shape`-dim Tensor containing the
                scalarized objective values.
        """
        pass  # pragma: no cover

    @abstractmethod
    def forward(self, Y: Tensor) -> Tensor:
        r"""Evaluate the scalarization functions on a set of points.
        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_scalars`-dim Tensor containing the
                scalarized objective values.
        """
        pass  # pragma: no cover


class ResidualBasedScalarizationFunction(ScalarizationFunction, ABC):
    r"""Abstract base class for scalarization functions that are based on both
    weights and reference points.
    """

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool,
        clip: bool,
    ) -> None:
        r"""Base constructor for the scalarization functions based on both weights
        and reference points.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clip the residual at zero.
        """
        super().__init__()
        if weights.ndim < 2:
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
            else:
                raise UnsupportedError(
                    "The weights should have a shape of "
                    "`batch_shape x num_weights x num_objectives`."
                )
        if ref_points.ndim < 2:
            if ref_points.ndim == 1:
                ref_points = ref_points.unsqueeze(0)
            else:
                raise UnsupportedError(
                    "The reference points should have a shape of "
                    "`batch_shape x num_ref x num_objectives`."
                )

        self.weights = weights
        self.ref_points = ref_points
        self.invert = invert
        self.clip = clip

    @staticmethod
    def compute_weighted_residual(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool,
        clip: bool,
    ) -> Tensor:
        r"""Compute the weighted residual.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `clip = True`:
                residual = max(residual, 0)

            weighted_residual = w * residual

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clip the residual at zero.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref x M`-dim Tensor
                containing the weighted residuals.
        """
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        if ref_points.ndim == 1:
            ref_points = ref_points.unsqueeze(0)

        sign = -1.0 if invert else 1.0
        # `batch_shape x num_points x num_ref x M`
        diff = sign * (ref_points.unsqueeze(-3) - Y.unsqueeze(-2))

        if clip:
            diff = torch.clamp(diff, min=0)

        # `batch_shape x num_points x num_weights x num_ref x M`
        return weights.unsqueeze(-2).unsqueeze(-4) * diff.unsqueeze(-3)


class LinearScalarization(ScalarizationFunction):
    r"""Linear scalarization function."""
    num_params = 1

    def __init__(self, weights: Tensor, negate: bool = False) -> None:
        r"""Linear scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor of weights.
            negate: If True, we negate the scalarization function.
        """
        super().__init__()
        if weights.ndim < 2:
            raise UnsupportedError(
                "The weights should have a shape of "
                "`batch_shape x num_weights x num_objectives`."
            )
        self.weights = weights
        self.negate = negate

    @staticmethod
    def evaluate(Y: Tensor, weights: Tensor, negate: bool = False) -> Tensor:
        r"""Computes the linear scalarization.

            If `negate=False`:
                s(Y) = sum(w * Y)
            Else:
                s(Y) = - sum(w * Y)

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.

        Returns:
            A `batch_shape x num_points x num_weights`-dim Tensor containing the
                scalarized objective values.
        """
        sign = -1.0 if negate else 1.0
        # `batch_shape x num_points x num_weights`
        return sign * torch.sum(weights.unsqueeze(-3) * Y.unsqueeze(-2), dim=-1)

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the linear scalarization.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_scalars`-dim Tensor containing the
                scalarized objective values.
        """
        scalarized_Y = self.evaluate(
            Y=Y,
            weights=self.weights,
            negate=self.negate,
        )
        return torch.flatten(scalarized_Y, start_dim=-self.num_params, end_dim=-1)


class LpScalarization(ResidualBasedScalarizationFunction):
    r"""Lp scalarization function."""
    num_params = 2

    def __init__(
        self,
        p: float,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = False,
        clip: bool = False,
        negate: bool = True,
    ) -> None:
        r"""Lp scalarization function.

        Args:
            p: The power of the Lp norm.
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clip the residual at zero.
            negate: If True, we negate the distance.
        """
        super().__init__(
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            clip=clip,
        )
        # TODO: check whether p >= 1
        self.p = p
        self.negate = negate

    @staticmethod
    def evaluate(
        Y: Tensor,
        p: float,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = False,
        clip: bool = False,
        negate: bool = True,
    ) -> Tensor:
        r"""Computes the Lp scalarization or one of its variants.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `clip = True`:
                residual = max(residual, 0)

            If `negate=False`:
                sign = 1
            Else:
                sign = -1

            s(Y) = sign * ||w * residual||_p

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            p: The power of the Lp norm.
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clip the residual at zero.
            negate: If True, we negate the distance.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective values.
        """
        sign = -1.0 if negate else 1.0

        # `batch_shape x num_points x num_weights x num_ref x M`
        weighted_diff = LpScalarization.compute_weighted_residual(
            Y=Y,
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            clip=clip,
        )

        # `batch_shape x num_points x num_weights x num_ref`
        return sign * torch.norm(weighted_diff, p=p, dim=-1)

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the Lp scalarization or one of its variants.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_scalars`-dim Tensor containing the
                scalarized objective values.
        """
        scalarized_Y = self.evaluate(
            Y=Y,
            p=self.p,
            weights=self.weights,
            ref_points=self.ref_points,
            invert=self.invert,
            clip=self.clip,
            negate=self.negate,
        )

        return torch.flatten(scalarized_Y, start_dim=-self.num_params, end_dim=-1)


class ChebyshevScalarization(ResidualBasedScalarizationFunction):
    r"""Chebyshev scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = False,
        pseudo: bool = False,
        clip: bool = False,
        negate: bool = True,
    ) -> None:
        r"""Chebyshev scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            pseudo: If True, then we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            clip: If True, we clip the distance at zero.
            negate: If True, we negate the distance.
        """
        super().__init__(
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            clip=clip,
        )
        self.pseudo = pseudo
        self.negate = negate

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = False,
        pseudo: bool = False,
        clip: bool = False,
        negate: bool = True,
    ) -> Tensor:
        r"""Computes the Chebyshev scalarization or one of its variants.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `clip = True`:
                residual = max(residual, 0)

            If `negate=False`:
                sign = 1
            Else:
                sign = -1

            If `pseudo=False`:
                s(Y) = sign * max(w * residual)
            Else:
                s(Y) = sign * min(w * residual)

        Args:
            Y: An `batch_shape x num_points M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            pseudo: If True, then we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            clip: If True, we clip the distance at zero.
            negate: If True, we negate the distance.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective values.
        """
        sign = -1.0 if negate else 1.0

        # `batch_shape x num_points x num_weights x num_ref x M`
        weighted_diff = ChebyshevScalarization.compute_weighted_residual(
            Y=Y,
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            clip=clip,
        )

        # `batch_shape x num_points x num_weights x num_ref`
        if pseudo:
            return sign * torch.min(weighted_diff, dim=-1).values
        else:
            return sign * torch.max(weighted_diff, dim=-1).values

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the Chebyshev scalarization or one of its variants.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_scalars`-dim Tensor containing the
                scalarized objective values.
        """
        scalarized_Y = self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            invert=self.invert,
            pseudo=self.pseudo,
            clip=self.clip,
            negate=self.negate,
        )

        return torch.flatten(scalarized_Y, start_dim=-self.num_params, end_dim=-1)


class LengthScalarization(ChebyshevScalarization):
    r"""Length scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = True,
        clip: bool = True,
    ) -> None:
        r"""Length scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clamp the length at zero.
        """
        # Note that we store weights, but actually use 1 / weights in evaluation.
        super().__init__(
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            pseudo=True,
            clip=clip,
            negate=False,
        )

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = True,
        clip: bool = True,
    ) -> Tensor:
        r"""Computes the length scalarization.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `clip = True`:
                residual = max(residual, 0)

            s(Y) = min(residual / w)

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clamp the length at zero.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective values.
        """
        # TODO: need to be careful about dividing by zero
        return ChebyshevScalarization.evaluate(
            Y=Y,
            weights=1 / weights,
            ref_points=ref_points,
            invert=invert,
            pseudo=True,
            clip=clip,
            negate=False,
        )

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the length scalarization.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_scalars`-dim Tensor containing the
                scalarized objective values.
        """
        scalarized_Y = self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            invert=self.invert,
            clip=self.clip,
        )

        return torch.flatten(scalarized_Y, start_dim=-self.num_params, end_dim=-1)


class HypervolumeScalarization(LengthScalarization):
    r"""Hypervolume scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = True,
        clip: bool = True,
    ) -> None:
        r"""Hypervolume scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            invert: If True, the residual is defined as `y-r`, else the residual is
                defined as `r-y`, where `y` is an objective vector and `r` is
                a reference point.
            clip: If True, we clamp the values at zero.
        """
        super().__init__(
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            clip=clip,
        )

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        invert: bool = True,
        clip: bool = True,
    ) -> Tensor:
        r"""Computes the hypervolume scalarization.

            c = π^(M/2) / (2^M Γ(M/2 + 1))

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `clip = True`:
                residual = max(residual, 0)

            s(Y) = c g(min(residual / w), M)

            where g(z, M) = sign(z) * abs(z)^M.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            clip: If True, we clamp the values at zero.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective values.
        """
        length = LengthScalarization.evaluate(
            Y=Y, weights=weights, ref_points=ref_points, invert=invert, clip=clip
        )
        M = weights.shape[-1]
        # We multiply the constant before taking the power for numerical stability.
        constant = pi ** (1 / 2) / (2 * gamma(M / 2 + 1) ** (1 / M))

        if clip:
            return torch.pow(constant * length, M)
        else:
            return torch.sign(length) * torch.pow(constant * abs(length), M)

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the hypervolume scalarization.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_scalars`-dim Tensor containing the
                scalarized objective values.
        """
        scalarized_Y = self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            invert=self.invert,
            clip=self.clip,
        )

        return torch.flatten(scalarized_Y, start_dim=-self.num_params, end_dim=-1)


class AugmentedChebyshevScalarization(ChebyshevScalarization):
    r"""Augmented Chebyshev scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5e-2,
        invert: bool = False,
        pseudo: bool = False,
        negate: bool = True,
    ) -> None:
        r"""Augmented Chebyshev scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            beta: The trade-off parameter controlling the L1 penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.
            pseudo: If True, we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            negate: If True, we negate the distance.
        """
        super().__init__(
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            clip=False,
            pseudo=pseudo,
            negate=negate,
        )

        self.beta = beta

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5e-2,
        invert: bool = False,
        pseudo: bool = False,
        negate: bool = True,
    ) -> Tensor:
        r"""Computes the augmented Chebyshev scalarization or one of its variants.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `negate=False`:
                sign = 1
            Else:
                sign = -1

            If `pseudo=False`:
                s(Y) = sign * (max(w * residual) + beta * sum(w * residual))
            Else:
                s(Y) = sign * (min(w * residual) + beta * sum(w * residual))

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            beta: The trade-off parameter controlling the L1 penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.
            pseudo: If True, we compute the pseudo Chebyshev scalarization function,
                which uses the minimum instead of the maximum.
            negate: If True, we negate the distance.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective values.
        """
        sign = -1.0 if negate else 1.0
        # `batch_shape x num_points x num_weights x num_ref x M`
        weighted_diff = AugmentedChebyshevScalarization.compute_weighted_residual(
            Y=Y,
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            clip=False,
        )
        # `num_points x batch_shape x num_weights x num_ref`
        penalty = torch.sum(weighted_diff, dim=-1)

        # `batch_shape x num_points x num_weights x num_ref`
        if pseudo:
            return sign * (torch.min(weighted_diff, dim=-1).values + beta * penalty)
        else:
            return sign * (torch.max(weighted_diff, dim=-1).values + beta * penalty)

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the augmented Chebyshev scalarization or one of its variants.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_scalars`-dim Tensor containing the
                scalarized objective values.
        """
        scalarized_Y = self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            beta=self.beta,
            invert=self.invert,
            pseudo=self.pseudo,
            negate=self.negate,
        )

        return torch.flatten(scalarized_Y, start_dim=-self.num_params, end_dim=-1)


class ModifiedChebyshevScalarization(ChebyshevScalarization):
    r"""Modified Chebyshev scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5e-2,
        invert: bool = False,
        pseudo: bool = False,
        negate: bool = True,
    ) -> None:
        r"""Modified Chebyshev scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            beta: The trade-off parameter controlling the L1 penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.
            pseudo: If True, we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            negate: If True, we negate the distance.
        """
        super().__init__(
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            clip=False,
            pseudo=pseudo,
            negate=negate,
        )

        self.beta = beta

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5e-2,
        invert: bool = False,
        pseudo: bool = False,
        negate: bool = True,
    ) -> Tensor:
        r"""Computes the modified Chebyshev scalarization function or one of its
        variants.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `negate=False`:
                sign = 1
            Else:
                sign = -1

            If `pseudo=False`:
                s(Y) = sign * max(w * residual + sign * beta * sum(w * residual))
            Else:
                s(Y) = sign * min(w * residual + sign * beta * sum(w * residual))

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            beta: The trade-off parameter controlling the L1 penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.
            pseudo: If True, we compute the pseudo Chebyshev scalarization
                function, which uses the minimum instead of the maximum.
            negate: If True, we negate the distance.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective values.
        """
        sign = -1.0 if negate else 1.0
        # `batch_shape x num_points x num_weights x num_ref x M`
        weighted_diff = ModifiedChebyshevScalarization.compute_weighted_residual(
            Y=Y,
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            clip=False,
        )
        penalty = sign * torch.sum(weighted_diff, dim=-1, keepdims=True)

        # `batch_shape x num_points x num_weights x num_ref`
        if pseudo:
            return sign * torch.min(weighted_diff + beta * penalty, dim=-1).values
        else:
            return sign * torch.max(weighted_diff + beta * penalty, dim=-1).values

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the modified Chebyshev scalarization function.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_scalars`-dim Tensor containing the
                scalarized objective values.
        """
        scalarized_Y = self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            beta=self.beta,
            invert=self.invert,
            pseudo=self.pseudo,
            negate=self.negate,
        )

        return torch.flatten(scalarized_Y, start_dim=-self.num_params, end_dim=-1)


class PBIScalarization(ResidualBasedScalarizationFunction):
    r"""Penalty boundary intersection scalarization function."""
    num_params = 2

    def __init__(
        self,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5,
        invert: bool = False,
        negate: bool = True,
    ) -> None:
        r"""Penalty boundary intersection scalarization function.

        Args:
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            beta: The trade-off parameter controlling the diversity penalty.
            invert: If True, the residual is defined as the difference from the
                reference point `y-r`, else the residual is defined as the difference
                towards the reference point `r-y`, where `y` is an objective vector
                and `r` is a reference point.
            negate: If True, we negate the convergence term.
        """
        super().__init__(
            weights=weights,
            ref_points=ref_points,
            invert=invert,
            clip=False,
        )

        self.beta = beta
        self.negate = negate

    @staticmethod
    def evaluate(
        Y: Tensor,
        weights: Tensor,
        ref_points: Tensor,
        beta: float = 5,
        invert: bool = False,
        negate: bool = True,
    ) -> Tensor:
        r"""Computes the penalty boundary intersection scalarization or inverted
        penalty boundary intersection scalarization.

            If `invert=False`:
                residual = r - Y
            Else:
                residual = Y - r

            If `negate=False`:
                sign = 1
            Else:
                sign = -1

            convergence(Y) = |sum(w * residual)|
            diversity(Y) = ||residual - convergence(Y) * w||_2

            s(Y) = sign * convergence(Y) - beta * diversity(Y)

            The sign of the convergence term depends on the optimization problem
            and the choice of reference point. For a maximization problem, the
            standard PBI uses the utopia reference point and sets `invert=False` and
            `negate=True`. Whereas, the standard inverted PBI uses the nadir
            reference point and sets `invert=True` and `negate=False`.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            weights: A `batch_shape x num_weights x M`-dim Tensor containing the
                weights `w`.
            ref_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                reference points `r`.
            beta: The trade-off parameter controlling the diversity penalty.
            invert: If True we compute with respect to the nadir, else we compute
                with respect to the utopia.
            negate: If True, we negate the convergence term.

        Returns:
            A `batch_shape x num_points x num_weights x num_ref`-dim Tensor
                containing the scalarized objective values.
        """
        sign = -1.0 if negate else 1.0
        # `batch_shape x num_points x num_weights x num_ref x M`
        diff = PBIScalarization.compute_weighted_residual(
            Y=Y,
            weights=torch.ones_like(weights),
            ref_points=ref_points,
            invert=invert,
            clip=False,
        )
        # `batch_shape x 1 x num_weights x 1 x M`
        expanded_weights = weights.unsqueeze(-2).unsqueeze(-4)

        # `batch_shape x num_points x num_weights x num_ref`
        convergence = torch.abs(torch.sum(expanded_weights * diff, dim=-1))

        # `batch_shape x num_points x num_weights x num_ref`
        diversity = torch.norm(
            diff - convergence.unsqueeze(-1) * expanded_weights, p=2, dim=-1
        )

        # `batch_shape x num_points x num_weights x num_ref`
        return sign * convergence - beta * diversity

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the penalty boundary intersection scalarization or inverted
        penalty boundary intersection scalarization.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_scalars`-dim Tensor containing the
                scalarized objective values.
        """
        scalarized_Y = self.evaluate(
            Y=Y,
            weights=self.weights,
            ref_points=self.ref_points,
            beta=self.beta,
            invert=self.invert,
            negate=self.negate,
        )

        return torch.flatten(scalarized_Y, start_dim=-self.num_params, end_dim=-1)


class KSScalarization(LengthScalarization):
    r"""Kalai-Smorodinsky scalarization function."""
    num_params = 1

    def __init__(
        self,
        utopia_points: Tensor,
        nadir_points: Tensor,
    ) -> None:
        r"""Kalai-Smorodinsky scalarization function.

        Args:
            utopia_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                utopia points, which are the best possible points.
            nadir_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                nadir points, which are the worst possible points.
        """

        super().__init__(
            weights=utopia_points - nadir_points,
            ref_points=nadir_points,
            invert=True,
            clip=False,
        )

        self.utopia_points = utopia_points
        self.nadir_points = nadir_points

    @staticmethod
    def evaluate(
        Y: Tensor,
        utopia_points: Tensor,
        nadir_points: Tensor,
    ) -> Tensor:
        r"""Computes the Kalai-Smorodinsky scalarization function.

            s(Y) = min((y - nadir) / (utopia - nadir))
                 = min((nadir - y) / (nadir - utopia))

        This scalarization function can be used for both minimization and
        maximization problems as long as the reference points are set accordingly.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.
            utopia_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                utopia points, which are the best possible points.
            nadir_points: A `batch_shape x num_ref x M`-dim Tensor containing the
                nadir points, which are the worst possible points.

        Returns:
            A `batch_shape x num_ref`-dim Tensor containing the scalarized objective
                values.
        """
        # `num_points x batch_shape x num_ref`
        return LengthScalarization.evaluate(
            Y=Y,
            weights=utopia_points - nadir_points,
            ref_points=nadir_points,
            invert=True,
            clip=False,
        ).diagonal(dim1=-2, dim2=-1)

    def forward(self, Y: Tensor) -> Tensor:
        r"""Computes the Kalai-Smorodinsky scalarization function.

        Args:
            Y: An `batch_shape x num_points x M`-dim Tensor containing the objective
                vectors.

        Returns:
            A `batch_shape x num_points x num_scalars`-dim Tensor containing the
                scalarized objective values.
        """
        scalarized_Y = self.evaluate(
            Y=Y,
            utopia_points=self.utopia_points,
            nadir_points=self.nadir_points,
        )

        return torch.flatten(scalarized_Y, start_dim=-self.num_params, end_dim=-1)
