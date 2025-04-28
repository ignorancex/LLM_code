from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import fft

from src.common_utils.custom_vars import LinearOperatorMethod, NormOperatorType


def tv_direct(x: np.ndarray) -> np.ndarray:
    if x.ndim < 3:
        raise ValueError("The array must have at least 3 dimensions.")
    rows_diff = np.diff(x, n=1, axis=0)
    rows_zero = np.zeros((1, *x.shape[1:]))
    rows_tv = np.concatenate((rows_diff, rows_zero), axis=0)
    rows_tv = rows_tv[..., np.newaxis]
    cols_diff = np.diff(x, n=1, axis=1)
    cols_zero = np.zeros((x.shape[0], 1, *x.shape[2:]))
    cols_tv = np.concatenate((cols_diff, cols_zero), axis=1)
    cols_tv = cols_tv[..., np.newaxis]
    u = np.concatenate((rows_tv, cols_tv), axis=-1)
    return u


def tv_adjoint(u: np.ndarray) -> np.ndarray:
    if u.ndim < 4:
        raise ValueError("The array must have at least 4 dimensions.")
    rows_tv = u[..., 0]
    rows_tv_diff = np.diff(rows_tv, n=1, axis=0)
    rows_first = rows_tv[0, :, ...][np.newaxis, :, ...]
    rows_adj = -np.concatenate((rows_first, rows_tv_diff), axis=0)
    cols_tv = u[..., 1]
    cols_tv_diff = np.diff(cols_tv, n=1, axis=1)
    cols_first = cols_tv[:, 0, ...][:, np.newaxis, ...]
    cols_adj = -np.concatenate((cols_first, cols_tv_diff), axis=1)
    x = rows_adj + cols_adj
    return x


# TODO: Find a better way than using Callables.. LinearOperator(ABC) => MatrixOperator & FunctionOperator
@dataclass(frozen=True)
class LinearOperator:
    direct: Callable[[np.ndarray], np.ndarray]
    adjoint: Callable[[np.ndarray], np.ndarray]
    norm: float
    inverse: Optional[Callable[[np.ndarray], np.ndarray]] = None

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> LinearOperator:
        return cls(
            direct=lambda x: matrix @ x,
            adjoint=lambda u: matrix.T @ u,
            norm=float(np.linalg.svd(matrix, compute_uv=False)[0]),
            inverse=lambda u: np.linalg.pinv(matrix) @ u,
        )

    @classmethod
    def from_method(cls, method: LinearOperatorMethod) -> LinearOperator:
        if method == LinearOperatorMethod.NOT_APPLICABLE:
            return cls(
                direct=lambda x: np.array(None),
                adjoint=lambda u: np.array(None),
                norm=0.,
                inverse=lambda u: np.array(None),
            )
        elif method == LinearOperatorMethod.IDENTITY:
            return cls(
                direct=lambda x: x,
                adjoint=lambda u: u,
                norm=1.,
                inverse=lambda u: u,
            )

        elif method == LinearOperatorMethod.DCT:
            return cls(
                direct=lambda x: fft.dct(x, norm="ortho", axis=-2),
                adjoint=lambda u: fft.idct(u, norm="ortho", axis=-2),
                norm=1.,
                inverse=lambda u: fft.idct(u, norm="ortho", axis=-2),
            )

        elif method == LinearOperatorMethod.TV:
            return cls(
                direct=tv_direct,
                adjoint=tv_adjoint,
                norm=np.sqrt(8),
                inverse=None,
            )

        else:
            ValueError(f"Option '{method}' is not supported.")


@dataclass(frozen=True)
class ProximalOperator(ABC):
    @abstractmethod
    def direct(self, x: np.ndarray, lambdaa: float) -> np.ndarray:
        pass

    @abstractmethod
    def proximal(self, x: np.ndarray, gamma: float) -> np.ndarray:
        """
        Proximal operator of a function f(.), i.e., prox_{lambda, f}
        """
        pass

    @abstractmethod
    def proximal_conjugate(self, x: np.ndarray, gamma: float, lambdaa: float) -> np.ndarray:
        """
        Proximal operator of the conjugate of a norm operator g(.), i.e., prox_{lambda, g*}
        """
        pass


@dataclass(frozen=True)
class CTVOperator(ProximalOperator):
    norm: NormOperator

    def direct(self, x: np.ndarray, lambdaa: float):
        return lambdaa * self.norm.direct(x)

    def proximal(self, x: np.ndarray, gamma: float):
        return self.norm.proximal(x, gamma)

    def proximal_conjugate(self, x: np.ndarray, _, lambdaa: float):
        if self.norm.proximal_conjugate:
            return self.norm.proximal_conjugate(x, lambdaa)
        else:
            raise ValueError("The proximal operator of the conjugate function of the norm is not implemented.")


@dataclass
class NormOperator:
    direct: Callable
    conjugate: Callable
    proximal: Callable
    proximal_conjugate: Callable

    @classmethod
    def from_norm(cls, norm: NormOperatorType) -> NormOperator:
        if norm == NormOperatorType.NOT_APPLICABLE:
            return cls(
                direct=lambda x: None,
                conjugate=lambda x: None,
                proximal=lambda x: None,
                proximal_conjugate=lambda x: None,
            )
        elif norm == NormOperatorType.L112 or norm == NormOperatorType.L121 or norm == NormOperatorType.L211:
            return prox_elements_l112(norm_label=norm[0], perm=norm[1])
        elif norm == NormOperatorType.L1O:
            return prox_elements_l1o()
        elif norm == NormOperatorType.L2O:
            return prox_elements_l2o()
        else:
            ValueError(f"Norm option {norm[0]} is not supported.")


# Collaborative norm operators
# TODO: Turn NormOperator into an ABC and the functions below into its Implementations
def prox_elements_l1o() -> NormOperator:
    def opprox(x, gamma):
        return prox_l1(x, gamma)

    def oppconj(x, lambdaa):
        return projball_linf(x, lambdaa)

    def opnorm(x):
        """
        Performs l1 norm on the first axis of the array.
        In the case of a spectrum, the first axis represents the channels.
        """
        return np.linalg.norm(x, ord=1, axis=0)

    def opnconj(x):
        return np.amax(np.absolute(x), axis=0)

    return NormOperator(
        proximal=opprox,
        proximal_conjugate=oppconj,
        direct=opnorm,
        conjugate=opnconj,
    )


def prox_elements_l2o() -> NormOperator:
    def opprox(x, gamma):
        return prox_l21(x, gamma)

    def oppconj(x, lambdaa):
        return projball_l2inf(x, lambdaa)

    def opnorm(x):
        """
        Performs l2 norm on the first axis of the array.
        In the case of a spectrum, the first axis represents the channels.
        """
        return np.linalg.norm(x, ord=2, axis=0)

    def opnconj(x):
        return np.linalg.norm(x, ord=2, axis=0)

    return NormOperator(
        proximal=opprox,
        proximal_conjugate=oppconj,
        direct=opnorm,
        conjugate=opnconj,
    )


def prox_elements_l112(norm_label: str, perm: tuple) -> NormOperator:
    def opprox(x, gamma):
        return permute_wrapper(perm, prox_l21, x, gamma)

    def oppconj(x, lambdaa):
        return permute_wrapper(perm, projball_l2inf, x, lambdaa)

    def opnorm(x: np.ndarray):
        for order in norm_label[1:]:
            x = np.linalg.norm(x, ord=order, axis=0)
        return x

    def opnconj(x):
        norm_dict: dict[str, Callable] = {"1": np.amax, "2": np.linalg.norm}
        x = np.absolute(x)
        for order in norm_label[1:]:
            x = norm_dict[order](x, axis=0)
        return x

    return NormOperator(
        proximal=opprox,
        proximal_conjugate=oppconj,
        direct=opnorm,
        conjugate=opnconj,
    )


# Elemental norm operators
def prox_l1(x: np.ndarray, gamma: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)


def projball_linf(x: np.ndarray, gamma: float) -> np.ndarray:
    return np.maximum(np.minimum(x, gamma), -gamma)


def prox_l21(x: np.ndarray, gamma: float) -> np.ndarray:
    return x - projball_l2inf(x, gamma)


def projball_l2inf(x: np.ndarray, gamma: float) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=0) / gamma, 1)


# utils
def permute_wrapper(perm, prox_func, x, *prox_func_args):
    x = np.transpose(x, axes=perm)
    x = prox_func(x, *prox_func_args)
    x = np.transpose(x, axes=np.argsort(perm))
    return x
