import torch
import numpy as np


class DefaultDevice:
    _default_device = 'cpu'
    @staticmethod
    def get() -> str:
        return DefaultDevice._default_device

    @staticmethod
    def set(default_device: str):
        DefaultDevice._default_device = default_device


# from https://github.com/pytorch/pytorch/issues/9244
def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


def project_psd(matrix: torch.Tensor) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.t().conj())  # symmetrize
    L, Q = torch.linalg.eigh(matrix.cdouble())
    L = torch.clamp(L, min=0.0)
    return Q @ torch.diag(L.cdouble()) @ Q.t().conj()


def maxeigh(matrix: torch.Tensor) -> torch.Tensor:
    # shift = torch.linalg.matrix_norm(matrix, ord=2)  # todo: this version doesn't seem to work
    # return torch.linalg.matrix_norm(matrix + shift*eye_like(matrix), ord=2) - shift
    # print(matrix)
    return torch.linalg.eigvalsh(matrix)[-1].item()


def norm_mat_exp_h(matrix: torch.Tensor) -> torch.Tensor:
    lmax = maxeigh(matrix)
    # print(f'{lmax=}')
    matrix = matrix - lmax * eye_like(matrix)
    mexp = torch.matrix_exp(matrix)
    # print(f'{mexp.trace()=}')
    return mexp / mexp.trace()


def norm_mat_exp(matrix: torch.Tensor) -> torch.Tensor:
    # lmax = np.sqrt(maxeigh(matrix @ matrix.t()))
    lmax = torch.linalg.eigvals(matrix).real.max()
    # print(f'{lmax=}')
    matrix = matrix - lmax * eye_like(matrix)
    mexp = torch.matrix_exp(matrix)
    # print(f'{mexp=}')
    # print(f'{mexp.trace()=}')
    return mexp / mexp.trace()


def logtrexp(matrix: torch.Tensor) -> torch.Tensor:
    lmax = torch.linalg.eigvalsh(matrix)[-1].item()
    matrix = matrix - lmax * eye_like(matrix)
    return torch.log(torch.trace(torch.matrix_exp(matrix)).real).item() + lmax


def matrix_log(matrix: torch.Tensor) -> torch.Tensor:
    L, Q = torch.linalg.eigh(matrix.cdouble())
    # print(f'{L=}')
    L[L <= 1e-30] = 1e-30  # todo: hack
    log_diag = torch.diag(torch.log(L).cdouble())
    return Q @ log_diag @ Q.t().conj()


def logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim) - np.log(x.shape[dim])


def tensor_prod(*vectors) -> torch.Tensor:
    n = len(vectors)
    expanded_vectors = [v[[None]*i + [slice(None)] + [None]*(n-1-i)] for i, v in enumerate(vectors)]
    result = expanded_vectors[0]
    for i in range(1, n):
        result = result * expanded_vectors[i]
    return result


def get_default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'
