from __future__ import annotations
import torch
from functools import wraps

"""Native PyTorch implementation of the LOBPCG algorithm for eigenvalue computation.

This code was copied (and translated into PyTorch) from Google's JAX implementation:
https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/linalg.py#L37-L105
"""

def torch_lobpcg(A, X, max_iter=100, tol=None):
    """Finds the top k eigenvalues/eigenvectors of the n x n linear operator A.
    
    Args:
      A (Callable: Array -> Array): function representing an [n x n] linear operator. 
        Passing in an [n x r] input should return an [n x r] output where A has
        been evaluated on each column of the input.
      X (Array): an initial guess for the top k eigenvectors, of shape [n x k].  This is
        how the function learns what k is.
      max_iter (int): the maximum number of LOBPCG iterations to run
      tol (float): the numerical tolerance
      
    Returns:
      top eigenvalues
      top eigenvectors
      number of iterations
    """
    n, k = X.shape
    
    A = create_safe_wrapper(A)

    if n < 4 * k:
        A_mat = A(torch.eye(n, dtype=X.dtype, device=X.device))
        evals, evecs = _eigh_ascending(A_mat)
        return evals[:k], evecs[:, :k], 1

    dt = X.dtype
    if tol is None:
        tol = torch.finfo(dt).eps

    X = _orthonormalize(X)
    P = _extend_basis(X, X.shape[1])
    AX = A(X)
    theta = torch.sum(X * AX, axis=0, keepdims=True)
    R = AX - theta * X
    converged = 0
    for i in range(max_iter):
        if converged >= k:
            break
        R = _project_out(torch.concatenate((X, P), axis=1), R)
        XPR = torch.concatenate((X, P, R), axis=1)
        theta, Q = _rayleigh_ritz_orth(A, XPR)
        theta = theta[:k]
        B = Q[:, :k]
        normB = torch.linalg.norm(B, ord=2, axis=0, keepdims=True)
        B /= normB
        X = _mm(XPR, B)
        normX = torch.linalg.norm(X, ord=2, axis=0, keepdims=True)
        X /= normX
        q, _ = torch.linalg.qr(Q[:k, k:].T)
        diff_rayleigh_ortho = _mm(Q[:, k:], q)
        P = _mm(XPR, diff_rayleigh_ortho)
        normP = torch.linalg.norm(P, ord=2, axis=0, keepdims=True)
        P /= torch.where(normP == 0, 1.0, normP)
        AX = A(X)
        R = AX - theta[None, :] * X
        resid_norms = torch.linalg.norm(R, ord=2, axis=0)
        reltol = 10 * n * (torch.linalg.norm(AX, ord=2, axis=0) + theta)
        res_converged = resid_norms < tol * reltol
        converged = torch.sum(res_converged)
    return theta, X, i

def _project_out(basis, U):
    for _ in range(2):
        U -= _mm(basis, _mm(basis.T, U))
        U = _orthonormalize(U)
    for _ in range(2):
        U -= _mm(basis, _mm(basis.T, U))
    normU = torch.linalg.norm(U, ord=2, axis=0, keepdims=True)
    U *= normU >= 0.99

    return U

def _rayleigh_ritz_orth(A, S):
    SAS = _mm(S.T, A(S))
    return _eigh_ascending(SAS)


def _orthonormalize(basis):
    for _ in range(2):
        basis = _svqb(basis)
    return basis


def _mm(a, b):
    return torch.matmul(a, b)


def _eigh_ascending(A):
    try:
        w, V = torch.linalg.eigh(A)
    except:
        print("uh oh")
        raise Exception("eigh failed")
    idx = torch.argsort(-w)
    return w[idx], V[:, idx]


def _svqb(X):
    norms = torch.linalg.norm(X, ord=2, axis=0, keepdims=True)
    X /= torch.where(norms == 0, 1.0, norms)
    inner = _mm(X.T, X)
    w, V = _eigh_ascending(inner)
    tau = torch.finfo(X.dtype).eps * w[0]
    padded = torch.maximum(w, tau)
    sqrted = torch.where(tau > 0, padded, 1.0) ** (-0.5)
    scaledV = V * sqrted[None, :]
    orthoX = _mm(X, scaledV)
    keep = ((w > tau) * (torch.diag(inner) > 0.0))[None, :]
    orthoX *= keep.type(orthoX.dtype)
    norms = torch.linalg.norm(orthoX, ord=2, axis=0, keepdims=True)
    keep *= (norms > 0.0).type(keep.dtype)
    orthoX /= torch.where(keep, norms, 1.0)
    return orthoX


def _extend_basis(X, m):
    n, k = X.shape
    Xupper, Xlower = X[:k], X[k:]
    u, s, vt = torch.linalg.svd(Xupper)
    y = torch.concatenate([Xupper + _mm(u, vt), Xlower], axis=0)
    other = torch.concatenate(
        [
            torch.eye(m, dtype=X.dtype, device=X.device),
            torch.zeros((n - k - m, m), dtype=X.dtype, device=X.device),
        ],
        axis=0,
    )
    w = _mm(y, vt.T * ((2 * (1 + s)) ** (-1 / 2))[None, :])
    h = -2 * torch.linalg.multi_dot([w, w[k:, :].T, other])
    h[k:] += other
    return h


# wrap the matvec function A with an error handler that, in the event
# of an OOM, returns the shape of the input tensor that triggered it
def create_safe_wrapper(A):
    @wraps(A)  # This preserves the metadata of the original function A
    def safe_A(tensor):
        try:
            return A(tensor)
        except torch.cuda.OutOfMemoryError as e:            
            raise torch.cuda.OutOfMemoryError(
                f"{str(e)}\nTensor shape that caused the OOM: {tuple(tensor.shape)}"
            )
    
    return safe_A
