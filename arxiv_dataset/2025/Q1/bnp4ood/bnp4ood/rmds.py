"""Implementation of the RMDS algorithm"""
from jaxtyping import Float
import torch
from torch import Tensor


def rmd_score_fun(X: Float[Tensor, "num_samples dim"],
                  muk: Float[Tensor, "num_classes dim"],
                  J: Float[Tensor, "num_classes dim dim"],
                  mu0: Float[Tensor, "dim"] | None = None,
                  L0: Float[Tensor, "dim dim"] | None = None,
                  return_preds: bool = False):
    """Helper to compute the RMDS scores"""
    diff = X[:,None,:] - muk[None, :, :]
    MDk = torch.einsum('nki,ij,nkj->nk', diff, J, diff)
    if mu0 is not None and L0 is not None:
        tmp = torch.linalg.solve_triangular(L0, (X - mu0).T, upper=False).T
        MD0 = torch.einsum('ni,ni->n', tmp, tmp)
        MDk = MDk - MD0[:,None]
    scores = -torch.min(MDk, axis=1).values
    if not scores.isfinite().all():
        raise ValueError("Scores are not finite! Check if degenerate dimensions caused nans to arise in L0.")
    if return_preds:
        preds = torch.argmin(MDk, axis=1)
        return scores, preds
    return scores


def compute_rmd_params(Ns: Float[Tensor, "num_classes"],
                       sumx: Float[Tensor, "num_classes dim"],
                       sumxxT: Float[Tensor, "num_classes dim dim"],
                      ) -> tuple[Float[Tensor, "dim"],
                                 Float[Tensor, "dim dim"],
                                 Float[Tensor, "num_classes dim"],
                                 Float[Tensor, "num_classes dim dim"]]:
    mu0 = sumx.sum(0) / Ns.sum(0)
    Sigma0 = (
        sumxxT.sum(0)
        - torch.outer(sumx.sum(0), mu0)
        - torch.outer(mu0, sumx.sum(0))
        + Ns.sum() * torch.outer(mu0, mu0)
    ) / Ns.sum()
    L0 = torch.linalg.cholesky(Sigma0)
    # Degenerate dimensions will be nan and must be set to a small value
    if not L0.isfinite().all():
        L0[~L0.isfinite()] = 1e-6
    muk = sumx / Ns[:, None]
    Sigma = (sumxxT.sum(0)
             - torch.einsum("ki,kj->ij", sumx, muk)
             - torch.einsum("ki,kj->ij", muk, sumx)
             + torch.einsum("k,ki,kj->ij", Ns, muk, muk)) / Ns.sum()
    J = torch.linalg.inv(Sigma)
    return mu0, L0, muk, J


def compute_rmd_score(Ns, sumx, sumxxT, X_te, with_background=True):
    mu0, L0, muk, J = compute_rmd_params(Ns, sumx, sumxxT)
    if with_background:
        return rmd_score_fun(X_te, muk, J, mu0=mu0, L0=L0)
    return rmd_score_fun(X_te, muk, J)


def compute_indep_rmd_params(Ns: Float[Tensor, "num_classes"],
                             sumx: Float[Tensor, "num_classes dim"],
                             sumxxT: Float[Tensor, "num_classes dim dim"],
                            ) -> tuple[Float[Tensor, "dim"],
                                        Float[Tensor, "dim dim"],
                                        Float[Tensor, "num_classes dim"],
                                        Float[Tensor, "num_classes dim dim"]]:
    """Compute params of the independent covariance model
    """
    mu0 = sumx.sum(axis=0) / Ns.sum()
    Sigma0 = (sumxxT.sum(0) # (D, D)
              + Ns.sum() * torch.einsum("i,j->ij", mu0, mu0)
              - torch.outer(sumx.sum(0), mu0)
              - torch.outer(mu0, sumx.sum(0))
             ) / Ns.sum()
    J0 = torch.linalg.inv(Sigma0)
    muk = sumx / Ns[:, None]
    Sigmak = ( # (K, D, D)
        sumxxT
        + torch.einsum("k,ki,kj->kij", Ns, muk, muk)
        - torch.einsum("ki,kj->kij", sumx, muk)
        - torch.einsum("ki,kj->kij", muk, sumx)
    ) / Ns[:, None, None]
    Jk = torch.linalg.inv(Sigmak)
    return mu0, J0, muk, Jk


def indep_rmd_score_fun(X: Float[Tensor, "num_samples dim"],
                        muk: Float[Tensor, "num_classes dim"],
                        Jk: Float[Tensor, "num_classes dim dim"],
                        mu0: Float[Tensor, "dim"] | None = None,
                        J0: Float[Tensor, "dim dim"] | None = None,
                        return_preds: bool = False,
                       ) -> Float[Tensor, "num_samples"]:
    # Compute RMD
    diff = X[:,None,:] - muk[None, :, :]
    MDk = torch.einsum('nki,kij,nkj->nk', diff, Jk, diff)
    if mu0 is not None and J0 is not None:
        diff = X - mu0
        MD0 = torch.einsum('ni,ij,nj->n', diff, J0, diff)
        MDk = MDk - MD0[:, None]
    scores = -torch.min(MDk, axis=1).values
    if return_preds:
        preds = torch.argmin(MDk, axis=1)
        return scores, preds
    return scores


def compute_indep_rmd_score(Ns, sumx, sumxxT, X_te, with_background=True):
    mu0, J0, muk, Jk = compute_indep_rmd_params(Ns, sumx, sumxxT)
    if with_background:
        return indep_rmd_score_fun(X_te, muk, Jk, mu0=mu0, J0=J0)
    return indep_rmd_score_fun(X_te, muk, Jk)
