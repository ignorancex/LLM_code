"""Implementation of the Tied DPMM algorithm"""
import logging

from jaxtyping import Float
import torch
from torch import Tensor
import torch.distributions as dists


logger = logging.getLogger(__name__)


def posterior_pred_logprob(
        X: Float[Tensor, "num_samples dim"],
        params: tuple[Float[Tensor, "... dim"], Float[Tensor, "... dim dim"]],
    ) -> Float[Tensor, "num_samples ..."]:
    """Compute the log probability of the data under the posterior predictive distribution"""
    px_y = dists.MultivariateNormal(params[0], scale_tril=params[1])
    return px_y.log_prob(X[:, None, :]).squeeze()


def make_psd(A):
    A = 0.5 * (A + A.T)
    A += torch.eye(A.shape[-1], device=A.device) * 1e-4
    return A


def compute_tied_params(Ns: Float[Tensor, "num_classes"],
                        sumx: Float[Tensor, "num_classes dim"],
                        sumxxT: Float[Tensor, "num_classes dim dim"],
                       ) -> tuple[
                           tuple[Float[Tensor, "dim"], Float[Tensor, "dim dim"],],
                           tuple[Float[Tensor, "num_classes dim"], Float[Tensor, "num_classes dim dim"]]
                       ]:
    D = sumx.shape[-1]
    K = sumx.shape[0]
    mu0 = sumx.sum(0) / Ns.sum(0)
    Sigma0 = (
        sumxxT.sum(0)
        - torch.outer(sumx.sum(0), mu0)
        - torch.outer(mu0, sumx.sum(0))
        + Ns.sum() * torch.outer(mu0, mu0)
    ) / Ns.sum()
    Sigma0 = make_psd(Sigma0)
    J0 = torch.linalg.inv(Sigma0)
    J0 = make_psd(J0)
    cluster_means = sumx / Ns[:, None]
    Sigma = (sumxxT.sum(0)
             - torch.einsum("ki,kj->ij", sumx, cluster_means)
             - torch.einsum("ki,kj->ij", cluster_means, sumx)
             + torch.einsum("k,ki,kj->ij", Ns, cluster_means, cluster_means)) / Ns.sum()
    Sigma = make_psd(Sigma)
    J = torch.linalg.inv(Sigma)
    J = make_psd(J)
    Sk = torch.linalg.inv(J0 + Ns[:, None, None] * J)
    muk = []
    for k in range(K):
        temp = J0 @ mu0 + Ns[k] * J @ cluster_means[k] # (D,)
        temp = Sk[k] @ temp
        muk.append(temp)
    muk = torch.stack(muk)
    L0 = torch.linalg.cholesky(Sigma0 + Sigma)
    Lk = torch.linalg.cholesky(Sk + Sigma)
    prior_params = (mu0, L0)
    post_params = (muk, Lk)
    return prior_params, post_params


def compute_scores(
        prior_params, posterior_params, X, Ns,
        return_preds=False,
        ):
    prior_pred_lps = posterior_pred_logprob(X, prior_params)[:, None] # (N, 1)
    posterior_pred_lps = posterior_pred_logprob(X, posterior_params) # (N, K)

    thetas = posterior_pred_lps - prior_pred_lps
    scores = torch.logsumexp(thetas + torch.log(Ns), axis=1)
    if return_preds:
        preds = torch.argmax(posterior_pred_lps, axis=-1)
        return scores, preds
    return scores
