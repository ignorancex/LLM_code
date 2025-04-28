"""Expectation maximization parameter estimation for hierarchical DPMMs."""
from jaxtyping import Float
import torch
import torch.distributions as dists
from torch import Tensor
from tqdm import trange

from .distributions import multivariate_t_logpdf


# Expectation Maximization algorithm
def niw_normalizer(nu: Float[Tensor, "num_classes"],
                   Sigma: Float[Tensor, "num_classes dim dim"],
                   kappa: Float[Tensor, "num_classes"]
                  ) -> Float[Tensor, "num_classes"]:
    """
    Compute the marginal likelihood of a normal inverse Wishart
    distribution as a function of hyperparameters

    nu: (K,)
    Sigma: (K, D, D)
    kappa: (K,)
    """
    D = Sigma.shape[-1]
    ll = -0.5 * D * torch.log(torch.as_tensor(kappa, device=Sigma.device))
    ll += torch.mvlgamma(0.5 * torch.as_tensor(nu), D)
    ll += 0.5 * nu * D * torch.log(torch.tensor(2.0))
    ll -= 0.5 * nu * torch.linalg.slogdet(Sigma)[1]
    return ll


def compute_posterior_params(
        Nk: Float[Tensor, "num_classes"],
        sumx: Float[Tensor, "num_classes dim"],
        sumxxT: Float[Tensor, "num_classes dim dim"],
        nu0: float | Float[Tensor, ""],
        Sigma0: Float[Tensor, "dim dim"],
        kappa0: float | Float[Tensor, ""],
        mu0: Float[Tensor, "dim"],
    ) -> tuple[Float[Tensor, "num_classes"],
               Float[Tensor, "num_classes dim dim"],
               Float[Tensor, "num_classes"],
               Float[Tensor, "num_classes dim"]]:
    """
    Compute the posterior parameters given the hyperparameters from the
    sufficient statistics.
    """
    D = Sigma0.shape[-1]
    assert nu0 > D + 1, "Invalid nu0"

    nus = nu0 + Nk
    kappas = kappa0 + Nk
    mus = (kappa0 * mu0 + sumx) / kappas[:, None]
    Sigmas = (nu0 - D - 1) * Sigma0 \
        + kappa0 * torch.outer(mu0, mu0) \
        + sumxxT \
        - kappas[:, None, None] * torch.einsum("ki,kj->kij", mus, mus)
    Sigmas = 0.5 * (Sigmas + torch.transpose(Sigmas, -1, -2))
    Sigmas += nus.reshape(-1, 1, 1) * 1e-4 * torch.eye(D, device=Sigmas.device)
    return nus, Sigmas, kappas, mus


def marginal_ll(
        Nk,
        nus_post: Float[Tensor, "num_classes"],
        Sigmas_post: Float[Tensor, "num_classes dim dim"],
        kappas_post: Float[Tensor, "num_classes"],
        nu0: float | Float[Tensor, ""],
        Sigma0: Float[Tensor, "dim dim"],
        kappa0: float | Float[Tensor, ""],
    ) -> Float[Tensor, "num_classes"]:
    """
    Compute the marginal log likelihood given the hyperparameters.
    """
    D = Sigma0.shape[-1]
    ll = niw_normalizer(nus_post, Sigmas_post, kappas_post)
    ll -= niw_normalizer(nu0, (nu0 - D - 1) * Sigma0, kappa0)
    ll -= 0.5 * D * Nk * torch.log(2 * torch.as_tensor(torch.pi, device=Sigma0.device))
    return ll

def multivariate_digamma(inpt: Float[Tensor, "N"], D: int) -> Float[Tensor, "N"]:
    """Compute multivariate digamma
    """
    return torch.sum(torch.digamma(inpt[:, None] + 0.5 * (1 - torch.arange(1, D+1, device=inpt.device)[None, :])), axis=-1)


def multivariate_polygamma(n: int, inpt: Float[Tensor, "N"], D: int) -> Float[Tensor, "N"]:
    """Compute multivariate polygamma function of order n, dimension D
    """
    return torch.sum(torch.polygamma(n, inpt + 0.5 * (1 - torch.arange(1, D+1, device=inpt.device))), axis=-1)


def e_step(
        Nk: Float[Tensor, "num_classes"],
        sumx: Float[Tensor, "num_classes dim"],
        sumxxT: Float[Tensor, "num_classes dim dim"],
        nu0: float | Float[Tensor, ""],
        Sigma0: Float[Tensor, "dim dim"],
        kappa0: float | Float[Tensor, ""],
        mu0: Float[Tensor, "dim"],
    ) -> tuple[tuple[Float[Tensor, "num_classes"],
                     Float[Tensor, "num_classes"],
                     Float[Tensor, "num_classes"]],
               Float[Tensor, ""]]:
    """
    Compute expected sufficient statistics
    """
    N = Nk.sum()
    D = Sigma0.shape[-1]
    nus_post, Sigmas_post, kappas_post, mus_post = \
        compute_posterior_params(Nk, sumx, sumxxT, nu0, Sigma0, kappa0, mu0)

    # Directly compute Tr(E[Sigma_k^{-1}] Sigma0) = Tr(\nu_k' * Sigma_k'^{-1} * Sigma0)
    i = 0
    while not dists.constraints.positive_definite.check(Sigmas_post).all() and i < 5:
        idxs = ~dists.constraints.positive_definite.check(Sigmas_post)
        Sigmas_post[idxs] += torch.eye(D, device=Sigmas_post.device)
        i += 1
    Ls_post = torch.linalg.cholesky(Sigmas_post)                                          # (K, D, D)
    tmp = torch.linalg.solve_triangular(Ls_post, Sigma0, upper=False)                     # (K, D, D)
    tmp = torch.linalg.solve_triangular(torch.swapaxes(Ls_post, -1, -2), tmp, upper=True) # (K, D, D)
    E_trace = torch.einsum('k,kii->k', nus_post, tmp)                                     # (K,)

    # Compute expected log determinant of Sigma_k
    E_logdet_Sigma = 2 * torch.einsum('kii->k', torch.log(Ls_post)) \
        - multivariate_digamma(0.5 * nus_post, D) - D * torch.log(torch.tensor(2.))

    # Compute expected Mahalanobis distance
    dmu = mus_post - mu0
    tmp = torch.linalg.solve_triangular(Ls_post, dmu[:, :, None], upper=False)[:, :, 0]  # (K, D)
    E_mahal = 1 / kappas_post + nus_post * torch.sum(tmp**2, axis=1)

    ll = torch.sum(marginal_ll(Nk, nus_post, Sigmas_post, kappas_post, nu0, Sigma0, kappa0)) / N
    return (E_trace, E_logdet_Sigma, E_mahal), ll


def m_step_kappa(
        D: int,
        expectations: tuple[
            Float[Tensor, "num_classes dim dim"],
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes"]]
    ) -> Float[Tensor, ""]:
    """
    Maximize the expected log likelihood wrt kappa
    """
    _, _, E_mahal = expectations
    K = E_mahal.shape[0]
    return (K * D) / E_mahal.sum()


def _L_nu(nu0: float | Float[Tensor, ""],
          Sigma0: Float[Tensor, "dim dim"],
          expectations: tuple[
            Float[Tensor, "num_classes dim dim"],
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes"]],
    ) -> Float[Tensor, ""]:
    """Compute the expected log likelihood as a function of nu
    """
    nu0 = torch.as_tensor(nu0)

    E_trace, E_logdet_Sigma, _ = expectations
    K = E_trace.shape[0]
    D = Sigma0.shape[-1]
    L = 0.5 * nu0 * K * D * torch.log(0.5 * (nu0 - D - 1))
    L += 0.5 * nu0 * K * torch.linalg.slogdet(Sigma0)[1]
    L -= 0.5 * nu0 * torch.sum(E_logdet_Sigma)
    L -= 0.5 * nu0 * torch.sum(E_trace)
    L -= K * torch.mvlgamma(0.5 * nu0, D)
    return L


def _dL_nu(
        D: int,
        nu0: float | Float[Tensor, ""],
        logdet_Sigma0: float | Float[Tensor, ""],
        expectations: tuple[
            Float[Tensor, "num_classes dim dim"],
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes"]],
    ) -> Float[Tensor, ""]:
    """Compute the derivative of the expected log likelihood wrt nu

    Raises
    ------
    AssertionError if the derivative is not finite
    """
    E_trace, E_logdet_Sigma, _ = expectations
    K = E_trace.shape[0]

    nu0 = torch.as_tensor(nu0, device=E_trace.device)

    dL = 0.5 * K * D * torch.log(0.5 * (nu0 - D - 1))
    dL += 0.5 * K * D  * nu0 / (nu0 - D - 1)
    dL += 0.5 * K * logdet_Sigma0
    dL -= 0.5 * torch.sum(E_logdet_Sigma)
    dL -= 0.5 * torch.sum(E_trace)
    dL -= 0.5 * K * multivariate_digamma(0.5 * nu0.unsqueeze(0), D)[0]
    assert torch.isfinite(dL)
    return dL


def _d2L_nu2(
        D: int,
        nu0: float | Float[Tensor, ""],
        expectations: tuple[
            Float[Tensor, "num_classes dim dim"],
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes"]],
    ) -> Float[Tensor, ""]:
    """Compute the second derivative of the expected log likelihood wrt nu
    """
    E_trace, _, _ = expectations
    K = E_trace.shape[0]

    nu0 = torch.as_tensor(nu0, device=E_trace.device)

    d2L = 0.5 * K * D / (nu0 - D - 1)
    d2L -= 0.5 * K * D * (D+1) / (nu0 - D - 1)**2
    d2L -= 0.25 * K * multivariate_polygamma(1, 0.5 * nu0, D)
    return d2L


def m_step_nu(
        D,
        nu0: float | Float[Tensor, ""],
        logdet_Sigma0: float | Float[Tensor, ""],
        expectations: tuple[
            Float[Tensor, "num_classes dim dim"],
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes"]],
        num_iters: int = 100
    ) -> Float[Tensor, ""]:
    """
    Maximize the expected log likelihood wrt nu

    Raises
    ------
    AssertionError if nu0 is not finite at any point during the optimization.
    """
    for i in range(num_iters):
        dL = _dL_nu(D, nu0, logdet_Sigma0, expectations)
        d2L = _d2L_nu2(D, nu0, expectations)
        nu0 = nu0**2 * d2L / (dL + nu0 * d2L)
        nu0 = torch.clamp(nu0, D + 1 + 1e-3)
        assert torch.isfinite(nu0)
    return nu0


def em(
        Nk: Float[Tensor, "num_classes"],
        sumx: Float[Tensor, "num_classes dim"],
        sumxxT: Float[Tensor, "num_classes dim dim"],
        nu0: float | Float[Tensor, ""],
        Sigma0: Float[Tensor, "dim dim"],
        kappa0: float | Float[Tensor, ""],
        mu0: Float[Tensor, "dim"],
        num_iter: int=50,
        verbose: bool=False,
        pbar: bool=True,
    ) -> tuple[list[float], list[float], list[float]]:
    """
    EM algorithm

    Raises
    ------
    AssertionError if nu0, kappa0, or the derivative of expected log likelihood
    wrt nu0 is not finite during optimization.
    """
    # Precompute some constants
    D = sumx.shape[-1]
    logdet_Sigma0 = torch.linalg.slogdet(Sigma0)[1]

    lls = []
    kappas = [kappa0.item()]
    nus = [nu0.item()]

    for i in trange(num_iter, disable=not pbar):
        # E step
        expectations, ll = e_step(Nk, sumx, sumxxT, nu0, Sigma0, kappa0, mu0)

        # M step
        kappa0 = m_step_kappa(D, expectations)
        nu0 = m_step_nu(D, nu0, logdet_Sigma0, expectations)
        assert torch.isfinite(kappa0)
        assert torch.isfinite(nu0)

        # Track results
        lls.append(ll.item())
        kappas.append(kappa0.item())
        nus.append(nu0.item())
        if verbose: print(f"\nITR {i}, LL {lls[-1]}, nu: {nus[-1]}, kappa: {kappas[-1]}")

    return lls, kappas, nus


def posterior_pred_logprob(X: Float[Tensor, "N dim"],
                           posterior_params: tuple) -> Float[Tensor, "N K"]:
    """Compute the posterior predictive log probability
    """
    xdev = X.device
    nu_post, Sigma_post, kappa_post, mu_post = posterior_params
    D = Sigma_post.shape[-1]

    pred_cov = ((kappa_post + 1) / (kappa_post * (nu_post - D + 1))).reshape(-1, 1, 1) * Sigma_post
    pred_cov = 0.5 * (pred_cov + torch.transpose(pred_cov, -1, -2))
    pred_cov += 1e-4 * torch.eye(D, device=xdev)
    return multivariate_t_logpdf(X, nu_post, mu_post,
                                 torch.linalg.cholesky(pred_cov.squeeze()), "full")


def compute_scores(prior_params,
                   posterior_params,
                   X,
                   Ns,
                   return_preds=False,
                  ):
    """
    Compute log likelihood ratio for each data point
    """
    D = X.shape[1]
    prior_pred_lps = posterior_pred_logprob(X, prior_params)[:,None] # (N, 1)
    posterior_pred_lps = posterior_pred_logprob(X, posterior_params)   # (N, K)

    scores_per_class = posterior_pred_lps - prior_pred_lps

    logNk = torch.log(Ns)
    scores = torch.logsumexp(scores_per_class + logNk, axis=1)
    if return_preds:
        preds = torch.argmax(posterior_pred_lps, -1)
        return scores, preds
    return scores