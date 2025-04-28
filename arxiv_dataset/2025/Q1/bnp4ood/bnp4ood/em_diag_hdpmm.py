"""Expectation maximization parameter estimation for hierarchical DPMMs
with diagonal covariance."""
from jaxtyping import Float
import torch
from torch import Tensor
from torch.distributions import StudentT
from tqdm import trange


# Expectation Maximization algorithm
def nix_normalizer(nu: Float[Tensor, "num_classes dim"],
                   sigmasq: Float[Tensor, "num_classes dim"],
                   kappa: Float[Tensor, "num_classes dim"],
                  ) -> Float[Tensor, "num_classes dim"]:
    """
    Compute the marginal likelihood of a normal inverse chi-squared
    distribution as a function of hyperparameters
    """
    kappa = torch.as_tensor(kappa, device=sigmasq.device)
    nu = torch.as_tensor(nu, device=sigmasq.device)

    ll = -0.5 * torch.log(kappa)
    ll += torch.special.gammaln(0.5 * nu)
    ll -= 0.5 * nu * torch.log(0.5 * nu * sigmasq)
    return ll


def compute_posterior_params(
        Nk: Float[Tensor, "num_classes"],
        sumx: Float[Tensor, "num_classes dim"],
        sumxsq: Float[Tensor, "num_classes dim"],
        nu0: float | Float[Tensor, "dim"],
        sigmasq0: Float[Tensor, "dim"],
        kappa0: float | Float[Tensor, "dim"],
        mu0: Float[Tensor, "dim"],
    ) -> tuple[Float[Tensor, "num_classes dim"],
               Float[Tensor, "num_classes dim"],
               Float[Tensor, "num_classes dim"],
               Float[Tensor, "num_classes dim"]]:
    """
    Compute the posterior parameters given the hyperparameters from the
    sufficient statistics.
    """
    assert torch.all(nu0 > 0), "Invalid nu0"

    nus = nu0 + Nk[:, None]                         # (K, D)
    kappas = kappa0 + Nk[:, None]                   # (K, D)
    mus = (kappa0 * mu0 + sumx) / kappas            # (K, D
    sigmasqs = \
        (nu0 * sigmasq0 \
        + kappa0 * mu0**2 \
        + sumxsq \
        - kappas * mus**2) / nus                    # (K, D)

    # Make sure it's not too small
    sigmasqs = torch.clamp(sigmasqs, 1e-4)

    return nus, sigmasqs, kappas, mus


def marginal_ll(
        Nk: Float[Tensor, "num_classes"],
        nus_post: Float[Tensor, "num_classes dim"],
        sigmasqs_post: Float[Tensor, "num_classes dim"],
        kappas_post: Float[Tensor, "num_classes dim"],
        nu0: float | Float[Tensor, "dim"],
        sigmasq0: Float[Tensor, "dim"],
        kappa0: float | Float[Tensor, "dim"],
    ) -> Float[Tensor, "num_classes dim"]:
    """
    Compute the marginal log likelihood given the hyperparameters.
    """
    ll = nix_normalizer(nus_post, sigmasqs_post, kappas_post)
    ll -= nix_normalizer(nu0, sigmasq0, kappa0)
    ll -= 0.5 * Nk[:, None] * torch.log(torch.as_tensor(2 * torch.pi, device=Nk.device))
    return ll

def e_step(
        Nk: Float[Tensor, "num_classes"],
        sumx: Float[Tensor, "num_classes dim"],
        sumxsq: Float[Tensor, "num_classes dim"],
        nu0: float | Float[Tensor, "dim"],
        sigmasq0: Float[Tensor, "dim"],
        kappa0: float | Float[Tensor, "dim"],
        mu0: Float[Tensor, "dim"],
    ) -> tuple[tuple[Float[Tensor, "num_classes dim"],
                     Float[Tensor, "num_classes dim"],
                     Float[Tensor, "num_classes dim"]],
               Float[Tensor, ""]]:
    """
    Compute expected sufficient statistics
    """
    N = Nk.sum()
    nus_post, sigmasq_post, kappas_post, mus_post = \
        compute_posterior_params(Nk, sumx, sumxsq, nu0, sigmasq0, kappa0, mu0)

    # E[1/\sigma_{k,d}^2]
    E_sigmasq_inv = 1 / sigmasq_post

    # E[\log \sigma_{k,d}^2]
    E_log_sigmasq = torch.log(0.5 * nus_post * sigmasq_post) - torch.digamma(0.5 * nus_post)

    # E[(\mu_{k,d} - \mu_0)^2 / \sigma_{k,d}^2]
    E_mahal = 1 / kappas_post + (mus_post - mu0)**2 / sigmasq_post

    ll = marginal_ll(Nk, nus_post, sigmasq_post, kappas_post, nu0, sigmasq0, kappa0)#, device=Nk.device)
    return (E_sigmasq_inv, E_log_sigmasq, E_mahal), ll.sum() / N


def m_step_kappa(
        expectations: tuple[
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"]]
    ) -> Float[Tensor, ""]:
    """
    Maximize the expected log likelihood wrt kappa
    """
    _, _, E_mahal = expectations
    K, _ = E_mahal.shape
    return K / E_mahal.sum(axis=0)


def _L_nu(nu0: float | Float[Tensor, "dim"],
          sigmasq0: Float[Tensor, "dim"],
          expectations: tuple[
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"]],
    ) -> Float[Tensor, "dim"]:
    """Compute the expected log likelihood as a function of nu_{0,d}
    """
    E_sigmasq_inv, E_log_sigmasq, _ = expectations
    K, _ = E_sigmasq_inv.shape
    L = 0.5 * nu0 * K * torch.log(0.5 * nu0)
    L += 0.5 * nu0 * K * torch.log(sigmasq0)
    L -= K * torch.special.gammaln(0.5 * nu0)
    L -= 0.5 * torch.sum(nu0 * E_log_sigmasq, axis=0)
    L -= 0.5 * torch.sum(nu0 * sigmasq0 * E_sigmasq_inv, axis=0)
    return L


def _dL_nu(
        nu0: float | Float[Tensor, "dim"],
        sigmasq0: float | Float[Tensor, "dim"],
        expectations: tuple[
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"]],
    ) -> Float[Tensor, "dim"]:
    """Compute the derivative of the expected log likelihood wrt nu_{0,d}

    Raises
    ------
    AssertionError if the derivative is not finite
    """
    E_sigmasq_inv, E_log_sigmasq, _ = expectations
    K, _ = E_sigmasq_inv.shape

    dL = 0.5 * K * (torch.log(0.5 * nu0) + 1)
    dL += 0.5 * K * torch.log(sigmasq0)
    dL -= 0.5 * torch.sum(E_log_sigmasq, axis=0)
    dL -= 0.5 * torch.sum(sigmasq0 * E_sigmasq_inv, axis=0)
    dL -= 0.5 * K * torch.special.digamma(0.5 * nu0)
    assert torch.all(torch.isfinite(dL))
    return dL


def _d2L_nu2(
        nu0: float | Float[Tensor, "dim"],
        expectations: tuple[
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"]],
    ) -> Float[Tensor, "dim"]:
    """Compute the second derivative of the expected log likelihood wrt nu_{0,d}
    """
    E_sigmasq_inv, _, _ = expectations
    K, _ = E_sigmasq_inv.shape

    d2L = 0.5 * K / nu0
    d2L -= 0.25 * K * torch.special.polygamma(1, 0.5 * nu0)
    return d2L


def m_step_nu(
        nu0: float | Float[Tensor, "dim"],
        sigmasq0: float | Float[Tensor, "dim"],
        expectations: tuple[
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"]],
        num_iters: int = 100
    ) -> Float[Tensor, "dim"]:
    """
    Maximize the expected log likelihood wrt nu

    Raises
    ------
    AssertionError if nu0 is not finite at any point during the optimization.
    """

    for i in range(num_iters):
        dL = _dL_nu(nu0, sigmasq0, expectations)
        d2L = _d2L_nu2(nu0, expectations)
        nu0 = nu0**2 * d2L / (dL + nu0 * d2L)
        nu0 = torch.clamp(nu0, 1 + 1e-3)
        assert torch.all(torch.isfinite(nu0))
    return nu0


def em(
        Nk: Float[Tensor, "num_classes"],
        sumx: Float[Tensor, "num_classes dim"],
        sumxsq: Float[Tensor, "num_classes dim"],
        nu0: Float[Tensor, "dim"],
        sigmasq0: Float[Tensor, "dim"],
        kappa0: Float[Tensor, "dim"],
        mu0: Float[Tensor, "dim"],
        fit_kappa0=True,
        num_iter: int=50,
        verbose: bool=False,
        pbar: bool=True,
    ) -> tuple[list[float], list[float], list[float]]:
    """
    EM algorithm for hGMM

    Raises
    ------
    AssertionError if nu0, kappa0, or the derivative of expected log likelihood
    wrt nu0 is not finite during optimization.
    """
    lls = []
    kappas = [kappa0.to('cpu')]
    nus = [nu0.to('cpu')]

    for i in trange(num_iter, disable=not pbar):
        # E step
        expectations, ll = e_step(Nk, sumx, sumxsq, nu0, sigmasq0, kappa0, mu0)

        # M step
        if fit_kappa0: kappa0 = m_step_kappa(expectations)
        nu0 = m_step_nu(nu0, sigmasq0, expectations)
        assert torch.all(torch.isfinite(kappa0))
        assert torch.all(torch.isfinite(nu0))

        # Track results
        lls.append(ll.item())
        kappas.append(kappa0.to('cpu'))
        nus.append(nu0.to('cpu'))
        if verbose: print(f"\nITR {i}, LL {lls[-1]}, nu: {nus[-1]}, kappa: {kappas[-1]}")

    return lls, torch.stack(kappas), torch.stack(nus)


def posterior_pred_logprob_perdim(
        X: Float[Tensor, "num_data dim"], posterior_params: tuple
        ) -> Float[Tensor, "num_data num_classes"]:
    """Compute the posterior predictive log probability
    """
    nu_post, sigmasq_post, kappa_post, mu_post = posterior_params
    K = nu_post.shape[0] if nu_post.ndim > 1 else 0
    scale_post = torch.sqrt((kappa_post + 1) / kappa_post * sigmasq_post)
    post = StudentT(nu_post, mu_post, scale_post)
    return post.log_prob(X[:, None, :]) if K > 0 else post.log_prob(X)


def compute_weights(X, Nk, sumx, sumxsq, nu0, sigmasq0, kappa0, mu0, rho,
                   ):
    """
    Compute the weights for each dimension
    """
    # Compute the posterior parameters for each cluster
    nus_post, sigmasq_post, kappas_post, mus_post = \
        compute_posterior_params(Nk, sumx, sumxsq, nu0, sigmasq0, kappa0, mu0)

    # Compute marginal likelihoods, assuming per-class means and variances
    lp1 = marginal_ll(Nk, nus_post, sigmasq_post, kappas_post, nu0, sigmasq0,
                      kappa0,
                     ).sum(axis=0)
    lp1 += torch.log(torch.as_tensor(rho, device=Nk.device))

    # Compute marginal likelihoods, assuming shared mean and variance
    lp0 = StudentT(nu0, mu0, torch.sqrt(sigmasq0)).log_prob(X).sum(axis=0)
    lp0 += torch.log(torch.as_tensor(1 - rho, device=Nk.device))

    log_weights = lp1 - torch.logsumexp(torch.stack([lp0, lp1]), axis=0)

    return torch.exp(log_weights)


def compute_scores(prior_params,
                   posterior_params,
                   Z_te,
                   Ns,
                   weights=None,
                   return_preds=False,
                  ):
    """
    Compute log likelihood ratio for each data point
    """
    D = Z_te.shape[1]
    if weights is None:
        weights = torch.ones(D, device=Z_te.device)

    prior_pred_lps_te = posterior_pred_logprob_perdim(Z_te, prior_params)[:,None,:] # (N, 1, D)
    posterior_pred_lps_te = posterior_pred_logprob_perdim(Z_te, posterior_params)   # (N, K, D)

    scores_per_class = torch.sum(weights * (posterior_pred_lps_te - prior_pred_lps_te), axis=2)
    logNk = torch.log(Ns)
    scores = torch.logsumexp(scores_per_class + logNk, axis=1) # \tilde{C} in the paper
    if return_preds:
        preds = torch.argmax(posterior_pred_lps_te.sum(-1), -1)
        return scores, preds
    return scores
