"""EM parameter estimation for coupled hierarchical DPMMs with diagonal covariance."""
from jaxtyping import Float, Int
import torch
from torch import Tensor
from torch.distributions import Gamma, StudentT
from tqdm import trange


# Expectation Maximization algorithm
def nix_normalizer(nu: Float[Tensor, "num_classes dim"],
                   sigmasq: Float[Tensor, "num_points num_classes dim"],
                   kappa: Float[Tensor, "num_classes dim"],
                  ) -> Float[Tensor, "num_classes dim"]:
    """
    Compute the marginal likelihood of a normal inverse chi-squared
    distribution as a function of hyperparameters
    """
    ll = -0.5 * torch.log(kappa) # (K, D) or (D)
    ll += torch.special.gammaln(0.5 * nu) # (K, D) or (D)
    ll = ll - 0.5 * nu * torch.log(0.5 * nu * sigmasq) # (P-1, K, D) or (P-1, 1, D)
    return ll


def marginal_ll(
        Nk: Float[Tensor, "num_classes"],
        nus_post: Float[Tensor, "num_classes dim"],
        sigmasqs_post: Float[Tensor, "num_classes num_points dim"],
        kappas_post: Float[Tensor, "num_classes dim"],
        nu0: Float[Tensor, "dim"],
        sigmasq0: Float[Tensor, "num_points dim"],
        kappa0: Float[Tensor, "dim"],
        logweights: Float[Tensor, "num_points"],
    ) -> float:
    """
    Compute the marginal log likelihood given the hyperparameters.
    """
    D = nu0.shape[0]
    ll = nix_normalizer(nus_post, sigmasqs_post.transpose(0,1), kappas_post)
    ll -= nix_normalizer(nu0, sigmasq0[:, None, :], kappa0)
    ll = logweights[:, None] + ll.sum(-1)
    ll = torch.logsumexp(ll, dim=0).sum()
    ll += 0.5 * Nk.sum() * D * torch.log(torch.as_tensor(2 * torch.pi, device=nus_post.device))
    return ll


def gammak_centers_weights(
        alpha0: float,
        gammaks: Float[Tensor, "num_points"]
        ) -> tuple[Float[Tensor, "num_points-1"],
                   Float[Tensor, "num_points-1"],
                   Float[Tensor, "num_points-1"]]:
    # gammak centers
    centers = 0.5 * (gammaks[1:] + gammaks[:-1])
    widths = gammaks[1:] - gammaks[:-1]
    logweights = Gamma(alpha0, alpha0).log_prob(centers) + torch.log(widths)
    logweights -= torch.logsumexp(logweights, dim=0)
    return centers, widths, logweights


def compute_posterior_params(
        Nk: Float[Tensor, "num_classes"],
        sumx: Float[Tensor, "num_classes dim"],
        sumxsq: Float[Tensor, "num_classes dim"],
        nu0: float | Float[Tensor, "dim"],
        sigmasq0: Float[Tensor, "dim"],
        kappa0: float | Float[Tensor, "dim"],
        mu0: Float[Tensor, "dim"],
        alpha0: float,
        gammaks: Float[Tensor, "num_points"],
    ) -> tuple[Float[Tensor, "num_classes dim"],
               Float[Tensor, "num_classes num_points-1 dim"],
               Float[Tensor, "num_classes dim"],
               Float[Tensor, "num_classes dim"],
               Float[Tensor, "num_classes num_points-1"],
               Float[Tensor, "num_classes num_points-1"]]:
    """
    Compute the posterior parameters given the hyperparameters from the
    sufficient statistics.
    """
    assert torch.all(nu0 > 0), "Invalid nu0"
    centers, widths, logweights = gammak_centers_weights(alpha0, gammaks)

    nus_post = nu0 + Nk[:, None]                         # (K, D)
    kappas_post = kappa0 + Nk[:, None]                   # (K, D)
    mus_post = (kappa0 * mu0 + sumx) / kappas_post            # (K, D)
    sigmasqs_post = \
        (nu0 * centers[:, None, None] * sigmasq0 \
        + kappa0 * mu0**2 \
        + sumxsq \
        - kappas_post * mus_post**2) / nus_post           # (P-1, K, D)

    # Make sure it's not too small
    sigmasqs_post = torch.clamp(sigmasqs_post, 1e-4)

    ll = nix_normalizer(nus_post, sigmasqs_post, kappas_post)   # (P-1, K, D)
    ll -= nix_normalizer(nu0, centers[:, None, None] * sigmasq0, kappa0) # (P-1, 1, D)
    ll -= 0.5 * Nk[None, :, None] * torch.log(torch.as_tensor(2 * torch.pi)) # (P-1, K, D)
    ll = ll.sum(-1) # (P-1, K)
    ll += logweights[:, None] # (P-1, K)
    log_gammak_post = ll - ll.logsumexp(0) # (P-1, K)
    gammak_post = torch.exp(log_gammak_post) # (P-1, K)
    # Reshape to K, ...
    log_gammak_post = log_gammak_post.transpose(0, 1) # (K, P-1)
    gammak_post = gammak_post.transpose(0, 1) # (K, P-1)
    sigmasqs_post = sigmasqs_post.transpose(0, 1) # (K, P-1, D)
    return nus_post, sigmasqs_post, kappas_post, mus_post, gammak_post, log_gammak_post


def e_step(
        Nk: Float[Tensor, "num_classes"],
        sumx: Float[Tensor, "num_classes dim"],
        sumxsq: Float[Tensor, "num_classes dim"],
        nu0: float | Float[Tensor, "dim"],
        sigmasq0: Float[Tensor, "dim"],
        kappa0: float | Float[Tensor, "dim"],
        mu0: Float[Tensor, "dim"],
        alpha0: float | Float[Tensor, ""],
        gammaks: Float[Tensor, "num_points"],
    ) -> tuple[tuple[Float[Tensor, "num_classes"],
                     Float[Tensor, "num_classes"],
                     Float[Tensor, "num_classes dim"],
                     Float[Tensor, "num_classes dim"],
                     Float[Tensor, "num_classes dim"]],
               Float[Tensor, ""]]:
    """
    """

    N = Nk.sum()
    nus_post, sigmasq_post, kappas_post, mus_post, gammak_post, _ = \
        compute_posterior_params(Nk, sumx, sumxsq, nu0, sigmasq0, kappa0, mu0, alpha0, gammaks)

    centers, _, logweights = gammak_centers_weights(alpha0, gammaks)

    # E[gamma_k]
    E_gammak = torch.sum(gammak_post * centers, axis=-1)

    # E[log gamma_k]
    E_log_gammak = torch.sum(gammak_post * torch.log(centers), axis=-1)

    # E[gamma_k/\sigma_{k,d}^2]
    E_gammak_sigmasq_inv = torch.einsum("kp,p,kpd->kd", gammak_post, centers, 1/sigmasq_post)

    # E[\log \sigma_{k,d}^2]
    E_log_sigmasq = torch.log(0.5 * nus_post[:, None, :] * sigmasq_post)
    E_log_sigmasq -= torch.digamma(0.5 * nus_post[:, None, :])
    E_log_sigmasq = torch.einsum("kp,kpd->kd", gammak_post, E_log_sigmasq)

    # E[(\mu_{k,d} - \mu_0)^2 / \sigma_{k,d}^2]
    E_mahal = 1 / kappas_post[:, None, :] + (mus_post[:, None, :] - mu0)**2 / sigmasq_post
    E_mahal = torch.einsum("kp,kpd->kd", gammak_post, E_mahal)

    ll = marginal_ll(Nk, nus_post, sigmasq_post, kappas_post,
                     nu0, centers[:, None] * sigmasq0, kappa0, logweights)
    return (E_gammak, E_log_gammak, E_gammak_sigmasq_inv, E_log_sigmasq, E_mahal), ll / N


def m_step_kappa(
        expectations: tuple[
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
        ],
    ) -> Float[Tensor, ""]:
    """
    Maximize the expected log likelihood wrt kappa
    """
    _, _, _, _, E_mahal = expectations
    K, _ = E_mahal.shape
    return K / E_mahal.sum(axis=0)


def _L_nu(nu0: float | Float[Tensor, "dim"],
          sigmasq0: Float[Tensor, "dim"],
          expectations: tuple[
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
        ],
    ) -> Float[Tensor, "dim"]:
    """Compute the expected log likelihood as a function of nu_{0,d}
    """
    _, E_log_gammak, E_gammak_sigmasq_inv, E_log_sigmasq, _ = expectations
    K, _ = E_gammak_sigmasq_inv.shape
    L = 0.5 * nu0 * K * torch.log(0.5 * nu0)
    L += 0.5 * nu0 * E_log_gammak.sum()
    L += 0.5 * nu0 * K * torch.log(sigmasq0)
    L -= 0.5 * torch.sum(nu0 * E_log_sigmasq, axis=0)
    L -= 0.5 * torch.sum(nu0 * sigmasq0 * E_gammak_sigmasq_inv, axis=0)
    L -= K * torch.special.gammaln(0.5 * nu0)
    return L


def _dL_nu(
        nu0: float | Float[Tensor, "dim"],
        sigmasq0: float | Float[Tensor, "dim"],
        expectations: tuple[
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
        ],
    ) -> Float[Tensor, "dim"]:
    """Compute the derivative of the expected log likelihood wrt nu_{0,d}

    Raises
    ------
    AssertionError if the derivative is not finite
    """
    _, E_log_gammak, E_gammak_sigmasq_inv, E_log_sigmasq, _ = expectations
    K, _ = E_gammak_sigmasq_inv.shape

    dL = 0.5 * K * (torch.log(0.5 * nu0) + 1)
    dL += 0.5 * E_log_gammak.sum()
    dL += 0.5 * K * torch.log(sigmasq0)
    dL -= 0.5 * torch.sum(E_log_sigmasq, axis=0)
    dL -= 0.5 * torch.sum(sigmasq0 * E_gammak_sigmasq_inv, axis=0)
    dL -= 0.5 * K * torch.special.digamma(0.5 * nu0)
    assert torch.all(torch.isfinite(dL))
    return dL


def _d2L_nu2(
        nu0: float | Float[Tensor, "dim"],
        expectations: tuple[
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
        ],
    ) -> Float[Tensor, "dim"]:
    """Compute the second derivative of the expected log likelihood wrt nu_{0,d}
    """
    _, _, E_sigmasq_inv, _, _ = expectations
    K, _ = E_sigmasq_inv.shape

    d2L = 0.5 * K / nu0
    d2L -= 0.25 * K * torch.special.polygamma(1, 0.5 * nu0)
    return d2L


def m_step_nu(
        nu0: Float[Tensor, "dim"],
        sigmasq0: Float[Tensor, "dim"],
        expectations: tuple[
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes"],
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


def _dL_alpha(alpha0: Float[Tensor, ""],
              expectations: tuple[
                  Float[Tensor, "num_classes"],
                  Float[Tensor, "num_classes"],
                  Float[Tensor, "num_classes dim"],
                  Float[Tensor, "num_classes dim"],
                  Float[Tensor, "num_classes dim"]],
             ) -> Float[Tensor, ""]:
    E_gammak, E_log_gammak, _, _, _ = expectations
    K = E_log_gammak.shape[0]
    return (
        K * (torch.log(alpha0) + 1 - torch.special.digamma(alpha0))
        + (E_log_gammak - E_gammak).sum())


def _d2L_alpha2(alpha0: Float[Tensor, ""],
                expectations: tuple[
                    Float[Tensor, "num_classes"],
                    Float[Tensor, "num_classes"],
                    Float[Tensor, "num_classes dim"],
                    Float[Tensor, "num_classes dim"],
                    Float[Tensor, "num_classes dim"]],
               ) -> Float[Tensor, ""]:
    E_gammak, _, _, _, _ = expectations
    K = E_gammak.shape[0]
    return K * (1. / alpha0 - torch.special.polygamma(1, alpha0))


def m_step_alpha(
    alpha0: Float[Tensor, ""],
    expectations: tuple[
            Float[Tensor, "num_classes"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"]],
        num_iters: int = 100
    ) -> Float[Tensor, ""]:
    """
    Maximize the expected log likelihood wrt alpha

    Raises
    ------
    AssertionError if alpha0 is not finite at any point during the optimization.
    """

    for i in range(num_iters):
        dL = _dL_alpha(alpha0, expectations)
        d2L = _d2L_alpha2(alpha0, expectations)
        alpha0 = alpha0**2 * d2L / (dL + alpha0 * d2L)
        alpha0 = torch.clamp(alpha0, 1e-3)
        assert torch.all(torch.isfinite(alpha0))
    return alpha0


def em(
        Nk: Float[Tensor, "num_classes"],
        sumx: Float[Tensor, "num_classes dim"],
        sumxsq: Float[Tensor, "num_classes dim"],
        nu0: Float[Tensor, "dim"],
        sigmasq0: Float[Tensor, "dim"],
        kappa0: Float[Tensor, "dim"],
        mu0: Float[Tensor, "dim"],
        alpha0: float | Float[Tensor, ""],
        fit_kappa0=True,
        fit_alpha0=True,
        num_iter: int=50,
        gammaks: Float[Tensor, "num_points"] = torch.logspace(-1, 1, 101, base=10),
        verbose: bool=False,
        pbar: bool=True,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
    lls = []
    kappas = [kappa0.to("cpu")]
    nus = [nu0.to("cpu")]
    alpha0 = torch.as_tensor(alpha0, device=sumx.device)
    alphas = [alpha0.to("cpu")]

    for i in trange(num_iter, disable=not pbar):
        # E step
        expectations, ll = e_step(Nk, sumx, sumxsq, nu0, sigmasq0, kappa0, mu0, alpha0, gammaks)

        # M step
        if fit_kappa0: kappa0 = m_step_kappa(expectations)
        if fit_alpha0: alpha0 = m_step_alpha(alpha0, expectations)
        nu0 = m_step_nu(nu0, sigmasq0, expectations)
        assert torch.all(torch.isfinite(kappa0))
        assert torch.all(torch.isfinite(nu0))
        assert torch.all(torch.isfinite(alpha0))

        # Track results
        lls.append(ll.item())
        kappas.append(kappa0.to('cpu'))
        nus.append(nu0.to('cpu'))
        alphas.append(alpha0.to('cpu'))
        if verbose: print(f"\nITR {i}, LL {lls[-1]}, nu: {nus[-1]}, kappa: {kappas[-1]}")

    return lls, torch.stack(kappas), torch.stack(nus), torch.stack(alphas)


def index_logsumexp(index: Int[Tensor, "num_idxs"], source: Float[Tensor, "num_data num_idxs"],
                    num_unique_idxs: int = None) -> Float[Tensor, "num_data num_unique_idxs"]:
    """Compute logsumexp over a set of indexs for a given dimension.

    Currently only supports indexing over last dimension.
    """
    dim = -1
    if num_unique_idxs is None:
        num_unique_idxs = index.unique().shape[0]
    K = num_unique_idxs
    other_dim = 0
    N = source.shape[other_dim]
    c = source.min() * torch.ones(N, K, device=source.device)
    # (N, K)
    c = torch.index_reduce(c, dim, index, source, "amax")
    # (N, num_idxs)
    exps = torch.exp(source - c[:, index])
    sum_exps = torch.zeros(N, K, device=source.device)
    sum_exps = torch.index_add(sum_exps, dim, index, exps)
    lse = c + torch.log(sum_exps)
    if not lse.isfinite().all():
        raise ValueError("Logsumexp is not finite")
    return lse


def posterior_pred_logprob(
        X: Float[Tensor, "num_data dim"],
        posterior_params: tuple[
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes num_points dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes dim"],
            Float[Tensor, "num_classes num_points"],
            Float[Tensor, "num_classes num_points"]],
        ) -> Float[Tensor, "num_data num_classes"]:
    """Compute the posterior predictive log probability
    """
    nu_post, sigmasq_post, kappa_post, mu_post, gammak_post, log_gammak_post = posterior_params
    K = nu_post.shape[0] if nu_post.ndim > 1 else 0
    if K == 0:
        kappa_post = kappa_post.unsqueeze(0)
        nu_post = nu_post.unsqueeze(0)
        sigmasq_post = sigmasq_post.unsqueeze(0)
        mu_post = mu_post.unsqueeze(0)
        gammak_post = gammak_post.unsqueeze(0)
        log_gammak_post = log_gammak_post.unsqueeze(0)
        K += 1

    # Remove gammaks that = 0
    mask = gammak_post > 0
    idxs = torch.where(mask)
    scale_post = torch.sqrt((kappa_post[idxs[0]] + 1) / kappa_post[idxs[0]] * sigmasq_post[mask])
    post = StudentT(
        nu_post[idxs[0]], # (mask.sum(), D)
        mu_post[idxs[0]], # (mask.sum(), D)
        scale_post, # (mask.sum(), D)
    )
    logprobs = log_gammak_post[mask] + post.log_prob(X[:, None, :]).sum(-1) # (N, mask.sum())
    logprobs = index_logsumexp(idxs[0], logprobs, num_unique_idxs=K) # (N, K)
    return logprobs.squeeze() # Remove K dimension if singleton


def compute_scores(prior_params,
                   posterior_params,
                   Z_te,
                   Ns,
                   return_preds=False,
                  ):
    """
    Compute log likelihood ratio for each data point
    """
    D = Z_te.shape[1]
    prior_pred_lps_te = posterior_pred_logprob(Z_te, prior_params)[:,None] # (N, 1)
    posterior_pred_lps_te = posterior_pred_logprob(Z_te, posterior_params)   # (N, K)

    scores_per_class = posterior_pred_lps_te - prior_pred_lps_te

    logNk = torch.log(Ns)
    scores = torch.logsumexp(scores_per_class + logNk, axis=1) # \tilde{C} in the paper
    if return_preds:
        preds = torch.argmax(posterior_pred_lps_te, -1)
        return scores, preds
    return scores
