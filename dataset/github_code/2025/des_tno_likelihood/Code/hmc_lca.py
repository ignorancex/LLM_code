from hmc_gradient import *

import jax
import jax.numpy as jnp


def beta_jax(x, Abar, s):
    alpha = Abar * jnp.power(10, s)
    beta = alpha / Abar - alpha
    return jax.scipy.stats.beta.pdf(x, alpha, beta)


grad_Abar = jax.jit(jax.vmap(jax.grad(beta_jax, (1)), in_axes=[0, None, None]))
grad_s = jax.jit(jax.vmap(jax.grad(beta_jax, (2)), in_axes=[0, None, None]))


@numba.njit(fastmath=True, error_model="numpy")
def gamma(z):
    """Numerical Recipes 6.1"""
    coefs = np.array(
        [
            57.1562356658629235,
            -59.5979603554754912,
            14.1360979747417471,
            -0.491913816097620199,
            0.339946499848118887e-4,
            0.465236289270485756e-4,
            -0.983744753048795646e-4,
            0.158088703224912494e-3,
            -0.210264441724104883e-3,
            0.217439618115212643e-3,
            -0.164318106536763890e-3,
            0.844182239838527433e-4,
            -0.261908384015814087e-4,
            0.368991826595316234e-5,
        ]
    )

    out = np.empty(z.shape[0])
    scale = np.ones_like(z)
    neg = True
    while neg:
        scale[z < 0] *= 1 / z[z < 0]
        z[z < 0] += 1
        if len(z[z < 0]) == 0:
            neg = False

    for i in range(z.shape[0]):
        y = z[i]
        tmp = z[i] + 5.24218750000000000
        tmp = (z[i] + 0.5) * np.log(tmp) - tmp
        ser = 0.999999999999997092

        n = coefs.shape[0]
        for j in range(n):
            y = y + 1.0
            ser = ser + coefs[j] / y

        out[i] = tmp + np.log(2.5066282746310005 * ser / z[i])
    return np.exp(out) * scale


@numba.njit(fastmath=True)
def normalize_beta(Abar, s):
    # Arange = np.linspace(0,1,100_000)
    # Aterm = np.power(Arange, Abar * np.power(10, s) - 1)
    # Aminus1term = np.power(1-Arange, (1-Abar) * np.power(10,s) - 1)
    # prod = Aterm * Aminus1term
    alpha = Abar * np.power(10, s)
    beta = alpha / Abar - alpha
    gammas = gamma(np.array([alpha, beta, alpha + beta]))
    return gammas[-1] / (gammas[0] * gammas[1])


@numba.njit(fastmath=True)
def beta(A, Abar, s, norm):
    Aterm = np.power(A, Abar * np.power(10, s) - 1)
    Aminus1term = np.power(1 - A, (1 - Abar) * np.power(10, s) - 1)
    # norm = normalize_beta(Abar, s)
    return Aterm * Aminus1term * norm


@numba.njit(parallel=True, fastmath=True)
def Rialpha_singleslope_lca(ncomp, nobj, jacob, pref, chain_H, chain_LCA, params):
    Rialpha = np.zeros((nobj, ncomp))
    for alpha in range(ncomp):
        slope = params[3 * alpha]
        Abar = params[3 * alpha + 1]
        s = params[3 * alpha + 2]
        norm = normalize_beta(Abar, s)
        for i in numba.prange(nobj):
            size_chain_comp = np.log(singleslope(chain_H[i], slope))
            lca_chain_comp = np.log(beta(chain_LCA[i], Abar, s, norm))

            Rialpha[i, alpha] = max(
                1e-45,
                np.sum(
                    np.exp(
                        jacob[i] + pref[i][:, alpha] + size_chain_comp + lca_chain_comp
                    )
                )
                / len(chain_H[i]),
            )

    return Rialpha

@numba.njit(parallel=True, fastmath=True)
def Rialpha_lca(ncomp, nobj, jacob, pref, chain_LCA, params):
    Rialpha = np.zeros((nobj, ncomp))
    for alpha in range(ncomp):
        Abar = params[2 * alpha ]
        s = params[2 * alpha + 1]
        norm = normalize_beta(Abar, s)
        for i in numba.prange(nobj):
            lca_chain_comp = np.log(beta(chain_LCA[i], Abar, s, norm))
            Rialpha[i, alpha] = max(
                1e-45,
                np.sum(
                    np.exp(
                        jacob[i] + pref[i][:, alpha] + lca_chain_comp
                    )
                )
                / len(chain_LCA[i]),
            )

    return Rialpha


@numba.njit(parallel=True, fastmath=True)
def Rialphaderiv_singleslope_lca(
    alpha,
    nobj,
    jacob,
    pref,
    chain_H,
    chain_LCA,
    beta_deriv_Abar,
    beta_deriv_s,
    slope,
    Abar,
    s,
):
    Rialpha_slope = np.zeros(nobj)
    Rialpha_Abar = np.zeros(nobj)
    Rialpha_s = np.zeros(nobj)

    norm = normalize_beta(Abar, s)
    for i in numba.prange(nobj):
        slope_deriv = singleslope_deriv(chain_H[i], slope)
        size_chain_comp = np.log(singleslope(chain_H[i], slope))
        lca_chain_comp = np.log(beta(chain_LCA[i], Abar, s, norm))

        Rialpha_slope[i] = np.sum(
            np.exp(jacob[i] + pref[i][:, alpha] + lca_chain_comp) * slope_deriv
        ) / len(chain_H[i])
        Rialpha_Abar[i] = np.sum(
            np.exp(jacob[i] + pref[i][:, alpha] + size_chain_comp) * beta_deriv_Abar[i]
        ) / len(chain_H[i])
        Rialpha_s[i] = np.sum(
            np.exp(jacob[i] + pref[i][:, alpha] + size_chain_comp) * beta_deriv_s[i]
        ) / len(chain_H[i])

    return Rialpha_slope, Rialpha_Abar, Rialpha_s

@numba.njit(parallel=True, fastmath=True)
def Rialphaderiv_lca(
    alpha,
    nobj,
    jacob,
    pref,
    chain_LCA,
    beta_deriv_Abar,
    beta_deriv_s,
):
    Rialpha_Abar = np.zeros(nobj)
    Rialpha_s = np.zeros(nobj)

    for i in numba.prange(nobj):
        Rialpha_Abar[i] = np.sum(
            np.exp(jacob[i] + pref[i][:, alpha] ) * beta_deriv_Abar[i]
        ) / len(chain_LCA[i])
        Rialpha_s[i] = np.sum(
            np.exp(jacob[i] + pref[i][:, alpha] ) * beta_deriv_s[i]
        ) / len(chain_LCA[i])

    return Rialpha_Abar, Rialpha_s


class LogLikeGMMLightCurve(LogLikeGMMSingleSlope):
    def __init__(
        self,
        gmm,
        chains_color,
        chains_H,
        chains_LCA,
        selection,
        ncomp,
        pref=-1,
        prior=None,
        prior_deriv=None,
    ):
        super().__init__(
            gmm,
            chains_color,
            chains_H,
            selection,
            ncomp,
            pref,
            prior,
            prior_deriv,
        )
        self.chains_LCA = numba.typed.List(chains_LCA)

        indexes = [len(i) for i in chains_LCA]
        self.indexes = np.cumsum(indexes)[:-1]
        self.chains_flat = np.concatenate(chains_LCA)
        if prior is None:
            self.prior = lambda x: 0
            self.prior_deriv = lambda x: np.zeros(3 * ncomp)
        else:
            self.prior = prior
            self.prior_deriv = prior_deriv

    def Rialpha(self, params):
        return Rialpha_singleslope_lca(
            self.ncomp,
            self.nobj,
            self.jacobian,
            self.S_obj_noamp,
            self.chains_H,
            self.chains_LCA,
            params,
        )

    def Salpha(self, params, fun):
        # Note that the integral of the LCA distribution times the selection function is constant (=1), so we don't need (Abar,s)_alpha in here
        # This is because p(s|A) ~ const for all A
        # Extra care is needed to get the parameters in the right place. We only need the slopes:
        params = params[[3 * i for i in range(self.ncomp)]]
        f = fun(params, self.H_bins)

        S = np.zeros(self.ncomp)
        for alpha in range(self.ncomp):
            integrand = self.selection[alpha] * f[:, alpha]

            S[alpha] = (
                np.sum((integrand[1:] + integrand[:-1]) * np.diff(self.H_bins)) / 2
            )
        return S

    def gradient_theta(self, falpha, params):
        ## Assumes we have already computed Rialpha and Salpha
        self.Rialpha_current = self.Rialpha(params)

        self.Salpha_current = self.Salpha(params, self.size_dist)

        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        # Salpha_current = self.Salpha(params, self.size_dist)
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand = lambda params, H: singleslope_deriv_multi(H, params)

        # sneaky bug!
        self.SalphatimesH = self.Salpha(params, SHintegrand)

        # this is where it gets tricky - this is now 3n d
        grad = np.zeros_like(params)

        for alpha in range(self.ncomp):
            # unpack
            slope = params[3 * alpha]
            Abar = params[3 * alpha + 1]
            s = params[3 * alpha + 2]

            # evaluate jax gradients at each parameter over the chains
            deriv_chain_Abar = np.array(grad_Abar(self.chains_flat, Abar, s))
            deriv_chain_s = np.array(grad_s(self.chains_flat, Abar, s))

            # now calculate the Rialpha sums
            Rialpha_slope, Rialpha_Abar, Rialpha_s = Rialphaderiv_singleslope_lca(
                alpha,
                self.nobj,
                self.jacobian,
                self.S_obj_noamp,
                self.chains_H,
                self.chains_LCA,
                self.splitArray(deriv_chain_Abar),
                self.splitArray(deriv_chain_s),
                slope,
                Abar,
                s,
            )

            gradRialpha_slope = np.sum(Rialpha_slope / self.falphaRalpha)
            gradRialpha_Abar = np.sum(Rialpha_Abar / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

            # now put things in place
            grad[3 * alpha] = (
                gradRialpha_slope * falpha[alpha]
                - self.nobj * falpha[alpha] * self.SalphatimesH[alpha] / Salpha_sum
                + prior_deriv[3 * alpha]
            )
            grad[3 * alpha + 1] = (
                gradRialpha_Abar * falpha[alpha] + prior_deriv[3 * alpha + 1]
            )
            grad[3 * alpha + 2] = (
                gradRialpha_s * falpha[alpha] + prior_deriv[3 * alpha + 2]
            )

        return self.pref * grad
    
    def splitArray(self, array):
        return numba.typed.List(np.split(array, self.indexes))

class LogLikeGMMNoColor(LogLikeGMMLightCurve):
    def __init__(
        self,
        gmm,
        chains_color,
        chains_H,
        chains_LCA,
        selection,
        ncomp,
        pref=-1,
        prior=None,
        prior_deriv=None,
    ):
        super().__init__(
            gmm,
            chains_color,
            chains_H,
            chains_LCA,
            selection,
            1,
            pref,
            prior,
            prior_deriv,
        )

    def compute_jacobian_chain(self):
        self.jacobian = []

        for i in range(self.nobj):
            self.jacobian.append(jacobian_H(self.chains_H[i]))

        self.jacobian = numba.typed.List(self.jacobian)

    def compute_gmm_chain(self):
        self.S_obj_noamp = []
        # logdet = np.linalg.slogdet(self.gmm.cov_best)[1]

        for x in range(len(self.chains_color)):
            self.S_obj_noamp.append(np.zeros((len(self.chains_H[x]), 1)))

        self.S_obj_noamp = numba.typed.List(self.S_obj_noamp)

    def likelihood(self, theta):
        return super().likelihood(np.array([1.0]), theta)

    def gradient_theta(self, theta):
        return super().gradient_theta(np.array([1.0]), theta)


class LogLikeGMMLightCurveNoH(LogLikeGMMLightCurve):
    def __init__(
        self,
        gmm,
        chains_color,
        chains_LCA,
        ncomp,
        pref=-1,
        prior=None,
        prior_deriv=None,
    ):
        self.gmm = gmm 
        self.chains_color = chains_color

        self.nobj = len(chains_LCA)

        self.ncomp = ncomp 

        self.rng = np.random.default_rng()
        
        self.chains_LCA = numba.typed.List(chains_LCA)

        indexes = [len(i) for i in chains_LCA]
        self.indexes = np.cumsum(indexes)[:-1]
        self.chains_flat = np.concatenate(chains_LCA)


        self.pref = pref
        self.compute_gmm_chain()
        self.compute_jacobian_chain()

        if prior is None:
            self.prior = lambda x : 0
            self.prior_deriv = lambda x : np.zeros(2*ncomp)
        else:
            self.prior = prior 
            self.prior_deriv = prior_deriv
	
    def compute_jacobian_chain(self):
        self.jacobian = []

        for i in range(self.nobj):
            self.jacobian.append(jacobian_color(self.chains_color[i]))

        self.jacobian = numba.typed.List(self.jacobian)

    def Rialpha(self, params):
        return Rialpha_lca(
            self.ncomp,
            self.nobj,
            self.jacobian,
            self.S_obj_noamp,
            self.chains_LCA,
            params,
        )
	
    def likelihood(self, falpha, params):
        self.Rialpha_current = self.Rialpha(params)
        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        prior = self.prior(params)

        return self.pref*(np.sum(np.log(self.falphaRalpha))  + prior)

    def gradient_theta(self, falpha, params):
        ## Assumes we have already computed Rialpha and Salpha
        self.Rialpha_current = self.Rialpha(params)
        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        grad = np.zeros_like(params)

        for alpha in range(self.ncomp):
            # unpack
            Abar = params[2* alpha ]
            s = params[2 * alpha + 1]

            # evaluate jax gradients at each parameter over the chains
            deriv_chain_Abar = np.array(grad_Abar(self.chains_flat, Abar, s))
            deriv_chain_s = np.array(grad_s(self.chains_flat, Abar, s))

            # now calculate the Rialpha sums
            Rialpha_Abar, Rialpha_s = Rialphaderiv_lca(
                alpha,
                self.nobj,
                self.jacobian,
                self.S_obj_noamp,
                self.chains_LCA,
                self.splitArray(deriv_chain_Abar),
                self.splitArray(deriv_chain_s),
            )

            gradRialpha_Abar = np.sum(Rialpha_Abar / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

            # now put things in place
            grad[2 * alpha ] = (
                gradRialpha_Abar * falpha[alpha] + prior_deriv[2 * alpha ]
            )
            grad[2 * alpha + 1] = (
                gradRialpha_s * falpha[alpha] + prior_deriv[2 * alpha + 1]
            )

        return self.pref * grad
    

    def gradient_f(self, falpha, params):
        grad = np.zeros_like(falpha)
        self.Rialpha_current = self.Rialpha(params)
        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        for alpha in range(self.ncomp):
            gradsumRialpha = np.sum(self.Rialpha_current[:,alpha]/self.falphaRalpha)
            grad[alpha] = gradsumRialpha

        return self.pref*grad


class LogLikeGMMLightCurveNoHSingle(LogLikeGMMLightCurveNoH):
    def likelihood(self, falpha, params):
        return super().likelihood(falpha, np.array([*params, *params]))
    def gradient_theta(self, falpha, params):
        return super().gradient_theta(falpha, np.array([*params, *params]))[2:]
    def gradient_f(self, falpha, params):
        return super().gradient_f(falpha, np.array([*params, *params]))    