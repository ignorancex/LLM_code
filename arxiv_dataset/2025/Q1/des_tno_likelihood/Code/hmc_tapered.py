from hmc_physical import *
from hmc_rolling_numerical import *
import jax
import jax.numpy as jnp

# cumulative form: 10^(alpha H) * exp(-10^(-beta*(H - Hc))) - note we are absorbing all scaling factors in here
# note also typo in Kavelaars et al - if beta < 0 in this form, this is not a proper cdf
# so the differential form is log(10) * 10^(alpha H) * exp(10^(-beta * (H-G))) * (alpha  - beta 10^(- beta* (H - Hc))))
# note we need to divide by the cumulative form at 3 and 10 to normalize - but no need to integrate numerically 
@numba.njit(fastmath=True)
def tapered_slope(H, alpha, b, H_crit):
    single = np.power(10, alpha * H)
    power10 = np.power(10, -b * (H - H_crit))
    taper = np.exp(-power10)
    
    diff = np.log(10) * single * taper * (alpha + b * power10)
    Hmm = np.array([3., 10.])

    norm = np.power(10, alpha * Hmm) * np.exp(-np.power(10, -b * (Hmm - H_crit)))
    
    return diff/(norm[1] - norm[0])


@jax.jit
def tapered_slope_jax(H, alpha, b, H_crit):
    single = jnp.power(10, alpha * H)
    power10 = jnp.power(10, -b * (H - H_crit))
    taper = jnp.exp(-power10)
    
    diff = jnp.log(10) * single * taper * (alpha + b * power10)
    Hmm = np.array([3., 10.])

    norm = jnp.power(10, alpha * Hmm) * jnp.exp(-jnp.power(10, -b * (Hmm - H_crit)))
    
    return diff/(norm[1] - norm[0])



Hbins = np.linspace(3, 10, 1000)

grad_alpha = jax.jit(
    jax.vmap(jax.grad(tapered_slope_jax, (1)), in_axes=[0, None, None, None])
)

grad_beta = jax.jit(
    jax.vmap(jax.grad(tapered_slope_jax, (2)), in_axes=[0, None, None, None])
)

grad_Hcrit = jax.jit(
    jax.vmap(jax.grad(tapered_slope_jax, (3)), in_axes=[0, None, None, None])
)



@numba.njit(parallel=True, fastmath=True, error_model="numpy")
def Rialpha_tapered_lca_physical(
    ncomp, nobj, nphys, jacob, pref, chain_H, chain_LCA, params
):
    Rialpha = np.zeros((nobj, ncomp))
    for alpha in range(nphys):
        a = params[5 * alpha]
        b = params[5 * alpha + 1]
        H_crit = params[5*alpha + 2]
        Abar = params[5 * alpha + 3]
        s = params[5 * alpha + 4]
        norm = normalize_beta(Abar, s)
        # parallelize here
        for i in numba.prange(nobj):
            size_chain_comp = np.log(
                tapered_slope(chain_H[i], a, b, H_crit) 
            )
            lca_chain_comp = np.log(beta(chain_LCA[i], Abar, s, norm))
            # and here?
            for comp in range(ncomp):
                if comp % nphys == alpha:
                    Rialpha[i, comp] = np.sum(
                        np.exp(
                            jacob[i]
                            + pref[i][:, comp]
                            + size_chain_comp
                            + lca_chain_comp
                        )
                    ) / len(chain_H[i])

    return Rialpha


@numba.njit(parallel=True, fastmath=True, error_model="numpy")
def Rialphaderiv_tapered_lca_physical(
    alpha,
    ncomp,
    nobj,
    nphys,
    jacob,
    pref,
    chain_H,
    chain_LCA,
    tapered_deriv_alpha,
    tapered_deriv_beta,
    tapered_deriv_Hcrit,
    beta_deriv_Abar,
    beta_deriv_s,
    a,
    b,
    H_crit,
    Abar,
    s,
):
    Rialpha_a = np.zeros((nobj, ncomp))
    Rialpha_b = np.zeros((nobj, ncomp))
    Rialpha_Hcrit = np.zeros((nobj, ncomp))
    Rialpha_Abar = np.zeros((nobj, ncomp))
    Rialpha_s = np.zeros((nobj, ncomp))

    norm = normalize_beta(Abar, s)

    for i in numba.prange(nobj):
        size_chain_comp = np.log(tapered_slope(chain_H[i], a, b, H_crit))
        lca_chain_comp = np.log(beta(chain_LCA[i], Abar, s, norm))
        for comp in range(ncomp):
            if comp % nphys == alpha:
                Rialpha_a[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + lca_chain_comp)
                    * tapered_deriv_alpha[i]
                ) / len(chain_H[i])
                Rialpha_b[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + lca_chain_comp)
                    * tapered_deriv_beta[i]
                ) / len(chain_H[i])
                Rialpha_Hcrit[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + lca_chain_comp)
                    * tapered_deriv_Hcrit[i]
                ) / len(chain_H[i])

                Rialpha_Abar[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    * beta_deriv_Abar[i]
                ) / len(chain_H[i])
                Rialpha_s[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    * beta_deriv_s[i]
                ) / len(chain_H[i])

    return Rialpha_a, Rialpha_b, Rialpha_Hcrit, Rialpha_Abar, Rialpha_s






class LogLikeGMMDynamicsTapered(LogLikeGMMDynamicsRolling):
    def Rialpha(self, params):
        return Rialpha_tapered_lca_physical(
            self.ncomp,
            self.nobj,
            self.nphys,
            self.jacobian,
            self.S_obj_noamp_physical,
            self.chains_H,
            self.chains_LCA,
            params,
        )

    def Salpha(self, params, fun):
        # Note that the integral of the LCA distribution times the selection function is constant (=1), so we don't need (Abar,s)_alpha in here
        # This is because p(s|A) ~ const for all A
        # Extra care is needed to get the parameters in the right place. We only need the slopes:
        a = params[[5 * i for i in range(self.nphys)]]
        b = params[[5 * i + 1 for i in range(self.nphys)]]
        Hcrit = params[[5*i + 2 for i in range(self.nphys)]]
        f = []
        for alpha in range(self.nphys):
            f.append(fun(self.H_bins, a[alpha], b[alpha], Hcrit[alpha]))

        S = np.zeros(self.ncomp)
        for alpha in range(self.ncomp):
            integrand = self.selection[alpha] * f[alpha % self.nphys]

            S[alpha] = (
                np.sum((integrand[1:] + integrand[:-1]) * np.diff(self.H_bins)) / 2
            )
        return S

    def size_dist(self, H, a, b, Hcrit):
        return tapered_slope(H, a, b, Hcrit)

    def gradient_theta(self, falpha, params):
        self.Rialpha_current = self.Rialpha(params)

        self.Salpha_current = self.Salpha(params, self.size_dist)

        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        # Salpha_current = self.Salpha(params, self.size_dist)
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        # different from other cases now
        self.SalphatimesH_a = self.Salpha(params, grad_alpha)
        self.SalphatimesH_b = self.Salpha(params, grad_beta)
        self.SalphatimesH_Hcrit = self.Salpha(params, grad_Hcrit)

        # this is where it gets tricky - this is now 3nphys d
        grad = np.zeros_like(params)

        for alpha in range(self.nphys):
            # unpack
            a = params[5 * alpha]
            b = params[5 * alpha + 1]
            H_crit = params[5*alpha + 2]
            Abar = params[5 * alpha + 3]
            s = params[5 * alpha + 4]
            # now calculate the Rialpha sums

            tapered_deriv_alpha = np.array(
                grad_alpha(self.chains_H_flat, a, b, H_crit)
            )
            tapered_deriv_beta = np.array(
                grad_beta(self.chains_H_flat, a, b, H_crit)
            )
            tapered_deriv_Hcrit = np.array(
                grad_Hcrit(self.chains_H_flat, a, b, H_crit)
            )
            beta_deriv_Abar = np.array(grad_Abar(self.chains_flat, Abar, s))
            beta_deriv_s = np.array(grad_s(self.chains_flat, Abar, s))

            (
                Rialpha_deriv_alpha,
                Rialpha_deriv_beta,
                Rialpha_deriv_Hcrit,
                Rialpha_deriv_Abar,
                Rialpha_deriv_s,
            ) = Rialphaderiv_tapered_lca_physical(
                alpha,
                self.ncomp,
                self.nobj,
                self.nphys,
                self.jacobian,
                self.S_obj_noamp_physical,
                self.chains_H,
                self.chains_LCA,
                self.splitArray(tapered_deriv_alpha),
                self.splitArray(tapered_deriv_beta),
                self.splitArray(tapered_deriv_Hcrit),
                self.splitArray(beta_deriv_Abar),
                self.splitArray(beta_deriv_s),
                a,
                b,
                H_crit,
                Abar,
                s,
            )

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_alpha_sum = np.sum(Rialpha_deriv_alpha * falpha, axis=1)
            Rialpha_beta_sum = np.sum(Rialpha_deriv_beta * falpha, axis=1)
            Rialpha_Hcrit_sum = np.sum(Rialpha_deriv_Hcrit * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_deriv_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_deriv_s * falpha, axis=1)

            gradRialpha_alpha = np.sum(Rialpha_alpha_sum / self.falphaRalpha)
            gradRialpha_beta = np.sum(Rialpha_beta_sum / self.falphaRalpha)
            gradRialpha_Hcrit = np.sum(Rialpha_Hcrit_sum/self.falphaRalpha)
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params)

            SalphatimesH_comp_a = np.zeros_like(self.SalphatimesH_a)
            SalphatimesH_comp_b = np.zeros_like(self.SalphatimesH_b)
            SalphatimesH_comp_Hcrit = np.zeros_like(self.SalphatimesH_Hcrit)
            
            for comp in range(self.ncomp):
                if comp % self.nphys == alpha:
                    SalphatimesH_comp_a[comp] = self.SalphatimesH_a[comp]
                    SalphatimesH_comp_b[comp] = self.SalphatimesH_b[comp]
                    SalphatimesH_comp_Hcrit[comp] = self.SalphatimesH_Hcrit[comp]
                    

            # now put things in place
            grad[5 * alpha] = (
                gradRialpha_alpha
                - self.nobj * np.dot(SalphatimesH_comp_a, falpha) / Salpha_sum
                + prior_deriv[5 * alpha]
            )
            grad[5 * alpha + 1] = (
                gradRialpha_beta
                - self.nobj * np.dot(SalphatimesH_comp_b, falpha) / Salpha_sum
                + prior_deriv[5 * alpha + 1]
            )
            grad[5 * alpha + 2] = (
                gradRialpha_Hcrit
                - self.nobj * np.dot(SalphatimesH_comp_Hcrit, falpha) / Salpha_sum
                + prior_deriv[5 * alpha + 2]
            )

            grad[5 * alpha + 3] = gradRialpha_Abar + prior_deriv[5 * alpha + 3]
            grad[5 * alpha + 4] = gradRialpha_s + prior_deriv[5 * alpha + 4]
        return self.pref * grad


class LogLikeGMMDynamicsTaperedSameSlope(LogLikeGMMDynamicsTapered):
    def __init__(
        self,
        gmm,
        chains_color,
        chains_H,
        chains_LCA,
        selection,
        ncomp,
        physical_assignment,
        pref=-1,
        prior=None,
        prior_deriv=None,
        nphys=None,
    ):
        super().__init__(
            gmm,
            chains_color,
            chains_H,
            chains_LCA,
            selection,
            ncomp,
            physical_assignment,
            pref,
            prior,
            prior_deriv,
            nphys,
        )
        self.eff_index = []
        for i in range(self.nphys):
            self.eff_index += [0, 1, 2]
            self.eff_index += [3 + 2 * i, 3 + 2 * i + 1]

    def Rialpha(self, params):
        eff_params = params[self.eff_index]
        return super().Rialpha(eff_params)

    def Salpha(self, params, fun):
        eff_params = params[self.eff_index]
        return super().Salpha(eff_params, fun)

    def gradient_theta(self, falpha, params):
        ## Assumes we have already computed Rialpha and Salpha
        self.Rialpha_current = self.Rialpha(params)

        self.Salpha_current = self.Salpha(params, self.size_dist)

        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        # Salpha_current = self.Salpha(params, self.size_dist)
        Salpha_sum = np.dot(falpha, self.Salpha_current)
       
        # different from other cases now
        self.SalphatimesH_a = self.Salpha(params, grad_alpha)
        self.SalphatimesH_b = self.Salpha(params, grad_beta)
        self.SalphatimesH_Hcrit = self.Salpha(params, grad_Hcrit)

        grad = np.zeros_like(params)

        a = params[0]
        b = params[1]
        Hcrit = params[2]
        tapered_deriv_alpha = np.array(grad_alpha(self.chains_H_flat, a, b, Hcrit))
        tapered_deriv_beta = np.array(grad_beta(self.chains_H_flat, a, b, Hcrit))
        tapered_deriv_Hcrit = np.array(grad_Hcrit(self.chains_H_flat, a, b, Hcrit))
        for alpha in range(self.nphys):
            # unpack
            Abar = params[2 * alpha + 3]
            s = params[2 * alpha + 4]

            # now calculate the Rialpha sums

            beta_deriv_Abar = np.array(grad_Abar(self.chains_flat, Abar, s))
            beta_deriv_s = np.array(grad_s(self.chains_flat, Abar, s))

            (
                Rialpha_deriv_alpha,
                Rialpha_deriv_beta,
                Rialpha_deriv_Hcrit,
                Rialpha_deriv_Abar,
                Rialpha_deriv_s,
            ) = Rialphaderiv_tapered_lca_physical(
                alpha,
                self.ncomp,
                self.nobj,
                self.nphys,
                self.jacobian,
                self.S_obj_noamp_physical,
                self.chains_H,
                self.chains_LCA,
                self.splitArray(tapered_deriv_alpha),
                self.splitArray(tapered_deriv_beta),
                self.splitArray(tapered_deriv_Hcrit),
                self.splitArray(beta_deriv_Abar),
                self.splitArray(beta_deriv_s),
                a,
                b,
                Hcrit,
                Abar,
                s,
            )
            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_alpha_sum = np.sum(Rialpha_deriv_alpha * falpha, axis=1)
            Rialpha_beta_sum = np.sum(Rialpha_deriv_beta * falpha, axis=1)
            Rialpha_Hcrit_sum = np.sum(Rialpha_deriv_Hcrit * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_deriv_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_deriv_s * falpha, axis=1)

            gradRialpha_alpha = np.sum(Rialpha_alpha_sum / self.falphaRalpha)
            gradRialpha_beta = np.sum(Rialpha_beta_sum / self.falphaRalpha)
            gradRialpha_Hcrit = np.sum(Rialpha_Hcrit_sum/self.falphaRalpha)
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params)

            SalphatimesH_comp_a = np.zeros_like(self.SalphatimesH_a)
            SalphatimesH_comp_b = np.zeros_like(self.SalphatimesH_b)
            SalphatimesH_comp_Hcrit = np.zeros_like(self.SalphatimesH_Hcrit)

            for comp in range(self.ncomp):
                if comp % self.nphys == alpha:
                    SalphatimesH_comp_a[comp] = self.SalphatimesH_a[comp]
                    SalphatimesH_comp_b[comp] = self.SalphatimesH_b[comp]
                    SalphatimesH_comp_Hcrit[comp] = self.SalphatimesH_Hcrit[comp]
                    

            # now put things in place
            grad[0] += (
                gradRialpha_alpha
                - self.nobj * np.dot(SalphatimesH_comp_a, falpha) / Salpha_sum
                + prior_deriv[0]
            )
            grad[1] += (
                gradRialpha_beta
                - self.nobj * np.dot(SalphatimesH_comp_b, falpha) / Salpha_sum
                + prior_deriv[1]
            )
            grad[2] += (gradRialpha_Hcrit - self.nobj*np.dot(SalphatimesH_comp_Hcrit, falpha)/Salpha_sum + prior_deriv[2])
            grad[2 * alpha + 3] = gradRialpha_Abar + prior_deriv[2 * alpha + 3]
            grad[2 * alpha + 4] = gradRialpha_s + prior_deriv[2 * alpha + 4]
        return self.pref * grad
