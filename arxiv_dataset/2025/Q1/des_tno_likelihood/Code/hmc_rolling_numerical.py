from hmc_physical import *

import jax
import jax.numpy as jnp


@numba.njit(fastmath=True)
def rolling_slope(H, theta1, theta2):
    Heff = H - 7
    roll = np.power(10, Heff * theta1 + Heff * Heff * theta2)
    return roll


@jax.jit
def rolling_slope_jax(H, theta1, theta2):
    Heff = H - 7
    roll = jnp.power(10, Heff * theta1 + Heff * Heff * theta2)
    return roll


@numba.njit(fastmath=True)
def normalize_rolling(slope1, slope2):
    roll_H = rolling_slope(Hbins, slope1, slope2)
    integral = np.sum((roll_H[1:] + roll_H[:-1]) * np.diff(Hbins) / 2)
    return integral


Hbins = np.linspace(5, 9, 1000)

grad_slope1 = jax.jit(
    jax.vmap(jax.grad(rolling_slope_jax, (1)), in_axes=[0, None, None])
)
grad_slope2 = jax.jit(
    jax.vmap(jax.grad(rolling_slope_jax, (2)), in_axes=[0, None, None])
)


def del_double_del_x(H, slope1, slope2, grad_x):
    norm = normalize_rolling(slope1, slope2)
    derivative = grad_x(H, slope1, slope2)
    roll = rolling_slope(H, slope1, slope2)
    deriv = np.array(grad_x(Hbins, slope1, slope2))
    integral = np.sum((deriv[1:] + deriv[:-1]) * np.diff(Hbins) / 2)

    return np.array(derivative - (roll / norm) * integral) / norm


@numba.njit(parallel=True, fastmath=True, error_model="numpy")
def Rialpha_rolling_lca_physical(
    ncomp, nobj, nphys, jacob, pref, chain_H, chain_LCA, params
):
    Rialpha = np.zeros((nobj, ncomp))
    for alpha in range(nphys):
        theta1 = params[4 * alpha]
        theta2 = params[4 * alpha + 1]
        Abar = params[4 * alpha + 2]
        s = params[4 * alpha + 3]
        norm = normalize_beta(Abar, s)
        norm_roll = normalize_rolling(theta1, theta2)
        # parallelize here
        for i in numba.prange(nobj):
            size_chain_comp = np.log(
                rolling_slope(chain_H[i], theta1, theta2) / norm_roll
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
def Rialphaderiv_rolling_lca_physical(
    alpha,
    ncomp,
    nobj,
    nphys,
    jacob,
    pref,
    chain_H,
    chain_LCA,
    rolling_deriv_theta1,
    rolling_deriv_theta2,
    beta_deriv_Abar,
    beta_deriv_s,
    theta1,
    theta2,
    Abar,
    s,
):
    Rialpha_theta1 = np.zeros((nobj, ncomp))
    Rialpha_theta2 = np.zeros((nobj, ncomp))
    Rialpha_Abar = np.zeros((nobj, ncomp))
    Rialpha_s = np.zeros((nobj, ncomp))

    norm = normalize_beta(Abar, s)
    norm_roll = normalize_rolling(theta1, theta2)

    for i in numba.prange(nobj):
        size_chain_comp = np.log(rolling_slope(chain_H[i], theta1, theta2) / norm_roll)
        lca_chain_comp = np.log(beta(chain_LCA[i], Abar, s, norm))
        for comp in range(ncomp):
            if comp % nphys == alpha:
                Rialpha_theta1[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + lca_chain_comp)
                    * rolling_deriv_theta1[i]
                ) / len(chain_H[i])
                Rialpha_theta2[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + lca_chain_comp)
                    * rolling_deriv_theta2[i]
                ) / len(chain_H[i])

                Rialpha_Abar[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    * beta_deriv_Abar[i]
                ) / len(chain_H[i])
                Rialpha_s[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    * beta_deriv_s[i]
                ) / len(chain_H[i])

    return Rialpha_theta1, Rialpha_theta2, Rialpha_Abar, Rialpha_s


@numba.njit(parallel=True, fastmath=True, error_model="numpy")
def Rialpha_rolling_physical(ncomp, nobj, nphys, jacob, pref, chain_H, params):
    Rialpha = np.zeros((nobj, ncomp))
    for alpha in range(nphys):
        theta1 = params[2 * alpha]
        theta2 = params[2 * alpha + 1]
        norm_roll = normalize_rolling(theta1, theta2)
        # parallelize here
        for i in numba.prange(nobj):
            size_chain_comp = np.log(
                rolling_slope(chain_H[i], theta1, theta2) / norm_roll
            )
            # and here?
            for comp in range(ncomp):
                if comp % nphys == alpha:
                    Rialpha[i, comp] = np.sum(
                        np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    ) / len(chain_H[i])

    return Rialpha


@numba.njit(parallel=True, fastmath=True, error_model="numpy")
def Rialphaderiv_rolling_physical(
    alpha,
    ncomp,
    nobj,
    nphys,
    jacob,
    pref,
    chain_H,
    rolling_deriv_theta1,
    rolling_deriv_theta2,
    theta1,
    theta2,
):
    Rialpha_theta1 = np.zeros((nobj, ncomp))
    Rialpha_theta2 = np.zeros((nobj, ncomp))


    for i in numba.prange(nobj):
        for comp in range(ncomp):
            if comp % nphys == alpha:
                Rialpha_theta1[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp]) * rolling_deriv_theta1[i]
                ) / len(chain_H[i])
                Rialpha_theta2[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp]) * rolling_deriv_theta2[i]
                ) / len(chain_H[i])

    return Rialpha_theta1, Rialpha_theta2


class LogLikeGMMDynamicsRolling(LogLikeGMMLCADynamics):
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
        self.chains_H_flat = np.concatenate(chains_H)

    def Rialpha(self, params):
        return Rialpha_rolling_lca_physical(
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
        theta1 = params[[4 * i for i in range(self.nphys)]]
        theta2 = params[[4 * i + 1 for i in range(self.nphys)]]
        f = []
        for alpha in range(self.nphys):
            f.append(fun(self.H_bins, theta1[alpha], theta2[alpha]))

        S = np.zeros(self.ncomp)
        for alpha in range(self.ncomp):
            integrand = self.selection[alpha] * f[alpha % self.nphys]

            S[alpha] = (
                np.sum((integrand[1:] + integrand[:-1]) * np.diff(self.H_bins)) / 2
            )
        return S

    def size_dist(self, H, theta1, theta2):
        norm = normalize_rolling(theta1, theta2)
        return rolling_slope(H, theta1, theta2) / norm

    def gradient_theta(self, falpha, params):
        ## Assumes we have already computed Rialpha and Salpha
        self.Rialpha_current = self.Rialpha(params)

        self.Salpha_current = self.Salpha(params, self.size_dist)

        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        # Salpha_current = self.Salpha(params, self.size_dist)
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand_theta1 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope1
        )
        SHintegrand_theta2 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope2
        )

        # different from other cases now
        self.SalphatimesH_theta1 = self.Salpha(params, SHintegrand_theta1)
        self.SalphatimesH_theta2 = self.Salpha(params, SHintegrand_theta2)

        # this is where it gets tricky - this is now 3nphys d
        grad = np.zeros_like(params)

        for alpha in range(self.nphys):
            # unpack
            theta1 = params[4 * alpha]
            theta2 = params[4 * alpha + 1]
            Abar = params[4 * alpha + 2]
            s = params[4 * alpha + 3]

            # now calculate the Rialpha sums

            rolling_deriv_theta1 = np.array(
                SHintegrand_theta1(self.chains_H_flat, theta1, theta2)
            )
            rolling_deriv_theta2 = np.array(
                SHintegrand_theta2(self.chains_H_flat, theta1, theta2)
            )
            beta_deriv_Abar = np.array(grad_Abar(self.chains_flat, Abar, s))
            beta_deriv_s = np.array(grad_s(self.chains_flat, Abar, s))

            (
                Rialpha_deriv_theta1,
                Rialpha_deriv_theta2,
                Rialpha_deriv_Abar,
                Rialpha_deriv_s,
            ) = Rialphaderiv_rolling_lca_physical(
                alpha,
                self.ncomp,
                self.nobj,
                self.nphys,
                self.jacobian,
                self.S_obj_noamp_physical,
                self.chains_H,
                self.chains_LCA,
                self.splitArray(rolling_deriv_theta1),
                self.splitArray(rolling_deriv_theta2),
                self.splitArray(beta_deriv_Abar),
                self.splitArray(beta_deriv_s),
                theta1,
                theta2,
                Abar,
                s,
            )

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_theta1_sum = np.sum(Rialpha_deriv_theta1 * falpha, axis=1)
            Rialpha_theta2_sum = np.sum(Rialpha_deriv_theta2 * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_deriv_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_deriv_s * falpha, axis=1)

            gradRialpha_theta1 = np.sum(Rialpha_theta1_sum / self.falphaRalpha)
            gradRialpha_theta2 = np.sum(Rialpha_theta2_sum / self.falphaRalpha)
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params)

            SalphatimesH_comp_theta1 = np.zeros_like(self.SalphatimesH_theta1)
            SalphatimesH_comp_theta2 = np.zeros_like(self.SalphatimesH_theta2)

            for comp in range(self.ncomp):
                if comp % self.nphys == alpha:
                    SalphatimesH_comp_theta1[comp] = self.SalphatimesH_theta1[comp]
                    SalphatimesH_comp_theta2[comp] = self.SalphatimesH_theta2[comp]

            # now put things in place
            grad[4 * alpha] = (
                gradRialpha_theta1
                - self.nobj * np.dot(SalphatimesH_comp_theta1, falpha) / Salpha_sum
                + prior_deriv[4 * alpha]
            )
            grad[4 * alpha + 1] = (
                gradRialpha_theta2
                - self.nobj * np.dot(SalphatimesH_comp_theta2, falpha) / Salpha_sum
                + prior_deriv[4 * alpha + 1]
            )
            grad[4 * alpha + 2] = gradRialpha_Abar + prior_deriv[4 * alpha + 2]
            grad[4 * alpha + 3] = gradRialpha_s + prior_deriv[4 * alpha + 3]
        return self.pref * grad


class LogLikeEigenGMMDynamicsRolling(LogLikeGMMDynamicsRolling):
    def compute_gmm_chain(self):
        self.eigen = np.linalg.eig(self.gmm.cov_best[0])[1][:, 0]
        self.S_obj_noamp = []
        logdet = np.linalg.slogdet(self.gmm.cov_best)[1]

        for x in self.chains_color:
            g = (
                ga.log_multigaussian_no_off(x, self.gmm.mean_best, self.gmm.cov_best)
                - logdet / 2
                - np.log(2 * np.pi) * 3 / 2
            )
            decomp = eigendecompose(x, self.eigen, self.gmm.mean_best[0])
            g[decomp > 0, 0] = -np.inf
            g[decomp < 0, 0] += np.log(2)

            g[decomp < 0, 1] = -np.inf
            g[decomp > 0, 1] += np.log(2)

            self.S_obj_noamp.append(g)

        self.S_obj_noamp = numba.typed.List(self.S_obj_noamp)

class LogLikeEigenGMMRollingShared(LogLikeEigenGMMDynamicsRolling):
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
        self.eff_index = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7]
    def Rialpha(self, params):
        eff_params = params[self.eff_index]
        return super().Rialpha(eff_params)

    def Salpha(self, params, fun):
        eff_params = params[self.eff_index]
        return super().Salpha(eff_params, fun)
    
    def gradient_theta(self, falpha, params): #can also write this as super()! 
        ## Assumes we have already computed Rialpha and Salpha
        eff_params = params[self.eff_index]
        self.Rialpha_current = self.Rialpha(params)

        self.Salpha_current = self.Salpha(params, self.size_dist)

        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        # Salpha_current = self.Salpha(params, self.size_dist)
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand_theta1 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope1
        )
        SHintegrand_theta2 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope2
        )

        # different from other cases now
        self.SalphatimesH_theta1 = self.Salpha(eff_params, SHintegrand_theta1)
        self.SalphatimesH_theta2 = self.Salpha(eff_params, SHintegrand_theta2)

        # this is where it gets tricky - this is now 3nphys d
        grad = np.zeros_like(eff_params)

        for alpha in range(self.nphys):
            # unpack
            theta1 = eff_params[4 * alpha]
            theta2 = eff_params[4 * alpha + 1]
            Abar = eff_params[4 * alpha + 2]
            s = eff_params[4 * alpha + 3]

            # now calculate the Rialpha sums

            rolling_deriv_theta1 = np.array(
                SHintegrand_theta1(self.chains_H_flat, theta1, theta2)
            )
            rolling_deriv_theta2 = np.array(
                SHintegrand_theta2(self.chains_H_flat, theta1, theta2)
            )
            beta_deriv_Abar = np.array(grad_Abar(self.chains_flat, Abar, s))
            beta_deriv_s = np.array(grad_s(self.chains_flat, Abar, s))

            (
                Rialpha_deriv_theta1,
                Rialpha_deriv_theta2,
                Rialpha_deriv_Abar,
                Rialpha_deriv_s,
            ) = Rialphaderiv_rolling_lca_physical(
                alpha,
                self.ncomp,
                self.nobj,
                self.nphys,
                self.jacobian,
                self.S_obj_noamp_physical,
                self.chains_H,
                self.chains_LCA,
                self.splitArray(rolling_deriv_theta1),
                self.splitArray(rolling_deriv_theta2),
                self.splitArray(beta_deriv_Abar),
                self.splitArray(beta_deriv_s),
                theta1,
                theta2,
                Abar,
                s,
            )

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_theta1_sum = np.sum(Rialpha_deriv_theta1 * falpha, axis=1)
            Rialpha_theta2_sum = np.sum(Rialpha_deriv_theta2 * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_deriv_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_deriv_s * falpha, axis=1)

            gradRialpha_theta1 = np.sum(Rialpha_theta1_sum / self.falphaRalpha)
            gradRialpha_theta2 = np.sum(Rialpha_theta2_sum / self.falphaRalpha)
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(eff_params)

            SalphatimesH_comp_theta1 = np.zeros_like(self.SalphatimesH_theta1)
            SalphatimesH_comp_theta2 = np.zeros_like(self.SalphatimesH_theta2)

            for comp in range(self.ncomp):
                if comp % self.nphys == alpha:
                    SalphatimesH_comp_theta1[comp] = self.SalphatimesH_theta1[comp]
                    SalphatimesH_comp_theta2[comp] = self.SalphatimesH_theta2[comp]

            # now put things in place
            grad[4 * alpha] = (
                gradRialpha_theta1
                - self.nobj * np.dot(SalphatimesH_comp_theta1, falpha) / Salpha_sum
                + prior_deriv[4 * alpha]
            )
            grad[4 * alpha + 1] = (
                gradRialpha_theta2
                - self.nobj * np.dot(SalphatimesH_comp_theta2, falpha) / Salpha_sum
                + prior_deriv[4 * alpha + 1]
            )
            grad[4 * alpha + 2] = gradRialpha_Abar + prior_deriv[4 * alpha + 2]
            grad[4 * alpha + 3] = gradRialpha_s + prior_deriv[4 * alpha + 3]
        
        gg = grad[[0, 1, 2, 3, 8, 9, 10, 11]]
        gg[0] += grad[4]
        gg[1] += grad[5]
        gg[2] += grad[6]
        gg[3] += grad[7]
        return self.pref * gg


class LogLikeGMMDynamicsRollingSameSlope(LogLikeGMMDynamicsRolling):
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
            self.eff_index += [0, 1]
            self.eff_index += [2 + 2 * i, 2 + 2 * i + 1]

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

        SHintegrand_theta1 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope1
        )
        SHintegrand_theta2 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope2
        )

        # different from other cases now
        self.SalphatimesH_theta1 = self.Salpha(params, SHintegrand_theta1)
        self.SalphatimesH_theta2 = self.Salpha(params, SHintegrand_theta2)

        # this is where it gets tricky - this is now 3nphys d
        grad = np.zeros_like(params)

        theta1 = params[0]
        theta2 = params[1]
        rolling_deriv_theta1 = np.array(
            SHintegrand_theta1(self.chains_H_flat, theta1, theta2)
        )
        rolling_deriv_theta2 = np.array(
            SHintegrand_theta2(self.chains_H_flat, theta1, theta2)
        )

        for alpha in range(self.nphys):
            # unpack
            Abar = params[2 * alpha + 2]
            s = params[2 * alpha + 3]

            # now calculate the Rialpha sums

            beta_deriv_Abar = np.array(grad_Abar(self.chains_flat, Abar, s))
            beta_deriv_s = np.array(grad_s(self.chains_flat, Abar, s))

            (
                Rialpha_deriv_theta1,
                Rialpha_deriv_theta2,
                Rialpha_deriv_Abar,
                Rialpha_deriv_s,
            ) = Rialphaderiv_rolling_lca_physical(
                alpha,
                self.ncomp,
                self.nobj,
                self.nphys,
                self.jacobian,
                self.S_obj_noamp_physical,
                self.chains_H,
                self.chains_LCA,
                self.splitArray(rolling_deriv_theta1),
                self.splitArray(rolling_deriv_theta2),
                self.splitArray(beta_deriv_Abar),
                self.splitArray(beta_deriv_s),
                theta1,
                theta2,
                Abar,
                s,
            )

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_theta1_sum = np.sum(Rialpha_deriv_theta1 * falpha, axis=1)
            Rialpha_theta2_sum = np.sum(Rialpha_deriv_theta2 * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_deriv_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_deriv_s * falpha, axis=1)
                        

            gradRialpha_theta1 = np.sum(Rialpha_theta1_sum / self.falphaRalpha)
            gradRialpha_theta2 = np.sum(Rialpha_theta2_sum / self.falphaRalpha)
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params)

            SalphatimesH_comp_theta1 = np.zeros_like(self.SalphatimesH_theta1)
            SalphatimesH_comp_theta2 = np.zeros_like(self.SalphatimesH_theta2)

            for comp in range(self.ncomp):
                if comp % self.nphys == alpha:
                    SalphatimesH_comp_theta1[comp] = self.SalphatimesH_theta1[comp]
                    SalphatimesH_comp_theta2[comp] = self.SalphatimesH_theta2[comp]

            # now put things in place
            grad[0] += (
                gradRialpha_theta1
                - self.nobj * np.dot(SalphatimesH_comp_theta1, falpha) / Salpha_sum
                + prior_deriv[0]
            )
            grad[1] += (
                gradRialpha_theta2
                - self.nobj * np.dot(SalphatimesH_comp_theta2, falpha) / Salpha_sum
                + prior_deriv[1]
            )
            grad[2 * alpha + 2] = gradRialpha_Abar + prior_deriv[2 * alpha + 2]
            grad[2 * alpha + 3] = gradRialpha_s + prior_deriv[2 * alpha + 3]
        return self.pref * grad

class LogLikeGMMDynamicsRollingSameSlopeNoLCA(LogLikeGMMDynamicsRolling):
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
            self.eff_index += [0, 1]

    def Rialpha(self, params):
        return Rialpha_rolling_physical(
            self.ncomp,
            self.nobj,
            self.nphys,
            self.jacobian,
            self.S_obj_noamp_physical,
            self.chains_H,
            params[self.eff_index],
        )
    def gradient_theta(self, falpha, params):
        ## Assumes we have already computed Rialpha and Salpha
        self.Rialpha_current = self.Rialpha(params)

        self.Salpha_current = self.Salpha(params, self.size_dist)

        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        # Salpha_current = self.Salpha(params, self.size_dist)
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand_theta1 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope1
        )
        SHintegrand_theta2 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope2
        )

        # different from other cases now
        self.SalphatimesH_theta1 = self.Salpha(params, SHintegrand_theta1)
        self.SalphatimesH_theta2 = self.Salpha(params, SHintegrand_theta2)

        # this is where it gets tricky - this is now 3nphys d
        grad = np.zeros_like(params)

        theta1 = params[0]
        theta2 = params[1]
        rolling_deriv_theta1 = np.array(
            SHintegrand_theta1(self.chains_H_flat, theta1, theta2)
        )
        rolling_deriv_theta2 = np.array(
            SHintegrand_theta2(self.chains_H_flat, theta1, theta2)
        )

        for alpha in range(self.nphys):

            (
                Rialpha_deriv_theta1,
                Rialpha_deriv_theta2,
            ) = Rialphaderiv_rolling_physical(
                alpha,
                self.ncomp,
                self.nobj,
                self.nphys,
                self.jacobian,
                self.S_obj_noamp_physical,
                self.chains_H,
                self.splitArray(rolling_deriv_theta1),
                self.splitArray(rolling_deriv_theta2),
                theta1,
                theta2,
            )

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_theta1_sum = np.sum(Rialpha_deriv_theta1 * falpha, axis=1)
            Rialpha_theta2_sum = np.sum(Rialpha_deriv_theta2 * falpha, axis=1)
                        

            gradRialpha_theta1 = np.sum(Rialpha_theta1_sum / self.falphaRalpha)
            gradRialpha_theta2 = np.sum(Rialpha_theta2_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params)

            SalphatimesH_comp_theta1 = np.zeros_like(self.SalphatimesH_theta1)
            SalphatimesH_comp_theta2 = np.zeros_like(self.SalphatimesH_theta2)

            for comp in range(self.ncomp):
                if comp % self.nphys == alpha:
                    SalphatimesH_comp_theta1[comp] = self.SalphatimesH_theta1[comp]
                    SalphatimesH_comp_theta2[comp] = self.SalphatimesH_theta2[comp]

            # now put things in place
            grad[0] += (
                gradRialpha_theta1
                - self.nobj * np.dot(SalphatimesH_comp_theta1, falpha) / Salpha_sum
                + prior_deriv[0]
            )
            grad[1] += (
                gradRialpha_theta2
                - self.nobj * np.dot(SalphatimesH_comp_theta2, falpha) / Salpha_sum
                + prior_deriv[1]
            )
        return self.pref * grad

    def Salpha(self, params, fun):
        # Note that the integral of the LCA distribution times the selection function is constant (=1), so we don't need (Abar,s)_alpha in here
        # This is because p(s|A) ~ const for all A
        # Extra care is needed to get the parameters in the right place. We only need the slopes:
        theta1 = params[0]
        theta2 = params[1]
        f = []
        for alpha in range(self.nphys):
            f.append(fun(self.H_bins, theta1, theta2))

        S = np.zeros(self.ncomp)
        for alpha in range(self.ncomp):
            integrand = self.selection[alpha] * f[alpha % self.nphys]

            S[alpha] = (
                np.sum((integrand[1:] + integrand[:-1]) * np.diff(self.H_bins)) / 2
            )
        return S


class LogLikeRolling(LogLikeSingleSlope):
    def __init__(
        self, chains_H, selection, ncomp, pref=-1, prior=None, prior_deriv=None
    ):
        self.chains_H = numba.typed.List(chains_H)
        self.chains_H_flat = np.concatenate(chains_H)

        self.selection = selection
        self.nobj = len(chains_H)
        self.ncomp = ncomp
        self.nphys = 1

        self.rng = np.random.default_rng()

        # to avoid rewriting code, let me do something a bit silly
        # this avoids the need to rewrite the Rialpha declarations
        self.S_obj_noamp = []
        for x in self.chains_H:
            self.S_obj_noamp.append(np.zeros((len(x), 1)))
            self.S_obj_noamp = numba.typed.List(self.S_obj_noamp)

        self.pref = pref
        self.compute_jacobian_chain()

        self.H_bins = np.arange(5.05, 9.05, 0.1)

        if prior is None:
            self.prior = lambda x: 0
            self.prior_deriv = lambda x: np.zeros(2)
        else:
            self.prior = prior
            self.prior_deriv = prior_deriv
        indexes = [len(i) for i in chains_H]
        self.indexes = np.cumsum(indexes)[:-1]

    def Rialpha(self, params):
        return Rialpha_rolling_physical(
            self.ncomp,
            self.nobj,
            self.nphys,
            self.jacobian,
            self.S_obj_noamp,
            self.chains_H,
            params,
        )

    def Salpha(self, params, fun):
        # Note that the integral of the LCA distribution times the selection function is constant (=1), so we don't need (Abar,s)_alpha in here
        # This is because p(s|A) ~ const for all A
        # Extra care is needed to get the parameters in the right place. We only need the slopes:
        theta1 = params[[2* i for i in range(self.nphys)]]
        theta2 = params[[2 * i + 1 for i in range(self.nphys)]]
        f = []
        for alpha in range(self.nphys):
            f.append(fun(self.H_bins, theta1[alpha], theta2[alpha]))

        S = np.zeros(self.ncomp)
        for alpha in range(self.ncomp):
            integrand = self.selection[alpha] * f[alpha % self.nphys]

            S[alpha] = (
                np.sum((integrand[1:] + integrand[:-1]) * np.diff(self.H_bins)) / 2
            )
        return S

    def size_dist(self, H, theta1, theta2):
        norm = normalize_rolling(theta1, theta2)
        return rolling_slope(H, theta1, theta2) / norm

    def gradient_theta(self, params):
        falpha = np.array([1.0])
        ## Assumes we have already computed Rialpha and Salpha
        self.Rialpha_current = self.Rialpha(params)

        self.Salpha_current = self.Salpha(params, self.size_dist)

        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        # Salpha_current = self.Salpha(params, self.size_dist)
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand_theta1 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope1
        )
        SHintegrand_theta2 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope2
        )

        # different from other cases now
        self.SalphatimesH_theta1 = self.Salpha(params, SHintegrand_theta1)
        self.SalphatimesH_theta2 = self.Salpha(params, SHintegrand_theta2)

        # this is where it gets tricky - this is now 3nphys d
        grad = np.zeros_like(params)

        theta1 = params[0]
        theta2 = params[1]

        rolling_deriv_theta1 = np.array(
            SHintegrand_theta1(self.chains_H_flat, theta1, theta2)
        )
        rolling_deriv_theta2 = np.array(
            SHintegrand_theta2(self.chains_H_flat, theta1, theta2)
        )

        Rialpha_deriv_theta1, Rialpha_deriv_theta2 = Rialphaderiv_rolling_physical(
            0,
            self.ncomp,
            self.nobj,
            self.nphys,
            self.jacobian,
            self.S_obj_noamp,
            self.chains_H,
            self.splitArray(rolling_deriv_theta1),
            self.splitArray(rolling_deriv_theta2),
            theta1,
            theta2,
        )

        # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
        Rialpha_theta1_sum = np.sum(Rialpha_deriv_theta1 * falpha, axis=1)
        Rialpha_theta2_sum = np.sum(Rialpha_deriv_theta2 * falpha, axis=1)

        gradRialpha_theta1 = np.sum(Rialpha_theta1_sum / self.falphaRalpha)
        gradRialpha_theta2 = np.sum(Rialpha_theta2_sum / self.falphaRalpha)

        prior_deriv = self.prior_deriv(params)

        # now put things in place
        grad[0] = (
            gradRialpha_theta1
            - self.nobj * np.dot(self.SalphatimesH_theta1, falpha) / Salpha_sum
            + prior_deriv[0]
        )
        grad[1] = (
            gradRialpha_theta2
            - self.nobj * np.dot(self.SalphatimesH_theta1, falpha) / Salpha_sum
            + prior_deriv[1]
        )
        return self.pref * grad
    def splitArray(self, array):
        return numba.typed.List(np.split(array, self.indexes))
