from hmc_lca import *


@numba.njit(parallel=True, fastmath=True)
def Rialpha_singleslope_lca_physical(
    ncomp, nobj, nphys, jacob, pref, chain_H, chain_LCA, params
):
    Rialpha = np.zeros((nobj, ncomp))
    for alpha in range(nphys):
        slope = params[3 * alpha]
        Abar = params[3 * alpha + 1]
        s = params[3 * alpha + 2]
        norm = normalize_beta(Abar, s)
        # parallelize here
        for i in numba.prange(nobj):
            size_chain_comp = np.log(singleslope(chain_H[i], slope))
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


@numba.njit(parallel=True, fastmath=True)
def Rialphaderiv_singleslope_lca_physical(
    alpha,
    ncomp,
    nobj,
    nphys,
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
    Rialpha_slope = np.zeros((nobj, ncomp))
    Rialpha_Abar = np.zeros((nobj, ncomp))
    Rialpha_s = np.zeros((nobj, ncomp))

    norm = normalize_beta(Abar, s)
    for i in numba.prange(nobj):
        slope_deriv = singleslope_deriv(chain_H[i], slope)
        size_chain_comp = np.log(singleslope(chain_H[i], slope))
        lca_chain_comp = np.log(beta(chain_LCA[i], Abar, s, norm))
        for comp in range(ncomp):
            if comp % nphys == alpha:
                Rialpha_slope[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + lca_chain_comp) * slope_deriv
                ) / len(chain_H[i])
                Rialpha_Abar[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    * beta_deriv_Abar[i]
                ) / len(chain_H[i])
                Rialpha_s[i, comp] = np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    * beta_deriv_s[i]
                ) / len(chain_H[i])

    return Rialpha_slope, Rialpha_Abar, Rialpha_s


@numba.njit(parallel=True, fastmath=True)
def Rialpha_singleslope_physical(ncomp, nobj, nphys, jacob, pref, chain_H, params):
    Rialpha = np.zeros((nobj, ncomp))
    for alpha in range(nphys):
        slope = params[alpha]
        # parallelize here
        for i in numba.prange(nobj):
            size_chain_comp = np.log(singleslope(chain_H[i], slope))
            # and here?
            for comp in range(ncomp):
                if comp % nphys == alpha:
                    Rialpha[i, comp] = np.sum(
                        np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    ) / len(chain_H[i])

    return Rialpha


@numba.njit(parallel=True, fastmath=True)
def Rialphaderiv_singleslope_physical(
    alpha, ncomp, nobj, nphys, jacob, pref, chain_H, slope
):
    Rialpha = np.zeros((nobj, ncomp))
    for i in numba.prange(nobj):
        for alpha in range(nphys):
            size_chain_comp = singleslope_deriv(chain_H[i], slope)
            for comp in range(ncomp):
                if comp % nphys == alpha:
                    Rialpha[i, comp] = np.sum(
                        np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    ) / len(chain_H[i])

    return Rialpha


class LogLikeGMMLCADynamics(LogLikeGMMLightCurve):
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
            pref,
            prior,
            prior_deriv,
        )
        if nphys is None:
            self.nphys = ncomp
        else:
            self.nphys = nphys

        self.assign_classes(physical_assignment)

    def assign_classes(self, physical_assignment):
        self.S_obj_noamp_physical = []

        for i in range(self.nobj):
            obj = []
            z = np.zeros_like(self.S_obj_noamp[i][:, 0])
            z -= np.inf
            for alpha in range(self.ncomp):
                if alpha in physical_assignment[i]:
                    obj.append(self.S_obj_noamp[i][:, alpha % self.nphys])
                else:
                    obj.append(z)
            obj = np.vstack(obj)
            self.S_obj_noamp_physical.append(obj.T)

        self.S_obj_noamp_physical = numba.typed.List(self.S_obj_noamp_physical)

    def Rialpha(self, params):
        return Rialpha_singleslope_lca_physical(
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
        params = params[[3 * i for i in range(self.nphys)]]
        f = fun(params, self.H_bins)

        S = np.zeros(self.ncomp)
        for alpha in range(self.ncomp):
            integrand = self.selection[alpha] * f[:, alpha % self.nphys]

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

        # this is where it gets tricky - this is now 3nphys d
        grad = np.zeros_like(params)

        for alpha in range(self.nphys):
            # unpack
            slope = params[3 * alpha]
            Abar = params[3 * alpha + 1]
            s = params[3 * alpha + 2]

            # evaluate jax gradients at each parameter over the chains
            deriv_chain_Abar = np.array(grad_Abar(self.chains_flat, Abar, s))
            deriv_chain_s = np.array(grad_s(self.chains_flat, Abar, s))

            # now calculate the Rialpha sums
            (
                Rialpha_slope,
                Rialpha_Abar,
                Rialpha_s,
            ) = Rialphaderiv_singleslope_lca_physical(
                alpha,
                self.ncomp,
                self.nobj,
                self.nphys,
                self.jacobian,
                self.S_obj_noamp_physical,
                self.chains_H,
                self.chains_LCA,
                numba.typed.List(np.split(deriv_chain_Abar, self.indexes)),
                numba.typed.List(np.split(deriv_chain_s, self.indexes)),
                slope,
                Abar,
                s,
            )

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_slope_sum = np.sum(Rialpha_slope * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_s * falpha, axis=1)

            gradRialpha_slope = np.sum(Rialpha_slope_sum / self.falphaRalpha)
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

            SalphatimesH_comp = np.zeros_like(self.SalphatimesH)

            for comp in range(self.ncomp):
                if comp % self.nphys == alpha:
                    SalphatimesH_comp[comp] = self.SalphatimesH[comp]

            # now put things in place
            grad[3 * alpha] = (
                gradRialpha_slope
                - self.nobj * np.dot(SalphatimesH_comp, falpha) / Salpha_sum
                + prior_deriv[3 * alpha]
            )
            grad[3 * alpha + 1] = gradRialpha_Abar + prior_deriv[3 * alpha + 1]
            grad[3 * alpha + 2] = gradRialpha_s + prior_deriv[3 * alpha + 2]

        return self.pref * grad


class LogLikeGMMLCADynamicsEigen(LogLikeGMMLCADynamics):
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


class LogLikeGMMDynamics(LogLikeGMMSingleSlope):
    def __init__(
        self,
        gmm,
        chains_color,
        chains_H,
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
            selection,
            ncomp,
            pref,
            prior,
            prior_deriv,
        )
        if nphys is None:
            self.nphys = ncomp
        else:
            self.nphys = nphys

        self.assign_classes(physical_assignment)

    def assign_classes(self, physical_assignment):
        self.S_obj_noamp_physical = []

        for i in range(self.nobj):
            obj = []
            z = np.zeros_like(self.S_obj_noamp[i][:, 0])
            z -= np.inf
            for alpha in range(self.ncomp):
                if alpha in physical_assignment[i]:
                    obj.append(self.S_obj_noamp[i][:, alpha % self.nphys])
                else:
                    obj.append(z)
            obj = np.vstack(obj)
            self.S_obj_noamp_physical.append(obj.T)

        self.S_obj_noamp_physical = numba.typed.List(self.S_obj_noamp_physical)

    def Rialpha(self, params):
        return Rialpha_singleslope_physical(
            self.ncomp,
            self.nobj,
            self.nphys,
            self.jacobian,
            self.S_obj_noamp_physical,
            self.chains_H,
            params,
        )

    def Salpha(self, params, fun):
        f = fun(params, self.H_bins)
        S = np.zeros(self.ncomp)
        for alpha in range(self.ncomp):
            integrand = self.selection[alpha] * f[:, alpha % self.nphys]

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

        # this is where it gets tricky - this is now 3nphys d
        grad = np.zeros_like(params)

        for alpha in range(self.nphys):
            # unpack
            slope = params[alpha]

            # now calculate the Rialpha sums

            Rialpha_slope = Rialphaderiv_singleslope_lca_physical(
                alpha,
                self.ncomp,
                self.nobj,
                self.nphys,
                self.jacobian,
                self.S_obj_noamp_physical,
                self.chains_H,
                slope,
            )

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_slope_sum = np.sum(Rialpha_slope * falpha, axis=1)

            gradRialpha_slope = np.sum(Rialpha_slope_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

            SalphatimesH_comp = np.zeros_like(self.SalphatimesH)

            for comp in range(self.ncomp):
                if comp % self.nphys == alpha:
                    SalphatimesH_comp[comp] = self.SalphatimesH[comp]

            # now put things in place
            grad[alpha] = (
                gradRialpha_slope
                - self.nobj * np.dot(SalphatimesH_comp, falpha) / Salpha_sum
                + prior_deriv[alpha]
            )
        return self.pref * grad
