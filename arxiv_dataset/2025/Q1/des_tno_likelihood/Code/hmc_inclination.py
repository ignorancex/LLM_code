from hmc_physical import * 

@numba.njit(fastmath=True)
def vonmises(theta, coeff):
    kappa = np.exp(coeff)
    a = kappa/(1-np.exp(-2*kappa))
    sin_thetahalf = np.sin(theta/2)
    b = np.sin(theta) * np.exp(-2 * kappa * sin_thetahalf * sin_thetahalf)
    return a*b

@numba.njit(fastmath = True)
def vonmises_grad(theta, coeff):
    kappa = np.exp(coeff)
    sin_theta = np.sin(theta)
    sin_thetahalf = np.sin(theta/2)
    one_minus_exp2k = 1 - np.exp(-2 * kappa)
    exp2ksin = np.exp(-2 * kappa * sin_thetahalf * sin_thetahalf)
    
    pref = sin_theta/one_minus_exp2k
    
    first = - 2 * kappa * exp2ksin * sin_thetahalf * sin_thetahalf 
    second = - 2 * kappa * exp2ksin * np.exp(-2*kappa)/one_minus_exp2k
    third = exp2ksin 
    
    return kappa*(first + second + third) * pref 

@numba.njit(fastmath = True, parallel = True)
def singleslope_inc_multi(slopes, concentrations, H, inc):
    z = np.zeros((len(H), len(inc), len(slopes)))
    for i in numba.prange(len(H)):
        for j in numba.prange(len(inc)):
            for alpha in range(len(slopes)):
                z[i,j,alpha] = singleslope(H[i], slopes[alpha]) * vonmises(inc[j], concentrations[alpha])
    return z

@numba.njit(fastmath = True, parallel = True)
def singleslope_deriv_inc_multi(slopes, concentrations, H, inc):
    z = np.zeros((len(H), len(inc), len(slopes)))
    for i in numba.prange(len(H)):
        for j in numba.prange(len(inc)):
            for alpha in range(len(slopes)):
                z[i,j,alpha] = singleslope_deriv(H[i], slopes[alpha]) * vonmises(inc[j], concentrations[alpha])
    return z

@numba.njit(fastmath = True, parallel = True)
def singleslope_inc_deriv_multi(slopes, concentrations, H, inc):
    z = np.zeros((len(H), len(inc), len(slopes)))
    for i in numba.prange(len(H)):
        for j in numba.prange(len(inc)):
            for alpha in range(len(slopes)):
                z[i,j,alpha] = singleslope(H[i], slopes[alpha]) * vonmises_grad(inc[j], concentrations[alpha])
    return z


## now this is where it gets tricky. Rialpha stays the same, but each i,alpha now has 
## p(inc_i|kappa_alpha) in front. Note that there is no inclination chain (no sum over j)
## as we can assume the measurements are ~perfect

# these are easily vectorizable - is this way actually faster?
@numba.njit(fastmath = True, parallel = True)
def Rialpha_inc_physical(thetas, kappas, nobj, ncomp, nphys):
    vM = np.zeros((nobj, ncomp))
    for alpha in range(nphys):
        for i in numba.prange(nobj):
            vm_this = vonmises(thetas[i], kappas[alpha])
            for comp in range(ncomp):
                if comp % nphys == alpha:
                    vM[i,comp] = vm_this
    return vM


@numba.njit(fastmath = True, parallel = True)
def Rialpha_inc_individual(thetas, kappas, nobj, ncomp):
    vM = np.zeros((nobj, ncomp))
    for alpha in range(ncomp):
        for i in numba.prange(nobj):
            vM[i,alpha] = vonmises(thetas[i], kappas[alpha])
    return vM

@numba.njit(fastmath = True, parallel = True)
def Rialpha_inc_deriv(thetas, kappas, nobj, ncomp):
    vM = np.zeros((nobj, ncomp))
    for alpha in range(ncomp):
        for i in numba.prange(nobj):
            vM[i,alpha] = vonmises_grad(thetas[i], kappas[alpha])
    return vM


@numba.njit(parallel=True, fastmath=True)
def Rialphaderiv_singleslope_lca_inc_physical(
    alpha,
    ncomp,
    nobj,
    nphys,
    jacob,
    pref,
    chain_H,
    chain_LCA,
    inclinations,
    beta_deriv_Abar,
    beta_deriv_s,
    slope,
    Abar,
    s,
    concentration,
):
    Rialpha_slope = np.zeros((nobj, ncomp))
    Rialpha_Abar = np.zeros((nobj, ncomp))
    Rialpha_s = np.zeros((nobj, ncomp))
    Rialpha_inc = np.zeros((nobj, ncomp))

    norm = normalize_beta(Abar, s)
    for i in numba.prange(nobj):
        slope_deriv = singleslope_deriv(chain_H[i], slope)
        inc_deriv = vonmises_grad(inclinations[i], concentration)
        size_chain_comp = np.log(singleslope(chain_H[i], slope))
        lca_chain_comp = np.log(beta(chain_LCA[i], Abar, s, norm))
        inc_comp = vonmises(inclinations[i], concentration)
        
        for comp in range(ncomp):
            if comp % nphys == alpha:
                Rialpha_slope[i, comp] = inc_comp * np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + lca_chain_comp) * slope_deriv
                ) / len(chain_H[i])
                Rialpha_Abar[i, comp] = inc_comp * np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    * beta_deriv_Abar[i]
                ) / len(chain_H[i])
                Rialpha_s[i, comp] = inc_comp * np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    * beta_deriv_s[i]
                ) / len(chain_H[i])
                Rialpha_inc[i,comp] = inc_deriv *  np.sum(np.exp(jacob[i] + pref[i][:, alpha] + size_chain_comp + lca_chain_comp))/len(chain_H[i])
    return Rialpha_slope, Rialpha_Abar, Rialpha_s, Rialpha_inc


@numba.njit(parallel=True, fastmath=True)
def Rialphaderiv_singleslope_lca_inc_physical_percomponent(
    alpha,
    ncomp,
    nobj,
    nphys,
    jacob,
    pref,
    chain_H,
    chain_LCA,
    inclinations,
    beta_deriv_Abar,
    beta_deriv_s,
    slope,
    Abar,
    s,
    concentrations,
):
    Rialpha_slope = np.zeros((nobj, ncomp))
    Rialpha_Abar = np.zeros((nobj, ncomp))
    Rialpha_s = np.zeros((nobj, ncomp))
    Rialpha_inc = np.zeros((nobj, ncomp))

    norm = normalize_beta(Abar, s)
    for i in numba.prange(nobj):
        slope_deriv = singleslope_deriv(chain_H[i], slope)
        size_chain_comp = np.log(singleslope(chain_H[i], slope))
        lca_chain_comp = np.log(beta(chain_LCA[i], Abar, s, norm))
        
        for comp in range(ncomp):
            if comp % nphys == alpha:
                inc_deriv = vonmises_grad(inclinations[i], concentrations[comp])
                inc_comp = vonmises(inclinations[i], concentrations[comp])
                Rialpha_slope[i, comp] = inc_comp * np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + lca_chain_comp) * slope_deriv
                ) / len(chain_H[i])
                Rialpha_Abar[i, comp] = inc_comp * np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    * beta_deriv_Abar[i]
                ) / len(chain_H[i])
                Rialpha_s[i, comp] = inc_comp * np.sum(
                    np.exp(jacob[i] + pref[i][:, comp] + size_chain_comp)
                    * beta_deriv_s[i]
                ) / len(chain_H[i])
                Rialpha_inc[i,comp] = inc_deriv *  np.sum(np.exp(jacob[i] + pref[i][:, alpha] + size_chain_comp + lca_chain_comp))/len(chain_H[i])
    return Rialpha_slope, Rialpha_Abar, Rialpha_s, Rialpha_inc


class LogLikeGMMLCAIncDynamics(LogLikeGMMLCADynamics):
    def __init__(self,
        gmm,
        chains_color,
        chains_H,
        chains_LCA,
        inclinations,
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
        self.inclinations = np.pi*inclinations/180
        self.param_indices = [i for i in range(4*(nphys)) if ((i+1)%4 != 0 or i == 0)]
        
        self.H_bins = np.arange(3.1,10.1,0.2)
        self.inc_bins = np.arange(1.5, 91.5, 3) * np.pi/180
        
        self.dH = 0.2 
        self.dInc = 3* np.pi/180

        
    def Rialpha(self, params):
        params_inc = params[3::4]
        params_phot = params[self.param_indices] #constructed during init
        Rialpha_phot =  Rialpha_singleslope_lca_physical(self.ncomp, self.nobj, self.nphys, self.jacobian, self.S_obj_noamp_physical, self.chains_H, self.chains_LCA, params_phot)
        Rialpha_inc = Rialpha_inc_physical(self.inclinations, params_inc, self.nobj, self.ncomp, self.nphys)
        return Rialpha_phot * Rialpha_inc 

    def Salpha(self, params, fun):
        # note that selection is independent of LCA - so int dLCA p(s|LCA) * p(LCA) = 1
        params_inc = params[3::4]
        params_abs = params[0::4]
        
        f = fun(params_abs, params_inc, self.H_bins, self.inc_bins)

        S = np.zeros(self.ncomp)
        for alpha in range(self.ncomp):
            S[alpha] = (
                np.trapz(np.trapz(self.selection[alpha]*f[:,:,alpha % self.nphys], dx = self.dH, axis=0),dx = self.dInc, axis=0)
            )
        return S

    def size_dist(self, params_abs, params_inc, H, inc):
        return singleslope_inc_multi(params_abs, params_inc, H, inc)

    
    def gradient_theta(self, falpha, params):
        self.Rialpha_current = self.Rialpha(params)
        self.Salpha_current = self.Salpha(params, self.size_dist)
        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand_slopederiv = lambda params_abs, params_inc, H, inc: singleslope_deriv_inc_multi(params_abs, params_inc, H, inc)
        SHintegrand_concederiv = lambda params_abs, params_inc, H, inc: singleslope_inc_deriv_multi(params_abs, params_inc, H, inc)

        self.SalphatimesH_slope = self.Salpha(params, SHintegrand_slopederiv)
        self.SalphatimesH_conce = self.Salpha(params, SHintegrand_concederiv)
        
        # this is where it gets tricky - this is now 4nphys d
        grad = np.zeros_like(params)

        for alpha in range(self.nphys):
            # unpack
            slope = params[4 * alpha]
            Abar = params[4 * alpha + 1]
            s = params[4 * alpha + 2]
            concentration = params[4 * alpha + 3]
            
            # evaluate jax gradients at each parameter over the chains
            deriv_chain_Abar = np.array(grad_Abar(self.chains_flat, Abar, s))
            deriv_chain_s = np.array(grad_s(self.chains_flat, Abar, s))

            # now calculate the Rialpha sums
            (
                Rialpha_slope,
                Rialpha_Abar,
                Rialpha_s,
                Rialpha_conce
            ) = Rialphaderiv_singleslope_lca_inc_physical(
                alpha,
                self.ncomp,
                self.nobj,
                self.nphys,
                self.jacobian,
                self.S_obj_noamp_physical,
                self.chains_H,
                self.chains_LCA,
                self.inclinations,
                numba.typed.List(np.split(deriv_chain_Abar, self.indexes)),
                numba.typed.List(np.split(deriv_chain_s, self.indexes)),
                slope,
                Abar,
                s,
                concentration,
            )

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_slope_sum = np.sum(Rialpha_slope * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_s * falpha, axis=1)
            Rialpha_conce_sum = np.sum(Rialpha_conce * falpha, axis=1)


            gradRialpha_slope = np.sum(Rialpha_slope_sum / self.falphaRalpha)
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)
            gradRialpha_conce = np.sum(Rialpha_conce_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

            SalphatimesH_slope_comp = np.zeros_like(self.SalphatimesH_slope)
            SalphatimesH_conce_comp = np.zeros_like(self.SalphatimesH_conce)

            for comp in range(self.ncomp):
                if comp % self.nphys == alpha:
                    SalphatimesH_slope_comp[comp] = self.SalphatimesH_slope[comp]
                    SalphatimesH_conce_comp[comp] = self.SalphatimesH_conce[comp]


            # now put things in place
            grad[4 * alpha] = (
                gradRialpha_slope
                - self.nobj * np.dot(SalphatimesH_slope_comp, falpha) / Salpha_sum
                + prior_deriv[4 * alpha]
            )
            grad[4 * alpha + 1] = gradRialpha_Abar + prior_deriv[4 * alpha + 1]
            grad[4 * alpha + 2] = gradRialpha_s + prior_deriv[4 * alpha + 2]
            grad[4 * alpha + 3] = gradRialpha_conce - self.nobj * np.dot(SalphatimesH_conce_comp, falpha)/Salpha_sum + prior_deriv[4*alpha+3]
        return self.pref * grad


class LogLikeGMMLCAIncPerComponent(LogLikeGMMLCAIncDynamics):
    def __init__(self,
        gmm,
        chains_color,
        chains_H,
        chains_LCA,
        inclinations,
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
        inclinations,
        selection,
        ncomp,
        physical_assignment,
        pref,
        prior,
        prior_deriv,
        nphys,
        )
        self.inclinations = np.pi*inclinations/180
        self.param_indices = [i for i in range(3*(nphys))]
        
        self.H_bins = np.arange(3.1,10.1,0.2)
        self.inc_bins = np.arange(1.5, 91.5, 3) * np.pi/180
        
        self.dH = 0.2 
        self.dInc = 3 * np.pi/180

        
    def Rialpha(self, params):
        params_inc = params[3*self.nphys::]
        params_phot = params[self.param_indices] #constructed during init
        # save to avoid recomputing in derivative
        self.Rialpha_phot =  Rialpha_singleslope_lca_physical(self.ncomp, self.nobj, self.nphys, self.jacobian, self.S_obj_noamp_physical, self.chains_H, self.chains_LCA, params_phot)
        self.Rialpha_inc = Rialpha_inc_individual(self.inclinations, params_inc, self.nobj, self.ncomp)
        return self.Rialpha_phot * self.Rialpha_inc 

    def Salpha(self, params, fun):
        # note that selection is independent of LCA - so int dLCA p(s|LCA) * p(LCA) = 1
        params_inc = params[3*self.nphys::]
        params_phot = params[self.param_indices]
        params_H = params_phot[::3]
        params_abs = np.tile(params_H, int(self.ncomp/self.nphys))
                
        f = fun(params_abs, params_inc, self.H_bins, self.inc_bins)
        
        S = np.zeros(self.ncomp)
        for alpha in range(self.ncomp):
            S[alpha] = (
                np.trapz(np.trapz(self.selection[alpha]*f[:,:,alpha], dx = self.dH, axis=0),dx = self.dInc, axis=0)
            )
        return S

    def gradient_theta(self, falpha, params):
        self.Rialpha_current = self.Rialpha(params)
        self.Salpha_current = self.Salpha(params, self.size_dist)
        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand_slopederiv = lambda params_abs, params_inc, H, inc: singleslope_deriv_inc_multi(params_abs, params_inc, H, inc)
        SHintegrand_concederiv = lambda params_abs, params_inc, H, inc: singleslope_inc_deriv_multi(params_abs, params_inc, H, inc)

        self.SalphatimesH_slope = self.Salpha(params, SHintegrand_slopederiv)
        self.SalphatimesH_conce = self.Salpha(params, SHintegrand_concederiv)
        
        # this is where it gets tricky - this is now 3nphys + ncomp d
        grad = np.zeros_like(params)
        params_phot = params[self.param_indices]
        params_inc = params[3*self.nphys::]
        for alpha in range(self.nphys):
            # unpack
            slope = params_phot[3 * alpha]
            Abar = params_phot[3 * alpha + 1]
            s = params_phot[3 * alpha + 2]
            
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

            #now, note that Rialpha for the inclination is constant here - so just take product
            Rialpha_slope = Rialpha_slope * self.Rialpha_inc
            Rialpha_Abar = Rialpha_Abar * self.Rialpha_inc
            Rialpha_s = Rialpha_s * self.Rialpha_inc 

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_slope_sum = np.sum(Rialpha_slope * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_s * falpha, axis=1)


            gradRialpha_slope = np.sum(Rialpha_slope_sum / self.falphaRalpha)
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

            SalphatimesH_slope_comp = np.zeros_like(self.SalphatimesH_slope)

            for comp in range(self.ncomp):
                if comp % self.nphys == alpha:
                    SalphatimesH_slope_comp[comp] = self.SalphatimesH_slope[comp]


            # now put things in place
            grad[3 * alpha] = (
                gradRialpha_slope
                - self.nobj * np.dot(SalphatimesH_slope_comp, falpha) / Salpha_sum
                + prior_deriv[3 * alpha]
            )
            grad[3 * alpha + 1] = gradRialpha_Abar + prior_deriv[3 * alpha + 1]
            grad[3 * alpha + 2] = gradRialpha_s + prior_deriv[3 * alpha + 2]
        
        # now proceed to compute concentration gradient
        # note that Rialpha_phot does not change - so we can re-use computation from early on
        Rialpha_inc = Rialpha_inc_deriv(self.inclinations, params_inc, self.nobj, self.ncomp) * self.Rialpha_phot 
        
        for alpha in range(self.ncomp):
            Rialphaderiv_comp = Rialpha_inc[:,alpha] 
            Salpha_comp = self.SalphatimesH_conce[alpha]

            gradRialpha = np.sum(Rialphaderiv_comp/self.falphaRalpha)
            prior_deriv = self.prior_deriv(params_inc[alpha])[3*self.nphys + alpha]
            

            grad[3*self.nphys + alpha] = gradRialpha*falpha[alpha] - self.nobj*falpha[alpha]*Salpha_comp/Salpha_sum + prior_deriv

        return self.pref * grad


class LogLikeGMMLCAIncSingle(LogLikeGMMLCAIncPerComponent):
    def __init__(self,
        gmm,
        chains_color,
        chains_H,
        chains_LCA,
        inclinations,
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
        inclinations,
        selection,
        ncomp,
        physical_assignment,
        pref,
        prior,
        prior_deriv,
        nphys,
        )
        
        self.ndist = int(self.ncomp/self.nphys)

    def Rialpha(self, params):
        params_inc = params[3*self.nphys::]
        params_inc = np.repeat(params_inc, self.nphys)
        params_phot = params[self.param_indices] #constructed during init
        # save to avoid recomputing in derivative
        self.Rialpha_phot =  Rialpha_singleslope_lca_physical(self.ncomp, self.nobj, self.nphys, self.jacobian, self.S_obj_noamp_physical, self.chains_H, self.chains_LCA, params_phot)
        self.Rialpha_inc = Rialpha_inc_individual(self.inclinations, params_inc, self.nobj, self.ncomp)
        return self.Rialpha_phot * self.Rialpha_inc 

    def Salpha(self, params, fun):
        # note that selection is independent of LCA - so int dLCA p(s|LCA) * p(LCA) = 1
        params_inc = params[3*self.nphys::]
        params_inc = np.repeat(params_inc, self.nphys)
        params_phot = params[self.param_indices]
        params_H = params_phot[::3]
        params_abs = np.tile(params_H, int(self.ncomp/self.nphys))
                
        f = fun(params_abs, params_inc, self.H_bins, self.inc_bins)
        
        S = np.zeros(self.ncomp)
        for alpha in range(self.ncomp):
            S[alpha] = (
                np.trapz(np.trapz(self.selection[alpha]*f[:,:,alpha], dx = self.dH, axis=0),dx = self.dInc, axis=0)
            )
        return S

    def gradient_theta(self, falpha, params):
        self.Rialpha_current = self.Rialpha(params)
        self.Salpha_current = self.Salpha(params, self.size_dist)
        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand_slopederiv = lambda params_abs, params_inc, H, inc: singleslope_deriv_inc_multi(params_abs, params_inc, H, inc)
        SHintegrand_concederiv = lambda params_abs, params_inc, H, inc: singleslope_inc_deriv_multi(params_abs, params_inc, H, inc)

        self.SalphatimesH_slope = self.Salpha(params, SHintegrand_slopederiv)
        self.SalphatimesH_conce = self.Salpha(params, SHintegrand_concederiv)
        
        # this is where it gets tricky - this is now 3nphys + ncomp d
        grad = np.zeros_like(params)
        params_phot = params[self.param_indices]
        params_inc = params[3*self.nphys::]
        params_inc = np.repeat(params_inc, self.nphys)

        for alpha in range(self.nphys):
            # unpack
            slope = params_phot[3 * alpha]
            Abar = params_phot[3 * alpha + 1]
            s = params_phot[3 * alpha + 2]
            
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

            #now, note that Rialpha for the inclination is constant here - so just take product
            Rialpha_slope = Rialpha_slope * self.Rialpha_inc
            Rialpha_Abar = Rialpha_Abar * self.Rialpha_inc
            Rialpha_s = Rialpha_s * self.Rialpha_inc 

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_slope_sum = np.sum(Rialpha_slope * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_s * falpha, axis=1)


            gradRialpha_slope = np.sum(Rialpha_slope_sum / self.falphaRalpha)
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

            SalphatimesH_slope_comp = np.zeros_like(self.SalphatimesH_slope)

            for comp in range(self.ncomp):
                if comp % self.nphys == alpha:
                    SalphatimesH_slope_comp[comp] = self.SalphatimesH_slope[comp]


            # now put things in place
            grad[3 * alpha] = (
                gradRialpha_slope
                - self.nobj * np.dot(SalphatimesH_slope_comp, falpha) / Salpha_sum
                + prior_deriv[3 * alpha]
            )
            grad[3 * alpha + 1] = gradRialpha_Abar + prior_deriv[3 * alpha + 1]
            grad[3 * alpha + 2] = gradRialpha_s + prior_deriv[3 * alpha + 2]
        
        # now proceed to compute concentration gradient
        # note that Rialpha_phot does not change - so we can re-use computation from early on
        Rialpha_inc = Rialpha_inc_deriv(self.inclinations, params_inc, self.nobj, self.ncomp) * self.Rialpha_phot 
        #params_inc = params[3*self.nphys::]

        for alpha in range(self.ncomp):
            Rialphaderiv_comp = Rialpha_inc[:,alpha] 
            Salpha_comp = self.SalphatimesH_conce[alpha]

            gradRialpha = np.sum(Rialphaderiv_comp/self.falphaRalpha)
            prior_deriv = self.prior_deriv(params_inc[alpha])[3*self.nphys + alpha]
            
            entry = alpha // self.nphys
            grad[3*self.nphys + entry] += gradRialpha*falpha[alpha] - self.nobj*falpha[alpha]*Salpha_comp/Salpha_sum + prior_deriv
        return self.pref * grad
    
    
class LogLikeGMMHalfInclination(LogLikeGMMLCAIncPerComponent):
    def compute_gmm_chain(self):
        self.eigen = np.linalg.eig(self.gmm.cov_best[0])[1][:,0]
        self.S_obj_noamp = []
        logdet = np.linalg.slogdet(self.gmm.cov_best)[1] 

        for x in self.chains_color:
            g = ga.log_multigaussian_no_off(x, self.gmm.mean_best, self.gmm.cov_best) - logdet/2 - np.log(2*np.pi)*3/2
            decomp = eigendecompose(x, self.eigen, self.gmm.mean_best[0]) 
            g[decomp > 0,0] = -np.inf
            g[decomp < 0,0] += np.log(2)

            g[decomp < 0,1] = -np.inf
            g[decomp > 0,1] += np.log(2)
            
            self.S_obj_noamp.append(g)

        self.S_obj_noamp = numba.typed.List(self.S_obj_noamp)
