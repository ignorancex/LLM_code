from hmc_inclination import * 
from hmc_rolling_numerical import * 
from hmc_tapered import * 

@numba.njit(fastmath = True, parallel = True)
def rolling_inc_multi(slopes, rolls, concentrations, H, inc):
    z = np.zeros((len(H), len(inc), len(slopes)))
    for alpha in range(len(slopes)):
        norm = normalize_rolling(slopes[alpha], rolls[alpha])
        for i in numba.prange(len(H)):
            for j in numba.prange(len(inc)):
                z[i,j,alpha] = rolling_slope(H[i], slopes[alpha], rolls[alpha]) * vonmises(inc[j], concentrations[alpha])/norm
    return z

@numba.njit(fastmath = True, parallel = True)
def tapered_inc_multi(a, b, H_crit, concentrations, H, inc):
    z = np.zeros((len(H), len(inc), len(a)))
    for alpha in range(len(a)):
        for i in numba.prange(len(H)):
            for j in numba.prange(len(inc)):
                z[i,j,alpha] = tapered_slope(H[i], a[alpha], b[alpha], H_crit[alpha]) * vonmises(inc[j], concentrations[alpha])
    return z



@numba.njit(fastmath = True, parallel = True)
def rolling_inc_deriv_multi(params_abs, concentrations, H, inc):
    slopes = params_abs[0::4]
    rolls = params_abs[1::4]
 
    z = np.zeros((len(H), len(inc), len(slopes)))
    for alpha in range(len(slopes)):
        norm = normalize_rolling(slopes[alpha], rolls[alpha])
        for i in numba.prange(len(H)):
            for j in numba.prange(len(inc)):
                z[i,j,alpha] = rolling_slope(H[i], slopes[alpha], rolls[alpha]) * vonmises_grad(inc[j], concentrations[alpha])/norm
    return z

def del_roll_inclination(params_abs, concentrations, H, inc, fun):
    slopes = params_abs[0::4]
    rolls = params_abs[1::4]
    
    z = np.zeros((len(H), len(inc), len(slopes)))
    
    for alpha in range(len(slopes)):
        ff = fun(H, slopes[alpha], rolls[alpha])
        cc = vonmises(inc, concentrations[alpha])
        z[:,:,alpha] = np.outer(ff, cc)
    return z
    

class LogLikeGMMLCAIncRolling(LogLikeGMMLCAIncPerComponent):
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
        self.param_indices = [i for i in range(4*(nphys))]
        self.inc_params = [4*self.nphys + i for i in range(self.ncomp)]
        
        self.H_bins = np.arange(3.1,10.1,0.2)
        self.inc_bins = np.arange(1.5, 91.5, 3) * np.pi/180
        
        self.dH = 0.2 
        self.dInc = 3 * np.pi/180
        self.chains_H_flat = np.concatenate(chains_H)
        indexes = [len(i) for i in chains_H]
        self.indexes = np.cumsum(indexes)[:-1]

    def size_dist(self, params_abs, params_inc, H, inc):
        theta = params_abs[0::4]
        thetaprime = params_abs[1::4]
        return rolling_inc_multi(theta, thetaprime, params_inc, H, inc)
        
    def Rialpha(self, params):
        params_inc = params[self.inc_params]
        params_phot = params[self.param_indices] #constructed during init
        # save to avoid recomputing in derivative
        self.Rialpha_phot =  Rialpha_rolling_lca_physical(self.ncomp, self.nobj, self.nphys, self.jacobian, self.S_obj_noamp_physical, self.chains_H, self.chains_LCA, params_phot)
        self.Rialpha_inc = Rialpha_inc_individual(self.inclinations, params_inc, self.nobj, self.ncomp)
        return self.Rialpha_phot * self.Rialpha_inc 

    def Salpha(self, params, fun):
        # note that selection is independent of LCA - so int dLCA p(s|LCA) * p(LCA) = 1
        params_inc = params[self.inc_params]
        params_phot = params[self.param_indices]
        params_abs = np.tile(params_phot, int(self.ncomp/self.nphys))
                
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

        SHintegrand_theta1 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope1
        )
        
        SHintegrand_theta1_conce = lambda params_abs, params_inc, H, inc : del_roll_inclination(params_abs, params_inc, H, inc, SHintegrand_theta1)
        
        SHintegrand_theta2 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope2
        )
        SHintegrand_theta2_conce = lambda params_abs, params_inc, H, inc : del_roll_inclination(params_abs, params_inc, H, inc, SHintegrand_theta2)

        # different from other cases now
        self.SalphatimesH_theta1 = self.Salpha(params, SHintegrand_theta1_conce)
        self.SalphatimesH_theta2 = self.Salpha(params, SHintegrand_theta2_conce)

        SHintegrand_concederiv = lambda params_abs, params_inc, H, inc: rolling_inc_deriv_multi(params_abs, params_inc, H, inc)

        self.SalphatimesH_conce = self.Salpha(params, SHintegrand_concederiv)
        
        # this is where it gets tricky - this is now 3nphys + ncomp d
        grad = np.zeros_like(params)
        params_phot = params[self.param_indices]
        params_inc = params[self.inc_params]
        for alpha in range(self.nphys):
            # unpack
            theta1 = params_phot[4 * alpha]
            theta2 = params_phot[4*alpha+1]
            Abar = params_phot[4 * alpha + 2]
            s = params_phot[4 * alpha + 3]
            
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

            #now, note that Rialpha for the inclination is constant here - so just take product
            Rialpha_theta1 = Rialpha_deriv_theta1 * self.Rialpha_inc
            Rialpha_theta2 = Rialpha_deriv_theta2 * self.Rialpha_inc
            Rialpha_Abar = Rialpha_deriv_Abar * self.Rialpha_inc
            Rialpha_s = Rialpha_deriv_s * self.Rialpha_inc 

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_theta1_sum = np.sum(Rialpha_theta1 * falpha, axis=1)
            Rialpha_theta2_sum = np.sum(Rialpha_theta2 * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_s * falpha, axis=1)


            gradRialpha_theta1 = np.sum(Rialpha_theta1_sum / self.falphaRalpha)
            gradRialpha_theta2 = np.sum(Rialpha_theta2_sum / self.falphaRalpha)
        
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

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
        
        # now proceed to compute concentration gradient
        # note that Rialpha_phot does not change - so we can re-use computation from early on
        Rialpha_inc = Rialpha_inc_deriv(self.inclinations, params_inc, self.nobj, self.ncomp) * self.Rialpha_phot 
        
        for alpha in range(self.ncomp):
            Rialphaderiv_comp = Rialpha_inc[:,alpha] 
            Salpha_comp = self.SalphatimesH_conce[alpha]

            gradRialpha = np.sum(Rialphaderiv_comp/self.falphaRalpha)
            prior_deriv = self.prior_deriv(params_inc[alpha])[4*self.nphys + alpha]
            

            grad[4*self.nphys + alpha] = gradRialpha*falpha[alpha] - self.nobj*falpha[alpha]*Salpha_comp/Salpha_sum + prior_deriv

        return self.pref * grad


class LogLikeGMMLCAIncRollingSingle(LogLikeGMMLCAIncRolling):
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
        params_inc = [4*self.nphys + i for i in range(self.ndist)]
        self.inc_params = np.repeat(params_inc, self.nphys)
        
        
    def gradient_theta(self, falpha, params):
        self.Rialpha_current = self.Rialpha(params)
        self.Salpha_current = self.Salpha(params, self.size_dist)
        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand_theta1 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope1
        )
        
        SHintegrand_theta1_conce = lambda params_abs, params_inc, H, inc : del_roll_inclination(params_abs, params_inc, H, inc, SHintegrand_theta1)
        
        SHintegrand_theta2 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope2
        )
        SHintegrand_theta2_conce = lambda params_abs, params_inc, H, inc : del_roll_inclination(params_abs, params_inc, H, inc, SHintegrand_theta2)

        # different from other cases now
        self.SalphatimesH_theta1 = self.Salpha(params, SHintegrand_theta1_conce)
        self.SalphatimesH_theta2 = self.Salpha(params, SHintegrand_theta2_conce)

        SHintegrand_concederiv = lambda params_abs, params_inc, H, inc: rolling_inc_deriv_multi(params_abs, params_inc, H, inc)

        self.SalphatimesH_conce = self.Salpha(params, SHintegrand_concederiv)
        
        # this is where it gets tricky - this is now 3nphys + ncomp d
        grad = np.zeros_like(params)
        params_phot = params[self.param_indices]
        params_inc = params[self.inc_params]
        for alpha in range(self.nphys):
            # unpack
            theta1 = params_phot[4 * alpha]
            theta2 = params_phot[4*alpha+1]
            Abar = params_phot[4 * alpha + 2]
            s = params_phot[4 * alpha + 3]
            
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

            #now, note that Rialpha for the inclination is constant here - so just take product
            Rialpha_theta1 = Rialpha_deriv_theta1 * self.Rialpha_inc
            Rialpha_theta2 = Rialpha_deriv_theta2 * self.Rialpha_inc
            Rialpha_Abar = Rialpha_deriv_Abar * self.Rialpha_inc
            Rialpha_s = Rialpha_deriv_s * self.Rialpha_inc 

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_theta1_sum = np.sum(Rialpha_theta1 * falpha, axis=1)
            Rialpha_theta2_sum = np.sum(Rialpha_theta2 * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_s * falpha, axis=1)


            gradRialpha_theta1 = np.sum(Rialpha_theta1_sum / self.falphaRalpha)
            gradRialpha_theta2 = np.sum(Rialpha_theta2_sum / self.falphaRalpha)
        
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

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
        
        # now proceed to compute concentration gradient
        # note that Rialpha_phot does not change - so we can re-use computation from early on
        Rialpha_inc = Rialpha_inc_deriv(self.inclinations, params_inc, self.nobj, self.ncomp) * self.Rialpha_phot 
        
        for alpha in range(self.ncomp):
            Rialphaderiv_comp = Rialpha_inc[:,alpha] 
            Salpha_comp = self.SalphatimesH_conce[alpha]

            gradRialpha = np.sum(Rialphaderiv_comp/self.falphaRalpha)
            prior_deriv = self.prior_deriv(params_inc[alpha])[4*self.nphys + alpha]
            
            entry = alpha // self.nphys
            grad[4*self.nphys + entry] = gradRialpha*falpha[alpha] - self.nobj*falpha[alpha]*Salpha_comp/Salpha_sum + prior_deriv

        return self.pref * grad            



class LogLikeGMMLCAIncRollingEigenShared(LogLikeGMMLCAIncRolling):
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
        nphys=None,):
        super().__init__(gmm,
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
        nphys,)
        
        self.eff_index = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7] + [8+i for i in range(self.ncomp)]
        self.eff_grad_index = [0, 1, 2, 3, 8, 9, 10, 11] + [12+i for i in range(self.ncomp)]
        self.param_indices = [i for i in range(4*(nphys))]
        self.inc_params = [4*self.nphys + i for i in range(self.ncomp)]
        
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
        
    def Rialpha(self, params):
        eff_params = params[self.eff_index]
        return super().Rialpha(eff_params)

    def Salpha(self, params, fun):
        eff_params = params[self.eff_index]
        return super().Salpha(eff_params, fun)
    
    def gradient_theta(self, falpha, params):
        self.Rialpha_current = self.Rialpha(params)
        self.Salpha_current = self.Salpha(params, self.size_dist)
        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand_theta1 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope1
        )
        
        SHintegrand_theta1_conce = lambda params_abs, params_inc, H, inc : del_roll_inclination(params_abs, params_inc, H, inc, SHintegrand_theta1)
        
        SHintegrand_theta2 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope2
        )
        SHintegrand_theta2_conce = lambda params_abs, params_inc, H, inc : del_roll_inclination(params_abs, params_inc, H, inc, SHintegrand_theta2)

        # different from other cases now
        self.SalphatimesH_theta1 = self.Salpha(params, SHintegrand_theta1_conce)
        self.SalphatimesH_theta2 = self.Salpha(params, SHintegrand_theta2_conce)

        SHintegrand_concederiv = lambda params_abs, params_inc, H, inc: rolling_inc_deriv_multi(params_abs, params_inc, H, inc)

        self.SalphatimesH_conce = self.Salpha(params, SHintegrand_concederiv)
        
        # this is where it gets tricky - this is now 3nphys + ncomp d
        # this is all in effective parameter space - then we re-join.
        eff_params = params[self.eff_index]
        grad = np.zeros_like(eff_params)

        params_phot = eff_params[self.param_indices]
        params_inc = eff_params[self.inc_params]
        for alpha in range(self.nphys):
            # unpack
            theta1 = params_phot[4 * alpha]
            theta2 = params_phot[4*alpha+1]
            Abar = params_phot[4 * alpha + 2]
            s = params_phot[4 * alpha + 3]
            
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

            #now, note that Rialpha for the inclination is constant here - so just take product
            Rialpha_theta1 = Rialpha_deriv_theta1 * self.Rialpha_inc
            Rialpha_theta2 = Rialpha_deriv_theta2 * self.Rialpha_inc
            Rialpha_Abar = Rialpha_deriv_Abar * self.Rialpha_inc
            Rialpha_s = Rialpha_deriv_s * self.Rialpha_inc 

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_theta1_sum = np.sum(Rialpha_theta1 * falpha, axis=1)
            Rialpha_theta2_sum = np.sum(Rialpha_theta2 * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_s * falpha, axis=1)


            gradRialpha_theta1 = np.sum(Rialpha_theta1_sum / self.falphaRalpha)
            gradRialpha_theta2 = np.sum(Rialpha_theta2_sum / self.falphaRalpha)
        
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

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
        
        # now proceed to compute concentration gradient
        # note that Rialpha_phot does not change - so we can re-use computation from early on
        Rialpha_inc = Rialpha_inc_deriv(self.inclinations, params_inc, self.nobj, self.ncomp) * self.Rialpha_phot 
        
        for alpha in range(self.ncomp):
            Rialphaderiv_comp = Rialpha_inc[:,alpha] 
            Salpha_comp = self.SalphatimesH_conce[alpha]

            gradRialpha = np.sum(Rialphaderiv_comp/self.falphaRalpha)
            prior_deriv = self.prior_deriv(params_inc[alpha])[4*self.nphys + alpha]
            

            grad[4*self.nphys + alpha] = gradRialpha*falpha[alpha] - self.nobj*falpha[alpha]*Salpha_comp/Salpha_sum + prior_deriv

             
                
        g = grad[self.eff_grad_index]
        g[0] += grad[4] 
        g[1] += grad[5]
        g[2] += grad[6] 
        g[3] += grad[7]
                
        return g*self.pref
        
        
class LogLikeGMMLCAIncRollingShared(LogLikeGMMLCAIncRolling):
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
        nphys=None,):
        super().__init__(gmm,
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
        nphys,)
        
        self.eff_index = [0, 1, 2, 3, 0, 1, 4, 5] + [6+i for i in range(self.ncomp)]
        self.eff_grad_index = [0, 1, 2, 3, 6, 7] + [8+i for i in range(self.ncomp)]
        self.param_indices = [i for i in range(4*(nphys))]
        self.inc_params = [4*self.nphys + i for i in range(self.ncomp)]
                
    def Rialpha(self, params):
        eff_params = params[self.eff_index]
        return super().Rialpha(eff_params)

    def Salpha(self, params, fun):
        eff_params = params[self.eff_index]
        return super().Salpha(eff_params, fun)
    
    def gradient_theta(self, falpha, params):
        self.Rialpha_current = self.Rialpha(params)
        self.Salpha_current = self.Salpha(params, self.size_dist)
        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand_theta1 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope1
        )
        
        SHintegrand_theta1_conce = lambda params_abs, params_inc, H, inc : del_roll_inclination(params_abs, params_inc, H, inc, SHintegrand_theta1)
        
        SHintegrand_theta2 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope2
        )
        SHintegrand_theta2_conce = lambda params_abs, params_inc, H, inc : del_roll_inclination(params_abs, params_inc, H, inc, SHintegrand_theta2)

        # different from other cases now
        self.SalphatimesH_theta1 = self.Salpha(params, SHintegrand_theta1_conce)
        self.SalphatimesH_theta2 = self.Salpha(params, SHintegrand_theta2_conce)

        SHintegrand_concederiv = lambda params_abs, params_inc, H, inc: rolling_inc_deriv_multi(params_abs, params_inc, H, inc)

        self.SalphatimesH_conce = self.Salpha(params, SHintegrand_concederiv)
        
        # this is where it gets tricky - this is now 3nphys + ncomp d
        # this is all in effective parameter space - then we re-join.
        eff_params = params[self.eff_index]
        grad = np.zeros_like(eff_params)

        params_phot = eff_params[self.param_indices]
        params_inc = eff_params[self.inc_params]
        for alpha in range(self.nphys):
            # unpack
            theta1 = params_phot[4 * alpha]
            theta2 = params_phot[4*alpha+1]
            Abar = params_phot[4 * alpha + 2]
            s = params_phot[4 * alpha + 3]
            
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

            #now, note that Rialpha for the inclination is constant here - so just take product
            Rialpha_theta1 = Rialpha_deriv_theta1 * self.Rialpha_inc
            Rialpha_theta2 = Rialpha_deriv_theta2 * self.Rialpha_inc
            Rialpha_Abar = Rialpha_deriv_Abar * self.Rialpha_inc
            Rialpha_s = Rialpha_deriv_s * self.Rialpha_inc 

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_theta1_sum = np.sum(Rialpha_theta1 * falpha, axis=1)
            Rialpha_theta2_sum = np.sum(Rialpha_theta2 * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_s * falpha, axis=1)


            gradRialpha_theta1 = np.sum(Rialpha_theta1_sum / self.falphaRalpha)
            gradRialpha_theta2 = np.sum(Rialpha_theta2_sum / self.falphaRalpha)
        
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

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
        
        # now proceed to compute concentration gradient
        # note that Rialpha_phot does not change - so we can re-use computation from early on
        Rialpha_inc = Rialpha_inc_deriv(self.inclinations, params_inc, self.nobj, self.ncomp) * self.Rialpha_phot 
        
        for alpha in range(self.ncomp):
            Rialphaderiv_comp = Rialpha_inc[:,alpha] 
            Salpha_comp = self.SalphatimesH_conce[alpha]

            gradRialpha = np.sum(Rialphaderiv_comp/self.falphaRalpha)
            prior_deriv = self.prior_deriv(params_inc[alpha])[4*self.nphys + alpha]
            

            grad[4*self.nphys + alpha] = gradRialpha*falpha[alpha] - self.nobj*falpha[alpha]*Salpha_comp/Salpha_sum + prior_deriv

             
                
        g = grad[self.eff_grad_index]
        g[0] += grad[4] 
        g[1] += grad[5]
                
        return g*self.pref

class LogLikeGMMLCAIncRollingSharedSingle(LogLikeGMMLCAIncRollingShared):
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
        nphys=None,):
        super().__init__(gmm,
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
        nphys,)
            
        self.ndist = int(self.ncomp/self.nphys)
        self.eff_index = [0, 1, 2, 3, 0, 1, 4, 5] + [6+i for i in range(self.ndist)]
        self.eff_grad_index = [0, 1, 2, 3, 6, 7] + [8+i for i in range(self.ndist)]
        self.param_indices = [i for i in range(4*(nphys))]
        params_inc = [4*self.nphys + i for i in range(self.ndist)]
        self.inc_params = np.repeat(params_inc, self.nphys)

    def gradient_theta(self, falpha, params):
        self.Rialpha_current = self.Rialpha(params)
        self.Salpha_current = self.Salpha(params, self.size_dist)
        self.falphaRalpha = np.sum(self.Rialpha_current * falpha, axis=1)
        
        Salpha_sum = np.dot(falpha, self.Salpha_current)

        SHintegrand_theta1 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope1
        )
        
        SHintegrand_theta1_conce = lambda params_abs, params_inc, H, inc : del_roll_inclination(params_abs, params_inc, H, inc, SHintegrand_theta1)
        
        SHintegrand_theta2 = lambda H, theta1, theta2: del_double_del_x(
            H, theta1, theta2, grad_slope2
        )
        SHintegrand_theta2_conce = lambda params_abs, params_inc, H, inc : del_roll_inclination(params_abs, params_inc, H, inc, SHintegrand_theta2)

        # different from other cases now
        self.SalphatimesH_theta1 = self.Salpha(params, SHintegrand_theta1_conce)
        self.SalphatimesH_theta2 = self.Salpha(params, SHintegrand_theta2_conce)

        SHintegrand_concederiv = lambda params_abs, params_inc, H, inc: rolling_inc_deriv_multi(params_abs, params_inc, H, inc)

        self.SalphatimesH_conce = self.Salpha(params, SHintegrand_concederiv)
        
        # this is where it gets tricky - this is now 3nphys + ncomp d
        # this is all in effective parameter space - then we re-join.
        eff_params = params[self.eff_index]
        grad = np.zeros_like(eff_params)

        params_phot = eff_params[self.param_indices]
        params_inc = eff_params[self.inc_params]
        for alpha in range(self.nphys):
            # unpack
            theta1 = params_phot[4 * alpha]
            theta2 = params_phot[4*alpha+1]
            Abar = params_phot[4 * alpha + 2]
            s = params_phot[4 * alpha + 3]
            
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

            #now, note that Rialpha for the inclination is constant here - so just take product
            Rialpha_theta1 = Rialpha_deriv_theta1 * self.Rialpha_inc
            Rialpha_theta2 = Rialpha_deriv_theta2 * self.Rialpha_inc
            Rialpha_Abar = Rialpha_deriv_Abar * self.Rialpha_inc
            Rialpha_s = Rialpha_deriv_s * self.Rialpha_inc 

            # some care is needed here: Rialpha_deriv is 0 for components outside the physical class
            Rialpha_theta1_sum = np.sum(Rialpha_theta1 * falpha, axis=1)
            Rialpha_theta2_sum = np.sum(Rialpha_theta2 * falpha, axis=1)
            Rialpha_Abar_sum = np.sum(Rialpha_Abar * falpha, axis=1)
            Rialpha_s_sum = np.sum(Rialpha_s * falpha, axis=1)


            gradRialpha_theta1 = np.sum(Rialpha_theta1_sum / self.falphaRalpha)
            gradRialpha_theta2 = np.sum(Rialpha_theta2_sum / self.falphaRalpha)
        
            gradRialpha_Abar = np.sum(Rialpha_Abar_sum / self.falphaRalpha)
            gradRialpha_s = np.sum(Rialpha_s_sum / self.falphaRalpha)

            prior_deriv = self.prior_deriv(params[alpha])

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
        
        # now proceed to compute concentration gradient
        # note that Rialpha_phot does not change - so we can re-use computation from early on
        Rialpha_inc = Rialpha_inc_deriv(self.inclinations, params_inc, self.nobj, self.ncomp) * self.Rialpha_phot 
        
        for alpha in range(self.ncomp):
            Rialphaderiv_comp = Rialpha_inc[:,alpha] 
            Salpha_comp = self.SalphatimesH_conce[alpha]

            gradRialpha = np.sum(Rialphaderiv_comp/self.falphaRalpha)
            prior_deriv = self.prior_deriv(params_inc[alpha])[4*self.nphys + alpha]
            
            entry = alpha // self.nphys
            grad[4*self.nphys + entry] = gradRialpha*falpha[alpha] - self.nobj*falpha[alpha]*Salpha_comp/Salpha_sum + prior_deriv

                
        g = grad[self.eff_grad_index]
        g[0] += grad[4] 
        g[1] += grad[5]
                
        return g*self.pref



class LogLikeGMMLCAIncTapered(LogLikeGMMLCAIncPerComponent):
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
        self.param_indices = [i for i in range(5*(nphys))]
        
        self.H_bins = np.arange(3.1,10.1,0.2)
        self.inc_bins = np.arange(1.5, 91.5, 3) * np.pi/180
        
        self.dH = 0.2 
        self.dInc = 3 * np.pi/180
    
    def size_dist(self, params_abs, params_inc, H, inc):
        alpha = params_abs[0::5]
        b = params_abs[1::5]
        H_crit = params_abs[2::5]
        return tapered_inc_multi(alpha, b, H_crit, params_inc, H, inc)
        
    def Rialpha(self, params):
        params_inc = params[5*self.nphys::]
        params_phot = params[self.param_indices] #constructed during init
        # save to avoid recomputing in derivative
        self.Rialpha_phot =  Rialpha_tapered_lca_physical(self.ncomp, self.nobj, self.nphys, self.jacobian, self.S_obj_noamp_physical, self.chains_H, self.chains_LCA, params_phot)
        self.Rialpha_inc = Rialpha_inc_individual(self.inclinations, params_inc, self.nobj, self.ncomp)
        return self.Rialpha_phot * self.Rialpha_inc 

    def Salpha(self, params, fun):
        # note that selection is independent of LCA - so int dLCA p(s|LCA) * p(LCA) = 1
        params_inc = params[5*self.nphys::]
        params_phot = params[self.param_indices]
        params_abs = np.tile(params_phot, int(self.ncomp/self.nphys))
                
        f = fun(params_abs, params_inc, self.H_bins, self.inc_bins)
        
        S = np.zeros(self.ncomp)
        for alpha in range(self.ncomp):
            S[alpha] = (
                np.trapz(np.trapz(self.selection[alpha]*f[:,:,alpha], dx = self.dH, axis=0),dx = self.dInc, axis=0)
            )
        return S

    def gradient_theta(self, falpha, params):
        raise NotImplementedError
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
        params_inc = params[4*self.nphys::]
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
