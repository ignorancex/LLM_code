import numpy as np

from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

class LPT_RSD_VelocityMoments(LPT_RSD):

    '''
    Class to compute the (redshift-space) velocity moments.
    
    Based on the full-resummed LPT power spectrum predictions in LPT_RSD.
    
    '''
    
    def __init__(self, *args, dlambda=0.01, ngauss=4, **kw):
    
        LPT_RSD.__init__(self, *args, use_Pzel=False, **kw)
        
        # Derivatives
        self.dlambda = dlambda
        
        # Set Up Multipole Grid
        self.ngauss = ngauss
        self.nus, self.ws = np.polynomial.legendre.leggauss(2*self.ngauss)
        self.nus_calc = self.nus[0:self.ngauss]
        
        self.L0 = np.polynomial.legendre.Legendre((1))(self.nus)
        self.L1 = np.polynomial.legendre.Legendre((0,1))(self.nus)
        self.L2 = np.polynomial.legendre.Legendre((0,0,1))(self.nus)
        self.L3 = np.polynomial.legendre.Legendre((0,0,0,1))(self.nus)
        self.L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(self.nus)

        self.Ls = [self.L0, self.L1, self.L2, self.L3, self.L4]

    
    def make_pltable(self,f, apar = 1, aperp = 1, kv = None, kmin = 1e-2, kmax = 0.25, nk = 50, nmax=8):
        '''
        Copy of function from LPT_RSD.
        '''

        if kv is None:
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            nk = len(kv)
            
        self.pknutable = np.zeros((len(self.nus),nk,self.num_power_components+10+4+4-1))
        
        
        # To implement AP:
        # Calculate P(k,nu) at the true coordinates, given by
        # k_true = k_apfac * kobs
        # nu_true = nu * a_perp/a_par/fac
        # Note that the integration grid on the other hand is never observed
        
        for ii, nu in enumerate(self.nus_calc):
        
            fac = np.sqrt(1 + nu**2 * ((aperp/apar)**2-1))
            k_apfac = fac / aperp
            nu_true = nu * aperp/apar/fac
            vol_fac = apar * aperp**2
        
            self.setup_rsd_facs(f,nu_true)
            
            for jj, k in enumerate(kv):
                ktrue = k_apfac * k
                pterms = self.p_integrals(ktrue,nmax=nmax)
                
                #self.pknutable[ii,jj,:-4] = pterms[:-1]
                cterm_ii = self.num_power_components - 1
                self.pknutable[ii,jj,:cterm_ii] = pterms[:-1]
                
                # counterterms
                self.pknutable[ii,jj,cterm_ii + 0] = 2 * (1 + f*nu_true**2) * ktrue**2 * pterms[-1] # beta00
                self.pknutable[ii,jj,cterm_ii + 1] = 2 * ktrue**2 * pterms[-1] # beta00 b1
                self.pknutable[ii,jj,cterm_ii + 2] = 2 * (1 + f*nu_true**2) * f * nu_true**2 * ktrue**2 * pterms[-1] # beta12
                self.pknutable[ii,jj,cterm_ii + 3] = 2 * f * nu_true**2 * ktrue**2 * pterms[-1] # beta12 b1
                self.pknutable[ii,jj,cterm_ii + 4] = 2 * (1 + f*nu_true**2) * f**2 * nu_true**2 * ktrue**2 * pterms[-1] # beta22
                self.pknutable[ii,jj,cterm_ii + 5] = 2 * f**2 * nu_true**2 * ktrue**2 * pterms[-1] # beta22 b1
                self.pknutable[ii,jj,cterm_ii + 6] = 2 * (1 + f*nu_true**2) * f**2 * nu_true**4 * ktrue**2 * pterms[-1] # beta24
                self.pknutable[ii,jj,cterm_ii + 7] = 2 * f**2 * nu_true**4 * ktrue**2 * pterms[-1] # beta24 b1
                self.pknutable[ii,jj,cterm_ii + 8] = 2 * (1 + f*nu_true**2) * f**3 * nu_true**4 * ktrue**2 * pterms[-1] # beta24
                self.pknutable[ii,jj,cterm_ii + 9] = 2 * f**3 * nu_true**4 * ktrue**2 * pterms[-1] # beta24 b1
                
                # stochastic terms
                sterm_ii = cterm_ii + 10
                self.pknutable[ii,jj,sterm_ii + 0] = 1
                self.pknutable[ii,jj,sterm_ii + 1] = ktrue**2
                self.pknutable[ii,jj,sterm_ii + 2] = f * ktrue**2 * nu_true**2
                self.pknutable[ii,jj,sterm_ii + 3] = f**2 * ktrue**2 * nu_true**2
                
                # FoG terms
                fterm_ii = sterm_ii + 4
                self.pknutable[ii,jj,fterm_ii + 0] = (1 + f*nu_true**2)**2 * f**4 * ktrue**4 * nu_true**4 * pterms[-1] # beta_fog
                self.pknutable[ii,jj,fterm_ii + 1] = 2 * (1 + f*nu_true**2) * f**4 * ktrue**4 * nu_true**4 * pterms[-1] # beta_fog b1^2
                self.pknutable[ii,jj,fterm_ii + 2] = f**4 * ktrue**4 * nu_true**4 * pterms[-1] # beta_fog b1^2
                self.pknutable[ii,jj,fterm_ii + 3] = f**4 * ktrue**4 * nu_true**4 # s44



        self.pknutable[self.ngauss:,:,:] = np.flip(self.pknutable[0:self.ngauss],axis=0)
        self.pknutable /= (apar * aperp**2)
        
        self.kv = kv
        self.p0ktable = 0.5 * np.sum((self.ws*self.L0)[:,None,None]*self.pknutable,axis=0)
        self.p2ktable = 2.5 * np.sum((self.ws*self.L2)[:,None,None]*self.pknutable,axis=0)
        self.p4ktable = 4.5 * np.sum((self.ws*self.L4)[:,None,None]*self.pknutable,axis=0)
        
        return self.pknutable, self.p0ktable, self.p2ktable, self.p4ktable

    def prepare_pknu_tables(self, f, lambda_center=1, apar = 1, aperp = 1, kv = None, kmin = 1e-2, kmax = 0.25, nk = 50, nmax=8):
        
        # These are factors such that nu_true is the "true" los angle and ktrue = ktrue_fac * k
        fac = np.sqrt(1 + self.nus**2 * ((aperp/apar)**2-1))
        self.nu_trues = self.nus * aperp/apar * fac
        self.ktrue_fac = fac / aperp

        self.make_pltable( f * (lambda_center - self.dlambda),\
                           apar=apar, aperp=aperp,\
                           kv=kv, kmin=kmin, kmax=kmax, nk=nk, nmax=nmax)

        self.pknutable_m = 1.0 * self.pknutable
        
        self.make_pltable( f * (lambda_center + self.dlambda),\
                           apar=apar, aperp=aperp,\
                           kv=kv, kmin=kmin, kmax=kmax, nk=nk, nmax=nmax)

        self.pknutable_p = 1.0 * self.pknutable

        self.make_pltable( f * lambda_center,\
                           apar=apar, aperp=aperp,\
                           kv=kv, kmin=kmin, kmax=kmax, nk=nk, nmax=nmax)
                           
        self.f = f
                           
        return self.kv, self.pknutable, self.pknutable_m, self.pknutable_p

    def make_Xi_n_table(self, f, lambda_center = 1,\
                        apar = 1, aperp = 1,
                        kv = None, kmin = 1e-2, kmax = 0.25, nk = 50, nmax=8):
    
        self.prepare_pknu_tables(f, lambda_center=lambda_center, apar = apar, aperp = aperp, kv = kv, kmin = kmin, kmax = kmax, nk = nk, nmax=nmax)

        # Compute derivatives
        self.dPdlambda = (self.pknutable_p - self.pknutable_m)/(2 * self.dlambda)
        self.d2Pdlambda2 = (self.pknutable_p - 2*self.pknutable + self.pknutable_m) / self.dlambda**2

        # Now Compute Multipoles
        self.kpars_true = (self.nu_trues * self.ktrue_fac)[:, None, None] * self.kv[None, :, None]
        
        self.Xi0_0 = 0.5 * np.sum((self.ws*self.L0)[:,None,None] * self.kpars_true**(0) * self.pknutable,axis=0)
        self.Xi0_2 = 2.5 * np.sum((self.ws*self.L2)[:,None,None] * self.kpars_true**(0) * self.pknutable,axis=0)
        self.Xi0_4 = 4.5 * np.sum((self.ws*self.L4)[:,None,None] * self.kpars_true**(0) * self.pknutable,axis=0)
        
        self.Xi1_1 = -1.5 * np.sum((self.ws*self.L1)[:,None,None] * self.kpars_true**(-1) * self.dPdlambda,axis=0)
        self.Xi1_3 = -3.5 * np.sum((self.ws*self.L3)[:,None,None] * self.kpars_true**(-1) * self.dPdlambda,axis=0)
    
        self.Xi2_0 = -0.5 * np.sum((self.ws*self.L0)[:,None,None] * self.kpars_true**(-2) * self.d2Pdlambda2,axis=0)
        self.Xi2_2 = -2.5 * np.sum((self.ws*self.L2)[:,None,None] * self.kpars_true**(-2) * self.d2Pdlambda2,axis=0)
        #self.Xi2_4 = -4.5 * np.sum((self.ws*self.L4)[:,None,None] * self.kpars_true**(-2) * self.d2Pdlambda2,axis=0)

        return self.kv, self.Xi0_0, self.Xi0_2, self.Xi0_4, self.Xi1_1, self.Xi1_3, self.Xi2_0, self.Xi2_2#, self.Xi2_4


    def combine_bias_terms_Xi_n(self, bvec):

        #b1, b2, bs, b3, beta00, beta21, beta22, beta42, beta43, s00, s02, s21, s22, beta_fog, s44 = bvec
        
                
        p0, p2, p4 = self.combine_bias_terms(bvec, self.Xi0_0), self.combine_bias_terms(bvec, self.Xi0_2), self.combine_bias_terms(bvec, self.Xi0_4)
        v1, v3 = self.combine_bias_terms(bvec, self.Xi1_1), self.combine_bias_terms(bvec, self.Xi1_3)
        s0, s2 = self.combine_bias_terms(bvec, self.Xi2_0), self.combine_bias_terms(bvec, self.Xi2_2)
        
        return self.kv, (p0,p2,p4), (v1,v3), (s0,s2)

    def combine_bias_terms(self, pvec, bias_table):
    
        b1, b2, bs, b3, beta00, beta12, beta22, beta24, beta34, s00, s20, s12, s22, beta_fog, s44 = pvec
        
        monomials = np.array([1, b1, b1**2,\
                              b2, b1*b2, b2**2,\
                              bs, b1*bs, b2*bs, bs**2,\
                              b3, b1*b3,\
                              beta00, b1*beta00,\
                              beta12, b1*beta12,\
                              beta22, b1*beta22,\
                              beta24, b1*beta24,\
                              beta34, b1*beta34,\
                              s00, s20, s12, s22,\
                              beta_fog, b1*beta_fog, b1**2 * beta_fog, s44])

        return np.sum(bias_table * monomials, axis=-1)
