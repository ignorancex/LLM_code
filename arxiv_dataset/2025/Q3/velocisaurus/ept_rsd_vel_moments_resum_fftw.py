import numpy as np

from velocileptors.Utils.pnw_dst import pnw_dst

from ept_rsd_vel_moments_fftw import RedshiftSpaceVelocityMomentsEPT

class RedshiftSpaceVelocityMomentsREPT:

    '''
    Class to compute IR-resummed RSD pairwise velocity power spectra.
    
    Based on velocileptors.
    
    '''
    
    def __init__(self, k, p, pnw=None, *args, rbao = 110, kmin = 1e-2, kmax = 0.5, nk = 100, sbao=None, **kw):

        self.nk, self.kmin, self.kmax = nk, kmin, kmax
        self.rbao = rbao
        
        self.ept = RedshiftSpaceVelocityMomentsEPT( k, p, kmin=kmin, kmax=kmax, nk = nk, **kw)
        self.ncols = self.ept.ncols
        
        if pnw is None:
            knw, pnw = pnw_dst(k, p, ii_l=135,ii_r=225)
        else:
            knw, pnw = k, pnw

        self.ept_nw = RedshiftSpaceVelocityMomentsEPT( knw, pnw, kmin=kmin, kmax=kmax, nk = nk, **kw)
        
        self.kv = self.ept.kv
        self.nus, self.ws, self.Lns = self.ept.nus, self.ept.ws, self.ept.Lns
        self.plin  = self.ept.plin
        self.plin_nw = self.ept_nw.plin
        self.plin_w = self.plin - self.plin_nw
        
        if sbao is None:
            self.sigma_squared_bao = np.interp(self.rbao, self.ept.qint, self.ept.Xlin + self.ept.Ylin)
        else:
            self.sigma_squared_bao = sbao
            
        self.damp_exp = - 0.5 * self.kv**2 * self.sigma_squared_bao
        self.damp_fac = np.exp(self.damp_exp)


    def setup_rsd_spectra(self,f):
    
        self.ept.setup_Xis_redshift_space(f)
        #self.ept.compute_Xiells_redshift_space()
        
        self.ept_nw.setup_Xis_redshift_space(f)
        #self.ept_nw.compute_Xiells_redshift_space()
        
        self.pskmutable_nw = self.ept_nw.pskmutable
        self.vskmutable_nw = self.ept_nw.vskmutable
        self.sskmutable_nw = self.ept_nw.sskmutable
        
        self.pskmutable_nw_linear = self.ept_nw.pskmutable_linear
        self.vskmutable_nw_linear = self.ept_nw.vskmutable_linear
        self.sskmutable_nw_linear = self.ept_nw.sskmutable_linear
        
        self.pskmutable_w = self.ept.pskmutable - self.ept_nw.pskmutable
        self.vskmutable_w = self.ept.vskmutable - self.ept_nw.vskmutable
        self.sskmutable_w = self.ept.sskmutable - self.ept_nw.sskmutable

        self.pskmutable_w_linear = self.ept.pskmutable_linear - self.ept_nw.pskmutable_linear
        self.vskmutable_w_linear = self.ept.vskmutable_linear - self.ept_nw.vskmutable_linear
        self.sskmutable_w_linear = self.ept.sskmutable_linear - self.ept_nw.sskmutable_linear

        
        # Compute the "bracketed" quantities
        kv = self.kv[None,:,None]
        nu = self.nus[:,None,None]
        
        K2SigmaNu = kv**2 * (1 + f*(2+f)*nu**2) * self.sigma_squared_bao
        f1pfKNuSigma = f*(1+f)*kv*nu *self.sigma_squared_bao
        
        self.pskmutable_nodisps = self.pskmutable_w + 0.5 * K2SigmaNu * self.pskmutable_w_linear
        self.pskmutable = self.pskmutable_nw + np.exp(-0.5*K2SigmaNu) * self.pskmutable_nodisps
        self.pselltable = self.compute_multipole_moments(self.pskmutable)
        
        self.vskmutable_nodisps = self.vskmutable_w + 0.5 * K2SigmaNu * self.vskmutable_w_linear - f1pfKNuSigma * self.pskmutable_w_linear
        self.vskmutable = self.vskmutable_nw + np.exp(-0.5*K2SigmaNu) * (self.vskmutable_nodisps + f1pfKNuSigma * self.pskmutable_nodisps)
        self.vselltable = self.compute_multipole_moments(self.vskmutable)
        
        self.sskmutable_nodisps = self.sskmutable_w + 0.5 * K2SigmaNu * self.sskmutable_w_linear\
                                     + 2*f1pfKNuSigma*self.vskmutable_w_linear\
                                     - f**2*self.sigma_squared_bao*self.pskmutable_w_linear
        self.sskmutable = self.sskmutable_nw + np.exp(-0.5*K2SigmaNu)\
                            * (self.sskmutable_nodisps - 2*f1pfKNuSigma*self.vskmutable_nodisps\
                               + (-f1pfKNuSigma**2 + f**2*self.sigma_squared_bao) * self.pskmutable_nodisps)
        self.sselltable = self.compute_multipole_moments(self.sskmutable)

    def compute_multipole_moments(self,bias_table_kmu):
    
        #  format ell, k, bias_index
    
        ret = np.zeros((len(self.Lns),self.nk, self.ncols))
        
        for ell, Ln in enumerate(self.Lns):
        
            ret[ell] = (2*ell+1)/2. * np.sum((self.ws*Ln)[:,None,None]*bias_table_kmu,axis=0)
            
        return ret
        
    def combine_bias_terms(self, pvec, bias_table):
    
        return self.ept.combine_bias_terms(pvec, bias_table)
        
        #b1, b2, bs, b3, beta00, beta12, beta22, beta24, beta34, s00, s20, s12, s22 = pvec
        
        #monomials = np.array([1, b1, b1**2,\
        #                      b2, b1*b2, b2**2,\
        #                      bs, b1*bs, b2*bs, bs**2,\
        #                      b3, b1*b3,\
        #                      beta00, b1*beta00,\
        #                      beta12, b1*beta12,\
        #                      beta22, b1*beta22,\
        #                      beta24, b1*beta24,\
        #                      beta34, b1*beta34,\
        #                      s00, s20, s12, s22])

        #return np.sum(bias_table * monomials, axis=-1)

