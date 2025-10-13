
import numpy as np

from velocileptors.EPT.ept_fftw import EPT

class RedshiftSpaceVelocityMomentsEPT(EPT):

    def __init__(self, *args, ngauss=4,  **kw):
        
        EPT.__init__(self, *args, **kw)
        
        # How many bias terms are there?
        self.ncols = 12 + 10 + 4 + 4 # 12 bias combinations (see below) + 10 counterterm combinations + 4 stochastic terms + 2 FoG terms
        
        # We will define all our calculations on this mu grid
        self.ngauss = ngauss
        self.nus, self.ws = np.polynomial.legendre.leggauss(2*ngauss)
        self.nus_calc = self.nus[0:ngauss]
        
        self.L0 = np.polynomial.legendre.Legendre((1))(self.nus)
        self.L1 = np.polynomial.legendre.Legendre((0,1))(self.nus)
        self.L2 = np.polynomial.legendre.Legendre((0,0,1))(self.nus)
        self.L3 = np.polynomial.legendre.Legendre((0,0,0,1))(self.nus)
        self.L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(self.nus)
        
        self.Lns = [self.L0, self.L1, self.L2, self.L3, self.L4]
        
        # Bias indices for each moment (only necessary because the old labeling was dumb.)
        # The bias index will be:
        # 0: 1, 1: b1, 2: b1sq, 3: b2, 4: b1b2, 5: b2sq, 6: bs, 7: b1bs, 8: b2bs, 9: bssq, 10: b3, 11: b1b3
        
        self.piis = np.array([2,4,7,5,8,9,11])
        self.viis = np.array([1,2,3,4,6,7,10])
        self.siis = np.array([0,1,2,3,6])
        self.giis = np.array([0,1])
        self.kiis = np.array([0,])
        
        self.setup_Xin_real_space()
        
    
    def setup_Xin_real_space(self):
    
        # The format here will be mu, k, bias_index
        # the biastable_ept format is k, bias index

        
        # Xi0: Power Spectrum
        self.pkmutable = np.zeros((2*self.ngauss,self.nk,self.ncols))
        self.pkmutable[:,:,self.piis] += self.pktable_ept[None,:,1:-1] * self.L0[:,None,None]
        self.pkmutable_linear = np.zeros((2*self.ngauss,self.nk,self.ncols))
        self.pkmutable_linear[:,:,self.piis] += self.pktable_ept_linear[None,:,1:-1] * np.ones(2*self.ngauss)[:,None,None]
        
        # Xi1: vk
        self.vkmutable = np.zeros((2*self.ngauss,self.nk,self.ncols))
        self.vkmutable[:,:,self.viis] += self.vktable_ept[None,:,1:-1] * self.L1[:,None,None]
        self.vkmutable_linear = np.zeros((2*self.ngauss,self.nk,self.ncols))
        self.vkmutable_linear[:,:,self.viis] += self.vktable_ept_linear[None,:,1:-1] * self.L1[:,None,None]
        
        # Xi2: sk
        self.skmutable = np.zeros((2*self.ngauss,self.nk,self.ncols))
        self.skmutable[:,:,self.siis] += self.s0ktable_ept[None,:,1:-1] * self.L0[:,None,None] + \
                            self.s2ktable_ept[None,:,1:-1] * self.L2[:,None,None]
        self.skmutable_linear = np.zeros((2*self.ngauss,self.nk,self.ncols))
        self.skmutable_linear[:,:,self.siis] += self.s0ktable_ept_linear[None,:,1:-1] * self.L0[:,None,None] + \
                            self.s2ktable_ept_linear[None,:,1:-1] * self.L2[:,None,None]
        
        # Xi3: gk
        self.gkmutable = np.zeros((2*self.ngauss,self.nk,self.ncols))
        self.gkmutable[:,:,self.giis] += self.g1ktable_ept[None,:,1:-1] * self.nus[:,None,None] + self.g3ktable_ept[None,:,1:-1] * (self.nus**3)[:,None,None]
        
        # Xi4: gk
        self.kkmutable = np.zeros((2*self.ngauss,self.nk,self.ncols))
        self.kkmutable[:,:,self.kiis] += self.k0[None,:,None] * self.L0[:,None,None] + self.k2[None,:,None] * (self.nus**2)[:,None,None] + self.k4[None,:,None] * (self.nus**4)[:,None,None]


    def setup_Xis_redshift_space(self, f):
        
        fKnu = f*self.kv[None,:] * self.nus[:,None]
        
        # Power Spectrum
        
        # loops
        self.pskmutable = self.pkmutable - fKnu[:,:,None]*self.vkmutable - 0.5*(fKnu**2)[:,:,None]*self.skmutable\
                               + 1./6*(fKnu**3)[:,:,None]*self.gkmutable + 1./24*(fKnu**4)[:,:,None]*self.kkmutable
        
        # counterterms
        k2Plin = self.kv[None,:]**2 * self.plin[None,:]
        self.pskmutable[:,:,12] = 2*f*self.nus[:,None]**2*k2Plin # beta00
        self.pskmutable[:,:,13] = 2*k2Plin # b1 beta00
        self.pskmutable[:,:,14] = 2*f**2*self.nus[:,None]**4*k2Plin # beta12
        self.pskmutable[:,:,15] = 2*f*self.nus[:,None]**2*k2Plin # b1 beta12
        self.pskmutable[:,:,16] = 2*f**3*self.nus[:,None]**4*k2Plin # beta22
        self.pskmutable[:,:,17] = 2*f**2*self.nus[:,None]**2*k2Plin # b1 beta22
        self.pskmutable[:,:,18] = 2*f**3*self.nus[:,None]**6*k2Plin # beta24
        self.pskmutable[:,:,19] = 2*f**2*self.nus[:,None]**4*k2Plin # b1 beta24
        self.pskmutable[:,:,20] = 2*f**4*self.nus[:,None]**6*k2Plin # beta34
        self.pskmutable[:,:,21] = 2*f**3*self.nus[:,None]**4*k2Plin # b1 beta34
        
        # stochastic
        self.pskmutable[:,:,22] = np.ones((2*self.ngauss,self.nk)) # s00
        self.pskmutable[:,:,23] = self.kv[None,:]**2 # s20
        self.pskmutable[:,:,24] = f * self.nus[:,None]**2 * self.kv[None,:]**2 # s12
        self.pskmutable[:,:,25] = f**2 * self.nus[:,None]**2 * self.kv[None,:]**2 # s22
        
        # FoG
        k4mu4Plin = self.nus[:,None]**4 * self.kv[None,:]**4 * self.plin[None,:]
        
        self.pskmutable[:,:,26] = f**4 * f**2 * self.nus[:,None]**4 * k4mu4Plin # beta_fog
        self.pskmutable[:,:,27] = f**4 * 2*f*self.nus[:,None]**2 * k4mu4Plin # b1 beta_fog
        self.pskmutable[:,:,28] = f**4 * k4mu4Plin # b1^2 beta_fog
        
        self.pskmutable[:,:,29] = f**4 * self.nus[:,None]**4 * self.kv[None,:]**4

        # linear theory
        self.pskmutable_linear = self.pkmutable_linear - fKnu[:,:,None]*self.vkmutable_linear - 0.5*(fKnu**2)[:,:,None]*self.skmutable_linear

        
        # Velocity Spectrum
        self.vskmutable = f * (self.vkmutable + fKnu[:,:,None]*self.skmutable \
                              - 0.5*(fKnu**2)[:,:,None]*self.gkmutable - 1./6*(fKnu**3)[:,:,None]*self.kkmutable)
        
        # 2*beta00*f*k*\[Mu] + 2*b1*beta12*f*k*\[Mu] + 4*b1*beta22*f**2*k*\[Mu] + 4*beta12*f**2*k*\[Mu]**3 + 4*b1*beta24*f**2*k*\[Mu]**3 + 6*beta22*f**3*k*\[Mu]**3 + 6*b1*beta34*f**3*k*\[Mu]**3 + 6*beta24*f**3*k*\[Mu]**5 + 8*beta34*f**4*k*\[Mu]**5
        kPlin = self.kv[None,:] * self.plin[None,:]

        self.vskmutable[:,:,12] = -2*f*self.nus[:,None]*kPlin # beta00
        self.vskmutable[:,:,13] = 0 # b1 beta00
        self.vskmutable[:,:,14] = -4*f**2*self.nus[:,None]**3*kPlin # beta12
        self.vskmutable[:,:,15] = -2*f*self.nus[:,None]*kPlin # b1 beta12
        self.vskmutable[:,:,16] = -6*f**3*self.nus[:,None]**3*kPlin # beta22
        self.vskmutable[:,:,17] = -4*f**2*self.nus[:,None]*kPlin # b1 beta22
        self.vskmutable[:,:,18] = -6*f**3*self.nus[:,None]**5*kPlin # beta24
        self.vskmutable[:,:,19] = -4*f**2*self.nus[:,None]**3*kPlin # b1 beta24
        self.vskmutable[:,:,20] = -8*f**4*self.nus[:,None]**5*kPlin # beta34
        self.vskmutable[:,:,21] = -6*f**3*self.nus[:,None]**3*kPlin # b1 beta34

        # stochastic
        self.vskmutable[:,:,22] = 0 # s00
        self.vskmutable[:,:,23] = 0 # s20
        self.vskmutable[:,:,24] = - f * self.nus[:,None] * self.kv[None,:] # s12
        self.vskmutable[:,:,25] = - 2*f**2 * self.nus[:,None] * self.kv[None,:] # s22

        # FoG
        k3mu3Plin = self.nus[:,None]**3 * self.kv[None,:]**3 * self.plin[None,:]
        
        self.vskmutable[:,:,26] = -6 * f**6 * self.nus[:,None]**4 * k3mu3Plin # beta_fog
        self.vskmutable[:,:,27] = -10 * f**5 * self.nus[:,None]**2 * k3mu3Plin # b1 beta_fog
        self.vskmutable[:,:,28] = -4 * f**4 * k3mu3Plin # b1^2 beta_fog
        
        self.vskmutable[:,:,29] = -4 * f**4 * self.nus[:,None]**3 * self.kv[None,:]**3 # s44

        self.vskmutable_linear = f*(self.vkmutable_linear + fKnu[:,:,None]*self.skmutable_linear)

        # Dispersion Spectrum
        # 4*b1*beta22*f**2 + 4*beta12*f**2*\[Mu]**2 + 4*b1*beta24*f**2*\[Mu]**2 + 12*beta22*f**3*\[Mu]**2 + 12*b1*beta34*f**3*\[Mu]**2 + 12*beta24*f**3*\[Mu]**4 + 24*beta34*f**4*\[Mu]**4
        self.sskmutable = f**2 * (self.skmutable \
                                   - fKnu[:,:,None]*self.gkmutable - 0.5*(fKnu**2)[:,:,None]*self.kkmutable)
            
        Plin = self.plin[None,:]
        self.sskmutable[:,:,12] = 0 # beta00
        self.sskmutable[:,:,13] = 0 # b1 beta00
        self.sskmutable[:,:,14] = -4*f**2*self.nus[:,None]**2*Plin # beta12
        self.sskmutable[:,:,15] = 0 # b1 beta12
        self.sskmutable[:,:,16] = -12*f**3*self.nus[:,None]**2*Plin # beta22
        self.sskmutable[:,:,17] = -4*f**2*Plin # b1 beta22
        self.sskmutable[:,:,18] = -12*f**3*self.nus[:,None]**4*Plin # beta24
        self.sskmutable[:,:,19] = -4*f**2*self.nus[:,None]**2*Plin # b1 beta24
        self.sskmutable[:,:,20] = -24*f**4*self.nus[:,None]**4*Plin # beta34
        self.sskmutable[:,:,21] = -12*f**3*self.nus[:,None]**2*Plin # b1 beta34

        # stochastic
        self.sskmutable[:,:,22] = 0 # s00
        self.sskmutable[:,:,23] = 0 # s20
        self.sskmutable[:,:,24] = 0 # s12
        self.sskmutable[:,:,25] = -2*f**2 # s22
        
        # FoG
        # -12 b^2 k^2 P \[Beta] \[Mu]^2 - 40 b f k^2 P \[Beta] \[Mu]^4 - 2 k^2 \[Mu]^2 (6 s + 15 f^2 P \[Beta] \[Mu]^4)
        k2mu2Plin = self.nus[:,None]**2 * self.kv[None,:]**2 * self.plin[None,:]
        
        self.sskmutable[:,:,26] = -30 * f**6 * self.nus[:,None]**4 * k2mu2Plin # beta_fog
        self.sskmutable[:,:,27] = -40 * f**5 * self.nus[:,None]**2 * k2mu2Plin # b1 beta_fog
        self.sskmutable[:,:,28] = -12 * f**4 * k2mu2Plin # b1^2 beta_fog
        
        self.sskmutable[:,:,29] = -12 * f**4 * self.nus[:,None]**2 * self.kv[None,:]**2 # s44

        self.sskmutable_linear = f**2 * self.skmutable_linear
        

    def compute_multipole_moments(self,bias_table_kmu):
    
        #  format ell, k, bias_index
    
        ret = np.zeros((len(self.Lns),self.nk, self.ncols))
        
        for ell, Ln in enumerate(self.Lns):
        
            ret[ell] = (2*ell+1)/2. * np.sum((self.ws*Ln)[:,None,None]*bias_table_kmu,axis=0)
            
        return ret
        

    def compute_Xiells_redshift_space(self):
    
        self.pselltable = self.compute_multipole_moments(self.pskmutable)
        self.pselltable_linear = self.compute_multipole_moments(self.pskmutable_linear)

        self.vselltable = self.compute_multipole_moments(self.vskmutable)
        self.vselltable_linear = self.compute_multipole_moments(self.vskmutable_linear)

        self.sselltable = self.compute_multipole_moments(self.sskmutable)
        self.sselltable_linear = self.compute_multipole_moments(self.sskmutable_linear)


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
