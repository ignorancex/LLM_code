import numpy as np

from scipy.interpolate import interp1d

from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

class LPT_RSD_VelocityMoments(LPT_RSD):

    '''
    Class to compute the (redshift-space) velocity moments.
    
    Based on the full-resummed LPT power spectrum predictions in LPT_RSD.
    
    '''
    
    def __init__(self, *args, dlambda=0.01, ngauss=4, **kw):
    
        LPT_RSD.__init__(self, *args, **kw)
                
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
        
        # Some correlators we need differently from the power spectrum
        self.Xlin_tot = self.qf.Xlin_lt + self.qf.Xlin_gt
        self.Ylin_tot = self.qf.Ylin_lt + self.qf.Ylin_gt
        
        if self.one_loop:
            self.X10loop12 = self.qf.X10loop12
            self.Y10loop12 = self.qf.Y10loop12
        else:
            self.X10loop12, self.Y10loop12 = (0,)*2

   
   ## New Angular Kernels Needed for Peculiar Velocities
   
    def _d3G0dC3_l_n(self,n,m):
        fnm = self.fnms[n,m]
        c = self.c
        s = self.s
    
        coeff1 = (-6*s**4 + 2*c**4*(1 + m)*(1 + 2*m - 2*n)*(-1 + 2*n) + s**2*c**2*(-5 - 14*m + 28*(1 + m)*n))
        coeff2 = (-(-1 + 2*n)*(6*s**4 + 2*c**4*(1 + 2*m - 2*n)*(1 + m - n) + s**2*c**2*(5 + 14*m - 4*(4 + m)*n)))
    
        ret = coeff1 * self.hyp1[n,m] + coeff2 * self.hyp2[n,m]
    
        ret *= 1/s
    
        return fnm * ret

    def _d3G0dC3_l(self,l,k,nmax=5):
        summand =  (k**(2* (l+np.arange(nmax))) * self.d3G0dC3_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**3
        
    def _d3G0dC2dA_l_n(self,n,m):
        fnm = self.fnms[n,m]
        c = self.c
        s = self.s
    
        coeff1 = 2*c**4*(m*(1 + 2*m) - 4*m*(1 + m)*n + 4*(1 + m)*n**2) - c**2*(1 - 10*m + 20*(1 + m)*n)*s**2 + 2*s**4
        coeff2 = (-1 + 2*n)*(2*c**4*(1 + 2*m - 2*n)*(m - n) - c**2*(1 - 10*m + 4*(3 + m)*n)*s**2 + 2*s**4)
    
        ret = coeff1 * self.hyp1[n,m] + coeff2 * self.hyp2[n,m]
        ret *= fnm/c
    
        return ret

    def _d3G0dC2dA_l(self,l,k,nmax=5):
        summand =  (k**(2* (l+np.arange(nmax))) * self.d3G0dC2dA_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**3
   
    def _d4G0dC2dA2_l_n(self,n,m):
        fnm = self.fnms[n,m]
        c = self.c
        s = self.s
    
        coeff1 = -(2*c**4*(m - 4*m**3 + 2*(1 + m)*(1 + 4*m**2)*n - 4*(1 + m)*(-1 + 4*m)*n**2 + 8*(1 + m)*n**3) -
                c**2*(3 + 12*m*(-2 + 3*m) - 80*(-1 + m**2)*n + 16*(1 + m)*(5 + m)*n**2)*s**2 + 12*(1 - 2*m + 4*(1 + m)*n)*s**4)
        coeff2 =  (-1 + 2*n)*(2*c**4*(-1 + 2*m - 2*n)*(1 + 2*m - 2*n)*(m - n) +
             c**2*(3 + 4*m**2*(9 - 4*n) + 4*n*(8 + 11*n) + 8*m*(-3 + 2*(-4 + n)*n))*s**2 - 12*(1 + 2*m*(-1 + n) + 3*n)*s**4)
    
        ret = coeff1 * self.hyp1[n,m] + coeff2 * self.hyp2[n,m]
    
        return fnm * ret

    def _d4G0dC2dA2_l(self,l,k,nmax=5):
        summand =  (k**(2* (l+np.arange(nmax))) * self.d4G0dC2dA2_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**4

    def _d4G0dCdA3_l_n(self,n,m):
        fnm = self.fnms[n,m]
        c = self.c
        s = self.s
    
        coeff1 = (4*c**4*(m**3*(-2 + 4*n) + m*(-1 + n + 4*n**3) + n*(7 + 4*n*(2 + n)) + m**2*(3 - 2*n*(1 + 4*n))) -
              4*c**2*(3 + 3*m*(-3 + 2*m) + 23*n + (9 - 14*m)*m*n + 2*(1 + m)*(7 + 2*m)*n**2)*s**2 + 3*(1 - 2*m + 4*(1 + m)*n)*s**4)
        coeff2 = (-1 + 2*n)*(-4*c**4*(-1 + 2*m - 2*n)*(-1 + m - n)*(m - n) +
               2*c**2*(-6 + 6*(3 - 2*m)*m - 25*n + 2*m*(7 + 4*m)*n - 8*(2 + m)*n**2)*s**2 + 3*(1 - 2*m + 4*(1 + m)*n)*s**4)
    
        ret = coeff1 * self.hyp1[n,m] + coeff2 * self.hyp2[n,m]
        ret *= s/c
    
        return fnm * ret

    def _d4G0dCdA3_l(self,l,k,nmax=5):
        summand =  (k**(2* (l+np.arange(nmax))) * self.d4G0dCdA3_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**4
   
   
    ### Generalized Function
    def setup_rsd_facs(self,f,nu,D=1,nmax=10):
        super().setup_rsd_facs(f,nu,D=D,nmax=nmax)
        
        self.d3G0dC3_l_ns = np.zeros( (self.jn,nmax) )
        self.d3G0dC2dA_l_ns = np.zeros( (self.jn,nmax) )
        self.d4G0dC2dA2_l_ns = np.zeros( (self.jn,nmax) )
        self.d4G0dCdA3_l_ns = np.zeros( (self.jn,nmax) )

        for ll in range(self.jn):
            for nn in range(nmax):
                self.d3G0dC3_l_ns[ll,nn] = self._d3G0dC3_l_n(ll+nn,ll)
                self.d3G0dC2dA_l_ns[ll,nn] = self._d3G0dC2dA_l_n(ll+nn,ll)
                self.d4G0dC2dA2_l_ns[ll,nn] = self._d4G0dC2dA2_l_n(ll+nn,ll)
                self.d4G0dCdA3_l_ns[ll,nn] = self._d4G0dCdA3_l_n(ll+nn,ll)


    ### Now define the actual integrals!

    def v_integrals(self, k, nmax=8):
        
        ksq = k**2
        Kfac = self.Kfac
        f = self.f
        nu = self.nu; kpar = k*nu
        Anu, Bnu = self.Anu, self.Bnu
        
        K = k*self.Kfac; Ksq = K**2
        Knfac = nu*(1+f); Kn = Knfac * k
        
        D2 = self.D**2; D4 = D2**2

        expon = np.exp(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*Ksq * D2* self.sigma)


        A = k*self.qint*self.c
        C = k*self.qint*self.s
        
        
        G0s =  [self._G0_l(ii,k,nmax=nmax)    for ii in range(self.jn)] + [0] + [0] + [0] + [0]
        dGdAs =  [self._dG0dA_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0] + [0]
        dGdCs = [self._dG0dC_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0] + [0]
        d2GdA2s = [self._d2G0dA2_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0]
        d2GdCdAs = [self._d2G0dCdA_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0] + [0]
        d2GdC2s = [self._d2G0dC2_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0] + [0]
        d3GdA3s = [self._d3G0dA3_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d3GdCdA2s = [self._d3G0dCdA2_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d3GdC2dAs = [self._d3G0dC2dA_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d3GdC3s = [self._d3G0dC3_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d4GdA4s = [self._d4G0dA4_l(ii,k,nmax=nmax) for ii in range(self.jn) ]
        d4GdCdA3s = [self._d4G0dCdA3_l(ii,k,nmax=nmax) for ii in range(self.jn) ]
        d4GdC2dA2s = [self._d4G0dC2dA2_l(ii,k,nmax=nmax) for ii in range(self.jn) ]

        G01s = [-(dGdAs[ii] + 0.5*A*G0s[ii-1])   for ii in range(self.jn)]
        G02s = [-(d2GdA2s[ii] + A * dGdAs[ii-1] + 0.5*G0s[ii-1] + 0.25 * A**2 *G0s[ii-2]) for ii in range(self.jn)]
        G03s = [d3GdA3s[ii] + 1.5*A*d2GdA2s[ii-1] + 1.5*dGdAs[ii-1] \
                 + 0.75*A**2*dGdAs[ii-2] + 0.75*A*G0s[ii-2] + A**3/8.*G0s[ii-3] for ii in range(self.jn)]
        G04s = [d4GdA4s[ii] + 2*A*d3GdA3s[ii-1] + 3*d2GdA2s[ii-1] \
                + 1.5*A**2*d2GdA2s[ii-2] + 3*A*dGdAs[ii-2] + 0.75*G0s[ii-2]\
                + 0.5*A**3*dGdAs[ii-3] + 0.75*A**2*G0s[ii-3]\
                + A**4/16. * G0s[ii-4] for ii in range(self.jn)]
                 
        G10s = [ dGdCs[ii] + 0.5*C*G0s[ii-1]  for ii in range(self.jn)]
        
        G11s = [ d2GdCdAs[ii] + 0.5*C*dGdAs[ii-1] + 0.5*A*dGdCs[ii-1] + 0.25*A*C*G0s[ii-2] for ii in range(self.jn)]
        G20s = [-(d2GdC2s[ii] + C * dGdCs[ii-1] + 0.5*G0s[ii-1] + 0.25 * C**2 *G0s[ii-2]) for ii in range(self.jn)]
        G12s = [-(d3GdCdA2s[ii] + 0.5*C*d2GdA2s[ii-1] + A*d2GdCdAs[ii-1] + 0.5*dGdCs[ii-1]\
                  + 0.5*A*C*dGdAs[ii-2] + 0.25*A**2*dGdCs[ii-2] + 0.25*C*G0s[ii-2] + A**2*C/8*G0s[ii-3])  for ii in range(self.jn)]

        G13s = [-(d4GdCdA3s[ii] + 1.5 * d2GdCdAs[ii-1] + 1.5 * A * d3GdCdA2s[ii-1] + 0.5 * C * d3GdA3s[ii-1]\
                + 0.75 * (A * dGdCs[ii-2] + C * dGdAs[ii-2] + A**2 * d2GdCdAs[ii-2] + A*C * d2GdA2s[ii-2])\
                + 0.125 * (3*A*C*G0s[ii-3] + A**3 * dGdCs[ii-3] + 3*A**2*C*dGdAs[ii-3]) + A**3*C/16*G0s[ii-4]) for ii in range(self.jn)]
                
        G21s = [ d3GdC2dAs[ii] + C * d2GdCdAs[ii-1] + 0.5 * dGdAs[ii-1] + 0.5 * A * d2GdC2s[ii-1]\
                + 0.25 * C**2 * dGdAs[ii-2] + 0.5 * A*C * dGdCs[ii-2] + 0.25 * A * G0s[ii-2]\
                + 0.125 * A*C**2 * G0s[ii-3] for ii in range(self.jn) ]
        
        G30s = [-(d3GdC3s[ii] + 1.5 * C* d2GdC2s[ii-1] + 1.5 * dGdCs[ii-1] + \
                  0.75 * C**2 * dGdCs[ii-2] + 0.75 * C * G0s[ii-2] + \
                  0.125 * C**3 * G0s[ii-3]) for ii in range(self.jn)]
        
        G22s = [ d4GdC2dA2s[ii] + 0.5 * d2GdC2s[ii-1] + A * d3GdC2dAs[ii-1] + 0.5 * d2GdA2s[ii-1] + C * d3GdCdA2s[ii-1] +\
                 0.25 * G0s[ii-2] + 0.5 * C * dGdCs[ii-2] + 0.25 * A**2 * d2GdC2s[ii-2] + \
                 0.5 * A * dGdAs[ii-2] + A*C * d2GdCdAs[ii-2] + 0.25 * C**2 * d2GdA2s[ii-2] +\
                 0.125 * (A**2 + C**2) * G0s[ii-3] + 0.25 * A*C * (A * dGdCs[ii-3] + C * dGdAs[ii-3]) +\
                 A**2*C**2/16 * G0s[ii-4] for ii in range(self.jn) ]

        ret = np.zeros(self.num_power_components)
            
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
                            
        for l in range(self.jn):
            
            mu0 = G0s[l]
            nq1 = self.Anu * G01s[l] + self.Bnu * G10s[l]
            mu_nq1 = self.Anu * G02s[l] + self.Bnu * G11s[l]
            nq2 = self.Anu**2 * G02s[l] + 2 * self.Anu * self.Bnu * G11s[l] + self.Bnu**2 * G20s[l]
            mu1 = G01s[l]
            mu2 = G02s[l]
            mu3 = G03s[l]
            mu2_nq1 = self.Anu * G03s[l] + self.Bnu * G12s[l]
            mu_nq2 = self.Anu**2 * G03s[l] + 2*self.Anu*self.Bnu*G12s[l] + self.Bnu**2*G21s[l]
            mu4 = G04s[l]
            mu3_nq1 = self.Anu * G04s[l] + self.Bnu * G13s[l]
            
            bias_integrands[0,:] = Kn * self.Xlin_tot * mu0 + K * mu_nq1 * self.Ylin_tot # za linear
            
            # factor of 2,3 from derivative of Psi{2,3}
            bias_integrands[0,:] += 2*(Kn + f*kpar) * self.X22 * mu0 + 2*(K * mu_nq1 + f*kpar*nq2) * self.Y22 # A22
            bias_integrands[0,:] += 3*(Kn*mu0 * self.X13 + K*mu_nq1 * self.Y13) # A13
            bias_integrands[0,:] += (Kn + 2*f*kpar) * self.X13 * mu0 + (K*mu_nq1 + 2*f*kpar*nq2) * self.Y13
            
            bias_integrands[0,:] += -0.5 * Ksq * (Kn * self.Xlin_tot * (self.Xlin_gt * mu0 + self.Ylin_gt * mu2) \
                                                   + K * self.Ylin_tot * (self.Xlin_gt * mu_nq1 + self.Ylin_gt * mu3_nq1 ) ) # Aloop
                                                        
            bias_integrands[0,:] += - (2*Kn*K*mu1 + K*Kn*mu1 + Ksq*nq1 + f*kpar*(K*mu1 + Kn*nq1)) * self.V1 \
                                    - (Ksq * nq1 + Kn * K*mu1 + f*kpar*Kn*nq1) * self.V3 \
                                    - (Ksq * mu2_nq1 + Ksq * mu2_nq1 + f*kpar*K*mu_nq2) * self.T #Wijk
            
            bias_integrands[1,:] = 2 * ( (1 - 0.5*Ksq*self.Xlin_gt) * self.Ulin * nq1 - 0.5*Ksq * self.Ylin_gt * self.Ulin * mu2_nq1 \
                                          - K*self.Ulin*(Kn*self.Xlin_tot*mu1 + K*mu2_nq1*self.Ylin_tot) \
                                          + 3 * self.U3 * nq1 + (3*Kn + f*kpar)*self.X10loop12*mu0 + (3*K*mu_nq1 + f*kpar*nq2)*self.Y10loop12)
                                 # b1 #in progress
                                   
            bias_integrands[2,:] = self.corlin * (Kn * self.Xlin_tot * mu0 + K*mu_nq1 * self.Ylin_tot)\
                                     + 2 * K * mu_nq1 * self.Ulin**2 + 2 * self.U11 * nq1 # b1sq # in progress
                                   
                                   
            bias_integrands[3,:] = 2 * self.U20 * nq1 + 2*K * self.Ulin**2 * mu_nq1 # b2
            bias_integrands[4,:] = 2 * self.corlin * self.Ulin * nq1 # b1b2
            #bias_integrands[5,:] = 0  # b2sq
            
            if self.shear or self.third_order:
                bias_integrands[6,:] = 4 * self.Us2 * nq1 + 2 * Kn * self.Xs2 * mu0 + 2 * K * self.Ys2 * mu_nq1 # bs should be both minus
                bias_integrands[7,:] = 2 * self.V * nq1 # b1bs
                #bias_integrands[8,:] = 0 # b2bs
                #bias_integrands[9,:] = 0 # bssq
                
            if self.third_order:
                bias_integrands[10,:] = 2 * self.Ub3 * nq1 #b3
                #bias_integrands[11,:] = 0 #b1 b3
            
            # Counterterm
            bias_integrands[-1,:] = self.corlin * mu0 # za
                                   
            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                bias_integrands -= bias_integrands[:,-1][:,None]
            else:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                                                                
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            ret += interp1d(ktemps, bias_ffts)(k)

        return 4*suppress*np.pi*ret

    def s_integrals(self, k, nmax=8):
        
        ksq = k**2
        Kfac = self.Kfac
        f = self.f
        nu = self.nu; kpar = k*nu
        Anu, Bnu = self.Anu, self.Bnu
        
        K = k*self.Kfac; Ksq = K**2
        Knfac = nu*(1+f); Kn = Knfac * k
        
        D2 = self.D**2; D4 = D2**2

        expon = np.exp(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*Ksq * D2* self.sigma)


        A = k*self.qint*self.c
        C = k*self.qint*self.s
        
        
        G0s =  [self._G0_l(ii,k,nmax=nmax)    for ii in range(self.jn)] + [0] + [0] + [0] + [0]
        dGdAs =  [self._dG0dA_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0] + [0]
        dGdCs = [self._dG0dC_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0] + [0]
        d2GdA2s = [self._d2G0dA2_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0]
        d2GdCdAs = [self._d2G0dCdA_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0] + [0]
        d2GdC2s = [self._d2G0dC2_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0] + [0]
        d3GdA3s = [self._d3G0dA3_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d3GdCdA2s = [self._d3G0dCdA2_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d3GdC2dAs = [self._d3G0dC2dA_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d3GdC3s = [self._d3G0dC3_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d4GdA4s = [self._d4G0dA4_l(ii,k,nmax=nmax) for ii in range(self.jn) ]
        d4GdCdA3s = [self._d4G0dCdA3_l(ii,k,nmax=nmax) for ii in range(self.jn) ]
        d4GdC2dA2s = [self._d4G0dC2dA2_l(ii,k,nmax=nmax) for ii in range(self.jn) ]

        G01s = [-(dGdAs[ii] + 0.5*A*G0s[ii-1])   for ii in range(self.jn)]
        G02s = [-(d2GdA2s[ii] + A * dGdAs[ii-1] + 0.5*G0s[ii-1] + 0.25 * A**2 *G0s[ii-2]) for ii in range(self.jn)]
        G03s = [d3GdA3s[ii] + 1.5*A*d2GdA2s[ii-1] + 1.5*dGdAs[ii-1] \
                 + 0.75*A**2*dGdAs[ii-2] + 0.75*A*G0s[ii-2] + A**3/8.*G0s[ii-3] for ii in range(self.jn)]
        G04s = [d4GdA4s[ii] + 2*A*d3GdA3s[ii-1] + 3*d2GdA2s[ii-1] \
                + 1.5*A**2*d2GdA2s[ii-2] + 3*A*dGdAs[ii-2] + 0.75*G0s[ii-2]\
                + 0.5*A**3*dGdAs[ii-3] + 0.75*A**2*G0s[ii-3]\
                + A**4/16. * G0s[ii-4] for ii in range(self.jn)]
                 
        G10s = [ dGdCs[ii] + 0.5*C*G0s[ii-1]  for ii in range(self.jn)]
        
        G11s = [ d2GdCdAs[ii] + 0.5*C*dGdAs[ii-1] + 0.5*A*dGdCs[ii-1] + 0.25*A*C*G0s[ii-2] for ii in range(self.jn)]
        G20s = [-(d2GdC2s[ii] + C * dGdCs[ii-1] + 0.5*G0s[ii-1] + 0.25 * C**2 *G0s[ii-2]) for ii in range(self.jn)]
        G12s = [-(d3GdCdA2s[ii] + 0.5*C*d2GdA2s[ii-1] + A*d2GdCdAs[ii-1] + 0.5*dGdCs[ii-1]\
                  + 0.5*A*C*dGdAs[ii-2] + 0.25*A**2*dGdCs[ii-2] + 0.25*C*G0s[ii-2] + A**2*C/8*G0s[ii-3])  for ii in range(self.jn)]

        G13s = [-(d4GdCdA3s[ii] + 1.5 * d2GdCdAs[ii-1] + 1.5 * A * d3GdCdA2s[ii-1] + 0.5 * C * d3GdA3s[ii-1]\
                + 0.75 * (A * dGdCs[ii-2] + C * dGdAs[ii-2] + A**2 * d2GdCdAs[ii-2] + A*C * d2GdA2s[ii-2])\
                + 0.125 * (3*A*C*G0s[ii-3] + A**3 * dGdCs[ii-3] + 3*A**2*C*dGdAs[ii-3]) + A**3*C/16*G0s[ii-4]) for ii in range(self.jn)]
                
        G21s = [ d3GdC2dAs[ii] + C * d2GdCdAs[ii-1] + 0.5 * dGdAs[ii-1] + 0.5 * A * d2GdC2s[ii-1]\
                + 0.25 * C**2 * dGdAs[ii-2] + 0.5 * A*C * dGdCs[ii-2] + 0.25 * A * G0s[ii-2]\
                + 0.125 * A*C**2 * G0s[ii-3] for ii in range(self.jn) ]
        
        G30s = [-(d3GdC3s[ii] + 1.5 * C* d2GdC2s[ii-1] + 1.5 * dGdCs[ii-1] + \
                  0.75 * C**2 * dGdCs[ii-2] + 0.75 * C * G0s[ii-2] + \
                  0.125 * C**3 * G0s[ii-3]) for ii in range(self.jn)]
        
        G22s = [ d4GdC2dA2s[ii] + 0.5 * d2GdC2s[ii-1] + A * d3GdC2dAs[ii-1] + 0.5 * d2GdA2s[ii-1] + C * d3GdCdA2s[ii-1] +\
                 0.25 * G0s[ii-2] + 0.5 * C * dGdCs[ii-2] + 0.25 * A**2 * d2GdC2s[ii-2] + \
                 0.5 * A * dGdAs[ii-2] + A*C * d2GdCdAs[ii-2] + 0.25 * C**2 * d2GdA2s[ii-2] +\
                 0.125 * (A**2 + C**2) * G0s[ii-3] + 0.25 * A*C * (A * dGdCs[ii-3] + C * dGdAs[ii-3]) +\
                 A**2*C**2/16 * G0s[ii-4] for ii in range(self.jn) ]

        ret = np.zeros(self.num_power_components)
            
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
                            
        for l in range(self.jn):
            
            mu0 = G0s[l]
            nq1 = self.Anu * G01s[l] + self.Bnu * G10s[l]
            mu_nq1 = self.Anu * G02s[l] + self.Bnu * G11s[l]
            nq2 = self.Anu**2 * G02s[l] + 2 * self.Anu * self.Bnu * G11s[l] + self.Bnu**2 * G20s[l]
            mu1 = G01s[l]
            mu2 = G02s[l]
            mu3 = G03s[l]
            mu2_nq1 = self.Anu * G03s[l] + self.Bnu * G12s[l]
            mu_nq2 = self.Anu**2 * G03s[l] + 2*self.Anu*self.Bnu*G12s[l] + self.Bnu**2*G21s[l]
            nq3 = self.Anu**3 * G03s[l] + 3*self.Anu**2*self.Bnu*G12s[l] + 3*self.Anu*self.Bnu**2*G21s[l] + self.Bnu**3*G30s[l]
            mu4 = G04s[l]
            mu3_nq1 = self.Anu * G04s[l] + self.Bnu * G13s[l]
            mu2_nq2 = self.Anu**2 * G04s[l] + 2*self.Anu*self.Bnu*G13s[l] + self.Bnu**2*G22s[l]
            
            bias_integrands[0,:] = self.Xlin_tot * mu0 + self.Ylin_tot * nq2 # za linear
            bias_integrands[0,:] += - (Kn**2 * self.Xlin_tot**2 * mu0 + 2 * self.Xlin_tot*self.Ylin_tot * Kn*K*mu_nq1 + self.Ylin_tot**2*K**2*mu2_nq2)\
                                   -0.5 * K**2 * self.Xlin_gt * (self.Xlin_tot * mu0 + self.Ylin_tot * nq2) \
                                   - 0.5 * K**2 * self.Ylin_gt * (self.Xlin_tot * mu2 + self.Ylin_tot * mu2_nq2)
            bias_integrands[0,:] += (6*self.X13 + 4*self.X22) * mu0 + (6*self.Y13 + 4*self.Y22) * nq2 # Aloop
            bias_integrands[0,:] += (-4 *(K*mu1 + Kn*nq1) - 2*(Kn + f*kpar)*nq1 ) * self.V1 + \
                                   (-4 * Kn * nq1 - K * mu1 - f*kpar*nq1)*self.V3  + (-5 * K * mu_nq2 - f*kpar * nq3) * self.T # Wijk
                                                
            bias_integrands[1,:] = -2 * self.Ulin * ( K*self.Xlin_tot*mu1 + K*self.Ylin_tot*mu_nq2 \
                                                      + 2*self.Xlin_tot*Kn*nq1 + 2*self.Ylin_tot*K*mu_nq2 )
            bias_integrands[1,:] += 4 * (self.X10 * mu0 + self.Y10 * nq2)  # b1
                                   
            bias_integrands[2,:] = 2 * self.Ulin**2 * nq2 + self.corlin * (self.Xlin_tot * mu0 + self.Ylin_tot * nq2) # b1sq
                                   
                                   
            bias_integrands[3,:] = 2 * self.Ulin**2 * nq2 # b2
            bias_integrands[4,:] = 0 # b1b2
            bias_integrands[5,:] = 0 # b2sq
            
            if self.shear or self.third_order:
                bias_integrands[6,:] = 2 * (self.Xs2 * mu0 + self.Ys2 * nq2) # bs should be both minus
                bias_integrands[7,:] = 0 # b1bs
                bias_integrands[8,:] = 0 # b2bs
                bias_integrands[9,:] = 0 # bssq
                
            if self.third_order:
                bias_integrands[10,:] = 0 #b3
                bias_integrands[11,:] = 0 #b1 b3
            
            # Counterterm
            bias_integrands[-1,:] =  self.corlin * mu0
                                   
            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                bias_integrands -= bias_integrands[:,-1][:,None]
            else:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                                                                
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            ret += interp1d(ktemps, bias_ffts)(k)

        return 4*suppress*np.pi*ret


    def make_table(self, f, nu, spectrum='vk', kv = None, kmin = 1e-2, kmax = 0.25, nk = 50,nmax=5):
    
        self.setup_rsd_facs(f,nu,nmax=nmax)
        
        if kv is None:
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            nk = len(kv)
        
        
        table = np.zeros([nk, self.num_power_components+1]) # one column for ks
        
        table[:, 0] = kv[:]
        for foo in range(nk):
            if spectrum == 'pk':
                table[foo, 1:] = self.p_integrals(kv[foo],nmax=nmax)

            if spectrum == 'vk':
                table[foo, 1:] = self.v_integrals(kv[foo],nmax=nmax)
                
            if spectrum == 'sk':
                table[foo, 1:] = self.s_integrals(kv[foo],nmax=nmax)
            
                
        # store a copy in pktables dictionary
        return table
        

    def make_multipoles_tables(self, ells, f, spectrum='vk',\
                               apar=1, aperp=1, kv=None, kmin=1e-2, kmax=0.25, nk = 50, nmax=5):

        if kv is None:
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            nk = len(kv)
    
        tables = {}
        eft_tables = {}
        
        table_nu = np.zeros((len(self.nus),nk,self.num_power_components-1))
        # zeros for counterterms and stochastic terms. pk has a k^2 and vk has kmu which are not isotropoic
        eft_table_nu = np.zeros((\
                                   len(self.nus),nk,\
                                 + 3*(spectrum=='vk' or spectrum=='sk')+4*(spectrum=='pk')\
                                 + 1*(spectrum=='sk') + 1*(spectrum=='vk') + 3*(spectrum=='pk')\
                                  ))

        
        #for ell in ells:
        #    tables[ell] = np.zeros((len(self.nus),nk,self.num_power_components))

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
                kmu = ktrue * nu_true
                
                if spectrum == 'pk':
                    pterms = self.p_integrals(ktrue,nmax=nmax)
                    table_nu[ii,jj,:] = pterms[:-1]
                    
                    za = pterms[-1]
                    eft_table_nu[ii,jj,:] = np.concatenate(\
                                                           (za * ktrue**2 * np.array([1, nu_true**2, nu_true**4, nu_true**6]),\
                                                            np.array([1, ktrue**2, ktrue**2 * nu_true**2]))\
                                                           ) # counterterms and stochastic terms
                    
                if spectrum == 'vk':
                    pterms = self.v_integrals(ktrue,nmax=nmax)
                    table_nu[ii,jj,:] = pterms[:-1]
                    
                    za = pterms[-1]
                    eft_table_nu[ii,jj,:] = np.concatenate(\
                                                           (za * ktrue * np.array([nu_true, nu_true**3, nu_true**5]),\
                                                           np.array([ktrue * nu_true]))\
                                                           ) # counterterms and stochastic terms
                    
                if spectrum == 'sk':
                    pterms = self.s_integrals(ktrue,nmax=nmax)
                    table_nu[ii,jj,:] = pterms[:-1]
                    
                    za = pterms[-1]
                    eft_table_nu[ii,jj,:] = np.concatenate(\
                                                           (za *  np.array([1, nu_true**2, nu_true**4]),\
                                                           [1])\
                                                           ) #counterterms

        table_nu[self.ngauss:,:,:] = (-1)**(ells[0]) * np.flip(table_nu[0:self.ngauss],axis=0) # we assume all the ells are the same parity
        eft_table_nu[self.ngauss:,:,:] = (-1)**(ells[0]) * np.flip(eft_table_nu[0:self.ngauss],axis=0)
        
        for ell in ells:
            tables[ell] = 0.5 * (2*ell+1) * np.sum((self.ws*self.Ls[ell])[:,None,None]*table_nu,axis=0) / vol_fac
            eft_tables[ell] = 0.5 * (2*ell+1) * np.sum((self.ws*self.Ls[ell])[:,None,None]*eft_table_nu,axis=0) / vol_fac

        return kv, tables, eft_tables


    def combine_bias_terms(self, bvec, ell, tables, eft_vec=None, eft_tables=None):
    
        '''
        Generic function to return the predicted power spectrum given a pair of bias vector and bias integral multipoles.
        
        The eft_vec should be of form [ct0, ct1, ct2, ..., sn0, sn2, ...] for however many rows are in eft_tables.
        
        '''
    
        b1, b2, bs, b3 = bvec
        bpoly = np.array([1, b1, b1**2,\
                         b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2,
                         b3, b1*b3])
                         
        ret = np.sum(bpoly[None,:] * tables[ell], axis=1)
        
        if eft_vec is not None:
            ret += np.sum(np.array(eft_vec)[None,:] * eft_tables[ell], axis=1)
                 
    
        return ret
