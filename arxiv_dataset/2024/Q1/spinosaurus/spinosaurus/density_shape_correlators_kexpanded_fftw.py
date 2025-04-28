import numpy as np

import time

from scipy.interpolate import interp1d

from  spinosaurus.Utils.loginterp import loginterp
from  spinosaurus.Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
from  spinosaurus.Utils.qfuncfft import QFuncFFT

from  spinosaurus.cleft_kexpanded_fftw import KECLEFT


class KEDensityShapeCorrelators(KECLEFT):
    '''
    Class based on cleft_fftw to compute density shape correlations.
    This is a placeholder for now.
    '''

    def __init__(self, *args, beyond_gauss = False, **kw):
        '''
        If beyond_gauss = True computes the third and fourth moments, otherwise
        default is to enable calculation of P(k), v(k) and sigma(k).
        
        Other keywords the same as the cleft_fftw class. Go look there!
        '''
        
        # Set up the configuration space quantities
        KECLEFT.__init__(self, *args, **kw)


        self.num_gd_components = 21
        self.sph_gd = SphericalBesselTransform(self.qint, L=self.jn, ncol=(self.num_gd_components), threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)



    def update_power_spectrum(self,k,p):
        '''
        Same as the one in cleft_fftw but also do the velocities.
        '''
        super(KEDensityShapeCorrelators,self).update_power_spectrum(k,p)
        self.setup_shape_correlators()


    def setup_shape_correlators(self):
        '''
        Set up Lagrangian correlators involved in shapes.
        '''
        self.J1 = self.qf.J1
        self.J2 = self.qf.J2
        self.J3 = self.qf.J3
        self.J4 = self.qf.J4
        
        self.xi20 = self.qf.xi20
        self.xi1m1 = self.qf.xi1m1
        self.xi3m1 = self.qf.xi3m1
        
        self.xi1m1_R1 = self.qf.xi1m1_R1
        self.xi3m1_R1 = self.qf.xi3m1_R1
        
        self.xi1m1_2ds = self.qf.xi1m1_2ds
        self.xi3m1_2ds = self.qf.xi3m1_2ds
        
        self.xi1m1_2s2 = self.qf.xi1m1_2s2
        self.xi3m1_2s2 = self.qf.xi3m1_2s2
        
        self.xi1m1_2L2 = self.qf.xi1m1_2L2
        self.xi3m1_2L2 = self.qf.xi3m1_2L2
        
        self.xi0_sDD = self.qf.xi0_sDD
        self.xi2A_sDD = self.qf.xi2A_sDD
        self.xi2B_sDD = self.qf.xi2B_sDD
        self.xi4_sDD = self.qf.xi4_sDD

        self.xi0_L2DD = self.qf.xi0_L2DD
        self.xi2A_L2DD = self.qf.xi2A_L2DD
        self.xi2B_L2DD = self.qf.xi2B_L2DD
        self.xi4_L2DD = self.qf.xi4_L2DD

        self.s2_sab_mu1 = 4./1575 * (-28*self.qf.xi00*self.qf.xi1m1 + 10*self.qf.xi20*self.qf.xi1m1 -63*self.qf.xi00*self.qf.xi3m1\
                                    + 60*self.qf.xi20*self.qf.xi3m1 + 108*self.qf.xi40*self.qf.xi1m1 + 18*self.qf.xi40*self.qf.xi3m1)
        self.s2_sab_mu3 = 4./1575 * (30*self.qf.xi20*self.qf.xi1m1 + 105*self.qf.xi00*self.qf.xi3m1 - 120*self.qf.xi20*self.qf.xi3m1\
                                    -180*self.qf.xi40*self.qf.xi1m1 + 90*self.qf.xi40*self.qf.xi3m1)
        
        self.delta_deltasab_mu1 = 1./15 * (-4*self.qf.xi00*self.qf.xi1m1 - 5*self.qf.xi20*self.qf.xi1m1 - 9*self.qf.xi00*self.qf.xi3m1)
        self.delta_deltasab_mu3 = self.qf.xi00*self.qf.xi3m1 + self.qf.xi20*self.qf.xi1m1
        
        self.delta_s2ab_mu1 = 2./45 * self.qf.xi20 * (self.qf.xi1m1 + 6*self.qf.xi3m1)
        self.delta_s2ab_mu3 = 2./45 * self.qf.xi20 * (3*self.qf.xi1m1 - 12*self.qf.xi3m1)
        
        self.delta2_deltasab_p2 = -4./3 * self.qf.xi00 * self.qf.xi20

        self.delta2_s2sab_p2 = 4./9 * self.qf.xi20**2
        
        self.s2_deltasab_p2 = 8./315 * self.qf.xi20 * (-7*self.qf.xi00 + 10*self.qf.xi20 -18*self.qf.xi40)
        
        self.s2_s2ab_p2 = - 8./6615 * (98*self.qf.xi00*self.qf.xi20 + 15*self.qf.xi20**2 + 72*self.qf.xi20*self.qf.xi40 - 90*self.qf.xi40**2)
        
        self.delta2_L2ab_p2 = 2./3 * 3./7 * self.qf.xi2_Q8 # 2/3 comes from kk - 1/3 delta -> P2(k)
        self.s2_L2ab_p2 = 2./3 * 1./7 * self.qf.xi2_Qs2
        
        self.xi3_deltasD = self.qf.xi3_deltasD
        self.xi1_deltasD = self.qf.xi1_deltasD
        
        self.xi3_deltaL2D = self.qf.xi3_deltaL2D
        self.xi1_deltaL2D = self.qf.xi1_deltaL2D
        
        self.xi1m1_b3 = self.qf.xi1m1_b3
        self.xi3m1_b3 = self.qf.xi3m1_b3
        self.xi20_b3 = self.qf.xi20_b3

        self.xi1m1_dt = self.qf.xi1m1_dt
        self.xi3m1_dt = self.qf.xi3m1_dt
        self.xi20_dt = self.qf.xi20_dt


    def compute_pgd_linear(self):
        self.p_linear_gd = np.zeros( (self.num_gd_components, self.N) )
        self.p_linear_gd[0,:] = 2./3 * self.pint
        self.p_linear_gd[4,:] = 2./3 * self.pint

    def compute_pgd_connected(self):
        self.p_connected_gd = np.zeros( (self.num_gd_components, self.N) )
    
        self.p_connected_gd[16,:] = 2./3 * self.qf.Rb3
        self.p_connected_gd[17,:] = 2./3 * self.qf.Rb3
        
        self.p_connected_gd[18,:] = 2./3 * self.qf.Adt
        self.p_connected_gd[19,:] = 2./3 * self.qf.Adt
        
        self.p_connected_gd[20,:] = self.kint**2 * self.pint

    def compute_pgd_k0(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgd_0 = np.zeros( (self.num_gd_components, self.N) )
        ret = np.zeros(self.num_gd_components)
        
        bias_integrands = np.zeros( (self.num_gd_components,self.N)  )
                
        for l in range(3):
            # l-dep functions
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)
            mu4fac = 0.2 * (l==0) - 4./7*(l==2) + 8./35*(l==4)
            mu5fac = (27. * (l==1) - 28. * (l==3) + 8. * (l==5))/63.
            mu6fac = (33 * (l==0) - 110 * (l==2) + 72 * (l==4) - 16 * (l==6))/231.

            # Alternatively in the Pell basis
            P1fac = mu1fac
            P2fac = 0.5 * (3*mu2fac - mu0fac ); mu2P2fac = 0.5 * (3*mu4fac - mu2fac)
            P3fac = 0.5 * (5*mu3fac - 3*mu1fac); mu2P3fac = 0.5 * (5*mu5fac - 3*mu3fac)
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac)
                        
            # (delta^2, delta sij)
            bias_integrands[9,:] = self.delta2_deltasab_p2 * P2fac
            
            # (delta^2, s^2_ij)
            bias_integrands[10,:] = self.delta2_s2sab_p2 * P2fac
            
            # (delta^2, tij)
            bias_integrands[11,:] = self.delta2_L2ab_p2 * P2fac
            
            # (s^2, delta sij)
            bias_integrands[13,:] = self.s2_deltasab_p2 * P2fac
            
            # (s^2, s^2_ij)
            bias_integrands[14,:] = self.s2_s2ab_p2 * P2fac
            
            # (s^2, tij)
            bias_integrands[15,:] = self.s2_L2ab_p2 * P2fac
            
            # Cubic bias terms
            
            # R1 terms
            #bias_integrands[17,:] = -2./3 * self.xi20_b3 * P2fac # (O^3, sab) or (delta, O^3_ab)
            
            # Rdt terms
            #bias_integrands[19,:] = -2./3 * self.xi20_dt * P2fac # (delta, O^3_ab)
            
            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gd.sph(l, bias_integrands)
            self.pgd_0 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    
    def compute_pgd_k1(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgd_1 = np.zeros( (self.num_gd_components, self.N) )
        
        bias_integrands = np.zeros( (self.num_gd_components,self.N)  )
                
        for l in range(4):
            # l-dep functions
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)
            mu4fac = 0.2 * (l==0) - 4./7*(l==2) + 8./35*(l==4)
            mu5fac = (27. * (l==1) - 28. * (l==3) + 8. * (l==5))/63.
            mu6fac = (33 * (l==0) - 110 * (l==2) + 72 * (l==4) - 16 * (l==6))/231.

            # Alternatively in the Pell basis
            P1fac = mu1fac
            P2fac = 0.5 * (3*mu2fac - mu0fac ); mu2P2fac = 0.5 * (3*mu4fac - mu2fac)
            P3fac = 0.5 * (5*mu3fac - 3*mu1fac); mu2P3fac = 0.5 * (5*mu5fac - 3*mu3fac)
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac)
            
            #bias_integrands[0,:] = - 4./3 * k*  self.J3 * mu1fac -  k * self.J4 * (mu3fac - mu1fac/3) # (1, cs)
            #bias_integrands[0,:] = - k*( (self.J2+2*self.J3)*mu1fac + self.J4*mu3fac )
            bias_integrands[0,:] = 2*self.xi1m1_R1*mu1fac - self.xi3m1_R1*P3fac
            
            # (1, delta s)
            bias_integrands[1,:] = 2*self.xi1m1_2ds*mu1fac - self.xi3m1_2ds*P3fac
            
            # (1, s^2_{ij})
            bias_integrands[2,:] = 2*self.xi3m1_2s2*mu1fac - self.xi1m1_2s2*P3fac
            
            # (1, L2_{ij})
            bias_integrands[3,:] = 2*self.xi3m1_2L2*mu1fac - self.xi1m1_2L2*P3fac
                                  
            # (delta, sij)
            #bias_integrands[4,:] = -2./3*P2fac * self.J1 - ksq * self.Ulin * ( (self.J2 + 2*self.J3) * mu2fac + self.J4 * mu4fac )\
            #                         + k * (2*self.xi1_deltasD*mu1fac - self.xi3_deltasD*P3fac)
            bias_integrands[4,:] = 2*self.xi1_deltasD*mu1fac - self.xi3_deltasD*P3fac
            
            # (delta, delta sij)
            bias_integrands[5,:] = -mu1fac*self.delta_deltasab_mu1 - mu3fac*self.delta_deltasab_mu3
            
            # (delta, s^2_ij)
            bias_integrands[6,:] = -mu1fac*self.delta_s2ab_mu1 - mu3fac*self.delta_s2ab_mu3

            # (delta, tij)
            bias_integrands[7,:] = 2*self.xi1_deltaL2D*mu1fac - self.xi3_deltaL2D*P3fac

            # (delta^2, sij)
            bias_integrands[8,:] = -2./3  * (mu1fac - 3*mu3fac) * self.J1 * self.Ulin
            
            # (s^2, sij)
            bias_integrands[12,:] = -mu1fac*self.s2_sab_mu1 - mu3fac*self.s2_sab_mu3
            
            # Cubic bias terms
            
            # R1 terms
            #bias_integrands[16,:] = - (-4./15*self.xi1m1_b3*mu1fac + 0.4*self.xi3m1_b3*P3fac) # (1, O^3 ab)

            # Rdt terms
            #bias_integrands[18,:] = - (-4./15*self.xi1m1_dt*mu1fac + 0.4*self.xi3m1_dt*P3fac) # (1, O^3 ab)

            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gd.sph(l, bias_integrands)
            self.pgd_1 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    def compute_pgd_k2(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgd_2 = np.zeros( (self.num_gd_components, self.N) )
        
        bias_integrands = np.zeros( (self.num_gd_components,self.N)  )
                
        for l in range(5):
            # l-dep functions
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)
            mu4fac = 0.2 * (l==0) - 4./7*(l==2) + 8./35*(l==4)
            mu5fac = (27. * (l==1) - 28. * (l==3) + 8. * (l==5))/63.
            mu6fac = (33 * (l==0) - 110 * (l==2) + 72 * (l==4) - 16 * (l==6))/231.

            # Alternatively in the Pell basis
            P1fac = mu1fac
            P2fac = 0.5 * (3*mu2fac - mu0fac); mu2P2fac = 0.5 * (3*mu4fac - mu2fac)
            P3fac = 0.5 * (5*mu3fac - 3*mu1fac); mu2P3fac = 0.5 * (5*mu5fac - 3*mu3fac)
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac)
            
            #bias_integrands[0,:] = - 4./3 * k*  self.J3 * mu1fac -  k * self.J4 * (mu3fac - mu1fac/3) # (1, cs)
            #bias_integrands[0,:] = - k*( (self.J2+2*self.J3)*mu1fac + self.J4*mu3fac )
            bias_integrands[0,:] = - 0.5 * (self.xi4_sDD*P4fac + (-2*self.xi2A_sDD+self.xi2B_sDD)*P2fac -2*self.xi0_sDD*mu0fac)
            
            # (1, delta s)
            bias_integrands[1,:] = - self.Ulin * ( (self.J2 + 2*self.J3) * mu2fac + self.J4 * mu4fac )
            
            # (1, s^2_{ij})
            bias_integrands[2,:] = - (self.J3**2 * mu0fac + \
                                            (self.J2**2 + 4*self.J2*self.J3 + 3*self.J3**2 + 2*self.J3*self.J4) * mu2fac + \
                                            (2*self.J2*self.J4 + 2*self.J3*self.J4 + self.J4**2) * mu4fac - 1./6 * (self.Xs2*mu0fac + mu2fac*self.Ys2) )
            
            # (1, L2_{ij})
            bias_integrands[3,:] = - 0.5 * (self.xi4_L2DD*P4fac + (-2*self.xi2A_L2DD+self.xi2B_L2DD)*P2fac -2*self.xi0_L2DD*mu0fac)
                                  
            # (delta, sij)
            bias_integrands[4,:] = -2./3 * ( -0.5*self.Xlin*P2fac - 0.5*self.Ylin*mu2P2fac ) * self.xi20\
                                    - self.Ulin * ( (self.J2 + 2*self.J3) * mu2fac + self.J4 * mu4fac )
            
            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gd.sph(l, bias_integrands)
            self.pgd_2 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    def compute_pgd_k3(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgd_3 = np.zeros( (self.num_gd_components, self.N) )
        
        bias_integrands = np.zeros( (self.num_gd_components,self.N)  )
                
        for l in range(6):
            # l-dep functions
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)
            mu4fac = 0.2 * (l==0) - 4./7*(l==2) + 8./35*(l==4)
            mu5fac = (27. * (l==1) - 28. * (l==3) + 8. * (l==5))/63.
            mu6fac = (33 * (l==0) + 110 * (l==2) + 72 * (l==4) + 16 * (l==6))/231.

            # Alternatively in the Pell basis
            P1fac = mu1fac
            P2fac = 0.5 * (3*mu2fac - mu0fac ); mu2P2fac = 0.5 * (3*mu4fac - mu2fac)
            P3fac = 0.5 * (5*mu3fac - 3*mu1fac); mu2P3fac = 0.5 * (5*mu5fac - 3*mu3fac)
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac)
            
            #bias_integrands[0,:] = - 4./3 * k*  self.J3 * mu1fac -  k * self.J4 * (mu3fac - mu1fac/3) # (1, cs)
            #bias_integrands[0,:] = - k*( (self.J2+2*self.J3)*mu1fac + self.J4*mu3fac )
            bias_integrands[0,:] = + 0.5 * (-4./15*self.xi1m1 * (self.Xlin*mu1fac + self.Ylin*mu3fac)\
                                                  + 0.4* self.xi3m1 * (self.Xlin*P3fac + self.Ylin*mu2P3fac)   )
            
                                                  
            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gd.sph(l, bias_integrands)
            self.pgd_3 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)


    def make_gdtable(self, D=1, nonlinear=True, kmin = 1e-3, kmax = 3, nk = 100):
    
        self.pktable_gd = np.zeros([nk, self.num_gd_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_gd[:, 0] = kv[:]
        for ii in range(self.num_gd_components):
            self.pktable_gd[:, ii+1] = interp1d(self.kint,\
                                             D**2 * self.p_linear_gd[ii,:] \
                                            + D**2 * nonlinear * (self.p_connected_gd[ii,:] + self.pgd_0[ii,:] \
                                            + self.kint * self.pgd_1[ii,:] + self.kint**2 * self.pgd_2[ii,:]\
                                            + self.kint**3 * self.pgd_3[ii,:]) )(kv)
                                                        
        return self.pktable_gd
        
    def combine_bias_terms_density_shape(self, density_bvec, shape_bvec, Pnoise=0):

        '''
        Compute the galaxy density x shape cross spectrum.
         
        Note that there is no shot noise term.
         
        The counterterm treatment is a bit ad-hoc, but we're just trying to be a bit symmetric in the inputs (really the cross counterterm is its own thing).

        Inputs:
            -density_bvec: scalar bias parameters of sample
            -shape_bvec: shape bias parameters of sample 
            -Pshot: constant shape noise amplitude, if auto-spectrum
            -sn21: first k2-dependent shot noise contribution, if auto-spectrum
            -sn22: second k2-dependent shot noise contribution, if auto-spectrum

        Outputs:
            -k: k scales at which LPT predictions have been computed
            -Pk: bias polynomial combination of parameters times basis spectra 
        
        '''


        b1, b2, bs, b3, alpha_d = density_bvec; b2 *= 0.5 # we use the 1/2 b2 delta^2 convention for some reason
        c_s, c_ds, c_s2, c_L2, c_3, c_dt, alpha_s = shape_bvec

        # The table is listed in order (1, Oab), (delta, Oab), (s2, Oab)
        bias_poly = np.array([c_s, c_ds, c_s2, c_L2,\
                              b1 * c_s, b1 * c_ds, b1 * c_s2, b1 * c_L2,\
                              b2 * c_s, b2 * c_ds, b2 * c_s2, b2 * c_L2,\
                              bs * c_s, bs * c_ds, bs * c_s2, bs * c_L2,\
                              c_3, b1 * c_3 + b3 * c_s, c_dt, b1 * c_dt, alpha_d + alpha_s])
        # If shape x density for same sample, there may be scale-dependent noise (see section 3.4)
        noiseterm = self.pktable_gd[:,0]**2 * Pnoise
        return self.pktable_gd[:,0], np.sum(bias_poly[None,:] * self.pktable_gd[:,1:], axis = 1) + noiseterm

