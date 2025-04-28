import numpy as np

import time

from scipy.interpolate import interp1d

from  spinosaurus.Utils.loginterp import loginterp
from  spinosaurus.Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
from  spinosaurus.Utils.qfuncfft import QFuncFFT

from  spinosaurus.cleft_fftw import CLEFT


class DensityShapeCorrelators(CLEFT):
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
        CLEFT.__init__(self, *args, **kw)


        self.num_gd_components = 21
        self.sph_gd = SphericalBesselTransform(self.qint, L=self.jn, ncol=(self.num_gd_components), threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)



    def update_power_spectrum(self,k,p):
        '''
        Same as the one in cleft_fftw but also do the velocities.
        '''
        super(DensityShapeCorrelators,self).update_power_spectrum(k,p)
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

    def gd_integrals(self,k):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        ksq = k**2; kcu = k**3
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*ksq *self.sigma)
        
        ret = np.zeros(self.num_gd_components)
        
        bias_integrands = np.zeros( (self.num_gd_components,self.N)  )
                
        for l in range(self.jn):
            # l-dep functions
            B = ksq * self.Ylin
            mu1fac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/B
            mu3fac = mu1fac * (1. - 2.*(l-1)/B) # mu3 terms start at j1 so l -> l-1
            mu4fac = 1 - 4*l/B + 4*l*(l-1)/B**2
            mu5fac = mu1fac * (1 - 4*(l-1)/B + 4*(l-1)*(l-2)/B**2)
            mu6fac = 1 - 6*l/B +12*l*(l-1)/B**2 - 8*l*(l-1)*(l-2)/B**3

            
            # Alternatively in the Pell basis
            P1fac = mu1fac
            P2fac = 0.5 * (3*mu2fac - 1 ); mu2P2fac = 0.5 * (3*mu4fac - mu2fac)
            P3fac = 0.5 * (5*mu3fac - 3*mu1fac); mu2P3fac = 0.5 * (5*mu5fac - 3*mu3fac)
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3)
            
            #bias_integrands[0,:] = - 4./3 * k*  self.J3 * mu1fac -  k * self.J4 * (mu3fac - mu1fac/3) # (1, cs)
            #bias_integrands[0,:] = - k*( (self.J2+2*self.J3)*mu1fac + self.J4*mu3fac )
            bias_integrands[0,:] = - k * (-4./15*self.xi1m1*mu1fac + 0.4*self.xi3m1*P3fac)\
                                  + 0.5 * k**3 * (-4./15*self.xi1m1 * (self.Xlin_gt*mu1fac + self.Ylin_gt*mu3fac)\
                                                  + 0.4* self.xi3m1 * (self.Xlin_gt*P3fac + self.Ylin_gt*mu2P3fac)   )\
                                   + k * (2*self.xi1m1_R1*mu1fac - self.xi3m1_R1*P3fac)\
                                   - 0.5 * ksq * (self.xi4_sDD*P4fac + (-2*self.xi2A_sDD+self.xi2B_sDD)*P2fac -2*self.xi0_sDD)

            
            # (1, delta s)
            bias_integrands[1,:] = - ksq * self.Ulin * ( (self.J2 + 2*self.J3) * mu2fac + self.J4 * mu4fac ) \
                                    + k * (2*self.xi1m1_2ds*mu1fac - self.xi3m1_2ds*P3fac)
            
            # (1, s^2_{ij})
            bias_integrands[2,:] = -ksq * ( self.J3**2 + \
                                            (self.J2**2 + 4*self.J2*self.J3 + 3*self.J3**2 + 2*self.J3*self.J4) * mu2fac + \
                                            (2*self.J2*self.J4 + 2*self.J3*self.J4 + self.J4**2) * mu4fac - 1./6 * (self.Xs2 + mu2fac*self.Ys2) )\
                                  + k * (2*self.xi3m1_2s2*mu1fac - self.xi1m1_2s2*P3fac)
            
            # (1, L2_{ij})
            bias_integrands[3,:] = k * (2*self.xi3m1_2L2*mu1fac - self.xi1m1_2L2*P3fac) \
                                     - 0.5 * ksq * (self.xi4_L2DD*P4fac + (-2*self.xi2A_L2DD+self.xi2B_L2DD)*P2fac -2*self.xi0_L2DD)
                                  
            # (delta, sij)
            #bias_integrands[4,:] = -2./3*P2fac * self.J1 - ksq * self.Ulin * ( (self.J2 + 2*self.J3) * mu2fac + self.J4 * mu4fac )\
            #                         + k * (2*self.xi1_deltasD*mu1fac - self.xi3_deltasD*P3fac)
            bias_integrands[4,:] = -2./3 * ( (1 - 0.5*ksq*self.Xlin_gt)*P2fac - 0.5 * ksq*self.Ylin_gt*mu2P2fac ) * self.xi20\
                                    - ksq * self.Ulin * ( (self.J2 + 2*self.J3) * mu2fac + self.J4 * mu4fac )\
                                     + k * (2*self.xi1_deltasD*mu1fac - self.xi3_deltasD*P3fac)
            
            # (delta, delta sij)
            bias_integrands[5,:] = -k*mu1fac*self.delta_deltasab_mu1 - k*mu3fac*self.delta_deltasab_mu3
            
            # (delta, s^2_ij)
            bias_integrands[6,:] = -k*mu1fac*self.delta_s2ab_mu1 - k*mu3fac*self.delta_s2ab_mu3

            # (delta, tij)
            bias_integrands[7,:] = k * (2*self.xi1_deltaL2D*mu1fac - self.xi3_deltaL2D*P3fac)

            # (delta^2, sij)
            bias_integrands[8,:] = -2./3 * k * (mu1fac - 3*mu3fac) * self.J1 * self.Ulin
            
            # (delta^2, delta sij)
            bias_integrands[9,:] = self.delta2_deltasab_p2 * P2fac
            
            # (delta^2, s^2_ij)
            bias_integrands[10,:] = self.delta2_s2sab_p2 * P2fac
            
            # (delta^2, tij)
            bias_integrands[11,:] = self.delta2_L2ab_p2 * P2fac
            
            # (s^2, sij)
            bias_integrands[12,:] = -k*mu1fac*self.s2_sab_mu1 - k*mu3fac*self.s2_sab_mu3

            # (s^2, delta sij)
            bias_integrands[13,:] = self.s2_deltasab_p2 * P2fac
            
            # (s^2, s^2_ij)
            bias_integrands[14,:] = self.s2_s2ab_p2 * P2fac
            
            # (s^2, tij)
            bias_integrands[15,:] = self.s2_L2ab_p2 * P2fac
            
            # Cubic bias terms
            
            # R1 terms
            bias_integrands[16,:] = - k * (-4./15*self.xi1m1_b3*mu1fac + 0.4*self.xi3m1_b3*P3fac) # (1, O^3 ab)
            
            bias_integrands[17,:] = -2./3 * self.xi20_b3 * P2fac # (O^3, sab) or (delta, O^3_ab)
            
            # Rdt terms
            bias_integrands[18,:] = - k * (-4./15*self.xi1m1_dt*mu1fac + 0.4*self.xi3m1_dt*P3fac) # (1, O^3 ab)
            
            bias_integrands[19,:] = -2./3 * self.xi20_dt * P2fac # (delta, O^3_ab)
            
            # Pure linear term for counterterm
            bias_integrands[20,:] = -2./3 * ksq * self.xi20 * P2fac
            
            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gd.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)

        
        return 4*suppress*np.pi*ret
    
    def make_gdtable(self, kmin = 1e-3, kmax = 3, nk = 100):
    
        self.pktable_gd = np.zeros([nk, 1+self.num_gd_components])
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable_gd[:, 0] = kv[:]
        for foo in range(nk):
            self.pktable_gd[foo,1:] = self.gd_integrals(kv[foo])
        
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
    
