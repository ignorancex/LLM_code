import numpy as np

import time

from scipy.interpolate import interp1d

from  spinosaurus.Utils.loginterp import loginterp
from  spinosaurus.Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
from  spinosaurus.Utils.qfuncfft import QFuncFFT

from  spinosaurus.density_shape_correlators_kexpanded_fftw import KEDensityShapeCorrelators


class KEShapeShapeCorrelators(KEDensityShapeCorrelators):
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
        KEDensityShapeCorrelators.__init__(self, *args, **kw)


        self.num_gg_components = 13
        self.sph_gg = SphericalBesselTransform(self.qint, L=self.jn, ncol=(self.num_gg_components), threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)



    def update_power_spectrum(self,k,p):
        '''
        Same as the one in cleft_fftw but also do the velocities.
        '''
        super(KEShapeShapeCorrelators,self).update_power_spectrum(k,p)
        self.setup_shape_correlators()
        
        self.pktables_gg = {}


    def setup_shape_correlators(self):
        '''
        Set up Lagrangian correlators involved in shapes.
        '''
        self.xi00 = self.qf.xi00
        self.xi20 = self.qf.xi20
        self.xi40 = self.qf.xi40
        
        self.J1 = self.qf.J1
        self.J2 = self.qf.J2
        self.J3 = self.qf.J3
        self.J4 = self.qf.J4
        self.J6 = self.qf.J6
        self.J7 = self.qf.J7
        self.J8 = self.qf.J8
        self.J9 = self.qf.J9

        # Components of (s, s)
        self.xi5_ssD  = self.qf.xi5_ssD
        self.xi3A_ssD = self.qf.xi3A_ssD
        self.xi3B_ssD = self.qf.xi3B_ssD
        self.xi1A_ssD = self.qf.xi1A_ssD
        self.xi1B_ssD = self.qf.xi1B_ssD
        
        self.s_s_0_P1 = 3*self.xi1A_ssD + 3./2*self.xi1B_ssD
        self.s_s_0_P3 = - 1.5*self.xi3B_ssD
        self.s_s_0_P5 = -1.5*self.xi5_ssD
        
        self.s_s_1_P1 = -6*self.xi1A_ssD-2.5*self.xi1B_ssD
        self.s_s_1_P3 = 2*self.xi3A_ssD-0.5*self.xi3B_ssD
        self.s_s_1_P5 = - 2*self.xi5_ssD
        
        self.s_s_2_P1 = 6*self.xi1A_ssD+self.xi1B_ssD
        self.s_s_2_P3 = 2*self.xi3A_ssD+self.xi3B_ssD
        self.s_s_2_P5 = - 0.5*self.xi5_ssD
        
        # Components of (s, L2)
        self.xi5_sL2D  = self.qf.xi5_sL2D
        self.xi3A_sL2D = self.qf.xi3A_sL2D
        self.xi3B_sL2D = self.qf.xi3B_sL2D
        self.xi1A_sL2D = self.qf.xi1A_sL2D
        self.xi1B_sL2D = self.qf.xi1B_sL2D
        
        self.s_L2_0_P1 = 3*self.xi1A_sL2D + 3./2*self.xi1B_sL2D
        self.s_L2_0_P3 = - 1.5*self.xi3B_sL2D
        self.s_L2_0_P5 = -1.5*self.xi5_sL2D
        
        self.s_L2_1_P1 = -6*self.xi1A_sL2D-2.5*self.xi1B_sL2D
        self.s_L2_1_P3 = 2*self.xi3A_sL2D-0.5*self.xi3B_sL2D
        self.s_L2_1_P5 = - 2*self.xi5_sL2D
        
        self.s_L2_2_P1 = 6*self.xi1A_sL2D+self.xi1B_sL2D
        self.s_L2_2_P3 = 2*self.xi3A_sL2D+self.xi3B_sL2D
        self.s_L2_2_P5 = - 0.5*self.xi5_sL2D
        
        
        # Components of (s_ab, delta s_cd) (these are multiplied by k, and are the imaginary parts of i <Delta_i s sd>
        self.s_ds_0_P1 = -2./105 * (self.J1*(28*self.J3 + 11*self.J4) - (105*self.J6 + 98*self.J8 + 11*self.J9)*self.Ulin)
        self.s_ds_0_P3 = -4./15 * (self.J1*(3*self.J3 + self.J4) - (3*self.J8 + self.J9)*self.Ulin)
        self.s_ds_0_P5 = -4./21 * (self.J1*self.J4 - self.J9*self.Ulin)
        
        self.s_ds_1_P1 = 4./35 * (self.J1*(7*self.J3 + 3*self.J4) - (35*self.J6 + 28*self.J8 + 3*self.J9)*self.Ulin)
        self.s_ds_1_P3 = -4./45 * (9*self.J1*self.J3 + self.J1*self.J4 + 9*self.J8*self.Ulin - self.J9*self.Ulin)
        self.s_ds_1_P5 = -16./63 * (self.J1*self.J4 - self.J9*self.Ulin)

        self.s_ds_2_P1 = -4./35 * (self.J1*self.J4 - (35*self.J6 + 14*self.J8 + self.J9)*self.Ulin)
        self.s_ds_2_P3 = 8./45 * (self.J1*self.J4 - (9*self.J8 + self.J9)*self.Ulin)
        self.s_ds_2_P5 = -4./63 * (self.J1*self.J4 - self.J9*self.Ulin)
        
        # Components of (s_ab, s^2_cd) (these are multiplied by k, and are the imaginary parts of i <Delta_i s sd>
        self.s_s2_0_P1 = 4./105 * (self.J4*(49*self.J6 + 11*self.J7 + 71*self.J8 + 11*self.J9) + self.J2*(105*self.J6 + 98*self.J8 + 11*self.J9) + self.J3*(140*self.J6 + 28*self.J7 + 25*(7*self.J8 + self.J9)))
        self.s_s2_0_P3 = 4./15 * (6*self.J3*self.J7 + 2*self.J2*(3*self.J8 + self.J9) + 5*self.J3*(3*self.J8 + self.J9) + self.J4*(3*self.J6 + 2*self.J7 + 7*self.J8 + 2*self.J9))
        self.s_s2_0_P5 = 8./21 * ((self.J2 + self.J3)*self.J9 + self.J4*(self.J7 + 2*self.J8 + self.J9))
        
        self.s_s2_1_P1 = -4./35 * (2*self.J4*(14*self.J6 + 3*self.J7 + 20*self.J8 + 3*self.J9) + self.J2*(70*self.J6 + 56*self.J8 + 6*self.J9) + self.J3*(70*self.J6 + 14*self.J7 + 91*self.J8 + 13*self.J9))
        self.s_s2_1_P3 = -4./45 * (self.J4*(9*self.J6 - 2*self.J7 + 5*self.J8 - 2*self.J9) + 2*self.J2*(9*self.J8 - self.J9) - self.J3*(18*self.J7 + 27*self.J8 + 11*self.J9))
        self.s_s2_1_P5 = 32./63 * ((self.J2 + self.J3)*self.J9 + self.J4*(self.J7 + 2*self.J8 + self.J9))

        self.s_s2_2_P1 = 8./35 * (self.J3*(7*self.J8 + self.J9) + self.J4*(7*self.J6 + self.J7 + 9*self.J8 + self.J9) + self.J2*(35*self.J6 + 14*self.J8 + self.J9))
        self.s_s2_2_P3 = -8./45 * (9*self.J4*self.J6 + 2*self.J4*self.J7 + 18*self.J2*self.J8 + 9*self.J3*self.J8 + 13*self.J4*self.J8 + 2*(self.J2 + self.J3 + self.J4)*self.J9)
        self.s_s2_2_P5 = 8./63 * ((self.J2 + self.J3)*self.J9 + self.J4*(self.J7 + 2*self.J8 + self.J9))

        # Components of (delta s_ab, delta s_cd)
        self.ds_ds_0_P0 = (2*(self.J1**2 + self.corlin*(15*self.J6 + 10*self.J8 + self.J9)))/15.
        self.ds_ds_0_P2 = (4*(self.J1**2 + self.corlin*(7*self.J8 + self.J9)))/21.
        self.ds_ds_0_P4 = (12*(self.J1**2 + self.corlin*self.J9))/35.
        
        self.ds_ds_1_P0 = (-4*(self.J1**2 + self.corlin*(15*self.J6 + 10*self.J8 + self.J9)))/15.
        self.ds_ds_1_P2 = (-4*(self.J1**2 + self.corlin*(7*self.J8 + self.J9)))/21.
        self.ds_ds_1_P4 = (16*(self.J1**2 + self.corlin*self.J9))/35.
        
        self.ds_ds_2_P0 = (4*(self.J1**2 + self.corlin*(15*self.J6 + 10*self.J8 + self.J9)))/15.
        self.ds_ds_2_P2 = (-8*(self.J1**2 + self.corlin*(7*self.J8 + self.J9)))/21.
        self.ds_ds_2_P4 = (4*(self.J1**2 + self.corlin*self.J9))/35.
        
        # Components of (delta s_ab, s^2_cd)
        self.ds_s2_0_P0 = 4./45 * self.xi20**2
        self.ds_s2_0_P2 = -4./2205 * self.xi20 * (49*self.xi00 + 15*self.xi20 + 36*self.xi40)
        self.ds_s2_0_P4 = 8./245 * self.xi20 * (2*self.xi20 - 5*self.xi40)
        
        self.ds_s2_1_P0 = (-8*self.xi20**2)/45.
        self.ds_s2_1_P2 = (4*self.xi20*(49*self.xi00 + 15*self.xi20 + 36*self.xi40))/2205.
        self.ds_s2_1_P4 = (32*self.xi20*(2*self.xi20 - 5*self.xi40))/735.
        
        self.ds_s2_2_P0 = (8*self.xi20**2)/45.
        self.ds_s2_2_P2 = (8*self.xi20*(49*self.xi00 + 15*self.xi20 + 36*self.xi40))/2205.
        self.ds_s2_2_P4 = (8*self.xi20*(2*self.xi20 - 5*self.xi40))/735.
        
        # Components of (s^2_ab, s^2_cd)
        #self.s2_s2_0_P0 = (2*(7*self.xi00**2 + 80*self.xi20**2 + 648*self.xi40**2))/14175.
        #self.s2_s2_0_P2 = (4*(98*self.xi00*self.xi20 + 475*self.xi20**2 + 432*self.xi20*self.xi40 + 3240*self.xi40**2))/138915.
        #self.s2_s2_0_P4 = (8*(85*self.xi20**2 + 2*(49*self.xi00 + 50*self.xi20)*self.xi40 + 162*self.xi40**2))/25725.

        #self.s2_s2_1_P0 = (-4*(7*self.xi00**2 + 80*self.xi20**2 + 648*self.xi40**2))/14175.
        #self.s2_s2_1_P2 = (-4*(98*self.xi00*self.xi20 + 475*self.xi20**2 + 432*self.xi20*self.xi40 + 3240*self.xi40**2))/138915.
        #self.s2_s2_1_P4 = (32*(85*self.xi20**2 + 2*(49*self.xi00 + 50*self.xi20)*self.xi40 + 162*self.xi40**2))/77175.

        #self.s2_s2_2_P0 = (4*(7*self.xi00**2 + 80*self.xi20**2 + 648*self.xi40**2))/14175.
        #self.s2_s2_2_P2 =  (-8*(98*self.xi00*self.xi20 + 475*self.xi20**2 + 432*self.xi20*self.xi40 + 3240*self.xi40**2))/138915.
        #self.s2_s2_2_P4 = (8*(85*self.xi20**2 + 2*(49*self.xi00 + 50*self.xi20)*self.xi40 + 162*self.xi40**2))/77175.
        
        # Components of (s^2_ab, s^2_cd)
        self.s2_s2_0_P0 = (2*(49*self.xi00**2 - 15*self.xi20**2 + 36*self.xi40**2))/4725.
        self.s2_s2_0_P2 = (4*(147*self.xi00*self.xi20 + 205*self.xi20**2 - 612*self.xi20*self.xi40 - 180*self.xi40**2))/46305.
        self.s2_s2_0_P4 = (8*(85*self.xi20**2 + 2*(49*self.xi00 + 50*self.xi20)*self.xi40 + 162*self.xi40**2))/25725.

        self.s2_s2_1_P0 = (-4*(49*self.xi00**2 - 15*self.xi20**2 + 36*self.xi40**2))/4725.
        self.s2_s2_1_P2 = (-4*(147*self.xi00*self.xi20 + 205*self.xi20**2 - 612*self.xi20*self.xi40 - 180*self.xi40**2))/46305.
        self.s2_s2_1_P4 = (32*(85*self.xi20**2 + 2*(49*self.xi00 + 50*self.xi20)*self.xi40 + 162*self.xi40**2))/77175.
        
        self.s2_s2_2_P0 = (4*(49*self.xi00**2 - 15*self.xi20**2 + 36*self.xi40**2))/4725.
        self.s2_s2_2_P2 =  (-8*(147*self.xi00*self.xi20 + 205*self.xi20**2 -612*self.xi20*self.xi40 -180*self.xi40**2))/46305.
        self.s2_s2_2_P4 = (8*(85*self.xi20**2 + 2*(49*self.xi00 + 50*self.xi20)*self.xi40 + 162*self.xi40**2))/77175.
        
        # Components of various things x L2
        self.xi0_Q2ds = self.qf.xi0_Q2ds
        self.xi2_Q2ds = self.qf.xi2_Q2ds
        self.xi4_Q2ds = self.qf.xi4_Q2ds
        
        self.xi0_Q2s2 = self.qf.xi0_Q2s2
        self.xi2_Q2s2 = self.qf.xi2_Q2s2
        self.xi4_Q2s2 = self.qf.xi4_Q2s2
        
        self.xi0_Q2L2 = self.qf.xi0_Q2L2
        self.xi2_Q2L2 = self.qf.xi2_Q2L2
        self.xi4_Q2L2 = self.qf.xi4_Q2L2
        
        # Cubic terms
        self.xi00_b3 = self.qf.xi00_b3
        self.xi20_b3 = self.qf.xi20_b3
        self.xi40_b3 = self.qf.xi40_b3

        self.xi00_dt = self.qf.xi00_dt
        self.xi20_dt = self.qf.xi20_dt
        self.xi40_dt = self.qf.xi40_dt
    
    def compute_ggm0_linear(self):
    
        self.pgg0_linear = np.zeros( (self.num_gg_components, self.N) )
        
        self.pgg0_linear[0,:] = 2./3 * self.pint
    
    def compute_ggm0_connected(self):
        
        self.pgg0_connected = np.zeros( (self.num_gg_components, self.N) )
        
        self.pgg0_connected[-1,:] = self.kint**2 * self.pint
        
    
    def compute_ggm0_k0(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgg0_0 = np.zeros( (self.num_gg_components, self.N) )
        
        bias_integrands = np.zeros( (self.num_gg_components,self.N)  )
                
        for l in range(self.jn):
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
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac); mu2P4fac = 1./8 * (35*mu6fac - 30*mu4fac + 3*mu2fac)
            P5fac = 1./8 * (63*mu5fac - 70*mu3fac + 15*mu1fac)
            
            # (delta s_ab, delta s_cd)
            bias_integrands[4,:] = self.ds_ds_0_P0 * mu0fac + self.ds_ds_0_P2 * P2fac + self.ds_ds_0_P4 * P4fac
            
            # (delta s_ab, s^2_cd)
            bias_integrands[5,:] = self.ds_s2_0_P0 * mu0fac + self.ds_s2_0_P2 * P2fac + self.ds_s2_0_P4 * P4fac
            
            # (delta s_ab, L2_cd)
            bias_integrands[6,:] = - (2./15*self.xi0_Q2ds * mu0fac -4./21*self.xi2_Q2ds*P2fac+12./35*self.xi4_Q2ds*P4fac)
            
            # (s^2_ab, s^2_cd)
            bias_integrands[7,:] = self.s2_s2_0_P0 * mu0fac + self.s2_s2_0_P2 * P2fac + self.s2_s2_0_P4 * P4fac
            
            # (s^2_ab, L2_cd)
            bias_integrands[8,:] = - (2./15*self.xi0_Q2s2 * mu0fac -4./21*self.xi2_Q2s2*P2fac+12./35*self.xi4_Q2s2*P4fac)
            
            # (L^2_ab, L^2_cd)
            bias_integrands[9,:] = - (2./15*self.xi0_Q2L2 * mu0fac -4./21*self.xi2_Q2L2*P2fac+12./35*self.xi4_Q2L2*P4fac)

            # Cubic Terms
            bias_integrands[10,:] = 2./15*self.xi00_b3 * mu0fac -4./21*self.xi20_b3*P2fac+12./35*self.xi40_b3*P4fac # b3 = 4/7 R1
            
            bias_integrands[11,:] = 2./15*self.xi00_dt * mu0fac -4./21*self.xi20_dt*P2fac+12./35*self.xi40_dt*P4fac # (s, delta t)

            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gg.sph(l, bias_integrands)
            self.pgg0_0 += 4*np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    def compute_ggm0_k1(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        

        self.pgg0_1 = np.zeros( (self.num_gg_components, self.N) )
        
        bias_integrands = np.zeros( (self.num_gg_components,self.N)  )
                
        for l in range(self.jn):
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
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac); mu2P4fac = 1./8 * (35*mu6fac - 30*mu4fac + 3*mu2fac)
            P5fac = 1./8 * (63*mu5fac - 70*mu3fac + 15*mu1fac)
            
            # (s_ab, s_cd)
            bias_integrands[0,:] = - ( self.s_s_0_P1*mu1fac + self.s_s_0_P3*P3fac + self.s_s_0_P5 * P5fac)
                
            # (s_ab, delta s_cd)
            bias_integrands[1,:] = - (self.s_ds_0_P1 * P1fac + self.s_ds_0_P3 * P3fac + self.s_ds_0_P5 * P5fac)
            
            # (s_ab, s^2_cd)
            bias_integrands[2,:] = - (self.s_s2_0_P1 * P1fac + self.s_s2_0_P3 * P3fac + self.s_s2_0_P5 * P5fac)
            
            # (s_ab, L2_cd)
            bias_integrands[3,:] =  - ( self.s_L2_0_P1*mu1fac + self.s_L2_0_P3*P3fac + self.s_L2_0_P5 * P5fac  ) # hack!

            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gg.sph(l, bias_integrands)
            self.pgg0_1 += 4*np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
            
    def compute_ggm0_k2(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgg0_2 = np.zeros( (self.num_gg_components, self.N) )
        
        bias_integrands = np.zeros( (self.num_gg_components,self.N)  )
                
        for l in range(self.jn):
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
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac); mu2P4fac = 1./8 * (35*mu6fac - 30*mu4fac + 3*mu2fac)
            P5fac = 1./8 * (63*mu5fac - 70*mu3fac + 15*mu1fac)
            
            # (s_ab, s_cd)
            bias_integrands[0,:] = (-0.5*self.Xlin)*(2./15*self.xi00*mu0fac-4./21*self.xi20*P2fac+12./35*self.xi40*P4fac) \
                - 0.5 * self.Ylin * (2/15*self.xi00*mu2fac -4./21*self.xi20*mu2P2fac+12./35*self.xi40*mu2P4fac)\
                - (1.5*self.J4**2*mu6fac + (4*self.J3*self.J4-self.J4**2)*mu4fac +   1./6*(16*self.J3**2-8*self.J3*self.J4+self.J4**2)*mu2fac )

            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gg.sph(l, bias_integrands)
            self.pgg0_2 +=  4*np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    
    def compute_ggm1_k0(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgg1_0 = np.zeros( (self.num_gg_components, self.N) )
        
        bias_integrands = np.zeros( (self.num_gg_components,self.N)  )
                
        for l in range(self.jn):
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
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac); mu2P4fac = 1./8 * (35*mu6fac - 30*mu4fac + 3*mu2fac)
            P5fac = 1./8 * (63*mu5fac - 70*mu3fac + 15*mu1fac)
            
            # All the cubic terms and L2 terms become P2(k) x P2(k), so vanish
            
            # (delta s_ab, delta s_cd)
            bias_integrands[4,:] = self.ds_ds_1_P0 * mu0fac + self.ds_ds_1_P2 * P2fac + self.ds_ds_1_P4 * P4fac
                        
            # (delta s_ab, s^2_cd)
            bias_integrands[5,:] = self.ds_s2_1_P0 * mu0fac + self.ds_s2_1_P2 * P2fac + self.ds_s2_1_P4 * P4fac
            
            # (delta s_ab, L2_cd)
            #bias_integrands[6,:] = -4/15*self.xi0_Q2ds * mu0fac +4./21*self.xi2_Q2ds*P2fac +16./35*self.xi4_Q2ds*P4fac
            
            # (s^2_ab, s^2_cd)
            bias_integrands[7,:] = self.s2_s2_1_P0 *mu0fac + self.s2_s2_1_P2 * P2fac + self.s2_s2_1_P4 * P4fac
            
            # (s^2_ab, L2_cd)
            #bias_integrands[8,:] = -4/15*self.xi0_Q2s2 * mu0fac +4./21*self.xi2_Q2s2*P2fac +16./35*self.xi4_Q2s2*P4fac
            
            # (L^2_ab, L^2_cd)
            #bias_integrands[9,:] = -4/15*self.xi0_Q2L2 * mu0fac +4./21*self.xi2_Q2L2*P2fac +16./35*self.xi4_Q2L2*P4fac

            # Cubic Terms
            #bias_integrands[10,:] = -4./15*self.xi00_b3 * mu0fac +4./21*self.xi20_b3*P2fac+16./35*self.xi40_b3*P4fac # b3 = 4/7 R1
            
            #bias_integrands[11,:] = -4./15*self.xi00_dt * mu0fac +4./21*self.xi20_dt*P2fac+16./35*self.xi40_dt*P4fac # (s, delta t)
            
            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gg.sph(l, bias_integrands)
            self.pgg1_0 += 4*np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    def compute_ggm1_k1(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgg1_1 = np.zeros( (self.num_gg_components, self.N) )
        
        bias_integrands = np.zeros( (self.num_gg_components,self.N)  )
                
        for l in range(self.jn):
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
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac); mu2P4fac = 1./8 * (35*mu6fac - 30*mu4fac + 3*mu2fac)
            P5fac = 1./8 * (63*mu5fac - 70*mu3fac + 15*mu1fac)
            
            # (s_ab, s_cd)
            #bias_integrands[0,:] = - ( self.s_s_1_P1*mu1fac + self.s_s_1_P3*P3fac + self.s_s_1_P5 * P5fac  )
               
            # (s_ab, delta s_cd)
            bias_integrands[1,:] = - (self.s_ds_1_P1 * P1fac + self.s_ds_1_P3 * P3fac + self.s_ds_1_P5 * P5fac)
            
            # (s_ab, s^2_cd)
            bias_integrands[2,:] = - (self.s_s2_1_P1 * P1fac + self.s_s2_1_P3 * P3fac + self.s_s2_1_P5 * P5fac)
            
            # (s_ab, L2_cd)
            #bias_integrands[3,:] = - (self.s_L2_1_P1*mu1fac + self.s_L2_1_P3*P3fac + self.s_L2_1_P5 * P5fac  )


            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gg.sph(l, bias_integrands)
            self.pgg1_1 += 4*np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
            
    def compute_ggm1_k2(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgg1_2 = np.zeros( (self.num_gg_components, self.N) )
        
        bias_integrands = np.zeros( (self.num_gg_components,self.N)  )
                
        for l in range(self.jn):
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
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac); mu2P4fac = 1./8 * (35*mu6fac - 30*mu4fac + 3*mu2fac)
            P5fac = 1./8 * (63*mu5fac - 70*mu3fac + 15*mu1fac)
            
            # (s_ab, s_cd)
            bias_integrands[0,:] = (-0.5*self.Xlin)*(-4./15*self.xi00*mu0fac+4./21*self.xi20*P2fac+16./35*self.xi40*P4fac) \
               - 0.5*self.Ylin * (-4/15*self.xi00*mu2fac+4./21*self.xi20*mu2P2fac+16./35*self.xi40*mu2P4fac)\
               - (-2*self.J3**2*mu0fac + 2*(self.J3**2-2*self.J3*self.J4)*mu2fac + 2*(2*self.J3*self.J4-self.J4**2)*mu4fac + 2*self.J4**2*mu6fac )
            
            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gg.sph(l, bias_integrands)
            self.pgg1_2 +=  4*np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    def compute_ggm2_k0(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgg2_0 = np.zeros( (self.num_gg_components, self.N) )
        
        bias_integrands = np.zeros( (self.num_gg_components,self.N)  )
                
        for l in range(self.jn):
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
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac); mu2P4fac = 1./8 * (35*mu6fac - 30*mu4fac + 3*mu2fac)
            P5fac = 1./8 * (63*mu5fac - 70*mu3fac + 15*mu1fac)
            
            # All the cubic terms and L2 terms become P2(k) x P2(k), so vanish
            
            # (delta s_ab, delta s_cd)
            bias_integrands[4,:] = self.ds_ds_2_P0 * mu0fac + self.ds_ds_2_P2 * P2fac + self.ds_ds_2_P4 * P4fac
            
            # (delta s_ab, s^2_cd)
            bias_integrands[5,:] = self.ds_s2_2_P0 * mu0fac + self.ds_s2_2_P2 * P2fac + self.ds_s2_2_P4 * P4fac
            
            # (delta s_ab, L2_cd)
            #bias_integrands[6,:] = 4/15*self.xi0_Q2ds * mu0fac + 8./21*self.xi2_Q2ds*P2fac + 4./35*self.xi4_Q2ds*P4fac
            
            # (s^2_ab, s^2_cd)
            bias_integrands[7,:] = self.s2_s2_2_P0 * mu0fac + self.s2_s2_2_P2 * P2fac + self.s2_s2_2_P4 * P4fac
            
            # (s^2_ab, L2_cd)
            #bias_integrands[8,:] = 4/15*self.xi0_Q2s2 * mu0fac + 8./21*self.xi2_Q2s2*P2fac + 4./35*self.xi4_Q2s2*P4fac
            
            # (L^2_ab, L^2_cd)
            #bias_integrands[9,:] = 4/15*self.xi0_Q2L2 * mu0fac + 8./21*self.xi2_Q2L2*P2fac + 4./35*self.xi4_Q2L2*P4fac
            
            # Cubic Terms
            #bias_integrands[10,:] = 4./15*self.xi00_b3 * mu0fac+8./21*self.xi20_b3*P2fac+4./35*self.xi40_b3*P4fac # b3 = 4/7 R1
            
            #bias_integrands[11,:] = 4./15*self.xi00_dt * mu0fac+8./21*self.xi20_dt*P2fac+4./35*self.xi40_dt*P4fac # (s, delta t)
            
            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gg.sph(l, bias_integrands)
            self.pgg2_0 += 4*np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    def compute_ggm2_k1(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgg2_1 = np.zeros( (self.num_gg_components, self.N) )
                
        bias_integrands = np.zeros( (self.num_gg_components,self.N)  )
                
        for l in range(self.jn):
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
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac); mu2P4fac = 1./8 * (35*mu6fac - 30*mu4fac + 3*mu2fac)
            P5fac = 1./8 * (63*mu5fac - 70*mu3fac + 15*mu1fac)
            
            # (s_ab, s_cd)
            #bias_integrands[0,:] = - ( self.s_s_2_P1*mu1fac + self.s_s_2_P3*P3fac + self.s_s_2_P5 * P5fac  )

            # (s_ab, delta s_cd)
            bias_integrands[1,:] = - (self.s_ds_2_P1 * P1fac + self.s_ds_2_P3 * P3fac + self.s_ds_2_P5 * P5fac)
            
            # (s_ab, s^2_cd)
            bias_integrands[2,:] = - (self.s_s2_2_P1 * P1fac + self.s_s2_2_P3 * P3fac + self.s_s2_2_P5 * P5fac)
            
            # (s_ab, L2_cd)
            #bias_integrands[3,:] = - ( self.s_L2_2_P1*mu1fac + self.s_L2_2_P3*P3fac + self.s_L2_2_P5 * P5fac  )


            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gg.sph(l, bias_integrands)
            self.pgg2_1 += 4*np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
            
    def compute_ggm2_k2(self):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        self.pgg2_2 = np.zeros( (self.num_gg_components, self.N) )
        
        bias_integrands = np.zeros( (self.num_gg_components,self.N)  )
                
        for l in range(self.jn):
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
            P4fac = 1./8 * (35*mu4fac - 30*mu2fac + 3*mu0fac); mu2P4fac = 1./8 * (35*mu6fac - 30*mu4fac + 3*mu2fac)
            P5fac = 1./8 * (63*mu5fac - 70*mu3fac + 15*mu1fac)
            
            # (s_ab, s_cd)
            bias_integrands[0,:] = (-0.5*self.Xlin)*(4./15*self.xi00 * mu0fac+8./21*self.xi20*P2fac+4./35*self.xi40*P4fac) \
             - 0.5*self.Ylin*(4/15*self.xi00*mu2fac+8./21*self.xi20*mu2P2fac+4./35*self.xi40*mu2P4fac)\
             - self.J4**2 * (0.5*mu2fac - mu4fac + 0.5*mu6fac)

            # multiply by IR exponent
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gg.sph(l, bias_integrands)
            self.pgg2_2 +=  4*np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)


    def make_ggtable(self, m, D=1, nonlinear=True, kmin = 1e-3, kmax = 3, nk = 100):
    
        pktable = np.zeros([nk, 1+self.num_gg_components])
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        pktable[:, 0] = kv[:]
        
        # numerical factors to convert from component spectra to P_22^m proper.
        for ii in range(self.num_gg_components):
            if m == 0:
                pktable[:, ii+1] = interp1d(self.kint,\
                                             D**2 * self.pgg0_linear[ii,:] \
                                            + D**2 * nonlinear * (self.pgg0_connected[ii,:] + self.pgg0_0[ii,:] \
                                            + self.kint * self.pgg0_1[ii,:] + self.kint**2 * self.pgg0_2[ii,:]) )(kv)
            if m == 1:
                pktable[:, ii+1] = -0.5 * interp1d(self.kint,\
                                            + D**2 * nonlinear * (self.pgg1_0[ii,:] \
                                            + self.kint * self.pgg1_1[ii,:] + self.kint**2 * self.pgg1_2[ii,:]) )(kv)
            if m == 2:
                pktable[:, ii+1] = 0.5 * interp1d(self.kint,\
                                            + D**2 * nonlinear * (self.pgg2_0[ii,:] \
                                            + self.kint * self.pgg2_1[ii,:] + self.kint**2 * self.pgg2_2[ii,:]) )(kv)
                
        self.pktables_gg[m] = pktable
                
        return pktable
        
    def combine_bias_terms_shape_shape(self, m, shape_bvec1, shape_bvec2, Pshot=0, sn21=0, sn22=0):

        '''
        Compute the galaxy shape x shape cross spectrum.
        
        The counterterm treatment is a bit ad-hoc, but we're just trying to be a bit symmetric in the inputs (really the cross counterterm is its own thing).

        Inputs:
            -m: helicity of the component spectrum to be computed
            -shape_bvec1: shape bias parameters of sample 1
            -shape_bvec2: shape bias parameters of sample 2
            -Pshot: constant shape noise amplitude, if auto-spectrum
            -sn21: first k2-dependent shot noise contribution, if auto-spectrum
            -sn22: second k2-dependent shot noise contribution, if auto-spectrum

        Outputs:
            -k: k scales at which LPT predictions have been computed
            -Pk: bias polynomial combination of parameters times basis spectra 
        
        '''

        c_s, c_ds, c_s2, c_L2, c_3, c_dt, alpha_s1 = shape_bvec1
        b_s, b_ds, b_s2, b_L2, b_3, b_dt, alpha_s2 = shape_bvec2

        # The table is listed in order (s_ab, Ocd), (delta sab, Ocd), (s^2_ab, Ocd)
        bias_poly = np.array([c_s * b_s, c_s * b_ds + b_s * c_ds, c_s * b_s2 + b_s * c_s2, c_s * b_L2 + b_s * c_L2,\
                              c_ds * b_ds, c_ds * b_s2 + c_s2 * b_ds, c_ds * b_L2 + c_L2 * b_ds,\
                              c_s2 * b_s2, c_s2 * b_L2 + c_L2 * b_s2,
                              c_L2 * b_L2,\
                              c_s * b_3 + c_3 * b_s, c_s * b_dt + c_dt * b_s,\
                              alpha_s1 + alpha_s2])

        pktable = self.pktables_gg[m]


        shot_term = 0

        #All m's have a constant noise term.
        #2 scale-dependent parameters for three noise contributions
        if shape_bvec1 == shape_bvec2:
            shot_term = 2 * Pshot
            if m == 0:
                shot_term += pktable[:,0]**2 * sn21
            if m == 1:
                shot_term += pktable[:,0]**2 * (sn21 + sn22)
            if m == 2:
                shot_term += pktable[:,0]**2 * (sn21 + 4*sn22)

        return pktable[:,0], np.sum(bias_poly[None,:] * pktable[:,1:], axis = 1) + shot_term

