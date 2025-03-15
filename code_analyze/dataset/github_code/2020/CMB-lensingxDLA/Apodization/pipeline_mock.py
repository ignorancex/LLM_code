import healpy as hp
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.signal import savgol_filter
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.4,Om0=0.315,Ob0=0.0224/(0.674)**2,Neff=2.99)
import astropy.constants as const
from scipy.interpolate import interp1d
import camb
from camb import model, initialpower

class mock:  # theoretical cross-power spectrum 
    
    def __init__(self,q_z,z_min,z_max,s,z_reso=100,lmax=1200): 
        
        '''
        q_z: 1-D array, the redshift distribution of QSO samples
        z_min: float, the minimum redshift
        z_max: float, the maximum redshift
        data: 3-D array, including index l, coefficient c_kl and its errorbar
        z_reso: int, the applied resolution for redshift
        lmax: int, the maximum spectrum index l
        '''

        self.q_z = q_z                                      # the redshift distribution of QSO samples
        self.z_range = np.linspace(z_min,z_max,z_reso)      # the redshift range the samples cover
        self.dz = (self.z_range[2:]-self.z_range[:-2])/2    # infinitesimal dz
        
        self.s = s 
        
        
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.4,
                           ombh2=0.0224, omch2=0.120,omk=0)  # cosmological model
        pars.InitPower.set_params(ns=0.965)
        
        pars.set_matter_power(redshifts=self.z_range)  
        pars.NonLinear = model.NonLinear_both
        
        self.pk_nonlin = camb.get_matter_power_interpolator(pars,nonlinear=True,           # the non-linear matter power spectrum
                                                            hubble_units=False,
                                                            k_hunit=False,kmax=20,
                                                            zmin=z_min-0.2,zmax=z_max+0.1)
        
        
        self.dN = None                         # the redshift distribution of QSO samples
                
        c = const.c.to('km/s').value           # light velocity in km/s
       
        h = cosmo.h                            # h corresponding to Hubble Constant 
        Om0 = cosmo.Om0
        self.z_cmb = 1100                      # the last scattering surface
        
        self.zcmb = np.linspace(0,self.z_cmb,10**5)
        self.dzcmb = np.diff(self.zcmb)
        self.zcmb = self.zcmb[1:]
        
        def H(z):                              # Hubble parameter
            return (cosmo.H(z)).value
            
        def chi(z):                            # comoving distance
            return (cosmo.comoving_distance(z).value)
        
        def W(z):                              # CMB lensing kernel 
            
            c1 = 3*(H(0)**2)*Om0/(2*c*H(z))
            c2 = (1+z)*chi(z)
            c3 = (1-chi(z)/chi(self.z_cmb))
            return c1*c2*c3
        
        self.z = self.z_range[1:-1]           # the used redshift range when intergrating
        
        self.win1 = H(self.z)/(chi(self.z)**2)*W(self.z)*(1/c) # part of the final integral term independent of bias and index l 
        self.win0 = H(self.z)/(chi(self.z)**2)*(1/c)                  
        self.win2 = H(self.zcmb)/(chi(self.zcmb)**2)*(W(self.zcmb)**2)*(1/c)
        

        bin_num=z_reso # calculate the redshift distribution based on QSO samples
        dz, bin_edges = np.histogram(self.q_z,
                                     bins=bin_num,density=True)
        dN = savgol_filter(dz,51,3) 
        
        self.dN = dN
        
        # mag_bia        
        maxqz = 100
        z_set = np.linspace(self.z,maxqz,10**3).T # shape = (z bins,1E5) 
        diff_z = np.diff(z_set,axis=1)  # shape = (z bins,1E5-1)
        inv_chi_zs = 1/(chi(z_set)[:,1:])      # shape = (z bins,1E5-1)
        chi_z_p = np.repeat(chi(self.z),diff_z.shape[-1]).reshape((len(self.z),diff_z.shape[-1])) # shape = (z bins,1E5-1)
        
        dNz = []
        for i in self.z:
            dN_z = np.histogram(self.q_z,range=(i,maxqz),bins=z_set.shape[1]-1)
            dN_z = dN_z[0]/(dN_z[1][2]-dN_z[1][1])/len(self.q_z)
            dNz.append(dN_z)
        dNz = np.array(dNz)
        #dNz = np.repeat(np.array([dNz]),len(self.z),axis=0).reshape((len(self.z),diff_z.shape[-1]))
        
        g0 = (dNz)*(1-chi_z_p*inv_chi_zs) # shape = (z bins,1E5-1)
        
        g = chi(self.z)/c*np.sum(g0*diff_z,axis=1) # shape = (z bins,)
        #print(np.dot(dNz[0],diff_z[0]))
        
        self.mag_f = 3/2/H(self.z)*Om0*H(0)**2*(5*self.s-2)*g
        
        
    def Ckq(self,b,x=range(1200),line=True):  # calculate the theoretical curve
        
        '''
        b: float, bias
        x: array, your wanted index l range
        line: bool, to obtain a curve (True) or several selected points (False) ?
        '''
        
        f = b*self.dN                   # window function for QSO 
        ff = f[1:-1]+self.mag_f         # to match the used redshift range
        
        w = np.ones(self.z.shape)       # weights for matter power spectrum
    

        
        
        win = (ff+self.mag_f)*self.win1              # part of the final in integral term independent of index l
        chi_z = (cosmo.comoving_distance(self.z).value)
        
        if line:                        # give a complete curve
            
            ls = np.arange(min(x),max(x)+1,dtype=np.float64) 
            cl = np.zeros(ls.shape)
            for i,l in enumerate(ls):
                k = l/chi_z
                cl[i] = np.dot(self.dz,w*self.pk_nonlin.P(self.z,k,grid=False)*win)  # integrate over redshift
                
            return ls,cl
        
        else:                         # give seleted data points 
            cl = np.zeros((len(x),1))
            for i,l in enumerate(x):
                k = l/chi_z
                cl[i] = np.dot(self.dz,w*self.pk_nonlin.P(self.z,k,grid=False)*win)                
            return cl
        
        
    def Cqq(self,b,x=range(1200),line=True):

        f = b*self.dN                   # window function for QSO 
        ff = f[1:-1]+self.mag_f         # to match the used redshift range

        w = np.ones(self.z.shape)       # weights for matter power spectrum




        win = ((ff+self.mag_f)**2)*self.win0              # part of the final in integral term independent of index l
        chi_z = (cosmo.comoving_distance(self.z).value)

        if line:                        # give a complete curve

            ls = np.arange(min(x),max(x)+1,dtype=np.float64) 
            cl = np.zeros(ls.shape)
            for i,l in enumerate(ls):
                k = l/chi_z
                w[:] = 1
                #w[k<1e-4] = 0
                #w[k>=20] = 0
                cl[i] = np.dot(self.dz,w*self.pk_nonlin.P(self.z,k,grid=False)*win)  # integrate over redshift

            return ls,cl

        else:                         # give seleted data points 
            cl = np.zeros((len(x),1))
            for i,l in enumerate(x):
                k = l/chi_z
                w[:] = 1
                w[k<1e-4] = 0
                w[k>=10] = 0
                cl[i] = np.dot(self.dz,w*self.pk_nonlin.P(self.z,k,grid=False)*win)                
            return cl

            
    def Ckk(self,x=range(1200),line=True):

        '''
        win = self.win2
        chi_z = (cosmo.comoving_distance(self.zcmb).value)
        
        if line:               # give a complete curve
            ls = np.arange(min(x),max(x)+1,dtype=np.float64) 
            cl = np.zeros(ls.shape)
            for i,l in enumerate(ls):
                k = l/chi_z
                cl[i] = np.dot(self.dzcmb,self.pk_nonlin.P(self.zcmb,k,grid=False)*win)  # integrate over redshift

            return ls,cl

        else:                         # give seleted data points 
            cl = np.zeros((len(x),1))
            for i,l in enumerate(x):
                k = l/chi_z
                cl[i] = np.dot(self.dzcmb,self.pk_nonlin.P(self.zcmb,k,grid=False)*win)                
            return cl
            
        '''
        
        
        nz = 100 #number of steps to use for the radial/redshift integration
        kmax=10  #kmax to use
        #First set up parameters as usual
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
        pars.InitPower.set_params(ns=0.965)

        #For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
        #so get background results to find chistar, set up a range in chi, and calculate corresponding redshifts
        results= camb.get_background(pars)
        chistar = results.conformal_time(0)- results.tau_maxvis
        chis = np.linspace(0,chistar,nz)
        zs=results.redshift_at_comoving_radial_distance(chis)
        #Calculate array of delta_chi, and drop first and last points where things go singular
        dchis = (chis[2:]-chis[:-2])/2
        chis = chis[1:-1]
        zs = zs[1:-1]

        #Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
        #Here for lensing we want the power spectrum of the Weyl potential.
        PK = camb.get_matter_power_interpolator(pars, nonlinear=True, 
            hubble_units=False, k_hunit=False, kmax=kmax,
            var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zs[-1])

        #Get lensing window function (flat universe)
        win = ((chistar-chis)/(chis**2*chistar))**2
        #Do integral over chi
        ls = np.arange(0,1200, dtype=np.float64)
        cl_kappa=np.zeros(ls.shape)
        w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
        for i, l in enumerate(ls):
            k=(l+0.5)/chis
            w[:]=1
            w[k<1e-4]=0
            w[k>=kmax]=0
            cl_kappa[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*win/k**4)
        cl_kappa*= (ls*(ls+1))**2
        
        return ls,cl_kappa
    
    
def bin_corr(c_kq,l_min=30,l_max=1200,band=15):   # bin the power-spectrum
    
    '''
    c_kq: array, cross-power spectrum to be binned
    l_min: int, the beginnning l you want
    l_max: int, the maximum l you want
    band: int, the total number of data points in the final results
    '''
    
    width = int((l_max-l_min)/band)                    # number of l in each bin
    bin_Cl = np.zeros((band,1))
    bin_l = np.zeros((band,1))
    
    for i in range(band):
        ell_min = l_min+width*i
        ell_max = l_min+(i+1)*width
        ell_seq = np.arange(ell_min,ell_max,1)
        weights = (2*ell_seq+1)/np.sum(2*ell_seq+1)
        bin_l[i] = np.sum(weights*ell_seq)
        bin_Cl[i] = np.sum(weights*c_kq[ell_min:ell_max])
    
    return bin_l, bin_Cl