from regpy.util import classlogger

from nbodykit.lab import *
from nbodykit import style, setup_logging

from reglyman.density import LinearPower

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.ndimage import gaussian_filter

from joblib import Parallel, delayed
import multiprocessing

import logging

setup_logging()

'''
Estimates peculiar velocities from the density field
'''

class Interpolation:
    log = classlogger
    def __init__(self, parameters_box, comoving_full_range, comoving, transfer="EisensteinHu", path_transfer=None, column=None, cosmo=None):
        chosen_cosmo=parameters_box['cosmology']
        #specifies specific cosmological model
        if chosen_cosmo=='Planck15':
            self.cosmo = cosmology.Planck15
        elif chosen_cosmo=='UserDefined':
            self.cosmo = cosmo
        else: 
            print(chosen_cosmo, 'not implemented right now')
            print('Continue with Planck15 cosmology instead')
            self.cosmo = cosmology.Planck15     

        self.redshift=parameters_box['redshift']
        
        self.Jeans_length=parameters_box['jeans_length']
        
        self.Plin = LinearPower(self.cosmo, self.redshift, transfer, path_transfer, column)
        
        #speify baryonic correlation function
        self.Plin_b=lambda k: 1/(1+self.Jeans_length**2*k**2)**2*self.Plin(k)
        self.Plin_b.redshift=self.redshift
        self.Plin_b.sigma8=self.cosmo.sigma8
        self.Plin_b.cosmo=self.cosmo
        
        k = np.linspace(10**(-5), 10, 10**6)
        size = len(k)//2
        Pk = self.Plin_b(k)
        fourier_coeff = np.abs(np.fft.fftn(Pk)[0:size+1])
        frqs = np.linspace(0, 0.1*size, size+1)
        self.cf_lin_dd = Spline(frqs, fourier_coeff)
        #self.cf_lin_dd=cosmology.CorrelationFunction(self.Plin_b)
        
        #v-delta power spectrum is given by iaHf*k_z/k**2*P_b(k)
        #In integration this can be rewritten using d/dz
        self.f = self.cosmo.scale_independent_growth_rate(self.redshift)
        self.norm=self.f * 100 * self.cosmo.efunc(self.redshift) / (1+self.redshift)
        self.Plin_vdelta=lambda k: self.norm*1/k**2*self.Plin_b(k)
        self.Plin_vdelta.redshift=self.redshift
        self.Plin_vdelta.sigma8=self.cosmo.sigma8
        self.Plin_vdelta.cosmo=self.cosmo
        
        Pk = self.Plin_vdelta(k)
        fourier_coeff = np.abs(np.fft.fftn(Pk)[0:size+1])
        frqs = np.linspace(0, 0.1*size, size+1)
        self.cf_lin_vd = Spline(frqs, fourier_coeff)
        #self.cf_lin_vd=cosmology.CorrelationFunction(self.Plin_vdelta)
        
        self.comoving_full_range=comoving_full_range
        self.comoving=comoving
        #Hubble constant
        self.H=100*self.cosmo.efunc(self.redshift)

        #Initialize meshgrid coordinates to compute distance between different points
        self.trans_x = np.linspace(0, parameters_box['background_box'][0], parameters_box['background_sampling'][0])
        self.trans_y = np.linspace(0, parameters_box['background_box'][1], parameters_box['background_sampling'][1])

        xv, yv, zv = np.meshgrid(self.trans_x, self.trans_y, self.comoving)
        self.xvf = xv.flatten()
        self.yvf = yv.flatten()
        self.zvf = zv.flatten()
        self.size_x = parameters_box['background_sampling'][0]
        self.size_y = parameters_box['background_sampling'][1]
        self.size_z = np.size(comoving)
        self.size_z_full = np.size(comoving_full_range)
        self.sampling = int(self.size_z_full/self.size_z) 
        assert self.sampling*self.size_z == self.size_z_full
        self.size = self.size_x*self.size_y*self.size_z

        #Initialize parallelization
        #self.num_cores = multiprocessing.cpu_count()
        
        return
        
    #compute correlation between velocity and logarithmic density perturbation        
    def Compute_V_Delta_Corr(self):
        #get distances between points
        diff=np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                diff[i, j]=np.sqrt( (self.xvf[i]-self.xvf[j])**2 + (self.yvf[i]-self.yvf[j])**2 + (self.zvf[i]-self.zvf[j])**2)
        #Compute covariance
        Covariance=self.cf_lin_vd(diff)
        Covariance /= Covariance[0, 0]
        #Compute final covariance by differentiating along z
        for i in range(self.size):
            Covariance[i, i]=1  
            array = Covariance[i, :].reshape((self.size_x, self.size_y, self.size_z))
            for l in range(self.size_x):
#                results=Parallel(n_jobs=self.num_cores)(delayed(self._compute_diffquotient)(array[l, m, :]) for m in range(self.size_y))
                for m in range(self.size_y):
                    array[l, m, :]=-self._compute_diffquotient(array[l, m, :])
#                    array[l, m, :] = -results[m]
            Covariance[i, :] = array.flatten()
        return Covariance
     
    #computes derivative along line of sight           
    def _compute_diffquotient(self, array):
        N=np.size(array)
        toret=np.zeros(N)
        for i in range(1, N-1):
            toret[i]=(array[i+1]-array[i-1])/(self.comoving[i+1]-self.comoving[i-1])            
        toret[0]=(array[1]-array[0])/(self.comoving[1]-self.comoving[0])
        toret[N-1]=(array[N-1]-array[N-2])/(self.comoving[N-1]-self.comoving[N-2])              
        return toret

    #computes the correlation between logarithmic density perturbation
    def Compute_Delta_Delta_Corr(self):
        #get distances between points
        diff=np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                diff[i, j]=np.sqrt( (self.xvf[i]-self.xvf[j])**2 + (self.yvf[i]-self.yvf[j])**2 + (self.zvf[i]-self.zvf[j])**2)
        Covariance=self.cf_lin_dd(diff)
        Covariance /= Covariance[0, 0]
        return Covariance 

    #compute all needed correlations for velocity estimation procedure
    def Compute_Correlations(self):
        cov_vd = self.Compute_V_Delta_Corr()
        if self.log.isEnabledFor(logging.INFO):
            self.log.info('Cov_vd computed')

        cov_dd = self.Compute_Delta_Delta_Corr()   
        if self.log.isEnabledFor(logging.INFO):
            self.log.info('Cov_dd computed')  

        cov_dd_inv = np.linalg.inv(cov_dd)
        if self.log.isEnabledFor(logging.INFO):
            self.log.info('Cov_dd inverted')

        return [cov_vd, cov_dd, cov_dd_inv]
    
    def Compute_Transverse_Correlation(self):
        size_trans = self.size_x * self.size_y
        diff = np.zeros((size_trans, size_trans))
        xv, yv = np.meshgrid(self.trans_x, self.trans_y)
        xvf = xv.flatten()
        yvf = yv.flatten()        
        for i in range(size_trans):
            for j in range(size_trans):
                diff[i, j] = np.sqrt( (xvf[i]-xvf[j])**2 + (yvf[i]-yvf[j])**2 )
        corr=self.cf_lin_dd(diff)
        corr /= corr[0, 0]
        return corr
    
    #performs estimation of the velocity field
    def Estimate_Velocity(self, delta, cov_dd_inv, cov_vd, sigma): 
        vel=cov_vd @ cov_dd_inv @ delta
        return gaussian_filter(vel, sigma=sigma)

    #interpolate the downsampled velocity to full grid
    def _interpolate_velocity(self, vel):
        return np.interp(self.comoving_full_range, self.comoving, vel)
    
    #correct density by new velocity estimate
    def _correct_density(self, delta, vel, i):
        array_d = delta.reshape((self.size_x, self.size_y, self.size_z_full))
        array_v = vel.reshape((self.size_x, self.size_y, self.size_z))
        for i in range(self.size_x):
            for j in range(self.size_y):
                vec=self._interpolate_velocity(array_v[i, j, :])
                xrange=self.H*self.comoving_full_range/(1+self.redshift)-0.5**i*vec
                array_d[i, j, :] = np.interp(self.H*self.comoving_full_range/(1+self.redshift), xrange, array_d[i, j, :])
        return array_d.flatten()
    
    #perform the estimaation
    def Perform_Estimation(self, delta, maxiter=10, sigma=0):
        #compute correlations
        self.cov_vd, self.cov_dd, self.cov_dd_inv = self.Compute_Correlations()
        #iteratively estimate velocity and calculate new density
        for i in range(maxiter):
            vel=self.Estimate_Velocity(delta[::self.sampling], self.cov_dd_inv, self.cov_vd, sigma=sigma)
            delta=self._correct_density(delta, vel, i+1)
        return [delta, vel]
    
    def ChooseLOS(self, matrixdata, index):
        shape = matrixdata.shape
        data = matrixdata[index, :]
        size=int(np.count_nonzero(index))
        return np.reshape(data, (size, shape[1]))
    
    def ChooseCov(self, matrixdata, index):
        size = int(np.count_nonzero(index))
        toret = matrixdata[index, :]
        toret = toret[:, index]
        return np.reshape(toret, (size, size))
    
    def ChooseCov3D(self, matrixdata, index):
        shape = matrixdata.shape
        size = int(np.count_nonzero(index))
        toret = matrixdata[:, index]
        return np.reshape(toret, (shape[0], size))
    
    def Perform_Interpolation(self, delta, index):
        corr = self.Compute_Transverse_Correlation()
        #transverse covariance matrix of observed (randomly chosen) lines of sight
        covariance_cross_observed=self.ChooseCov(corr, index)
        #invert covariance
        covariance_cross_observed_inv=np.linalg.inv(covariance_cross_observed)
        #transverse covariance between observed lines of sight and the complete
        #set of lines of sight
        covariance_cross_observed_3D=self.ChooseCov3D(corr, index)
            
        #Choose observed lines of sight 
        reco=self.ChooseLOS(delta, index)
        
        return covariance_cross_observed_3D @ covariance_cross_observed_inv @ reco