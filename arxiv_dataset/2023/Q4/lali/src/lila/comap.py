import os
import glob
import h5py
import numpy as np
import dask.array as da
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from nbodykit import setup_logging
import logging
import logging.config
from nbodykit.lab import ArrayMesh

from . import lim

class MockComap(lim.MockLim):
    """Generate mocks for Exlaim survey"""
    def __init__(self, snap, axis=3, basepath=None, boxsize=None, fine_Nmesh=None,
                survey_params={'beam_fwhm':4.5,'freq_res': 31.25, 'tempsys':44.0, 'nfeeds': 19, 
                'deltanu':15.625, 'patch': 4, 'tobs':1500, 'noise_per_voxel':17.8, 'nu_rest':115.27,  
                'nu_co_rest':115.27},  Li16_params={'alpha':1.17, 'beta':-0.21, 'sigma_co':0.37, 'delta_mf':1,
                'behroozi_avsfr':'/central/groups/carnegie_poc/mqezlou/lim/behroozi+13/sfr_release.dat', 
                'sfr_type':'behroozi+13'}, COMAP21_params={'A':-2.85, 'B':-0.42, 'C':10**10.63, 'M':10**12.3, 
                'sigma_co':0.42}, co_model='COMAP+21',**kwargs ):
        
        self.Li16_params = Li16_params
        self.COMAP21_params = COMAP21_params
        self.co_model = co_model
        self.noise_per_voxel = survey_params['noise_per_voxel']
    
        super().__init__(snap=snap, axis=axis, basepath=basepath, boxsize=boxsize, 
                         fine_Nmesh=fine_Nmesh, survey_params=survey_params, **kwargs)
    def get_res_par(self):
        """Calculate the spatial resolution along the lne-of-sight in units of comoving Mpc/h
        freq_res is the frequncy resolution of the survey."""
        return cosmo.h*((1+self.z)**2 *const.c*self.survey_params['freq_res']/(1e6*cosmo.H(self.z)*self.survey_params['nu_rest'])).value
        
    def fiducial_comap_2021(self, mvir):
        """Adopted from Chung+21, early science COMAP"""
        lco_p =  self.COMAP21_params['C'] / ((mvir/(cosmo.h*self.COMAP21_params['M']))**self.COMAP21_params['A'] +
                                             (mvir/(cosmo.h*self.COMAP21_params['M']))**self.COMAP21_params['B'])
        
        lco_p= 10**np.random.normal(np.log10(lco_p), self.COMAP21_params['sigma_co'])
        # Convert the units to L_sol
        return self.fix_co_lum_unit(lco_p)
    
    def co_lum_to_temp(self, lco, vol):
        """
        Convert CO luminosity in units of r$L_{\odot}$ in a volume of `vol` to brightness
        temperature in units of r'$\mu k$'. See Appendix B.1. in Li et al. (2016) arXiv:1503.08833.

        Parameters:
        lco (numpy array): CO luminosity in L_sol units
        vol (float): The volume in which the CO luminosity is measured, in units of (cMpc/h)^3.

        Returns:
        numpy array: The temperature brightness in that volume in units of \nu K.
        """
        temp = lco * (3.1e4 * (1 + self.z) ** 2 * self.survey_params['nu_rest'] ** -3 *
                    cosmo.H(self.z).value ** -1 * (vol / cosmo.h ** 3) ** -1)
        return temp

    def get_halo_luminosity(self):
        """Return the CO luminosity of the subhalos in L_sun unit"""
        if self.co_model=='Li+16':
            if not self.silent_mode:
                self.logger.info('Li+16 CO emission model with params:')
                self.logger.info(f'{self.Li16_params}')
            # Appendix B.1. in Li et. al. 2016 arxiv 1503.08833
            l_ir = self.get_ir_lum_halos()
            # We assign L_CO to zero for subhalos with sfr=0
            ind = np.where(l_ir > 0)[0]
            l_co = np.zeros_like(l_ir)
            l_co[ind] = 10**((np.log10(l_ir[ind]) - self.Li16_params['beta'])/self.Li16_params['alpha'])
            # Add a log-normal scatter
            l_co[ind] = 10**(np.log10(l_co[ind])+np.random.normal(0, self.Li16_params['sigma_co'], ind.size))
            # Convert the units to L_sol
            l_co[ind] = self.fix_co_lum_unit(l_co[ind])
            
        elif self.co_model=='COMAP+21':
            if not self.silent_mode:
                self.logger.info('COMAP+21 CO emission model with params:')
                self.logger.info(f'{self.COMAP21_params}')
            l_co = self.fiducial_comap_2021(mvir=self.halos[self.halo_type+'Mass'].compute()*1e10)
            self.halo_id = np.arange(0,l_co.size)
        else:
            raise NameError("Selected CO model is not supported!")

        return l_co  

    def fix_co_lum_unit(self, l_co_p):
        """Convert the units in L_CO_p from K (km/s) pc^2 to L_solar
        Li at. al. 2016 eq 4.
        """
        return  4.9e-5 * (self.survey_params['nu_rest']/self.survey_params['nu_co_rest'])**3 * l_co_p
    

    def get_ir_lum_halos(self):
        """Return the IR luminosity of the subhalos"""
        self._load_sfr()
        return self.sfr*1e10 / self.Li16_params['delta_mf']
    
    def get_lim_map(self):
        """Calculate the CO temperature map on a uniform grid. The final map used for
            calculating the power-spectrum.
        Returns:
            numpy array: The temperature on a grid in units of r'$\mu K$'. The grid is a fine
                mesh defined with `self.fine_Nmesh` which is different from the resolution of
                the actual survey, i.e. `self.Nmesh`.
           """
        if not self.silent_mode:
            self.logger.info('Getting the CO map')
        co_mesh = self.get_voxel_luminosity()
        def func(coord, lco):
            return self.co_lum_to_temp(lco=lco, vol=self.fine_vol_vox)
        co_mesh = co_mesh.apply(func, kind='index', mode='real')
        co_temp = co_mesh.compute()

        if not self.silent_mode:
            self.logger.info(' co_temp range : %s, %s,  co_temp mean : %s', 
                    np.min(co_temp), np.max(co_temp), np.mean(co_temp))
        self.lim_map = ArrayMesh(co_temp, BoxSize=self.boxsize)
        del co_temp

    def get_lim_noise_pk(self):
        """Get CO noise power spectrum which is a white noise (Chung+18 eq 15):
            P_n = sigma_n^2 V_vox 
        Returns: 
            The CO noise power spectrum in units of (\mu K)^2 * (cMpc/h)^3 
        """
        self.get_noise_per_voxel()
        self.noise_pk = (self.noise_per_voxel**2)*self.vol_vox

    def get_noise_per_voxel(self):
        """Get the amplitude of the co lim noise. Assuming a Gaussian random noise we only
        need the diagonal term in the covariance matrix, sigma_n. 
        We follow the prescription in Chung+18 eq 16.
        Returns:
            The noise rms error in units of \mu K
        """
        if self.noise_per_voxel is None:
            # Get the average survey time per pixel in seconds
            dx = self.survey_params['beam_fwhm']/(60*np.sqrt(8*np.log(2)))
            taupix = self.survey_params['tobs']*3600*dx*dx/self.survey_params['patch']
            # factors are simplified in the fraction. T: K -> \mu K and deltanu: MHz -> Hz
            self.noise_per_voxel = (self.survey_params['tempsys']*1e3 / 
                                    np.sqrt(self.survey_params['nfeeds'] *
                                            self.survey_params['deltanu']*taupix) )