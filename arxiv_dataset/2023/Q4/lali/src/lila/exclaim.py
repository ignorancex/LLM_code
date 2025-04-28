import os
import glob
import h5py
import numpy as np
import dask.array as da
from astropy import constants, units
from astropy.cosmology import Planck15 as cosmo
from nbodykit import setup_logging
import logging
import logging.config
from nbodykit.lab import ArrayMesh
from nbodykit.lab import HDFCatalog
from nbodykit import CurrentMPIComm
from . import lim

class MockExclaim(lim.MockLim):
    """Generate mocks for Exlaim survey"""
    def __init__(self, snap, axis=3, basepath= None, boxsize=None, fine_Nmesh=None, 
                survey_params={'beam_fwhm':4.33,'spec_res': 512,
                'deltanu':15.625, 'patch': 2.5, 'tobs':10.5, 'noise_per_voxel':None, 'nu_rest':1.901e6,  
                'nu_co_rest':115.27}, padmanabhan19_params={'M1':2.39e-5, 'N1':4.19e11}, 
                cii_model='Padmanabhan+19', noise_pk=None, **kwargs ):
        self.cii_model = cii_model
        self.padmanabhan19_params = padmanabhan19_params
        super().__init__(snap=snap, axis=axis, basepath=basepath, boxsize=boxsize, 
                         fine_Nmesh=fine_Nmesh, survey_params=survey_params, noise_pk=noise_pk, **kwargs)
        

    def get_res_par(self):
        """Eq 20 of Pullen+22 
        """
        return cosmo.h*(constants.c*(1+self.z)/(1e3*cosmo.H(self.z)*self.survey_params['spec_res'])).value
    def get_lim_noise_pk(self):
        """Table 3 from Pullen+22
        returns the instrument noise power on units of 
        """
        if self.noise_pk is None:
            if self.z < 2.8:
                self.noise_pk = 4.789e10#2.51e7 #107561.08108501034
            elif self.z <3.0:
                self.noise_pk = 10638.
            self.logger.info(f'noise_pk is set to : {self.noise_pk}')
        else:
            self.logger.info(f'noise_pk is passed as : {self.noise_pk}')
            pass
        
    def cii_padmanabhan19(self, mvir):
        """CII emission model, Padmanabhan et al 209
        Eq 3 in Pullen+22
        """
        halo_lum =  ( (mvir/self.padmanabhan19_params['M1'])**0.49 * 
                 np.exp(-self.padmanabhan19_params['N1']/mvir) * 
                 ( (1+self.z)**2.7 / (1 + ((1+self.z)/2.9)**5.6) )**1.79)
        return halo_lum

    def get_halo_luminosity(self):
        if not self.silent_mode:
            self.logger.info('CII model :%s', self.cii_model)
        if self.cii_model == 'Padmanabhan+19':
            l_cii = self.cii_padmanabhan19(mvir=self.halos[self.halo_type+'Mass'].compute()*1e10)
            self.halo_id = np.arange(0,l_cii.size)
        else:
            raise NameError("Selected CII model is not supported!")
        return l_cii

    def get_lim_map(self):
        """Get the CII luminosity map on a uniform grid of self.fine_Nmesh
        """
        if not self.silent_mode:
            self.logger.info('Getting the CII map')
        cii_mesh = self.get_voxel_luminosity()
        cii_map = cii_mesh.compute()
        cii_map *=  self.lsol_to_kjy()
        if not self.silent_mode:
            self.logger.info(' cii_map range : %s, %s,  cii_map mean : %s',
                    np.min(cii_map), np.max(cii_map), np.mean(cii_map))
        self.lim_map = ArrayMesh(cii_map, BoxSize=self.boxsize)
        del cii_map
        
    def lsol_to_kjy(self):
        """
        Convert the units of the lumonosity map from solar luminosity to KJy
        i.e. the prefactor in Eq1 of Pullen+22
        """
        return ((constants.c.to('km/s') / (4*np.pi*self.survey_params['nu_rest']*units.Hz*1e6 * cosmo.H(self.z)) *
                  1/(self.fine_vol_vox*(units.Mpc/((1+self.z)*cosmo.h))**3) * units.solLum.to('W')*units.W).to('Jy')).value/1e3