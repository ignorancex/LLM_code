import os
import h5py
import numpy as np
from nbodykit.lab import ArrayMesh
from astropy.cosmology import Planck15
from lytomo_watershed import spectra_mocking as sm
from lytomo_watershed import z_bin
from scipy.ndimage import gaussian_filter as gf
from nbodykit import setup_logging
import logging
import logging.config

class MockLya():
    """Work with the Lya tomography maps"""
    def __init__(self, noiseless_file, boxsize=None, brange=None, num_spectra=None,
                 dperp=2.5, sn=2, spec_file=None,silent_mode=True, transpose=(0,1,2), flux_mode=False, 
                 compensated=True, source_pk_file=None, HCD_mask={'thresh':None, 'type':None, 'vel_width':1235}):
        """
        noiseless_file : Path to the noiseless Lya tomography map.
        boxsdize : Boxsize in cMpc/h
        num_spectra : Number of spectra observed in the Lya tomography
        sn : float, average signal-to-noise of the spectra per Angstrom
             Default = 2, typical required for Lya tomography surveys, McQuinn+White+11 (arxiv:1102.1752)
        source_pk_file: str, default None. Path to the file storing the galaxy/quasar source power spectrum.
                        It is an h5py file of 2D power spectrum (k,mu). We only need the P(k, mu=0) to
                        account for the source clustering in the Lya noise power. Refer to discussion in the last
                        paragraph of arxiv:1102.1752
        HCD_mask_type: str or None, 'EW' or 'NHI'. The type of masking we want to apply to high column dnesity
                        absorbors of the Lya forest.
        HCD_mask_file: str, Address to the mask file, only if HCD_mask_type = 'EW'. It will avoid computing the mask
                            as it is expensive to calculate. 
        HCD_mask['thresh']: float, if HCD_mask_type='NHI': pass the high column dnesity objects where N_HI > HCD_mask['thresh'] will be
                        replaced with random non HCDs.
                           if HCD_mask_type='EW': masking all the regions with equivalent width > HCD_mask['thresh'] (in units
                            of Ansgtrom) with (1/<F>) -1
        
        """
        self.silent_mode = silent_mode
        if not self.silent_mode:
            setup_logging()
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
            # create logger
            self.logger = logging.getLogger('MockLya')
            self.logger.info('Starting')
            self.logger.info('dperp = %s', dperp)

        self.noiseless_file = noiseless_file
        self.brange = brange
        if boxsize is None:
            self.boxsize = (self.brange[1]-self.brange[0],
                            self.brange[3]-self.brange[2],
                            self.brange[5]-self.brange[4],)
        else:
            if isinstance(boxsize, int):
                self.boxsize = [boxsize, boxsize, boxsize]
            else:
                self.boxsize = boxsize
            
        self.num_spectra = num_spectra
        self.spec_file = spec_file
        with h5py.File(self.spec_file,'r') as f:
            self.cosmo = Planck15.clone(name='MockLya Cosmo', 
                    H0 = 100*f['Header'].attrs['hubble'],
                    Om0 = f['Header'].attrs['omegam'],
                    Ob0 = f['Header'].attrs['omegab'])
        
        self.compensated =compensated
        self.dperp = dperp
        self.flux_mode = flux_mode
        self.HCD_mask = HCD_mask
        self.sn = sn
        self.mock_map = None
        self.noiseless_map = None
        self.transpose = transpose
        with h5py.File(self.noiseless_file,'r') as f:
            self.z = f['redshift'][()]
        self.noiseless_map = self.get_map_noiseless()
        self.Nmesh = [self.noiseless_map.shape[0], 
                    self.noiseless_map.shape[1], 
                    self.noiseless_map.shape[2]]
                            
        if source_pk_file is not None:
            if not self.silent_mode:
                self.logger.info('Loading the source angular power spectrum')
            with h5py.File(self.source_pk_file,'r') as f:
                ang_pk_source = {}
                self.source_pk['power'] = f['pkmu'][:,0]
                self.source_pk['k_perp'] = f['k'][:,0]
        else:
            self.source_pk = None
    
    def get_map_noiseless(self):
        """Openning and trimming the noiseless map"""
        if self.noiseless_map is None:
            lya_map = np.transpose(h5py.File(self.noiseless_file,'r')['map'][:], axes=self.transpose)
            if self.HCD_mask['type'] == 'NHI':
                lya_map = self._mask_HCDs_NHI_based(lya_map)

            self.noiseless_map = self._get_map(lya_map)

        return self.noiseless_map
        
    def _get_map(self, lya_map):
        """Trim the lya maps"""
        if self.flux_mode:
            # Convert delta_F to F, so at the end, flux will be the output map
            lya_map = self._dF_to_F(lya_map)

        if self.brange is not None:
            # Take a subset of the full box
            ind = np.meshgrid(np.arange(self.brange[0], self.brange[1]+1),
                              np.arange(self.brange[2], self.brange[3]+1),
                              np.arange(self.brange[4], self.brange[5]+1),
                              indexing='ij')
            ind = (ind[0], ind[1], ind[2])
            ly_map = lya_map[ind]
        return lya_map
    
    def _dF_to_F(self, lya_map):
        """Convert delta_F to Flux
        Paramters: 
            lya_map: delta_F map
        Returns:
            The flux map
        """
        lya_map +=1 
        lya_map *=self.get_mean_flux(z=self.z)
        if not self.silent_mode:
            self.logger.info('dF -> F')
        
        return lya_map

    def _mask_HCDs_NHI_based(self):
        """A function to replace sightlines containing an HCD (high-column density)
           with random chunck of spectra which is not an HCD. HCDs are defines as N_HI > self.HCD_mask['thresh'].
           The column density (NHI) is the integrated value over a width of vel_width
           along the line-of-sight.
        Parameters:
            lya_map: delta_F map
            vel_width: float, velocity width  in km/s.
        Returns:
            lya_map: A deltaF map without any HCDs

        """
        self.logger.info('Masking HCDs with log(NHI) >= '+str(np.log10(self.HCD_mask['thresh'])))
        # The number of adjacent pixels should be added to get NHI :
        with h5py.File(self.spec_file,'r') as f:
            tau_map = f['tau/H/1/1215'][:]
        
        L = np.shape(tau_map)[1]
        vmax = self.cosmo.H(self.z).value*(1/(1+self.z))*self.boxsize[2]/self.cosmo.h
        addpix = np.around( self.HCD_mask['vel_width'] / ( vmax / L ) )
        
        t = np.arange(0,L+1,addpix).astype(int)
        NHI_map = h5py.File(self.spec_file,'r')['colden/H/1'][:]
        NHI_map_summed = np.zeros(shape=(NHI_map.shape[0], t.size-1))
        for i in range(t.size-1):
            NHI_map_summed[:,i] = np.sum(NHI_map[:,t[i]:t[i+1]], axis=1)
        del NHI_map
        self.logger.info('NHI is summed over '+str(addpix)+' pixels')
        
        ind_HCD = np.where(NHI_map_summed > self.HCD_mask['thresh'])
        ind_HCD = np.array(ind_HCD).astype(int)
        mask = np.zeros(shape=(NHI_map.shape[0],), dtype=np.int)
        mask[ind_HCD[0]]  = 1
        ind_HCD = np.where(mask)

        self.logger.info('HCD fraction = '+str(ind_HCD[0].size/(lya_map.shape[0]*lya_map.shape[1])))
        
        while ind_HCD[0].size !=0:
            ind_rep = np.random.randint(0, NHI_map.shape[0], (ind_HCD[0].size,) )
            tau_map[ind_HCD[0],:] = tau_map[ind_rep, :]
            NHI_map_summed[ind_HCD[0], :] = NHI_map_summed[ind_rep, :]
            ind_HCD = np.where(NHI_map_summed > self.HCD_mask['thresh'])
            ind_HCD = np.array(ind_HCD).astype(int)
            mask = np.zeros(shape=(NHI_map.shape[0],), dtype=np.int)
            mask[ind_HCD[0]]  = 1
            ind_HCD = np.where(mask)
        return tau_map

    
    def get_CNR(self, num_spectra, z, QSO=[], DCV13_model=True, seed=14):
        """ Calculate Continuum to noise ratio (signal to noise ratio) modeled in LATIS
        QSO contains the sightline number of quasars.
        """
        CNR = np.zeros(num_spectra)
        np.random.seed(seed)

        for ii in range(num_spectra):
            if ii in QSO:
                if DCV13_model:
                    CNR[ii] = np.exp(np.random.normal(2.3, 1.2))

            else:
                mean = 0.84 + 0.99 * (z - 2.5)- 1.82*(z - 2.5)**2
                CNR[ii] = np.exp(np.random.normal(mean, .43))
        return CNR

    def get_CE(self, CNR) :
        """ Calculate Continuum noise for each spectra modeled in LATIS"""
        CE = 0.24*CNR**(-0.86)
        CE[np.where(CE < 0.05)] = 0.05

        return CE

    def get_mean_flux(self, z, metal=False) :
        """ get the mean flux used in LATIS Faucher-Giguere 2008"""
        if metal :
            # The below is not good for HI absorption as includes the effect of metals
            return np.exp(-0.001845*(1+self.z)**3.924)
        else :
            # The below is good for only HI absorptions, does not include metal absorption
            return np.exp(-0.001330*(1+self.z)**4.094)

    def get_lya_sigma_n(self):
        """Get the amplitdue of the Lya noise. Actual Lya tomography noise covariance 
        however has different diagonal terms. For now, we choose the median of those terms.
        Returns:
            The median noise amplitude in Lya tomo map
        """
        CNR = self.get_CNR(num_spectra= self.num_spectra, z=self.z)
        sigma_lya = np.sqrt(self.get_CE(CNR)**2+(1/CNR)**2)/self.get_mean_flux(self.z)
        sigma_lya[sigma_lya < 0.2] = 0.2
        
        return np.median(sigma_lya)
    
    def load_noiseless(self):
        df_map = h5py.File(self.noiseless_file,'r')['map'][:]
        return df_map
    
    def estimate_noise(self):
        """ Estimate the noise in the Lya delta_F map as the rms of the deviation
            of the mock maps from the noiseless map
        """
        noiseless_map = self.get_map_noiseless()
        mock_map = self.get_map_mock()
        
        diff = mock_map - noiseless_map
        
        return diff
    
    def _get_redshift_width(self):
        """Calculates the width of the simulaion box in redshift space"""
        med_comoving = self.cosmo.comoving_distance(self.z).value - self.cosmo.comoving_distance(2.2).value
        z = z_bin.cmpch_to_redshift(d=[med_comoving-self.boxsize[2]/2, med_comoving+self.boxsize[2]/2])
        self.z_width =  z[1]- z[0]

    def get_galaxy_counts(self, pixfile='/run/media/mahdi/HD2/Lya/spectra/maps/mapsv13/dcv13pix.dat',
                        mapfile='/run/media/mahdi/HD2/Lya/spectra/maps/mapsv13/dcv13map.dat',
                        idsfile='/run/media/mahdi/HD2/Lya/spectra/maps/mapsv13/dcv13ids.dat',
                        n_av = 7e-4):
        """
        Estimnate the galaxy counts in this Lya tomography surveys
            It is estimated by counting the avergae number of sightlines droping from z-deltaz/2
            to z+deltaz/2 where z is the redshift of the snapshot and deltaz is the redshift width
            of the simulation box. 
        Cautions :
            1. This ignores the buffer between start of the Lya forest and the galaxy's position
            2. Now, it only works for LATIS, I should re-design it to get any n(z)
        Returns:
            The number of galaxies that Lya forest secures their reshift in the
            simulated box
        """
        if n_av is None:
            self._get_redshift_width()
            init = sm.get_mock_sightline_number(z_range=[self.z - self.z_width/2, self.z], 
                                        pixfile=pixfile, mapfile=mapfile, idsfile=idsfile)
            final = sm.get_mock_sightline_number(z_range=[self.z , self.z + self.z_width/2], 
                                        pixfile=pixfile, mapfile=mapfile, idsfile=idsfile)
            self.galaxy_counts = int(init - final)
        self.galaxy_counts = np.product(self.boxsize)*n_av



