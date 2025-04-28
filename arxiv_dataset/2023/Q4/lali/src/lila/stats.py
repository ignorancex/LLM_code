import numpy as np
import h5py
import astropy.constants as const
from astropy.cosmology import Planck15 as cosmo
from nbodykit.lab import FFTPower
from nbodykit.lab import ArrayMesh
from nbodykit import utils
from lytomo_watershed import spectra_mocking as sm
import lim_lytomo
from . import git_handler
from nbodykit import setup_logging
import logging
import logging.config

class Stats():
    """A class to calculate the auto/cross power spectrum"""
    def __init__(self, z=None, mock_lim=None, mock_lya=None, mock_galaxy=None, kmin=None, kmax=1, dk=0.03, Nmu=30,
                k_par_min = None, los=[0,0,1], vol_ratio=1.0, Pco=None, save_3d_power=False):
        """
        Parameters
        ----------
            mock_lya, mock_lim : MockLya, MockLim instances, optional
                Instanses mocking the corresponding surveys
            kmin, kmax, dk : float, optional
                The limits and the steps for the power spectrum calculations
            los : (array_like , optional)
                the direction to use as the line-of-sight; must be a unit vector
            vol_ratio : float, Optional
                a correction factor to the # of available modes in uncertainty calculations. It is
                a quick correction for volume effect in power spectrum uncertainity calculation 
                if the simulated volume is few factors larger than the survey. The value is the
                ratio (Vol_survey/Vol_simulation)
            Pco : numpy array of floats, default= None
                The CO power spctrum. If None, it does calculate it.
        """

        setup_logging()
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        # create logger
        self.logger = logging.getLogger('Stats')
        self.logger.info('Starting')        
        self.mock_lim = mock_lim
        self.mock_lya = mock_lya
        self.mock_galaxy = mock_galaxy

        self.kmin = kmin
        if self.kmin is None:
            if self.mock_lim is not None:
                self.kmin =  1 / self.mock_lim.boxsize[0]
            elif self.mock_lya is not None:
                self.kmin =  1 / self.mock_lya.boxsize[0]
            elif self.mock_galaxy is not None:
                self.kmin =  1 / self.mock_galaxy.boxsize[0]
        self.kmax = kmax
        self.dk = dk 
        self.Nmu = Nmu
        self.k_par_min = k_par_min
        if self.k_par_min is not None:
            self.sigma_foreground = 1/self.k_par_min

        self.los = los
        self.vol_ratio = vol_ratio

        # Quantities to calculate
        self.lim_pk = Pco
        self.lim_pkmu = None
        self.lim_pk3d= None
        self.lya_pk = None
        self.lya_pkmu = None
        self.lya_pk3d= None
        self.gal_pk = None
        self.gal_pkmu = None
        self.gal_pk3d= None
        self.lim_lya_pk = None
        self.lim_lya_pkmu = None
        self.lim_lya_pk3d = None
        self.lim_gal_pk = None
        self.lim_gal_pkmu = None
        self.lim_gal_pk3d = None
        self.lim_noise_pk = None
        self.lya_noise_pk = None
        self.lim_noise_pk_av = None
        self.lya_noise_pk_av = None
        self.gal_noise_pk_av = None
        self.gal_noise_pk = None
        self.sigma_lim_pk = None
        self.sigma_lya_pk = None
        self.sigma_gal_pk = None
        self.sigma_lim_lya_pk = None
        self.sigma_lim_gal_pk = None
        self.lim_sn = None
        self.lya_sn = None
        self.gal_sn = None
        self.lim_lya_sn = None
        self.lim_gal_sn = None
        self.z = z
        if self.z is None:
            if self.mock_lim is not None:
                self.z = self.mock_lim.z
            elif self.mock_lya is not None:
                self.z = self.mock_lya.z
            elif self.mock_galaxy is not None:
                self.z = self.mock_galaxy.z
        self.save_3d_power = save_3d_power
        # MPI
        """
        self.MPI = CurrentMPIComm
        self.comm = self.MPI.get()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        """
        

    def get_lim_pk(self, mode='1d'):
        """Calculate the spehrically averaged CO temperature power spectrum"""
        if self.mock_lim.lim_map is None:
            self.mock_lim.get_lim_map()
        if not self.mock_lim.silent_mode:
            self.logger.info('Calculating the LIM auto power, mode: '+mode)
        
        return  FFTPower(self.mock_lim.lim_map, BoxSize=self.mock_lim.boxsize, 
                            mode=mode, kmin=self.kmin, kmax=self.kmax,dk=self.dk,
                            los=self.los, Nmu=self.Nmu, save_3d_power=self.save_3d_power)
    
    def get_lya_pk(self, savefile=None, mode='1d'):
        """Calculate the spehrically averaged deltaF power spectrum"""
        lya_map = self.mock_lya.get_map_noiseless()
        lya_mesh= ArrayMesh(lya_map, BoxSize= self.mock_lya.boxsize)
        if not self.mock_lya.silent_mode:
            self.logger.info('Calculating the Lya auto power, mode: '+mode)        
        return FFTPower(lya_mesh, mode=mode, BoxSize=self.mock_lya.boxsize,
                            kmin=self.kmin, kmax=self.kmax, dk=self.dk, 
                            Nmu=self.Nmu, los=self.los, save_3d_power=self.save_3d_power)


    def get_gal_pk(self, mode='1d'):
        """Calculate the spehrically averaged galaxy power spectrum"""
        if not self.mock_galaxy.silent_mode:
            self.logger.info('Calculating the galaxy power, mode: '+mode)
        return FFTPower(self.mock_galaxy.map, BoxSize=self.mock_galaxy.boxsize, 
                            mode=mode, kmin=self.kmin, kmax=self.kmax, dk=self.dk,
                            Nmu=self.Nmu, los=self.los, save_3d_power=self.save_3d_power)

    def get_lim_lya_pk(self, mode='1d'):
        """Calculate the spehrically averaged cross power spectrum between LIM LIM and Lya Tomography"""
        lya_map = self.mock_lya.get_map_noiseless()
        lya_mesh= ArrayMesh(lya_map, BoxSize= self.mock_lya.boxsize)
        if self.mock_lim.lim_map is None:
            self.mock_lim.get_lim_map()
        if not self.mock_lim.silent_mode:
            self.logger.info('Calculating the LIMXLya power, mode: '+mode)            
        return FFTPower(first=self.mock_lim.lim_map, second=lya_mesh, 
                                BoxSize=self.mock_lim.boxsize, mode=mode,
                                kmin=self.kmin, kmax=self.kmax, dk=self.dk,
                                Nmu = self.Nmu, los=self.los, save_3d_power=self.save_3d_power)
    
    def get_lim_gal_pk(self, mode='1d'):
        """Calculate the spehrically averaged cross power spectrum between CO LIM and Galaxy Overdensities"""
        if self.mock_lim.lim_map is None:
            self.mock_lim.compute_lim_map()
        if not self.mock_lim.silent_mode:
            self.logger.info('Calculating the COXgal power, mode: '+mode)            
        return  FFTPower(first=self.mock_lim.lim_map, second=self.mock_galaxy.map, 
                                BoxSize = self.mock_lim.boxsize, mode=mode, 
                                kmin=self.kmin, kmax=self.kmax, dk=self.dk, 
                                Nmu = self.Nmu, los=self.los, save_3d_power=self.save_3d_power)
        
    def get_uncertainty_lim_pk(self):
        """Get the uncertainty in the CO power spectrum which is:
            simgma_{P_CO} = (P_{CO}(k) + P_{n, LIM} ) ) / sqrt(N_{modes}
            - Some bins have 0 mode, so nan is returned
        """
        if self.lim_pk is None:
            fftpow = self.get_lim_pk(mode='1d')
            self.lim_pk = fftpow.power
            self.lim_pk3d = fftpow.pk3d
        if self.lim_pkmu is None:
            fftpow = self.get_lim_pk(mode='2d')
            self.lim_pkmu = fftpow.power
            self.lim_pk3d = fftpow.pk3d
        if self.lim_noise_pk is None:
            self.mock_lim.get_lim_noise_pk()
        w2kmu = self._attenuation(k=self.lim_pkmu['k'], mu=self.lim_pkmu['mu'], sig_par=self.mock_lim.res_par,
                                    sig_perp = self.mock_lim.res_perp/2, foreground_explode=True)
        self.sigma_lim_pkmu = (self.lim_pkmu['power'] + self.mock_lim.noise_pk/w2kmu)/(np.sqrt(self.vol_ratio*self.lim_pkmu['modes']))
        self.sigma_lim_pk = self.inverse_variance_weighting(self.lim_pk['k'], self.sigma_lim_pkmu)
        self.lim_noise_pk_av = self.inverse_variance_weighting(self.lim_pk['k'], self.mock_lim.noise_pk/w2kmu)
        self.logger.info('lim_noise_pk_av: '+str(self.lim_noise_pk_av))            

    def get_lya_noise_pk(self, k_par, k_perp=None):
        """Get Lya tomography  noise power spectrum which is estimated by
            comparing to the noiseless map. 
        
        Parameters:
        ------------------
        Returns: 
            The Lya noise power spectrum in units of (cMpc/h)^3 
        """
        self.lya_noise_pk = {}
        if not self.mock_lya.silent_mode:
            self.logger.info('getting 1D Pk for Lya')
        k ,lya_pk_los = self.get_1D_lya_pk()
        lya_pk_los = np.interp(k_par, k, lya_pk_los)
        n2D_eff = self.get_n2D_eff(lya_pk_los)
        noise =  lya_pk_los / n2D_eff
        if self.mock_lya.source_pk is not None:
            source_pk = np.interp(k_perp, self.mock_lya.source_pk['k_perp'], self.mock_lya.source_pk['power'])
            noise *= (1 + source_pk/(self.mock_lya.dp**2))
        return noise
        

    
    def get_n2D_eff(self, lya_pk_los, lam0=1216):
        """Get the effecvtive 2D shightline density of spectra. eq 13. in McQuinn+White+11 (arxiv:1102.1752)
        Prameters:
        ------------------------------
        lya_pk_los : The line-of-sight power spectrum
        lam0 : The Lya resframe wavelength
        """
        self.logger.info('H: '+str(self.mock_lya.cosmo.H(self.z)))
        self.logger.info('h: '+str(self.mock_lya.cosmo.h))
        self.logger.info('sn: '+str(self.mock_lya.sn))
        P_N = (sm.get_mean_flux(self.z)/self.mock_lya.sn)**2 *  lam0 * (self.mock_lya.cosmo.H(self.z)*self.mock_lya.cosmo.h/const.c.to('km/s')).value
        P_N = 0
        q = lya_pk_los / (lya_pk_los + P_N)
        return (1/(self.mock_lya.dperp**2))*q

    
    def get_uncertainty_lya_pk(self, savefile=None):
        """Get the uncertainty in the CO power spectrum which is:
            simgma_{P_Lya} = (P_{Lya}(k) + P_{n, Lya} ) / sqrt(N_{modes})
        """
        if self.lya_pk is None:
            fftpow = self.get_lya_pk(mode='1d')
            self.lya_pk = fftpow.power
            self.lya_pk3d = fftpow.pk3d
        if self.lya_pkmu is None:
            fftpow = self.get_lya_pk(mode='2d')
            self.lya_pkmu = fftpow.power
            self.lya_pk3d = fftpow.pk3d
        if self.lya_noise_pk is None:
            k_par = np.ravel(self.lya_pkmu['k']*self.lya_pkmu['mu'])
            self.lya_noise_pk = self.get_lya_noise_pk(k_par).reshape(self.lya_pkmu['power'][:].shape)

        self.sigma_lya_pkmu = (self.lya_pkmu['power'] + self.lya_noise_pk)/(np.sqrt(self.vol_ratio*self.lya_pkmu['modes']))
        if not self.mock_lya.silent_mode:
            self.logger.info('inverse var weighting, auto Lya')
        self.sigma_lya_pk = self.inverse_variance_weighting(self.lya_pk['k'], self.sigma_lya_pkmu)
        # save P_noise,Lya just in case:
        self.lya_noise_pk_av = self.inverse_variance_weighting(self.lya_pk['k'], self.lya_noise_pk)

    def get_gal_noise_pk(self):
        """Get galaxy noise power spectrum which is a poisson noise from the discrite
           galaxy distribution.
        Returns: 
            The Galaxy noise power spectrum in units of (cMpc/h)^3 
        """
        # Save the 3D number density of the galaxies since it will be 
        # used as the shot noise in the power spectrum modelling :
        self.n3D = self.mock_galaxy.galaxy_count/np.product(self.mock_galaxy.boxsize)
        self.gal_noise_pk = 1/self.n3D


        
    def get_uncertainty_gal_pk(self):
        """Get the uncertainty in the galaxy power spectrum which is:
            simgma_{P_gal} = (P_{gal}(k) + P_{n, gal} ) / sqrt(N_{modes})
        """
        if self.gal_pk is None:
            fftpow = self.get_gal_pk(mode='1d')
            self.gal_pk = fftpow.power
            self.gal_pk3d = fftpow.pk3d
        if self.gal_pkmu is None:
            fftpow = self.get_gal_pk(mode='2d')
            self.gal_pkmu = fftpow.power
            self.gal_pk3d = fftpow.pk3d
        if self.gal_noise_pk is None:
            self.get_gal_noise_pk()
        w2kmu = self._attenuation(k=self.gal_pkmu['k'], mu=self.gal_pkmu['mu'], sig_par=self.mock_galaxy.res_par,
                                                                                sig_perp = self.mock_galaxy.res_perp)
        self.sigma_gal_pkmu = (self.gal_pkmu['power'] + self.gal_noise_pk/w2kmu)/(np.sqrt(self.vol_ratio*self.gal_pkmu['modes']))        
        if not self.mock_galaxy.silent_mode:
            self.logger.info('inverse var weighting, auto gal')
        self.sigma_gal_pk = self.inverse_variance_weighting(self.gal_pk['k'], self.sigma_gal_pkmu)
        # save P_noise,Gal just in case:
        self.gal_noise_pk_av = self.inverse_variance_weighting(self.gal_pk['k'], self.gal_noise_pk/w2kmu)

  
    def get_uncertainty_lim_lya(self, sigma_lya_pk_file=None):
        """Get the uncertainty in the CO LIM * Lya Tomo cross power spectrum. Eq 12 in Chung+18.
        """
        if self.lim_lya_pk is None:
            fftpow = self.get_lim_lya_pk(mode='1d')
            self.lim_lya_pk = fftpow.power
            self.lim_lya_pk3d = fftpow.pk3d
        if self.lim_lya_pkmu is None:
            fftpow = self.get_lim_lya_pk(mode='2d')
            self.lim_lya_pkmu = fftpow.power
            self.lim_lya_pk3d = fftpow.pk3d
        if self.sigma_lim_pk is None:
            self.get_uncertainty_lim_pk()
        if self.sigma_lya_pk is None:
            self.get_uncertainty_lya_pk()

        self.sigma_lim_lya_pkmu = np.abs(np.sqrt( (self.sigma_lim_pkmu*self.sigma_lya_pkmu)/2+ 
                                (self.lim_lya_pkmu['power']**2)/(2*self.vol_ratio*self.lim_lya_pkmu['modes'])))
        if not self.mock_lim.silent_mode:
            self.logger.info('inverse var weighting, COXLya')
        self.sigma_lim_lya_pk = self.inverse_variance_weighting(self.lim_lya_pk['k'], self.sigma_lim_lya_pkmu)
    
    def get_uncertainty_lim_gal(self):
        """Get the uncertainty in the CO LIM x galaxy survey power spectrum. Eq 12 in Chung+18.
        """
        if self.lim_gal_pk is None:
            fftpow = self.get_lim_gal_pk(mode='1d')
            self.lim_gal_pk = fftpow.power
            self.lim_gal_pk3d = fftpow.pk3d
        if self.lim_gal_pkmu is None:
            fftpow = self.get_lim_gal_pk(mode='2d')
            self.lim_gal_pkmu = fftpow.power
            self.lim_gal_pk3d = fftpow.pk3d
        if self.sigma_lim_pk is None:
            self.get_uncertainty_lim_pk()
        if self.sigma_gal_pk is None:
            self.get_uncertainty_gal_pk()

        self.sigma_lim_gal_pkmu = np.abs(np.sqrt( (self.sigma_lim_pkmu*self.sigma_gal_pkmu)/2+ 
                                (self.lim_gal_pkmu['power']**2)/(2*self.vol_ratio*self.lim_gal_pkmu['modes'])))
        if not self.mock_lim.silent_mode:
            self.logger.info('inverse var weighting, COXgal')
        self.sigma_lim_gal_pk = self.inverse_variance_weighting(self.lim_gal_pk['k'], self.sigma_lim_gal_pkmu)

    def get_lim_sn(self):
        """Get the signal to noise of the CO LIM auto power in each k
        """
        if self.sigma_lim_pk is None:
            self.get_uncertainty_lim_pk()
        self.lim_sn = np.abs(self.lim_pk['power']/self.sigma_lim_pk)
    
    def get_lya_sn(self):
        """Get the signal to noise of the Lya tomography auto power in each k
        """
        if self.sigma_lya_pk is None:
            self.get_uncertainty_lya_pk()
        self.lya_sn = np.abs(self.lya_pk['power']/self.sigma_lya_pk)
    
    def get_gal_sn(self):
        """Get the signal to noise of the galaxy survey auto power in each k
        """
        if self.sigma_gal_pk is None:
            self.get_uncertainty_gal_pk()
        self.gal_sn = np.abs(self.gal_pk['power']/self.sigma_gal_pk)
    
    def get_lim_lya_sn(self, sigma_lya_pk_file=None):
        """Get the signal to noise of the cross power in each k
        """
        if self.sigma_lim_lya_pk is None:
            self.get_uncertainty_lim_lya(sigma_lya_pk_file=sigma_lya_pk_file)
        self.lim_lya_sn = np.abs(self.lim_lya_pk['power']/self.sigma_lim_lya_pk)
    
    def get_lim_gal_sn(self):
        """Get the signal to noise of the CO LIM x Galaxy survey cross power in each k
        """
        if self.sigma_lim_gal_pk is None:
            self.get_uncertainty_lim_gal()
        self.lim_gal_sn = np.abs(self.lim_gal_pk['power']/self.sigma_lim_gal_pk)

    def _attenuation(self,k, mu, sig_par, sig_perp, foreground_explode = False):
        """Signal attenuation due to finite resolution.
            The spherically averaged auto/cross power spectra should be correctd with
            this window function :
                        W(k)^2 = <exp(-k_{\perp}^2 sigma_{\perp}^2)>
                        W(k)^2 = <exp(-k_{\||}^2 sigma_{\||}^2)>
            The averaging is done within the shells used to calculaete P(k). Basically, implemnting
            Appendix C.3 in Lee+16 and eq 19 and 20 in Chung+18.
            
        Parameters: K : ndarray
                        an array of ks to get the attenuation at.
                    sig_par: 
                        resolution along line-of-sight in cMpc/h (units should match k's unit)
                    sig_perp : 
                        resolution along perp direction
        Returns: ndarray
            W(k)^2 : multiply the output by the auto/cross power spectrum
        """
        foreground = np.ones_like(k)
        if foreground_explode and (self.k_par_min is not None):
            ind = np.where(mu*k <= self.k_par_min)
            if ind[0].size != 0 :
                foreground[ind] *= 1e-9
                #foreground[ind] *= np.exp(-((mu[ind]*k[ind] - self.k_par_min)*self.sigma_foreground)**2 )
                #foreground[ind] *=  np.exp(-(mu[ind]*k[ind] / self.k_par_min)**2 )
                self.logger.info('Foreground effects, median correction=%s', np.median(foreground[ind]))

        return np.exp(-(k*sig_perp)**2) * np.exp(- (mu*k)**2 * (sig_par**2 - sig_perp**2)) * foreground
    
    def _get_kcut(self):
        if not self.mock_lim.silent_mode:
            self.logger.info('Cut foreground, i.e k < %s', self.k_par_min)
        def _foreground_exp(k_par):
            return np.zeros_like(k_par)
            #return np.exp(-((k_par - self.k_par_min)*self.sigma_foreground)**2 )
        kcut={'k_par_min':self.k_par_min}
        kcut['func'] = _foreground_exp

        return kcut

    def _do_save_power(self, p, savefile):
        """Save the power spectrum on file
        Parameters:
            p : A dictionary with keys: 'power' and 'k'
        """
        lim_pk = utils.GatherArray(data=np.abs(p['power']), comm=self.comm, root=0)
        k = utils.GatherArray(data=p['k'], comm=self.comm, root=0)
        if self.rank == 0:
            with h5py.File(savefile, 'w') as fw:
                fw['power'] = np.abs(p['power'][:])
                fw['k'] = p['k'][:]
    
    def get_1D_lya_pk(self):
        """Get 1D flux power spectrum, to be used for a simple lya noise mdoel McQuinn+14 eq 12,13 arxiv:1102.1752 """
        from fake_spectra import fluxstatistics as fs
        from lytomo_watershed import spectra_mocking as sm
        with h5py.File(self.mock_lya.spec_file,'r') as f:
            tau = f['tau/H/1/1215'][:]
            mf = sm.get_mean_flux(z=f['Header'].attrs['redshift'])
        pk1d = fs.flux_power(tau, vmax = self.mock_lya.boxsize[2], mean_flux_desired=mf, window=False)
        return pk1d
    
    def inverse_variance_weighting(self, k, sigma_pkmu):
        """calculate the sigma_pk with the inverse variance weighting in mu bins
            eq 6 in Furlanetto, Lidz 2014 arxiv:0611274
            Paramters:
            --------------------
            k : k-bins
            sigma_pkmu : numpy array, the uncertainty in each (k, mu) bins
            Returns :
                The weighted noise in '1D' k-space.
            """
        sigma_pk = 1 / np.sqrt(np.nansum(1/np.abs(sigma_pkmu)**2, axis=1))
        return sigma_pk

    def save_stat(self, savefile):
        """Save all computed statistics on an hdf5 file
        Parameters:
        -------------------
        savefile : an hdf5 file path
        """
        from . import comap, exclaim

        self.logger.info('saving at %s', savefile)
        with h5py.File(savefile, 'w') as fw:
            if self.lim_pk is not None:
                try:
                    for k, v in self.mock_lim.survey_params.items():
                        fw[f'lim_survey_params/{k}'] = v
                except:
                    pass
                fw['lim_pk/power'] = self.lim_pk['power']
                fw['lim_pk/k'] = self.lim_pk['k']
                fw['lim_pk/modes'] = self.lim_pk['modes']
                try:
                    fw['lim_pk/noise'] = self.lim_noise_pk_av
                except:
                    pass
            # Write down the LIM emission model parameters
            if isinstance(self.mock_lim, comap.MockComap):
                if self.mock_lim.co_model == 'COMAP+21':
                    for k, v in self.mock_lim.COMAP21_params.items():
                        fw[f'lim_model/COMAP+21/{k}'] = v
                if self.mock_lim.co_model == 'Li+16':
                    for k, v in self.mock_lim.Li16_params.items():
                        fw[f'lim_model/Li+16/{k}'] = v
            if isinstance(self.mock_lim, exclaim.MockExclaim):
                if self.mock_lim.co_model == 'Padmanabhan+19':
                    for k, v in self.mock_lim.padmanabhan19_params.items():
                        fw[f'lim_model/padmanabhan+19/{k}'] = v

            if self.lya_pk is not None:
                fw['lya_dperp'] = self.mock_lya.dperp
                fw['lya_pk/power'] = self.lya_pk['power']
                fw['lya_pk/k'] = self.lya_pk['k']
                fw['lya_pk/modes'] = self.lya_pk['modes']
                if self.save_3d_power:
                    fw['lya_pk3d/power'] = self.lya_pk3d[:]
                if self.lya_noise_pk_av is not None:
                    fw['lya_pk/noise'] = self.lya_noise_pk_av

            if self.lya_pkmu is not None:
                fw['lya_pkmu/power'] = self.lya_pkmu['power']
                fw['lya_pkmu/k'] = self.lya_pkmu['k']
                fw['lya_pkmu/mu'] = self.lya_pkmu['mu']
                fw['lya_pkmu/modes'] = self.lya_pkmu['modes']

            if self.gal_pk is not None:
                try:
                    fw['gal_Rz'] = self.mock_galaxy.Rz
                except:
                    pass
                fw['gal_pk/power'] = self.gal_pk['power']
                fw['gal_pk/k'] = self.gal_pk['k']
                fw['gal_pk/modes'] = self.gal_pk['modes']
                fw['gal_pk/n3D'] = self.n3D
            if self.gal_pkmu is not None:
                fw['gal_pkmu/power'] = self.gal_pkmu['power']
                fw['gal_pkmu/k'] = self.gal_pkmu['k']
                fw['gal_pkmu/mu'] = self.gal_pkmu['mu']
                fw['gal_pkmu/modes'] = self.gal_pkmu['modes']
            
                

            if self.sigma_lim_pk is not None:
                fw['sigma_lim_pk'] = np.abs(self.sigma_lim_pk)
            if self.sigma_lya_pk is not None:
                fw['sigma_lya_pk'] = np.abs(self.sigma_lya_pk)
            if self.sigma_gal_pk is not None:
                fw['sigma_gal_pk'] = np.abs(self.sigma_gal_pk)
                if self.gal_noise_pk_av is not None:
                    fw['gal_pk/noise'] = self.gal_noise_pk_av

            if self.lim_lya_pk is not None:
                fw['lim_lya_pk/power'] = self.lim_lya_pk['power']
                fw['lim_lya_pk/k'] = self.lim_lya_pk['k']
                fw['lim_lya_pk/modes'] = self.lim_lya_pk['modes']
            if self.lya_pkmu is not None:
                fw['lim_lya_pkmu/power'] = self.lim_lya_pkmu['power']
                fw['lim_lya_pkmu/k'] = self.lim_lya_pkmu['k']
                fw['lim_lya_pkmu/mu'] = self.lim_lya_pkmu['mu']
                fw['lim_lya_pkmu/modes'] = self.lim_lya_pkmu['modes']

            if self.lim_gal_pk is not None:
                fw['lim_gal_pk/power'] = self.lim_gal_pk['power']
                fw['lim_gal_pk/k'] = self.lim_gal_pk['k']
                try:
                    fw['lim_gal_pk/modes'] = self.lim_gal_pk['modes']
                except: pass
            if self.lim_gal_pkmu is not None:
                fw['lim_gal_pkmu/power'] = self.lim_gal_pkmu['power']
                fw['lim_gal_pkmu/k'] = self.lim_gal_pkmu['k']
                fw['lim_gal_pkmu/mu'] = self.lim_gal_pkmu['mu']
                fw['lim_gal_pkmu/modes'] = self.lim_gal_pkmu['modes']

            if self.sigma_lim_lya_pk is not None:
                fw['sigma_lim_lya_pk'] = np.abs(self.sigma_lim_lya_pk)
            if self.sigma_lim_gal_pk is not None:
                fw['sigma_lim_gal_pk'] = np.abs(self.sigma_lim_gal_pk)

            # Save the commit hash of the code used for this run
            # when reading this convert it to a list with :
            # `f['Git'].attrs['HEAD_HASH'].tolist()`
            fw.create_group('Git')
            head_hash = git_handler.get_head_hash(lim_lytomo)
            fw['Git'].attrs["HEAD_HASH"] = np.void(head_hash)
    
    def load_stat(self, stat_file, load_2D_stats=False, load_3D_stats=False):
        """ Load the stats from an `hdf5` file
        Parameters:
        stat_file: the file address to load the file from
        --------------------------
        Returns: an instance of stats.Stat()
        """
        with h5py.File(stat_file,'r') as f:
            try:
                self.lim_pk = {}
                self.lim_pkmu = {}
                self.lim_pk['k'] = f['lim_pk/k'][:]
                self.lim_pk['modes'] = f['lim_pk/modes'][:]
                self.lim_pk['power'] = f['lim_pk/power'][:]
                try:
                    self.lim_noise_pk = f['lim_pk/noise'][:]
                except:
                    pass
                self.sigma_lim_pk = f['sigma_lim_pk'][:]/np.sqrt(self.vol_ratio)
                self.get_lim_sn()
                if load_2D_stats:
                    try:
                        self.lim_pkmu['mu'] = f['lim_pkmu/mu'][:]
                        self.lim_pkmu['modes'] = f['lim_pkmu/modes'][:]
                        self.lim_pkmu['power'] = f['lim_pkmu/power'][:]
                    except:
                        self.logger.info('P_LIM(k,mu) is not stored.')
            except:
                self.lim_pk = None
                self.lim_pkmu = None
                self.logger.info('Could not load LIM')
                pass
            try:
                self.lya_pk = {}
                self.lya_pkmu = {}
                self.lya_pk3d = {}
                self.lim_lya_pk = {}
                self.lya_pk['k'] = f['lya_pk/k'][:]
                self.lya_pk['modes'] = f['lya_pk/modes'][:]
                self.lya_pk['power'] = f['lya_pk/power'][:]
                try:
                    self.lya_noise_pk = f['lya_pk/noise'][:]
                except:
                    pass
                try:
                    self.sigma_lya_pk = f['sigma_lya_pk'][:]/np.sqrt(self.vol_ratio)
                    self.get_lya_sn()
                except:
                    self.logger.info('No Lya noise stats')

                if load_2D_stats:
                    try:
                        self.lya_pkmu['mu'] = f['lya_pkmu/mu'][:]
                        self.lya_pkmu['modes'] = f['lya_pkmu/modes'][:]
                        self.lya_pkmu['power'] = f['lya_pkmu/power'][:]
                    except:
                        self.logger.info('P_Lya(k,mu) is not stored.')
                if load_3D_stats:
                    try:
                        self.lya_pk3d['power'] = f['lya_pk3d/power'][:]
                    except:
                        pass
                
                try:
                    self.lim_lya_pk['k'] = f['lim_lya_pk/k'][:]
                    self.lim_lya_pk['power'] = f['lim_lya_pk/power'][:]
                    self.sigma_lim_lya_pk = f['sigma_lim_lya_pk'][:]/np.sqrt(self.vol_ratio)
                    self.get_lim_lya_sn()
                except:
                    self.logger.info('No LIMXLya stats')
                    pass

            except:
                self.lya_pk = None
                self.lya_pkmu = None
                self.logger.info('No Lya stats')
                pass
            try:
                self.gal_pk = {}
                self.gal_pkmu = {}
                self.lim_gal_pk = {}
                self.gal_pk['k'] = f['gal_pk/k'][:]
                self.gal_pk['modes'] = f['gal_pk/modes'][:]
                self.gal_pk['power'] = f['gal_pk/power'][:]
                try:
                    self.n3D = f['gal_pk/n3D'][()]
                except:
                    pass
                try:
                    self.gal_noise_pk = f['gal_pk/noise'][:]
                except:
                    pass

                self.sigma_gal_pk = f['sigma_gal_pk'][:]/np.sqrt(self.vol_ratio)
                self.get_gal_sn()
                if load_2D_stats:
                    try:
                        self.gal_pkmu['mu'] = f['gal_pkmu/mu'][:]
                        self.gal_pkmu['modes'] = f['gal_pkmu/modes'][:]
                        self.gal_pkmu['power'] = f['gal_pkmu/power'][:]
                    except:
                        self.logger.info('P_Lya(k,mu) is not stored.')
                
                try:
                    self.lim_gal_pk['k'] = f['lim_gal_pk/k'][:]
                    self.lim_gal_pk['power'] = f['lim_gal_pk/power'][:]
                    self.sigma_lim_gal_pk = f['sigma_lim_gal_pk'][:]/np.sqrt(self.vol_ratio)
                    
                    self.get_lim_gal_sn()
                except:
                    self.logger.info('No LIMXGal stats')
                    pass

            except:
                self.gal_pk = None
                self.gal_pkmu = None
                self.logger.info('No Galaxy stats')
                pass
            
