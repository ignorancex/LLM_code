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
from nbodykit.lab import ArrayMesh, HDFCatalog, CurrentMPIComm



class MockLim():
    """A class for mock Line Intensit Map """
    def __init__(self, snap, survey_params,  axis=3, basepath=None, boxsize=None, brange =None, 
                 fine_Nmesh=None, seed=None, halo_type='Subhalo', sim_type= 'TNG',
                 compensated=True, rsd=True, mass_cut=None, silent_mode=True, sfr_file=None, noise_pk=None):
        """
        Generate a mock Line intensity maps from a hydro/DM-only simulation.
        The default arguments are for COMAP Early science(Chung+21)
        Paramterts:
            snap : Snapshot id, int
            axis : Either 1 or 2 or 3 indicating the line of sight axis, x or y or z
            basepath: The path to the directory of postprocessed data of the simulations
            sfr_file : The file of averaged star formation in the subhalos. Not rquired if 
                        the instantaneous sfr is used.
            fine_Nmesh : (array_like, shape=(3,) ) number of grid mesh cells along each axis. It should match that of Lya/galaxy maps.
            brange: (array_like, shape=(6,)) [xmin,xmax,ymin,ymax,zmin,zmax], subbox coordinates 
                    in cMpc/h of the simulation to make mock for
            boxsize : int or tuple of 3 ints
                    Simulation box Size in comoving  cMpc/h
            freq_res : Frequency resolution in MHz
            beam_fwhm : Beam FWHM in arcminutes
            tempsys : System temperature in K
            nfeeds: Bumber of feeds
            deltanu : Frequency resolution in MHz
            patch : survey area per patch in dega^2
            tobs : Total survey time on this patch in hours
            noise_per_voxel : Noise per voxel in \mu K. Defult on the  COMAP+21 Y5 projection, 
                        Table 4 in Chung+21 Arxiv:2111.05931. It overrides the calculation from 
                        the sys temperature. 
            nu_rest : rest frequncy of the line in GHZ
            nu_co_rest : rest frequncy of the CO line in GHZ
            Li16_params: Optional, used only if co_model=='Li16'. The model parameters for painting CO
                         emission on halo catalog. Li+16: arxiv:1503.08833
                         alpha, beta, sigma_co : The partamters in converting L_IR to L_CO, values from
                                                Chung et. al. 2018 arxiv:1809.04550 and
                                                
                         delta_mf : The paramter to convert star formation rate to L_IR
            COMAP21_params : Optiona, only if co_model=='COMAP21'. The model parameters for painting CO
                             emission on halo catalog. Chung et. al. 2021 arxiv:2111.05931
                             
            seed : Random seed to generate the scatter in L_CO
            halo_type : Either 'Subhalo' or 'Group'
            sfr_type: The options for sfr_type are :
                    'SFR_MsunPerYrs_in_InRad_50Myrs', 'SFR_MsunPerYrs_in_InRad_100Myrs',
                    'SFR_MsunPerYrs_in_InRad_200Myrs', 'SubahaloSFR', 'GroupSFR' ,'behroozi+13'
                     The first 3 are average star formation rates from Donanari+2019 and Pillepich+2019
                     The 4th and 5th are the instantaneous start fromation in hydro TNG
                     The last is the average sfr-vs-DM halo mass relation from behroozi+13
            co_model : st, Fiducial model (prior) for COMAP. Options are :
                        'Li+16' : A model based on Li+16
                        'COMAP+21'  : A model based on chung+21. COMAP early science fiducial model
                                     which is a data driven prior (UM+COLDz+COPSS): Arxiv:2111.05931                   
            behroozi_avsfr : str, path to average SFR from behroozi+13
            sim_type : str, simulation type either 'ASTRID', 'TNG' or 'MDPL2'
            compensated: If True, makes the correction in the fourier space of the field interpolated
                         by CIC.
            mass_cut : Default 10, log of min halo mass in M_sol/h units. If not None, only halos with 
                        log_10(M_halo) > mass_cut [M_sol/h] emmit CO
            rsd : whether to apply redshift space distortion to along line-of-sight
        """
        # MPI
        self.MPI = CurrentMPIComm
        self.comm = self.MPI.get()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.snap = snap
        self.axis= axis
        self.sight_vec = [0,0,0]
        self.sight_vec[int(axis-1)] = 1
        self.brange = brange
        self.sfr = []
        self.halos = {}
        self.groups = {}
        self.halo_id = []
        self.lim_map = None
        self.basepath = basepath
        if self.basepath is None:
            self.halo_file = None
        else :
            self.halo_file = os.path.join(basepath, 'groups_'+str(self.snap)+'/fof*')
        self.sfr_file = sfr_file
        if boxsize is None:
            self.boxsize = (brange[1]-brange[0],
                             brange[3]-brange[2],
                             brange[5]-brange[4])
        else:
            self.boxsize = boxsize

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.halo_type = halo_type
        self.sim_type = sim_type

        self.compensated = compensated
        self.mass_cut = mass_cut
        self.rsd = rsd
        self.silent_mode =silent_mode
        if not self.silent_mode:
            setup_logging()
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
            # create logger
            self.logger = logging.getLogger('MockLim')
            self.logger.info('Starting')
        ## Suvey parameters
        self.survey_params  = survey_params
        self.noise_pk = noise_pk
        

        self.fine_Nmesh = fine_Nmesh
        # Comoving volume of a voxel in (Mpc/h)^3
        self.fine_vol_vox = np.product(self.boxsize) / np.product(np.product(self.fine_Nmesh))
        # Laod halos on an Nbodykit catalog
        if self.halo_file is not None:
            self._load_halos()
        else:
            if not self.silent_mode:
                self.logger.warning("Not making the LIM map, if you would like so set the basepath for the simualtion.")
        self.res_par = self.get_res_par()
        self.res_perp = self.get_res_perp()
        self.Nmesh = [np.around(self.boxsize[0]/(self.res_perp)), 
                    np.around(self.boxsize[1]/(self.res_perp)),
                    np.around(self.boxsize[2]/(self.res_par))]
        self.vol_vox = np.product(self.boxsize) / np.product(np.product(self.Nmesh))
        if not self.silent_mode:
            self.logger.info('LIM :')
            self.logger.info('resolution : perp= %s, par= %s, boxsize=%s', self.res_perp, self.res_par, self.boxsize)
            self.logger.info('Nmesh = %s', str(self.Nmesh))
        if self.halo_file is not None:
            self.get_lim_map()
            del self.halos

    def get_res_par(self):
        """Calculate the spatial resolution along the lne-of-sight in units of comoving Mpc/h
        freq_res is the frequncy resolution of the survey."""
        raise NotImplementedError
        
    def get_res_perp(self):
        """Calculate the spatial resolution in transverse direction to the line-of-sight in
        comoving Mpc/h.
        The angular resolution of the survey is in arcminutes"""
        if not 'angular_res' in self.survey_params:
            self.survey_params['angular_res'] = 2*self.survey_params['beam_fwhm']/np.sqrt(8*np.log(2))
        return cosmo.h*cosmo.comoving_distance(z=self.z).value*self.survey_params['angular_res']*np.pi/(180*60)
    
    def get_halo_luminosity(self):
        """ A function which takes halo masses and returns the luminosity of each in L_solar.
        Should be overwritten by the function in each instance of this class.
        """
        raise NotImplementedError

    def get_lim_map(self):
        raise NotImplementedError

    def get_voxel_luminosity(self):
        """
        Calculate the 3D CO luminotity map on a unifrom grid of self.fine_Nmesh.

        Returns : 
        numpy array shape=self.fine_Nmesh: CO luminosity map in L_sun unit 
        """
        halo_lum = self.get_halo_luminosity()
        # CIC interpolation : NOT SURE THIS IS THE BEST CHOICE
        voxel_lum_mesh = self.do_cic(q=halo_lum, qlabel='halo_lum')
        return voxel_lum_mesh
    
    def do_cic(self, q, qlabel):
        """Similar to LyTomo_watershed.get_Density()
        CIC a sfr_type on a regular mesh grid
        q : The quantity to paint on mesh
        """
        if self.halos['Coordinates'].size ==0:
            raise IOError('No halos on Rank '+str(self.rank))
        
        if self.rsd:
            self._apply_rsd()
        
        qtemp = np.zeros(shape=(self.halos['Coordinates'][:].shape[0],))
        qtemp[self.halo_id] = q
        self.halos[qlabel] = qtemp
        if not self.silent_mode:
            self.logger.info('self.fine_Nmesh = %s', self.fine_Nmesh)
            #self.logger.info('Max Coord ', np.max(self.halos['Coordinates'].compute()))
            #self.logger.info('BoxSize :', self.boxsize)
            self.logger.info('compensated = %s', self.compensated)

        assert(np.all(qtemp >= 0))
        mesh = self.halos.to_mesh(Nmesh=self.fine_Nmesh, position='Coordinates', value=qlabel,
                                  BoxSize=self.boxsize, compensated=self.compensated)

        correc_fac = self.halos.csize / np.prod(self.fine_Nmesh)
        if not self.silent_mode:
            self.logger.info('q/<q> to q correction factor = %s', correc_fac)
        delta_to_value = lambda x,y : y*correc_fac
        mesh = mesh.apply(delta_to_value, kind='index', mode='real')
        #assert (np.all(mesh.compute() >= 0))
        
        return mesh
    
    def _apply_rsd(self):
        """Apply Redshift space distortion, for sub/halos the velocities are in km/s"""
        if not self.silent_mode:
            self.logger.info('applying rsd')
        # Note: in Subfind,  halo velocities do not have the extra sqrt(a) fcator
        if self.halo_type=='Subhalo':
            self.halos['Coordinates'] = (self.halos['Coordinates']+
                                         (self.halos[self.halo_type+'Vel']*self.sight_vec*
                                          cosmo.h/cosmo.H(self.z).value))%self.boxsize
        elif self.halo_type=='Group':
            ## I need to add a new formula since the vel is in km/s/a units
            raise NotImplemented 
        if not self.silent_mode:
            self.logger.info('RSD applied')


    def _load_halos(self):
        """Load the subhalo catalogue using nbodykit
        The catalouge is not loaded on memory since it uses Dask
        """
        if not self.silent_mode:
            self.logger.info('Loading Halo Catalogue')
        fn = glob.glob(self.halo_file)[0]
        with h5py.File(fn,'r') as f:
            self.z = f['Header'].attrs['Redshift']
        self.halos = HDFCatalog(self.halo_file, dataset=self.halo_type)
        if not self.silent_mode:
            self.logger.info('Halo Cat is loaded ')
        
        # Some unit corrections are needed for each simulation
        if self.sim_type=='MDPL2':
            self.halos['SubhaloMass'] = self.halos['SubhaloMass']/1e10

        if self.sim_type=='TNG' or self.sim_type == 'ASTRID':
            # for TNG convert ckpc/h tp cMpc/h
            self.halos['Coordinates'] = self.halos['SubhaloPos']/1e3

        if self.mass_cut is not None:
            ind = da.greater(self.halos['SubhaloMass'] , 10**(self.mass_cut-10))
            self.halos = self.halos[ind]
        
        if self.brange is not None:
            ind = self.halos['Coordinates'] >= np.array([[ self.brange[0],self.brange[2], self.brange[4] ]])
            ind *= self.halos['Coordinates'] <= np.array([[ self.brange[1],self.brange[3], self.brange[5] ]])
            ind = da.prod(ind, axis=1, dtype=bool)
            self.halos = self.halos[ind]
        
    def _load_sfr(self):
        """Read the sfr of the subhalos
        depending on the sfr_type passed to it, it either uses the mean
        SFR or the instantaneous SFR.
        """
        # Look at the mean SFR table
        if self.Li16_params['sfr_type'][0:7]=='SFR_Msu':
            with h5py.File(self.sfr_file,'r') as f:
                self.sfr = f['Snapshot_'+str(self.snap)][self.sfr_type][:]
                self.halo_id = f['Snapshot_'+str(self.snap)]['SubfindID'][:]
        # Look at the instantaneous SFR in subhalos
        elif self.Li16_params['sfr_type'] == 'SubhaloSFR':
            self.sfr = self.halos['SubhaloSFR'].compute()
            self.halo_id = np.arange(self.sfr.size)
            if self.mass_cut is not None:
                ind = np.where(self.halos['SubhaloMass'] >= 10**(self.mass_cut-10))
                self.sfr = self.sfr[ind]
                self.halo_id = self.halo_id[ind]
        # Look at the instantaneous SFR in halos
        elif self.Li16_params['sfr_type'] == 'GroupSFR':
            self.sfr = self.halos['GroupSFR'].compute()
            self.halo_id = np.arange(self.sfr.size)
        # Look at the average SFR -vs halo masss from Behroozi+13
        elif self.Li16_params['sfr_type'] == 'behroozi+13':
            self.sfr = self.sfr_behroozi()
            self.halo_id = np.arange(self.sfr.size)
            if self.mass_cut is not None:
                ind = np.where(self.halos['SubhaloMass'].compute() >= 10**(self.mass_cut-10))
                self.sfr = self.sfr[ind]
                self.halo_id = self.halo_id[ind]
        else:
            raise NameError("Wrong sfr_type")
            
    def sfr_behroozi(self):
        """Get the sfr for halos from average sfr-DM halo Mass relation from behroozi+13.
            This is the method adopted by Chung+18 and Li+16 for making mock CO LIM
        """
        intp = self._sfr_behroozi_interpolator()
        # Masses in TNG are in M_sol /h but in Behroozi are in M_sol
        hmass = 10+np.log10(self.halos[self.halo_type+'Mass'].compute())-np.log10(cosmo.h)
        return 10**(intp(hmass))
    
    def _sfr_behroozi_interpolator(self):
        """Set up the 2D interpol"""
        from scipy.interpolate import interp1d
        #bh = self._load_behroozi_data(zrange=(self.z-0.05, self.z+0.05))
        bh = self._load_behroozi_data()
        intp = interp1d(bh[:,1], bh[:,2], kind='linear', fill_value='extrapolate')
        return intp      
    
    def _load_behroozi_data(self, zrange=None):
        
        if self.Li16_params['behroozi_avsfr'] is None:
            raise IOError("pass the path for behroozi+13 average sfr-vs- halo mass relation")
        bh = np.loadtxt(self.Li16_params['behroozi_avsfr'])
        bh[:,0] -= 1
        zdiff  = np.abs(bh[:,0] - self.z)
        ind = np.where(zdiff==zdiff.min())[0]
        return bh[ind,:]
    
    
    def upsample(self, co_temp, method='linear', final_shape = (205,205,205)):
        """Upsample (increase the resolution) for the CO map by interpolation. It is not used for
        power spectrum calculations.
        
        Parameters:
        ----------------------------
        co_temp : The low res map
        method : Either 'linear' or 'nearest'
        final_shape : The high-res map's shape

        Returns : 
        -------------------
        The hig-res map
        """
        from scipy.interpolate import RegularGridInterpolator
        init_shape = co_temp.shape
        
        
        x, y, z = (np.arange(init_shape[0])*final_shape[0]/(init_shape[0]-1),
                    np.arange(init_shape[1])*final_shape[1]/(init_shape[1]-1), 
                    np.arange(init_shape[2])*final_shape[2]/(init_shape[2]-1))
        
        intp = RegularGridInterpolator((x, y, z), co_temp, method=method)
        
        xu, yu, zu = np.meshgrid(np.arange(final_shape[0]),
                                np.arange(final_shape[1]),
                                np.arange(final_shape[2]))
        del co_temp
        coords = np.zeros(shape=(xu.shape[0]*xu.shape[1]*xu.shape[2], 3 ))
        coords[:,0]  = xu.ravel()
        coords[:,1]  = yu.ravel()
        coords[:,2]  = zu.ravel()
        co_temp_u = intp(coords)

        co_temp_u = co_temp_u.reshape(final_shape)
        co_temp_u = np.transpose(co_temp_u, axes=[1,0,2])

        return co_temp_u
