import os
import glob
import copy
from matplotlib.pyplot import axis
from pandas import to_timedelta
import h5py
import numpy as np
import dask.array as da
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from nbodykit import setup_logging
import logging
import logging.config
from nbodykit.lab import ArrayMesh
from nbodykit.lab import HDFCatalog
from nbodykit import CurrentMPIComm
from . import git_handler

class MockGalaxy():
    """A class for mock Line Intensit Map """
    def __init__(self, snap, axis=3, basepath=None, boxsize=None, brange =None, Rz= 0.02, 
                seed=None, halo_type='Subhalo', sim_type= 'TNG', compensated=True, map_path=None,
                mass_cut=11+np.log10(5), num_maps=1, sampling_rate =1, rsd= True, silent_mode=True, save_path=None):
        """
        Generate a mock Gaklaxy density from a hydro/DM-only simulation.
        The default arguments are for COMAP 2018(Chung+18, arxiv:1809.04550) 

        Parameterts:
        ---------------------------
            snap : Snapshot id, int
            axis : Either 1 or 2 or 3 indicating the line of sight axis, x or y or z
            basepath: The path to the directory of postprocessed data of the simulations
            Nmesh : (array_like, shape=(3,) ) number of grid mesh cells along each axis
            brange: (array_like, shape=(6,)) [xmin,xmax,ymin,ymax,zmin,zmax], subbox coordinates 
                    in cMpc/h of the simulation to make mock for
            boxsize : int or tuple of 3 ints
                    Simulation box Size in comoving  cMpc/h
            Rz : float, redshift accuracy, i.e. sigma_z/(1+z), of the galaxy survey.
            res_par : float, spatial resolution along line-of-sight in cMcp/h
            res_perp: float, spatial resolution in transverse direction in cMpc/h
            seed : Random seed to sample for redshift uncertainty of galaxies
            halo_type : Either 'Subhalo' or 'Group'                
            sim_type : str, simulation type either 'TNG' or 'MDPL2'
            compensated: If True, makes the correction in the fourier space of the field interpolated
                         by CIC.
            mass_cut : log of min subhalo mass in M_sol/h units, Default is ~ 11.7. If not None, only halos with 
                        log_10(M_halo) > mass_cut [M_sol/h] are considered

            rsd : whether to apply redshift space distortion to along line-of-sight
            are selected.

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
        self.halos = None
        self.halos_base = None
        self.groups = {}
        self.halo_id = []
        self.basepath = basepath
        if self.basepath is None:
            self.halo_file = None
        else :
            self.halo_file = os.path.join(basepath, 'groups_'+str(self.snap)+'/fof*')
        self.map_path = map_path
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
        self.num_maps = num_maps
        self.sampling_rate = sampling_rate
        self.rsd = rsd
        self.silent_mode =silent_mode
        if not self.silent_mode:
            setup_logging()
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
            # create logger
            self.logger = logging.getLogger('MockGal')
            self.logger.info('Starting')
        
        self.map = None
        self.perfect_map = None
        ## Suvey parameters
        self.Rz = Rz
        self.fine_res_par = 0.25
        self.fine_res_perp = 0.25

        # Set the grid numbers based on the spatial resolution of the map
        self.fine_Nmesh = [np.around(self.boxsize[0]/(self.fine_res_perp)), 
                           np.around(self.boxsize[1]/(self.fine_res_perp)),
                           np.around(self.boxsize[2]/(self.fine_res_par))]
        
        # Make maps :
        if self.map_path is not None:
            if not self.silent_mode:
                self.logger.info("Loading a precomputed galaxy map: ")
                self.logger.info(map_path)
            self.map_file = h5py.File(self.map_path,'r')
            self.map = self.map_file['map'][:]
            assert (self.map.shape[0]==self.fine_Nmesh[0])*(self.map.shape[0]==self.fine_Nmesh[0])*(self.map.shape[0]==self.fine_Nmesh[0])
            self.map = ArrayMesh(self.map, BoxSize=self.boxsize)
            self.galaxy_count = self.map_file['galaxy_count'][()]
            self.z = self.map_file['z'][()]
            self.res_par = self.get_res_par()
            self.res_perp = self.get_res_perp()
            
        else:
            # Laod halos on an Nbodykit catalog
            self._load_halos()
            #self.halos_base = copy.deepcopy(self.halos)
            self.galaxy_count = self.halos.csize
            self.res_par = self.get_res_par()
            self.res_perp = self.get_res_perp()
            # First get the perfect map (Rz=0) and then the mock map. Note: Only apply rsd for the firs time since 
            # it dispalces the halos the first time.
            self.map = self.get_galaxy_map(Rz=0, rsd=True)
            if save_path is not None:
                if not self.silent_mode:
                    self.logger.info('Saving the gal map on %s', save_path)
                self.save_map(save_path)

            """
            for i in range(self.num_maps):
                self.map = self.get_galaxy_map(Rz=0, rsd=True)
                if save_path is not None:
                    if self.sampling_rate != 1:
                        if i== 0:
                            save_path_old = save_path
                            save_path = save_path[:-5]+'_n0.hdf5'
                        else:
                            # Realod the halos, to avoid applying multiple 
                            # random smapling 
                            self._load_halos()
                            save_path = save_path_old[:-6]+str(i)+'.hdf5'
                    self.logger.info('Saving the gal map on %s', save_path)
                    self.save_map(save_path)
            """
            del self.halos

        if not self.silent_mode:
            self.logger.info('resolution : perp= %s, par= %s, cMcp/h, boxsize=%s', self.fine_res_perp, self.fine_res_par, self.boxsize)
    
    def get_res_par(self):
        """Calculate the spatial resolution along the lne-of-sight in units of comoving Mpc/h,
            It is the redshift uncertainty for the glaaxy survey.
            """
        delta_z = self.Rz*(1+self.z)
        delta_d = (const.c.to('km/s')*delta_z/cosmo.H(z=self.z)).value 
        # In units of cMpc/h :
        delta_d *= cosmo.h
        if not self.silent_mode:
            self.logger.info('sigma_z/(1+z) : %s', self.Rz)
            self.logger.info('delta_d : %s', delta_d)
        return delta_d
        
    def get_res_perp(self):
        """Calculate the spatial resolution in transverse direction to the line-of-sight in
        comoving Mpc/h. It is 0 for galaxy surveys
        """
        return 0

    def get_galaxy_map(self, Rz=None, mesh=True, rsd=True):
        """Calculate  the 3D galaxy overdensity map
           Returning the could-in-cell interpolated (CIC) galaxy overdensity : (n / <n>) - 1
           Be careful : We run this method for perfect_map first (i.e. mock=False) and then for the mock map (i.e. mock=True) in 
           the __init__()  method 
           
           Parameters:
           ------------------------
           mesh : bool. If True,  mesh the map in self.map as a pmesh object, otherwise, return
                the map as a numpy array without storing the result. 

           Returns : If mesh=False, An array of glaxay overdensity with shape=(Nmesh*Nmesh*Nmesh), Otherwise
                    does not return anything
           """
        if Rz is None:
            Rz = self.Rz
        if self.sampling_rate != 1:
            # sample the halos
            ind_rand = da.random.randint(0, self.halos.size, 
                size=int(self.sampling_rate*self.halos.size))
            selection = np.zeros(shape=(self.halos.csize,), dtype=bool)
            selection[ind_rand] = True
            self.halos['Selection'] = selection

        gal_density = self.do_CIC(Rz, rsd=rsd)
        to_delta = lambda x,y : y-1
        gal_density = gal_density.apply(to_delta, kind='index', mode='real')
        
        if mesh:
            return gal_density
        else:
            return gal_density.compute(Nmesh=self.fine_Nmesh)
    
    def do_CIC(self, Rz, rsd):
        """Similar to LyTomo_watershed.get_Density()
        Deposit the halos on a grid with cloud-in-cell method.
        """
        if self.halos['Coordinates'].size ==0:
            raise IOError('No halos on Rank '+str(self.rank))
        
        # Apply redshift space distortion
        # If you have calculated the perfect map, you had applied this before, so
        # do not apply it anymore
        if rsd:
            self._apply_rsd()

        # Apply an scatter due to redshift uncertainties        
        if Rz > 0 :
            self._apply_redshift_uncertainty(Rz)

        mesh = self.halos.to_mesh(Nmesh=self.fine_Nmesh, position='Coordinates',
                                  BoxSize=self.boxsize, compensated=self.compensated,
                                  selection='Selection')
        return mesh

    def _apply_rsd(self):
        """Apply Redshift space distortion, for sub/halos the velocities are in km/s 
        Note: in AREPO,  halo velocities do not have the extra sqrt(a) fcator
        """
        if not self.silent_mode:
            self.logger.info('applying rsd')
        self.halos['Coordinates'] = (self.halos['Coordinates']+
                                    (self.halos[self.halo_type+'Vel']*self.sight_vec*
                                    cosmo.h/cosmo.H(self.z).value))%self.boxsize
        if not self.silent_mode:
            self.logger.info('RSD applied')

    def _load_halos(self):
        """Load the subhalo catalogue using nbodykit
        The catalouge is not loaded on memory since it uses Dask.
        """
        if self.halos_base is not None:
            del self.halos
            self.halos = self.halos_base
        else:
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

            if self.sim_type=='TNG' or self.sim_type=='ASTRID':
                # for TNG convert ckpc/h tp cMpc/h
                self.halos['Coordinates'] = self.halos['SubhaloPos']/1e3

            if self.mass_cut is not None:
                if not self.silent_mode:
                    self.logger.info('Filtering halos for M_h > 10^ %s', self.mass_cut)
                ind = da.greater(self.halos['SubhaloMass'] , 10**(self.mass_cut-10))
                if not self.silent_mode:
                    self.logger.info('found the halos > mass_cut')
                self.halos = self.halos[ind]
                if not self.silent_mode:
                    self.logger.info('Halos are filtered')
            if self.brange is not None:
                ind = self.halos['Coordinates'] >= np.array([[ self.brange[0],self.brange[2], self.brange[4] ]])
                ind *= self.halos['Coordinates'] <= np.array([[ self.brange[1],self.brange[3], self.brange[5] ]])
                ind = da.prod(ind, axis=1, dtype=bool)
                self.halos = self.halos[ind]
    
                
    def save_map(self,save_path):
        """Save the galaxy map on an hd5 file for later usage"""
        import lila
        with h5py.File(save_path,'w') as fw:
            if not self.silent_mode:
                self.logger.info('Computing the mesh density')
            fw['map'] = self.map.compute(mode='real', Nmesh = self.fine_Nmesh)

            fw['galaxy_count'] = self.galaxy_count
            fw['z'] = self.z
            # Save the commit hash of the code used for this run
            # when reading this convert it to a list with :
            # `f['Git'].attrs['HEAD_HASH'].tolist()`
            fw.create_group('Git')
            head_hash = git_handler.get_head_hash(lila)
            fw['Git'].attrs["HEAD_HASH"] = np.void(head_hash)
