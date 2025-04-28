import numpy as np
from nbodykit.lab import cosmology

class Infer():
    """ 
    some helper routines for inference from a simulation.
    Used in notebooks/inference.ipynb
    """
    def __init__(self, st, iter_sampling=100_000, chains=6, linear=True, rsd=False, dm_file = None, kmin=None, kmax =None, k_non=None, lam=1):
        """"
        Paramters :
        linear : bool, if True use linear matter pwoer spectrum
        rsd : bool, if linear = True and rsd =True, account for the redshift space distrotion in the 
            linear matter power spectrum, it is the Kaiser and Finger of god's effects.
        
        """
        self.st = st
        self.iter_sampling = iter_sampling
        self.chains = chains
        self.linear = linear
        self. rsd = rsd
        self.dm_file = dm_file
        self.samples = None
        self.modes = None
        self.medians = None
        self.std_modes = None
        self.std_medians = None
        if kmin is None:
            self.kmin = st.kmin
        else:
            self.kmin = kmin
        if kmax is None:
            self.kmax = st.kmax
        else:
            self.kmax = kmax
        self.k_non = k_non
        self.lam = lam

    def get_power_from_map(self):
        from nbodykit.lab import ArrayMesh
        from nbodykit.lab import FFTPower
        import h5py
        with h5py.File(self.dm_file,'r') as f:
            dm = f['DM/dens'][:]
            mesh = ArrayMesh(dm, BoxSize=205)
        pow = FFTPower(mesh, mode='2d', los=[0,0,1], kmin=1/205, kmax=1, dk=0.03).run()[0]
        return pow

    def get_pm(self, k, z):
        """Linear Matter power spectrum
        z : redshift
        k: the k bins
        Return: the linear matter power P(k)"""
        if self.linear:
            cosmo_nbkit = cosmology.Planck15
            Plin = cosmology.LinearPower(cosmo=cosmo_nbkit, redshift=z, transfer='CLASS')
            if self.rsd :
                raise NotImplementedError('The Kaiser andFinger of god rsd effects are not '+
                                        +'implemented in the inference. To use the simple linear matter power, pass `rsd=False` ')
            
            return Plin(k)
        elif self.dm_file is None:
            pn = np.loadtxt('/run/media/mahdi/HD2/LIM/powerspectrum-0.2275.txt')
            pn = np.interp(k, pn[:,0], pn[:,1])
            return pn
        else:
            pk = self.get_power_from_map()
            pn = np.abs(np.interp(k, pk['k'][:,-1], pk['power'][:,-1]))
            return pn

    def fit_auto_lim(self, stan_model):
        """ For Auto LIM """
        data = {}
        ind = np.where( (self.st.lim_pk['k'][:] >= self.kmin) * (self.st.lim_pk['k'][:] <= self.kmax))
        data['N'] = self.st.lim_pk['k'][ind].shape[0]
        data['pm'] = self.get_pm(k=self.st.lim_pk['k'][ind], z=self.st.z)
        data["lim_pk"] = np.abs(self.st.lim_pk['power'][ind])
        data['sigma_lim_pk'] = np.abs(self.st.sigma_lim_pk[ind])
        # run HMC for 1000 iterations with 4 chains
        fit_auto = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)
        self.samples = np.empty(shape=(fit_auto.stan_variable('clustering').size, 2))
        self.samples[:,0] = fit_auto.stan_variable('clustering')
        self.samples[:,1] = fit_auto.stan_variable('pshot_lim')

        self.get_mode_std()
        self.clustering = self.medians[0]
        self.pshot_lim = self.medians[1]

        self.lim_pk_model = self.clustering**2 * data['pm'] + self.pshot_lim
    
    def fit_lim_gal(self, stan_model):
        ## For Galaxies :
        # put data in dict to pass to the CmdStanModel
        # the keys have to match the variable names in the Stan file data block
        data = {}
        ind = np.where( (self.st.lim_pk['k'][:] >= self.kmin) * (self.st.lim_pk['k'][:] <= self.kmax))[0]
        if self.k_non is not None:
            correction = self.st.gal_pk['k'][ind] - self.k_non
            indk0 = np.where(correction < 0)
            print(indk0)
            correction[indk0] = 0
            correction = correction**2
            print(correction)
        else:
            correction = 0
        sigma_lim_gal_pk = np.abs(self.st.sigma_lim_gal_pk[ind])*(1 + self.lam*correction)
        sigma_gal_pk = np.abs(self.st.sigma_gal_pk[ind])*(1 + self.lam*correction)


        data['N'] = self.st.lim_pk['k'][ind].shape[0]
        data['n3D'] = self.st.n3D
        data['pm'] = self.get_pm(k=self.st.lim_pk['k'][ind], z=self.st.z)
        data["lim_pk"] = np.abs(self.st.lim_pk['power'][ind])
        data['gal_pk'] = np.abs(self.st.gal_pk['power'][ind])
        data["lim_gal_pk"] = np.abs(self.st.lim_gal_pk['power'][ind])
        data['sigma_lim_pk'] = np.abs(self.st.sigma_lim_pk[ind])
        data['sigma_gal_pk'] = sigma_gal_pk
        data['sigma_lim_gal_pk'] = sigma_lim_gal_pk
        # run HMC for 1000 iterations with 4 chains
        fit_lim_gal = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)
        self.samples = np.empty(shape=(fit_lim_gal.stan_variable('clustering').size,5))
        self.samples[:,0] = fit_lim_gal.stan_variable('clustering')
        self.samples[:,1] = fit_lim_gal.stan_variable('pshot_lim')
        self.samples[:,2] = fit_lim_gal.stan_variable('bgal')
        self.samples[:,3] = fit_lim_gal.stan_variable('pshot_gal')
        self.samples[:,4] = fit_lim_gal.stan_variable('pshot_lim_gal')
        
        self.get_mode_std()
        self.clustering = self.medians[0]
        self.pshot_lim = self.medians[1]
        self.bgal = self.medians[2]
        self.pshot_gal = self.medians[3]
        self.pshot_lim_gal = self.medians[4]
        
        size = self.samples.shape[0]       
        
        lim_pk_model = fit_lim_gal.stan_variable('clustering').reshape(size, -1)**2 * data['pm'] + fit_lim_gal.stan_variable('pshot_lim').reshape(size, -1)
        self.lim_pk_bounds = np.quantile(lim_pk_model, q=[0.16,0.84], axis=0)
        del lim_pk_model
        self.lim_pk_model = self.clustering**2 * data['pm'] + self.pshot_lim

        gal_pk_model = fit_lim_gal.stan_variable('bgal').reshape(size, -1)**2 * data['pm'] + fit_lim_gal.stan_variable('pshot_gal').reshape(size, -1)
        self.gal_pk_bounds = np.quantile(gal_pk_model, q = [0.16,0.84], axis=0)
        del gal_pk_model
        self.gal_pk_model = self.bgal**2 * data['pm'] + self.pshot_gal

        lim_gal_pk_model = (fit_lim_gal.stan_variable('bgal').reshape(size, -1)*fit_lim_gal.stan_variable('clustering').reshape(size, -1))*data['pm'] + fit_lim_gal.stan_variable('pshot_lim_gal').reshape(size, -1)
        self.lim_gal_pk_bounds = np.quantile(lim_gal_pk_model, q=[0.16,0.84], axis=0)
        del lim_gal_pk_model
        self.lim_gal_pk_model = self.clustering * self.bgal * data['pm'] + self.pshot_lim_gal

    def fit_lim_lya(self, stan_model, pass_rk=False):
        ## For Lya :
        data = {}
        ind = np.where( (self.st.lim_pk['k'][:] >= self.kmin) * (self.st.lim_pk['k'][:] <= self.kmax))
        data['N'] = self.st.lim_pk['k'][ind].shape[0]
        data['pm'] = self.get_pm(k=self.st.lim_pk['k'][ind], z=self.st.z)
        data["lim_pk"] = np.abs(self.st.lim_pk['power'][ind])
        data['lya_pk'] = np.abs(self.st.lya_pk['power'][ind])
        if pass_rk:
            rk = np.abs(self.st.lim_lya_pk['power'][:]/ np.sqrt(self.st.lya_pk['power'][:]*
                                            self.st.lim_pk['power'][:]))
            data['rk'] = rk
        data["lim_lya_pk"] = np.abs(self.st.lim_lya_pk['power'][ind])
        data['sigma_lim_pk'] = np.abs(self.st.sigma_lim_pk[ind])
        data['sigma_lya_pk'] = np.abs(self.st.sigma_lya_pk[ind])
        data['sigma_lim_lya_pk'] = self.st.sigma_lim_lya_pk[ind]
 
        # run HMC for 1000 iterations with 4 chains        
        fit = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)
        self.samples = np.empty(shape=(fit.stan_variable('clustering').size,3))
        self.samples[:,0] = fit.stan_variable('clustering')
        self.samples[:,1] = fit.stan_variable('pshot_lim')
        self.samples[:,2] = fit.stan_variable('blya')
        
        self.get_mode_std()
        self.clustering = self.medians[0]
        self.pshot_lim = self.medians[1]
        self.blya = self.medians[2]

        size = self.samples.shape[0]       
        lim_pk_model = fit.stan_variable('clustering').reshape(size,-1)**2 * data['pm'] + fit.stan_variable('pshot_lim').reshape(size, -1)
        self.lim_pk_bounds = np.quantile(lim_pk_model, q=[0.16,0.84], axis=0)
        del lim_pk_model
        self.lim_pk_model = self.clustering**2 * data['pm'] + self.pshot_lim

        lya_pk_model = fit.stan_variable('blya').reshape(size, -1)**2 * data['pm']
        self.lya_pk_bounds = np.quantile(lya_pk_model, q=[0.16,0.84], axis=0)
        del lya_pk_model
        self.lya_pk_model = self.blya**2 * data['pm']

        lim_lya_pk_model = fit.stan_variable('clustering').reshape(size,-1)*fit.stan_variable('blya').reshape(size, -1)* data['pm']
        self.lim_lya_pk_bounds = np.quantile(lim_lya_pk_model, q=[0.16,0.84], axis=0)
        del lim_lya_pk_model
        self.lim_lya_pk_model = self.clustering*self.blya*data['pm']

    def fit_co_cross_gal(self, stan_model):
        ## For Galaxies :
        # put data in dict to pass to the CmdStanModel
        # the keys have to match the variable names in the Stan file data block
        data = {}
        ind = np.where( (self.st.lim_pk['k'][:] >= self.kmin) * (self.st.lim_pk['k'][:] <= self.kmax))
        if self.k_non is not None:
            correction = self.st.gal_pk['k'][ind] - self.k_non
            indk0 = np.where(correction < 0)
            print(indk0)
            correction[indk0] = 0
            correction = correction**2
            print(correction)
        else:
            correction = 0
        sigma_lim_gal_pk = np.abs(self.st.sigma_lim_gal_pk[ind])*(1 + self.lam*correction)
        sigma_gal_pk = np.abs(self.st.sigma_gal_pk[ind])*(1 + self.lam*correction)


        data['N'] = self.st.lim_pk['k'][ind].shape[0]
        data['pm'] = self.get_pm(k=self.st.lim_pk['k'][ind], z=self.st.z)
        data['gal_pk'] = np.abs(self.st.gal_pk['power'][ind])
        data["lim_gal_pk"] = np.abs(self.st.lim_gal_pk['power'][ind])
        data['sigma_gal_pk'] = sigma_gal_pk
        data['sigma_lim_gal_pk'] = sigma_lim_gal_pk
        # run HMC for 1000 iterations with 4 chains
        fit_lim_gal = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)
        self.samples = np.empty(shape=(fit_lim_gal.stan_variable('clustering').size,4))
        self.samples[:,0] = fit_lim_gal.stan_variable('clustering')
        self.samples[:,1] = fit_lim_gal.stan_variable('bgal')
        self.samples[:,2] = fit_lim_gal.stan_variable('pshot_gal')
        self.samples[:,3] = fit_lim_gal.stan_variable('pshot_lim_gal')
        
        self.get_mode_std()
        self.clustering = self.medians[0]
        self.bgal = self.medians[1]
        self.pshot_gal = self.medians[2]
        self.pshot_lim_gal = self.medians[3]
        
        self.gal_pk_model = self.bgal**2 * data['pm'] + self.pshot_gal
        self.lim_gal_pk_model = self.bgal*self.clustering * data['pm'] + self.pshot_lim_gal

    def fit_co_cross_lya(self, stan_model):
        ## For Lya :
        data = {}
        ind = np.where( (self.st.lim_pk['k'][:] >= self.kmin) * (self.st.lim_pk['k'][:] <= self.kmax))
        data['N'] = self.st.lim_pk['k'][ind].shape[0]
        data['pm'] = self.get_pm(k=self.st.lim_pk['k'][ind], z=self.st.z)
        data['lya_pk'] = np.abs(self.st.lya_pk['power'][ind])
        data["lim_lya_pk"] = np.abs(self.st.lim_lya_pk['power'][ind])
        data['sigma_lya_pk'] = np.abs(self.st.sigma_lya_pk[ind])
        data['sigma_lim_lya_pk'] = self.st.sigma_lim_lya_pk[ind]
 
        # run HMC for 1000 iterations with 4 chains        
        fit = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)
        self.samples = np.empty(shape=(fit.stan_variable('clustering').size,3))
        self.samples[:,0] = fit.stan_variable('clustering')
        self.samples[:,2] = fit.stan_variable('blya')
        
        self.get_mode_std()
        self.clustering = self.medians[0]
        self.blya = self.medians[1]
        
        self.lya_pk_model = self.blya**2 * data['pm']
        self.lim_lya_pk_model = self.clustering*self.blya * data['pm']

    def fit_co_coXlya(self, stan_model, pass_rk=False):
        ## For Lya :
        data = {}
        ind = np.where( (self.st.lim_pk['k'][:] >= self.kmin) * (self.st.lim_pk['k'][:] <= self.kmax))
        data['N'] = self.st.lim_pk['k'][ind].shape[0]
        data['pm'] = self.get_pm(k=self.st.lim_pk['k'][ind], z=self.st.z)
        data["lim_pk"] = np.abs(self.st.lim_pk['power'][ind])
        if pass_rk:
            rk = np.abs(self.st.lim_lya_pk['power'][:]/ np.sqrt(self.st.lya_pk['power'][:]*
                                            self.st.lim_pk['power'][:]))
            data['rk'] = rk
        data["lim_lya_pk"] = np.abs(self.st.lim_lya_pk['power'][ind])
        data['sigma_lim_pk'] = np.abs(self.st.sigma_lim_pk[ind])
        data['sigma_lim_lya_pk'] = self.st.sigma_lim_lya_pk[ind]
 
        # run HMC for 1000 iterations with 4 chains        
        fit = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)
        self.samples = np.empty(shape=(fit.stan_variable('clustering').size,3))
        self.samples[:,0] = fit.stan_variable('clustering')
        self.samples[:,1] = fit.stan_variable('pshot_lim')
        self.samples[:,2] = fit.stan_variable('blya')
        
        self.get_mode_std()
        self.clustering = self.medians[0]
        self.pshot_lim = self.medians[1]
        self.blya = self.medians[2]

        size = self.samples.shape[0]       
        lim_pk_model = fit.stan_variable('clustering').reshape(size,-1)**2 * data['pm'] + fit.stan_variable('pshot_lim').reshape(size, -1)
        self.lim_pk_bounds = np.quantile(lim_pk_model, q=[0.16,0.84], axis=0)
        del lim_pk_model
        self.lim_pk_model = self.clustering**2 * data['pm'] + self.pshot_lim

        lya_pk_model = fit.stan_variable('blya').reshape(size, -1)**2 * data['pm']
        self.lya_pk_bounds = np.quantile(lya_pk_model, q=[0.16,0.84], axis=0)
        del lya_pk_model
        self.lya_pk_model = self.blya**2 * data['pm']

        lim_lya_pk_model = fit.stan_variable('clustering').reshape(size,-1)*fit.stan_variable('blya').reshape(size, -1)* data['pm']
        self.lim_lya_pk_bounds = np.quantile(lim_lya_pk_model, q=[0.16,0.84], axis=0)
        del lim_lya_pk_model
        self.lim_lya_pk_model = self.clustering*self.blya*data['pm']


    def fit_co_gal_fixed_pshot(self, stan_model, pshot_gal, pshot_lim_gal):
        ## For Galaxies :
        # put data in dict to pass to the CmdStanModel
        # the keys have to match the variable names in the Stan file data block
        data = {}
        ind = np.where( (self.st.lim_pk['k'][:] >= self.kmin) * (self.st.lim_pk['k'][:] <= self.kmax))
        if self.k_non is not None:
            correction = self.st.gal_pk['k'][ind] - self.k_non
            indk0 = np.where(correction < 0)
            print(indk0)
            correction[indk0] = 0
            correction = correction**2
            print(correction)
        else:
            correction = 0
        sigma_lim_gal_pk = np.abs(self.st.sigma_lim_gal_pk[ind])*(1 + self.lam*correction)
        sigma_gal_pk = np.abs(self.st.sigma_gal_pk[ind])*(1 + self.lam*correction)


        data['N'] = self.st.lim_pk['k'][ind].shape[0]
        data['pshot_gal'] = pshot_gal
        data['pshot_lim_gal'] = pshot_lim_gal
        data['n3D'] = self.st.n3D
        data['pm'] = self.get_pm(k=self.st.lim_pk['k'][ind], z=self.st.z)
        data["lim_pk"] = np.abs(self.st.lim_pk['power'][ind])
        data['gal_pk'] = np.abs(self.st.gal_pk['power'][ind])
        data["lim_gal_pk"] = np.abs(self.st.lim_gal_pk['power'][ind])
        data['sigma_lim_pk'] = np.abs(self.st.sigma_lim_pk[ind])
        data['sigma_gal_pk'] = sigma_gal_pk
        data['sigma_lim_gal_pk'] = sigma_lim_gal_pk
        # run HMC for 1000 iterations with 4 chains
        fit_lim_gal = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)
        self.samples = np.empty(shape=(fit_lim_gal.stan_variable('clustering').size,5))
        self.samples[:,0] = fit_lim_gal.stan_variable('clustering')
        self.samples[:,1] = fit_lim_gal.stan_variable('pshot_lim')
        self.samples[:,2] = fit_lim_gal.stan_variable('bgal')

        
        self.get_mode_std()
        self.clustering = self.medians[0]
        self.pshot_lim = self.medians[1]
        self.bgal = self.medians[2]
        self.pshot_gal = pshot_gal
        self.pshot_lim_gal = pshot_lim_gal

        
        size = self.samples.shape[0]       
        
        lim_pk_model = fit_lim_gal.stan_variable('clustering').reshape(size, -1)**2 * data['pm'] + fit_lim_gal.stan_variable('pshot_lim').reshape(size, -1)
        self.lim_pk_bounds = np.quantile(lim_pk_model, q=[0.16,0.84], axis=0)
        del lim_pk_model
        self.lim_pk_model = self.clustering**2 * data['pm'] + self.pshot_lim

        gal_pk_model = fit_lim_gal.stan_variable('bgal').reshape(size, -1)**2 * data['pm'] + pshot_gal
        self.gal_pk_bounds = np.quantile(gal_pk_model, q = [0.16,0.84], axis=0)
        del gal_pk_model
        self.gal_pk_model = self.bgal**2 * data['pm'] + self.pshot_gal

        lim_gal_pk_model = (fit_lim_gal.stan_variable('bgal').reshape(size, -1)*fit_lim_gal.stan_variable('clustering').reshape(size, -1))*data['pm'] + pshot_lim_gal
        self.lim_gal_pk_bounds = np.quantile(lim_gal_pk_model, q=[0.16,0.84], axis=0)
        del lim_gal_pk_model
        self.lim_gal_pk_model = self.clustering * self.bgal * data['pm'] + self.pshot_lim_gal
    
    def point_estimate_co_gal(self, stan_model):
        ## For Galaxies :
        # put data in dict to pass to the CmdStanModel
        # the keys have to match the variable names in the Stan file data block
        data = {}
        ind = np.where( (self.st.lim_pk['k'][:] >= self.kmin) * (self.st.lim_pk['k'][:] <= self.kmax))
        if self.k_non is not None:
            correction = self.st.gal_pk['k'][ind] - self.k_non
            indk0 = np.where(correction < 0)
            print(indk0)
            correction[indk0] = 0
            correction = correction**2
            print(correction)
        else:
            correction = 0
        sigma_lim_gal_pk = np.abs(self.st.sigma_lim_gal_pk[ind])*(1 + self.lam*correction)
        sigma_gal_pk = np.abs(self.st.sigma_gal_pk[ind])*(1 + self.lam*correction)


        data['N'] = self.st.lim_pk['k'][ind].shape[0]
        data['n3D'] = self.st.n3D
        data['pm'] = self.get_pm(k=self.st.lim_pk['k'][ind], z=self.st.z)
        data["lim_pk"] = np.abs(self.st.lim_pk['power'][ind])
        data['gal_pk'] = np.abs(self.st.gal_pk['power'][ind])
        data["lim_gal_pk"] = np.abs(self.st.lim_gal_pk['power'][ind])
        data['sigma_lim_pk'] = np.abs(self.st.sigma_lim_pk[ind])
        data['sigma_gal_pk'] = sigma_gal_pk
        data['sigma_lim_gal_pk'] = sigma_lim_gal_pk
        # run HMC for 1000 iterations with 4 chains
        point_estimate = stan_model.optimize(data=data)
        return point_estimate

    def fit_co_no_gal(self, stan_model):
        ## For Galaxies :
        # put data in dict to pass to the CmdStanModel
        # the keys have to match the variable names in the Stan file data block
        data = {}
        ind = np.where( (self.st.lim_pk['k'][:] >= self.kmin) * (self.st.lim_pk['k'][:] <= self.kmax))
        if self.k_non is not None:
            correction = self.st.gal_pk['k'][ind] - self.k_non
            indk0 = np.where(correction < 0)
            print(indk0)
            correction[indk0] = 0
            correction = correction**2
            print(correction)
        else:
            correction = 0
        sigma_lim_gal_pk = np.abs(self.st.sigma_lim_gal_pk[ind])*(1 + self.lam*correction)
        sigma_gal_pk = np.abs(self.st.sigma_gal_pk[ind])*(1 + self.lam*correction)


        data['N'] = self.st.lim_pk['k'][ind].shape[0]
        data['pm'] = self.get_pm(k=self.st.lim_pk['k'][ind], z=self.st.z)
        data["lim_pk"] = np.abs(self.st.lim_pk['power'][ind])
        data["lim_gal_pk"] = np.abs(self.st.lim_gal_pk['power'][ind])
        data['sigma_lim_pk'] = np.abs(self.st.sigma_lim_pk[ind])
        data['sigma_lim_gal_pk'] = sigma_lim_gal_pk
        # run HMC for 1000 iterations with 4 chains
        fit_lim_gal = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)
        self.samples = np.empty(shape=(fit_lim_gal.stan_variable('clustering').size,4))
        self.samples[:,0] = fit_lim_gal.stan_variable('clustering')
        self.samples[:,1] = fit_lim_gal.stan_variable('pshot_lim')
        self.samples[:,2] = fit_lim_gal.stan_variable('bgal')
        self.samples[:,3] = fit_lim_gal.stan_variable('pshot_lim_gal')
        
        self.get_mode_std()
        self.clustering = self.medians[0]
        self.pshot_lim = self.medians[1]
        self.bgal = self.medians[2]
        self.pshot_lim_gal = self.medians[3]
        
        self.lim_pk_model = self.clustering**2 * data['pm'] + self.pshot_lim
        self.lim_gal_pk_model = self.bgal*self.clustering * data['pm'] + self.pshot_lim_gal

    def fit_co_no_gal_bgal_fixed(self, stan_model, bgal):
        ## For Galaxies :
        # put data in dict to pass to the CmdStanModel
        # the keys have to match the variable names in the Stan file data block
        data = {}
        ind = np.where( (self.st.lim_pk['k'][:] >= self.kmin) * (self.st.lim_pk['k'][:] <= self.kmax))
        if self.k_non is not None:
            correction = self.st.gal_pk['k'][ind] - self.k_non
            indk0 = np.where(correction < 0)
            print(indk0)
            correction[indk0] = 0
            correction = correction**2
            print(correction)
        else:
            correction = 0
        sigma_lim_gal_pk = np.abs(self.st.sigma_lim_gal_pk[ind])*(1 + self.lam*correction)
        sigma_gal_pk = np.abs(self.st.sigma_gal_pk[ind])*(1 + self.lam*correction)


        data['N'] = self.st.lim_pk['k'][ind].shape[0]
        data['pm'] = self.get_pm(k=self.st.lim_pk['k'][ind], z=self.st.z)
        data["lim_pk"] = np.abs(self.st.lim_pk['power'][ind])
        data["lim_gal_pk"] = np.abs(self.st.lim_gal_pk['power'][ind])
        data['sigma_lim_pk'] = np.abs(self.st.sigma_lim_pk[ind])
        data['sigma_lim_gal_pk'] = sigma_lim_gal_pk
        # run HMC for 1000 iterations with 4 chains
        fit_lim_gal = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)
        self.samples = np.empty(shape=(fit_lim_gal.stan_variable('clustering').size,3))
        self.samples[:,0] = fit_lim_gal.stan_variable('clustering')
        self.samples[:,1] = fit_lim_gal.stan_variable('pshot_lim')
        self.samples[:,2] = fit_lim_gal.stan_variable('pshot_lim_gal')
        
        self.get_mode_std()
        self.clustering = self.medians[0]
        self.pshot_lim = self.medians[1]
        self.bgal = bgal
        self.pshot_lim_gal = self.medians[2]
        
        self.lim_pk_model = self.clustering**2 * data['pm'] + self.pshot_lim
        self.lim_gal_pk_model = self.bgal*self.clustering * data['pm'] + self.pshot_lim_gal



    def fit_auto_gal(self, stan_model):
        data = {}
        ind = np.where( (self.st.gal_pk['k'][:] >= self.kmin) * (self.st.gal_pk['k'][:] <= self.kmax))
        if self.k_non is not None:
            correction = self.st.gal_pk['k'][ind] - self.k_non
            indk0 = np.where(correction < 0)
            print(indk0)
            correction[indk0] = 0
            correction = correction**2
            print(correction)
        else:
            correction = 0
        sigma_lim_gal_pk = np.abs(self.st.sigma_lim_gal_pk[ind])*(1 + self.lam*correction)
        sigma_gal_pk = np.abs(self.st.sigma_gal_pk[ind])*(1 + self.lam*correction)


        data['N'] = self.st.gal_pk['k'][ind].shape[0]
        data['pm'] = self.get_pm(k=self.st.gal_pk['k'][ind], z=self.st.z)
        data['gal_pk'] = np.abs(self.st.gal_pk['power'][ind])
        data['sigma_gal_pk'] = sigma_gal_pk
        # run HMC for 1000 iterations with 4 chains
        fit_lim_gal = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)
        self.samples = np.empty(shape=(fit_lim_gal.stan_variable('bgal').size,2))
        self.samples[:,0] = fit_lim_gal.stan_variable('bgal')
        self.samples[:,1] = fit_lim_gal.stan_variable('pshot_gal')
        
        self.get_mode_std()
        self.bgal = self.medians[0]
        self.pshot_gal = self.medians[1]
        
        self.gal_pk_model = self.bgal**2 * data['pm'] + self.pshot_gal

    def fit_auto_lya(self, stan_model):
        ## For Lya :
        data = {}
        ind = np.where( (self.st.lim_pk['k'][:] >= self.kmin) * (self.st.lim_pk['k'][:] <= self.kmax))
        data['N'] = self.st.lim_pk['k'][ind].shape[0]
        data['pm'] = self.get_pm(k=self.st.lim_pk['k'][ind], z=self.st.z)
        data['lya_pk'] = np.abs(self.st.lya_pk['power'][ind])
        data['sigma_lya_pk'] = np.abs(self.st.sigma_lya_pk[ind])
 
        # run HMC for 1000 iterations with 4 chains        
        fit = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)
        self.samples = np.empty(shape=(fit.stan_variable('blya').size,1))
        self.samples[:,0] = fit.stan_variable('blya')
        
        self.get_mode_std()
        self.blya = self.medians[0]
        
        self.lya_pk_model = self.blya**2 * data['pm']

    def fit_lya_gal(self, st, stan_model):
        ## For All :
        data = {}
        data['N'] = self.st.lim_pk['k'][:].shape[0]
        data['n3D'] = self.st.n3D
        data['pm'] = self.get_pm(self.st.lim_pk['k'][:], self.st.z)
        data["lim_pk"] = np.abs(self.st.lim_pk['power'][:])
        data["lya_pk"] = np.abs(self.st.lya_pk['power'][:])
        data["gal_pk"] = np.abs(self.st.gal_pk['power'][:])
        data["lim_lya_pk"] = np.abs(self.st.lim_lya_pk['power'][:])
        data["lim_gal_pk"] = np.abs(self.st.lim_gal_pk['power'][:])
        data['sigma_lya_pk'] = np.abs(self.st.sigma_lya_pk[:])
        data['sigma_gal_pk'] = np.abs(self.st.sigma_gal_pk[:])
        data['sigma_lim_pk'] = np.abs(self.st.sigma_lim_pk[:])
        data['sigma_lim_lya_pk'] = np.abs(self.st.sigma_lim_lya_pk[:])
        data['sigma_lim_gal_pk'] = np.abs(self.st.sigma_lim_gal_pk[:])

        # run HMC for 1000 iterations with 4 chains
        fit = stan_model.sample(data=data, iter_sampling=self.iter_sampling, chains=self.chains)

        self.samples = np.empty(shape=(fit.stan_variable('clustering').size,5))
        self.samples[:,0] = fit.stan_variable('clustering')
        self.samples[:,1] = fit.stan_variable('pshot_lim')
        self.samples[:,2] = fit.stan_variable('blya')
        self.samples[:,3] = fit.stan_variable('bgal')
        self.samples[:,4] = fit.stan_variable('pshot_lim_gal')

    def get_mode_std(self):
        """get mode and std of each parameter in self.samples
        Paramters:
            self.samples: an (m,n) dimensional array. m paramters and n
            smaples.
        Returns :
            modees, std around modes, medians and std around medians
        """
        if self.modes is None:
            modes, std_modes = [], []
            medians, std_medians= [], []
            for i in range(self.samples.shape[1]):
                v, c = np.unique(self.samples[:,i], return_counts=True)
                ind = np.argmax(c)
                mod = v[ind]
                med = np.median(self.samples[:,i])
                modes.append(mod)
                medians.append(med)
                std_medians.append(np.sqrt(np.mean((self.samples[:,i]-med)**2)))
                std_modes.append(np.sqrt(np.mean((self.samples[:,i]-mod)**2)))
            
            self.modes = np.around(modes, 10)
            self.std_modes = np.around(std_modes, 10)
            self.medians = np.around(medians,10)
            self.std_medians =  np.around(std_medians, 10)
    
    def print_summary_stats(self, param_labels, survey_label):
        """Print the mode+- std for all paramters and all self.samples"""
        from IPython.display import display, Math
    
        
        print(survey_labels[n])
        self.get_mode_std()
        for i in range(len(param_labels)):
            frac = np.around(self.std_modes[i]/self.modes[i],5)
            frac_r = np.around(self.modes[i]/self.std_modes[i], 5)
            display(Math(param_labels[i]+' = '+str(self.modes[i])+' \pm '+str(self.std_modes[i])+', \ frac ='+str(frac)+',\ 1/frac = '+str(frac_r)))

    def get_true_paramters(self, lim_mock):
        """Using the generative CO mode, calculate the true parameters
        num: The index to the paramter set in the MCMC collection of the CO model 
        Returns: (Tb, P_shot)
            Tb: <T_CO>*b_CO
            P_shot: CO auto shot noise power
        """
        co_temp = lim_mock.co_temp.compute()
        mean_Tco = np.mean(co_temp)
        

    def get_b_lya(self,z):
        """Get  b_Lya from the Lya maps
        """
        b_lya = - 0.20 # I should be able to use our mock map instead ?
        return b_lya



    def _get_bco(self):
        """Calculate the CO bias for a catalog from the CO model :
        b_CO = \int L_{CO}(M) b(M) dn/dM dM / \int L_{CO}(M) dn/dM dM
        Wehre b(M) is the mass dependant halo bias and dn/dM the halo 
        mass function.
        Returns: b_CO
        """
        pass

    def _get_Pshot_co(self, lim_mock):
        """Get the CO shot noise power :
        (e.g. eq 2 from Keenan+21 arxive:2110.02239)
        P_{shot, CO} = C  \int L^{2}_{CO} dn/dL dL
        Where the C is the conversion factor from CO luminocity to 
        brightness temperature, same as in mock_lim.py
        --------------------------------
        Paramters : 
        lim_mock : an instance of lim_lytomo.mock_lim.Mock_Lim 
        Returns: p_{shot, co} in appropriate units
        """
        lco = lim_mock.get_co_lum_halos()
        assert len(lim_mock.boxsize )==3
        vt = np.prod(lim_mock.boxsize)
    
        return np.sum(lim_mock.co_lum_to_temp(lco=lco, vol=vt)**2)*vt
    
    def rsd_term(self):
        """An analytic term in the power spectum formalism to take the 
        redshift space distrotion(RSD) into account"""

        return 0