from regpy.util import classlogger

from nbodykit.lab import *
from nbodykit import style, setup_logging

from reglyman.density import LinearPower

from nbodykit.mockmaker import lognormal_transform, gaussian_real_fields
from pmesh.pm import ParticleMesh
from nbodykit.mpirng import MPIRandomState

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import mpsort

import logging

setup_logging()

'''
Computes the creation of synthetic overdensities
'''

class Data_Generation():
    log=classlogger
    
    def __init__(self, parameters_box, parameters_igm, use_baryonic=False, transfer="EisensteinHu", path_transfer=None, column=None, cosmo=None):
        self.redshift= parameters_box['redshift']
        self.width=parameters_box['width']
        chosen_cosmo=parameters_box['cosmology']
        
        #Use the baryonic power spectrum instead
        #Then the resulting overdensity field does not correspond to cdm-density, but to baryonic density
        #Jeans length needed [h**-1 Mpc]
        #Note that Jeans length scales with redshift
        #For proper values look: Choudhury, Sraianda ... 2001
        self.use_baryonic=use_baryonic
        self.Jeans_length=parameters_box['jeans_length']

        #specifies specific cosmological model
        if chosen_cosmo=='Planck15':
            self.cosmo = cosmology.Planck15
        elif chosen_cosmo=='UserDefined':
            self.cosmo = cosmo
        else: 
            print(chosen_cosmo, 'not implemented right now')
            print('Continue with Planck15 cosmology instead')
            self.cosmo = cosmology.Planck15     

        self.comoving_distance=self.cosmo.comoving_distance(self.redshift)

        #linear power spectrum
        Plin = LinearPower(self.cosmo, self.redshift, transfer, path_transfer, column)
        if self.use_baryonic:
            self.Plin=lambda k: 1/(1+self.Jeans_length**2*k**2)**2*Plin(k)
            self.Plin.redshift=self.redshift
            self.Plin.sigma8=self.cosmo.sigma8
            self.Plin.cosmo=self.cosmo
            if self.log.isEnabledFor(logging.INFO):
                self.log.info('Start computation of baryonic density field') 
        else:
            self.Plin=Plin
            if self.log.isEnabledFor(logging.INFO):
                self.log.info('Start computation of cdm density field')            
        
        #Parameters describing the box
        self.background_field_x=parameters_box['background_box'][0]
        self.background_field_y=parameters_box['background_box'][1]
        
        self.background_sampling_x=parameters_box['background_sampling'][0]
        self.background_sampling_y=parameters_box['background_sampling'][1]
        
        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Background has been specified:')
            self.log.info('-->Comoving length: {}'.format([self.background_field_x, self.background_field_y]))
            self.log.info('-->sampled:{}'.format([self.background_sampling_x, self.background_sampling_y]))

        #initializes pseudo-gaussian random generator
        self.seed=parameters_box['seed']

        #if True, the seed gaussian has unitary_amplitude
        self.unitary_amplitude=True
        self.inverted_phase=False
        #Also compute displacement
        self.compute_displacement=True
                
        self.BoxSize=[self.background_field_x, self.background_field_y, self.width]
        self.N_pix=parameters_box['nr_pix']

        self.bias=2

	#The following values describe the state of the IGM and  have to match whatever specified in the inversion        
        self.beta=parameters_igm['beta']
        self.alpha=2-1.4*self.beta
        self.tildeC=parameters_igm['tilde_c']
        
        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Generator initialized with parameters')
            self.log.info('-->Cosmology: {}'.format(chosen_cosmo))
            self.log.info('-->Box size: {}'.format(self.BoxSize))
            self.log.info('-->Number pixels: {}'.format(self.N_pix))
            self.log.info('-->Redshift: {}'.format(self.redshift))
            self.log.info('-->seed: {}'.format(self.seed)) 
            self.log.info('-->Jeans: {}'.format(self.Jeans_length))

        # the particle mesh for gridding purposes
        _Nmesh = np.empty(3, dtype='i8')
        _Nmesh[0] = self.background_sampling_x
        _Nmesh[1] = self.background_sampling_y
        _Nmesh[2] = self.N_pix
        self.pm = ParticleMesh(BoxSize=self.BoxSize, Nmesh=_Nmesh, dtype='f4')
        return

	#Compute Density Field. The function strongly matches the implementation in nbodykit.
	#Return:
        #delta: Density Perturbation (rho/rho_med)
        #comoving: Comoving distance to each point at line of sight [h^-1 Mpc]
        #vel: Peculiar velocity field [km/s]
    def Compute_Density_Field(self):
        # growth rate to do RSD in the Zel'dovich approximation
        f = self.cosmo.scale_independent_growth_rate(self.redshift)

        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Growth rate is {}'.format(f))

	#Lineat density and displacement field
        delta, disp = gaussian_real_fields(self.pm, self.Plin, self.seed,
            unitary_amplitude=self.unitary_amplitude,
            inverted_phase=self.inverted_phase,
            compute_displacement=self.compute_displacement)

        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Gaussian field generated')

        # apply the lognormal transformation to the initial conditions density
        # this creates a positive-definite delta (necessary for Poisson sampling)
        lagrangian_bias = self.bias - 1
        delta = lognormal_transform(delta, bias=lagrangian_bias)
        
        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Density field projected to lognormal')
	
	#Compute peculiar velocity field from linear displacement field
        if self.compute_displacement:
            self.displacement=disp
            velocity_norm = f * 100 * self.cosmo.efunc(self.redshift) / (1+self.redshift)
            vel_x  = velocity_norm * disp[0]
            vel_y = velocity_norm * disp[1]
            vel_z = velocity_norm * disp[2]
            vel=[vel_x.value, vel_y.value, vel_z.value]
            
        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Velocity field computed')        
        
	#Computes comoving distances along line of sight
        comoving=self.cosmo.comoving_distance(self.redshift)-self.width+self.width/self.N_pix*np.linspace(1, self.N_pix, self.N_pix)

        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Density field succesfully computed')
        
        return [delta, vel, comoving]

#Computes the Covariance matrix, depending on the linear autocorrelation
#Correlation may look different in nonlinear and mildly nonlinear regime (due to lognormal transform)
#In case of non biased gaussian fields (expectation value equal to zero) the output matches the covariance matrix
#diff: The difference value between two points in real space
#Autocorrelation is given by the inverse Fourier transform of power spectrum (Wiener-Khinchin-Theorem)
#comoving: vector of comoving coordinates on which the density field is evaluated
#WARNING: Due to rebinning the comoving field could have changed and is not equal binned anymore  
#As Covariance matrix is symmetric, only the upper half is computed, then both merged together by transposition
    def Compute_Linear_Covariance(self, comoving):
        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Computation of prior covariance started')
            
        Covariance=np.zeros((np.size(comoving), np.size(comoving)))

        k = np.linspace(10**(-5), 10, 10**6)
        size = len(k)//2
        Pk = self.Plin(k)
        fourier_coeff = np.abs(np.fft.fftn(Pk)[0:size+1])
        frqs = np.linspace(0, 0.1*size, size+1)
        cf_lin = Spline(frqs, fourier_coeff)
        
        diff=np.zeros((np.size(comoving), np.size(comoving)))
        for i in range(np.size(comoving)):
            for j in range(np.size(comoving)):
                diff[i, j]=np.abs(comoving[i]-comoving[j])
                
        Covariance=cf_lin(diff)
        
        Covariance /= Covariance[0, 0]

        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Prior Covariance computed')  

        return Covariance

#This Function computes the selection of a single line of sight from the delta sample
#i, j: indices of background sources at maximal redshift
#Return: Density-field along one line-of-sight, evaluated at the pixels of comoving
    def Select_LOS(self, delta, vel, index):
        i=index[0].astype(int)
        j=index[1].astype(int)
        if self.log.isEnabledFor(logging.INFO):                
            self.log.info('Line of sight selected: {}'.format(index))
        return [delta[i, j, :], vel[i, j, :]]

        
#Poisson sample to overdensity field,
#matches the nbodykit routine 
    def PoissonSample(self, delta, parameters_sampling):
        nbar=parameters_sampling['nbar']
        seed1=parameters_sampling['seed1']
        seed2=parameters_sampling['seed2']

        comm = self.pm.comm
        # mean number of objects per cell
        H = self.BoxSize / self.pm.Nmesh
        overallmean = H.prod() * nbar

        # number of objects in each cell (per rank, as a RealField)
        cellmean = delta * overallmean

        # create a random state with the input seed
        rng = MPIRandomState(seed=seed1, comm=comm, size=delta.size)

        # generate poissons. Note that we use ravel/unravel to
        # maintain MPI invariane.
        Nravel = rng.poisson(lam=cellmean.ravel())
        N = self.pm.create(type='real')
        N.unravel(Nravel)

        Ntot = N.csum()
        if self.log.isEnabledFor(logging.INFO):
            self.log.info('Poisson sampling done, total number of objects is {}'.format(Ntot))

        pos_mesh = self.pm.generate_uniform_particle_grid(shift=0.0)
        disp_mesh = np.empty_like(pos_mesh)

        # no need to do decompose because pos_mesh is strictly within the
        # local volume of the RealField.
        N_per_cell = N.readout(pos_mesh, resampler='nnb')
        for i in range(N.ndim):
            disp_mesh[:, i] = self.displacement[i].readout(pos_mesh, resampler='nnb')

        # fight round off errors, if any
        N_per_cell = np.int64(N_per_cell + 0.5)

        pos = pos_mesh.repeat(N_per_cell, axis=0)
        disp = disp_mesh.repeat(N_per_cell, axis=0)

        del pos_mesh
        del disp_mesh

        if self.log.isEnabledFor(logging.INFO):
            self.log.info("Catalog produced. Assigning in cell shift.")

        # FIXME: after pmesh update, remove this
        orderby = np.int64(pos[:, 0] / H[0] + 0.5)
        for i in range(1, delta.ndim):
            orderby[...] *= self.pm.Nmesh[i]
            orderby[...] += np.int64(pos[:, i] / H[i] + 0.5)

        # sort by ID to maintain MPI invariance.
        pos = mpsort.sort(pos, orderby=orderby, comm=comm)
        disp = mpsort.sort(disp, orderby=orderby, comm=comm)

        if self.log.isEnabledFor(logging.INFO):
            self.log.info("Sorting done")

        rng_shift = MPIRandomState(seed=seed2, comm=comm, size=len(pos))
        in_cell_shift = rng_shift.uniform(0, H[i], itemshape=(delta.ndim,))

        pos[...] += in_cell_shift
        pos[...] %= self.pm.BoxSize

        if self.log.isEnabledFor(logging.INFO):
            self.log.info("Catalog shifted.")
            
        #Catalog needs to be shifted in z-coordinate, such that pos and comoving match
        pos[...,0]+=(self.comoving_distance-self.width)*np.ones(pos.shape[0])

        return pos, disp
 
    #Projects baryonic density field to neutral hydrogen overdensity field
    def Find_Neutral_Hydrogen_Fraction(self, density, redshift):
        density_h1=self.tildeC*(1+redshift)**6*density**(self.alpha)
        if self.log.isEnabledFor(logging.INFO):
            self.log.info("Density field projected to neutral hydrogen density field")        
        return density_h1
    
    def Project_Velocity(self, vel, direction):
        return vel[direction]

        
        
    
    
