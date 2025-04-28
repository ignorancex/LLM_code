from reglyman.density import Data_Generation, Cosmo_Translate
from reglyman.tomography import Interpolation

import regpy.stoprules as rules

import numpy as np
import logging

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

'''
Performs the interpolation the density field to a finer grid.

WARNING: This method is untested alpha-quality software, do expect errors.

First a synthetic baryonic density is created. The creation of synthetic data
analogous to the creation of synthetic data in the inversion examples and the
data generation examples. We refer to these documentations. 

The density is interpolated by using the logarithmic density perturbation d by:
    <d> = C^{3D}_{d, d} C^{obs}_{d, d}^(-1) d
where C denotes the correlation of the logarithmic density perturbation between
the observed lines of sight (obs) and the covariance between all lines of sight
(3D).

This relation has been established by Pichon 2001,
but in opposition we use the synthetic correlations instead of any analytic choice.
The relation can be proven in the context of Bayesian inference with Gaussian
distributed velocity and a lognormal distributed density perturbation.

If the covariance between two points (x1, y1, z1) and (x2, y2, z2) can be separated
as C = fC(x1, x2, y1, y2) * g(x3, y3), then only the transverse parts of the 
covariances have to be computed since the longitudinal part exactly cancels. 
We assume that this assumption is satisfied and only compute the transverse covarinaces
which reduces the computational cost heavily.
'''

'''
Parameters for the creation of synthetic data
->cosmology: The used cosmology. Specified keywords must match the keywords of available in nbodykit.
->jeans_length: Jeans Length for computing the matter power spectrum.
->background_box, width: We distinguish between the dimension along the line of sight (width, length of the Ly-alpha forest)
	 and perpendicular to the line of sight (background_box, field of view of illuminating quasars in the background).
	The density is computed in a box with size: background_box[0] x background_box[1] x width.
->background_sampling, nr_pix: Sampling in the background field (background box) and along the line of sight (nr_pix). 
->redshift: redshift of the box.
->seed: Seed for random Gaussian field.

->beta, t_med: Equation of state: T(x, z) = T_0 (z) delta_b^(2 beta), where t_med=T_0 is the temperature at mean density
->tilde_c: proportionality factor between neutral hydrogen density and baryonic density perturbation: n_{HI} = tilde_c * (1+z)^6 delta_b
Note n_{HI} is the density and delta_b the density perturbation (i.e. n_b divided by the mean baryonic density)
'''

parameters_box={	
	'cosmology' : 'Planck15',
	'jeans_length' : 0.16,
	'background_box' : np.array([1, 1]),
	'background_sampling' : np.array([10, 10]),
	'width' : 10,
	'nr_pix' : 300,
	'redshift' : 2.5,
	'seed' : 12
	}

parameters_igm={
'beta' : 0.2,
'tilde_c' : 1.2*10**(-8),
't_med' : 1
}

#Computes the Baryon overdensity and peculiar velocities
Generator_baryon=Data_Generation(parameters_box, parameters_igm, use_baryonic=True)
delta_baryon, vel_baryon, comoving=Generator_baryon.Compute_Density_Field()

#Translates the comoving distance to a Hubble distance and redshift (by the use of astropy)
Trans=Cosmo_Translate(comoving, parameters_box['cosmology'])
hubble_vel=Trans.Compute_Hubble_Velocity()
hubble_flow=Trans.Convert_To_Numpy(hubble_vel)
redshift=Trans.Compute_Redshift()
hubble=3*10**5*redshift/(1+parameters_box['redshift'])

#Interpolates the computed densities and velocities on a uniform grid
delta_baryon, vel_baryon, hubble, comoving, redshift=Trans.Rebin_all(delta_baryon.copy(), vel_baryon.copy(), hubble, parameters_box['nr_pix'])
###############################################################################
  
#Choose randomly a subset of lines of sight to interpolate 
Nr_LOS = np.prod( parameters_box['background_sampling'] )
sep = 2

mean_separation = sep
    
Nr_obs=int(Nr_LOS/mean_separation**2)

np.random.seed(12)
choice = np.random.choice(int(Nr_LOS), Nr_obs, replace=False)
choice.sort()
index = [int(i) in choice for i in range(int(Nr_LOS))]
###############################################################################

#Initialize Interpolation object
Estimator= Interpolation(parameters_box, comoving, comoving[::10])
#Reshape delta baryon
delta_baryon = delta_baryon.reshape((Nr_LOS, parameters_box['nr_pix']))
#Perform interpolation for logarithmic density pertrubation
result = Estimator.Perform_Interpolation(np.log(delta_baryon), index)
#project to density perturbation again
result = np.exp(result)

      
        
        
