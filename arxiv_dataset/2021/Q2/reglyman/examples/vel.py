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
Performs the estimation of peculiar velocities from the density field.

WARNING: This method is untested alpha-quality software, do expect errors.

First a synthetic baryonic density is created. The creation of synthetic data
analogous to the creation of synthetic data in the inversion examples and the
data generation examples. We refer to these documentations. 

The velocity is computed from the logarithmic density perturbation d by:
    <v> = C_{v, d} C_{d, d}^(-1) d
where C denotes the correlation. This relation has been established by Pichon 2001,
but in opposition we use the synthetic correlations instead of any analytic choice.
The relation can be proven in the context of Bayesian inference with Gaussian
distributed velocity and a lognormal distributed density perturbation.

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
	'background_box' : np.array([10, 10]),
	'background_sampling' : np.array([15, 15]),
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

#For simplicty we assume vanishing peculiar velocities here. Just needs to be commented if peculiar velocities should be taken into account
vel_baryon = 1*vel_baryon

####################################################################################################################################################
#Initialize Velocity Calculator
#To save computation time we only estimate the velocity from ten bins wide pixels
Estimator= Interpolation(parameters_box, comoving, comoving[::10])
#Perform estimation with maximally 10 iterations of self calibration of density
#and smoothing of density field with Gaussian blur with width sigma=1
#Density must be provided as flat array
#Density perturbation is corrected for typical bias
delta_est, vel_est = Estimator.Perform_Estimation(np.log(delta_baryon.flatten()), maxiter=10, sigma=1)

#Plot results
import matplotlib.pyplot as plt 

#The correlations are normalized. Hence, we need to find the correct amplitude of the velocities here.
#In application of realistic data, this has to be specified prior to analysis,
#i.e. one needs to specify the variance and mean of the correct densitty field. 
factor = np.std(vel_baryon)/np.std(vel_est)
vel_est *= factor
vel_est -= np.mean(vel_est)

plt.plot(vel_est[500:600], label='est')
plt.plot(vel_baryon[2, :, :, :].flatten()[5000:6000:10], label='true')
plt.legend()
plt.show()

print(np.mean(vel_baryon))
print(np.mean(vel_est))