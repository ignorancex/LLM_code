import numpy as np
from reglyman.density import Data_Generation, Cosmo_Translate, Analysis_Tools

'''
In this file we compute the nonlinear density field, velocity perturbation and mock data by the lognormal approach.
The computation is strongly based on the computation implemented in nbodykit.

In:
->cosmology: The used cosmology. Specified keywords must match the keywords of available in nbodykit.
->jeans_length: Jeans Length for computing the matter power spectrum.
->background_box, width: We distinguish between the dimension along the line of sight (width, length of the Ly-alpha forest)
	 and perpendicular to the line of sight (background_box, field of view of illuminating quasars in the background).
	The density is computed in a box with size: background_box[0] x background_box[1] x width.
->background_sampling, nr_pix: Sampling in the background field (background box) and along the line of sight (nr_pix). 
->redshift: redshift of the box.
->seed: Seed for random Gaussian field.

->beta: Slope of the Eq. of state.
->tilde_c: proportionality factor between neutral hydrogen density and baryonic density perturbation: n_{HI} = tilde_c * (1+z)^6 (1+delta_b)
'''

parameters_box={	
	'cosmology' : 'Planck15',
	'jeans_length' : 0.16,
	'background_box' : np.array([100, 100]),
	'background_sampling' : np.array([100, 100]),
	'width' : 10,
	'nr_pix' : 200,
	'redshift' : 2.5,
	'seed' : 12
}

parameters_igm={
	'beta' : 0.2,
	'tilde_c' : 1.2*10**(-8)
}

#parameters for poisson sampling of the final density field in order to compute mock galaxies in the box
parameters_sampling={
	'seed1' : 1,
	'seed2' : 1,
	'nbar' : 10**(-3)
}

#Compute CDM overdensity delta_cdm and the peculiar velocity field of the cold dark matter.
#comoving contains the comoving distance to each pixel (array of same length as delta_cdm)
Generator_cdm=Data_Generation(parameters_box, parameters_igm)
delta_cdm, vel_cdm, comoving=Generator_cdm.Compute_Density_Field()

#Translates the comoving distance to a Hubble distance and redshift (by the use of astropy)
Trans=Cosmo_Translate(comoving, parameters_box['cosmology'])
hubble_vel=Trans.Compute_Hubble_Velocity()
hubble=Trans.Convert_To_Numpy(hubble_vel)
redshift=Trans.Compute_Redshift()

#Computes the Baryon overdensity and peculiar velocities
Generator_baryon=Data_Generation(parameters_box, parameters_igm, use_baryonic=True)
delta_baryon, vel_baryon, _=Generator_baryon.Compute_Density_Field()

#Projects the baryonic matter overdensity to the number density of neutral hydrogen
dens_hydrogen=Generator_baryon.Find_Neutral_Hydrogen_Fraction(delta_baryon, redshift)

#Interpolates the computed densities and velocities on a regular grid
delta_cdm_rebin, vel_cdm_rebin, hubble_rebin, comoving_rebin, redshift_rebin=Trans.Rebin_all(delta_cdm.copy(), vel_cdm.copy(), hubble, parameters_box['nr_pix'])
delta_baryon_rebin, vel_baryon_rebin=Trans.Rebin_delta_vel(delta_baryon.copy(), vel_baryon.copy(), hubble, hubble_rebin, parameters_box['nr_pix'])
dens_hydrogen_rebin=Trans.Rebin_delta(dens_hydrogen.copy(), hubble, hubble_rebin, parameters_box['nr_pix'])

#Find mean and standard deviation of lognormal approximation, used as input for PC and RPC inversion
Analysis=Analysis_Tools()
gaussian=Analysis.Find_Gaussian(dens_hydrogen/np.mean(dens_hydrogen))

#Computes the covariance matrix of baryonic matter and cdm along the line of sight, used as input for IRGN inversion-
covariance_baryon=Generator_baryon.Compute_Linear_Covariance(comoving)
covariance_cdm=Generator_cdm.Compute_Linear_Covariance(comoving)

#Computes mock galaxies with position pos and displacement disp
pos, disp=Generator_baryon.PoissonSample(delta_baryon, parameters_sampling)
