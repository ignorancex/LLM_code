from reglyman.operators import LymanAlphaHydrogen, ProbCons, ProbData
from regpy.discrs import UniformGrid
from regpy.hilbert import L2, Sobolev
from regpy.solvers import HilbertSpaceSetting
from reglyman.solvers import Gradient_Descent
from reglyman.density import Data_Generation, Cosmo_Translate

import regpy.stoprules as rules

import numpy as np
import logging
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

'''
Performs the whole inversion, i.e. creation of synthetic data, forwardly mapping to Ly-alpha forest flux, adding noise and invert the spectra by the RPC approach
The first two steps have been discussed in more details in /synthethic_data/ files
Note that with the PC method the density is computed in redshift space.

exact_solution holds the exact neutral hydrogen density
exact_data holds the exact flux data
data holds the noisy data vector used for inversion
reco holds the recovered neutral hydrogen density

All these arrays have shape (Nr_LOS, N_space) where Nr_LOS denotes the number of chosen lines of sights and N_space the number of pixels along a single line of sight.
The redshifts and Hubble velocities for each pixel along the line of sight are stored in redshift and Hubble_vel respectively.
'''
##############################################################################################################################################################
'''
Create synthetic overdensity data. More explained in the file synthetic_data/generate_data.py
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

#Parameters for the creation of synthetic data
parameters_box={	
	'cosmology' : 'Planck15',
	'jeans_length' : 0.16,
	'background_box' : np.array([100, 100]),
	'background_sampling' : np.array([100, 100]),
	'width' : 10,
	'nr_pix' : 572,
	'redshift' : 2.5,
	'seed' : 12
	}

parameters_igm={
	'beta' : 0.2,
	'tilde_c' : 1.2*10**(-8),
	't_med' : 1
	}

#Inversion parameters are not needed for the inversion in opposition to the RL and IRGN method

#Noise model; sigma_F^2=F^2/SNR^2+sigma_0^2
parameters_noise = {
        "snr" : 50,
        "sigma_0" : 0.01,
        "seed" : 4
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

dens_hydrogen=Generator_baryon.Find_Neutral_Hydrogen_Fraction(delta_baryon, redshift)

#Interpolates the computed densities and velocities on a uniform grid
dens_hydrogen, vel_baryon, hubble, comoving, redshift=Trans.Rebin_all(dens_hydrogen.copy(), vel_baryon.copy(), hubble, parameters_box['nr_pix'])

#For simplicity we assume vanishing peculiar velocities here. Could be just commented out if peculiar velocities are desired
vel_baryon = 0*vel_baryon
####################################################################################################################################################################
'''
Compute Ly-alpha forest flux from computed overdensities. More explained in the file synthetic_data/....
'''
#Which lines of sights to select from the box
array=10+20*np.linspace(0, 4, 5)

#Number of lines of sights
Nr_LOS=array.shape[0]**2

#Define domain and codomain for forward operator
#Domain and codomain are defined on a uniform grid
coords=np.linspace(1, parameters_box['nr_pix'], parameters_box['nr_pix'])
domain=UniformGrid(coords, dtype=float)

coords=np.linspace(1, parameters_box['nr_pix'], parameters_box['nr_pix'])
codomain=UniformGrid(coords, dtype=float)

#parameters describing the forward operator
#->redshift_back: redshift of box
#->width: length of line of sight in [h^{-1}Mpc]
#->n_pix: (size domain, size codomain)
#->beta, t_med, tilde_c: three parameters describing the forward problem
parameters_forward = {
        "redshift_back" : parameters_box['redshift'],
        "width" : parameters_box['width'],
        "sampling" : (parameters_box['nr_pix'], parameters_box['nr_pix']),
        "beta" : parameters_igm['beta'],
        "t_med" : parameters_igm['t_med'],
        "tilde_c" : parameters_igm['tilde_c']
        }

#Project velocity to velocity in the direction along the line of sight
vel_all=vel_baryon[2, :, :, :]
vel_all*=(hubble[-1]-hubble[0])/(hubble_flow[-1]-hubble_flow[0])
hubble_spect=hubble
redshift_spect=redshift

#Allocate the exact density and exact data
exact_solution=np.zeros((Nr_LOS, parameters_box['nr_pix']))
exact_data=np.zeros((Nr_LOS, parameters_box['nr_pix']))

#Compute for every selected line of sight the exact Ly-alpha forest data
#LymanAlphaBar is the forward operator that computes the Ly-alpha forest flux from the baryonic overdensity
vel=np.zeros((Nr_LOS, parameters_box['nr_pix']))
counter=0
for i in array.astype(int):
    for j in array.astype(int):
        vel[counter, :]=vel_all[i, j, :]
        op = LymanAlphaHydrogen(domain, parameters_forward, redshift, redshift_spect, hubble, hubble_spect, vel[counter, :])
        exact_solution[counter, :]=dens_hydrogen[i, j, :]
        exact_data[counter, :]=op(exact_solution[counter, :])
        counter+=1
        print('LOS: ', counter, 'computed')

#Random numbers for noise contribution    
np.random.seed(parameters_noise["seed"])
random=np.random.randn(Nr_LOS, parameters_box['nr_pix'])

###################################################################################################################################################################
'''
Add noise
For RPC: sigma is not used in inversion algorithm
'''

#Actually computes the noisy data
#Noise model; sigma_F^2=F^2/SNR^2+sigma_0^2
SNRatio=parameters_noise["snr"]
data=np.zeros((Nr_LOS, parameters_box['nr_pix']))
sigma=np.zeros((Nr_LOS, parameters_box['nr_pix']))
sigma_0=parameters_noise["sigma_0"]

#Add noise to data
for i in range(Nr_LOS):
    sigma[i, :]=np.sqrt((exact_data[i, :]/SNRatio)**2+sigma_0**2)
    noise = sigma[i, :]*random[i, :]
    data[i, :] = exact_data[i, :] + noise

##################################################################################################################################################################
'''
Rebin flux
'''

N_spect = int(parameters_box['nr_pix']/2-1)
N_space = int(parameters_box['nr_pix']/2-1)

def rebin(y):
    lower=y[1:parameters_box['nr_pix']-1:2]
    upper=y[2:parameters_box['nr_pix']-1:2]
    return (lower+upper)/2

hubble=rebin(hubble)
redshift=rebin(redshift)
comoving=rebin(comoving)

vel_rebinning = np.zeros([Nr_LOS, N_spect])
for i in range(Nr_LOS):
    vel_rebinning[i, :]=rebin(vel[i, :])
vel=vel_rebinning

exact_data_rebinned=np.zeros([Nr_LOS, N_spect])
for i in range(Nr_LOS):
    exact_data_rebinned[i,:]=rebin(exact_data[i, :])
    print('Smoothing+Rebinning: ', i, 'computed')    
exact_data=exact_data_rebinned

exact_solution_rebinned=np.zeros([Nr_LOS, N_spect])
for i in range(Nr_LOS):
    exact_solution_rebinned[i, :]=rebin(exact_solution[i, :])
exact_solution=exact_solution_rebinned
        
data_rebinned=np.zeros((Nr_LOS, N_spect))
for i in range(Nr_LOS):
    data_rebinned[i, :]=rebin(data[i, :])
data=data_rebinned

####################################################################################################################################################################
'''
Initial Guess
'''

from nbodykit.lab import *
cosmo=cosmology.Planck15
pre=100*cosmo.h/(4.45*10**(-22)*cosmo.C)/(3.08*10**22)
alpha=2-1.4*parameters_igm['beta']
init=np.zeros((Nr_LOS, N_space))
for i in range(Nr_LOS):
    flux=np.where(data[i, :]>1, 0.99, data[i, :])
    flux=np.where(data[i, :]<0, 0.01, flux)
    init[i, :] = 10*pre*(1-flux)
####################################################################################################################################################################
'''
Performs the actual inversion
'''

sep=0.05

coords=sep*np.linspace(1, N_space, N_space)
domain=UniformGrid(coords, dtype=float)

coords=sep*np.linspace(1, N_spect, N_spect)
codomain=UniformGrid(coords, dtype=float)

y=np.log(exact_solution)
mu=np.mean(y)
sigma=np.std(y)

reco=np.zeros((Nr_LOS, N_space))

H1_domain=Sobolev(domain, index=1)
L2_domain=L2(domain)

penalty = lambda x: H1_domain.gram(x)-L2_domain.gram(x)

Preprocess=ProbData(0.99, 0, mu, sigma)
Probability_data, Delta_bright=Preprocess.perform(data)
Probability_data.reshape((Nr_LOS, N_spect))

op = ProbCons(domain, mu, sigma, Delta_bright, codomain=codomain, exp=True)

setting = HilbertSpaceSetting(
    op=op,
    Hdomain=Sobolev(index=0),
    Hcodomain=Sobolev(index=0))


def perform_inversion(Probability_data, init):

    descent = Gradient_Descent(setting, Probability_data, np.log(init), stepsize=0.5, alpha=0.2, penalty=penalty)
    
    stoprule = (
    rules.CombineRules(
    (rules.CountIterations(500),
    rules.RelativeChangeData(setting.Hcodomain.norm, Probability_data, 0))))
    
    return descent.run(stoprule)
    

results=Parallel(n_jobs=num_cores)(delayed(perform_inversion)(Probability_data[i, :], init[i, :]) for i in range(Nr_LOS))

for i in range(Nr_LOS):
    reco[i, :]=np.exp(results[i][0])
