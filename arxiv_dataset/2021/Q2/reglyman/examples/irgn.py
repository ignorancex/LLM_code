from reglyman.operators import LymanAlphaBar
from reglyman.util import EnergyNorm
from regpy.discrs import UniformGrid
from regpy.hilbert import L2
from regpy.solvers import HilbertSpaceSetting
from reglyman.solvers import IRGN
from reglyman.density import Data_Generation, Cosmo_Translate

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
Performs the whole inversion, i.e. creation of synthetic data, forwardly mapping to Ly-alpha forest flux, adding noise and invert the spectra by IRGN approach
The first two step have been discussed in more details in /synthethic_data/ files.
The inversion is performed in real space.

exact_solution holds the exact baryonic density perturbation
exact_data holds the exact flux data
data holds the noisy data vector used for inversion
reco holds the recovered density perturbation
reco_data holds the recovered flux (i.e. the evaluation of the recovered density profile)

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

#Parameters assumed for the inversion procedure. These are crucial for the performance of the inversion algorithm!
#However, often the parameters describing the EOS of state of the IGM (beta, T_0) are not known.
parameters_inversion = {
        "redshift_back" : 2.5,
        "width" : 10,
        "sampling" : (285, 285),
        "beta" : 0.2,
        "t_med" : 1,
        "tilde_c" : 1.2*10**(-8)
        }

#Noise model; sigma_F^2=F^2/SNR^2+sigma_0^2
parameters_noise = {
        "snr" : 50,
        "sigma_0" : 0.01,
        "seed" : 4
}

#Computes the Baryon overdensity and peculiar velocities
Generator_baryon=Data_Generation(parameters_box, parameters_igm, use_baryonic=True)
delta_baryon, vel_baryon, comoving=Generator_baryon.Compute_Density_Field()

#Compute Cosomology object
cosmology_model = Generator_baryon.cosmo

#Translates the comoving distance to a Hubble distance and redshift (by the use of astropy)
Trans=Cosmo_Translate(comoving, parameters_box['cosmology'])
hubble_vel=Trans.Compute_Hubble_Velocity()
hubble_flow=Trans.Convert_To_Numpy(hubble_vel)
redshift=Trans.Compute_Redshift()
hubble=3*10**5*redshift/(1+parameters_box['redshift'])

#Interpolates the computed densities and velocities on a uniform grid
delta_baryon, vel_baryon, hubble, comoving, redshift=Trans.Rebin_all(delta_baryon.copy(), vel_baryon.copy(), hubble, parameters_box['nr_pix'])

#For simplicty we assume vanishing peculiar velocities here. Just needs to be commented if peculiar velocities should be taken into account
vel_baryon = 0*vel_baryon
#################################################################################################################################################################
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
        op = LymanAlphaBar(domain, parameters_forward, redshift, redshift_spect, hubble, hubble_spect, vel[counter, :])
        exact_solution[counter, :]=delta_baryon[i, j, :]
        exact_data[counter, :]=op(exact_solution[counter, :])
        counter+=1
        print('LOS: ', counter, 'computed')

#Random numbers for noise contribution    
np.random.seed(parameters_noise["seed"])
random=np.random.randn(Nr_LOS, parameters_box['nr_pix'])

###################################################################################################################################################################
'''
Add noise
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

N_spect = parameters_inversion['sampling'][0]
N_space = parameters_inversion['sampling'][1]

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
sigma_rebinned=np.zeros((Nr_LOS, N_spect))

for i in range(Nr_LOS):
    sigma_rebinned[i, :]=rebin(sigma[i, :])*1/np.sqrt(2)
    data_rebinned[i, :]=rebin(data[i, :])
    
data=data_rebinned
sigma=sigma_rebinned

##################################################################################################################################################################
'''
Initial Guess
'''

pre=100*cosmology_model.h*cosmology_model.efunc(parameters_inversion['redshift_back'])/(4.45*10**(-22)*cosmology_model.C)/(3.08*10**22)
init=np.zeros((Nr_LOS, N_spect))
alpha=2-1.4*parameters_inversion['beta']
for i in range(Nr_LOS):
    flux=np.where(data[i, :]>1, 0.99, data[i, :])
    flux=np.where(data[i, :]<0, 0.01, flux)
    init[i, :] = pre*(1-flux)
init = (init/(parameters_inversion['tilde_c']*(1+redshift)**6))**(1/alpha)
init=np.log(init)

####################################################################################################################################################################
'''
Performs the actual inversion
'''
#Domain and codomain for unversion
coords=np.linspace(1, N_space, N_space)
domain=UniformGrid(coords, dtype=float)

coords=np.linspace(1, N_spect, N_spect)
codomain=UniformGrid(coords, dtype=float)

reco=np.zeros((Nr_LOS, N_space))
reco_data=np.zeros((Nr_LOS, N_spect))

#Compute the covariance matrix of baryonic matter along the line of sight
parameters_box['nr_pix']=N_spect
Generator_covariance=Data_Generation(parameters_box, parameters_igm, use_baryonic=True)
covariance_baryon=Generator_covariance.Compute_Linear_Covariance(comoving)
C_0=covariance_baryon
M_0= np.zeros(N_space)

def perform_inversion(data, init, sigma, vel):
    C_D=sigma
        
    op = LymanAlphaBar(domain, parameters_inversion, redshift, redshift, hubble, hubble, vel, gamma=True, codomain=codomain)

    setting = HilbertSpaceSetting(
        op=op,
        Hdomain=L2,
        Hcodomain=L2)

    solver = IRGN(setting, data, init, C_0, M_0, C_D**2, maxit=15, tol=1e-3, restart=10)
    
    energy=EnergyNorm(sigma)
    
    stoprule = (
    rules.CombineRules(
    (rules.CountIterations(10),
    rules.Discrepancy(energy.norm, data, noiselevel=np.sqrt(N_spect), tau=1),
    rules.RelativeChangeData(energy.norm, data, 0.1))))
    
    return solver.run(stoprule)
    

results=Parallel(n_jobs=num_cores)(delayed(perform_inversion)(data[i, :], init[i, :], sigma[i, :], vel[i, :]) for i in range(Nr_LOS))

for i in range(Nr_LOS):
    reco[i, :]=np.exp(results[i][0])
    reco_data[i, :]=results[i][1]

