'''
Computes the Ly-alpha forest flux from the neutral hydrogen density field.

In:
->redshift, hubble, vel, delta: Typically computed by the file generate_data.py, for demonstration issues here chosen a priori
	redshift and hubble are the redshift and hubble distance of each pixel in the line of sight, vel the field of peculiar velocities.
'''

from reglyman.operators import LymanAlphaHydrogen
from regpy.discrs import UniformGrid

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

#Number of pixels in spectrum along single line of sight (N_spect) and in density space (N_space). 
#Typical N_space=N_spect
N_space=200
N_spect=200

#Which lines of sights to select from the box
array=10+20*np.linspace(0, 4, 5)

#Number of lines of sights
Nr_LOS=array.shape[0]**2

#Define domain and codomain for forward operator
#Domain and codomain are defined on a uniform grid
coords=np.linspace(1, N_space, N_space)
domain=UniformGrid(coords, dtype=float)

coords=np.linspace(1, N_spect, N_spect)
codomain=UniformGrid(coords, dtype=float)

#parameters describing the forward operator
#->redshift_back: redshift of box
#->width: length of line of sight in [h^{-1}Mpc]
#->n_pix: (size domain, size codomain)
#->beta, t_med, tilde_c: three parameters describing the forward problem
parameters = {
        "redshift_back" : 2.5,
        "width" : 10,
        "sampling" : (N_space, N_spect),
        "beta" : 0.2,
        "t_med" : 1,
        "tilde_c" : 1.2*10**(-8)
        }

#Mock redshifts, typically computed by generate_data
redshift=2.5-6.15*10**(-5)*np.linspace(0, N_space, N_space)
hubble=431647.83-8.51*np.linspace(0, N_space, N_space)
vel=np.zeros((3, 100, 100, N_space))

#Project velocity to velocity in the direction along the line of sight
vel=vel[2, :, :, :]

#Interpolate hubble and redshift to the values in the codomain
#Only works for N_space/N_spect=int, otherwise has to be replaced by a more serious interpolation scheme
hubble_spect=hubble[0:N_space:int(N_space/N_spect)]
redshift_spect=redshift[0:N_space:int(N_space/N_spect)]

#Mock overdensity. Typically computed by generate_data.py
dens_hydrogen=np.zeros((100, 100, N_space))
for i in range(100):
    for j in range(100):
        dens_hydrogen[i, j, :]=parameters['tilde_c']*(1+parameters['redshift_back'])**6*(1.1+np.sin(i+j+np.linspace(0, 2*np.pi, N_space)))**(2-1.4*parameters['beta'])

#Allocate the exact density and exact data
exact_solution=np.zeros((Nr_LOS, N_space))
exact_data=np.zeros((Nr_LOS, N_spect))

#Compute for every selected line of sight the exact Ly-alpha forest data
#LymanAlphaBar is the forward operator that computes the Ly-alpha forest flux from the baryonic overdensity
counter=0
for i in array.astype(int):
    for j in array.astype(int):
        op = LymanAlphaHydrogen(domain, parameters, redshift, redshift_spect, hubble, hubble_spect, vel_pec=vel[i, j, :])
        exact_solution[counter, :]=dens_hydrogen[i, j, :]
        exact_data[counter, :]=op(exact_solution[counter, :])
        counter+=1
        print('LOS: ', counter, 'computed')

###############################################################################

#Noise model; sigma_F^2=F^2/SNR^2+sigma_0^2
parameters_noise = {
        "snr" : 50,
        "sigma_0" : 0.01,
        "seed" : 4
}

#Random numbers for noise contribution    
np.random.seed(parameters_noise["seed"])
random=np.random.randn(Nr_LOS, N_spect)

#Actually computes the noisy data
#Noise model; sigma_F^2=F^2/SNR^2+sigma_0^2
SNRatio=parameters_noise["snr"]
data=np.zeros((Nr_LOS, N_spect))
sigma=np.zeros((Nr_LOS, N_spect))
sigma_0=parameters_noise["sigma_0"]

#Add noise to data
for i in range(Nr_LOS):
    sigma[i, :]=np.sqrt((exact_data[i, :]/SNRatio)**2+sigma_0**2)
    noise = sigma[i, :]*random[i, :]
    data[i, :] = exact_data[i, :] + noise

#Optional: Data rebinning, smoothing to mimic instrumental broadening



