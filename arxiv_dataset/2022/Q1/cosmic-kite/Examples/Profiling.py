from cosmic_kite import cosmic_kite
import matplotlib.pyplot as plt
import camb
import numpy as np
from time import time


pars = camb.read_ini('/home/martin/Descargas/CAMB/inifiles/planck_2018.ini')

# Let's create a fiducial planck spectra
H0_true  = 67.32117
omb_true = 0.0223828
omc_true = 0.1201075
n_true   = 0.9660499
tau_true = 0.05430842
As_true  = 2.100549e-09

H0_sigma  = 0.92
omb_sigma = 0.00022
omc_sigma = 0.0021
n_sigma   = 0.0057
tau_sigma = 0.008
As_sigma  = 0.034e-9

# Let's choose 1000 random cosmologies around planck values
nran = 1000
params = np.random.normal(loc = [omb_true, omc_true, H0_true, n_true, tau_true, As_true], scale = [omb_sigma, omc_sigma, H0_sigma, n_sigma, tau_sigma, As_sigma],size = (nran, 6))

# Let's measure the time to compute 1000 spectra with CAMB
camb_start = time()

for i in range(nran):
  pars.set_cosmology(H0 = params[i,2], ombh2 = params[i,0], omch2 = params[i,1], tau = params[i,4])
  pars.InitPower.set_params(As = params[i,5], ns = params[i,3])
  results = camb.get_results(pars)
  powers  = results.get_cmb_power_spectra(pars, CMB_unit='muK')

camb_end = time()
print(f'CAMB took {camb_end - camb_start} seconds to compute the PS for ', nran, ' random cosmologies!')

# Let's measure the time to compute 1000 spectra with Cosmic-kite in a loop

ck_start = time()
for i in range(nran):
  ps = cosmic_kite.pars2ps(params[i,:].reshape(1, -1))

ck_end = time()
print(f'Cosmic-kite took {ck_end - ck_start} seconds to compute the power spectra for ',nran,' random cosmologies in loop mode!')
print('This is ' + str( (camb_end - camb_start) / (ck_end - ck_start) ) + ' faster than CAMB')

# Let's measure the time to compute 1000 spectra with Cosmic-kite

ck_start = time()

ps = cosmic_kite.pars2ps(params)

ck_end = time()
print(f'Cosmic-kite took {ck_end - ck_start} seconds to compute the PS for ',nran,' random cosmologies!')
print('This is ' + str( (camb_end - camb_start) / (ck_end - ck_start) ) + ' faster than CAMB')

