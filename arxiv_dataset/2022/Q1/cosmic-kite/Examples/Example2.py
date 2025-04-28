from cosmic_kite import cosmic_kite
import matplotlib.pyplot as plt
import camb
import numpy as np

# Let's compute a spectra with CAMB for random cosmologies

pars = camb.read_ini('/home/martin/Descargas/CAMB/inifiles/planck_2018.ini')

H0  = np.random.uniform(61, 74)
omb = np.random.uniform(0.0213, 0.023)
omc = np.random.uniform(0.11, 0.125)
n   = np.random.uniform(0.94, 0.985)
tau = np.random.uniform(0.02, 0.092)
As  = np.random.uniform(1.95e-9, 2.2e-9)

#calculate results for these parameters
pars.set_cosmology(H0 = H0, ombh2 = omb, omch2 = omc, tau = tau)
pars.set_for_lmax(3000)
pars.InitPower.set_params(As = As, ns = n)
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
powers  = results.get_cmb_power_spectra(pars, CMB_unit='muK')
camb_tt = powers['total'][2:2650,0]
camb_ee = powers['total'][2:2650,1]
camb_te = powers['total'][2:2650,3]
l       = np.arange(2,2650)

camb_ps = np.concatenate((camb_tt, camb_ee, camb_te))

# The input of the ps2pars function must be an array of shape (n, 2450) where n is the number of cosmological models to be computed

pred_pars = cosmic_kite.ps2pars(camb_ps.reshape(1,-1))[0]
rel_diff = ([omb, omc, H0, n, tau, As] - pred_pars)/[omb, omc, H0, n, tau, As]

plt.scatter(['omb', 'omc', 'H0', 'n', 'tau', 'As'],rel_diff)
plt.ylabel(r'$(\Omega_{R}-\Omega_{P})/\Omega_{R}$')
plt.show()

