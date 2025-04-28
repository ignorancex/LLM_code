from cosmic_kite import cosmic_kite
import matplotlib.pyplot as plt
import camb
import numpy as np

# Let's compute a fiducal planck spectrum with CAMB

pars = camb.read_ini('/home/martin/Descargas/CAMB/inifiles/planck_2018.ini')

H0_true  = np.random.normal(67.32117, 5)
omb_true = np.random.normal(0.0223828, 0.0001)
omc_true = np.random.normal(0.1201075, 0.01)
n_true   = np.random.normal(0.9660499, 0.01)
tau_true = np.random.normal(0.05430842, 0.001)
As_true  = np.random.normal(2.100549e-9, 0.1e-9)

#calculate results for these parameters
pars.set_cosmology(H0 = H0_true, ombh2 = omb_true, omch2 = omc_true, tau = tau_true)
pars.set_for_lmax(3000)
pars.InitPower.set_params(As = As_true, ns = n_true)
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
powers  = results.get_cmb_power_spectra(pars, CMB_unit='muK')
camb_tt = powers['total'][2:2650,0]
camb_ee = powers['total'][2:2650,1]
camb_te = powers['total'][2:2650,3]
l       = np.arange(2,2650)

true_pars = np.array([omb_true, omc_true, H0_true, n_true, tau_true, As_true]).reshape(1,-1)

# The input of the pars2ps function must be an array of shape (n, 6) where n is the number of cosmological models to be computed
ps = cosmic_kite.pars2ps(true_pars)
tt_pred = ps[0][0]
ee_pred = ps[1][0]
te_pred = ps[2][0]

fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.05, 'wspace': 0.})
axs[0].plot(l, camb_tt, label = 'CAMB')
axs[0].plot(l, tt_pred, label = 'Cosmic-kite', linestyle = ':')
axs[0].legend()
axs[0].set(ylabel = r'$D^{TT}_{l}$')

axs[1].plot(camb_tt-tt_pred, label = 'CAMB - Cosmic-kite')
axs[1].legend()
axs[1].set(ylabel = r'$D^{TT}_{l,CAMB}-D^{TT}_{l,CK}$')
axs[1].set(xlabel = 'l')
plt.show()

fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.05, 'wspace': 0.})
axs[0].plot(l, camb_ee, label = 'CAMB')
axs[0].plot(l, ee_pred, label = 'Cosmic-kite', linestyle = ':')
axs[0].legend()
axs[0].set(ylabel = r'$D^{EE}_{l}$')

axs[1].plot(camb_ee-ee_pred, label = 'CAMB - Cosmic-kite')
axs[1].legend()
axs[1].set(ylabel = r'$D^{EE}_{l,CAMB}-D^{EE}_{l,CK}$')
axs[1].set(xlabel = 'l')
plt.show()

fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.05, 'wspace': 0.})
axs[0].plot(l, camb_te, label = 'CAMB')
axs[0].plot(l, te_pred, label = 'Cosmic-kite', linestyle = ':')
axs[0].legend()
axs[0].set(ylabel = r'$D^{TE}_{l}$')

axs[1].plot(camb_te-te_pred, label = 'CAMB - Cosmic-kite')
axs[1].legend()
axs[1].set(ylabel = r'$D^{TE}_{l,CAMB}-D^{TE}_{l,CK}$')
axs[1].set(xlabel = 'l')
plt.show()

