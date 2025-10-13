################### survey specifications and fiducial parameters #####################
import numpy as np
import seaborn as sns
sns.set_context('notebook')
sns.set_palette('husl')

# basic settings
f_sky=0.4
ell_min=51
ell_max=3000
label='CMB-S4'
FWHM=np.array([3.0])/60*np.pi/180 # rad, 150 GHz
sigma2_T=(np.array([1.0])/60*np.pi/180)**2 # (muK rad)^2
sigma2_P=(np.array([1.41])/60*np.pi/180)**2 # (muK rad)^2
k_max_transfer=10 # 1/Mpc

# accuracy settings
nonlinear=True
halofit_ver='mead2020' # 'mead2020': more accurate for massive neutrinos, higher errors expected
AccuracyBoost=2 # 3
lAccuracyBoost=2 # 3
lens_accuracy=2 # 1 is only for Planck-like level
k_per_logint=100 # 50
# CAMB/fortran/halofit.f90 tolerance=1e-6

# constants
nu_mass_num=1

# fiducial values
omega_m0=0.32
omega_b0=0.049
h=0.67
n_s=0.96
sigma_8=0.81
m_nu=0.06
N_eff=3.044
tau=0.054
w0=-0.758
wa=-0.82

# variables
var_name=['omega_m0','omega_b0','h','n_s','sigma_8','m_nu','N_eff','tau','w0','wa']
var_exp=[r'$\Omega_\mathrm{m,0}$',r'$\Omega_\mathrm{b,0}$',r'$h$',r'$n_\mathrm{s}$',r'$\sigma_8$',r'$\sum m_\nu$',r'$N_\mathrm{eff}$',r'$\tau$',r'$w_0$',r'$w_a$']
var_num=len(var_name)
cosmo_value=[omega_m0,omega_b0,h,n_s,sigma_8,m_nu,N_eff,tau,w0,wa]
cosmo_num=len(cosmo_value)
powers=0