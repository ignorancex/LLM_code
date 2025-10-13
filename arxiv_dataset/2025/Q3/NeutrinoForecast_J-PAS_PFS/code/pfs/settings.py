################### survey specifications and fiducial parameters #####################
import numpy as np
import seaborn as sns
sns.set_context('notebook')
sns.set_palette('husl')

# function of z
z_min=np.array([0.6,0.8,1.0,1.2,1.4,1.6,2.0])
z_max=np.array([0.8,1.0,1.2,1.4,1.6,2.0,2.4])
z=(z_min+z_max)/2
# V_fid=np.array([0.59,0.79,0.96,1.09,1.19,2.58,2.71])*1e9 # (Mpc/h)^3
V_fid=[]
n_fid=np.array([1.9,6.0,5.8,7.8,5.5,3.1,2.7])*1e-4 # (h/Mpc)^3, so power spectra in (Mpc/h)^3
b_fid=[1.18,1.26,1.34,1.42,1.50,1.62,1.78]

# basic settings
label='PFS'
A_sky=1464 # deg^2
sigma_0z=0.0007
k_min=1e-4 # h/Mpc
k_max=0. # h/Mpc
k_max_transfer=10 # 1/Mpc
window_len=599
k_fid=0
mu_fid=np.arange(-1,1+0.01,0.01)
k_points=0
mu_points=mu_fid.size

# accuracy settings
nonlinear=True
halofit_ver='mead2020' # 'mead2020': more accurate for massive neutrinos, higher errors expected
AccuracyBoost=2 # 3
lAccuracyBoost=2 # 3
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
w0=-1
wa=0

# variables
var_name=['omega_m0','omega_b0','h','n_s','sigma_8','m_nu','N_eff','w0','wa','p_s1','p_s2','p_s3','p_s4','p_s5','p_s6','p_s7','lnbs1','lnbs2','lnbs3','lnbs4','lnbs5','lnbs6','lnbs7',]
var_exp=[r'$\Omega_\mathrm{m,0}$',r'$\Omega_\mathrm{b,0}$',r'$h$',r'$n_\mathrm{s}$',r'$\sigma_8$',r'$\sum m_\nu$',r'$N_\mathrm{eff}$',r'$w_0$',r'$w_a$',r'$p_\mathrm{s1}$',r'$p_\mathrm{s2}$',r'$p_\mathrm{s3}$',r'$p_\mathrm{s4}$',r'$p_\mathrm{s5}$',r'$p_\mathrm{s6}$',r'$p_\mathrm{s7}$',r'$\ln(b\sigma_8)_1$',r'$\ln(b\sigma_8)_2$',r'$\ln(b\sigma_8)_3$',r'$\ln(b\sigma_8)_4$',r'$\ln(b\sigma_8)_5$',r'$\ln(b\sigma_8)_6$',r'$\ln(b\sigma_8)_7$',]
var_num=len(var_name)
cosmo_value=[omega_m0,omega_b0,h,n_s,sigma_8,m_nu,N_eff,w0,wa]
cosmo_num=len(cosmo_value)
powers=0
temp_path='d:/large_array_temp/'
save_path='../data/PFS/'

k_max=0.25 #0.1/Dz # h/Mpc
k_fid=10**np.arange(np.log10(k_min),np.log10(np.max(k_max))+0.001,0.001)
k_points=k_fid.size