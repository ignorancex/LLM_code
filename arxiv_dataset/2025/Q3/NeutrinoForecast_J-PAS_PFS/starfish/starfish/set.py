import os
import importlib

__all__ = ["get_settings", "settings"]


def get_settings():
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, "settings.py")
    if os.path.exists(file_path):
        return importlib.import_module("settings")
    else:
        raise FileNotFoundError("settings.py not found in this directory!")


################### survey specifications and fiducial parameters #####################
# import numpy as np
# import seaborn as sns

# sns.set_context('notebook')
# sns.set_palette('husl')

# # function of z
# z_min=np.array([])
# z_max=np.array([])
# z=(z_min+z_max)/2
# V_fid=[]
# n_fid=np.array([]) # (h/Mpc)^3, so power spectra in (Mpc/h)^3
# b_fid=[]

# # basic settings
# A_sky=0 # deg^2
# sigma_0z=0.0000
# k_min=1e-4 # h/Mpc
# k_max=0.25 # h/Mpc
# k_max_transfer=0 # 1/Mpc
# window_len=0
# k_fid=10**np.arange(np.log10(k_min),np.log10(k_max)+0.001,0.001)
# mu_fid=np.arange(-1,1+0.01,0.01)
# k_points=k_fid.size
# mu_points=mu_fid.size
# # basic settings
# f_sky=0
# ell_min=0
# ell_max=0
# FWHM=np.array([]) # rad
# sigma2_T=(np.array([]))**2 # (muK rad)^2
# sigma2_P=(np.array([]))**2 # (muK rad)^2


# # accuracy settings
# nonlinear=True
# halofit_ver='mead2020' # 'mead2020': more accurate for massive neutrinos, higher errors expected
# AccuracyBoost=1 # 3
# lAccuracyBoost=1 # 3
# k_per_logint=100 # 50
# # CAMB/fortran/halofit.f90 tolerance=1e-6

# # constants
# nu_mass_num=1

# # fiducial values
# omega_m0=0
# omega_b0=0
# h=0
# n_s=0
# sigma_8=0
# m_nu=0
# N_eff=0
# w0=0
# wa=0

# # variables
# var_name=[]
# var_exp=[]
# var_num=len(var_name)
# cosmo_value=[]
# cosmo_num=len(cosmo_value)
# temp_path=''
# save_path=''

settings = get_settings()
