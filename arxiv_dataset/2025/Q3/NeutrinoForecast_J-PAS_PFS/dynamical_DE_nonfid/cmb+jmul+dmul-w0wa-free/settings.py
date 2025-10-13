################### survey specifications and fiducial parameters #####################
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context('notebook')
sns.set_palette('husl')
        
# fiducial values
omega_m0=0.32
omega_b0=0.049
h=0.67
n_s=0.96
sigma_8=0.81
m_nu=0.06
N_eff=3.044
w0=-0.758
wa=-0.82

# variables
var_name=['omega_m0','omega_b0','h','n_s','sigma_8','m_nu','N_eff','w0','wa']
var_exp=[r'$\Omega_\mathrm{m,0}$',r'$\Omega_\mathrm{b,0}$',r'$h$',r'$n_\mathrm{s}$',r'$\sigma_8$',r'$\sum m_\nu\;[\mathrm{eV}]$',r'$N_\mathrm{eff}$',r'$w_0$',r'$w_a$']
var_num=len(var_name)
cosmo_value=[omega_m0,omega_b0,h,n_s,sigma_8,m_nu,N_eff,w0,wa]
cosmo_num=len(cosmo_value)