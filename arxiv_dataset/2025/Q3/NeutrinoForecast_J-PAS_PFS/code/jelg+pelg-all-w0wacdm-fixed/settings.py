################### survey specifications and fiducial parameters #####################
import numpy as np
import seaborn as sns
sns.set_context('notebook')
sns.set_palette('husl')

var_name=['omega_m0','omega_b0','h','n_s','sigma_8','m_nu','N_eff','w0','wa']

z_JPAS_ELG=np.array([0.3,0.5,0.7,0.9,1.1,1.3])
name_JPAS_ELG=var_name.copy()
for z in z_JPAS_ELG:
    name_JPAS_ELG.append(f'ps_{z}')
for z in z_JPAS_ELG:
    name_JPAS_ELG.append(f'lnbs_{z}')
    
z_PFS_ELG=np.array([0.7,0.9,1.1,1.3,1.5,1.8,2.2])
name_PFS_ELG=var_name.copy()
for z in z_PFS_ELG:
    name_PFS_ELG.append(f'ps_{z}')
for z in z_PFS_ELG:
    name_PFS_ELG.append(f'lnbs_{z}')

z=np.array([0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.8,2.2])
name=var_name.copy()
for zz in z:
    name.append(f'ps_{zz}')
for zz in z:
    name.append(f'lnbs_{zz}')

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
var_exp=[r'$\Omega_\mathrm{m,0}$',r'$\Omega_\mathrm{b,0}$',r'$h$',r'$n_\mathrm{s}$',r'$\sigma_8$',r'$\sum m_{\nu}\;[\mathrm{eV}]$',r'$w_0$',r'$w_a$']
var_num=len(var_name)
cosmo_value=[omega_m0,omega_b0,h,n_s,sigma_8,m_nu,w0,wa]
cosmo_num=len(cosmo_value)