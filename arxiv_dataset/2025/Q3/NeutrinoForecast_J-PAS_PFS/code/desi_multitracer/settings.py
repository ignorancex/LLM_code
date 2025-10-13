################### survey specifications and fiducial parameters #####################
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context('notebook')
sns.set_palette('husl')

tracers=['BGS','LRG','ELG','QSO']
def read_info():
    df=pd.read_excel('tracers.xlsx',header=0,index_col=0)
    nuisance_name=[]
    nuisance_exp=[]
    for i in range(1,df.shape[0]+1):
        for t in tracers:
            nuisance_name.append(f'p_s{i}({t})')
            nuisance_exp.append(rf'$p_\mathrm{{s{i}}}(\mathrm{{{t}}})$')
    for i in range(1,df.shape[0]+1):
        for t in tracers:
            nuisance_name.append(f'lnbs{i}({t})')
            nuisance_exp.append(rf'$\ln(b\sigma_8)_{i}(\mathrm{{{t}}})$')
    
    return df.index.values,df.values,nuisance_name,nuisance_exp
        
# function of z
z,n_fid,nuisance_name,nuisance_exp=read_info()
z_min=z-0.1
z_max=z+0.1
b_fid=[]
V_fid=[]

# basic settings
label='DESI'
A_sky=14000 # deg^2
sigma_0z=[0.0005,0.0005,0.0005,0.001]
k_min=1e-4 # h/Mpc
k_max=0. # h/Mpc
k_max_transfer=10 # 1/Mpc
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
var_name=['omega_m0','omega_b0','h','n_s','sigma_8','m_nu','N_eff','w0','wa']+nuisance_name
var_exp=[r'$\Omega_\mathrm{m,0}$',r'$\Omega_\mathrm{b,0}$',r'$h$',r'$n_\mathrm{s}$',r'$\sigma_8$',r'$\sum m_\nu\;[\mathrm{eV}]$',r'$N_\mathrm{eff}$',r'$w_0$',r'$w_a$']+nuisance_exp
var_num=len(var_name)
cosmo_value=[omega_m0,omega_b0,h,n_s,sigma_8,m_nu,N_eff,w0,wa]
cosmo_num=len(cosmo_value)
powers=0
temp_path='d:/large_array_temp/'
save_path=f'../data/DESI_all/'

def get_Dz():
    import camb
    
    k_eval=1e-5
    z_eval=np.concatenate(([0],z,))
    
    omnuh2 = m_nu / 94.07 * (N_eff / 3) ** (3 / 4)
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100 * h,
        ombh2=omega_b0 * h**2,
        omch2=(omega_m0 - omega_b0) * h**2 - omnuh2,
        omk=0,
        num_massive_neutrinos=nu_mass_num,
        mnu=m_nu,
        nnu=N_eff,
    )
    pars.InitPower.set_params(ns=n_s)
    pars.set_matter_power(
        redshifts=[0],
        kmax=k_max_transfer,
        k_per_logint=k_per_logint,
        nonlinear=nonlinear,
        accurate_massive_neutrino_transfers=True,
    )
    pars.NonLinearModel.halofit_version = halofit_ver
    pars.set_accuracy(
        AccuracyBoost=1, lAccuracyBoost=1
    )

    # scale sigma_8 of *matter*
    A_s = pars.InitPower.As * (sigma_8 / camb.get_results(pars).get_sigma8_0()) ** 2
    pars.InitPower.set_params(As=A_s, ns=n_s)

    pars.set_matter_power(
        redshifts=np.concatenate(([0],z,)),
        kmax=k_max_transfer,
        k_per_logint=k_per_logint,
        nonlinear=nonlinear,
        accurate_massive_neutrino_transfers=True,
        silent=True,
    )
    res = camb.get_results(pars)
    PK = res.get_matter_power_interpolator(
        nonlinear=nonlinear,
        var1="delta_nonu",
        var2="delta_nonu",
        hubble_units=False,
        k_hunit=True,
        log_interp=True,
        silent=True,
    )
    
    pks=PK.P(z_eval,k_eval)*h**3 # in (Mpc/h)^3
    Dz=np.sqrt(pks[1:]/pks[0])
    
    return Dz

Dz=get_Dz()
b_fid=np.zeros((len(z),len(tracers)))
b_fid[:,0]=1.34/Dz
b_fid[:,1]=1.7/Dz
b_fid[:,2]=0.84/Dz
b_fid[:,3]=0.53+0.289*(1+z)**2
window_len=599

k_max=0.25 #0.1/Dz # h/Mpc
k_fid=10**np.arange(np.log10(k_min),np.log10(np.max(k_max))+0.001,0.001)
k_points=k_fid.size