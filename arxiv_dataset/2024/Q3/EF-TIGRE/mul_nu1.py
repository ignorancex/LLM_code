import numpy as np
import os
from scipy.interpolate import interp1d

#You need to make sure that the current directory is the local oath to the repository, other the file names will not be found.

def interpolate_mul_nu1(base_dir=''):
    return [interpolate_mu0(base_dir=base_dir), interpolate_mu2(base_dir=base_dir), interpolate_mu4(base_dir=base_dir), interpolate_nu1(base_dir=base_dir)]

def interpolate_mu0(base_dir=''):
    data = np.loadtxt(os.path.join(base_dir, 'mul_nu1/mu0_z10.0_fid.dat'))
    val_d = data[:, 0]
    val_mu0 = data[:, 1]
    mu0 = interp1d(val_d, val_mu0, kind='cubic')
    return mu0 

def interpolate_mu2(base_dir=''):
    data = np.loadtxt(os.path.join(base_dir,'mul_nu1/mu2_z10.0_fid.dat'))
    val_d = data[:, 0]
    val_mu2 = data[:, 1]
    mu2 = interp1d(val_d, val_mu2, kind='cubic')
    return mu2 

def interpolate_mu4(base_dir=''):   
    data = np.loadtxt(os.path.join(base_dir,'mul_nu1/mu4_z10.0_fid.dat'))
    val_d = data[:, 0]
    val_mu4 = data[:, 1]
    mu4 = interp1d(val_d, val_mu4, kind='cubic')
    return mu4 

def interpolate_nu1(base_dir=''): #Note: This is in fact the function nu_1 without the factor d*H0
    data = np.loadtxt(os.path.join(base_dir,'mul_nu1/nu1_z10.0_fid.dat'))
    val_d = data[:, 0]
    val_nu1 = data[:, 1]
    nu1 = interp1d(val_d, val_nu1, kind='cubic')
    return nu1 

def interpolate_Pin(base_dir=''):
    data = np.loadtxt(os.path.join(base_dir,'mul_nu1/Pkz10.0.dat'))
    val_k = data[:, 0]
    val_Pk = data[:, 1]
    Pk = interp1d(val_k, val_Pk, kind='cubic', fill_value='extrapolate')
    
    # Extrapolation for large k-values
    alpha = (np.log(val_Pk[-1])-np.log(val_Pk[-2]))/(np.log(val_k[-1])-np.log(val_k[-2]))
    A = val_Pk[-1]/(val_k[-1]**alpha)
    
    # Define function including extrapolation
    def Pk_all(k):
        return np.where(k < val_k[-1], Pk(k), A*k**alpha)
    
    return Pk_all
