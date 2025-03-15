#imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import warnings
from joblib import Parallel, delayed

from functions.utils import der_diff
from functions.grid_functions import reduce_res

def nu_calc_sparse(fit_grid_new_z, fit_pot_new_z, R_g_sparse, tol, z_tol, window, g):
    '''
    Computes the vertical epicyclic frequency nu for a sparse set of guiding radii R_g_sparse.
    Returns a list of calculated (R_g, nu, nu_linear) values to be used for interpolation.
    '''
    sparse_nu_values = []
    for R_g in R_g_sparse:
        mask_derz = (abs(fit_grid_new_z[:, 0] - R_g) < tol)
        xz = np.unique(fit_grid_new_z[mask_derz][:, 1])
        yz = np.mean(fit_pot_new_z[mask_derz].reshape(len(xz), int(fit_grid_new_z[mask_derz].shape[0] / len(xz))), axis=1)

        # Smooth the potential in the z direction
        Phi_gaussz = savgol_filter(yz, window, 2)
        Phi_gaussz = gaussian_filter1d(Phi_gaussz,g,mode='nearest')
        Phi_gaussz = gaussian_filter1d(Phi_gaussz,g,mode='nearest')
        Phi_gaussz = gaussian_filter1d(Phi_gaussz, g, mode='nearest')
        tz, cz, kfitz = interpolate.splrep(xz, Phi_gaussz, s=50, k=3)
        zPhi_b = interpolate.BSpline(tz, cz, kfitz, extrapolate=True)

        xx_z = np.linspace(-1000, 1000, 2000)
        yy_bz = zPhi_b(xx_z)
        der1 = der_diff(yy_bz, xx_z)[0]
        # print('got der in nu_calc_sparse function')
      
        # Fitting a straight line to derivative to estimate linear part
        slope, intercept = np.polyfit(xx_z[500:-501], der1[500:-500], 1)
        # print('line fit for linear nu')
      
        # Calculate nu using the smoothed derivatives
        nu = np.nanmean(1000*np.sqrt(np.diff(der1)/np.diff(xx_z)[1:])[abs(xx_z[1:-1])<z_tol])
        nu_linear = np.nanmean(1000*np.sqrt((np.diff(xx_z*slope+intercept)/np.diff(xx_z)))[(np.argsort(yy_bz)-2)[:100]])
        sparse_nu_values.append((R_g, nu, nu_linear))
    
    return sparse_nu_values
  
def create_nu_interpolation_function(nu_values):
    '''
    Interpolates nu and nu_linear as functions of R_g using sparse values, with separate handling for each.
    '''
    # Separate the valid (R_g, nu) and (R_g, nu_linear) pairs
    R_g_sparse_nu = np.array([item[0] for item in nu_values if not np.isnan(item[1])])
    nu_values_clean = np.array([item[1] for item in nu_values if not np.isnan(item[1])])

    R_g_sparse_nu_linear = np.array([item[0] for item in nu_values if not np.isnan(item[2])])
    nu_linear_values_clean = np.array([item[2] for item in nu_values if not np.isnan(item[2])])

    # Interpolate with smoothing splines or BSpline for smoothness
    t, c, k = interpolate.splrep(R_g_sparse_nu, nu_values_clean, s=1e3, k=3)
    nu_func = interpolate.BSpline(t, c, k, extrapolate=True)

    t_lin, c_lin, k_lin = interpolate.splrep(R_g_sparse_nu_linear, nu_linear_values_clean, s=1e3, k=3)
    nu_linear_func = interpolate.BSpline(t_lin, c_lin, k_lin, extrapolate=True)

    return nu_func, nu_linear_func


def nu(i, fit_grid, fit_pot, R_g_b, sparse_step=200):
    '''
    Overall function to compute nu values efficiently using sparse grid interpolation.
    '''
    print('Reducing fitgrid res for nu calc')
    #     ######(20000,2000) is grid_R[1:] which I removed from the fitgrid.py file and directly inputting the shape here in the reduce_res function
    fit_grid,fit_pot, _,_ = reduce_res(fit_grid,fit_pot,1,10,20000,2000)
    print('Resolution of fit_grid reduced. Trying to get nu')
    
    # Define sparse guiding radii
    R_g_min, R_g_max = max(2e3, R_g_b.min()), min(17500, R_g_b.max())
    print(f'minR={R_g_min},Rmax={R_g_max}')
    R_g_sparse = np.arange(R_g_min, R_g_max, sparse_step)
    if R_g_b.shape[0]<10:
        print(R_g_b.shape[0])
        print('using default R_g min max values for grid')
        R_g_sparse = np.arange(2e3,17500,sparse_step)

    # Calculate sparse nu values
    sparse_nu_values = nu_calc_sparse(fit_grid, fit_pot, R_g_sparse, tol=1, z_tol=250, window=101, g=2)
    sparse_array = np.array(sparse_nu_values)
    print(f'nan values before getting function (Rg,nu,nu_linear- {np.isnan(sparse_array[:,0]).sum(),np.isnan(sparse_array[:,1]).sum(),np.isnan(sparse_array[:,2]).sum(), sparse_array[:,2]}')

    # Create interpolation functions
    nu_func, nu_linear_func = create_nu_interpolation_function( sparse_nu_values)

    # Get interpolated nu values for all stars' R_g
    nu_values = nu_func(R_g_b)
    print(f'nan values after func-{np.isnan(nu_values).sum()}')
    nu_linear_values = nu_linear_func(R_g_b)
    print(f'nan values after func-{np.isnan(nu_linear_values).sum()}')

    print(f'Calculated interpolated nu values for {nu_values.shape} stars at i={i} Myr)')

    return nu_linear_values, nu_values
