#imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

from functions.utils import der

def kappa(i, C_cyl, V_cyl, RPhi):
    '''
    Takes in C_cyl, V_cyl returned by mid_res_info and RPhi returned by potential_der 
    and calculates kappa values for stars . It also returns R_g (guiding radius), 
    L_z (Angular momentum in z), L_z_galaxy (R_g^2*omega(@R_g)) and 
    mask - which can be applied to M, ID ,C_cyl etc to get the same dimensions
    as R_g and kappa
    '''
    extra_dis = 5000
    R_max = C_cyl[:, 1].max() + extra_dis
    R_min = C_cyl[:, 1].min() - extra_dis
    R_max = min(R_max, 17.5e3)
    R_min = max(R_min, 2)
    N_R = round((R_max - R_min) / 0.5)
    xx_c1 = np.linspace(R_min, R_max, N_R)
    yy_c1 = RPhi(xx_c1)
    print('Resolution in R for R_g calc (pc): ' + str((R_max - R_min) / N_R))
    print(f'N_R={N_R}')

    # Calculate L_z  angular momentum in z direction= R*v_phi/1000 (kpc km/s)
  
    L_z = C_cyl[:, 1] * V_cyl[:, 0] / 1000 
  
    # calculate L_z_grid for grid point, fit a function to it
    #to get R_g=f(L_z), put stars' L_z in it and get their guiding radius
  
    omega = np.sqrt(der(yy_c1, xx_c1)[0] / xx_c1[2:-2])  
    #units of omega km/(s*pc)
    print(f'got omega - shape {omega.shape}')
  
    L_z_grid = ((xx_c1[2:-2] ** 2) * omega/ 1000)
    t,c,kb = interpolate.splrep(L_z_grid,xx_c1[2:-2],s=2,k=3)
    Rg_Lz = interpolate.BSpline(t,c,kb,extrapolate=True)
    R_g = Rg_Lz(L_z)
    mask = (R_g>2e3)&(R_g<16000)
    R_g = R_g[mask]
    print(f'got R_g of {R_g.shape} stars')

    ##get kappa valueas for each grid point, fit a function to it
    ##to get kappa as a function of R_g. then put stars' R_g values
    #in to get their kappas
  
    K1 = np.diff((omega * 1000) ** 2) / np.diff(xx_c1[2:-2])
    kappa_grid = np.sqrt(xx_c1[2:-3] * K1 + 4 * (omega[1:] * 1000) ** 2)
    t,c,kb = interpolate.splrep(xx_c1[502:-3],kappa_grid[500:],s=3,k=3)
    kappa_Rg = interpolate.BSpline(t,c,kb,extrapolate=True)
    kappa_stars = kappa_Rg(R_g) ## kappa in units km/(s*kpc)
    print(f'got kappa of {kappa_stars.shape} stars')

    ##getting L_z_galaxy = R_g^2*omega(Rg)/1000 in units kpc km/s
    #same procedure with fitting function to the grid
  
    L_z_galaxy_grid = xx_c1[2:-2]**2*omega/1000
    t,c,kb = interpolate.splrep(xx_c1[502:-2],L_z_galaxy_grid[500:],s=3,k=3)
    Lzgal_Rg = interpolate.BSpline(t,c,kb,extrapolate=True)
    L_z_galaxy = Lzgal_Rg(C_cyl[:,1]) #in units kpc km/s
    print('got L_z_gal')
    print(f'Outputting kappa for i={i} i.e {i} Myr')
  
    return kappa_stars, R_g, L_z[mask], L_z_galaxy[mask], mask
