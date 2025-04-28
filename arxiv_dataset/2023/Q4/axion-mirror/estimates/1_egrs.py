import os
import sys
import argparse

import numpy as np
from scipy.signal import convolve2d

from config import pc_dict

sys.path.append("..")
import axionmirror.units_constants as uc
from axionmirror.nfw import rho_integral
from axionmirror.spectral import prefac
from axionmirror.egrs import egrs_list_all

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"


def egrs(pc, verbose=False):
    """Makes signal maps for extragalactic radio sources (EGRSs)."""
    
    #===== settings =====
    smooth = False
    include_forwardschein = False
    
    #===== extragalactic radio source map =====
    egrs_list = egrs_list_all()
    egrs_map = np.zeros_like(pc.ra_grid)

    n_included_keuhr = 0
    n_included_cygA = 0
    n_included_cora = 0
    n_total_keuhr = np.sum([egrs.catalog == 'keuhr' for egrs in egrs_list])
    n_total_cygA = np.sum([egrs.catalog == 'Cyg A' for egrs in egrs_list])
    n_total_cora = np.sum([egrs.catalog == 'cora' for egrs in egrs_list])

    for egrs in egrs_list:
        
        if egrs.anti_dec < pc.dec_edges[0] or egrs.anti_dec > pc.dec_edges[-1]:
            continue

        if egrs.catalog == 'keuhr':
            n_included_keuhr += 1
        elif egrs.catalog == 'Cyg A':
            n_included_cygA += 1
        elif egrs.catalog == 'cora':
            n_included_cora += 1

        i_dec = np.searchsorted(pc.dec_edges, egrs.anti_dec) - 1
        i_ra  = np.searchsorted(pc.ra_edges, egrs.anti_ra) - 1
        delta_dec = np.diff(pc.dec_edges)[i_dec]
        delta_ra  = np.diff(pc.ra_edges)[i_ra]
        
        pixel_area = delta_dec * delta_ra * np.cos(egrs.anti_dec) # [rad^2]
        I = egrs.Snu_no_extrap(pc.nu)/pixel_area # [Jy sr^-1]
        # I = 2 * nu^2 kb T / c0^2
        # [MHz^2 g sr^-1] = [MHz^2] [cm^2 MHz^2 g] [cm^-2 MHz^-2]
        T = I * uc.Jy / (2 * pc.nu**2 * uc.kb / uc.c0**2)
        
        egrs_map[i_dec, i_ra] += T

    if verbose:
        print(f'keuhr: {n_included_keuhr}/{n_total_keuhr}', end=' ')
        print(f'cygA: {n_included_cygA}/{n_total_cygA}', end=' ')
        print(f'cora: {n_included_cora}/{n_total_cora}')
    
    #===== rho_DM =====
    rho_integral_map = rho_integral(pc.lb_flat).reshape(pc.radec_shape)
    gegen_map = prefac(pc.nu) * egrs_map * rho_integral_map
    total_map = gegen_map
    if include_forwardschein:
        forward_map = prefac(pc.nu) * egrs_map * rho_integral_map
        total_map += forward_map
    
    #===== smooth for numerical stability =====
    if smooth:
        sigma = 1 # [pixel]
        n_sigma = 5
        n_pix_side = 2 * n_sigma * sigma - 1
        i_c = (n_pix_side - 1) / 2
        kernel = np.zeros((n_pix_side, n_pix_side))
        
        for i in range(n_pix_side):
            for j in range(n_pix_side):
                kernel[i, j] = np.exp( - ((i-i_c)**2 + (j-i_c)**2) / (2*sigma**2) )
        kernel /= np.sum(kernel)
                
        total_map = convolve2d(total_map, kernel, mode='same')
        
    #===== save =====
    temp_name = 'egrs'
    temp_map = total_map
    os.makedirs(f"{pc.save_dir}/{temp_name}", exist_ok=True)
    np.save(f'{pc.save_dir}/{temp_name}/{temp_name}-{pc.postfix}.npy', temp_map)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    
    pc = pc_dict[args.config]
    pc.iter_over_func(egrs, verbose=args.verbose)