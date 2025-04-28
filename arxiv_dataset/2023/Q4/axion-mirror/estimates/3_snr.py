import os
import sys
import argparse
from tqdm import tqdm

import numpy as np

from config import pc_dict

sys.path.append("..")
import axionmirror.units_constants as uc
from axionmirror.snr import load_snr_list, add_image_to_map

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"


def snr(pc, snr_pop=..., snr_list_samples=..., var_flag=...):
    """Makes signal maps for supernova remnants (SNRs)."""
    
    #===== settings =====
    n_sigma = 4 # size of SNR image added to map
    use_gaussian_intg = False
    Sgfgnu1GHz_threshold = 1e-7 # [Jy]

    #===== add snr to map =====
    sig_temp_map_snr_samples = []

    for snr_list in snr_list_samples:
        
        sig_temp_map_snr = np.zeros_like(pc.ra_grid)
        
        for snr in snr_list:
            snr_anti_ra  = snr.ra + np.pi if snr.ra < np.pi else snr.ra - np.pi
            snr_anti_dec = - snr.dec

            #----- gegenschein -----
            if snr.Sgnu(1000) > Sgfgnu1GHz_threshold:
                #T_submap = snr.Sgnu(nu)*Jy * tmpl * c0**2 / (2 * nu**2 * pixel_area_submap * kb)
                T_submap_prefactor = snr.Sgnu(pc.nu)*uc.Jy * 1 * uc.c0**2 / (2 * pc.nu**2 * 1 * uc.kb)
                # [K] = [MHz^2 g] [cm MHz]^2 [MHz]^-2 [cm^2 MHz^2 g K^-1]^-1
                add_image_to_map(
                    sig_temp_map_snr, pc.ra_s, pc.dec_s, pc.ra_edges, pc.dec_edges,
                    pc.pixel_area_map,
                    source_ra = snr_anti_ra,
                    source_dec = snr_anti_dec,
                    image_sigma = snr.image_sigma, n_sigma=n_sigma,
                    T_submap_prefactor = T_submap_prefactor,
                    use_gaussian_intg=use_gaussian_intg, modify_mode='add'
                )

            #----- Front gegenschein -----
            if snr.Sfgnu(1000) > Sgfgnu1GHz_threshold:
                #T_submap = snr.Sgnu(nu)*Jy * tmpl * c0**2 / (2 * nu**2 * pixel_area_submap * kb)
                T_submap_prefactor = snr.Sfgnu(pc.nu)*uc.Jy * 1 * uc.c0**2 / (2 * pc.nu**2 * 1 * uc.kb)
                # [K] = [MHz^2 g] [cm MHz]^2 [MHz]^-2 [cm^2 MHz^2 g K^-1]^-1
                add_image_to_map(
                    sig_temp_map_snr, pc.ra_s, pc.dec_s, pc.ra_edges, pc.dec_edges,
                    pc.pixel_area_map,
                    source_ra = snr.ra,
                    source_dec = snr.dec,
                    image_sigma = snr.image_sigma_fg, n_sigma=n_sigma,
                    T_submap_prefactor = T_submap_prefactor,
                    use_gaussian_intg=use_gaussian_intg, modify_mode='add'
                )
                
        sig_temp_map_snr_samples.append(sig_temp_map_snr.copy())

    sig_temp_map_snr_samples = np.array(sig_temp_map_snr_samples)

    #===== save =====
    temp_name = f'snr-{snr_pop}-{var_flag}'
    temp_map = sig_temp_map_snr_samples
    os.makedirs(f"{pc.save_dir}/{temp_name}", exist_ok=True)
    np.save(f'{pc.save_dir}/{temp_name}/{temp_name}-{pc.postfix}.npy', temp_map)

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config')
    parser.add_argument('--pop', type=str, required=True, help='{fullinfo, partialinfo, graveyard}')
    parser.add_argument('--var', type=str, default='base', help='variation flag')
    args = parser.parse_args()
    
    pc = pc_dict[args.config]
    n_samples = 300
    
    snr_list_samples = []
    for i_sample in tqdm(range(n_samples)):
        snr_list_samples.append(
            load_snr_list(f"../outputs/snr/{args.pop}_samples_{args.var}/{args.pop}_{i_sample}.json")
        )
    pc.iter_over_func(snr, snr_pop=args.pop, snr_list_samples=snr_list_samples, var_flag=args.var)