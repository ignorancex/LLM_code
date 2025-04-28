import os
import sys
import h5py
import argparse

import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import jax.numpy as jnp

from config import pc_dict

sys.path.append("..")
from axionmirror.spectral import prefac
from axionmirror.map_utils import pad_mbl, interp2d_vmap

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"


def gsr(pc, field_model=...):
    """Make signal and background maps from galactic synchrotron radiation (GSR)."""

    #===== settings =====
    remove_GCantiGC = True
    
    #===== source (Haslam) map =====
    nu_haslam = 408 # [MHz]
    beta = -2.5
    # destriped (not desourced) map for background
    haslam_ds_map_hp = hp.read_map('../data/gsr/haslam408_ds_Remazeilles2014.fits')
    haslam_ds_map_hp *= (pc.nu/nu_haslam) ** beta
    haslam_ds_map = hp.pixelfunc.get_interp_val(
        haslam_ds_map_hp, np.rad2deg(pc.l_grid), np.rad2deg(pc.b_grid), lonlat=True
    )
    
    #===== signal =====
    l_grid_zc = jnp.where(pc.l_grid > np.pi, pc.l_grid - 2*np.pi, pc.l_grid) # zero centered
    bl_flat = jnp.stack([pc.b_grid.ravel(), l_grid_zc.ravel()], axis=-1)
    
    with h5py.File(f"../outputs/gsr/Ta_408MHz_field{field_model}.h5") as hf: # Ta: signal temperature for all
        padded_Ta, padded_b, padded_l = pad_mbl(hf['Ta'][:], hf['b_s'][:], hf['l_s'][:])
    
    sig_temp_map = interp2d_vmap(
        jnp.asarray(padded_Ta),
        jnp.asarray(padded_b),
        jnp.asarray(padded_l),
        bl_flat
    ).reshape(pc.radec_shape)
    
    sig_temp_map = np.array(sig_temp_map) * (pc.nu/nu_haslam)**beta * (prefac(pc.nu)/prefac(nu_haslam))
    
    #----- remove GC pixel and anti-GC pixel -----
    if remove_GCantiGC:
        coord_GC = SkyCoord(l=0*u.rad, b=0*u.rad, frame='galactic')
        ra_GC = coord_GC.icrs.ra.rad
        dec_GC = coord_GC.icrs.dec.rad

        i_ra_GC = np.searchsorted(pc.ra_edges, ra_GC) - 1 # v_edges[i-1] < v < v_edges[i]
        i_dec_GC = np.searchsorted(pc.dec_edges, dec_GC) - 1
        if 0 < i_dec_GC and i_dec_GC < pc.n_dec and 0 < i_ra_GC and i_ra_GC < pc.n_ra:
            sig_temp_map[i_dec_GC, i_ra_GC] = 0

        ra_antiGC = ra_GC+np.pi if ra_GC < np.pi else ra_GC-np.pi
        dec_antiGC = - dec_GC

        i_ra_antiGC = np.searchsorted(pc.ra_edges, ra_antiGC) - 1 # v_edges[i-1] < v < v_edges[i]
        i_dec_antiGC = np.searchsorted(pc.dec_edges, dec_antiGC) - 1
        if 0 < i_dec_antiGC and i_dec_antiGC < pc.n_dec and 0 < i_ra_antiGC and i_ra_antiGC < pc.n_ra:
            sig_temp_map[i_dec_antiGC, i_ra_antiGC] = 0

    #===== background =====
    bkg_temp_map = haslam_ds_map + pc.telescope.T_rec # [K/eta_sig]

    #===== save =====
    for temp_name, temp_map in zip([f'gsr{field_model}', 'bkg'], [sig_temp_map, bkg_temp_map]):
        os.makedirs(f"{pc.save_dir}/{temp_name}", exist_ok=True)
        np.save(f'{pc.save_dir}/{temp_name}/{temp_name}-{pc.postfix}.npy', temp_map)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config')
    parser.add_argument('--field_model', type=str, required=True, help='field model')
    args = parser.parse_args()
    
    pc = pc_dict[args.config]
    pc.iter_over_func(gsr, field_model=args.field_model)
