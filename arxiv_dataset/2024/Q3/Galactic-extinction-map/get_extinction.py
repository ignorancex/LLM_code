# This script provides a function to retrieve extinction values based on Galactic coordinates
# and distance. The get_extinction function takes Galactic longitude (glon), Galactic latitude
# (glat), and distance (dpc) as input parameters. Optionally, it can also return the reddening
# curve parameter R(V) if get_rv is set to True.
#
# Example usage:
#   >>> from get_extinction import get_extinction
#   >>> get_extinction(glon=166.4628, glat=-23.6146, dpc=135.8, get_rv=False)
#   >>> get_extinction(glon=166.4628, glat=-23.6146, dpc=135.8, get_rv=True)

import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.interpolate import interp1d

def get_extinction(glon, glat, dpc, get_rv=False):
    extinction_map = 'extinction_map'
    if get_rv:
        extinction_map = 'extinction_map2'
    
    ebv_avg = fits.open(extinction_map+'_ebv.fits')[0].data
    ebv_avg_err = fits.open(extinction_map+'_ebverr.fits')[0].data
    dmn_min = fits.open(extinction_map+'_mindmn.fits')[0].data
    dmn_max = fits.open(extinction_map+'_maxdmn.fits')[0].data
    
    if get_rv:
        rv_avg = fits.open(extinction_map+'_rv.fits')[0].data
    
    coords = SkyCoord(l=glon, b=glat, frame='galactic', unit=(u.deg, u.deg))
    
    pix = hp.ang2pix(2**7, coords.l.deg, coords.b.deg, nest=True, lonlat=True)
    
    dmn = 5.0 * np.log10(dpc) - 5.0
    darr = np.arange(5.0, 12.42, 0.02)
    
    ebv = interp1d(darr, ebv_avg[:, pix], kind='linear')(dmn)
    ebverr = interp1d(darr, ebv_avg_err[:, pix], kind='linear')(dmn)
    
    if get_rv:
        rv = rv_avg[pix]
    
    dpc_min = 10**((dmn_min[pix] + 5.0) / 5.0)
    dpc_max = 10**((dmn_max[pix] + 5.0) / 5.0)
    
    print(f'E(B-V) = {ebv:.3f} +/- {ebverr:.3f}')
    if get_rv:
        print(f'R(V) = {rv:.3f}')
    print(f'Note that useful E(B-V) is found within {int(dpc_min)} - {int(dpc_max)} pc.')

if __name__ == "__main__":
    # Example usage:
    get_extinction(glon=166.4628, glat=-23.6146, dpc=135.8, get_rv=False)
    get_extinction(glon=166.4628, glat=-23.6146, dpc=135.8, get_rv=True)
    pass


