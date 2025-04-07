"""
Given a FITS file in which the data is equispaced in sine(latitude), remesh the data and save a FITS file where the data is equispaced in latitude

Call this script by passing filenames of FITS files as arguments.
"""

import sys
import numpy as np
import os
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator as RGI

def remesh(fname, out):
	"""
	Handles a single FITS file
	
	Arguments:
		fname: str, path to the FITS file
		out: str, path to save the FITS file containing the remeshed data
	"""
	with fits.open(fname) as f:
		hdu = f[0]
		header = hdu.header
		data = hdu.data
	
	header['CUNIT2'] = "" #sine(latitude) is dimensionless, but the HMI maps erroneously have this field nonempty.
	
	i_sinlat = np.arange(np.shape(data)[0])
	i_lon = np.arange(np.shape(data)[1])
	
	w = WCS(header)
	_, sinlat = w.array_index_to_world_values(i_sinlat, np.zeros_like(i_sinlat))
	lon ,_ = w.array_index_to_world_values(np.zeros_like(i_lon), i_lon)
	
	#See https://stackoverflow.com/questions/51474792/2d-interpolation-with-nan-values-in-python/51500842
	data_no_nan = np.nan_to_num(data)
	lat = np.arcsin(sinlat)*180/np.pi #get latitude in degrees
	interp = RGI((lat, lon), data_no_nan, method='cubic', bounds_error=False)
	interp_where_nan = RGI((lat, lon), np.where(np.isnan(data), 1, 0).astype(float), method='linear', bounds_error=False)
	
	lat_new = np.linspace(lat[0], lat[-1], len(lat))
	new_grid = tuple(np.meshgrid(lat_new, lon, indexing='ij'))
	data_remeshed = interp(new_grid)
	data_remeshed[interp_where_nan(new_grid) > 0.5] = np.nan
	
	#Prepare header for the new FITS file
	header['CRPIX2'] = len(lat_new)/2 + 0.5
	header.comments['CRPIX2'] = "[pixel]"
	header['CRVAL2'] = (lat_new[-1]+lat_new[0])/2
	header.comments['CRVAL2'] = ""
	header['CDELT2'] = lat_new[1]-lat_new[0]
	header.comments['CDELT2'] = "[degree/pixel]"
	header['CUNIT2'] = "deg"
	header.comments['CUNIT2'] = "CUNIT2: degree"
	header.comments['CTYPE2'] = "Carrington Latitude"
	header.add_comment("  Rebinned to have the y-axis equispaced in degrees by Kishore G. (kishore96@gmail.com).")
	
	fits.writeto(out, data_remeshed, header=header)

def gen_out_name(fname):
	"""
	Generate name of output file from the name of the input file. Just inserts the word 'rebinned' after the second dot in fname
	"""
	dirname = os.path.dirname(fname)
	out = os.path.basename(fname)
	
	out = out.split('.')
	out.insert(2, "rebinned")
	out = ".".join(out)
	
	out = os.path.join(dirname, out)
	
	assert out != fname #Just a sanity check
	return out

if __name__ == "__main__":
	for fname in sys.argv[1:]:
		out = gen_out_name(fname)
		if os.path.exists(out):
			print(f"Skipping {fname}")
		else:
			print(f"Remeshing {fname}")
			remesh(fname, out)
