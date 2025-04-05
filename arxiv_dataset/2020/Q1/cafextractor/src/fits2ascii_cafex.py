from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
import glob
from astropy.io import ascii
import numpy as np
import os
import argparse

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to data folder")
    args = parser.parse_args()
    return args


args = cli()
path = args.path


for file in glob.glob(path+'/*red.fits'): 

	a = fits.open(file)
	try:
		a["FNORM"].data[0,250:-200]
	except:
		print "ERROR in file: "+file+" --> not all extensions fouond (probably FNORM)"
		continue

	w,f,ef,fnorm,order = [],[],[],[],[]
	norders, npix = np.shape(a["WAVELENGTH"].data)
	xmin,xmax = 250,-200 #100, -100 # 

	for oo in range(norders)[::-1]:
		w.append(a["WAVELENGTH"].data[oo,xmin:xmax])
		f.append(a["FLUX"].data[oo,xmin:xmax])
		ef.append(a["EFLUX"].data[oo,xmin:xmax])
		fnorm.append(a["FNORM"].data[oo,xmin:xmax])
		order.append(oo+np.zeros(len(a["WAVELENGTH"].data[oo,xmin:xmax])))
		
		
	w,f,ef,fnorm,order = np.array(w),np.array(f),np.array(ef),np.array(fnorm),np.array(order)
	
	w = [y for x in w for y in x]
	f = [y for x in f for y in x]
	ef = [y for x in ef for y in x]
	fnorm = [y for x in fnorm for y in x]
	order = [y for x in order for y in x]

	data = Table([w, f, ef, fnorm, order], names=['# Wave', 'Flux', 'eFlux', 'Fnorm', 'Order'])
	ascii.write(data, os.path.splitext(file)[0]+'.txt', format='tab',overwrite=True)


	data = Table([a["WMERGE1D"].data, a["FMERGE1D"].data], names=['# Wave', 'FMERGE1D'])
	ascii.write(data, os.path.splitext(file)[0]+'_merge1d.txt', format='tab',overwrite=True)
