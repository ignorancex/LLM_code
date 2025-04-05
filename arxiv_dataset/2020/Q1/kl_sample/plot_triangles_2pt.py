import numpy as np
from astropy.io import fits
import kl_sample.io as io

io.print_info_fits('/users/groups/damongebellini/kl_sample/data/data_real.fits')

io.print_info_fits('/users/groups/damongebellini/kl_sample/data/data_fourier.fits')

'''
d_r=fits.open("/users/groups/damongebellini/kl_sample/data/data_real.fits")
d_f=fits.open("/users/groups/damongebellini/kl_sample/data/data_fourier.fits")

def unwrap_cells(ell :
    

ells=d_f[1].data
cells=d_f[2].data
nells=d_f[2].data
print(ells.shape,cells.shape,nells.shape)
'''
