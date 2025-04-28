import numpy as np
from astropy.io import fits
from molecular_columns.common_functions import J_nu
import astropy.units as u

file_C18O_Tp = 'data/NGC1333_C18O_Tp.fits'
file_C18O_tau = 'data/NGC1333_C18O_tau_v1.fits'

Tp_C18O, hd_C18O = fits.getdata(file_C18O_Tp, header=True)
hd_C18O['BUNIT'] = ('', 'opacity')
freq_C18O = 329.33055250 * u.GHz #

tau_C18O = -np.log(1.0 - Tp_C18O / 
                   (J_nu(Tex=12.0*u.K, freq=freq_C18O) - 
                    J_nu(Tex=2.73*u.K, freq=freq_C18O)).value)

fits.writeto(file_C18O_tau, tau_C18O, hd_C18O, overwrite=True)
