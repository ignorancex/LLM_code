import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Rectangle

from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval,PercentileInterval,MinMaxInterval
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astroquery.ned import Ned
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo

ll      = 2000                      # Physical size of the image
itvl    = 'PercentileInterval(99)'  # Colorbar interval
cmap    = 'gist_heat_r'             # Color map
num_cont= 10                        # Number of contours

fname   = otpt + '.fits'
exportfits(imagename=otpt+'.image', fitsimage=fname, overwrite=True)

hdu     = fits.open(fname)[0]
wcs     = WCS(hdu.header, naxis=2)
newdata = np.squeeze(hdu.data)

lvls    = np.array([-1.0])
for i in np.arange(num_cont):
    lvls = np.append(lvls, np.sqrt(2**i))
lvls    = 3 * img_rms * lvls

try:
    # Creating cutouts
    s1      = (ll/cosmo.kpc_proper_per_arcmin(z).value)
    size    = u.Quantity((s1,s1), u.arcmin)
    x0      = hdu.header['CRPIX1']
    y0      = hdu.header['CRPIX2']
    ra      = float(Ned.query_object(cluster)['RA'])
    dec     = float(Ned.query_object(cluster)['DEC'])
    ra0     = hdu.header['CRVAL1']
    dec0    = hdu.header['CRVAL2']
    del_a   = hdu.header['CDELT1']
    del_d   = hdu.header['CDELT2']
    x       = int(np.round(x0 + (ra-ra0)*np.cos(np.deg2rad(np.mean((dec,dec0))))/del_a))
    y       = int(np.round(y0 + (dec-dec0)/del_d))
    pos     = (x,y)
    cutout  = Cutout2D(newdata, pos, size, wcs=wcs)
    newx0   = cutout.wcs.wcs.crpix[0]
    newy0   = cutout.wcs.wcs.crpix[1]
    newx    = len(cutout.data)
    newy    = len(cutout.data)

    # Setting colorbar levels
    interval= eval(itvl)
    vmin, vmax = interval.get_limits(cutout.data)

    # Plotting
    im = plt.imshow(cutout.data, vmin=vmin, vmax=vmax, origin='lower',cmap=cmap)
    plt.contour(cutout.data, levels=lvls, colors='blue', alpha=0.5)
    plt.grid(color='white', ls='dashed', alpha=0.5)
    freq = hdu.header['CRVAL3']
    plt.title('{:s} ({:.0f} MHz) [rms={:.2f} mJy/beam]'.format(cluster, freq/1.e6, img_rms))
    plt.colorbar(im)

    plt.pause(10)
    plt.close("all")

except Exception as e:
    print('ERROR for {}:\n{}\n'.format(cluster,e))
####--------------------------------XXXX------------------------------------####
