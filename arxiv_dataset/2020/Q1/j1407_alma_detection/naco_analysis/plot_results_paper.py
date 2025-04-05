# -*- coding: utf-8 -*-
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

__author__ = "Alexander Bohn"

"""
Plotting the results of the reduction
"""

import numpy as np
import os
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from pynpoint.util.image import crop_image

# Input directory
path = "/Volumes/BOHN/work/data/targets/J1407/2019-03-01/NACO/L_prime/0_2/Results/"
path = "."
# Saving directory
path_save = "/Users/alex/surfdrive/PhD/Research/Papers/2019/A&A/J1407/data/"
path_save = "./"
# Data parameters
target = "J1407"
filt = "L_prime"

# Instrument
instrument = "NACO"
pixscale = 0.02719

# image parameters
im_size = 101
flux_size = 11


im = fits.open(os.path.join(path,"J1407_L_prime_res_median_norm_False_pca_10.fits"))[0].data
flux = fits.open(os.path.join(path,"J1407_L_prime_derotated_median.fits"))[0].data

# crop images to imsize
im_crop = crop_image(im[6,], None, im_size)
flux_crop = crop_image(flux, None, flux_size)

im_crop_shape = im_crop.shape

# Set masked pixels to minimum value
im_crop[im_crop==0.] = np.min(im_crop)
im_crop.reshape(im_crop_shape)

############
# Plot
############

plt.set_cmap("viridis")
# plt.set_cmap("inferno")
# plt.set_cmap("magma") # Alternative color scheme

# Figure B_H median
f_median, ax_median = plt.subplots()
ax1_zoom = zoomed_inset_axes(ax_median,
                             1.,
                             loc="lower left")

ax_median.spines['right'].set_color('none')
ax_median.spines['top'].set_color('none')
ax_median.spines['left'].set_color('none')
ax_median.spines['bottom'].set_color('none')

ax_median.imshow(im_crop, origin="lower",
                 extent=[-im_crop.shape[1] / 2., im_crop.shape[1] / 2., -im_crop.shape[0] / 2., im_crop.shape[0] / 2.])
ax1_zoom.imshow(flux_crop, origin="lower")

# Make x axis
ax_median.set_xticks(np.arange(-1, 1.5, 0.5) / pixscale)
ax_median.xaxis.set_major_locator(MultipleLocator(0.5 / pixscale))
ax_median.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_median.xaxis.set_minor_locator(MultipleLocator(0.1 / pixscale))
ax_median.set_xticklabels(["%.1f" % i for i in np.arange(-1.5, 2., 0.5)[::-1]])
ax_median.set_xlabel(r"$\Delta$RA [arcsec]")

# Make y axis
ax_median.set_yticks(np.arange(-1, 1.5, .5) / pixscale)
ax_median.yaxis.set_major_locator(MultipleLocator(.5 / pixscale))
ax_median.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_median.yaxis.set_minor_locator(MultipleLocator(0.1 / pixscale))
ax_median.set_yticklabels(["%.1f" % i for i in np.arange(-1.5, 1.5, .5)])
ax_median.set_ylabel(r"$\Delta$Dec [arcsec]")

# tick parameters
ax_median.tick_params(axis="both",
                      reset=False,
                      which="major",
                      direction="in",
                      length=6.,
                      width=2.,
                      colors="white",
                      labelcolor="black",
                      top=True,
                      bottom=True,
                      right=True,
                      left=True,
                      labelbottom=True,
                      labeltop=True,
                      labelright=True,
                      labelleft=True)

# tick parameters
ax_median.tick_params(axis="both",
                      reset=False,
                      which="minor",
                      direction="in",
                      length=3.,
                      width=1,
                      colors="white",
                      labelcolor="black",
                      top=True,
                      bottom=True,
                      right=True,
                      left=True,
                      labelbottom=True,
                      labeltop=True,
                      labelright=True,
                      labelleft=True)

# add flux PSF
ax1_zoom.tick_params(axis="both",
                     reset=False,
                     which="both",
                     direction="in",
                     length=3.,
                     width=1,
                     colors="white",
                     labelcolor="black",
                     top=False,
                     bottom=False,
                     right=False,
                     left=False,
                     labelbottom=False,
                     labeltop=False,
                     labelright=False,
                     labelleft=False)

ax1_zoom.spines['bottom'].set_color('white')
ax1_zoom.spines['top'].set_color('white')
ax1_zoom.spines['right'].set_color('white')
ax1_zoom.spines['left'].set_color('white')

leg = ax_median.legend(loc=0,fontsize=12,frameon=False)
# plt.show()
f_median.savefig(os.path.join(path_save, "J1407_NACO_result.pdf"), bbox_inches='tight', transparent=True, pad_inches=0)

hdu = fits.PrimaryHDU(im_crop)
hdul = fits.HDUList([hdu])
hdul.writeto('J1407_NACO_result.fits')
plt.close(f_median)
