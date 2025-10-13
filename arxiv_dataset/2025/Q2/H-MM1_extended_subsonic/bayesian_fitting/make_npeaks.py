""" Generating figures for the upcoming letter """

import numpy as np
import astropy.units as u
from skimage import morphology
from astropy.io import fits


mle_x1_file = 'nested-sampling/H-MM1-mle-x1.fits'
gascontourfile = 'data/sample_sig.fits'
Kfile = 'nested-sampling/H-MM1-Ks.fits'
Kcut = 5 # the heuristical ln(Z1/Z2) cut for model selection
distance = 138.4 # in pc
region_lab = 'H-MM1'
beam_text = 'GBT Beam'

def header_flatten(head):
    """ Flattens 3D header into 2D. Surely it exists somewhere already... """
    flathead = head.copy()
    for key in flathead.keys():
        if key.endswith('3'):
            flathead.pop(key)
    flathead['NAXIS'] = 2
    flathead['WCSAXES'] = 2

    return flathead


def show_filtered_contours(fig, Karr, Kcut=Kcut, skimage_kwargs={},
                           header=header_flatten(fits.getheader(Kfile)),
                           **kwargs):
    """ Overplots contours with small features removed """
    min_size = skimage_kwargs.pop('min_size', 3)
    levels = kwargs.pop('level', [0.5])
    Karr_clean = morphology.remove_small_objects(Karr > Kcut,
                                                 min_size=min_size).astype(int)
    hdu_clean = fits.PrimaryHDU(Karr_clean,
                                header=header_flatten(fits.getheader(Kfile)))
    fig.show_contour(data=hdu_clean, levels=levels, **kwargs)


Ks = fits.getdata(Kfile)

gas_contour_kwargs = dict(levels=[0], linewidths=1,
                          colors='#ef6548', linestyles='--')
gas_contour_kwargs_on_v = dict(levels=[0], linewidths=0.5,
                          colors='0.45', linestyles='--')
kcut_contour_kwargs = dict(linewidths=0.5, colors='white')

hdu_K01 = fits.PrimaryHDU(Ks[0], header=header_flatten(fits.getheader(Kfile)))

hdu_K21 = fits.PrimaryHDU(Ks[1],
                          header=header_flatten(fits.getheader(Kfile)))

hdu_K32 = fits.PrimaryHDU(Ks[2],
                          header=header_flatten(fits.getheader(Kfile)))

# make the ln(K)>Kcut based map of LoS component numbers
npeaks_map = np.full_like(Ks[0], np.nan)
npeaks_map[Ks[0] <= Kcut] = 0
npeaks_map[Ks[0] > Kcut] = 1
Karr_clean = morphology.remove_small_objects(Ks[0] > Kcut,
                                             min_size=3).astype(int)

npeaks_map[(Karr_clean == 0) & np.isfinite(npeaks_map)] = 0

npeaks_map[(Ks[0] > Kcut) & (Ks[1] > Kcut)] = 2
npeaks_map[(Ks[0] > Kcut) & (Ks[1] > Kcut) & (Ks[2] > Kcut)] = 3
#npeaks_map[(Ks[0] > Kcut) & (Ks[1] > Kcut) & (Ks[2] > Kcut) & (Ks[3] > Kcut)] = 4
hdu_npeaks = fits.PrimaryHDU(npeaks_map,
                             header=header_flatten(fits.getheader(mle_x1_file)))
hdu_npeaks.writeto('nested-sampling/npeaks_cut5.fits', overwrite=True)
