#!/usr/bin/env python3

# Place import files below
import os
import warnings

import numpy as np

warnings.filterwarnings(action='ignore', category=RuntimeWarning)


def convert_Lsun_to_abs_mag(L, band='V'):
    """Converts Luminosity in Lsun to absolute magnitude.

    Args:
        L (fl/arr): Luminosity [Lsun]
        band (str, optional): Select from available bands.
            Defaults to 'V'.

    Returns:
        fl/arr: Luminosity of the object(s) in absolute magnitudes.
    """
    m_sun = {
        'U': 5.61,
        'B': 5.44,
        'V': 4.81,
        'K': 3.27,
        'g': 5.23,
        'r': 4.53,
        'i': 4.19,
        'z': 4.01
    }
    return m_sun[band] - 2.5 * np.log10(L)


def convert_Lsun_per_pc_to_mag_arcsec(L_per_pc, band='V'):
    """Convert surface brightness in Lsun/pc^2 to mag arcsec^-2

    Args:
        L_per_pc (fl/arr): Luminosity per pc^2 [Lsun pc^-2]
        band (str, optional): Select from available bands.
            Defaults to 'V'.

    Returns:
        fl/arr: Surface brightness of the object(s) in mag arcsec^-2.
    """
    return convert_Lsun_to_abs_mag(L_per_pc, band) + 21.572


def embed_symbols(pdf_file):
    """Embeds symobls in pdf files.

    Args:
        pdf_file (str): Filepath to the file

    Returns:
        None
    """
    os.system('gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress '
              '-dEmbedAllFonts=true -sOutputFile={} -f {}'.format(
                  pdf_file.replace('.pdf', '_embedded.pdf'), pdf_file))
    return None


def make_cumulative_function(values,
                             min_val=np.nan,
                             max_val=np.nan,
                             bins=None):
    if bins is not None:
        x = np.concatenate(([min_val], np.sort(bins), [max_val]))
        in_bins = np.isin(np.sort(bins), values)
        n = np.concatenate(([0.], np.cumsum(in_bins), [np.sum(in_bins)]))
    else:
        x = np.sort(values)
        x = np.concatenate(([min_val], x, [max_val]))
        n = np.concatenate(([0.], np.arange(len(values)) + 1, [len(values)]))

    rem_nan = ~np.isnan(x)

    return x[rem_nan], n[rem_nan]


def save_figures(fig, location, embed=False):
    """Saves svg and pdf versions of figures.

    Args:
        fig (Matplotlib figure object): The figure to save
        location (str): Filepath to the save file
        embed (bool): If True, embeds the symbols in the pdf file.
            Default: False.

    Returns:
        None
    """
    if '.pdf' in location:
        pdf_file = location
        svg_file = location.replace('.pdf', '.svg')
    else:
        pdf_file = location + '.pdf'
        svg_file = location + '.svg'

    fig.savefig(pdf_file, dpi=600, format='pdf', transparent=False)
    fig.savefig(svg_file, dpi=600, format='svg', transparent=False)

    if embed:
        embed_symbols(pdf_file)

    return None


def survey_cone(survey_area):
    """ Computes the fractional sky area coverage and cosine of the
         corresponding opening angle of a mock conical survey region.

   Args:
      survey_area = Float. Survey sky area coverage in deg^2.

   Returns:
      Tuple in the form (Fractional survey area,Cosine{survey angle}).
   """
    fullSkyArea = 4 * np.pi * (180. / np.pi)**2  # in deg^2
    fracArea = survey_area / fullSkyArea
    cos_surveyAngle = 1. - 2. * fracArea

    return (fracArea, cos_surveyAngle)


def v_sphere(r):
    """Calculates the volume of a sphere.

    Args:
        r (fl/arr): Float or array of floats specifying the radius of
            the spheres.

    Returns:
        fl/arr: Volume of the sphere(s).
    """
    return 4. * np.pi * r**3 / 3.
