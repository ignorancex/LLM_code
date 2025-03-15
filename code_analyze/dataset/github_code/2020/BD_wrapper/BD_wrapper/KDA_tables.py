import os
import numpy as np
from astropy.table import Table
from astropy import units as u


def kda_info_table(t, glon='', glat='', vlsr='', dvlsr='', kda='',
                   a='', b='', pa='', factor_dvlsr=1., factor_a=1., factor_b=1.,
                   offset_pa=0, unit_dvlsr=u.km/u.s, unit_a=u.deg, unit_b=u.deg,
                   unit_pa=u.deg, add_cols=[]):
    """Create info table for kinematic distance ambiguity solutions.

    Parameters
    ----------
    t : astropy.table.table.Table
        Table containing sources with solved kinematic distance ambiguities.
    glon : str
        Column name of the Galactic longitude values of the sources.
    glat : str
        Column name of the Galactic latitude values of the sources.
    vlsr : str
        Column name of the velocity values of the sources (measured to the local standard of rest).
    dvlsr : str
        Column name of the measured linewidth of the sources.
    kda : str
        Column name of the kinematic distance ambiguity solution of the sources ('N', 'F', or 'T').
    a : str
        Column name of the semi-major axis of the sources (corresponding to half the FWHM extent of the sources).
    b : str
        Column name of the semi-minor axis of the sources (corresponding to half the FWHM extent of the sources).
    pa : str
        Column name of the position angle of the sources (corresponding to half the FWHM extent of the sources).
    factor_dvlsr : type
        Description of parameter `factor_dvlsr`.
    factor_a : type
        Description of parameter `factor_a`.
    factor_b : type
        Description of parameter `factor_b`.
    offset_pa : float [deg]
        Offset of the position angle to the positive Galactic longitude axis. If position angle is measured counterclockwise from Galactic north, `offset_pa=90`.
    unit_dvlsr : type
        Description of parameter `unit_dvlsr`.
    unit_a : type
        Description of parameter `unit_a`.
    unit_b : type
        Description of parameter `unit_b`.
    unit_pa : type
        Description of parameter `unit_pa`.
    add_cols : list
        List of additional names of columns from the table that should be included.

    Returns
    -------
    type
        Description of returned object.

    """
    p_far = t[kda].copy()
    keep = np.where(np.logical_or(p_far == 'N', p_far == 'F'))[0]

    p_far = np.where(p_far == 'N', -0.5, p_far)
    p_far = np.where(p_far == 'F', 0.5, p_far)
    p_far = p_far[keep].astype('float')

    factor_a = factor_a * (unit_a).to(u.deg)
    factor_b = factor_b * (unit_b).to(u.deg)
    factor_dvlsr = factor_dvlsr * (unit_dvlsr).to(u.km / u.s)

    col_glon = t[glon][keep]
    col_glat = t[glat][keep]
    col_vlsr = t[vlsr][keep]
    col_dvlsr = t[dvlsr][keep] * factor_dvlsr
    col_a = factor_a * t[a][keep]
    col_b = factor_a * t[b][keep]

    if pa:
        col_pa = np.deg2rad(t[pa][keep]) + np.deg2rad(offset_pa)
    else:
        col_pa = np.zeros(len(keep))

    col_cos_pa = np.cos(col_pa)
    col_sin_pa = np.sin(col_pa)
    col_aa = (t[a][keep] * factor_a)**2
    col_bb = (t[b][keep] * factor_b)**2

    added_data = []
    for col in add_cols:
        added_data.append(t[col][keep])

    data = [col_glon, col_glat, col_vlsr, col_a, col_b, np.rad2deg(col_pa),
            col_dvlsr, p_far, col_cos_pa, col_sin_pa, col_aa, col_bb] + added_data
    names = ['GLON', 'GLAT', 'VLSR', 'a', 'b', 'pa',
             'd_VLSR', 'p_far', 'cos_pa', 'sin_pa', 'aa', 'bb'] + add_cols

    t_sel = Table(data=data, names=names)

    for col in ['GLON', 'GLAT', 'a', 'b']:
        t_sel[col].unit = 'deg'
        t_sel[col].format = "{0:.4f}"
    for col in ['VLSR', 'd_VLSR']:
        t_sel[col].unit = 'km/s'
        t_sel[col].format = "{0:.4f}"
    t_sel['pa'].format = "{0:.0f}"

    return t_sel


def kda_info_table_ini(reference, weight_cat=1, threshold_spatial=1,
                       threshold_spectral=1, size='fwhm', linewidth='fwhm'):
    str_default = str("[DEFAULT]\n\n")
    str_reference = str(
        "# reference to the table [str]\n"
        "reference = '{}'\n\n".format(reference)
    )
    str_weight_cat = str(
        "# weight for the reliability of the catalogue; should be between 0 and 1 [float]\n"
        "weight_cat = {}\n\n".format(weight_cat)
    )
    str_threshold_spatial = str(
        "# minimum weight for spatial associations; should be between 0 and 1 [float]\n"
        "threshold_spatial = {}\n\n".format(threshold_spatial)
    )
    str_threshold_spectral = str(
        "# minimum weight for spectral associations; should be between 0 and 1 [float]\n"
        "threshold_spectral = {}\n\n".format(threshold_spectral)
    )
    str_size = str(
        "# source size (semi-major and semi-minor axes) given in terms of FWHM ('fwhm') or standard deviation ('std') [str]\n"
        "size = '{}'\n\n".format(size)
    )
    str_linewidth = str(
        "# linewidth measurement given in terms of FWHM ('fwhm') or standard deviation ('std') [str]\n"
        "linewidth = '{}'\n\n".format(linewidth)
    )

    text_ini = (
        str_default +
        str_reference +
        str_weight_cat +
        str_threshold_spatial +
        str_threshold_spectral +
        str_size +
        str_linewidth
    )

    return text_ini
