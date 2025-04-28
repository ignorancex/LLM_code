"""
Functions for Solene's DLA catalog, from
    The Completed SDSS-IV extended Baryon Oscillation
    Spectroscopic Survey: The Damped Lyman-α systems Catalog
    https://arxiv.org/abs/2107.09612

Solene's catalogue:
https://drive.google.com/drive/folders/1UaFHVwSNPpqkxTbcbR8mVRJ5BUR9KHzA
"""

import numpy as np

from astropy.io import fits

def solene_eBOSS_cuts(z_qsos: int, zwarning: int, bal_prob: float) -> np.ndarray:
    """
    DLA_CAT_SDSS_DR16.fits (Chabanier+21)
    *************************************************************************************
    Results of the DLA search using the CNN from Parks+18 on the 263,201 QSO spectra from
    the SDSS-IV quasar catalog from DR16 (Lyke+20) with

    (1) 2 <= Z_QSO <= 6

    (2) Z_WARNING != 
        SKY (0),
        LITTLE_COVERAGE (1),
        UNPLUGGED (7),
        BAD_TARGET (8) or
        NODATA (9).

    (3) BAL_PROB = 0
    """
    # 1) Z_QSO
    ind = (2 <= z_qsos) & (z_qsos <= 6)

    # converting decimal ZWARNING to binary
    zwarning_b = [format(z, "b") for z in zwarning]

    # 2) ZWARNING: filtering based on Solene's condition
    solene_filter = [0, 1, 7, 8, 9]

    ind_z = np.ones(z_qsos.shape, dtype=np.bool_)

    for i,zw in enumerate(zwarning_b):
        for b in solene_filter:
            fiter_yes = bitget_string(zw, b)
            if fiter_yes:
                ind_z[i] = False
                break

    # 3) BAL prob
    ind_bal = (bal_prob == 0)

    ind = ind & ind_z & ind_bal

    return ind

def bitget_string(z: str, bit: int):
    """
    get the bit value at position bit:int, assume it's binary.
    """
    if bit >= len(z):
        return False

    bit_str = z[::-1][bit]

    return bool(int(bit_str))
