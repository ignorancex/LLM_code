"""Class and functions for extra-galactic radio sources"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd


@dataclass
class EGRS:
    """
    Extra-galactic radio source.

    Args:
        name (str): Name of the source.
        l (float): Galactic longitude [rad].
        b (float): Galactic latitude [rad].
        ra (float): Right ascension [rad].
        dec (float): Declination [rad].
        spec (np.ndarray): Spectral data. Must be sorted by frequency. Frequency in [MHz], flux density in [Jy].
        si (float): Spectral index.
        min_extrap_freq (float): Minimum frequency in extrapolation [MHz].
        max_extrap_freq (float): Maximum frequency in extrapolation [MHz].
        catalog (str): Catalog name.
    """
    name: str = None
    l: float = None # [rad] | galactic longitude
    b: float = None # [rad] | galactic latitude
    ra: float = None # [rad]
    dec: float = None # [rad]
    spec: np.ndarray = None # [[freq, flux], [freq, flux], ...] must be sorted
    si: float = None
    min_extrap_freq: ClassVar[float] = 30 # [MHz]
    max_extrap_freq: ClassVar[float] = 1500 # [MHz]
    catalog: str = None
    
    def __post_init__(self):
        self.spec = self.spec[np.argsort(self.spec[:,0])]

    @property
    def l_0center(self):
        return self.l if self.l < np.pi else self.l - 2*np.pi
    
    @property
    def min_data_freq(self):
        return self.spec[0,0]
    @property
    def max_data_freq(self):
        return self.spec[-1,0]
    @property
    def min_data_Snu(self):
        return self.spec[0,1]
    @property
    def max_data_Snu(self):
        return self.spec[-1,1]
    
    @property
    def anti_ra(self):
        return self.ra + np.pi if self.ra < np.pi else self.ra - np.pi
    @property
    def anti_dec(self):
        return - self.dec
    @property
    def anti_l(self):
        return self.l + np.pi if self.l < np.pi else self.l - np.pi
    @property
    def anti_b(self):
        return - self.b
        
    def Snu_no_extrap(self, nu):
        """
        Given frequency nu [MHz] returns flux density Snu [Jy].
        No extrapolation. (If nu is outside of the data range, returns 0.)
        """
        if nu < self.min_data_freq or nu > self.max_data_freq:
            return 0.
        else:
            return np.interp(nu, self.spec[:,0], self.spec[:,1])

    def Snu(self, nu):
        """
        Given frequency nu [MHz] returns flux density Snu [Jy].
        With extrapolation.
        """
        if nu < self.min_data_freq:
            return self.min_data_Snu * (nu/self.min_data_freq) ** self.si
        elif nu > self.max_data_freq:
            return self.max_data_Snu * (nu/self.max_data_freq) ** self.si
        else:
            return self.Snu_no_extrap(nu)
        
    @property
    def lower_extrap_spec(self):
        """Extrapolation points below min_data_freq."""
        if self.min_data_freq < EGRS.min_extrap_freq:
            return None
        else:
            nu = EGRS.min_extrap_freq
            return np.array([[nu, self.Snu(nu)], self.spec[0]])
    @property
    def upper_extrap_spec(self):
        """Extrapolation points above max_data_freq."""
        if self.max_data_freq > EGRS.max_extrap_freq:
            return None
        else:
            nu = EGRS.max_extrap_freq
            return np.array([self.spec[-1], [nu, self.Snu(nu)]])
        
    def plot_spec(self, ax, extrap=True, interp_color='k', extrap_color='C1', alpha=1., markeralpha=1.):
        """Plot spectrum."""
        ax.plot(self.spec[:,0], self.spec[:,1], color=interp_color, alpha=alpha)
        ax.plot(self.spec[:,0], self.spec[:,1], '+', color=interp_color, alpha=markeralpha)
        if extrap:
            lower_extrap_spec = self.lower_extrap_spec
            if lower_extrap_spec is not None:
                ax.plot(lower_extrap_spec[:,0], lower_extrap_spec[:,1], color=extrap_color, alpha=alpha)
            upper_extrap_spec = self.upper_extrap_spec
            if upper_extrap_spec is not None:
                ax.plot(upper_extrap_spec[:,0], upper_extrap_spec[:,1], color=extrap_color, alpha=alpha)
        
        
#===== utils =====

def egrs_list_keuhr(include_cygA=True):
    
    with fits.open("../data/egrs/keuhr_catalog.fits") as hdul:
        data = hdul[1].data
        
    egrs_list = []
    i_d = 0

    while i_d < len(data):
        d = data[i_d]
        name = d['NAME']
        i_d_end = i_d
        while i_d_end < len(data) and data[i_d_end]['NAME'] == name:
            i_d_end += 1

        egrs = EGRS(
            name = d['NAME'],
            l = np.deg2rad(d['LII']),
            b = np.deg2rad(d['BII']),
            ra = np.deg2rad(d['RA']),
            dec = np.deg2rad(d['DEC']),
            spec = np.stack([data[i_d:i_d_end]['FREQUENCY'],
                             data[i_d:i_d_end]['FLUX_RADIO']/1000.], axis=-1),
            si = d['SPECTRAL_INDEX'],
            catalog = 'keuhr',
        )
        egrs_list.append(egrs)

        i_d = i_d_end

    if include_cygA:
        l = 76.18988064623 * u.deg
        b =  5.75538801499 * u.deg
        c = SkyCoord(l=l, b=b, frame="galactic")
        cygA = EGRS(
            name = 'Cyg A',
            l = l.to(u.rad).value,
            b = b.to(u.rad).value,
            ra = c.icrs.ra.rad,
            dec = c.icrs.dec.rad,
            spec = np.loadtxt('../data/egrs/cygA_spec.txt'),
            si = None,
            catalog = 'Cyg A',
        )
        egrs_list.append(cygA)
        
    return egrs_list


def egrs_list_cora():

    df = pd.read_csv('../data/egrs/cora/combinedps_egrs.dat', delim_whitespace=True)

    egrs_list = []
    for _, row in df.iterrows():
        name = row['NAME']
        if name.startswith('?'):
            continue
        ra = np.deg2rad(row['RA'])
        dec = np.deg2rad(row['DEC'])
        coord = SkyCoord(ra=ra, dec=dec, unit='rad', frame='icrs')
        egrs = EGRS(
            name = name,
            l = coord.galactic.l.rad,
            b = coord.galactic.b.rad,
            ra = ra,
            dec = dec,
            spec = np.array([[74., row['S74']], [600., row['S600']], [1400., row['S1400']]]),
            catalog = 'cora',
        )
        egrs_list.append(egrs)

    return egrs_list


def concat_with_exclusion(egrs_list_1, egrs_list_2, exclude_radius=np.deg2rad(1.)):
    """Concatenate egrs_list_1 and egrs_list_2, excluding sources in egrs_list_2 within exclude_radius [rad] of existing sources."""
    egrs_list = [egrs for egrs in egrs_list_1]

    l_s = np.array([egrs.l for egrs in egrs_list_1])
    b_s = np.array([egrs.b for egrs in egrs_list_1])
    for egrs in egrs_list_2:
        l = egrs.l
        b = egrs.b
        if np.all(np.sqrt((l_s - l)**2 + (b_s - b)**2) > exclude_radius):
            egrs_list.append(egrs)

    return egrs_list


def egrs_list_all():
    egrs_list = concat_with_exclusion(egrs_list_keuhr(), egrs_list_cora())
    return egrs_list