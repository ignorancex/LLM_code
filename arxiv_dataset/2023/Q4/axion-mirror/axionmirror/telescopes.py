"""Class for telescopes"""

import sys
from dataclasses import dataclass

import numpy as np

sys.path.append("..")
import axionmirror.units_constants as uc
from axionmirror.spectral import dnu

data_dir = "../data"


@dataclass
class Telescope:
    """
    A dataclass for drift scan dish telescopes.

    Args:
        name (str)
        nu_min (float): Min frequency of telescope [MHz].
        nu_max (float): Max frequency of telescope [MHz].
        dec (float): Telescope declination.
        pointing (bool): Whether or not the telescope is pointing.
        survey_ra_max (float): Max right ascension of survey [rad].
        survey_ra_min (float): Min right ascension of survey [rad].
        fixed_survey_za_max (float): Fixed max zenith angle of survey in declination [rad]. If None, use half primary beam (dec) size.
        size_ra (float): Array size in right ascension [cm].
        size_dec (float): Array size in declination [cm].
        eta_f_ra (float): Filling factor in right ascension.
        eta_f_dec (float): Filling factor in declination.
        eta_a_ra (float): Aperture efficiency in right ascension.
        eta_a_dec (float): Aperture efficiency in declination.
        primary_beam_baseline_ra (float): Primary beam baseline in right ascension [cm].
        primary_beam_baseline_dec (float): Primary beam baseline in declination [cm].
        eta_sig (float): Signal chain efficiency.
        T_rec_raw (float): Raw receiver temperature [K].
        t_obs_days (float): Observation time [day].
        n_pol (int): Number of polarizations observed.

    Note:
        Default length measurements are in [cm], angles are in [rad], temperature is in [K].
    """

    name: str
    nu_min: float
    nu_max: float
    dec: float
    pointing: bool
    survey_ra_max: float
    survey_ra_min: float
    fixed_survey_za_max: float
    size_ra: float
    size_dec: float
    eta_f_ra: float
    eta_f_dec: float
    eta_a_ra: float
    eta_a_dec: float
    primary_beam_baseline_ra: float
    primary_beam_baseline_dec: float
    eta_sig: float
    T_rec_raw: float
    t_obs_days: float
    n_pol: int

        
    def ra_pixel_size(self, nu, ra):
        """Pixel size in right ascension [rad]."""
        return (uc.c0 / nu) / (self.eta_f_ra * self.eta_a_ra * self.size_ra)
    
    def dec_pixel_size(self, nu, dec):
        """Pixel size in declination [rad]."""
        denom = self.eta_f_dec * self.eta_a_dec * self.size_dec * np.cos(dec)
        if not self.pointing:
            denom *= np.cos(dec - self.dec)
        return (uc.c0 / nu) / denom
    
    def primary_beam_size_ra(self, nu):
        """Primary beam size in right ascension direction (but not in right ascension coordinates) [rad]."""
        return (uc.c0 / nu) / (self.eta_a_ra * self.primary_beam_baseline_ra)
    
    def primary_beam_size_dec(self, nu):
        """Primary beam size in declination [rad]."""
        return (uc.c0 / nu) / (self.eta_a_dec * self.primary_beam_baseline_dec)
    
    def survey_za_max(self, nu):
        """Maximum survey zenith angle [rad]."""
        if self.fixed_survey_za_max is not None:
            return self.fixed_survey_za_max
        else:
            if self.pointing:
                raise NotImplementedError
            return self.primary_beam_size_dec(nu) / 2
        
    def survey_dec_max(self, nu):
        """Maximum survey declination [rad]."""
        return np.minimum(self.dec + self.survey_za_max(nu), np.pi/2)
    
    def double_pass_dec(self, nu):
        """Declination beyond which points on the sky enters FOV twice per day [rad]."""
        survey_dec_max = self.dec + self.survey_za_max(nu)
        if survey_dec_max > np.pi/2:
            return np.pi - survey_dec_max
        else:
            return None
        
    def survey_dec_min(self, nu):
        """Minimum survey declination [rad]."""
        survey_dec_min = self.dec - self.survey_za_max(nu)
        if survey_dec_min < -np.pi/2:
            raise NotImplementedError
        return np.maximum(survey_dec_min, -np.pi/2)
    
    def extent(self, nu):
        """Extent of survey in [deg]."""
        return tuple(np.rad2deg(
            [self.survey_ra_max, self.survey_ra_min, self.survey_dec_min(nu), self.survey_dec_max(nu)]
        ))
    
    @property
    def T_rec(self):
        """Receiver temperature [K]."""
        return self.T_rec_raw / self.eta_sig
    
    def t_obs(self, nu, dec):
        """Total observation time [s]. dec can take any shape."""
        t_per_day = 86400 # [s]
        if self.pointing:
            t_per_day *= self.primary_beam_size_dec(nu) / (self.survey_dec_max(nu) - self.survey_dec_min(nu))
        t_obs_per_day = t_per_day * self.primary_beam_size_ra(nu) / (2*np.pi * np.cos(dec))
        if self.double_pass_dec(nu) is not None:
            t_obs_per_day *= ((dec > self.double_pass_dec(nu)) + 1)
        
        return self.t_obs_days * t_obs_per_day
    
    def Aeff_zenith(self, nu):
        """Effective area at zenith [cm^2]."""
        return self.eta_a_ra * self.eta_a_dec * self.eta_f_ra * self.eta_f_dec * self.size_ra * self.size_dec
    
    def Aeff(self, nu, dec):
        """Effective area [cm^2]."""
        return self.Aeff_zenith * np.cos(dec - self.dec)
    
    def survey_area(self, nu):
        """Survey area [sr]."""
        return (self.survey_ra_max - self.survey_ra_min) * (np.sin(self.survey_dec_max(nu)) - np.sin(self.survey_dec_min(nu)))
    
    def instantaneous_fov(self, nu):
        """Instantaneous field of view [sr]."""
        if self.pointing:
            return self.primary_beam_size_ra(nu) * self.primary_beam_size_dec(nu)
        else:
            return self.primary_beam_size_ra(nu) * 2 * self.survey_za_max(nu)
        
    def eff_etendue(self, nu):
        """Effective etendue [cm^2 sr]."""
        return self.Aeff_zenith(nu) * self.instantaneous_fov(nu)
    
    def merit(self, nu):
        """Figure of merit [arbitrary]. Ignores frequency scaling for now."""
        return np.sqrt(self.n_pol * self.eff_etendue(nu))
    
    def sens_estimate(self, nu_s):
        """Estimate of g_agg sensitivity [GeV^-1] at given frequencies [MHz]."""
        T_sig = 1.5e-5 # [K] at 408 MHz
        si_sig = -2.5 # S ~ nu^2 T Omega ~ nu^2 T lambda^2 ~ T. ??
        T_bkg = 35 # [K] at 408 MHz, mean of Haslam map
        si_bkg = -2.5
        nu_haslam = 408 # [MHz]

        gagg_s = np.zeros_like(nu_s)

        for i_nu, nu in enumerate(nu_s):
            pixel_size = (uc.c0 / nu) ** 2 / self.Aeff_zenith(nu) # [rad^2]
            n_pixel = self.survey_area(nu) / pixel_size
            t_obs = self.t_obs(nu, self.dec)
            snratio_per_pixel = T_sig * (nu/nu_haslam)**si_sig / (T_bkg * (nu/nu_haslam)**si_bkg + self.T_rec) * np.sqrt(self.n_pol * dnu(nu) * 1e6 * t_obs)
            snratio = np.sqrt(n_pixel) * snratio_per_pixel
            gagg_s[i_nu] = (uc.gagg_CAST/uc.invGeV) / np.sqrt(snratio)

        return gagg_s


@dataclass
class AntennaArray (Telescope):
    """
    A dataclass for drift scan antenna arrays.

    Args:
        fixed_primary_beam_size_ra (float): Fixed primary beam size in right ascension [rad]. If None, use primary beam baseline.
        fixed_primary_beam_size_dec (float): Fixed primary beam size in declination [rad]. If None, use primary beam baseline.
        n_element_ra (int): Number of elements in right ascension.
        n_element_dec (int): Number of elements in declination.
    """

    fixed_primary_beam_size_ra: float
    fixed_primary_beam_size_dec: float
    n_element_ra: int
    n_element_dec: int

    def ra_pixel_size(self, nu, ra):
        """Pixel size in right ascension [rad]."""
        return self.fixed_primary_beam_size_ra / self.n_element_ra
    
    def dec_pixel_size(self, nu, dec):
        """Pixel size in declination [rad]."""
        return self.fixed_primary_beam_size_dec / (self.n_element_dec * np.cos(dec - self.dec) * np.cos(dec))
    
    def primary_beam_size_ra(self, nu):
        """Primary beam size in right ascension direction (but not in right ascension coordinates) [rad]."""
        return self.fixed_primary_beam_size_ra
    
    def primary_beam_size_dec(self, nu):
        """Primary beam size in declination [rad]."""
        return self.fixed_primary_beam_size_dec
    
    def survey_za_max(self, nu):
        """Maximum survey zenith angle [rad]."""
        return self.primary_beam_size_dec(nu) / 2
    
    def Aeff_zenith(self, nu):
        """Effective area at zenith [cm^2]."""
        lmd = uc.c0 / nu # [cm]
        baseline_ra = lmd / self.fixed_primary_beam_size_ra
        baseline_dec = lmd / self.fixed_primary_beam_size_dec
        return baseline_ra * baseline_dec * self.n_element_ra * self.n_element_dec
    

#===== telescope instances =====

CHIME = Telescope(
    name = 'CHIME',
    nu_min = 400, nu_max = 800,
    dec = np.deg2rad(49.3),
    pointing = False,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = np.deg2rad(60.),
    size_ra = 86 * 100, size_dec = 78 * 100,
    eta_f_ra = 80/86, eta_f_dec = 1.,
    eta_a_ra = 0.5, eta_a_dec = 1.,
    primary_beam_baseline_ra = 20 * 100, primary_beam_baseline_dec = None,
    eta_sig = 1.,
    T_rec_raw = 40.,
    t_obs_days = 5 * 365.25,
    n_pol = 2,
)

HERA = Telescope(
    name = 'HERA',
    nu_min = 50, nu_max = 250,
    dec = np.deg2rad(-30.7),
    pointing = False,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = None, # use primary beam size (dec) / 2
    size_ra = 243 * 100, size_dec = 243 * 100,
    eta_f_ra = np.sqrt(0.834), eta_f_dec = np.sqrt(0.834),
    eta_a_ra = np.sqrt(0.6), eta_a_dec = np.sqrt(0.6),
    primary_beam_baseline_ra = 14 * 100, primary_beam_baseline_dec = 14 * 100,
    eta_sig = 1.,
    T_rec_raw = 100.,
    t_obs_days = 5 * 365.25,
    n_pol = 2,
)

CHORD = Telescope(
    name = 'CHORD',
    nu_min = 300, nu_max = 1500,
    dec = np.deg2rad(49.3),
    pointing = True,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = np.deg2rad(30.),
    size_ra = 22 * 7 * 100, size_dec = 23 * 9 * 100,
    eta_f_ra = np.sqrt(0.45), eta_f_dec = np.sqrt(0.45),
    eta_a_ra = np.sqrt(0.5), eta_a_dec = np.sqrt(0.5),
    primary_beam_baseline_ra = 6 * 100, primary_beam_baseline_dec = 6 * 100,
    eta_sig = 1.,
    T_rec_raw = 30.,
    t_obs_days = 5 * 365.25,
    n_pol = 2,
)

HIRAX256 = Telescope(
    name = 'HIRAX256',
    nu_min = 400, nu_max = 800,
    dec = np.deg2rad(-30.7),
    pointing = True,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = np.deg2rad(30.),
    size_ra = 16 * 7 * 100, size_dec = 16 * 9 * 100,
    eta_f_ra = np.sqrt(0.45), eta_f_dec = np.sqrt(0.45),
    eta_a_ra = np.sqrt(0.5), eta_a_dec = np.sqrt(0.5),
    primary_beam_baseline_ra = 6 * 100, primary_beam_baseline_dec = 6 * 100,
    eta_sig = 1.,
    T_rec_raw = 50.,
    t_obs_days = 5 * 365.25,
    n_pol = 2,
)

HIRAX1024 = Telescope(
    name = 'HIRAX1024',
    nu_min = 400, nu_max = 800,
    dec = np.deg2rad(-30.7),
    pointing = True,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = np.deg2rad(30.),
    size_ra = 32 * 7 * 100, size_dec = 32 * 9 * 100,
    eta_f_ra = np.sqrt(0.45), eta_f_dec = np.sqrt(0.45),
    eta_a_ra = np.sqrt(0.5), eta_a_dec = np.sqrt(0.5),
    primary_beam_baseline_ra = 6 * 100, primary_beam_baseline_dec = 6 * 100,
    eta_sig = 1.,
    T_rec_raw = 50.,
    t_obs_days = 5 * 365.25,
    n_pol = 2,
)

BURSTT256 = AntennaArray(
    name = 'BURSTT256',
    nu_min = 300, nu_max = 800,
    dec = np.deg2rad(23.7),
    pointing = False,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,

    fixed_survey_za_max = None,
    size_ra = None, size_dec = None,
    eta_f_ra = None, eta_f_dec = None,
    eta_a_ra = None, eta_a_dec = None,
    primary_beam_baseline_ra = None, primary_beam_baseline_dec = None,
    
    fixed_primary_beam_size_ra = np.deg2rad(60),
    fixed_primary_beam_size_dec = np.deg2rad(60),
    n_element_ra = 16,
    n_element_dec = 16,
    eta_sig = 1.,
    T_rec_raw = 30.,
    t_obs_days = 5 * 365.25,
    n_pol = 1,
)

BURSTT2048 = AntennaArray(
    name = 'BURSTT2048',
    nu_min = 300, nu_max = 800,
    dec = np.deg2rad(23.7),
    pointing = False,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,

    fixed_survey_za_max = None,
    size_ra = None, size_dec = None,
    eta_f_ra = None, eta_f_dec = None,
    eta_a_ra = None, eta_a_dec = None,
    primary_beam_baseline_ra = None, primary_beam_baseline_dec = None,

    fixed_primary_beam_size_ra = np.deg2rad(60),
    fixed_primary_beam_size_dec = np.deg2rad(60),
    n_element_ra = 45,
    n_element_dec = 45,
    eta_sig = 1.,
    T_rec_raw = 30.,
    t_obs_days = 5 * 365.25,
    n_pol = 1,
)


#===== telescopes for simple estimates =====

PUMA5k = Telescope(
    name = 'PUMA5k',
    nu_min = 200, nu_max = 1100,
    dec = None,
    pointing = True,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = None,
    size_ra = 71 * 6 * 100, size_dec = 71 * 6 * 100,
    eta_f_ra = np.sqrt(np.pi/4), eta_f_dec = np.sqrt(np.pi/4),
    eta_a_ra = 1., eta_a_dec = 1.,
    primary_beam_baseline_ra = 6 * 100, primary_beam_baseline_dec = 6 * 100,
    eta_sig = None,
    T_rec_raw = None,
    t_obs_days = 5 * 365.25,
    n_pol = 2,
)

PUMA5kC = Telescope(
    name = 'PUMA5kC',
    nu_min = 200, nu_max = 1100,
    dec = None,
    pointing = True,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = None,
    size_ra = 376 * 100, size_dec = 376 * 100,
    eta_f_ra = 1., eta_f_dec = 1.,
    eta_a_ra = 1., eta_a_dec = 1.,
    primary_beam_baseline_ra = 376 / np.sqrt(5000) * 100, primary_beam_baseline_dec = 376 / np.sqrt(5000) * 100,
    eta_sig = None,
    T_rec_raw = None,
    t_obs_days = 5 * 365.25,
    n_pol = 2,
)

PUMA32k = Telescope(
    name = 'PUMA32k',
    nu_min = 200, nu_max = 1100,
    dec = None,
    pointing = True,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = None,
    size_ra = 179 * 6 * 100, size_dec = 179 * 6 * 100,
    eta_f_ra = np.sqrt(np.pi/4), eta_f_dec = np.sqrt(np.pi/4),
    eta_a_ra = 1., eta_a_dec = 1.,
    primary_beam_baseline_ra = 6 * 100, primary_beam_baseline_dec = 6 * 100,
    eta_sig = None,
    T_rec_raw = None,
    t_obs_days = 5 * 365.25,
    n_pol = 2,
)

FARSIDE = Telescope(
    name = 'FARSIDE',
    nu_min = 300, nu_max = 500,
    dec = None,
    pointing = True,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = None,
    size_ra = 1e5 * 100, size_dec = 1e5 * 100,
    eta_f_ra = 1., eta_f_dec = 1.,
    eta_a_ra = 1., eta_a_dec = 1.,
    primary_beam_baseline_ra = 1e5/316 * 100, primary_beam_baseline_dec = 1e5/316 * 100,
    eta_sig = None,
    T_rec_raw = None,
    t_obs_days = 5 * 365.25,
    n_pol = 2,
)