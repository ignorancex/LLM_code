"""Functions and class for supernova remnants (SNR)"""

import sys
import json
from dataclasses import dataclass, asdict

import numpy as np
from scipy import integrate
from astropy.coordinates import SkyCoord
import jax.numpy as jnp

sys.path.append("..")
from axionmirror import units_constants as uc
from axionmirror.spectral import prefac
from axionmirror.math import right_geom_trapz
from axionmirror.nfw import rho_NFW


#===== image =====

def gaussian_val(sigma, x0, y0, x, y):
    """x, y can take any shape. Use np to be variable."""
    return np.exp( -((x-x0)**2+(y-y0)**2)/(2*sigma**2) )

def gaussian_integral_estimate(x0, y0, sigma,
                               x_range, y_range, sample_n=(10, 10)):
    """Rough estimate of 2D gaussian integral in rectangles by sampling points
    uniformly.
    x0, y0, sigma: parameters for the gaussian.
    x/y_range: (x/y_start, x/y_end).
    sample_n: (# of pts sampled in x, ... in y)
    """
    x_edges = jnp.linspace(*x_range, sample_n[0]+1)
    y_edges = jnp.linspace(*y_range, sample_n[1]+1)
    xs = (x_edges[1:] + x_edges[:-1]) / 2
    ys = (y_edges[1:] + y_edges[:-1]) / 2
    x_grid, y_grid = jnp.meshgrid(xs, ys)
    x = x_grid.flatten()
    y = y_grid.flatten()
    z = jnp.exp( -((x-x0)**2+(y-y0)**2)/(2*sigma**2) )
    return jnp.sum(z) * (x_range[1]-x_range[0]) * (y_range[1]-y_range[0]) \
           / (sample_n[0]*sample_n[1])

def norm_gaussian(sigma, x0=0, y0=0):
    """Returns a 2D gaussian function (normalized) as a function of sigma [rad]."""
    return lambda x, y: 1/(2*np.pi*sigma**2) * np.exp( -((x-x0)**2+(y-y0)**2)/(2*sigma**2) )

def gaussian(sigma, x0=0, y0=0):
    """Returns a 2D gaussian function (unnormalized) as a function of sigma [rad]."""
    return lambda x, y: np.exp( -((x-x0)**2+(y-y0)**2)/(2*sigma**2) )

def add_image_to_map(fullmap, ra_s, dec_s, ra_edges, dec_edges, pixel_area_map,
                     source_ra, source_dec, image_sigma, n_sigma, T_submap_prefactor,
                     use_gaussian_intg=False, modify_mode='add', debug=False):
    """Add a gaussian image to a map."""
    
    if source_dec < dec_edges[0] or source_dec > dec_edges[-1]: # can't see source
        return
    
    i_ra_st  = int(np.searchsorted(ra_s,  source_ra  - n_sigma*image_sigma))
    i_ra_ed  = int(np.searchsorted(ra_s,  source_ra  + n_sigma*image_sigma))
    i_dec_st = int(np.searchsorted(dec_s, source_dec - n_sigma*image_sigma))
    i_dec_ed = int(np.searchsorted(dec_s, source_dec + n_sigma*image_sigma))
    if i_ra_ed == i_ra_st:
        if i_ra_ed == len(ra_s):
            i_ra_st -= 1
        else:
            i_ra_ed += 1
    if i_dec_ed == i_dec_st:
        if i_dec_ed == len(dec_s):
            i_dec_st -= 1
        else:
            i_dec_ed += 1

    ra_subgrid, dec_subgrid = np.meshgrid(ra_s[i_ra_st:i_ra_ed], dec_s[i_dec_st:i_dec_ed])

    if ra_subgrid.shape == (1, 1):
        tmpl = np.array([[1.]])
    else:
        if use_gaussian_intg:
            tmpl = np.zeros_like(ra_subgrid)
            for i_ra in range(i_ra_st, i_ra_ed):
                for i_dec in range(i_dec_st, i_dec_ed):
                    tmpl[i_dec-i_dec_st,i_ra-i_ra_st] = gaussian_integral_estimate(
                        source_ra, source_dec, image_sigma,
                        (ra_edges[i_ra],ra_edges[i_ra+1]), (dec_edges[i_dec],dec_edges[i_dec+1])
                    )
        else:
            tmpl = gaussian_val(image_sigma, source_ra, source_dec, ra_subgrid, dec_subgrid)
    pixel_area_submap = pixel_area_map[i_dec_st:i_dec_ed, i_ra_st:i_ra_ed]
    tmpl = np.array(tmpl) * pixel_area_submap
    tmpl /= np.sum(tmpl)

    #T_submap = snr.Sgnu(freq)*Jy * tmpl * c0**2 / (2 * freq**2 * pixel_area_submap * kb)
    #T_submap_prefactor = snr.Sgnu(freq)*Jy * 1 * c0**2 / (2 * freq**2 * 1 * kb)
    T_submap = T_submap_prefactor * tmpl / pixel_area_submap
    # [K] = [MHz^2 g] [cm MHz]^2 [MHz]^-2 [cm^2 MHz^2 g K^-1]^-1
    
    if modify_mode == 'add':
        fullmap[i_dec_st:i_dec_ed, i_ra_st:i_ra_ed] += T_submap
    elif modify_mode == 'replace':
        fullmap[i_dec_st:i_dec_ed, i_ra_st:i_ra_ed] = T_submap
    else:
        raise NotImplementedError(modify_mode)
        
    if debug:
        print(np.max(T_submap))


#===== SNR =====

def lightcurve_scale(t_pk, t):
    """S/S_pk as a function of t/t_pk. Vectorized in t."""
    return np.exp(1.5*(1-t_pk/t)) * (t/t_pk)**(-1.5)


@dataclass
class SNR:
    """Class for supernova remnants (SNRs).
    Units:
        nu... [MHz]
        angles [rad] | coordinates, sizes, image_sigma..., theta...
        distances [kpc] | d..., x..., Gr...
        t... [yr]
        Snu... [Jy]
    """
    ID: str = None
    name_alt: str = None
    snr_type: str = None
    snrcat_dict: dict = None
    green_dict: dict = None
    l: float = None # [rad] | galactic longitude
    b: float = None # [rad] | galactic latitude
    size: float = None # [rad] | size estimate
    d: float = None # [kpc] | distance estimate
    d_min: float = None # [kpc]
    d_max: float = None # [kpc]
    t_free: float = None # [yr] | t=0 at birth
    t_now: float = None  # [yr] | t=0 at birth
    t_now_min: float = None  # [yr]
    t_now_max: float = None  # [yr]
    t_MFA: float = None  # [yr] | onset time of magnetic field amplification
    si: float = None # [1] | si>0 | spectral index = (p-1)/2
    si_guessed: bool = None
    Snu1GHz: float = None # [Jy] | Required by observed SNRs
    Snu1GHz_pk: float = None # [Jy] | Required by graveyard SNRs
    t_pk: float = None # [yr] | Time at peak luminosity
    
    #===== options =====
    tiop: str = None # '1' or '2' | option for t index of powerlaw for adiabatic expansion, depending on electron model
    integrate_method: str = None # 'quad' or 'trapz' | integration method for gegenschein
    use_lightcurve: bool = None # Whether to use lightcurve for free expansion
    infer_lightcurve_from_now: bool = None # Whether lightcurve inferred from information at present day
    
    #===== below are dependent quantities =====
    cl: float = None
    cb: float = None
    ra: float = None
    dec: float = None
    hemisph: str = None
    thetaGCCS: float = None
    thetaGCS: float = None
    p: float = None
    ti1: float = None
    ti2: float = None
    ti: float = None
    Snu1GHz_t_free: float = None
    nu_ref: float = None
    
    Sgnu_ref: float = None
    image_sigma: float = None
    Sfgnu_ref: float = None
    image_sigma_fg: float = None
    
        
    def __post_init__(self):
        if self.l == 0 and self.b == 0:
            self.b = np.deg2rad(0.1)

    
    def build(self, rho_DM=rho_NFW, tiop='2', integrate_method='quad',
              use_lightcurve=False, infer_lightcurve_from_now=False):
        
        #========== variables ==========
        self.cl = self.l + np.pi - 2*np.pi*int(self.l>=np.pi) # [rad] | countersource l | [-pi,pi)
        self.cb = - self.b # [rad] | countersource b
        coord = SkyCoord(l=self.l, b=self.b, frame='galactic', unit='rad')
        self.ra  = coord.icrs.ra.rad
        self.dec = coord.icrs.dec.rad
        self.hemisph = 'N' if self.dec > 0 else 'S'
        self.thetaGCCS = np.arccos( -np.cos(self.l)*np.cos(self.b) ) # [rad] | GC and counter source
        self.thetaGCS = np.pi - self.thetaGCCS # [rad] | GC and source
        self.p   = 2*self.si+1 # si determines p
        self.ti1 = -2*(self.p+1)/5
        self.ti2 = -4*self.p/5
        self.tiop = tiop
        self.ti = self.ti1 if self.tiop=='1' else self.ti2
        
        #========== Snu(nu, t) ==========
        self.use_lightcurve = use_lightcurve
        self.infer_lightcurve_from_now = infer_lightcurve_from_now
        if self.use_lightcurve:
            if self.infer_lightcurve_from_now:
                # requires Snu1GHz, t_pk, t_free
                self.Snu1GHz_t_free = self.Snu1GHz * (self.t_now/self.t_free)**(-self.ti) # ti < 0
                self.Snu1GHz_pk = self.Snu1GHz_t_free / lightcurve_scale(self.t_pk, self.t_free)
            else:
                # requires Snu1GHz_pk, t_pk, t_free
                self.Snu1GHz_t_free = self.Snu1GHz_pk * lightcurve_scale(self.t_pk, self.t_free)
                self.Snu1GHz = self.Snu1GHz_t_free * (self.t_now/self.t_free)**self.ti # ti < 0
        else:
            # requires Snu1GHz, t_MFA
            pass
            
        #========== integration setup ==========
        EPSILON = 1e-8
        self.integrate_method = integrate_method
        if self.integrate_method == 'quad':
            def integrator(*args):
                return integrate.quad(*args)[0]
        elif self.integrate_method == 'trapz':
            def integrator(*args):
                return right_geom_trapz(*args, offset=0.01*uc.c0_kpc_yr) # [0.01 yr]
        else:
            raise NotImplementedError(self.integrate_method)
            
        self.nu_ref = 1000. # [MHz]
        
        #========== gegenschein & front gegenschein ==========
        intgd = lambda xp: self.Snu_t(self.nu_ref, self.t(xp)) * rho_DM(np.maximum(self.Gr(xp), 1e-3)) * uc.kpc
        # converted intgd*dx from [Jy g/cm^3 kpc] to [Jy g/cm^3 cm] to bring numerical value closer to 1
        intg = integrator(intgd, 0, self.xp(0)-EPSILON) # [Jy g/cm^2]
        self.Sgnu_ref = prefac(self.nu_ref) * intg # = [g^-1 cm^2] [Jy g cm^-2] = [Jy]
        imsz_intgd = lambda xp: self.image_sigma_at(xp) * self.Snu_t(self.nu_ref, self.t(xp)) * rho_DM(self.Gr(xp)) * uc.kpc
        imsz_intg = integrator(imsz_intgd, 0, self.xp(0)-EPSILON)
        self.image_sigma = imsz_intg/intg # [arcmin]

        intgd = lambda xp: self.Snu_t(self.nu_ref, self.t_fg(xp)) * rho_DM(np.maximum(self.Gr_fg(xp), 1e-3)) * uc.kpc
        # converted intgd*dx from [Jy g/cm^3 kpc] to [Jy g/cm^3 cm] to bring numerical value closer to 1
        intg = integrator(intgd, self.d, self.xp_fg(0)-EPSILON) # [Jy g/cm^2]
        self.Sfgnu_ref = prefac(self.nu_ref) * intg # = [g^-1 cm^2] [Jy g cm^-2] = [Jy]
        imsz_intgd = lambda xp: self.image_sigma_at_fg(xp) * self.Snu_t(self.nu_ref, self.t_fg(xp)) * rho_DM(self.Gr_fg(xp)) * uc.kpc
        imsz_intg = integrator(imsz_intgd, self.d, self.xp_fg(0)-EPSILON)
        self.image_sigma_fg = imsz_intg/intg # [arcmin]
        
        if not np.all(np.isfinite([self.Sgnu_ref, self.Sfgnu_ref, self.image_sigma, self.image_sigma_fg])):
            print(self.Sgnu_ref, self.Sfgnu_ref, self.image_sigma, self.image_sigma_fg)
            raise ValueError(f'{self.ID}: integrated quantities contains non-finite values.')
            
        
    def name(self):
        """Returns name_alt or ID if name_alt is not present."""
        return self.name_alt if self.name_alt != '' else self.ID
    
    def size_t(self, t):
        """Size [rad] as a function of t [yr] (can be a vector)."""
        return np.where(
            t > self.t_free, self.size * (t/self.t_now)**0.4,
            np.where(
                t > 0, self.size * (self.t_free/self.t_now)**0.4 * (t/self.t_free),
                np.zeros_like(t),
            )
        )
        
    def xp(self, t):
        """x' (x_d) [kpc] as a function of SNR t [yr] (can be a vector)."""
        return uc.c0_kpc_yr * (self.t_now-t) / 2
    
    def xp_fg(self, t):
        """x' (x_d) [kpc] as a function of SNR t [yr] (can be a vector).
        Front gegenschein."""
        return uc.c0_kpc_yr * (self.t_now-t) / 2 + self.d
    
    def t(self, xp):
        """SNR t [yr] as a function of x' (x_d) [kpc] (can be a vector)."""
        return self.t_now - 2*xp/uc.c0_kpc_yr
    
    def t_fg(self, xp):
        """SNR t [yr] as a function of x' (x_d) [kpc] (can be a vector).
        Front gegenschein."""
        return self.t_now - 2*(xp - self.d)/uc.c0_kpc_yr
    
    def blur_sigma(self, xp):
        """Blur sigma [rad] as a function of x' (x_d) [kpc] (can be a vector)."""
        return (1 + xp/self.d) * 2 * uc.sigmad_over_c
    
    def blur_sigma_fg(self, xp):
        """Blur sigma [rad] as a function of x' (x_d) [kpc] (can be a vector).
        Front gegenschein."""
        return (xp/self.d - 1) * 2 * uc.sigmad_over_c
    
    def Gr(self, xp):
        """Distance to GC [kpc] as a function of x' (x_d) [kpc] (can be a vector)."""
        return np.sqrt(xp**2 + uc.r_Sun**2 - 2*uc.r_Sun*xp*np.cos(self.thetaGCCS))
    
    def Gr_fg(self, xp):
        """Distance to GC [kpc] as a function of x' (x_d) [kpc] (can be a vector).
        Front gegenschein."""
        return np.sqrt(xp**2 + uc.r_Sun**2 - 2*uc.r_Sun*xp*np.cos(self.thetaGCS))
    
    def image_sigma_at(self, xp):
        """Gegenschein image sigma [rad] as a function of x' (x_d) [kpc]
        (can be a vector)."""
        return np.sqrt((self.size_t(self.t(xp))/4)**2 + self.blur_sigma(xp)**2)
    
    def image_sigma_at_fg(self, xp):
        """Front gegenschein image sigma [rad] as a function of x' (x_d) [kpc]
        (can be a vector)."""
        return np.sqrt((self.size_t(self.t_fg(xp))/4)**2 + self.blur_sigma_fg(xp)**2)
    
    
    def Snu(self, nu):
        """Specific flux [Jy] as a function of frequency nu [MHz]."""
        return self.Snu1GHz * (nu/1000)**(-self.si)
    
    def Snu_t_lightcurve(self, nu, t):
        """Snu: lightcurve --t_free--> t^ti. Doesn't care about t_MFA.
        units: [Jy]([MHz], [yr]). Vectorized in t."""
        return np.where(
            t <= 0, np.zeros_like(t),
            np.where(
                t < self.t_free, self.Snu1GHz_pk * lightcurve_scale(self.t_pk, t) * (nu/1000)**(-self.si),
                self.Snu1GHz_t_free * (t/self.t_free)**self.ti * (nu/1000)**(-self.si)
            )
        )
    
    def Snu_t_fl(self, nu, t):
        """Snu: constant --t_MFA--> t^ti. Doesn't care about t_free.
        units: [Jy]([MHz], [yr]). Vectorized in t."""
        return np.where(
            t < 0, np.zeros_like(t),
            np.where(
                t > self.t_MFA, self.Snu(nu) * (t/self.t_now)**self.ti,
                self.Snu(nu) * (self.t_MFA/self.t_now)**self.ti
            )
        )
    
    def Snu_t(self, nu, t):
        """Switch between Snu_t_lightcurve and Snu_t_fl. Vectorized in t."""
        if self.use_lightcurve:
            return self.Snu_t_lightcurve(nu, t)
        else:
            return self.Snu_t_fl(nu, t)
        
    def L_t(self, nu, t):
        """Luminosity [erg/s/Hz] as a function of frequency nu [MHz] and time t [yr]. Vectorized in t."""
        return 4*np.pi * (self.d*uc.kpc)**2 * self.Snu_t(nu, t) * uc.Jy * uc.sec**2
        
    def Sgnu(self, nu):
        """Gegenschein flux [Jy] as a function of frequency nu [MHz]."""
        return self.Sgnu_ref * (prefac(nu)/prefac(self.nu_ref)) * (self.Snu(nu)/self.Snu1GHz)
    
    def Sfgnu(self, nu):
        """Front gegenschein flux [Jy] as a function of frequency nu [MHz]."""
        return self.Sfgnu_ref * (prefac(nu)/prefac(self.nu_ref)) * (self.Snu(nu)/self.Snu1GHz)
        

#===== SNR utilities =====

def GID(l, b):
    """Standard ID for a SNR with l [deg], b [deg]."""
    if isinstance(l, str):
        l = float(l)
    if isinstance(b, str):
        sign = b[0]
        b = float(b)
    else:
        sign = '+' if b>0 else '-'
    return 'G' + ('%05.1f'%l) + sign + ('%04.1f'%np.abs(b))

def get_snr(name, snr_list):
    """Get a SNR object from a list of SNR objects by name or alternative name."""
    for snr in snr_list:
        if snr.ID==name or snr.name_alt==name:
            return snr
    return None


def dump_snr_list(snr_list, fn):
    """Save a list of SNR objects to a json file."""
    config_list = [asdict(snr) for snr in snr_list]
    json.dump(config_list, open(fn, 'w'))
    
def load_snr_list(fn):
    """Load a list of SNR objects from a json file."""
    config_list = json.load(open(fn, 'r'))
    return [SNR(**config) for config in config_list]

def best_snrs(snr_list, top_n, nu_ref=1000, t_ref=50/365.25):
    """Return the top_n SNRs with the highest Snu_t(nu_ref [MHz], t_ref [yr])."""
    inds = np.argsort([snr.Snu_t(nu_ref, t_ref) for snr in snr_list])[::-1]
    return [snr_list[i] for i in inds[:top_n]]