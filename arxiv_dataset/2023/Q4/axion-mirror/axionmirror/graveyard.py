"""Functions for graveyard SNRs"""

import sys
import json
from tqdm import tqdm

import numpy as np
from scipy import stats
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

sys.path.append("..")
from axionmirror import units_constants as uc
from axionmirror.geometry import Glbd, GCstz, GCxyz_stz
from axionmirror.stats import sample_from_pdf, sample_from_pdf_jax, poisson_process
from axionmirror.nfw import rho_NFW
from axionmirror.snr import SNR, lightcurve_scale


#===== rate =====

def snr_forming_rate_CC(value='est'):
    """CC SNR formation rate [1/yr].
    From 1306.0559 Adams Kochanek Beacom Vagins Stanek.
    3.2+7.3-2.6 / 100yr
    """
    if value == 'est':
        return 3.2 / 100
    elif value == 'upper':
        return (3.2+7.3) / 100
    elif value == 'lower':
        return (3.2-2.6) / 100
    else:
        raise ValueError(value)

def snr_forming_rate_Ia(value='est'):
    """Ia SNR formation rate [1/yr].
    From 1306.0559 Adams Kochanek Beacom Vagins Stanek.
    1.4+1.4-0.8 / 100 yr
    """
    if value == 'est':
        return 1.4 / 100
    elif value == 'upper':
        return (1.4+1.4) / 100
    elif value == 'lower':
        return (1.4-0.8) / 100
    else:
        raise ValueError(value)

def snr_forming_rate_tot(value='est'):
    """Total SNR formation rate [1/yr].
    From 1306.0559 Adams Kochanek Beacom Vagins Stanek.
    4.6+7.4-2.7 / 100 yr
    """
    if value == 'est':
        return 4.6 / 100
    elif value == 'upper':
        return (4.6+7.4) / 100
    elif value == 'lower':
        return (4.6-2.7) / 100
    else:
        raise ValueError(value)
        

#===== location =====

def snr_stz_pdf_AKBVS(stz):
    """CC SNR location pdf.
    From 1306.0559 Adams Kochanek Beacom Vagins Stanek.
    Vectorized in first dimension.
    """
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    s0 = 2.9 # [kpc]
    z0 = 0.095 # [kpc]
    return np.exp(-s/s0) * np.exp(-np.abs(z)/z0)

def sample_snr_stz_AKBVS(num_samples=1):
    """SNR location sampler.
    From 1306.0559 Adams Kochanek Beacom Vagins Stanek.
    """
    s0 = 2.9 # [kpc]
    s_pdf = lambda s: np.exp(-s/s0)
    s_start, s_end = 0, 15 # [kpc]
    
    z0 = 0.095 # [kpc]
    z_pdf = lambda z: np.exp(-np.abs(z)/z0)
    z_start, z_end = -1, 1 # [kpc]
    
    s_samples = sample_from_pdf(s_pdf, s_start, s_end, num_samples)
    z_samples = sample_from_pdf(z_pdf, z_start, z_end, num_samples)
    t_samples = np.random.uniform(0, 2*np.pi, num_samples)
    
    return np.stack([s_samples,
                     t_samples,
                     z_samples], axis=-1)

def snr_stz_pdf_G(stz):
    """SNR location pdf.
    Following 1508.02931 Green.
    Vectorized in first dimension.
    
    Take z to be Uniform(-0.1, 0.1).
    SNR type: no mention
    """
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    
    R_sun = 8.5 # [kpc]
    alpha = 1.09 # [1]
    beta = 3.87 # [1]
    return (s/R_sun)**alpha * jnp.exp(-beta*(s-R_sun)/R_sun) * (jnp.abs(z) < 0.1)

def snr_stz_pdf_G_softz(stz):
    """Same as above except distribution on z is replace with exponential falloff.
    This is for SNR distance sampling (to avoid sampling outside of the support of pdf).
    """
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    
    R_sun = 8.5 # [kpc]
    z_scale = 0.1 # [kpc]
    alpha = 1.09 # [1]
    beta = 3.87 # [1]
    return (s/R_sun)**alpha * jnp.exp(-beta*(s-R_sun)/R_sun) * jnp.exp(-jnp.abs(z)/z_scale)

def snr_fi_pdf(stz, kernel, z_scale):
    """Pdf from kde of observed fullinfo SNR. Vectorized in first dimension."""
    xyz = GCxyz_stz(stz)
    xyz_in = jnp.stack([xyz[:,0], xyz[:,1], xyz[:,2]*z_scale], axis=0)
    return kernel.pdf(xyz_in)

def sample_snr_stz_G(num_samples=1):
    """SNR location sampler.
    Following 1508.02931 Green.
    Take z to be Uniform(-0.1, 0.1).
    """
    R_sun = 8.5 # [kpc]
    alpha = 1.09 # [1]
    beta = 3.87 # [1]
    s_pdf = lambda s: (s/R_sun)**alpha * np.exp(-beta*(s-R_sun)/R_sun)
    s_start, s_end = 0, 15 # [kpc]
    
    s_samples = sample_from_pdf(s_pdf, s_start, s_end, num_samples)
    z_samples = np.random.uniform(-0.1, 0.1, num_samples)
    t_samples = np.random.uniform(0, 2*np.pi, num_samples)
    
    return np.stack([s_samples,
                     t_samples,
                     z_samples], axis=-1)

@partial(jit, static_argnames=['stz_pdf_func', 'lowerbound', 'upperbound', 'num_samples'])
def sample_snr_d(rng_key, stz_pdf_func, l, b, lowerbound, upperbound, num_samples=1):
    """SNR distance sampler.
    Given stz_pdf_func(callable), l [rad], b [rad], lowerbound [kpc], upperbound [kpc],
    sample 1 instance of d [kpc].
    """
    def d_pdf(d):
        stz = GCstz(jnp.array([[l, b, d]]))
        return stz_pdf_func(stz)[0] * d**2
    
    return sample_from_pdf_jax(rng_key, vmap(d_pdf), lowerbound, upperbound, num_samples)


#===== spectral index =====

si_fit_dict = json.load(open('../outputs/snr/si_fit_dict.json', 'r'))
def sample_si(num_samples=1):
    return stats.skewnorm.rvs(si_fit_dict['skewness'], loc=si_fit_dict['loc'], scale=si_fit_dict['scale'], size=num_samples)


#===== size =====

size_fit_dict = json.load(open('../outputs/snr/size_fit_dict.json', 'r'))
def sample_size_1kyr(num_samples=1):
    return stats.skewnorm.rvs(size_fit_dict['skewness'], loc=size_fit_dict['loc'], scale=size_fit_dict['scale'], size=num_samples)

def sample_size_now(num_samples=1, t_now=...):
    """t_now in [yr], can be a vector."""
    return sample_size_1kyr(num_samples) * (1000/t_now)**0.4


#===== lightcurve =====

t_pk_mean = 50 / 365.25 # [yr]
t_pk_stddex = 0.9 # [1]
L_pk_mean = 3e25 # [erg/s/Hz]
L_pk_stddex = 1.5 # [1]

def log10_t_pk_pdf(log10_t_pk):
    """Unnormalized pdf for log_10(t_pk/yr), can be a vector."""
    return np.exp(- (log10_t_pk - np.log10(t_pk_mean))**2 / (2 * t_pk_stddex**2))

def log10_L_pk_pdf(log10_L_pk):
    """Unnormalized pdf for L_pk in [erg/s/Hz], can be a vector."""
    return np.exp(- (log10_L_pk - np.log10(L_pk_mean))**2 / (2 * L_pk_stddex**2))

def sample_t_pk(num_samples=1):
    """Sample t_pk [yr]"""
    return 10**(stats.norm.rvs(loc=np.log10(t_pk_mean), scale=t_pk_stddex, size=num_samples))

def sample_L_pk(num_samples=1):
    """Sample L_pk [erg/s/Hz]"""
    return 10**(stats.norm.rvs(loc=np.log10(L_pk_mean), scale=L_pk_stddex, size=num_samples))

def sample_t_pk_L_pk(snr, tiop='2', num_samples=1):
    """Sample t_pk [yr] conditioned on Snu(now)"""

    def L_pk(t_pk):
        """Vectorized in t_pk."""
        # part of the build function
        p   = 2*snr.si+1 # si determines p
        ti1 = -2*(p+1)/5
        ti2 = -4*p/5
        ti  = ti1 if tiop=='1' else ti2 # ti < 0
        Snu1GHz_t_free = snr.Snu1GHz * (snr.t_now/snr.t_free)**(-ti)
        Snu1GHz_pk = Snu1GHz_t_free / lightcurve_scale(t_pk, snr.t_free)
        Snu6p3GHz_pk = Snu1GHz_pk * (6.3/1)**(-snr.si)
        L_pk = 4*np.pi * (snr.d*uc.kpc)**2 * Snu6p3GHz_pk * uc.Jy * uc.sec**2
        return L_pk

    def conditioned_log10_t_pk_pdf(log10_t_pk):
        return log10_t_pk_pdf(log10_t_pk) * log10_L_pk_pdf(np.log10(L_pk(10**log10_t_pk)))
    
    log10_t_pk = sample_from_pdf(conditioned_log10_t_pk_pdf, np.log10(t_pk_mean/1e4), np.log10(snr.t_free), num_samples)
    return 10**log10_t_pk

def sample_Snu1GHz_pk(num_samples=1, si=..., d=...):
    """d in [kpc]. d and si can be vectors."""
    Snu6p3GHz_pk = sample_L_pk(num_samples) / (4*np.pi*(d*uc.kpc)**2) / uc.sec**2 / uc.Jy # [Jy]
    return Snu6p3GHz_pk * np.sqrt(4*10) ** si
    
    
#===== t_now (age) =====

log10age_fit_dict = json.load(open('../outputs/snr/log10age_fit_dict.json', 'r'))
def sample_t_now(num_samples=1):
    return 10**stats.skewnorm.rvs(log10age_fit_dict['skewness'], loc=log10age_fit_dict['loc'], scale=log10age_fit_dict['scale'], size=num_samples)


#===== t_free =====

def fixed_t_free(value='est'):
    """[yr]"""
    if value == 'est':
        return 100.
    elif value == 'upper':
        return 30.
    elif value == 'lower':
        return 300.
    else:
        raise ValueError(value)
    

#===== full graveyard sample =====
    
def sample_graveyard_snrs(t_cutoff=100000, verbose=1, build=False, var_flag='base'):
    """t_cutoff in [yr], set at end of adiabatic phase typically."""

    if var_flag == 'ti1':
        tiop = '1'
    else:
        tiop = '2'

    if var_flag == 'tf30':
        t_free = 30. # [yr]
    elif var_flag == 'tf300':
        t_free = 300. # [yr]
    else:
        t_free = 100. # [yr]

    if var_flag == 'srhigh':
        snr_rate_str = 'upper'
    elif var_flag == 'srlow':
        snr_rate_str = 'lower'
    else:
        snr_rate_str = 'est'

    t_now_arr = np.array(poisson_process(snr_forming_rate_tot(value=snr_rate_str), t_cutoff))
    n_snr = len(t_now_arr)
    if verbose >= 1:
        print(f'n_snr={n_snr}')
    
    lbd_arr = np.array(Glbd(sample_snr_stz_G(n_snr)), dtype=np.float64)
    si_arr = sample_si(n_snr)
    size_arr = sample_size_now(n_snr, t_now=t_now_arr)
    t_pk_arr = sample_t_pk(n_snr)
    t_free_arr = np.full((n_snr,), t_free)
    Snu1GHz_pk_arr = sample_Snu1GHz_pk(n_snr, si=si_arr, d=lbd_arr[:, 2])
    
    snr_list = []
    for i in (tqdm(range(n_snr)) if verbose >= 1 else range(n_snr)):
        
        Snu1GHz_pk = Snu1GHz_pk_arr[i]
        while True:
            snr = SNR(
                ID = f'Graveyard-{i}',
                l = lbd_arr[i, 0],
                b = lbd_arr[i, 1],
                d = lbd_arr[i, 2],
                size = size_arr[i],
                t_now = t_now_arr[i],
                t_free = t_free_arr[i],
                t_pk = t_pk_arr[i],
                Snu1GHz_pk = Snu1GHz_pk,
                si = si_arr[i],
            )
            snr.build(rho_DM=rho_NFW, use_lightcurve=True, integrate_method='trapz', tiop=tiop)
            if snr.Snu_t(400., snr.t_now) < 4657.:
                break
            else:
                if verbose >= 1:
                    print(f'resampling {i}')
                Snu1GHz_pk = sample_Snu1GHz_pk(1, si=si_arr[i], d=lbd_arr[i, 2])[0]
            
        snr_list.append(snr)
    return snr_list