"""Functions for the NFW halo"""

import sys

import jax.numpy as jnp
from jax import jit, vmap

sys.path.append("..")
from axionmirror import units_constants as uc
from axionmirror.geometry import Gr


#===== constants =====

rho_r_Sun = 0.44*uc.GeV / uc.c0**2 # [g/cm^3]
intg_d_s = jnp.concatenate((jnp.linspace(0, 20, 200, endpoint=False), # [kpc]
                            jnp.linspace(20, 200, 200, endpoint=False),
                            jnp.linspace(200, 1000, 100, endpoint=True)))


#===== functions =====

def rho_NFW(r, gamma=1., r_s=16.):
    """Generalized NFW halo density rho [g/cm^3] as a function of distance to
    galactic center r [kpc], gamma [1], and scale radius r_s [kpc]. Vectorized
    manually (first entry).
    """
    rho_unnorm = (r/r_s)**(-gamma) * (1 + r/r_s)**(-3+gamma)
    rho_unnorm_Sun = (uc.r_Sun/r_s)**(-gamma) * (1 + uc.r_Sun/r_s)**(-3+gamma)
    return (rho_r_Sun/rho_unnorm_Sun) * rho_unnorm

rho_integral_ref_length = 10 * uc.kpc # [cm]
rho_integral_ref = rho_NFW(uc.r_Sun) * rho_integral_ref_length # [g/cm^3] [cm] = [g/cm^2]


@jit
@vmap
def rho_integral(lb):
    """NFW halo density integral [g/cm^2] as a function of galactic coordinates
    (l, b) [rad, rad]. Vmapped.
    """
    l, b = lb # single numbers
    lbd_s = jnp.stack([ jnp.full_like(intg_d_s, l),
                        jnp.full_like(intg_d_s, b),
                        intg_d_s ], axis=-1)
    rho_s = rho_NFW(Gr(lbd_s))
    return jnp.trapz(rho_s, intg_d_s) * uc.kpc