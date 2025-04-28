"""Galactic B-field and electron density models

Units:
    length: [kpc]
    B field: [muGauss]
"""

import sys

import jax.numpy as jnp
from jax import vmap
from functools import partial

sys.path.append("..")
from axionmirror import units_constants as uc


#===== spectral indices =====

spec_ind_p = 3
spec_ind_alpha = (spec_ind_p+1)/2


#===== JF: B field =====

@partial(vmap, in_axes=(None, 0, 0, None))
def is_in_arm(stz, r_nx_low, r_nx_high, tani):
    
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    
    s_nx = s / jnp.exp((t-jnp.pi)*tani)
    in_arm_inner = jnp.logical_and(s_nx > r_nx_low, s_nx < r_nx_high)
    in_arm_outer = jnp.logical_and(s_nx > r_nx_low * jnp.exp(2*jnp.pi*tani),
                                   s_nx < r_nx_high * jnp.exp(2*jnp.pi*tani))
    in_s_bound = jnp.logical_and(s > 5, s < 20)
    return jnp.logical_and(jnp.logical_or(in_arm_inner, in_arm_outer), in_s_bound)


def logistic(z, h, w):
    return 1 / (1 + jnp.exp(-2*(jnp.abs(z)-h)/w))


def B_JF(stz):
    """B field model described in 1204.3662.
    B (stz vector field) ([muG], [muG], [muG]) as a function of stz ([kpc],
    [rad], [kpc]). Vectorized manually; batch dimension is the first.
    """
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    batchsize = stz.shape[0]
    
    ## disk
    i_angle = jnp.deg2rad(11.5)
    b_ring = 0.1 # [muG]
    b_arms = jnp.array([0.1, 3.0, -0.9, -0.8, -2.0, -4.2, 0.0, 2.7]) # [muG]
    r_nx_s = jnp.array([5.1, 6.3, 7.1, 8.3, 9.8, 11.4, 12.7, 15.5, 18.31]) # [kpc]
    h_disk = 0.4 # [kpc]
    w_disk = 0.27 # [kpc]
    
    # disk: inner
    frac_disk = 1 - logistic(z, h_disk, w_disk)
    B_inner_t = jnp.where(s < 5, b_ring, 0) * frac_disk
    
    # disk: arms
    is_in_arm_arr = is_in_arm(stz, r_nx_s[:-1], r_nx_s[1:], jnp.tan(i_angle))
    B_arms_5kpc = jnp.dot(b_arms, is_in_arm_arr) * frac_disk
    B_arms_s = jnp.sin(i_angle) * B_arms_5kpc * (5/s)
    B_arms_t = jnp.cos(i_angle) * B_arms_5kpc * (5/s)
    
    ## toroidal halo
    B_n, B_s = 1.4, -1.1 # [muG]
    r_n, r_s = 9.22, 16.7 # [kpc]
    w_h = 0.2 # [kpc]
    z_0 = 5.3 # [kpc]
    B_halo_t = jnp.exp(-jnp.abs(z)/z_0) * (1-frac_disk) * jnp.where( z > 0,
        B_n * (1 - logistic(s, r_n, w_h)),
        B_s * (1 - logistic(s, r_s, w_h)),
    )
    
    ## X halo
    B_X = 4.6 # [muG]
    Theta_0X = jnp.deg2rad(49)
    r_cX = 4.8 # [kpc]
    r_X = 2.9 # [kpc]
    
    # X halo: constant region
    rp_const = jnp.clip(s - jnp.abs(z) / jnp.tan(Theta_0X), 0, None)
    B_X_const_mag = B_X * jnp.exp(-rp_const/r_X) * (rp_const/s) * (s >= r_cX)
    B_X_const_s = B_X_const_mag * jnp.cos(Theta_0X) * jnp.sign(z)
    B_X_const_z = B_X_const_mag * jnp.sin(Theta_0X)
    
    # X halo: linear region
    rp_lin = s * r_cX / (r_cX + jnp.abs(z)/jnp.tan(Theta_0X))
    Theta_X = jnp.arctan(jnp.abs(z) / (s - rp_lin + 1e-5))
    B_X_lin_mag = B_X * jnp.exp(-rp_lin/r_X) * (rp_lin/s)**2 * (s < r_cX)
    B_X_lin_s = B_X_lin_mag * jnp.cos(Theta_X) * jnp.sign(z)
    B_X_lin_z = B_X_lin_mag * jnp.sin(Theta_X)
    
    return jnp.stack([B_arms_s + B_X_const_s + B_X_lin_s,
                      B_inner_t + B_arms_t + B_halo_t,
                      B_X_const_z + B_X_lin_z], axis=-1)


#===== JF: n_e =====

def n_e_WMAP(stz):
    """Electron density model from JF (WMAP).
    n_e [unnormalized] as a function of stz ([kpc], [rad], [kpc]). Vectorized
    manually; batch dimension is the first."""
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    hr = 5 # [kpc]
    hz = 1 # [kpc]
    return jnp.exp(-s/hr) * jnp.cosh(z/hz)**(-2)


#===== SRWE: B field =====

def B_SRWE_AH(stz_s):
    return B_SRWE_disk_ASR(stz_s) + B_SRWE_halo(stz_s)


def B_SRWE_BH(stz_s):
    return B_SRWE_disk_BS(stz_s) + B_SRWE_halo(stz_s)


def B_SRWE_disk_ASR(stz_s):
    """B_disk in Axi-symmetric spiral model + ring model (ASR) from 0711.1572.
    B (stz vector field) ([muG], [muG], [muG]) as a function of stz ([kpc],
    [rad], [kpc]). Vectorized manually; batch dimension is the first.
    """
    R_s, t_s, z_s = stz_s[:,0], stz_s[:,1], stz_s[:,2]
    R0, z0, Rc, Rsun = 10, 1, 5, 8.5 # all [kpc]
    B0, Bc = 2, 2 # all [muG]
    
    D1_s = jnp.where(R_s < Rc, Bc, B0*jnp.exp( -(R_s-Rsun)/R0-jnp.abs(z_s)/z0 ))
    D2_s = jnp.array(
        jnp.logical_or( (7.5<R_s), jnp.logical_and(5<R_s, R_s<6) ),
        dtype=float) * 2 - 1
    BDt_s = - D1_s * D2_s
    return jnp.stack([jnp.zeros_like(BDt_s),
                      BDt_s,
                      jnp.zeros_like(BDt_s)], axis=-1)


def B_SRWE_disk_BS(stz_s):
    """B_disk in Bi-symmetric spiral model (BS) from 0711.1572.
    B (stz vector field) ([muG], [muG], [muG]) as a function of stz ([kpc],
    [rad], [kpc]). Vectorized manually; batch dimension is the first.
    """
    R_s, t_s, z_s = stz_s[:,0], stz_s[:,1], stz_s[:,2]
    R0, z0, Rc, Rsun = 6, 1, 3, 8.5 # all [kpc]
    B0, Bc = 2, 2 # all [muG]
    
    D1_s = jnp.where(R_s < Rc, Bc, B0*jnp.exp( -(R_s-Rsun)/R0-jnp.abs(z_s)/z0 ))
    Rb_s = jnp.where(R_s > 6,  9,  6)
    p_s  = jnp.where(R_s > 6, jnp.deg2rad(-10), jnp.deg2rad(-15))
    beta_s = 1/jnp.tan(p_s)
    D2_s = -jnp.cos(t_s + beta_s*jnp.log(R_s/Rb_s))
    D1D2_s = D1_s*D2_s
    return jnp.stack([D1D2_s*jnp.sin(p_s),
                      -D1D2_s*jnp.cos(p_s),
                      jnp.zeros_like(D1D2_s)], axis=-1)


def B_SRWE_halo(stz_s):
    """B_halo model from 0711.1572.
    B (stz vector field) ([muG], [muG], [muG]) as a function of stz ([kpc],
    [rad], [kpc]). Vectorized manually; batch dimension is the first.
    """
    R_s, t_s, z_s = stz_s[:,0], stz_s[:,1], stz_s[:,2]
    BH0, RH0, zH0 = 10, 4, 1.5
    zH1_s = jnp.where(jnp.abs(z_s) < zH0, 0.2, 0.4)
    return jnp.stack([jnp.zeros_like(zH1_s),
                      BH0 / (1+((jnp.abs(z_s)-zH0)/zH1_s)**2)*R_s/RH0*jnp.exp(-(R_s-RH0)/RH0),
                      jnp.zeros_like(zH1_s)], axis=-1)


#===== SRWE: n_e =====

def n_e_SRWE(stz_s):
    """Electron density model from NE2001.
    n_e [unnormalized] as a function of stz ([kpc], [rad], [kpc]). Vectorized
    manually; batch dimension is the first.
    """
    s, t, z = stz_s[:,0], stz_s[:,1], stz_s[:,2]
    return jnp.where(jnp.abs(z)>1, 0.0,
                     jnp.exp(-(jnp.maximum(s,3)-uc.r_Sun)/8-jnp.abs(z)))