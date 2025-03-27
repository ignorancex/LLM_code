import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
from time import time

from profiley.helpers import lss


def main(config='mass_modeling_params.txt', threads=24):

    params = read_params(config)
    # cosmology
    cosmo = ccl.Cosmology(
        Omega_c=params['Omega_c'], Omega_b=params['Omega_b'],
        h=params['h'], A_s=params['As'], n_s=params['ns'])
    # mass range
    m = 10**np.arange(
        params['logm_min'], params['logm_max']+params['logm_bin'],
        params['logm_bin'])
    # mass definition
    mdef = ccl.halos.MassDef(params['delta'], params['background'])
    # redshift
    z = np.arange(
        params['z_min'], params['z_max']+1e-4, params['z_bin'])
    # scale factor
    a = 1 / (1+z)

    # matter power spectrum
    k = np.logspace(-15, 15, 10000)
    lnk = np.log(k)
    Pk = np.array([ccl.linear_matter_power(cosmo, k, ai) for ai in a])
    # halo bias
    bias = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef)
    bh = np.array([bias.get_halo_bias(cosmo, m, ai) for ai in a])
    # galaxy-matter power spectrum
    Pgm = bh[:,:,None] * Pk[:,None]
    lnPgm = np.log(Pgm)

    # correlation function
    Rxi = np.logspace(-3, 4, 100)
    xi = np.zeros((z.size,m.size,Rxi.size))
    for i in range(z.size):
        for j in range(m.size):
            lnPgm_lnk = interp1d(lnk, lnPgm[i,j])
            xi[i,j] = lss.power2xi(lnPgm_lnk, Rxi)

    # angular separations. To do the filtering we need to know the
    # boundaries of the angular binning, so that's what's defined in
    # the parameters file. These are in arcmin
    theta_bins = np.linspace(
        params['theta_min'], params['theta_max'], 1+params['theta_bins'])
    theta = (theta_bins[:-1]+theta_bins[1:]) / 2
    Dc = ccl.background.comoving_radial_distance(cosmo, a)
    # mean matter density at a=1
    rho_m = ccl.background.rho_x(cosmo, 1, 'matter')
    # comoving angular distance in Mpc
    R = theta * Dc[:,None] * (np.pi/180/60)

    # finally, calculate the surface density
    ti = time()
    sigma = lss.xi2sigma(R, Rxi, xi, rho_m, threads=threads, full_output=False)
    print('sigma in {0:.2f} min'.format((time()-ti)/60))
    print('shape =', sigma.shape)
    cosmo_params = {key: val for key, val in params.items()
                    if key in ('h','Omega_m','Omega_c','Omega_b','As','ns')}
    lss.save_profiles(
        'sigma_2h.txt', z, np.log10(m), theta, sigma, xlabel='z', ylabel='m',
        label='sigma_2h', R_units='arcmin', cosmo_params=cosmo_params)
    return


def read_params(config):
    params = {}
    with open(config) as f:
        for line in f:
            if line.strip()[0] == '#': continue
            key, val = line.split()
            for dtype in (int,float):
                try:
                    val = dtype(val)
                    break
                except ValueError:
                    pass
            params[key] = val
    if 'H0' in params:
        if 'h' in params and params['h'] != params['H0']/100:
            msg = f'inconsistent definitions of h and H0 in {config}.' \
                  ' Defining H0 is sufficient.'
            raise ValueError(msg)
        params['h'] = params['H0'] / 100
    if 'Omega_bh2' in params and 'Omega_b' not in params:
        params['Omega_b'] = params['Omega_bh2'] / params['h']**2
    if 'Omega_c' in params:
        if 'Omega_m' in params and 'Omega_b' in params \
                and params['Omega_c'] != params['Omega_m'] - params['Omega_b']:
            msg = 'inconsistent definitions of Omega_m, Omega_c and Omega_b' \
                  f' in {config}.'
            raise ValueError(msg)
    else:
        params['Omega_c'] = params['Omega_m'] - params['Omega_b']
    return params


if __name__ == '__main__':
    main()


