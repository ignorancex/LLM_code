import healpy as hp
import numpy as np
from utils import tsz_spectral_response


def generate_freq_maps(inp, save=True, band_limit=False, same_noise=True):

    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    save: Bool, whether to save frequency map files
    band_limit: Bool, whether or not to remove all power in weight maps above ellmax
    same_noise: Bool, whether to use the same or different white noise level in the two frequently channels
            (currently, if False, sets the noise in the higher frequency channel to be slightly higher)

    RETURNS
    -------
    power spectra of CMB, tSZ, and noise (CC, T, N)
    cmb_map, tsz_map, noise1_map, noise2_map (amplified depending on scaling)

    '''
    l_arr, m_arr = hp.Alm.getlm(3*inp.nside-1)

    #Read tSZ halosky map
    tsz_map = hp.read_map(inp.tsz_map_file)
    tsz_map = inp.tsz_amp*hp.ud_grade(tsz_map, inp.nside)
    if band_limit:
        tsz_alm = hp.map2alm(tsz_map)
        tsz_alm = tsz_alm*(l_arr<=inp.ellmax)
        tsz_map = hp.alm2map(tsz_alm, nside=inp.nside)
    tsz_cl = hp.anafast(tsz_map, lmax=inp.ell_sum_max)

    #realization of CMB from lensed alm
    cmb_map = hp.read_map(inp.cmb_map_file)
    cmb_map = hp.ud_grade(cmb_map, inp.nside)
    if band_limit:
        cmb_alm = hp.map2alm(cmb_map)
        cmb_alm = cmb_alm*(l_arr<=inp.ellmax)
        cmb_map = hp.alm2map(cmb_alm, nside=inp.nside)
    cmb_cl = hp.anafast(cmb_map, lmax=inp.ell_sum_max)

    #noise map realizations
    theta_fwhm = (1.4/60.)*(np.pi/180.)
    sigma = theta_fwhm/np.sqrt(8.*np.log(2.))
    W1 = (inp.noise/60.)*(np.pi/180.)
    if same_noise:
        W2 = W1
    else:
        W2 = (inp.noise*np.sqrt(1.5)/60.)*(np.pi/180.)
    ells = np.arange(3*inp.nside)
    noise1_cl = W1**2*np.exp(ells*(ells+1)*sigma**2)*10**(-12)
    noise2_cl = W2**2*np.exp(ells*(ells+1)*sigma**2)*10**(-12)
    np.random.seed(0)
    noise1_map = hp.synfast(noise1_cl, inp.nside)
    np.random.seed(1)
    noise2_map = hp.synfast(noise2_cl, inp.nside)
    if band_limit:
        noise1_alm = hp.map2alm(noise1_map)
        noise1_alm = noise1_alm*(l_arr<=inp.ellmax)
        noise1_map = hp.alm2map(noise1_alm, nside=inp.nside)
        noise2_alm = hp.map2alm(noise2_map)
        noise2_alm = noise2_alm*(l_arr<=inp.ellmax)
        noise2_map = hp.alm2map(noise2_alm, nside=inp.nside)
    noise1_cl = hp.anafast(noise1_map, lmax=inp.ell_sum_max)
    noise2_cl = hp.anafast(noise2_map, lmax=inp.ell_sum_max)

    #tSZ spectral response
    g1, g2 = tsz_spectral_response(inp.freqs)

    #create maps at freq1 and freq2 (in GHz)
    sim_map_1 = cmb_map + g1*tsz_map + noise1_map
    sim_map_2 = cmb_map + g2*tsz_map + noise2_map #make noise different in both maps
    if save:
        map1_fname = f'{inp.output_dir}/maps/freq1.fits'
        map2_fname = f'{inp.output_dir}/maps/freq2.fits'
        hp.write_map(map1_fname, sim_map_1, overwrite=True, dtype=np.float32)
        hp.write_map(map2_fname, sim_map_2, overwrite=True, dtype=np.float32)
        if inp.verbose:
            print(f'created {map1_fname} and {map2_fname}', flush=True)

    return cmb_cl, tsz_cl, noise1_cl, noise2_cl, cmb_map, tsz_map, noise1_map, noise2_map
