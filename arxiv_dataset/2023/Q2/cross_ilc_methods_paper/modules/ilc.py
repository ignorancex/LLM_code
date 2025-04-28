import numpy as np, sys, os, glob
from scipy.io import readsav
from scipy import interpolate as intrp
import scipy as sc
from scipy import integrate, stats

def get_teb_spec_combination(cl_dict):

    """
    uses cl_dict to determine if we are using ILC jointly for T/E/B.

    Parameters
    ----------
    cl_dict : dict
        dictionary containing (signal+noise) auto- and cross- spectra of different freq. channels.

    Returns
    -------
    nspecs : int
        tells if we are performing ILC for T alone or T/E/B together.
        default is 1. For only one map component.

    specs : list
        creates ['TT', 'EE', 'TE', ... etc.] based on cl_dict that is supplied.
        For example:
        ['TT'] = ILC for T-only
        ['EE'] = ILC for E-only
        ['TT', 'EE'] = ILC for T and E separately.
        ['TT', 'EE', 'TE'] = ILC for T and E jointly.
    """

    # fix-me. Do this in a better way.
    specs = sorted(list(cl_dict.keys()))

    if specs == ['TT'] or specs == ['EE'] or specs == ['BB']:  # only TT is supplied
        nspecs = 1
    elif specs == sorted(['TT', 'EE']) or specs == sorted(
        ['TT', 'EE', 'TE']
    ):  # TT/EE/TE are supplied
        nspecs = 2
    elif specs == sorted(['TT', 'EE', 'BB']) or specs == sorted(
        ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
    ):  # TT/EE/BB are supplied
        nspecs = 3
    else:
        logline = 'cl_dict must contain TT/EE/BB spectra or some combination of that'
        raise ValueError(logline)

    return nspecs, specs

def create_covariance(bands, elcnt, cl_dict):

    """
    Creates band-band covariance matrix at each el

    Parameters
    ----------
    bands : array
        array of frequency bands for which we need the covariance.
    elcnt : int
        ell index.
    cl_dict : dict
        dictionary containing (signal+noise) auto- and cross- spectra of different freq. channels.

    Returns
    -------
    cov: array
        band-band covariance matrix at each ell. dimension is nband x nband.
    """

    nc = len(bands)
    nspecs, specs = get_teb_spec_combination(cl_dict)
    cov = np.zeros((nspecs * nc, nspecs * nc))

    for specind, spec in enumerate(specs):
        curr_cl_dict = cl_dict[spec]
        if nspecs == 1:  # cov for TT or EE or BB
            for ncnt1, band1 in enumerate(bands):
                for ncnt2, band2 in enumerate(bands):
                    j, i = ncnt2, ncnt1
                    cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
        else:  # joint or separate TT/EE constraints #fix me: include BB for joint constraints.
            if spec == 'TT':
                for ncnt1, band1 in enumerate(bands):
                    for ncnt2, band2 in enumerate(bands):
                        j, i = ncnt2, ncnt1
                        cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
            elif spec == 'EE':
                for ncnt1, band1 in enumerate(bands):
                    for ncnt2, band2 in enumerate(bands):
                        j, i = ncnt2 + nc, ncnt1 + nc
                        cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
            elif spec == 'TE':
                for ncnt1, band1 in enumerate(bands):
                    for ncnt2, band2 in enumerate(bands):
                        j, i = ncnt2 + nc, ncnt1
                        cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
                        cov[i, j] = curr_cl_dict[(band1, band2)][elcnt]

    return cov

def get_frequency_response(bands, final_comp = 'cmb', spec_index_rg = -0.76, freqcalib_fac = None, nspecs = 1):

    """
    get frequency dependence of a sky signal.

    Parameters
    ----------
    bands : array
        array of frequency bands for which we need the covariance.

    component : str
        It can be
        'cmb' for CMB spectra
        'tsz' or 'y' for thermal SZ
        'radio' for radio galaxies (see get_radio_freq_dep() function)
        'cib' or 'cibpo' for Poisson component of the CIB.
        'cibclus' for clustered component of the CIB.
        'misc_cib_tcibxx_betayy' for misc CIB with T_d=xx and beta=yy.

    spec_index_rg: float
        Spectral index to be used for radio.
        Default is -0.76 (R21 value).

    freq_cal : array
        array containing calibration factors / mis-matches between different bands.
        default is None which is equivalent to [1, 1, 1, ... 1] in all bands.

    nspecs : int
        tells if we are performing ILC for T alone or T/E/B together.
        default is 1. For only one map component.

    Returns
    -------
    avec : array
        freq. dependene of the respective sky component.
        for example: CMB will be  [1., 1., ...., 1.] in all bands.
    """


    nc = len(bands)

    if freqcalib_fac is None: freqcalib_fac = np.ones(nc)

    if final_comp.lower() == 'cmb':
        freqscale_fac = np.ones(nc)

    elif final_comp.lower() == 'tsz' or final_comp.lower() == 'y':

        freqscale_fac = []
        for freq in sorted( bands ):
            freqscale_fac.append( compton_y_to_delta_Tcmb(freq * 1e9) )

        freqscale_fac = np.asarray( freqscale_fac )

    elif final_comp.lower() == 'cib' or final_comp.lower() == 'cibpo':
        freqscale_fac = []
        for freq in sorted( bands ):
            freqscale_fac.append( get_cib_freq_dep(freq * 1e9) )

        freqscale_fac = np.asarray( freqscale_fac )
        freqscale_fac /= np.max(freqscale_fac)
        
    elif final_comp.lower() == 'cibclus':
        freqscale_fac = []
        for freq in sorted( bands ):
            freqscale_fac.append( get_cib_freq_dep(freq * 1e9, beta = 2.505) )

        freqscale_fac = np.asarray( freqscale_fac )
        freqscale_fac /= np.max(freqscale_fac)

    elif final_comp.lower().find('misc_cib')>-1:

        #default values
        misc_tcib = 20.
        misc_beta = 1.505
        tcib_tmp =  re.findall('tcib\d*\.?\d+', final_comp.lower())
        if len(tcib_tmp)>0:
            tcib_tmp = tcib_tmp[0]
            misc_tcib = float(tcib_tmp.replace('tcib', ''))

        beta_tmp =  re.findall('beta\d*\.?\d+', final_comp.lower())
        if len(beta_tmp)>0:
            beta_tmp = beta_tmp[0]
            misc_beta = float(beta_tmp.replace('beta', ''))

        #bands = [30, 44, 70, 100, 150, 217, 353, 545]
        freqscale_fac = []
        for freq in sorted( bands ):
            freqscale_fac.append( get_cib_freq_dep(freq * 1e9, Tcib = misc_tcib, beta = misc_beta) )

        freqscale_fac = np.asarray( freqscale_fac )
        freqscale_fac /= np.max(freqscale_fac)

    elif final_comp.lower() == 'radio':
        freqscale_fac = []
        for freq in sorted( bands ):
            freqscale_fac.append( get_radio_freq_dep(freq, spec_index_rg = spec_index_rg) )

        freqscale_fac = np.asarray( freqscale_fac )

    acap = np.zeros(nc) + (freqscale_fac * freqcalib_fac) #assuming CMB is the same and calibrations factors are same for all channel

    if nspecs>1:
        acap_full = np.zeros( (nspecs, len(acap) * nspecs) )
        acap_full[0,:len(acap)] = acap
        if final_comp.lower() == 'cmb':
            acap_full[1,len(acap):] = acap
        else: #polarisation weights are zero for other foregrounds
            acap_full[1,len(acap):] = 0.

        acap_full = np.mat(acap_full).T #should be nspecs*nc x nspecs
        acap = acap_full
    else:
        acap = np.mat(acap).T #should be nspecs*nc x nspecs
    
    return acap

def get_mvilc_residual_and_weights(bands, el, cl_dict, final_comp = 'cmb', freqcalib_fac = None, lmin = 10):

    """
    get MV ILC residuals and weights

    Parameters
    ----------
    bands: array
        array of frequency bands for which we need the covariance.

    el: array
        Multipoles over which cl_dict is defined.

    cl_dict : dict
        dictionary containing signal auto- and cross- spectra of different freq. bands.

    final_comp: str
        It can be
        'cmb' for CMB spectra
        'tsz' or 'y' for thermal SZ
        'radio' for radio galaxies (see get_radio_freq_dep() function)
        'cib' or 'cibpo' for Poisson component of the CIB.
        'cibclus' for clustered component of the CIB.
        'misc_cib_tcibxx_betayy' for misc CIB with T_d=xx and beta=yy.

    freqcalib_fac : array
        array containing calibration factors / mis-matches between different bands.
        default is None which is equivalent to [1, 1, 1, ... 1] in all bands.

    lmin : int
        Minimum multipole to be used.
        Default is 10.

    Returns
    -------
    avec : array
        freq. dependene of the respective sky component.
        for example: CMB will be  [1., 1., ...., 1.] in all bands.
    """

    nspecs, specs = get_teb_spec_combination(cl_dict)
    acap = get_frequency_response(bands, final_comp = final_comp, freqcalib_fac = freqcalib_fac, nspecs = nspecs)

    nc = len(bands)
    weightsarr = np.zeros( (nspecs * nc, nspecs, len( el ) ) )
    cl_residual = np.zeros( (3, len(el)) )

    cl_residual_tmp = []

    for elcnt, currel in enumerate(el):
        if currel <= lmin: continue ## or el>=lmax: continue
        clmat = create_covariance(bands, elcnt, cl_dict) 
        clinv = np.linalg.pinv(clmat)

        nr = np.dot(clinv, acap)
        dr = np.dot( acap.T, np.dot(clinv, acap) )
        #drinv = sc.linalg.pinv2(dr)
        drinv = np.linalg.pinv(dr)

        weight = np.dot(nr, drinv)

        #ILC residuals
        if nspecs>1:
            cl_residual_tt, cl_residual_ee, cl_residual_te = drinv[0,0], drinv[1,1], drinv[0,1]
            cl_residual[:, elcnt] = cl_residual_tt, cl_residual_ee, cl_residual_te
        else:
            cl_residual[0, elcnt] = drinv[0]

        weightsarr[:, :, elcnt] = weight

        cl_residual_tmp.append( drinv )

    weightsarr = np.asarray( weightsarr )
    cl_residual = np.asarray( cl_residual )

    cl_residual[np.isinf(cl_residual)] = 0.
    cl_residual[np.isnan(cl_residual)] = 0.
    
    return cl_residual, weightsarr

def get_ilc_residual_using_weights(cl_dic, wl, bands, wl2 = None, lmax = 10000, el = None):

    """
    get ILC residuals for a given compnent given the freqeuency dependent weights.
    If wl2 is None, then this function returns the auto-ILC residuals.

    Parameters
    ----------
    cl_dict : dict
        dictionary containing signal auto- and cross- spectra of the component in 
        different freq. bands.

    wl: array
        freqeuency dependent weights.

    bands: array
        array of frequency bands.

    wl2: array
        same as wl but for a different ILC.
        Default is None.

    lmax: int
        Maximum multipole for computation.
        Default is 10000.

    el: array
        Multipole array. 
        Default is None.

    Returns
    -------
    avec : array
        freq. dependene of the respective sky component.
        for example: CMB will be  [1., 1., ...., 1.] in all bands.
    """

    if wl2 is None:
        wl2 = wl
    if el is None:
        el = np.arange(lmax)
    res_ilc = []
    for elcnt, currel in enumerate(el):
        clmat = np.mat( create_covariance(bands, elcnt, cl_dic) )
        currw_ilc1 = np.mat( wl[:, elcnt] )
        currw_ilc2 = np.mat( wl2[:, elcnt] )
        curr_res_ilc = np.asarray(np.dot(currw_ilc1, np.dot(clmat, currw_ilc2.T)))[0][0]
        res_ilc.append( curr_res_ilc )

    res_ilc = np.asarray(res_ilc)
    res_ilc[np.isnan(res_ilc)] = 0.
    res_ilc[np.isinf(res_ilc)] = 0.

    return res_ilc
