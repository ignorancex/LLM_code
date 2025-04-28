import numpy as np, sys, os, glob
from scipy.io import readsav
from scipy import interpolate as intrp
import scipy as sc
from scipy import integrate, stats
import ilc
import fisher_tools


h, k, c, temp=6.62607e-34, 1.38065e-23, 2.9979e8, 2.725

################################################################
################################################################
def Jy_Sr_K(band, Jy_K = True, K_Jy = False):
    """
    Conversion from Jy/Sr to Kelvin and vice versa.

    Parameters
    ----------
    band: float
        freqeuncy in GHz.
    Jy_K: bool
        Default is True. Convert from flux to temperature.
    k_Jy: bool
        Default is False. Convert from temperature to flux.

    Returns
    -------
    conv: float
        conversion factor from Jy/Sr or vice versa.
    """
    band *= 1e9

    x=h*band/(k*temp)
    
    dbnu = 2.*k*(band**2/c**2)*(x**2*np.exp(x))/(np.exp(x)-1.)**2.0
    conv=1e26*dbnu #Jy to K

    if Jy_K:
        return 1./conv
    else:
        return conv

def interpolate_dn_ds(s, dnds, increasing_spacing_by = 10):
    """
    interpolate dN/ds to increase the resolution.
    This is particularly important for high flux (S150>=2 mJy) sources.

    Parameters
    ----------
    s: array
        flux bins.
    dnds: array
        source count in each flux bin.
    increasing_spacing_by: int
        interploation factor. Default is x10.
    
    Returns
    -------
    s_ip: array
        interpolated flux bins.
    dnds_ip: array
        interpolated source counts in each flux bin.
    """

    lns = np.log(s)
    dlns =  (lns[1]-lns[0])

    #get a new closely separated range using interpolation
    dlns_ip = dlns / increasing_spacing_by
    #lns_ip = np.logspace(min(lns), max(lns), dlns_ip)
    s_ip = np.exp( np.arange(min(lns), max(lns), dlns_ip) )
    dnds_ip = np.interp(s_ip, s, dnds)
        
    return s_ip, dnds_ip

def get_poisson_source_counts(band, min_flux_mJy = -1., max_flux_mJy = 6.4e-3, band0 = 150., which_dnds = 'lagache', spec_index_radio = -0.76, dnds_ip_factor = 10):

    """
    get radio source population dN/ds for the desired band.

    Parameters
    ----------
    band: float
        freqeuncy in GHz.
    min_flux_mJy: float
        min. flux required in mJy.
        default is -1 --> no minimum threshold.
    max_flux_mJy: float
        min. flux required in mJy.
    band0: float
        default band in which the source count file is defined.
        default is 150 GHz. Does not work for others.
    which_dnds: str
        dn/ds model to be used.
        default in lagache.
        options are lagache and dezotti.
    spec_index_radio: float
        radio spectral index to be used to scale from band0 to other bands.
        default is -0.7 (R21 SPT result).
    dnds_ip_factor: int
        interploation factor for dN/ds. Default is x10.
    
    Returns
    -------
    s: array
        flux array.
    nsources: array
        total source in each flux bin.
    dndlns: array
        logrithminc source counts in each flux bin.
    """
    if which_dnds == 'dezotti':
        assert band0 == 150
        de_zotti_number_counts_file = 'publish/data/counts_150GHz_de_Zotti_radio.res'
        radio_log_s,radio_x,radio_y,radio_logs25n = np.loadtxt(de_zotti_number_counts_file, skiprows = 12, unpack = True)

        #first get the number of sources in each flux range
        s = 10**radio_log_s

        s150_flux = np.copy(s)
        ###print(s); sys.exit()
        #perform scaling
        s = (band/band0)**spec_index_radio * s
        s25n = 10**radio_logs25n
        dnds = s25n / s**2.5

        #20221101 - perform interpolation
        s_ip, dnds_ip = interpolate_dn_ds(s, dnds, increasing_spacing_by = dnds_ip_factor)
        s, dnds = s_ip, dnds_ip
        s150_flux = np.copy(s)
        s = (band/band0)**spec_index_radio * s

        ###print(s[-50:]); sys.exit()

        #get dn/ds and number of sources
        lns = np.log(s)
        dlns = lns[1] - lns[0]
        dndlns = dnds * s
        nsources = dndlns * dlns  # number of sources obtained

    elif which_dnds == 'lagache':
        assert band0 == 150
        lagache_number_counts_file = 'publish/data/lagache_2019_ns150_radio.dat'
        s, dnds = np.loadtxt(lagache_number_counts_file, unpack = True)

        #20221101 - perform interpolation
        s_ip, dnds_ip = interpolate_dn_ds(s, dnds, increasing_spacing_by = dnds_ip_factor)
        s, dnds = s_ip, dnds_ip

        s150_flux = np.copy(s)
        #perform scaling
        s = (band/band0)**spec_index_radio * s
        lns = np.log(s)
        dlns =  (lns[1]-lns[0])
        dndlns = dnds*s
        nsources = dndlns * dlns  # number of sources obtained

    elif which_dnds == 'agora':
        assert band0 == 150
        mdpl2_number_counts_searchstr = 'publish/data/mdpl2_radio/mdpl2_radiocat_len_universemachine_trinity_95_150_220ghz_randflux_datta2018_truncgauss_*'
        ##print(mdpl2_number_counts_searchstr); sys.exit()
        mdpl2_number_counts_flist = sorted( glob.glob( mdpl2_number_counts_searchstr ) )
        band_ind_dic = {90: 3, 95: 3, 150: 6, 220: 9} #total LK + HK sources
        colind = band_ind_dic[band0]
        s_arr, dnds_s2p5_arr = [], []
        for mdpl2_number_counts_fname in mdpl2_number_counts_flist:
            s, dnds_s2p5 = np.loadtxt(mdpl2_number_counts_fname, usecols = [0, colind], unpack = True) #Flux S is in Jy.
            s_arr.append(s)
            dnds_s2p5_arr.append(dnds_s2p5)
        s_arr = np.asarray( s_arr )
        dnds_s2p5_arr = np.asarray( dnds_s2p5_arr )
        s, dnds_s2p5 = np.mean(s_arr, axis = 0), np.mean(dnds_s2p5_arr, axis = 0)
        dnds = dnds_s2p5/s**2.5

        s_ip, dnds_ip = interpolate_dn_ds(s, dnds, increasing_spacing_by = dnds_ip_factor)
        s, dnds = s_ip, dnds_ip
        s150_flux = np.copy(s)
        s = (band/band0)**spec_index_radio * s

        #get dn/ds and number of sources
        lns = np.log(s)
        dlns = lns[1] - lns[0]
        dndlns = dnds * s
        nsources = dndlns * dlns  # number of sources obtained

    #pick the required flux indices
    sinds = np.where((s150_flux>min_flux_mJy) & (s150_flux<=max_flux_mJy))
    s, nsources, dndlns = s[sinds], nsources[sinds], dndlns[sinds]

    return s, nsources, dndlns

def get_poisson_power_with_spectra_index_scatter(band1, band2 = None, min_flux_mJy_band0 = -1., max_flux_mJy_band0 = 6e-3, band0 = 150., which_dnds = 'dezotti', spec_index_radio = -0.76, spec_index_radio_scatter = 0.6, units = 'uk', dnds_ip_factor = 1, max_radio_band = 230.):

    flux_arr_band0, counts_arr_band0, dndlns_arr_band0 = get_poisson_source_counts(band0, min_flux_mJy = min_flux_mJy_band0, max_flux_mJy = max_flux_mJy_band0, band0 = band0, which_dnds = which_dnds, spec_index_radio = spec_index_radio, dnds_ip_factor = dnds_ip_factor)
    flux_arr, counts_arr, dndlns_arr = flux_arr_band0, counts_arr_band0, dndlns_arr_band0

    if band2 is None: band2 = band1

    s_arr = flux_arr
    #print(s_arr); sys.exit()
    lns = np.log(s_arr)
    dlns =  (lns[1]-lns[0])

    ##print(dlns, s_arr, min_flux_mJy_band0, max_flux_mJy_band0, s_arr**2. * dndlns_arr, 'hihihi'); sys.exit()
    if spec_index_radio_scatter == 0.:
        min_alpha, max_alpha = spec_index_radio - 5. * 0.6, spec_index_radio + 5. * 0.6
        alpha_arr_for_int = np.arange(min_alpha, max_alpha, 0.01)
    else:
        min_alpha, max_alpha = spec_index_radio - 5. * spec_index_radio_scatter, spec_index_radio + 5. * spec_index_radio_scatter
        alpha_arr_for_int = np.arange(min_alpha, max_alpha, 0.01)

    int_arr = []
    for s, dndlns in zip(s_arr, dndlns_arr):
        def integrand_flux_dalpha(s, dndlns, dlns, alpha, alpha_mean, alpha_sigma):                
            return s**2.*(dndlns) * dlns * (band1 * band2 / band0**2.)**alpha * stats.norm.pdf(alpha, alpha_mean, alpha_sigma)
            
        int_val = integrate.simps( integrand_flux_dalpha(s, dndlns, dlns, alpha_arr_for_int, spec_index_radio, spec_index_radio_scatter), x=alpha_arr_for_int )
        ##int_val = s**2.*(dndlns) * dlns
        int_arr.append( int_val )

    integral = np.sum(int_arr)

    Jy_to_K_conv = Jy_Sr_K(band1) * Jy_Sr_K(band2)
    #print(Jy_Sr_K(band1)*1e6, Jy_Sr_K(band2)*1e6); sys.exit()
    cl_poisson = integral * Jy_to_K_conv

    if units == 'uk': cl_poisson *= 1e12

    if band1 > max_radio_band or band2 > max_radio_band: cl_poisson *= 0.

    return cl_poisson

def get_radio_ilc_residuals(els, freqarr, wl1, wl2, which_dnds_arr, min_flux_mJy=0.1e-3, max_flux_mJy=4e-3, spec_index_radio = -0.76, spec_index_radio_scatter_arr = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01], dnds_ip_factor = 10, quiet = False):
    radio_cl_dic = {}
    for which_dnds in which_dnds_arr:
        if not quiet: print('\tdN/ds = %s' %(which_dnds))
        radio_cl_dic[which_dnds] = {}
        for spec_index_radio_scatter in spec_index_radio_scatter_arr:
            if not quiet: print('\t\talpha = %.3f; scatter = %.3f' %(spec_index_radio, spec_index_radio_scatter))
            radio_cl_dic[which_dnds][spec_index_radio_scatter]={}
            radio_cl_dic[which_dnds][spec_index_radio_scatter]['cl_dic'] = {}
            for freq1 in freqarr:
                for freq2 in freqarr:
                    cl_analytic = get_poisson_power_with_spectra_index_scatter(freq1, band2 = freq2, min_flux_mJy_band0 = min_flux_mJy, max_flux_mJy_band0 = max_flux_mJy, which_dnds = which_dnds, spec_index_radio = spec_index_radio, spec_index_radio_scatter = spec_index_radio_scatter, dnds_ip_factor = dnds_ip_factor)
                    cl_analytic = np.tile(cl_analytic, len(els) )
                    radio_cl_dic[which_dnds][spec_index_radio_scatter]['cl_dic'][(freq1, freq2)] = cl_analytic
            #print(radio_cl_dic[which_dnds][spec_index_radio_scatter].keys())


            tmp_dic_dic = {}
            tmp_dic_dic['TT'] = radio_cl_dic[which_dnds][spec_index_radio_scatter]['cl_dic']
            res_ilc = ilc.get_ilc_residual_using_weights(tmp_dic_dic, wl1, freqarr, wl2 = wl2, el = els)
            radio_cl_dic[which_dnds][spec_index_radio_scatter]['res_ilc'] = res_ilc

    return radio_cl_dic

##############

def get_cmb_spectra(els = None, which_spectra = 'lensed', planck_cosmo_version = 2018, quiet = False):

    camb_folder = 'publish/data/CAMB/planck_%s/' %(planck_cosmo_version)

    #CMB stuffs
    #get fiducual LCDM power spectra computed using CAMB
    if not quiet:
        print('get fiducual LCDM power spectra computed using CAMB')
    camb_fname = '%s/cmb_spectra_%s.txt' %(camb_folder, which_spectra)
    cl_camb = np.loadtxt(camb_fname)
    el_camb = cl_camb[:,0] 
    cl_camb_tt = cl_camb[:,1]
    dl_fac_camb = el_camb * (el_camb+1)/2/np.pi
    cl_dic = {}
    cl_dic['TT'] = cl_camb[:,1]
    cl_dic['EE'] = cl_camb[:,2]
    cl_dic['TE'] = cl_camb[:,3]

    if els is None:
        els = np.copy(el_camb)

    #get derivatives of CMB power spectrum for different LCDM parameters.
    #They are already computed and stored.
    if not quiet:
        print('get/read derivatives')
    camb_deriv_fname = '%s/cmb_spectra_derivs_%s.npy' %(camb_folder, which_spectra)
    cl_deriv_dic_tmp = np.load(camb_deriv_fname, allow_pickle = 1).item()
    cl_deriv_dic = {}
    param_names = []
    for p in sorted( cl_deriv_dic_tmp ):
        if p == 'ell': continue    
        cl_deriv_dic[p]={}
        if planck_cosmo_version == 2018:
            cl_deriv_dic[p]['TT'] = cl_deriv_dic_tmp[p]['TT']
            cl_deriv_dic[p]['EE'] = cl_deriv_dic_tmp[p]['EE']
            cl_deriv_dic[p]['TE'] = cl_deriv_dic_tmp[p]['TE']
        else:
            cl_deriv_dic[p]['TT'] = cl_deriv_dic_tmp[p][0]
            cl_deriv_dic[p]['EE'] = cl_deriv_dic_tmp[p][1]
            cl_deriv_dic[p]['TE'] = cl_deriv_dic_tmp[p][2]
        
        if p == 'As' and planck_cosmo_version == 2015:
            cl_deriv_dic[p]['TT'] *= 1e9
            cl_deriv_dic[p]['EE'] *= 1e9
            cl_deriv_dic[p]['TE'] *= 1e9
            
        param_names.append( p )
        
    #interpolate CAMB spectra / derivatives on the desired els.
    for which_spec in cl_dic:
        cl_dic[which_spec] = np.interp(els, el_camb, cl_dic[which_spec], right=0.)
    for p in sorted( param_names ):
        for which_spec in cl_deriv_dic[p]:
            cl_deriv_dic[p][which_spec] = np.interp(els, el_camb, cl_deriv_dic[p][which_spec], right=0.)
            
    return cl_dic, cl_deriv_dic

def get_ilc_stuffs(expname, which_fg_model = 'agora'):

    #ILC files
    fname = 'publish/ilc/ilc_weights_residuals_%s_fg_model.npy' %(which_fg_model)
    fname_withccatp = 'publish/ilc/ilc_weights_residuals_%s_fg_model_withccatp.npy' %(which_fg_model)

    ilc_dict = np.load(fname, allow_pickle = True).item()
    ilc_dict_withccatp = np.load(fname_withccatp, allow_pickle = True).item()
    
    #total ILC residuals
    total_ilc_residuals_dict = ilc_dict['total_ilc_residuals']
    total_ilc_residuals_dict_withccatp = ilc_dict_withccatp['total_ilc_residuals']
    #print(total_ilc_residuals_dict.keys())

    #get experiment Nl_dict
    nl_TP_dict = ilc_dict['nl_TP_dict']
    nl_TP_dict_withccatp = ilc_dict_withccatp['nl_TP_dict']

    #weights
    weights_dict = ilc_dict['weights']
    weights_dict_withccatp = ilc_dict_withccatp['weights']


    return ilc_dict, ilc_dict_withccatp, total_ilc_residuals_dict, total_ilc_residuals_dict_withccatp, nl_TP_dict, nl_TP_dict_withccatp, weights_dict, weights_dict_withccatp

def get_temperature_ilc_residuals(expname, reqd_ilc_keyname_1, reqd_ilc_keyname_2, quiet = False, which_fg_model = 'agora'): #Temperature ILC residuals for Nl

    ilc_dict, ilc_dict_withccatp, total_ilc_residuals_dict, total_ilc_residuals_dict_withccatp, nl_TP_dict, nl_TP_dict_withccatp, weights_dict, weights_dict_withccatp = get_ilc_stuffs(expname, which_fg_model = which_fg_model)

    if not quiet:
        print('\tget ILC residuals for Nl')
    if expname.find('withccatp')>-1:
        dict_to_use = total_ilc_residuals_dict_withccatp
        expname_to_use = expname.replace('_withccatp', '')
    else:
        dict_to_use = total_ilc_residuals_dict
        expname_to_use = expname
    if reqd_ilc_keyname_1 != reqd_ilc_keyname_2:
        reqd_ilc_keyname_12 = '%sx%s' %(reqd_ilc_keyname_1, reqd_ilc_keyname_2)
    else:
        reqd_ilc_keyname_12 = reqd_ilc_keyname_1
    els, total_ilc_residual_mv = dict_to_use[expname_to_use]['mv']
    els, total_ilc_residual_1 = dict_to_use[expname_to_use][reqd_ilc_keyname_1]
    els, total_ilc_residual_2 = dict_to_use[expname_to_use][reqd_ilc_keyname_2]
    els, total_ilc_residual_12 = dict_to_use[expname_to_use][reqd_ilc_keyname_12]
    
    return els, total_ilc_residual_mv, total_ilc_residual_1, total_ilc_residual_2, total_ilc_residual_12    

def get_polarisation_mvilc_residuals(els, expname, which_fg_model = 'agora'):
    #get MV-ILC for polarisation
    """
    Note that foregrounds are assumed to be unpolarised here.
    So this should simply be a MV noise estimate after taking beams into account.
    """

    ilc_dict, ilc_dict_withccatp, total_ilc_residuals_dict, total_ilc_residuals_dict_withccatp, nl_TP_dict, nl_TP_dict_withccatp, weights_dict, weights_dict_withccatp = get_ilc_stuffs(expname, which_fg_model = which_fg_model)

    dict_for_ilc = {}
    if expname.find('withccatp')>-1:
        expname_to_use = expname.replace('_withccatp', '')
        dict_for_ilc['EE'] = nl_TP_dict_withccatp[expname_to_use]['P']
    else:
        dict_for_ilc['EE'] = nl_TP_dict[expname]['P']
    bands = get_exp_bands(expname)
    mvilc_pol_residuals, mvilc_pol_weights = ilc.get_mvilc_residual_and_weights(bands, els, dict_for_ilc)
    mvilc_pol_residuals = mvilc_pol_residuals[0]

    if (0):#expname == 's4_wide': #show plot for MV-ILC for pol and compare that will noise.
        clf()
        fsval = 14
        band_color_dict = {95: 'navy', 150: 'darkgreen', 220: 'goldenrod', 285: 'orangered', 345: 'darkred', 
                            '410': 'hotpink', 850: 'black'}
        ax=subplot(111, yscale = 'log')
        noise_arr = []
        for (nu1, nu2) in dict_for_ilc['EE']:
            if nu1 != nu2: continue
            curr_nl = dict_for_ilc['EE'][(nu1, nu2)]

            plot(els, curr_nl, color = band_color_dict[nu1], label = r'%s GHz' %(nu1))
            noise_arr.append(curr_nl)

        #MV-ILC for pol
        plot(els, mvilc_pol_residuals, color = 'black', label = r'MV-ILC')

        if (0): #simple MV noise for pol as a sanity check.
            noise_arr = np.asarray(noise_arr)
            mv_noise_pol = ( np.sum(noise_arr**-2, axis = 0) )**-0.5
            plot(els, mv_noise_pol, color = 'hotpink', lw = 2., ls = '-.', label = r'MV noise for pol.')

        legend(loc = 1, fontsize = fsval - 6, ncol = 5)
        xlabel(r'Multipole $\ell$', fontsize = fsval)
        ylabel(r'Polarisation noise: $C_{\ell}$ [$\mu$K$^{2}$]', fontsize = fsval-2)
        xlim(0., 7000.); ylim(1e-7, .01)
        title_str = r'%s: Polarisation noise + MV-ILC' %(exp_specs_dict[expname][0])
        title(title_str, fontsize = fsval)
        show()
        
    return els, mvilc_pol_residuals

def get_exp_bands(expname):
    if expname in ['s4_wide', 's4_deep', 'so_baseline', 'so_goal']:        
        bands = [95, 150, 220, 285]
    #elif expname == 'spt3g':
    elif expname.lower() in ['spt3g', 'sptsummer', 'sptext6k']:
        bands = [95, 150, 220]#, 600, 857]
    elif expname == 'spt4':
        bands = [95, 150, 220, 285, 345]
    elif expname in ['s4_wide_withccatp', 'so_baseline_withccatp', 'so_goal_withccatp']: 
        bands = [95, 150, 220, 285, 345, 410, 850]
        
    return bands

def wrapper_get_delta_cl(els, cl_dic, fsky, 
                            total_ilc_residual_1, total_ilc_residual_2, total_ilc_residual_12, 
                            total_ilc_residual_mv, mvilc_pol_residuals,
                           ):    
    #get delta_Cl using Knox formula.
    nl11_dic = {}
    nl11_dic['TT'] = total_ilc_residual_1
    nl11_dic['EE'] = mvilc_pol_residuals
    nl11_dic['TE'] = np.copy(total_ilc_residual_1) * 0.

    nl22_dic = {}
    nl22_dic['TT'] = total_ilc_residual_2
    nl22_dic['EE'] = mvilc_pol_residuals
    nl22_dic['TE'] = np.copy(total_ilc_residual_2) * 0.

    nl12_dic = {}
    nl12_dic['TT'] = total_ilc_residual_12
    nl12_dic['EE'] = mvilc_pol_residuals
    nl12_dic['TE'] = np.copy(total_ilc_residual_12) * 0.

    delta_cl_dic = fisher_tools.get_knox_errors_parent(els, cl_dic, nl11_dic, fsky, nl22_dic = nl22_dic, nl12_dic = nl12_dic)
    #print(delta_cl_dic.keys())

    if (1): #MV ILC
        nl_mv_dic = {}
        nl_mv_dic['TT'] = total_ilc_residual_mv
        nl_mv_dic['EE'] = mvilc_pol_residuals
        nl_mv_dic['TE'] = np.copy(total_ilc_residual_mv) * 0.
        delta_cl_dic_mv = fisher_tools.get_knox_errors_parent(els, cl_dic, nl_mv_dic, fsky)
        #print(delta_cl_dic_mv.keys()) 
    
    return delta_cl_dic, delta_cl_dic_mv

#radio stuffs
def get_radio_residuals(els, expname, reqd_ilc_keyname_1, reqd_ilc_keyname_2, which_dnds = 'lagache', 
                        min_flux_mJy=0.1e-3, max_flux_mJy = 3e-3, 
                        spec_index_radio = -0.76, 
                        spec_index_radio_scatter = 0.2,
                        quiet = True): 

    ilc_dict, ilc_dict_withccatp, total_ilc_residuals_dict, total_ilc_residuals_dict_withccatp, nl_TP_dict, nl_TP_dict_withccatp, weights_dict, weights_dict_withccatp = get_ilc_stuffs(expname)

    #get radio spectrum now for a given masking threshold.
    if expname.find('withccatp')>-1:
        expname_to_use = expname.replace('_withccatp', '')        
        w1, w2 = weights_dict_withccatp[expname_to_use][reqd_ilc_keyname_1], weights_dict_withccatp[expname_to_use][reqd_ilc_keyname_2] #weights for the two ILC maps.
    else:
        expname_to_use = expname
        w1, w2 = weights_dict[expname_to_use][reqd_ilc_keyname_1], weights_dict[expname_to_use][reqd_ilc_keyname_2] #weights for the two ILC maps.
    bands = get_exp_bands(expname)
    ##print(bands, len(bands), w1.shape, w2.shape); sys.exit()
    radio_cl_dict = get_radio_ilc_residuals(els, bands, w1, w2, [which_dnds], 
                                                  min_flux_mJy = min_flux_mJy, max_flux_mJy = max_flux_mJy, 
                                                  spec_index_radio = spec_index_radio, 
                                                  spec_index_radio_scatter_arr = [spec_index_radio_scatter],
                                                 quiet = quiet)

    res_cl_radio = radio_cl_dict[which_dnds][spec_index_radio_scatter]['res_ilc']
    
    return res_cl_radio

def get_radio_spectrum_and_derivatives(els, expname, cl_dic, cl_deriv_dic, reqd_ilc_keyname_1, reqd_ilc_keyname_2 = None, which_dnds = 'lagache', 
                        min_flux_mJy=0.1e-3, max_flux_mJy = 3e-3, 
                        spec_index_radio = -0.76, 
                        spec_index_radio_scatter = 0.2, 
                        spec_index_radio_step = 0.002,
                        spec_index_radio_scatter_step = 0.002,
                        quiet = True):
    

    #get radio spectrum
    if not quiet: print('get radio spectrum')
    cl_radio = get_radio_residuals(els, expname, reqd_ilc_keyname_1, reqd_ilc_keyname_2, max_flux_mJy = max_flux_mJy, spec_index_radio = spec_index_radio, spec_index_radio_scatter = spec_index_radio_scatter, quiet=quiet)
    #add radio to cl_dic['TT']
    ##cl_dic['TT'] += cl_radio## - already included in total residuals.

    #get deriviatives for spec_index_radio
    if not quiet: print('get deriviatives for alpha_radio')
    cl_radio_alpha_radio_low = get_radio_residuals(els, expname, reqd_ilc_keyname_1, reqd_ilc_keyname_2, max_flux_mJy = max_flux_mJy, spec_index_radio = spec_index_radio - spec_index_radio_step, spec_index_radio_scatter = spec_index_radio_scatter, quiet=quiet)
    cl_radio_alpha_radio_high = get_radio_residuals(els, expname, reqd_ilc_keyname_1, reqd_ilc_keyname_2, max_flux_mJy = max_flux_mJy, spec_index_radio = spec_index_radio + spec_index_radio_step, spec_index_radio_scatter = spec_index_radio_scatter, quiet=quiet)
    cl_deriv_dic['alpha_radio'] = {}
    cl_deriv_dic['alpha_radio']['TT'] = (cl_radio_alpha_radio_high-cl_radio_alpha_radio_low)/( 2 * spec_index_radio_step)
    cl_deriv_dic['alpha_radio']['EE'] = cl_deriv_dic['alpha_radio']['TT'] * 0.
    cl_deriv_dic['alpha_radio']['TE'] = cl_deriv_dic['alpha_radio']['TT'] * 0.

    #get deriviatives for spec_index_radio_scatter
    if not quiet: print('get deriviatives for alpha_radio_sigma')
    cl_radio_alpha_radio_sigma_low = get_radio_residuals(els, expname, reqd_ilc_keyname_1, reqd_ilc_keyname_2, max_flux_mJy = max_flux_mJy, 
                                                         spec_index_radio = spec_index_radio, 
                                                         spec_index_radio_scatter = spec_index_radio_scatter - spec_index_radio_scatter_step,
                                                        quiet=quiet)
    cl_radio_alpha_radio_sigma_high = get_radio_residuals(els, expname, reqd_ilc_keyname_1, reqd_ilc_keyname_2, max_flux_mJy = max_flux_mJy, 
                                                          spec_index_radio = spec_index_radio, 
                                                          spec_index_radio_scatter = spec_index_radio_scatter + spec_index_radio_scatter_step,
                                                         quiet=quiet)
    cl_deriv_dic['alpha_radio_sigma'] = {}
    cl_deriv_dic['alpha_radio_sigma']['TT'] = (cl_radio_alpha_radio_sigma_high-cl_radio_alpha_radio_sigma_low)/( 2 * spec_index_radio_scatter_step)
    cl_deriv_dic['alpha_radio_sigma']['EE'] = cl_deriv_dic['alpha_radio_sigma']['TT'] * 0.
    cl_deriv_dic['alpha_radio_sigma']['TE'] = cl_deriv_dic['alpha_radio_sigma']['TT'] * 0.
    
    return cl_radio, cl_dic, cl_deriv_dic

#CIB+tSZ residuals
def get_cib_tsz_spectrum_and_derivatives(els, expname, cl_dic, cl_deriv_dic, reqd_ilc_keyname_1, reqd_ilc_keyname_2 = None, Acibtsz = 1., Acibtsz_step = None):

    ilc_dict, ilc_dict_withccatp, total_ilc_residuals_dict, total_ilc_residuals_dict_withccatp, nl_TP_dict, nl_TP_dict_withccatp, weights_dict, weights_dict_withccatp = get_ilc_stuffs(expname)
    if reqd_ilc_keyname_1 != reqd_ilc_keyname_2:
        reqd_ilc_keyname_str = '%sx%s' %(reqd_ilc_keyname_1, reqd_ilc_keyname_2)
    else:
        reqd_ilc_keyname_str = reqd_ilc_keyname_1
    cl_cibtsz_ori = ilc_dict['cib_plus_tsz_residuals']['s4_wide'][reqd_ilc_keyname_str]
    cl_cibtsz = Acibtsz * np.copy( cl_cibtsz_ori )
    #add CIB+tSZ to cl_dic['TT']
    ##cl_dic['TT'] += cl_cibtsz## - already included in total residuals.
    
    #now get derivatives
    if Acibtsz_step == None: Acibtsz_step = Acibtsz / 100.
        
    cl_cibtsz_low = np.copy( cl_cibtsz_ori ) * ( Acibtsz - Acibtsz_step)
    cl_cibtsz_high = np.copy( cl_cibtsz_ori ) * ( Acibtsz + Acibtsz_step)
    cl_deriv_dic['Acibtsz'] = {}
    cl_deriv_dic['Acibtsz']['TT'] = (cl_cibtsz_high-cl_cibtsz_low)/( 2 * Acibtsz_step)
    cl_deriv_dic['Acibtsz']['EE'] = cl_deriv_dic['Acibtsz']['TT'] * 0.
    cl_deriv_dic['Acibtsz']['TE'] = cl_deriv_dic['Acibtsz']['TT'] * 0.
        
    return cl_cibtsz, cl_dic, cl_deriv_dic

##############
def fn_format_axis(ax,fx,fy):
    for label in ax.get_xticklabels(): label.set_fontsize(fx)
    for label in ax.get_yticklabels(): label.set_fontsize(fy)
    if (0):
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.xaxis.set_minor_locator(ticker.MaNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaNLocator(4))
        ax.yaxis.set_minor_locator(ticker.MaNLocator(4))
    return ax    
