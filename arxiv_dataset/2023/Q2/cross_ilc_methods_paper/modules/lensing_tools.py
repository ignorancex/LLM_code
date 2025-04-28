import numpy as np, os, glob, sys
import fisher_tools
from pylab import *

lensing_xcorr_folder = 'publish/lensing/lensing_xls/' #lensing bias files for S4-Wide.
lensing_n0_folder = 'publish/lensing/lensing_n0/'

def ngal_in_zbin(reqd_zbin, units = 'sr'):
    ngal_dic = {'zbin1': 1.78, 'zbin2': 1.77, 'zbin3': 1.78, 'zbin4': 1.78, 'zbin5': 1.78} #per arcmin^2
    ngal_per_arcmin2 = ngal_dic[reqd_zbin]
    ngal_per_sr = ngal_per_arcmin2 / np.radians(1./60.)**2.
    if units == 'sr':
        return ngal_per_sr
    elif units == 'arcmin2':
        return ngal_per_arcmin2

def get_shot_noise(reqd_zbin, els = None):
    ngal_per_sr = ngal_in_zbin(reqd_zbin)
    shot_noise = 1./ngal_per_sr
    if els is not None:
        shot_noise = np.tile(shot_noise, len(els))
    return shot_noise

def lensing_estimator_folder_str(expname, estimator, lmaxtt = 3000):

    estimator_str_dic = {'cross-ilc': 'x', 'MH18': 'mh', 'qe': 'qewithcmbmv'}

    estimator_str = estimator_str_dic[estimator]
    if lmaxtt != 3000:
        estimator_str = '%s_lmaxt%s' %(estimator_str, lmaxtt)

    expname_str_dic = {'s4_wide': 'cmbs4w', 'so_baseline': 'sobl', 'so_goal': 'sog', 'spt3g': 'spt3g'}
    expname_str = expname_str_dic[expname]

    return expname_str, estimator_str

def get_lensing_xcorr_input(expname, cross_corr_component_keyname, estimator, reqd_zbin = 'zbin3', lmaxtt = 3000, els = None):
    
    expname_str, estimator_str = lensing_estimator_folder_str(expname, estimator, lmaxtt = lmaxtt)
    folderpath = '%s/%s/lensrec_%s/' %(lensing_xcorr_folder, expname_str, estimator_str)

    #Input cross-correlation spectrum (Unbiased.)
    fname_unbiased_searchstr = '%s/*_input_%s_%s.dat' %(folderpath, cross_corr_component_keyname, reqd_zbin)
    #print(fname_unbiased_searchstr); ##sys.exit()
    fname_unbiased = glob.glob( fname_unbiased_searchstr )[0]
    els_, cl_kx_input = np.loadtxt(fname_unbiased, unpack = True) 
    if els is None:
        els = els_
    cl_kx_input = np.interp(els, els_, cl_kx_input)

    return els, cl_kx_input

def get_lensing_xcorr_biases(fg_component, expname, cross_corr_component_keyname, estimator, reqd_zbin = 'zbin3', bias_mul_fac_dict = None, lmaxtt = 3000):

    expname_str, estimator_str = lensing_estimator_folder_str(expname, estimator, lmaxtt = lmaxtt)
    folderpath = '%s/%s/lensrec_%s/' %(lensing_xcorr_folder, expname_str, estimator_str)

    #Input cross-correlation spectrum (Unbiased.)
    els, cl_kx_unbiased = get_lensing_xcorr_input(expname, cross_corr_component_keyname, estimator, reqd_zbin = reqd_zbin, lmaxtt = lmaxtt)

    #Output cross-correlation spectrum (Biased by different foregrounds.)
    fname_biased_searchstr = '%s/*_%s_%s_%s.dat' %(folderpath, fg_component, cross_corr_component_keyname, reqd_zbin)
    fname_biased = glob.glob( fname_biased_searchstr )[0]
    els, cl_kx_biased = np.loadtxt(fname_biased, unpack = True)

    #get the bias
    if bias_mul_fac_dict is None:
        bias_mul_fac = 1.
    else:
        bias_mul_fac = bias_mul_fac_dict[fg_component]

    curr_bias_normed = cl_kx_biased/cl_kx_unbiased * bias_mul_fac

    return els, cl_kx_unbiased, cl_kx_biased, curr_bias_normed

def get_lensing_errors(els, expname, cross_corr_component_keyname, estimator, reqd_zbin = 'zbin3', lmaxtt = 3000, fsky = 1., debug = False):

    #cl_kk + nl_kk
    cl_kk, nl_kk, cl_plus_nl_kk = get_clkk_nlkk(els, expname, estimator, lmaxtt = lmaxtt)

    #cl_xx + nl_xx
    cl_xx, nl_xx, cl_plus_nl_xx = get_clxx_nlxx(els, expname, cross_corr_component_keyname, estimator, reqd_zbin = reqd_zbin, lmaxtt = lmaxtt)
        
    #cl_kx (since nl_kx is zero)
    els, cl_kx_unbiased = get_lensing_xcorr_input(expname, cross_corr_component_keyname, estimator, reqd_zbin = reqd_zbin, els = els, lmaxtt = lmaxtt)
    cl_plus_nl_kx = np.copy(cl_kx_unbiased)

    #knox errors
    delta_cl_kx = fisher_tools.get_knox_errors(els, cl_plus_nl_kk, fsky, cl22 = cl_plus_nl_xx, cl12 = cl_plus_nl_kx)

    if debug: #plot signal and error bars now
        delta_el = 250
        binned_el = np.arange(0., max(els), delta_el)
        binned_cl_kx_unbiased = np.interp( binned_el, els, cl_kx_unbiased)
        binned_delta_cl_kx = np.interp( binned_el, els, delta_cl_kx)

        close('all')
        clf()
        ax = subplot(111, yscale='log')
        plot(els, cl_kx_unbiased, color='darkgray', lw=0.5, label = r'Signal'); 
        #plot(binned_el, binned_cl_kx_unbiased, color='black', lw=2., label = r'Signal: Binned'); 
        #errorbar(binned_el, binned_cl_kx_unbiased, yerr = binned_delta_cl_kx, color='red', marker ='o', capsize = 2.)
        bar(binned_el, 2*binned_delta_cl_kx, bottom = binned_cl_kx_unbiased-binned_delta_cl_kx, 
                    width = delta_el/2., color = 'orangered', alpha = 0.3)
        #plot(binned_el, binned_cl_kx_unbiased-binned_delta_cl_kx/2, color='red'); 
        #plot(binned_el, binned_cl_kx_unbiased+binned_delta_cl_kx/2, color='green'); 
        xlabel(r'Multipole $L$', fontsize = 14)
        ylabel(r'$C_{L}^{\kappa {\rm x}}$', fontsize = 14)
        xlim(150., 3000.); #ylim(1e-9, 1.)
        legend(loc = 1, fontsize = 10.)
        #title(r'CMB-S4 Wide: Cross-ILC', fontsize = 14)
        show(); sys.exit()

    return delta_cl_kx

def get_clxx_nlxx(els, expname, cross_corr_component_keyname, estimator, reqd_zbin = 'zbin3', lmaxtt = 3000, debug = False):

    expname_str, estimator_str = lensing_estimator_folder_str(expname, estimator, lmaxtt = lmaxtt)

    #cl_xx + nl_xx
    cl_xx_fname = '%s/cl_density_shear_allzbins.npy' %(lensing_xcorr_folder)
    if cross_corr_component_keyname == 'dens': #include shot noise term for density

        #signal
        cl_xx = np.load(cl_xx_fname, allow_pickle = True).item()[cross_corr_component_keyname][reqd_zbin]
        els_ = np.arange(len(cl_xx))
        cl_xx = np.interp(els, els_, cl_xx)

        #shot noise
        nl_xx = get_shot_noise(reqd_zbin, els = els)

        #signal + shot noise
        cl_plus_nl_xx = cl_xx + nl_xx

    else: #shear already has shape noise
        #signal + shape noise
        cl_plus_nl_xx = np.load(cl_xx_fname, allow_pickle = True).item()[cross_corr_component_keyname][reqd_zbin]
        els_ = np.arange(len(cl_plus_nl_xx))
        cl_plus_nl_xx = np.interp(els, els_, cl_plus_nl_xx)
        cl_xx, nl_xx = None, None

    if debug: #make a quick plot
        close('all')
        clf()
        ax = subplot(111, yscale='log')
        dl_fac = els * (els+1)/2/np.pi
        if cross_corr_component_keyname == 'dens':                        
            plot(els, dl_fac*cl_xx, color='darkgray', lw=2.); 
            plot(els, dl_fac*nl_xx, color='darkgreen', label = 'Noise'); 
            plot(els, dl_fac*cl_plus_nl_kk, color='darkred', label = 'Total'); 
        else:
            plot(els, dl_fac*cl_plus_nl_kk, color='darkred', label = 'Total')
        xlabel(r'Multipole $L$', fontsize = 14)
        ylabel(r'$D_{L}^{xx}$', fontsize = 14)
        xlim(0., 3000.); ylim(1e-9, 1.)
        legend(loc = 4, fontsize = 12)
        show();
        sys.exit()                    


    return cl_xx, nl_xx, cl_plus_nl_xx

def get_clkk_nlkk(els, expname, estimator, lmaxtt = 3000, cl_kk_fname = 'publish/data/CAMB/planck_2018/cmb_spectra_lensed_with_lensing.npy', debug = False):
    
    #cl_kk    
    tmp = np.load(cl_kk_fname, allow_pickle = True).item()
    els_, cl_kk = tmp['els'], tmp['Cl_dic']['PP']
    cl_kk = np.interp(els, els_, cl_kk)

    #nl_kk
    expname_str, estimator_str = lensing_estimator_folder_str(expname, estimator, lmaxtt = lmaxtt)
    if estimator_str == 'x_lmaxt5000': estimator_str = 'x'
    #print(estimator_str); sys.exit()
    nl_kk_fname = '%s/clkk_n0_%s_%s.dat' %(lensing_n0_folder, estimator_str, expname_str)
    if estimator_str == 'x':
        nl_kk_fname = nl_kk_fname.replace('_x_', '_xilc_')
    elif estimator_str == 'x_lmaxt5000': 
        nl_kk_fname = nl_kk_fname.replace('.dat', '_lmaxt5000.dat')
    els_, nl_kk = np.loadtxt(nl_kk_fname, unpack = True)
    nl_kk = np.interp(els, els_, nl_kk)

    cl_plus_nl_kk = cl_kk + nl_kk

    if debug: #make a quick plot
        close('all')
        clf()
        ax = subplot(111, yscale='log')#, xscale='log')
        plot(els, cl_kk, color='darkgray', lw=2.); 
        plot(els, nl_kk, color='darkgreen', label = 'N0'); 
        plot(els, cl_plus_nl_kk, color='darkred', label = 'Total')
        xlabel(r'Multipole $L$', fontsize = 14)
        ylabel(r'$C_{L}^{\kappa \kappa}$', fontsize = 14)
        xlim(0., 5000.); ylim(1e-9, 1e-5)
        legend(loc = 2, fontsize = 12)
        show();
        sys.exit()


    return cl_kk, nl_kk, cl_plus_nl_kk
