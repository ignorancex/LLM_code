import os, copy, pickle
from pylab import *
if (1):
    from matplotlib import rc;rc('text', usetex=True);rc('font', weight='bold');matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
    rcParams['font.family'] = 'serif'
    rc('text.latex',preamble=r'\usepackage{/Users/sraghunathan/.configs/apjfonts}')
    #from matplotlib import transforms

#import fisher_tools
import scipy as sc, warnings, argparse
np.seterr(divide='ignore', invalid='ignore')
import warnings, matplotlib.cbook
warnings.filterwarnings('ignore', category=UserWarning)

from matplotlib.patches import Ellipse

sys.path.append('/Users/sraghunathan/Research/SPTPol/analysis/git/DRAFT/modules/')
import foregrounds as fg, misc, ilc 
#sys.path.append('/Users/sraghunathan/Research/SPTPol/analysis/git/fisher_cosmo/modules/')
#sys.path.append('/Users/sraghunathan/Research/SPTpol/analysis/2020_07/ksz_ps/for_cross_ilc_paper/cross_ilc_methods_paper/modules/')
import tmp_tools as tools

parser = argparse.ArgumentParser(description='')
parser.add_argument('-which_spectra', dest='which_spectra', action='store', help='which_spectra', type=str, default='lensed_scalar_Alens0.30')
parser.add_argument('-use_thetastar', dest='use_thetastar', action='store', help='use_thetastar', type=int, default=1)
parser.add_argument('-set_YHe_using_BBN', dest='set_YHe_using_BBN', action='store', help='set_YHe_using_BBN', type=int, default=1)
parser.add_argument('-lmax', dest='lmax', action='store', help='lmax', type=int, default=5000)

parser.add_argument('-galmask', dest='galmask', action='store', help='galmask', type=int, default=2)
parser.add_argument('-use_cleanest_mask', dest='use_cleanest_mask', action='store', help='use_cleanest_mask', type=int, default=0)
parser.add_argument('-galmask2', dest='galmask2', action='store', help='galmask2', type=int, default=-1)

#datasets
parser.add_argument('-add_lensing', dest='add_lensing', action='store', help='add_lensing', type=int, default=0)
parser.add_argument('-add_planck', dest='add_planck', action='store', help='add_planck', type=int, default=1)#1)
parser.add_argument('-force_fsky_val', dest='force_fsky_val', action='store', type=float, default= -1., help='force_fsky_val')

parser.add_argument('-pspecstr', dest='pspecstr', action='store', help='pspecstr', type=str, default='TT-EE-TE')
parser.add_argument('-lmin_for_gal', dest='lmin_for_gal', action='store', help='lmin_for_gal', type=int, default=0)
parser.add_argument('-lmax_for_gal', dest='lmax_for_gal', action='store', help='lmax_for_gal', type=int, default=5000)
parser.add_argument('-gal_res_frac', dest='gal_res_frac', action='store', help='gal_res_frac', type=float, default=0.01)
parser.add_argument('-total_obs_time', dest='total_obs_time', action='store', type=float, default= 7., help='total_obs_time')
parser.add_argument('-use_JM_fisher_mat', dest='use_JM_fisher_mat', action='store', help='use_JM_fisher_mat', type=int, default=0)

#how to add bias
parser.add_argument('-use_parameter_shifts_for_syserrors', dest='use_parameter_shifts_for_syserrors', action='store', help='use_parameter_shifts_for_syserrors', type=int, default=1)#0)
parser.add_argument('-fix_other_params', dest='fix_other_params', action='store', help='fix_other_params', type=int, default=1)#0)

#fitting for galaxy
parser.add_argument('-cilc_nulled_comp', dest='cilc_nulled_comp', action='store', type=str, help='cilc_nulled_comp', default='')
parser.add_argument('-fit_for_galdust', dest='fit_for_galdust', action='store', type=int, help='fit_for_galdust', default= 1)
parser.add_argument('-expname', dest='expname', action='store', type=str, default= 's4wide', help='expname')
parser.add_argument('-Tdust', dest='Tdust', action='store', type=float, default= 20., help='Tdust')
parser.add_argument('-betadust', dest='betadust', action='store', type=float, default= 1.54, help='betadust')

#perform MLE estimate
parser.add_argument('-get_mle', dest='get_mle', action='store', help='get_mle', type=int, default=0)

args = parser.parse_args()
args_keys = args.__dict__
for kargs in args_keys:
    param_value = args_keys[kargs]
    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

##if cilc_nulled_comp != '': fit_for_galdust = 0
########################################################
########################################################
lmax = int(lmax_for_gal)
add_lensing = 0
use_thetastar = 1
if use_parameter_shifts_for_syserrors:
    params_to_shift =['neff']
    shift_frac = 0.5
    
prior_dic = {'tau': 0.007}
fix_params = ['ws', 'Alens', 'omk', 'mnu']#, 'thetastar', 'omch2', 'As', 'tau']
if (0):
    #fix_params = ['ws', 'mnu', 'Alens', 'omk', 'As', 'tau', 'omch2', 'ombh2', 'thetastar', 'ns']
    fix_params = ['ws', 'mnu', 'Alens', 'omk', 'tau', 'omch2', 'ombh2', 'thetastar']
    #fix_params = ['ws', 'mnu', 'Alens', 'omk', 'As', 'tau', 'omch2', 'ombh2']
if (use_parameter_shifts_for_syserrors or get_mle) and (fix_other_params):
    fix_params = ['ws', 'mnu', 'Alens', 'omk', 'As', 'tau', 'omch2', 'ombh2', 'thetastar', 'ns']
#fix_params = ['ws', 'mnu', 'Alens', 'omk', 'As', 'tau', 'omch2', 'ombh2', 'thetastar', 'neff']
#folders containing inputs
#camb_folder = 'data/CMB_spectra_derivatives_for_code_comparison/derivs_v2_20210114_lowres/'
parent_folder = '/Users/sraghunathan/Research/SPTPol/analysis/git/fisher_cosmo/'
camb_folder = '%s/data/CMB_spectra_derivatives_for_code_comparison/' %(parent_folder)
###draft_results_folder = 'data/DRAFT_results_20200601/s4like_mask/TT-EE-TE/baseline/'

########################################################
########################################################
#get fiducual LCDM power spectra computed using CAMB
print('\n\n')
print('\tget/read fiducual LCDM power spectra computed using CAMB')
if (0):
    camb_fname = '%s/cmb_spectra_%s_Srini.txt' %(camb_folder, which_spectra.replace('_scalar',''))
    cl_camb = np.loadtxt(camb_fname)[:lmax, :]
    els = np.arange(lmax)
    #els = cl_camb[:,0]
    #el_camb = cl_camb[:,0] 
    #cl_camb_tt = cl_camb[:,1]
    cl_dic = {}
    cl_dic['TT'] = cl_camb[:,1]
    cl_dic['EE'] = cl_camb[:,2]
    cl_dic['TE'] = cl_camb[:,3]
else:
    camb_fname = '%s/cmb_spectra_%s_Srini.npy' %(camb_folder, which_spectra.replace('_scalar',''))    
    camb_dic = np.load(camb_fname, allow_pickle=1).item()
    #els = camb_dic['els']
    cl_dic = camb_dic['Cl_dic']
    for k in cl_dic:
        cl_dic[k] = cl_dic[k][:lmax]
    els = np.arange(lmax)
########################################################
########################################################
#read derivatives
print('\tget/read derivatives')
camb_deriv_fname = '%s/cmb_spectra_derivs_%s_Srini.npy' %(camb_folder, which_spectra.replace('_scalar',''))
cl_deriv_dic_tmp = np.load(camb_deriv_fname, allow_pickle = 1).item()
cl_deriv_dic = {}
param_names = []
for p in sorted( cl_deriv_dic_tmp ):
    #if p not in fisher_param_names: continue
    if p == 'ell': continue
    cl_deriv_dic[p]={}
    cl_deriv_dic[p]['TT'] = cl_deriv_dic_tmp[p]['TT'][:lmax]
    cl_deriv_dic[p]['EE'] = cl_deriv_dic_tmp[p]['EE'][:lmax]
    cl_deriv_dic[p]['TE'] = cl_deriv_dic_tmp[p]['TE'][:lmax]
    param_names.append( p )
param_names = np.asarray( param_names )
##print(param_names, fix_params); sys.exit()
########################################################
########################################################
#read fisher matrix now
#fisher_folder = 'results/20200701/pinv_issue_20210108/Yp_using_BBN/with_thetastar/lensed_scalar_Alens0.30/s4like_mask_v2/TT-EE/baseline/'
#fisher_folder = 'results/20200701/Yp_using_BBN/with_thetastar/lensed_scalar_Alens0.30/s4like_mask_v2/TT-EE/baseline/'

if (1): #20210426 - using updated ILC curves / Fisher results
    #fisher_folder = 'results/20200701/Yp_using_BBN/with_thetastar/%s/s4like_mask_v2/TT-EE/baseline/' %(which_spectra)
    #nlfolder = 'data/DRAFT/results/20200701/s4like_mask_v2/TT-EE/baseline/'

    '''
    fisher_folder = 'results/20210423_with202102designtoolinputforpySM3sims/Yp_using_BBN/with_thetastar/%s/s4like_mask_v2/TT-EE/baseline/' %(which_spectra)
    nlfolder = 'data/DRAFT/results/20210423_with202102designtoolinputforpySM3sims/s4like_mask_v2/TT-EE/baseline/'
    '''

    fisher_folder = '%s/results/20210506_with202102designtoolinputforpySM3sims_sedscalingfordust/Yp_using_BBN/with_thetastar/%s/s4like_mask_v2/TT-EE/baseline/' %(parent_folder, which_spectra)
    nlfolder = '%s/data/DRAFT/results/20210506_with202102designtoolinputforpySM3sims_sedscalingfordust/s4like_mask_v2/TT-EE/baseline/' %(parent_folder)

    if cilc_nulled_comp !='':
        nlfolder = nlfolder.replace('/s4like_mask_v2/', '/%s/s4like_mask_v2/' %(cilc_nulled_comp))

if use_cleanest_mask:
    fisher_folder = fisher_folder.replace('s4like_mask_v2/', 's4like_mask_v3/')
    nlfolder = nlfolder.replace('s4like_mask_v2/', 's4like_mask_v3/')
lmin = 30.0
extrastr = ''
if add_planck:
    fisher_folder = '%s/with_planck_on_large_scales/' %(fisher_folder)
    lmin = 2.0
    extrastr = '%s_withplanckonlargescales' %(extrastr)
if add_lensing:
    fisher_folder = '%s/with_lensing/' %(fisher_folder)
    pspecstr = '%s-PP' %(pspecstr)
    extrastr = '%s_pluslensing' %(extrastr)

if (1): #20210426
    param_names_str = 'Alens-As-mnu-neff-ns-ombh2-omch2-omk-tau-thetastar-10cosmo'
    if fit_for_galdust:
        fisher_folder = '%s/fitting_for_galdust' %(fisher_folder)
        param_names_str = 'Alens-As-Tdust-betadust-mnu-neff-ns-ombh2-omch2-omk-tau-thetastar-12cosmo'

    if cilc_nulled_comp != '':
        fisher_folder = '%s/%s' %(fisher_folder, cilc_nulled_comp)

if pspecstr == 'TT-EE-TE':
    lminT, lmaxT, lminP, lmaxP = float(lmin), float(lmax), float(lmin), float(lmax)
elif pspecstr == 'TT':
    lminT, lmaxT, lminP, lmaxP = float(lmin), float(lmax), None, float(lmax)
elif pspecstr == 'EE':
    lminT, lmaxT, lminP, lmaxP = None, float(lmax), float(lmin), float(lmax)

'''
if galmask == 2:
    fisher_file = '%s/s4_0.57fsky_2.0minltemp_5000.0maxltemp_2.0minlpol_5000.0maxlpol_TT-EE-TE_galaxy1_galmask2__withplanckonlargescales_As-mnu-neff-ns-ombh2-omch2-tau-thetastar-8cosmo.npy' %(fisher_folder)
elif galmask == 5:
    fisher_file = '%s/s4_0.11fsky_30.0minltemp_5000.0maxltemp_30.0minlpol_5000.0maxlpol_TT-EE-TE-PP_galaxy1_galmask5__pluslensing_As-mnu-neff-ns-ombh2-omch2-tau-thetastar-8cosmo.npy' %(fisher_folder)
'''
def get_fsky(galmask, use_cleanest_mask = 0):
    if galmask == 2:
        fsky = 0.57
    elif galmask == 5:
        fsky = 0.11

    if use_cleanest_mask:
        if galmask == 0:
            fsky = 0.15
        elif galmask == 1:
            fsky = 0.42
        elif galmask == 2:
            fsky = 0.11    

    return fsky

fsky = get_fsky(galmask, use_cleanest_mask = use_cleanest_mask)
if force_fsky_val != -1.:
    fsky = force_fsky_val
#fisher_file = '%s/s4_%sfsky_%.1fminltemp_%.1fmaxltemp_%.1fminlpol_%.1fmaxlpol_%s_galaxy1_galmask%s_%s_As-mnu-neff-ns-ombh2-omch2-tau-thetastar-ws-9cosmo.npy' %(fisher_folder, fsky, lmin, lmax, lmin, lmax, pspecstr, galmask, extrastr)

#20210120 - check TT/EE/TT-EE-TE separately
#fisher_file = '%s/s4_%sfsky_%sminltemp_%smaxltemp_%sminlpol_%smaxlpol_%s_galaxy1_galmask%s_%s_Alens-As-mnu-neff-ns-ombh2-omch2-omk-tau-thetastar-10cosmo.npy' %(fisher_folder, fsky, lminT, lmaxT, lminP, lmaxP, pspecstr, galmask, extrastr)
fisher_file = '%s/s4_%sfsky_%sminltemp_%smaxltemp_%sminlpol_%smaxlpol_%s_galaxy1_galmask%s_%s_%s_for%dyears.npy' %(fisher_folder, fsky, lminT, lmaxT, lminP, lmaxP, pspecstr, galmask, extrastr, param_names_str, total_obs_time)
tmp_JM_opfname_str = None
if (0): #use FIsher computed with JM derivatives
    fisher_file = '%s/s4_%sfsky_%sminltemp_%smaxltemp_%sminlpol_%smaxlpol_%s_galaxy1_galmask%s_%s_As-mnu-neff-ns-ombh2-omch2-tau-thetastar-8cosmowithJMderivs.npy' %(fisher_folder, fsky, lminT, lmaxT, lminP, lmaxP, pspecstr, galmask, extrastr)  
    #print(fisher_file); sys.exit()
    tmp_JM_opfname_str = 'withJMspecderivs'
    JM_ret_dic = tools.get_JM_fisher_derivatives(els, which_spectra, galmask, get_fisher = 0, get_derivatives = 1)
    cl_deriv_dic = JM_ret_dic['cl_deriv_dic']

if galmask2 != -1:
    fsky2 = get_fsky(galmask2, use_cleanest_mask = use_cleanest_mask)
    fisher_file2 = '%s/s4_%sfsky_%sminltemp_%smaxltemp_%sminlpol_%smaxlpol_%s_galaxy1_galmask%s_%s_Alens-As-mnu-neff-ns-ombh2-omch2-omk-tau-thetastar-10cosmo.npy' %(fisher_folder, fsky2, lminT, lmaxT, lminP, lmaxP, pspecstr, galmask, extrastr)
    print(fisher_file2)
    sys.exit()

if (0): #spt3g
    fix_params = ['ws', 'mnu', 'Alens', 'omk']

    fsky = 0.036
    lmin_tmp, lmax_tmp = 100, 3500
    lmax_for_gal = lmax_tmp
    fisher_folder = 'results/spt3g/20210115//with_thetastar/lensed_scalar/'
    extrastr = ''
    fisher_file = '%s/spt3g_%sfsky_%dminltemp_%dmaxltemp_%dminlpol_%dmaxlpol_%s_galaxy0_galmask-1_%sAlens-As-mnu-neff-ns-ombh2-omch2-omk-tau-thetastar-10cosmo.npy' %(fisher_folder, fsky, lmin_tmp, lmax_tmp, lmin_tmp, lmax_tmp, pspecstr, extrastr)

fisher_dic = np.load(fisher_file, allow_pickle = 1, encoding = 'latin1').item()
param_names = np.asarray( fisher_dic['param_names'] )
param_names_JM = np.copy(param_names)
param_dict = fisher_dic['param_dict']
#print(sorted(param_dict.keys()));sys.exit()
F_mat = fisher_dic['F_mat']

#print(param_names, fisher_param_names, F_mat.shape); sys.exit()
##print(param_names, F_mat.shape, fix_params); sys.exit()

#20230413 - not reading Joel's Fisher matrix
if (1): #use Joel's Fisher matrix
    if use_JM_fisher_mat:

        JM_ret_dic = tools.get_JM_fisher_derivatives(els, which_spectra, galmask, get_fisher = 1, get_derivatives = 1)
        param_names_JM = JM_ret_dic['param_names']
        F_mat_JM = JM_ret_dic['F_mat']
        cl_sysdic_JM = JM_ret_dic['cl_sysdic']

        F_mat = np.copy( F_mat_JM )
        param_names = np.copy( param_names_JM )
        if (1):
            cl_deriv_dic_JM = JM_ret_dic['cl_deriv_dic']
            #sys.exit()

            if (0): #plot and compare derivatives
                #JM old derivs
                import glob
                fd = '/Users/sraghunathan/Dropbox/S4_Neff/results/CMB_spectra_derivs/'                
                searchstr_suff = 'cmb_derivs_*_%s_Joel.txt' %(which_spectra.replace('_scalar', ''))
                searchstr = '%s/%s' %(fd, searchstr_suff)
                flist = glob.glob(searchstr)
                cl_deriv_dic_JM_old = {}
                for f in flist:
                    keyname = f.split('/')[-1].replace('cmb_derivs_','').replace('_%s_Joel.txt' %(which_spectra.replace('_scalar', '')), '')
                    if keyname not in params_mapper_dic: continue
                    pmod = params_mapper_dic[keyname]
                    cl_deriv_dic_JM_old[pmod] = {}
                    el, tt_der, ee_der, te_der = np.loadtxt(f, unpack = 1)
                    #inds = np.where( (el>=minel) & (el<=maxel))[0]
                    tt_der = np.interp(els, np.arange(len(tt_der)), tt_der)                
                    ee_der = np.interp(els, np.arange(len(ee_der)), ee_der)
                    te_der = np.interp(els, np.arange(len(te_der)), te_der)
                    cl_deriv_dic_JM_old[pmod]['TT'], cl_deriv_dic_JM_old[pmod]['EE'], cl_deriv_dic_JM_old[pmod]['TE'] = tt_der, ee_der, te_der

                fname = '%s/cmb_spectra_derivs_lensed_Srini.npy' %(fd)
                cl_deriv_dic_old_tmp = np.load(fname, allow_pickle = 1, encoding = 'latin1').item()
                cl_deriv_dic_old = {}
                tmp_el = cl_deriv_dic_old_tmp['ell']
                for p in cl_deriv_dic_old_tmp:
                    if p == 'ell': continue
                    tt_der, ee_der, te_der = cl_deriv_dic_old_tmp[p]
                    tt_der = np.interp(els, np.arange(len(tt_der)), tt_der)                
                    ee_der = np.interp(els, np.arange(len(ee_der)), ee_der)
                    te_der = np.interp(els, np.arange(len(te_der)), te_der)
                    cl_deriv_dic_old[p] = {}
                    cl_deriv_dic_old[p]['TT'], cl_deriv_dic_old[p]['EE'], cl_deriv_dic_old[p]['TE'] = tt_der, ee_der, te_der

                which_spec = 'TT'
                tr, tc = 2, 4
                single_panel = 0
                if not single_panel:
                   fig = figure(figsize=(13, 8.))
                   subplots_adjust(hspace = 0.1)

                plfolder = 'plots/20200701/derivs_checks'; 
                if not os.path.exists(plfolder): os.system('mkdir -p %s' %(plfolder))
                for pcntr, p in enumerate( sorted(param_names) ):
                    print(p)
                    if not single_panel:
                        ax = subplot(tr, tc, pcntr + 1, yscale = 'log')
                        lwval = 1.
                        fsval = 10
                    else:
                        clf()
                        ax = subplot(111, yscale = 'log')
                        lwval = 1.
                        fsval = 14
                    to_plot_JM, to_plot_JM_old = cl_deriv_dic_JM[p][which_spec], cl_deriv_dic_JM_old[p][which_spec]
                    to_plot, to_plot_old = cl_deriv_dic[p][which_spec], cl_deriv_dic_old[p][which_spec]
                    colorarr = ['black', 'goldenrod', 'lime', 'orangered']
                    labarr = ['SR', 'SR: old', 'CT/JM', 'JM: old']
                    to_plot_arr = [to_plot, to_plot_old, to_plot_JM, to_plot_JM_old]
                    for cntr, curr_to_plot in enumerate( to_plot_arr ):
                        curr_to_plot[np.isnan(curr_to_plot)] = 0.
                        curr_to_plot[np.isinf(curr_to_plot)] = 0.
                        labval = None
                        if pcntr == 0 or single_panel: labval = r'%s' %(labarr[cntr])
                        plot(curr_to_plot, color = colorarr[cntr], label = labval, lw = lwval); 
                        if min(curr_to_plot)<0:
                            curr_to_plot_neg = curr_to_plot * -1.
                            plot(curr_to_plot_neg, color = colorarr[cntr], lw = 0.5, ls = '--'); 
                        if pcntr == 0 or single_panel: 
                            if cntr == len(to_plot_arr)-1: plot([],[],lw=0.5, ls='--',label = r'Negatives');
                            legend(loc = 1)
                    
                    tit = tools.get_latex_param_str(p)
                    title(tit, fontsize = fsval)
                    xlim(0, 5000)
                    if not single_panel:
                        if pcntr<tc:
                            setp(ax.get_xticklabels(), visible=False)
                        else:
                            xlabel(r'Multipole $\ell$', fontsize = fsval)
                    else:
                        xlabel(r'Multipole $\ell$', fontsize = fsval)

                        plname = '%s/deriv_for_%s.png' %(plfolder, p)
                        savefig(plname, dpi = 200.)
                        close()
                        #show()
                if not single_panel:
                    plname = '%s/%s_%sderivs.png' %(plfolder, which_spectra, which_spec)
                    savefig(plname, dpi = 200.)
                    close()
                #show(); 
                sys.exit()

            cl_deriv_dic = cl_deriv_dic_JM

    if (0):
        powersFid=data['powersFid']
        cosmoFid=data['cosmoFid']
        cmbNoiseSpectra=data['cmbNoiseSpectra']
        biasVector=data['biasVector']
        forecastConfig=data['forecastConfig']
        sysSpectrum=data['sysSpectrum']
        deflectionNoises=data['deflectionNoises']    

#add priors
print('\tadding prior: %s' %(list(prior_dic.keys())))

F_mat = tools.fn_add_prior(np.copy(F_mat), param_names, prior_dic)

#fix params
print('\tfixing some parameters: %s' %(fix_params))
F_mat, param_names = tools.fn_fix_params(F_mat, param_names, fix_params)
param_names = np.asarray(param_names)
##print(param_names); sys.exit()

C_mat = np.linalg.inv(F_mat)
sigma_dic = {}
param_errors = np.diag(C_mat)**0.5
for pcntr, p in enumerate( param_names ):
    sigma_dic[p] = param_errors[pcntr]

if (0):
    print(F_mat.shape, np.diag(C_mat)**0.5, param_names)
    sys.exit()
##print(sigma_dic); sys.exit()
########################################################
########################################################
#get nl file
print('\tget nl')
#nlfile = '%s/s4wide_ilc_galaxy1_27-39-93-145-225-278_TT-EE_galmask%s_AZ.npy' %(nlfolder, galmask)
nlfile = '%s/%s_ilc_galaxy1_27-39-93-145-225-278_TT-EE_galmask%s_AZ_for%dyears.npy' %(nlfolder, expname, galmask, total_obs_time)
if cilc_nulled_comp != '':
    nlfile = nlfile.replace('TT-EE_', 'TT-EE_%s_' %(cilc_nulled_comp))
print('\n\tNl file is: %s\n' %(nlfile))
nl_dic, fsky = tools.get_nldic(nlfile, els)
'''
nl_dic_master, fsky = tools.get_nldic(nlfile, els)
import copy
if pspecstr == 'TT-EE-TE':
    nl_dic = copy.deepcopy(nl_dic_master)
elif pspecstr == 'TT':
    nl_dic = {}
    nl_dic['TT'] = nl_dic_master['TT']
elif pspecstr == 'EE':
    nl_dic = {}
    nl_dic['EE'] = nl_dic_master['EE']
'''

#get galactic residuals
if (1): #20210426
    ###galresfile = '%s/s4wide_ilc_galaxy1_27-39-93-145-225-278_TT-EE_galmask%s_AZ_galresiduals.npy' %(nlfolder, galmask)
    ###gal_dustsync_resdic = np.load(galresfile, allow_pickle = 1, encoding = 'latin1').item()

    ilc_res_dic = np.load(nlfile, allow_pickle = True).item()
    fg_res_dic = ilc_res_dic['fg_res_dic']
    gal_dustsync_resdic = {}
    for which_spec in fg_res_dic:
        gal_dustsync_resdic[which_spec] = {}
        for signal in fg_res_dic[which_spec]:
            fg_res = fg_res_dic[which_spec][signal][:5000]
            #fg_res = np.interp( els, np.arange( len(fg_res) ), fg_res)
            if signal == 'galdust':
                if (0):
                    print('\n\n\t\t\t\tchange me: nulling gal dust for %s' %(which_spec))
                    fg_res = fg_res * 0.                    
                gal_dustsync_resdic[which_spec]['gal_dust'] = fg_res
            elif signal == 'galsync':
                if (1):
                    print('\n\n\t\t\t\tchange me: nulling gal sync for %s' %(which_spec))
                    fg_res = fg_res * 0.
                gal_dustsync_resdic[which_spec]['gal_sync'] = fg_res

if not fit_for_galdust: #modify galres based on intended fraction
    for which_spec in gal_dustsync_resdic:
        for keyname in gal_dustsync_resdic[which_spec]:
            gal_dustsync_resdic[which_spec][keyname] *= gal_res_frac
            null_gal_inds = np.where( (els<=lmin_for_gal) | (els>=lmax_for_gal) )[0]
            gal_dustsync_resdic[which_spec][keyname][null_gal_inds] = 0.
else:
    gal_res_frac = 1.

    #first calculate the residual galaxy power using freq. dep. weights.
    beam_noise_dic = ilc_res_dic['beam_noise_dic']
    freqarr = sorted(beam_noise_dic['T'].keys())
    ilc_weights_dic = ilc_res_dic['weights']

    #get beams
    bl_dic = misc.get_beam_dic(freqarr, beam_noise_dic['T'], max(els)+1)

    #get residual galactic dust
    param_dict['Tdust'] = Tdust
    param_dict['betadust'] = betadust    
    cl_gal_dust_res_dic = tools.get_gal_dust_residuals(els, expname, freqarr, galmask, param_dict, bl_dic, ilc_weights_dic)

    import copy
    ori_gal_dustsync_resdic = copy.deepcopy( gal_dustsync_resdic )
    for which_spec in gal_dustsync_resdic:
        gal_dustsync_resdic[which_spec]['gal_dust'] = gal_dustsync_resdic[which_spec]['gal_dust'] - cl_gal_dust_res_dic[which_spec]

    #we must also compute derivatives for these parameters.
    params_for_galdust_fitting = ['Tdust', 'betadust']#, Adust, alphadust]
    cosmo_param_dict = {}
    cosmo_param_dict['Tdust'] = (0.01,(1.,0.0001))
    cosmo_param_dict['betadust'] = (0.01,(1.,0.0001))
    which_spec_arr = ['TT', 'EE', 'BB', 'TE', 'PP', 'Tphi', 'Ephi']
    cl_gal_params_deriv_dic = tools.get_gal_dust_derivatives(els, expname, freqarr, galmask, params_for_galdust_fitting, param_dict, cosmo_param_dict, bl_dic, ilc_weights_dic, which_spec_arr = which_spec_arr)

    for p in cl_gal_params_deriv_dic:
        cl_deriv_dic[p] = cl_gal_params_deriv_dic[p]

    '''
    print(param_names); sys.exit()
    param_names.tolist().extend( params_for_galdust_fitting )
    param_names = np.asarray( sorted( param_names ) )
    print(param_names); sys.exit()
    '''
########################################################
########################################################
#separate gal residual from total nl
nl_nogal_dic = {}
galres_dic = {}
for which_spec in nl_dic:
    galres = np.zeros(lmax)
    if which_spec in gal_dustsync_resdic:
        galrestmp = gal_dustsync_resdic[which_spec]['gal_dust'] + gal_dustsync_resdic[which_spec]['gal_sync']
        galres = galrestmp[:lmax]#np.interp(els, np.arange(len(galrestmp)), galrestmp)
        ####print('\n\tchange me\n')
        ####galres[:len(galrestmp)-1] = galrestmp[1:]#np.interp(els, np.arange(len(galrestmp)), galrestmp)
    galres_dic[which_spec] = galres
    nl_nogal_dic[which_spec] = nl_dic[which_spec] - galres
import copy
cl_sysdic = copy.deepcopy(galres_dic)
########################################################
########################################################
#replace sys dic with parameter shifts
import copy #new addition
if use_parameter_shifts_for_syserrors:
    #import time
    #start = time.time()
    param_dict_low_res = copy.deepcopy(param_dict)
    if (1):
        param_dict_low_res['max_l_limit']=lmax
        param_dict_low_res['max_eta_k']=7000.0
        param_dict_low_res['max_eta_k_tensor']=3000.0
        param_dict_low_res['AccuracyBoost']=1
        param_dict_low_res['lAccuracyBoost']=1
        param_dict_low_res['lSampleBoost']=1

    print('\t\tShifting parameters for systematic error computations')
    import camb
    param_dict_fid = copy.deepcopy(param_dict_low_res)
    if which_spectra.find('Alens')>-1:
        tmp = which_spectra.split('_Alens')
        which_spectra_mod = tmp[0]
        Alens = float(tmp[1].replace('_Alens', ''))
        param_dict_fid['Alens'] = Alens
    else:
        which_spectra_mod = which_spectra
    pars_, els_, cl_dic_fid = tools.fn_set_CAMB_como(param_dict_fid, which_spectra_mod, add_lensing = add_lensing, use_thetastar = use_thetastar)

    #now compute modified cl with shifted parameters
    param_dict_mod = copy.deepcopy(param_dict_low_res)
    for p in params_to_shift:
        shift_val = sigma_dic[p] * shift_frac
        #shift_val = -0.008
        param_dict_mod[p] = param_dict_mod[p] + shift_val
        print('\t\t\tshifting %s: fiducial = %s; shift = %s; modified = %s' %(p, param_dict[p], shift_val, param_dict_mod[p]))

    pars_, els_, cl_dic_mod = tools.fn_set_CAMB_como(param_dict_mod, which_spectra_mod, add_lensing = add_lensing, use_thetastar = use_thetastar)

    if (0):
        ax  =subplot(111, yscale = 'log')
        plot(els_, cl_dic_fid['TT'], color = 'black')
        plot(els_, cl_dic_mod['TT'], color = 'red')
        plot(els_, cl_dic_fid['TT'] - cl_dic_mod['TT'], color = 'lime')
        xlim(0., 5000.); ylim(1e-8, 1e-2); 
        show(); sys.exit()

    #now compute cl_sys = cl_dic_fid - cl_dic_mod
    cl_sysdic = {}
    for which_spec in nl_dic:
        cl_sys = cl_dic_fid[which_spec] - cl_dic_mod[which_spec]
        cl_sysdic[which_spec] = np.interp(els, np.arange(len(cl_sys)), cl_sys)
    #end = time.time(); print((end-start)/60.); sys.exit()
########################################################
########################################################

#add systematic signal to cl now
clplussys_dic = {}
for which_spec in nl_dic:
    if which_spec in cl_sysdic:
        cl_sys = cl_sysdic[which_spec]
    else:
        cl_sys = np.zeros(len(cl_dic[which_spec]))
    clplussys_dic[which_spec] = cl_dic[which_spec] + cl_sys

if (1):
    #from IPython import embed; embed()
    close()
    figure(figsize=(8., 3.5))
    for which_spec_cntr, which_spec in enumerate( gal_dustsync_resdic ):
        ax = subplot(1,2,which_spec_cntr+1, yscale = 'log')
        plot(cl_dic[which_spec], 'gray')
        clsys = cl_sysdic[which_spec]
        if use_parameter_shifts_for_syserrors:
            labsys = 'Systematic: parameter shifts'
        else:
            labsys = 'Systematic: gal residuals'
        plot(clsys, 'darkred', label = labsys)
        if fit_for_galdust:
            #ori_cl_gal = cl_gal_dust_res_dic[which_spec] - (gal_dustsync_resdic[which_spec]['gal_dust'] + gal_dustsync_resdic[which_spec]['gal_sync'])
            plot(cl_gal_dust_res_dic[which_spec], 'goldenrod', label = 'Fit/scaling for galactic residuals')
            plot(ori_gal_dustsync_resdic[which_spec]['gal_dust'][:lmax], 'orangered', label = 'Original galactic residuals')
        else:
            plot(clsys / gal_res_frac, 'orangered', label = 'Original galactic residuals')
        if min(clsys)<0.:
            plot(-clsys, 'darkred', ls = '--')
        #plot(nl_nogal_dic[which_spec], 'darkgreen', label = r'ILC Noise')
        plot(nl_dic[which_spec], 'darkgreen', label = r'ILC Noise')
        xlim(0, 5000); #ylim(1e-9, 1e-2); 
        ylim(1e-8, 1e-2); 
        if galmask == 2:
            galmask_str = 'Mask: S4-Clean'
        elif galmask == 5:
            galmask_str = 'Mask: S4-Dirty'
        else:
            galmask_str = 'Mask: %s' %(galmask)

        if use_parameter_shifts_for_syserrors:
            params_to_shift_str = '-'.join(params_to_shift)
            titstr = r'%s: Systematic shifts for [%s]= %s$\sigma$; %s' %(which_spec, params_to_shift_str, shift_frac, galmask_str)
        else:
            if not fit_for_galdust:
                titstr = r'%s: Residual gal = %s; %s' %(which_spec, gal_res_frac, galmask_str)
            else:
                titstr = r'%s: Residual gal; %s' %(which_spec, galmask_str)

        title(titstr, fontsize = 10)
        if which_spec_cntr == 0:
            legend(loc = 1)#, fontsize = 8)
            ylabel(r'C$_{\ell}$ [$\mu$K$^{2}$]')
        xlabel(r'Multipole $\ell$')
    if use_parameter_shifts_for_syserrors:
        plname = 'plots/20200701/fisher_bias/sys_signal_shifted_params_%s.png' %(params_to_shift_str)
    else:
        plname = 'plots/20200701/fisher_bias/sys_signal_galmask%s_galesfrac%s.png' %(galmask, gal_res_frac)
    ###savefig(plname, dpi = 200.)
    show(); ##sys.exit()

########################################################
########################################################
get_bias_vector = 1
if get_bias_vector:

    #calculate bias vector
    #Eq.(8) of https://arxiv.org/pdf/0710.5171.pdf

    #get delta_cl first
    exp_dic = {}
    exp_dic['fsky'] = fsky
    delta_cl_dic = tools.fn_delta_Cl(els, clplussys_dic, exp_dic, Nldic = nl_nogal_dic)   

    if (0):
        cl_dic.pop('PP')
        cl_dic.pop('Tphi')
        cl_dic.pop('Ephi')
        cl_dic.pop('BB')
        delta_cl_cmb_ilcnoise_dic = tools.fn_delta_Cl(els, cl_dic, exp_dic, Nldic = nl_nogal_dic)   
        from IPython import embed; embed()
        for which_spec_cntr, which_spec in enumerate( gal_dustsync_resdic ):            
            plot(cl_sysdic[which_spec]/delta_cl_cmb_ilcnoise_dic[which_spec], label = r'%s' %(which_spec))
        legend(loc = 2, fancybox = True)
        xlabel(r'Multipole $\ell$')
        ylabel(r'C$_{\ell}^{res. gal}$/$\Delta$C$_{\ell}$', fontsize = 16)
        title(r'%d\%% residual galaxy' %(gal_res_frac * 100.))
        show(); sys.exit()

    print('\n\tcalculating bias vector now')
    if (0):
        bias_ind_spec_dic = {}
        bias_vector = []
        for pcntr, p in enumerate( param_names ):
            bias_ind_spec_dic[p] = {}
            for which_spec in cl_deriv_dic[p]:
                bias_ind_spec_dic[p][which_spec] = np.sum( galres_dic[which_spec] * cl_deriv_dic[p][which_spec] / (delta_cl_dic[which_spec])**2.)
            bias_vector.append( np.sum( list(bias_ind_spec_dic[p].values()) ) )
            #print('\t\t%s, %s' %(p, bias_vector[pcntr]))

    #from IPython import embed; embed()
    check_with_trace = 0
    if check_with_trace:
        bias_vector_with_trace = []
    bias_vector = []
    for pcntr, p in enumerate( param_names ):
        curr_bias_vector_with_trace = []
        curr_bias_vector = []
        for l in range(lmax):
            #if l<=200: continue
            TT, EE, TE = delta_cl_dic['TT'][l], delta_cl_dic['EE'][l], delta_cl_dic['TE'][l]
            TT_der, EE_der, TE_der = cl_deriv_dic[p]['TT'][l], cl_deriv_dic[p]['EE'][l], cl_deriv_dic[p]['TE'][l]
            TT_gal, EE_gal, TE_gal = cl_sysdic['TT'][l], cl_sysdic['EE'][l], cl_sysdic['TE'][l]
            if pspecstr == 'TT':
                EE, TE = 0., 0.
                EE_der, TE_der = 0., 0.
                EE_gal, TE_gal = 0., 0.
            elif pspecstr == 'EE':
                TT, TE = 0., 0.
                TT_der, TE_der = 0., 0.
                TT_gal, TE_gal = 0., 0.

            if check_with_trace:

                PP, TP, EP  = 0., 0., 0.
                PP_der, TP_der, EP_der  = 0., 0., 0.

                if 'PP' in delta_cl_dic:
                    PP, TP, EP = delta_cl_dic['PP'][l], delta_cl_dic['TP'][l], delta_cl_dic['EP'][l]
                    PP_der, TP_der, EP_der  = cl_deriv_dic[p]['PP'][l], cl_deriv_dic[p]['Tphi'][l], cl_deriv_dic[p]['Ephi'][l]

                curr_fisher_cov = tools.get_cov_for_fisher_with_trace(TT, EE, TE, PP, TP, EP)
                #print(curr_fisher_cov); sys.exit()
                inv_curr_fisher_cov = sc.linalg.pinv2(curr_fisher_cov)
                #inv_curr_fisher_cov = np.linalg.inv(curr_fisher_cov)

                galres_vec = tools.get_cov_for_fisher_with_trace(TT_gal, EE_gal, TE_gal, 0., 0., 0.)
                der_vec = tools.get_cov_for_fisher_with_trace(TT_der, EE_der, TE_der, PP_der, TP_der, EP_der)
                curr_bias_val_with_trace = np.trace( np.dot( np.dot(inv_curr_fisher_cov, galres_vec), np.dot(inv_curr_fisher_cov, der_vec) ) )

                curr_bias_vector_with_trace.append( curr_bias_val_with_trace )

                #print(l, curr_bias_vector_with_trace, 'trace')

            if (1):

                curr_fisher_cov = tools.get_cov_for_fisher(TT, EE, TE)
                inv_curr_fisher_cov = sc.linalg.pinv2(curr_fisher_cov)

                galres_vec = [TT_gal, EE_gal, TE_gal]
                der_vec = [TT_der, EE_der, TE_der]
                curr_bias_val = np.dot(galres_vec, np.dot( inv_curr_fisher_cov, der_vec ))
                #print(l, galres_vec)

                if (0):#l == 1000:
                    print(p, curr_fisher_cov)
                    print(galres_vec, der_vec, curr_bias_val)
                    sys.exit()
                    #from IPython import embed; embed()


                curr_bias_vector.append(curr_bias_val)
        #sys.exit()

        bias_vector.append( np.sum(curr_bias_vector) )
        if check_with_trace: bias_vector_with_trace.append( np.sum(curr_bias_vector_with_trace) )
        if (0):
            bias_vector = [-10916.1982595, 11926.1337747, 2.84570434619e+12, -3497705.388, -1146.04970381, 254711.509444, 8110.20114502]
        print('\t\t%s, %s' %(p, bias_vector[pcntr]))
    #sys.exit()

    bias_vector = np.asarray(bias_vector)

    if check_with_trace: 
        bias_vector_with_trace = np.asarray( bias_vector_with_trace )
        print(bias_vector)
        print(bias_vector_with_trace)
        sys.exit()

    if (0):
        tmp_bias_dic = {}
        for pind, p in enumerate( param_names ):
            tmp_bias_dic[p] = bias_vector[pind]
        for p in param_names_JM:
            if p in fix_params: continue
            print('\t\t%s %s' %(p, tmp_bias_dic[p]))
        print(biasVector_JM); sys.exit()

    ########################################################
    ########################################################
    #get final bias vector
    #C_mat = sc.linalg.pinv(F_mat)
    C_mat = np.linalg.inv(F_mat)
    #print(bias_vector.shape, C_mat.shape); sys.exit()
    final_bias_vector = np.asarray( np.dot( np.mat(bias_vector), C_mat ) )[0]
    if (0):
        bias_vector_mod = np.asarray( [bias_vector[0], bias_vector[2], bias_vector[1]] )
        F_mat_mod = np.array([[ 8.77664579e+04, -1.24294816e+07, -6.44112583e+05],
       [-1.24294816e+07,  3.23510755e+09,  9.94326484e+07],
       [-6.44112583e+05,  9.94326484e+07,  5.44382434e+06]])
        C_mat_mod = np.linalg.inv(F_mat_mod)
        final_bias_vector_mod = np.asarray( np.dot( np.mat(bias_vector_mod), C_mat_mod ) )[0]
        param_names_mod = [param_names[0], param_names[2], param_names[1]]
        C_mat = np.copy(C_mat_mod)
        final_bias_vector = np.copy(final_bias_vector_mod)

    if (1):
        Bdic, bias_dic, width_dic = {}, {}, {}
        width_vector = np.sqrt( np.diag(C_mat) )
        for pind, p in enumerate( param_names ):
            bias_dic[p] = final_bias_vector[pind]
            width_dic[p] = width_vector[pind]
            Bdic[p] = bias_vector[pind]

        print('\n\tFinal bias vector is')
        print('\t\t#param width bias fracsigma')
        for p in param_names_JM:
            if p in fix_params: continue
            print('\t\t%s %g %g %.2f$\sigma$' %(p, width_dic[p], bias_dic[p], bias_dic[p]/width_dic[p]))

    if (0):#not get_mle and not use_parameter_shifts_for_syserrors:
        #from IPython import embed; embed()
        #print(use_JM_fisher_mat); sys.exit()
        opfname = '%s_fisherbias_galresfrac%s' %(fisher_file.replace('.npy', ''), gal_res_frac)
        if use_JM_fisher_mat:
            opfname = '%s_JMfishermat' %(opfname)
        opfname = '%s.npy' %(opfname)
        op_dic = {}
        op_dic['ori_Fisher'] = fisher_dic
        op_dic['fix_params'] = fix_params
        op_dic['prior_dic'] = prior_dic
        op_dic['param_names'] = param_names
        op_dic['F_mat'] = F_mat
        op_dic['C_mat'] = C_mat
        op_dic['Bdic'] = Bdic
        op_dic['width_dic'] = width_dic
        op_dic['bias_dic'] = bias_dic
        np.save(opfname, op_dic)

        print('\n\tcheck this output file: %s' %(opfname))

get_mle = 1
if not get_mle: sys.exit()
########################################################
if (0): #perform likelihood calculation now for just single "neff" parameter - extend this later
    p = param_names[0]
    if p == 'neff':
        deltax, epsilon_x = 0.5, 0.025
    x = param_dict[p]
    x1, x2 = x-deltax, x+deltax
    x_arr = np.arange(x1, x2, epsilon_x)

    #create datavector and covariance vec
    data_only_tt, data_only_ee, data_only_te = cl_dic['TT'], cl_dic['EE'], cl_dic['TE']
    data_tt, data_ee, data_te = clplussys_dic['TT'] + nl_nogal_dic['TT'], clplussys_dic['EE'] + nl_nogal_dic['EE'], clplussys_dic['TE'] + nl_nogal_dic['TE']
    deltacl_tt, deltacl_ee, deltacl_te = delta_cl_dic['TT'], delta_cl_dic['EE'], delta_cl_dic['TE']

    #data_tt, data_ee, data_te = clplussys_dic['TT']+nl_dic['TT'], clplussys_dic['EE'], clplussys_dic['TE']

    datavec = np.concatenate( (data_tt, data_ee, data_te))
    deltaclvec =  np.concatenate( (deltacl_tt, deltacl_ee, deltacl_te))
    cinvvec = 1./deltaclvec

    modneff_camb_dic_fname = '%s/modneff_camb_dic.npy' %(camb_folder)
    if os.path.exists(modneff_camb_dic_fname):
        modneff_camb_dic = np.load(modneff_camb_dic_fname, allow_pickle = 1).item()
    else:
        modneff_camb_dic = {}

    logL_arr = []
    param_dict_mod = param_dict.copy()
    if which_spectra.find('_')>-1:
        which_spectra_mod = which_spectra.split('_')[0]
    else:
        which_spectra_mod = which_spectra

    diff_dic = {}
    for xcntr, xval in enumerate( x_arr ):
        keyname = round(xval, 4)
        if keyname in modneff_camb_dic:
            els_, curr_cl_dic = modneff_camb_dic[keyname]
        else:
            print('%s of %s: %s = %s' %(xcntr+1, len(x_arr), p, keyname))
            param_dict_mod[p] = xval
            pars, els, curr_cl_dic = tools.fn_set_CAMB_como(param_dict_mod, '%s_scalar' %(which_spectra_mod), use_thetastar = use_thetastar)
            modneff_camb_dic[keyname] = [els, curr_cl_dic]
            np.save(modneff_camb_dic_fname, modneff_camb_dic)

        model_tt = np.interp(els, els_, curr_cl_dic['TT'])
        model_ee = np.interp(els, els_, curr_cl_dic['EE'])
        model_te = np.interp(els, els_, curr_cl_dic['TE'])

        modelvec = np.concatenate( (model_tt, model_ee, model_te))
        logLval = 0.
        for lcntr in els:
            currcov = tools.get_cov_for_fisher(deltacl_tt[lcntr], deltacl_ee[lcntr], deltacl_te[lcntr])
            curr_datavec = [ data_tt[lcntr], data_ee[lcntr], data_te[lcntr] ]
            curr_modelvec = [ model_tt[lcntr], model_ee[lcntr], model_te[lcntr] ]

            '''
            currcov = currcov[-1, -1]
            curr_datavec = [ data_te[lcntr] ]
            curr_modelvec = [ model_te[lcntr] ]
            if xcntr%5 == 0:
                if keyname not in diff_dic:
                    diff_dic[keyname] = []
                diff_dic[keyname].append( curr_datavec[0] - curr_modelvec[0] )
            '''
            logLval += tools.get_likelihood( np.asarray( curr_datavec ), np.asarray( curr_modelvec ), currcov)

        print('%s of %s: %s = %s: logL = %s' %(xcntr+1, len(x_arr), p, keyname, logLval))

        logL_arr.append( logLval )

    if (0):
        #from IPython import embed; embed()
        print(diff_dic.keys())
        colorarr = [cm.jet(int(d)) for d in np.linspace(0, 255, len(diff_dic))]
        for cntr, k in enumerate( diff_dic ):
            ax = subplot(111, yscale = 'log')
            tmp = np.asarray( diff_dic[k] )
            plot(abs(tmp), color = colorarr[cntr], label = r'%s' %(k))
            '''
            ax = subplot(212, yscale = 'log')
            els_, curr_cl_dic = modneff_camb_dic[k]
            plot(curr_cl_dic['EE'], color = colorarr[cntr], label = r'%s' %(k))
            plot(cl_dic['EE'], color = 'k')
            '''
        legend(loc = 1)
        show(); sys.exit()

    #logL to likelihoods
    L_arr = np.exp( logL_arr - np.max(logL_arr) )
    L_arr /= np.max(L_arr)

    #get best-fit value and width
    recov_fit, recov_fit_low_err, recov_fit_high_err = np.asarray( tools.get_width_from_sampling(x_arr, L_arr) )
    recov_fit_width = (recov_fit_high_err + recov_fit_low_err)/2.
    print(recov_fit_width, p)

    #get Fisher constraints for overplotting
    xshift = recov_fit - x
    hor_fisher, ver_fisher = tools.get_Gaussian(x, width_dic[p], x1, x2)
    plot(hor_fisher + xshift, ver_fisher, color = 'black', lw = 0.2, ls = '-.')


    plot(x_arr, L_arr); axvline(x); 
    xlabel(r'N$_{\rm eff}$'); ylabel(r'Normalised likelihood $\mathcal{L}$')


    show()
    print(param_names)
    sys.exit()


########################################################
########################################################
#make corner plots
tr = tc = len(param_names)
desired_params_to_plot = []
for p in param_names:
    #if p == 'thetastar': continue
    desired_params_to_plot.append(p)
#fix_params = []

if (1):##not old_stepsizes: #based on Joel Meyers' step sizes
    cosmo_param_dict = {\
    'ombh2' : [0.0008,[0.0005, 0.000001]],\
    'omch2' : [0.0030,[0.005, 0.00001]],
    'tau' : [0.020, [0.02, 0.0001]],\
    'As' : [0.1e-9,[0.1e-9,1e-14]],\
    'ns' : [0.010, [0.02, 0.0001]],\
    'ws' : [-1e-2,[0.5, 0.0001]],\
    'neff': [0.080,[0.2,0.01]],\
    'mnu': [0.02,[1.,0.0001]],\
    ##############'YHe': [0.005,[1.,0.0001]],\
    ###'Alens': [1e-2, [1., 0.3]],\
    ###'Aphiphi': [1e-2, [1., 0.3]],\
    }
    if use_thetastar:
        cosmo_param_dict['thetastar'] = [0.000050, [0.00001, 0.0000001]]
        #cosmo_param_dict['thetastar'] = [0.000010, [0.05,0.001]]
    elif use_cosmomc_theta:
        cosmo_param_dict['cosmomc_theta'] = [0.000050, [0.00010,0.001]]
    else:
        #cosmo_param_dict['h']= [.00001, [0.05,0.001]]
        cosmo_param_dict['h']= [.005, [0.05,0.001]]
    if not set_YHe_using_BBN:
        cosmo_param_dict['YHe']= [0.005,[1.,0.0001]]

    if (0):#get_bias_vector:
        for pind, p in enumerate( param_names ):
            cosmo_param_dict[p][1][0] = cosmo_param_dict[p][1][0] + final_bias_vector[pind]

close()
fsval = 12
tr, tc = len(desired_params_to_plot), len(desired_params_to_plot)
if len(desired_params_to_plot)>1:
    figure(figsize=(tr + 3, tc + 3))
    subplots_adjust(hspace = 0.1, wspace = 0.2)
else:
    fsval = 18
if use_JM_fisher_mat:
    exparr = ['S4-Wide', 'S4-Wide with bias']
    colorarr = ['k', 'darkred', 'orangered']
    lsarr = ['-', '-', '-.']
else:
    exparr = [r'Fiducial', 'With bias']    
    colorarr = ['k', 'orangered']
    if (0): #20230414 - not using JM Fisher matrix.
        exparr = [r'SR (TT/EE/TE) + {\it Planck}', 'CT/JM (TT/EE/TE)']#, 'SR: With bias']    
        colorarr = ['k', 'lime', 'orangered']
    lsarr = ['-', '-', '-.']
import copy

if tmp_JM_opfname_str is not None:
    exparr[0] = '%s: JM spec/derivs' %(exparr[0])

use_percent = 0
for expcntr, exp in enumerate( exparr ):
    F_dic, color_dic, ls_dic = {}, {}, {}
    curr_F_mat = np.copy(F_mat)
    curr_param_names = np.copy(param_names)
    if exp.lower().find('ct/jm')>-1:
        curr_F_mat = np.copy(F_mat_JM)
        curr_param_names = np.copy(param_names_JM)

        #add priors
        print('\tadding prior for JM: %s' %(list(prior_dic.keys())))
        curr_F_mat = tools.fn_add_prior(np.copy(curr_F_mat), curr_param_names, prior_dic)

        #fix params
        print('\tfixing some parameters for JM: %s' %(fix_params))
        curr_F_mat, curr_param_names = tools.fn_fix_params(curr_F_mat, curr_param_names, fix_params)
        curr_param_names = np.asarray(curr_param_names)

    bias_dic = None
    param_dict_copy = copy.deepcopy( param_dict )
    if exp.lower().find('with bias')>-1:
        for pind, p in enumerate( curr_param_names ):
            #if p == 'As': final_bias_vector[pind] *= 1e-9
            param_dict_copy[p] = param_dict_copy[p] + final_bias_vector[pind]
        bias_dic = {}
        for pind, p in enumerate( curr_param_names ):
            bias_dic[p] = final_bias_vector[pind]

    print(exp)
    F_dic[exp] = curr_F_mat
    color_dic[exp] = [colorarr[expcntr]]
    ls_dic[exp] = [lsarr[expcntr]]

    tools.make_triangle_plot([exp], F_dic, param_dict_copy, cosmo_param_dict, tr, tc, curr_param_names, desired_params_to_plot, fix_params, color_dic, one_or_two_sigma = 1, fsval = fsval, noofticks = 3, use_percent = use_percent, ls_dic = ls_dic, bias_dic = bias_dic)

if (1):
    if len(param_names)>5:
        legsbpl = 6
    elif len(param_names)>1:
        legsbpl = 3
    elif len(param_names)==1:
        legsbpl = 1
    legsbpl = tc
    ax  = subplot(tr, tc, legsbpl)
    for expcntr, exp in enumerate( exparr ):
        labval = r'%s' %(exp)
        colorval = colorarr[expcntr]
        lsval = lsarr[expcntr]
        plot([],[], color = colorval, label = labval, ls = lsval)
        if legsbpl > 1:
            axis('off')
    if len(prior_dic)>0:
        priorstr = '-'.join( list(prior_dic.keys()) )
    else:
        priorstr = 'None'

    if which_spectra == 'lensed_scalar_Alens0.30':
        spec_legstr = 'Delensed'
    else:
        spec_legstr = which_spectra.replace('_scalar', '').capitalize()

    if use_parameter_shifts_for_syserrors:
        params_to_shift_str = '-'.join(params_to_shift)
        #legstr = r'Systematic shifts for [%s]= %s$\sigma$; Mask = %s; {\it Planck} = %s; Lensing = %s; Priors: %s; $(\ell_{\rm min}, \ell_{\rm max}) = (%d,%d)$' %(params_to_shift_str, shift_frac, galmask, add_planck, add_lensing, priorstr, lmin, lmax)
        legstr = r'%s: Systematic shifts for [%s]= %s$\sigma$; Mask = %s; {\it Planck} = %s; Lensing = %s; Priors: %s; $(\ell_{\rm min}, \ell_{\rm max}) = (%d,%d)$' %(spec_legstr, params_to_shift_str, shift_frac, galmask, add_planck, add_lensing, priorstr, lmin, lmax)
    else:
        #legstr = r'Resdial gal = %s; Mask = %s; {\it Planck} = %s; Lensing = %s; Priors: %s; $(\ell_{\rm min}, \ell_{\rm max}) = (%d,%d)$' %(gal_res_frac, galmask, add_planck, add_lensing, priorstr, lmin, lmax)
        legstr = r'%s: Resdial gal = %s; Mask = %s; {\it Planck} = %s; Lensing = %s; Priors: %s; $\ell_{\rm max_{\rm gal}} = %d$' %(spec_legstr, gal_res_frac, galmask, add_planck, add_lensing, priorstr, lmax_for_gal)
    #legstr = '%s' %(spec_legstr)
    #legstr1 = r'Mask = %s; {\it Planck} = %s; Lensing = %s$' %(galmask, add_planck, add_lensing)
    #legstr2 = r'Priors: %s; $(\ell_{\rm min}, \ell_{\rm max}) = (%d,%d)$' %(priorstr, lmin, lmax)
    #legstr = '%s\\%s' %(legstr1, legstr2)
    if legsbpl > 1:
        leg = legend(loc = 1, fontsize = fsval-1, framealpha = 0, handletextpad=0.8, handlelength = 1.5, numpoints = 1, columnspacing = 1, title = legstr)#, title_fontsize = fsval-1)
        leg._legend_box.align = 'left'
    else:
        title(legstr, fontsize = fsval-8)

    if (0):#len(exparr) > 1 and get_bias_vector:
        txtstr = r'$\ell_{\rm min} = %d; \ell_{\rm max} = %d$' %(lmin_for_gal, lmax_for_gal)
        figtext(0.7, 0.7, txtstr, fontsize = fsval)

    '''
    fishermat_txtstr = r'Fisher matrix from'
    if use_JM_fisher_mat:
        fishermat_txtstr = '%s CT/JM' %(fishermat_txtstr)
    else:
        fishermat_txtstr = '%s SR' %(fishermat_txtstr)
    figtext(0.7, 0.8, fishermat_txtstr, fontsize = fsval+2)
    '''

param_names_str = '-'.join(param_names)
plfolder = 'plots/20200701/fisher_bias/'
galbiastr = ''
if get_bias_vector:
    galbiastr = 'galbiasfor'
addplanckstr = ''
if add_planck:
    addplanckstr = '_withplanckonlargescales'
addlensingstr = ''
if add_lensing:
    addlensingstr = '_withlensing'
if use_parameter_shifts_for_syserrors:
    plname = '%s/galmask%s_%s%s_%s_%dshiftsmin%dlmaxforgal%s%s.png' %(plfolder, galmask, galbiastr, param_names_str, params_to_shift_str, lmin_for_gal, lmax_for_gal, addplanckstr, addlensingstr)
else:
    plname = '%s/galmask%s_%s%s_%sfrac_%dlmin%dlmaxforgal%s%s.png' %(plfolder, galmask, galbiastr, param_names_str, gal_res_frac, lmin_for_gal, lmax_for_gal, addplanckstr, addlensingstr)

#plname = '%s/SR_CTJMnoplanck.png' %(plfolder)
print(plname)
##savefig(plname, dpi = 200.)
show(); sys.exit()

##########
#sys.exit()
#compute systematic signal bsaed on parameter bias values
if (1):#use_parameter_shifts_for_syserrors:
    param_dict_low_res = copy.deepcopy(param_dict)
    if (1):
        param_dict_low_res['max_l_limit']=lmax
        param_dict_low_res['max_eta_k']=7000.0
        param_dict_low_res['max_eta_k_tensor']=3000.0
        param_dict_low_res['AccuracyBoost']=1
        param_dict_low_res['lAccuracyBoost']=1
        param_dict_low_res['lSampleBoost']=1

    print('\t\tShifting parameters for systematic error computations')
    import camb
    param_dict_fid = copy.deepcopy(param_dict_low_res)
    pars_, els_, cl_dic_fid = tools.fn_set_CAMB_como(param_dict_fid, which_spectra, add_lensing = add_lensing, use_thetastar = use_thetastar)

    #now compute modified cl with shifted parameters
    param_dict_biased = copy.deepcopy(param_dict_low_res)
    for p in bias_dic:
        shift_val = bias_dic[p]
        param_dict_biased[p] = param_dict_biased[p] + shift_val
        print('\t\t\tshifting %s: fiducial = %s; shift = %s; modified = %s' %(p, param_dict[p], shift_val, param_dict_biased[p]))

    pars_, els_, cl_dic_biased = tools.fn_set_CAMB_como(param_dict_biased, which_spectra, add_lensing = add_lensing, use_thetastar = use_thetastar)

    #now compute cl_sys_recov = cl_dic_fid - cl_dic_biased
    cl_sysdic_recov = {}
    for which_spec in nl_dic:
        cl_sys_recov = cl_dic_fid[which_spec] - cl_dic_biased[which_spec]
        cl_sysdic_recov[which_spec] = np.interp(els, np.arange(len(cl_sys_recov)), cl_sys_recov)
    #end = time.time(); print((end-start)/60.); sys.exit()


    if (1):
        close()
        figure(figsize=(10, 6))
        for which_spec_cntr, which_spec in enumerate( gal_dustsync_resdic ):
            ax = subplot(1,2,which_spec_cntr+1, yscale = 'log')
            plot(cl_dic[which_spec], 'gray')
            clsys = cl_sysdic[which_spec]
            clsys_recov = cl_sysdic_recov[which_spec]
            if use_parameter_shifts_for_syserrors:
                labsys = 'Systematic: parameter shifts'
            else:
                labsys = 'Systematic: gal residuals'
            plot(clsys, 'darkred', lw = 2., label = labsys)
            plot(clsys_recov, 'goldenrod', lw = 2., ls = '-.', label = '%s: recovered' %(labsys))
            plot(nl_nogal_dic[which_spec], 'darkgreen', label = r'Noise')
            xlim(0, 5000); ylim(1e-9, 1e-2); 
            if use_parameter_shifts_for_syserrors:
                params_to_shift_str = '-'.join(params_to_shift)
                titstr = r'%s: Systematic shifts for [%s]= %s$\sigma$; Mask = %s' %(which_spec, params_to_shift_str, shift_frac, galmask)
            else:
                titstr = r'%s: Residual gal = %s; Mask = %s' %(which_spec, gal_res_frac, galmask)

            title(titstr)
            if which_spec_cntr == 0:
                legend(loc = 1)#, fontsize = 8)
        show(); ##sys.exit()

print('hi');
sys.exit()
#close()
########################################################
#perform likelihood calculation now
p = param_names[0]
deltax, epsilon_x = cosmo_param_dict[p][1]
x = param_dict[p]
x1, x2 = x-deltax, x+deltax
x_arr = np.arange(x1, x2, epsilon_x)

logL_arr = []
param_dict_mod = param_dict.copy()
#datavec = 
for xval in x_arr:
    if (1):
        param_dict_mod['max_l_limit']=lmax
        param_dict_mod['max_eta_k']=10000.0
        param_dict_mod['max_eta_k_tensor']=3000.0
        param_dict_mod['AccuracyBoost']=1
        param_dict_mod['lAccuracyBoost']=1
        param_dict_mod['lSampleBoost']=1
    param_dict_mod[p] = xval
    pars, els, curr_cl_dic = tools.fn_set_CAMB_como(param_dict_mod, '%s_scalar' %(which_spectra), use_thetastar = use_thetastar)
    modelvec = np.concatenate( (curr_cl_dic['TT'], curr_cl_dic['EE'], curr_cl_dic['TE']))
print(param_names)
sys.exit()


########################################################
########################################################

