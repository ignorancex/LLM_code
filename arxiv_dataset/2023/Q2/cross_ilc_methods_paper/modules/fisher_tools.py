#For fisher forecasting
import numpy as np, sys, os, glob
from scipy.io import readsav
from scipy import interpolate as intrp
import scipy as sc
from scipy import integrate, stats
from pylab import *

def get_cl_ksz(els, Aksz = 1., dl_ksz_amp_total = 3.):
    dl_fac = els * (els+1)/2/np.pi
    dl_ksz = np.tile(dl_ksz_amp_total, len(els))
    cl_ksz = dl_ksz / dl_fac
    cl_ksz[np.isnan(cl_ksz)] = 0.
    cl_ksz[np.isinf(cl_ksz)] = 0.
    return Aksz * cl_ksz

def get_homo_ksz(els, Aksz_h = 1., alphaksz_h = 0., el_norm = 3000):
    dl_fac = els * (els+1)/2/np.pi
    dl_ksz_homo = Aksz_h * (els/el_norm)**alphaksz_h
    cl_ksz_homo = dl_ksz_homo/dl_fac

    return cl_ksz_homo

def get_knox_errors_parent(els, cl_dic, nl11_dic, fsky, nl22_dic = None, nl12_dic = None):

    delta_cl_dic = {}
    for XX in cl_dic:
        if XX == 'TT':
            nl11 = nl11_dic['TT']
            if nl22_dic is not None:
                nl22 = nl22_dic['TT']
                nl12 = nl12_dic['TT']
        elif XX == 'EE' or XX == 'BB':
            nl11 = nl11_dic['EE']
            if nl22_dic is not None:
                nl22 = nl22_dic['EE']
                nl12 = nl12_dic['EE']
        elif XX == 'TE':
            nl11 = np.copy(nl11_dic['TT']) * 0.
            if nl22_dic is not None:
                nl22 = np.copy(nl11) * 0.
                nl12 = np.copy(nl11) * 0.
        elif XX == 'PP':
            nl11 = np.copy(nl11_dic['PP'])
            if nl22_dic is not None:
                nl22 = np.copy(nl11) * 0.
                nl12 = np.copy(nl11) * 0.

        cl11 = cl_dic[XX] + nl11
        if nl22_dic is not None:
            cl22 = cl_dic[XX] + nl22
            cl12 = cl_dic[XX] + nl12
        else:
            cl22, cl12 = None, None

        delta_cl_dic[XX] = get_knox_errors(els, cl11, fsky, cl22 = cl22, cl12 = cl12)

    return delta_cl_dic

def get_knox_errors(els, cl11, fsky, cl22 = None, cl12 = None):

    """
    get Knox bandpower errors.

    els: multipoles
    cl11: signal + noise in the first map.
    cl22: signal + noise in the second map.
    cl12: signal + noise cross spectrum of the two maps.
    """

    delta_el = np.diff(els)[0]

    if cl22 is not None and cl12 is not None:
        cl_total = np.sqrt( (cl12**2. + cl11 * cl22)/2. )
    else:
        cl_total = cl11

    cl_knox_err = np.sqrt(2./ (2.*els + 1.) / fsky / delta_el ) * (cl_total)
    cl_knox_err[np.isnan(cl_knox_err)] = 0.
    cl_knox_err[np.isinf(cl_knox_err)] = 0.

    return cl_knox_err


def fix_params(F_mat, param_names, fix_params_arr):

    #remove parameters that must be fixed    
    F_mat_refined = []
    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):
            if p1 in fix_params_arr or p2 in fix_params_arr: continue
            F_mat_refined.append( (F_mat[pcntr2, pcntr1]) )

    totparamsafterfixing = int( np.sqrt( len(F_mat_refined) ) )
    F_mat_refined = np.asarray( F_mat_refined ).reshape( (totparamsafterfixing, totparamsafterfixing) )

    param_names_refined = []
    for p in param_names:
        if p in fix_params_arr: continue
        param_names_refined.append(p)

    param_names_refined = np.asarray(param_names_refined)


    return F_mat_refined, param_names_refined

def add_priors(F_mat, param_names, prior_dic):

    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):
            if p1 == p2 and p1 in prior_dic:
                prior_val = prior_dic[p1]
                F_mat[pcntr2, pcntr1] += 1./prior_val**2.

    return F_mat

def get_cov(TT, EE, TE, PP = 0., TP = 0., EP = 0.):

    C = np.zeros( (3,3) ) #TT, EE, PP
    C[0,0] = TT
    C[1,1] = EE
    C[0,1] = C[1,0] = TE

    C[2,2] = PP
    C[0,2] = C[2,0] = TP
    C[1,2] = C[2,1] = EP ##0. ##EP

    return np.mat( C )

def get_fisher_matrix(els, cl_deriv_dic, delta_cl_dic, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None):

    if min_l_temp is None: min_l_temp = 0
    if max_l_temp is None: max_l_temp = 10000

    if min_l_pol is None: min_l_pol = 0
    if max_l_pol is None: max_l_pol = 10000

    npar = len(params)
    F = np.zeros([npar,npar])
    #els = np.arange( len( delta_cl_dic.values()[0] ) )

    with_lensing = 0
    if 'PP' in pspectra_to_use:
        with_lensing = 1

    all_pspectra_to_use = []
    for tmp in pspectra_to_use:
        if isinstance(tmp, list):      
            all_pspectra_to_use.extend(tmp)
        else:
            all_pspectra_to_use.append(tmp)

    for lcntr, l in enumerate( els ):

        TT, EE, TE = 0., 0., 0.
        Tphi = Ephi = PP = 0.
        if 'TT' in delta_cl_dic:
            TT = delta_cl_dic['TT'][lcntr]
        if 'EE' in delta_cl_dic:
            EE = delta_cl_dic['EE'][lcntr]
        if 'TE' in delta_cl_dic:
            TE = delta_cl_dic['TE'][lcntr]
        if with_lensing:
            Tphi, Ephi, PP = delta_cl_dic['Tphi'][lcntr], delta_cl_dic['Ephi'][lcntr], delta_cl_dic['PP'][lcntr]

        null_TT, null_EE, null_TE = 0, 0, 0
        if l<min_l_temp or l>max_l_temp:
            null_TT = 1
        if l<min_l_pol or l>max_l_pol: 
            null_EE = 1
            null_TE = 1
        null_PP = 0 #Lensing noise curves already have pretty large noise outside desired L range
        #if l<min_l_TE or l>max_l_TE:  
        #    null_TE = 1

        #20200611
        if 'TT' not in all_pspectra_to_use:
            null_TT = 1
        if 'EE' not in all_pspectra_to_use:
            null_EE = 1
        if 'TE' not in all_pspectra_to_use:
            #if 'TT' not in pspectra_to_use and 'EE' not in pspectra_to_use:
            #    null_TE = 1
            if 'TT' in pspectra_to_use and 'EE' in pspectra_to_use:
                null_TE = 0
            else:
                null_TE = 1
        if ['TT', 'EE', 'TE'] in pspectra_to_use:
            null_TT = 0
            null_EE = 0
            null_TE = 0
        #20200611

        ##if (null_TT and null_EE and null_TE): continue# and null_PP): continue
        if (null_TT and null_EE and null_TE and null_PP): continue

        param_combinations = []
        for pcnt,p in enumerate(params):
            for pcnt2,p2 in enumerate(params):
                ##if [p2,p,pcnt2,pcnt] in param_combinations: continue
                param_combinations.append([p,p2, pcnt, pcnt2])

        def get_cov(TT, EE, TE, PP, TP, EP):

            C = np.zeros( (3,3) ) #TT, EE, PP
            C[0,0] = TT
            C[1,1] = EE
            C[0,1] = C[1,0] = TE

            C[2,2] = PP
            C[0,2] = C[2,0] = TP
            C[1,2] = C[2,1] = EP ##0. ##EP

            return np.mat( C )

        #nulling unwanted fields
        if null_TT and null_TE: TT = 0
        if null_EE and null_TE: EE = 0
        #if null_TE and (null_TT and null_EE): TE = 0
        if null_TE: 
            if not null_TT and not null_EE:
                pass
            else:
                TE = 0
        if null_PP: PP = Tphi = EPhi = 0
        if null_TT: Tphi = 0
        if null_EE: Ephi = 0
        #nulling unwanted fields

        COV_mat_l = get_cov(TT, EE, TE, PP, Tphi, Ephi)
        inv_COV_mat_l = sc.linalg.pinv2(COV_mat_l)
        #inv_COV_mat_l = np.linalg.inv(COV_mat_l)

        if (0):##l%500 == 0: 
            from IPython import embed; embed()
            print(l, null_TT, null_EE, null_TE, null_PP)
            print(COV_mat_l)

        for (p,p2, pcnt, pcnt2) in param_combinations:


            TT_der1, EE_der1, TE_der1 = 0., 0., 0.
            TT_der2, EE_der2, TE_der2 = 0., 0., 0.

            if 'TT' in cl_deriv_dic[p]:
                TT_der1 = cl_deriv_dic[p]['TT'][lcntr]
                TT_der2 = cl_deriv_dic[p2]['TT'][lcntr]
            if 'EE' in cl_deriv_dic[p]:
                EE_der1 = cl_deriv_dic[p]['EE'][lcntr]
                EE_der2 = cl_deriv_dic[p2]['EE'][lcntr]
            if 'TE' in cl_deriv_dic[p]:
                TE_der1 = cl_deriv_dic[p]['TE'][lcntr]
                TE_der2 = cl_deriv_dic[p2]['TE'][lcntr]


            if with_lensing:
                PP_der1, TPhi_der1, EPhi_der1 = cl_deriv_dic[p]['PP'][lcntr], cl_deriv_dic[p]['Tphi'][lcntr], cl_deriv_dic[p]['Ephi'][lcntr]
                PP_der2, TPhi_der2, EPhi_der2 = cl_deriv_dic[p2]['PP'][lcntr], cl_deriv_dic[p2]['Tphi'][lcntr], cl_deriv_dic[p2]['Ephi'][lcntr]
            else:
                PP_der1 = PP_der2 = 0.
                TPhi_der1 = TPhi_der2 = 0. 
                EPhi_der1 = EPhi_der2 = 0.


            if null_TT: TT_der1 = TT_der2 = TPhi_der1 = TPhi_der2 = 0
            if null_EE: EE_der1 = EE_der2 = EPhi_der1 = EPhi_der2 = 0
            if null_TE: TE_der1 = TE_der2 = 0
            if null_PP: PP_der1 = PP_der2 = 0

            fprime1_l_vec = get_cov(TT_der1, EE_der1, TE_der1, PP_der1, TPhi_der1, EPhi_der1)
            fprime2_l_vec = get_cov(TT_der2, EE_der2, TE_der2, PP_der2, TPhi_der2, EPhi_der2)

            curr_val = np.trace( np.dot( np.dot(inv_COV_mat_l, fprime1_l_vec), np.dot(inv_COV_mat_l, fprime2_l_vec) ) )

            F[pcnt2,pcnt] += curr_val

    return F   

def get_fisher_inv(F_mat):

    F_mat = np.asarray(F_mat)

    Flen = len(F_mat)
    all_inds = np.arange(Flen)

    F_mat_diag = np.diag(F_mat)
    good_inds = np.where(F_mat_diag > 0)[0]
    Flen_refined = len(good_inds)

    #from IPython import embed; embed(); sys.exit()

    F_mat_refined = []
    used_i, used_j = [], []
    for i in all_inds:
        for j in all_inds:
            if i in good_inds and j in good_inds: 
                F_mat_refined.append( (F_mat[j, i]) )
                used_i.append(i)
                used_j.append(j)
    used_i = np.asarray(used_i)
    used_j = np.asarray(used_j)
    F_mat_refined = np.asarray( F_mat_refined ).reshape( (Flen_refined, Flen_refined) )
    C_mat_refined = np.linalg.inv(F_mat_refined)

    C_mat = np.zeros(F_mat.shape)
    C_mat[used_j, used_i] = C_mat_refined.flatten()

    return C_mat

def get_sigma_of_a_parameter(F_mat, param_names, desired_param, prior_dic = None, fix_params_arr = None):

    F_mat_mod = np.copy(F_mat)
    param_names_mod = np.copy(param_names)

    param_names_mod = np.asarray( param_names_mod )
    fix_params_arr = np.asarray( fix_params_arr )

    if prior_dic is not None: #add priors.
        F_mat_mod = add_priors(F_mat_mod, param_names_mod, prior_dic)

    if fix_params_arr is not None:
        F_mat_mod, param_names_mod = fix_params(F_mat_mod, param_names_mod, fix_params_arr)

    cov_mat = get_fisher_inv(F_mat_mod)
    
    param_names_mod = np.asarray( param_names_mod )
    pind = np.where(param_names_mod == desired_param)[0][0]
    pcntr1, pcntr2 = pind, pind
    cov_inds_to_extract = [(pcntr1, pcntr1), (pcntr1, pcntr2), (pcntr2, pcntr1), (pcntr2, pcntr2)]
    cov_extract = np.asarray( [cov_mat[ii] for ii in cov_inds_to_extract] ).reshape((2,2))
    sigma_val = cov_extract[0,0]**0.5

    return sigma_val

def get_bias_vector(els, F_mat, param_names, delta_cl_dic, cl_deriv_dic, cl_sys_dic, max_l = 5000, pspectra_to_use = ['TT', 'EE', 'TE']):
    null_TT, null_EE, null_TE = 0, 0, 0
    bias_vector = []
    for pcntr, p in enumerate( param_names ):
        curr_bias_vector = []
        for l in range(max_l):
            TT, EE, TE = delta_cl_dic['TT'][l], delta_cl_dic['EE'][l], delta_cl_dic['TE'][l]
            TT_der, EE_der, TE_der = cl_deriv_dic[p]['TT'][l], cl_deriv_dic[p]['EE'][l], cl_deriv_dic[p]['TE'][l]
            TT_sys, EE_sys, TE_sys = cl_sys_dic['TT'][l], cl_sys_dic['EE'][l], cl_sys_dic['TE'][l]

            if 'TT' not in pspectra_to_use:
                null_TT = 1
            if 'EE' not in pspectra_to_use:
                null_EE = 1
            if 'TE' not in pspectra_to_use:
                if 'TT' in pspectra_to_use and 'EE' in pspectra_to_use:
                    null_TE = 0
                else:
                    null_TE = 1
            if ['TT', 'EE', 'TE'] in pspectra_to_use:
                null_TT = 0
                null_EE = 0
                null_TE = 0

            #nulling unwanted fields
            if null_TT and null_TE: 
                TT = TT_der = TT_sys = 0
            if null_EE and null_TE: 
                EE = EE_der = EE_sys = 0
            if null_TE: 
                if not null_TT and not null_EE:
                    pass
                else:
                    TE = TE_der = TE_sys = 0
            #nulling unwanted fields


            curr_fisher_cov = get_cov(TT, EE, TE)
            inv_curr_fisher_cov = get_fisher_inv(curr_fisher_cov)

            sysres_vec = get_cov(TT_sys, EE_sys, TE_sys)
            der_vec = get_cov(TT_der, EE_der, TE_der)
            curr_bias_val_with_trace = np.trace( np.dot( np.dot(inv_curr_fisher_cov, sysres_vec), np.dot(inv_curr_fisher_cov, der_vec) ) )
            ##print(curr_bias_val_with_trace); sys.exit()

            curr_bias_vector.append( curr_bias_val_with_trace )
        bias_vector.append( np.sum(curr_bias_vector) )
        ##print('\t\t%s, %s' %(p, bias_vector[pcntr]))
        
    bias_vector = np.asarray(bias_vector)
    C_mat = np.linalg.inv(F_mat)
    final_bias_vector = np.asarray( np.dot( np.mat(bias_vector), C_mat ) )[0]
    
    return final_bias_vector

def get_cmb_power_spectra_from_camb(param_dict, raw_cl =  True, which_spectra = 'lensed'):
    which_spectra = '%s_scalar' %(which_spectra)
    print(which_spectra)
    import camb
    pars = camb.CAMBparams(max_l_tensor = param_dict['max_l_tensor'], max_eta_k_tensor = param_dict['max_eta_k_tensor'])
    pars.set_for_lmax(param_dict['max_l_limit'], lens_potential_accuracy=param_dict['lens_potential_accuracy'])
    pars.set_accuracy(AccuracyBoost = param_dict['AccuracyBoost'], lAccuracyBoost = param_dict['lAccuracyBoost'], lSampleBoost = param_dict['lSampleBoost'],\
        DoLateRadTruncation = param_dict['do_late_rad_truncation'])
    pars.set_cosmology(thetastar=param_dict['thetastar'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], \
        omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], \
        num_massive_neutrinos = param_dict['num_nu_massive']) 

    pars.set_for_lmax(int(param_dict['max_l_limit']), lens_potential_accuracy=param_dict['lens_potential_accuracy'],\
        max_eta_k = param_dict['max_eta_k'],\
        #lens_k_eta_reference = param_dict['max_eta_k'],\
        )
    pars.InitPower.set_params(ns=param_dict['ns'], r=param_dict['r'], As = param_dict['As'])

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, raw_cl = raw_cl, lmax = param_dict['max_l_limit'])#, spectra = [which_spectra])#, CMB_unit=None, raw_cl=False)
    if pars.OutputNormalization == 1:
        powers[which_spectra] = param_dict['T_cmb']**2. *  powers[which_spectra]

    cl_tt, cl_ee, cl_te, cl_bb = powers[which_spectra].T * 1e12
    els = np.arange( len(cl_tt) )

    if (0):
        ax  =subplot(111, yscale = 'log')
        plot(els, cl_tt, color = 'black')
        xlim(0., 5000.); ylim(1e-8, 1e-2); 
        show(); sys.exit()
        sys.exit()

    cl_dic={}
    cl_dic['TT'], cl_dic['EE'], cl_dic['TE'], cl_dic['BB'] = cl_tt, cl_ee, cl_te, cl_bb

    return els, cl_dic

def simple_fisher_bias_check_introduce_sys_by_changing_cosmo_params(sys_bias_param_name_arr, sys_bias_param_shift_arr, raw_cl = True, sys_bias_spectra_arr_to_be_shifted = ['TT', 'EE', 'TE'], which_spectra = 'lensed', debug = False, fpath = 'publish/data/params_planck_r_0.0_2018_cosmo.txt', tmpels = None, tmpcltt = None):

    import tools_for_plotting

    parent_cl_dic = {}
    for iter in range(2):

        param_dict = tools_for_plotting.get_ini_param_dict(fpath = fpath)
        if iter == 1:
            for sys_bias_param_name, sys_bias_param_shift in zip(sys_bias_param_name_arr, sys_bias_param_shift_arr):
                param_dict[sys_bias_param_name] = sys_bias_param_shift

        #print(sys_bias_param_name, param_dict[sys_bias_param_name])
        ##print(param_dict)
        els, cl_dic = get_cmb_power_spectra_from_camb(param_dict, raw_cl =  raw_cl, which_spectra = which_spectra)

        parent_cl_dic[iter] = cl_dic
        ##print(cl_dic.keys()); 
    ##sys.exit()
        
    cl_dic_fid = parent_cl_dic[0]

    if (0):
        els_, cl_dic_fid, cl_dic_mod = els, parent_cl_dic[0], parent_cl_dic[1]
        ax  =subplot(111, yscale = 'log')
        plot(tmpels, tmpcltt, color = 'gray')
        plot(els_, cl_dic_fid['TT'], color = 'black')
        plot(els_, cl_dic_mod['TT'], color = 'red')
        plot(els_, cl_dic_fid['TT'] - cl_dic_mod['TT'], color = 'lime')
        xlim(0., 5000.); ylim(1e-8, 1e-2); 
        show(); sys.exit()

    cl_sys_shift_dic = {}
    for sys_bias_spectra_to_be_shifted in sys_bias_spectra_arr_to_be_shifted:
        cl_sys_shift = parent_cl_dic[0][sys_bias_spectra_to_be_shifted] - parent_cl_dic[1][sys_bias_spectra_to_be_shifted]
        cl_sys_shift_dic[sys_bias_spectra_to_be_shifted] = cl_sys_shift
    if debug:
        color_arr = ['navy', 'red']
        for iter in range(2):
            dl_fac = els * (els+1)
            ax = subplot(111, yscale = 'log')
            plot(els, dl_fac * parent_cl_dic[iter][sys_bias_spectra_to_be_shifted], color = color_arr[iter])
        plot(els, dl_fac * cl_sys_shift, color = 'black')
        plot(els, dl_fac * -cl_sys_shift, color = 'black', ls= '-.')
        show(); sys.exit()

    return els, cl_dic_fid, cl_sys_shift_dic

def simple_fisher_bias_check_introduce_sys_by_changing_cosmo_params_single_parameter(sys_bias_param_name, sys_bias_param_shift, raw_cl = True, sys_bias_spectra_arr_to_be_shifted = ['TT', 'EE', 'TE'], which_spectra = 'lensed_scalar', debug = False, fpath = 'publish/data/params_planck_r_0.0_2018_cosmo.txt', tmpels = None, tmpcltt = None):

    import tools_for_plotting
    
    parent_cl_dic = {}
    for iter in range(2):

        param_dict = tools_for_plotting.get_ini_param_dict(fpath = fpath)
        if iter == 1:
            param_dict[sys_bias_param_name] = sys_bias_param_shift

        #print(sys_bias_param_name, param_dict[sys_bias_param_name])
        els, cl_dic = get_cmb_power_spectra_from_camb(param_dict, raw_cl =  raw_cl, which_spectra = which_spectra)

        parent_cl_dic[iter] = cl_dic

    cl_dic_fid = parent_cl_dic[0]

    if (0):
        els_, cl_dic_fid, cl_dic_mod = els, parent_cl_dic[0], parent_cl_dic[1]
        ax  =subplot(111, yscale = 'log')
        plot(tmpels, tmpcltt, color = 'gray')
        plot(els_, cl_dic_fid['TT'], color = 'black')
        plot(els_, cl_dic_mod['TT'], color = 'red')
        plot(els_, cl_dic_fid['TT'] - cl_dic_mod['TT'], color = 'lime')
        xlim(0., 5000.); ylim(1e-8, 1e-2); 
        show(); sys.exit()

    cl_sys_shift_dic = {}
    for sys_bias_spectra_to_be_shifted in sys_bias_spectra_arr_to_be_shifted:
        cl_sys_shift = parent_cl_dic[0][sys_bias_spectra_to_be_shifted] - parent_cl_dic[1][sys_bias_spectra_to_be_shifted]
        cl_sys_shift_dic[sys_bias_spectra_to_be_shifted] = cl_sys_shift
    if debug:
        color_arr = ['navy', 'red']
        for iter in range(2):
            dl_fac = els * (els+1)
            ax = subplot(111, yscale = 'log')
            plot(els, dl_fac * parent_cl_dic[iter][sys_bias_spectra_to_be_shifted], color = color_arr[iter])
        plot(els, dl_fac * cl_sys_shift, color = 'black')
        plot(els, dl_fac * -cl_sys_shift, color = 'black', ls= '-.')
        show(); sys.exit()

    return els, cl_dic_fid, cl_sys_shift_dic
