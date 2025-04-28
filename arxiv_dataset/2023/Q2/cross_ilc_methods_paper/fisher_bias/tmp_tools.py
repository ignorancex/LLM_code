import numpy as np, camb, sys, scipy as sc, os
from camb import model, initialpower
from pylab import *
from copy import deepcopy


########################################################################################################################

def fn_ini_param_dic(fpath = 'params/params_planck_r_0.0_2015_cosmo_lensed_LSS.txt'):
    """
    read params file and initialise cosmology
    """
    try:
        params = np.recfromtxt(fpath, delimiter = '=', encoding = 'utf-8')
    except:
        params = np.recfromtxt(fpath, delimiter = '=')
    param_dict = {}
    for rec in params:
        val = rec[1].strip()##.decode("utf-8")
        try:
            if val.find('.')>-1:
                val = float(val)
            else:
                val = int(val)
        except:
            val = str(val)

        if val == 'None':
            val = None
        paramname = rec[0].strip()#.decode("utf-8")
        param_dict[paramname] = val

    return param_dict

########################################################################################################################

def fn_set_CAMB_como(param_dict, which_spectra, param_dict_derivatives = None, raw_cl = 1, high_low = 0, add_lensing = 0, use_thetastar = 0, use_cosmomc_theta = 0, derived_param = None, both_Cl_param = 0):

    """
    set CAMB cosmology and get power spectra
    """

    ###print( '\n\n\tcheck order here: https://camb.readthedocs.io/en/latest/camb.html\n\n' )

    #20200623 - setting accuracy/lmax seprately
    '''
    pars = camb.CAMBparams(min_l = param_dict['min_l_limit'], max_l = param_dict['max_l_limit'], \
        AccuracyBoost = param_dict['AccuracyBoost'], lAccuracyBoost = param_dict['lAccuracyBoost'], lSampleBoost = param_dict['lSampleBoost'],\
        max_l_tensor = param_dict['max_l_tensor'], max_eta_k = param_dict['max_eta_k'], max_eta_k_tensor = param_dict['max_eta_k_tensor'],\
        DoLateRadTruncation = param_dict['do_late_rad_truncation'],\
        )
    '''
    pars = camb.CAMBparams(max_l_tensor = param_dict['max_l_tensor'], max_eta_k_tensor = param_dict['max_eta_k_tensor'])
    #20200623 - setting accuracy/lmax seprately
    pars.set_accuracy(AccuracyBoost = param_dict['AccuracyBoost'], lAccuracyBoost = param_dict['lAccuracyBoost'], lSampleBoost = param_dict['lSampleBoost'],\
        DoLateRadTruncation = param_dict['do_late_rad_truncation'])
    ###pars.set_for_lmax(int(param_dict['max_l_limit']), lens_potential_accuracy=param_dict['lens_potential_accuracy'])

    #now see if mod_param_dict exists and change the cosmology accordingly
    if param_dict_derivatives is not None:
        param_dict_mod = param_dict.copy()
        #param_dict_mod = param_dict.deepcopy()
        #print param_dict_mod        

        for keyname in param_dict_derivatives.keys():
            if derived_param is None:
                print('\t\tModifying %s for derivative now: (%s,%s)' %(keyname, param_dict_mod[keyname], param_dict_derivatives[keyname]))
            if high_low == 0:
                param_dict_mod[keyname] = param_dict_mod[keyname] + param_dict_derivatives[keyname]
            else:
                param_dict_mod[keyname] = param_dict_mod[keyname] - param_dict_derivatives[keyname]

        #Alens or Aphiphi for derivatives
        if keyname == 'Aphiphi':
            Alensval = param_dict_mod[keyname]
        else:
            Alensval = param_dict_mod['Alens']

        pars.set_dark_energy(param_dict_mod['ws'])
        ###pars.InitPower.set_params(ns=param_dict_mod['ns'], r=param_dict_mod['r'], As = param_dict_mod['As'])
        if use_thetastar:
            pars.set_cosmology(thetastar=param_dict_mod['thetastar'], ombh2=param_dict_mod['ombh2'], omch2=param_dict_mod['omch2'], nnu = param_dict_mod['neff'], mnu=param_dict_mod['mnu'], \
                #omk=param_dict_mod['omk'], tau=param_dict_mod['tau'], YHe = param_dict_mod['YHe'], Alens = param_dict_mod['Alens'], \
                omk=param_dict_mod['omk'], tau=param_dict_mod['tau'], YHe = param_dict_mod['YHe'], Alens = Alensval, \
                num_massive_neutrinos = param_dict_mod['num_nu_massive'])            
            #pars.set_cosmology(cosmomc_theta=param_dict_mod['thetastar'], ombh2=param_dict_mod['ombh2'], omch2=param_dict_mod['omch2'], nnu = param_dict_mod['neff'], mnu=param_dict_mod['mnu'], omk=param_dict_mod['omk'], tau=param_dict_mod['tau'], YHe = param_dict_mod['YHe'], Alens = param_dict_mod['Alens'], num_massive_neutrinos = param_dict_mod['num_nu_massive'])            
        elif use_cosmomc_theta:
            pars.set_cosmology(cosmomc_theta=param_dict_mod['cosmomc_theta'], ombh2=param_dict_mod['ombh2'], omch2=param_dict_mod['omch2'], nnu = param_dict_mod['neff'], mnu=param_dict_mod['mnu'], \
                #omk=param_dict_mod['omk'], tau=param_dict_mod['tau'], YHe = param_dict_mod['YHe'], Alens = param_dict_mod['Alens'], \
                omk=param_dict_mod['omk'], tau=param_dict_mod['tau'], YHe = param_dict_mod['YHe'], Alens = Alensval, \
                num_massive_neutrinos = param_dict_mod['num_nu_massive'])
        else:
            pars.set_cosmology(H0=param_dict_mod['h']*100., ombh2=param_dict_mod['ombh2'], omch2=param_dict_mod['omch2'], nnu = param_dict_mod['neff'], mnu=param_dict_mod['mnu'], \
                #omk=param_dict_mod['omk'], tau=param_dict_mod['tau'], YHe = param_dict_mod['YHe'], Alens = param_dict_mod['Alens'], \
                omk=param_dict_mod['omk'], tau=param_dict_mod['tau'], YHe = param_dict_mod['YHe'], Alens = Alensval, \
                num_massive_neutrinos = param_dict_mod['num_nu_massive'])

        #20200619
        #print('\n\tswitching order on 20200619 following https://camb.readthedocs.io/en/latest/camb.html\n')
        #pars.set_for_lmax(int(param_dict['max_l_limit']), lens_potential_accuracy=param_dict['lens_potential_accuracy'])
        pars.set_for_lmax(int(param_dict_mod['max_l_limit']), lens_potential_accuracy=param_dict_mod['lens_potential_accuracy'],\
            max_eta_k = param_dict_mod['max_eta_k'],\
            )
        if param_dict_mod['As']>3.:
            pars.InitPower.set_params(ns=param_dict_mod['ns'], r=param_dict_mod['r'], As = np.exp(param_dict_mod['As'])/1e10)
        else:
            pars.InitPower.set_params(ns=param_dict_mod['ns'], r=param_dict_mod['r'], As = param_dict_mod['As'])
        #20200619

    else:
        pars.set_dark_energy(param_dict['ws'])
        ###pars.InitPower.set_params(ns=param_dict['ns'], r=param_dict['r'], As = param_dict['As'])
        if use_thetastar:
            pars.set_cosmology(thetastar=param_dict['thetastar'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], \
                omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], \
                num_massive_neutrinos = param_dict['num_nu_massive'])
            #pars.set_cosmology(cosmomc_theta=param_dict['thetastar'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], num_massive_neutrinos = param_dict['num_nu_massive'])
        elif use_cosmomc_theta:
            pars.set_cosmology(cosmomc_theta=param_dict['cosmomc_theta'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], \
                omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], \
                num_massive_neutrinos = param_dict['num_nu_massive'])
            #pars.set_cosmology(cosmomc_theta=param_dict['thetastar'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], num_massive_neutrinos = param_dict['num_nu_massive'])
        else:
            pars.set_cosmology(H0=param_dict['h']*100., ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], \
                omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], \
                num_massive_neutrinos = param_dict['num_nu_massive'])
        #20200619
        #print('\n\tswitching order on 20200619 following https://camb.readthedocs.io/en/latest/camb.html\n')
        #pars.set_for_lmax(int(param_dict['max_l_limit']), lens_potential_accuracy=param_dict['lens_potential_accuracy'])
        pars.set_for_lmax(int(param_dict['max_l_limit']), lens_potential_accuracy=param_dict['lens_potential_accuracy'],\
            max_eta_k = param_dict['max_eta_k'],\
            #lens_k_eta_reference = param_dict['max_eta_k'],\
            )

        if param_dict['As']>3.:
            pars.InitPower.set_params(ns=param_dict['ns'], r=param_dict['r'], As = np.exp(param_dict['As'])/1e10)
        else:
            pars.InitPower.set_params(ns=param_dict['ns'], r=param_dict['r'], As = param_dict['As'])
        #20200619

    '''
    #20200623
    pars.WantTransfer = True
    pars.set_matter_power(redshifts=[0.], kmax=2.0)
    print('\n\tincluding pars.WantTransfer for sigma8 calculation\n')
    #20200623
    '''

    ##from IPython import embed; embed()

    #get results
    results = camb.get_results(pars)
    #print(pars.InitPower.As)
    #print(results.get_derived_params()['thetastar'])


    if param_dict_derivatives is not None:
        param_to_be_mod = list(param_dict_derivatives.keys())[0]
    else:
        param_to_be_mod = None

    if param_to_be_mod is not None and param_to_be_mod != 'mnu' and param_to_be_mod != 'neff':
        line_to_print = 'param to be mod = %10s' %(param_to_be_mod)
        line_to_print = '%s; As = %.3f; ns = %.4f; 100ombh2 = %.4f; omch2 = %.4f' %(line_to_print, pars.InitPower.As * 1e9, pars.InitPower.ns, 100.*pars.ombh2, pars.omch2)
        line_to_print = '%s; tau = %.4f; 100thetastar = %.7f; 100thetaMC = %.7f; H0 = %.4f' %(line_to_print, pars.Reion.optical_depth, 100.*results.get_derived_params()['thetastar'], 100.*results.cosmomc_theta(), 100.*pars.h)
        print(line_to_print)
        #print(param_dict_derivatives.keys(), pars.ombh2, pars.omch2, results.get_derived_params()['thetastar'], results.cosmomc_theta(), pars.InitPower.ns, pars.InitPower.As, pars.Reion.optical_depth, pars.h, derived_param_val)

    if derived_param is not None:
        derived_param = np.asarray( derived_param )
        if np.ndim(derived_param) > 1:
            derived_param_val_arr = []
            for tmp in derived_param:
                derived_param_name, derived_param_cmd = tmp
                derived_param_val = eval( derived_param_cmd )
                derived_param_val_arr.append( derived_param_val )
            derived_param_val = np.asarray( derived_param_val_arr )
        else:
            derived_param_name, derived_param_cmd = derived_param
            derived_param_val = eval( derived_param_cmd )

        #from IPython import embed; embed()
        #print(param_dict_derivatives.keys(), pars.ombh2, pars.omch2, results.get_derived_params()['thetastar'], results.cosmomc_theta(), pars.InitPower.ns, pars.InitPower.As, pars.Reion.optical_depth, pars.h, derived_param_val)

        if not both_Cl_param:
            return derived_param_val

    #get dictionary of CAMB power spectra
    powers = results.get_cmb_power_spectra(pars, lmax = param_dict['max_l_limit'], raw_cl = raw_cl)#, spectra = [which_spectra])#, CMB_unit=None, raw_cl=False)
    #powers = results.get_cmb_power_spectra(pars, lmax = param_dict['max_l_limit'], raw_cl = raw_cl, CMB_unit = 'muK')#, spectra = [which_spectra])#, CMB_unit=None, raw_cl=False)
    #powers_v2 = results.get_cmb_power_spectra(pars, lmax = param_dict['max_l_limit'], raw_cl = raw_cl, CMB_unit = 'muK')#, spectra = [which_spectra])#, CMB_unit=None, raw_cl=False)

    #from IPython import embed; embed()
    if (0):
        from IPython import embed; embed()
        unlensed, lensed, phi = powers['unlensed_scalar'], powers['lensed_scalar'], powers['lens_potential']
        total = powers['total']
        #unlensed_v2, lensed_v2, phi_v2 = powers['unlensed_scalar'], powers['lensed_scalar'], powers['lens_potential']

        #powers = results.get_cmb_power_spectra(pars, lmax = param_dict['max_l_limit'], raw_cl = 0)#, spectra = [which_spectra])#, CMB_unit=None, raw_cl=False)
        #unlensed, lensed, phi = powers['unlensed_scalar'], powers['lensed_scalar'], powers['lens_potential']

        ax = subplot(111, yscale = 'log');plot(unlensed[:,0], 'k-'); # plot(unlensed_v2[:,0], 'lime'); show()
        ax = subplot(111, yscale = 'log');plot(lensed[:,0], 'g-'); #plot(lensed_v2[:,0]/lensed[:,0]); show()
        ax = subplot(111, yscale = 'log');plot(total[:,0], 'r-'); #plot(lensed_v2[:,0]/lensed[:,0]); show()
        show()
        ax = subplot(111, yscale = 'log');plot(phi[:,0], 'b-'); show()#plot(phi_v2[:,0], 'lime'); show()
        sys.exit()

    #get only the required ell range since powerspectra start from ell=0 by default
    for keyname in powers:
        powers[keyname] = powers[keyname][param_dict['min_l_limit']:, :]

    els = np.arange(param_dict['min_l_limit'], param_dict['max_l_limit']+1)
    if not raw_cl: #20200529: also valid for lensing (see https://camb.readthedocs.io/en/latest/_modules/camb/results.html#CAMBdata.get_lens_potential_cls)
        powers[which_spectra] = powers[which_spectra] * 2 * np.pi / (els[:,None] * (els[:,None] + 1 ))

    if which_spectra == 'lens_potential':

        Cl_phiphi, Cl_Tphi, Cl_Ephi = powers[which_spectra].T

        #K or uK
        if param_dict['uK'] == 1:
            Cl_Tphi *= 1e6##1e12
            Cl_Ephi *= 1e6##1e12

        if (1):
            Cl_phiphi = Cl_phiphi * (els * (els+1))**2. /(2. * np.pi)
            Cl_Tphi = Cl_Tphi * (els * (els+1))**1.5 /(2. * np.pi)
            Cl_Ephi = Cl_Ephi * (els * (els+1))**1.5 /(2. * np.pi)
        
        Cl_dic = {}
        Cl_dic['PP'] = Cl_phiphi
        Cl_dic['Tphi'] = Cl_Tphi
        Cl_dic['Ephi'] = Cl_Ephi

        if (0):
            phi_kappa_fac =1.#( (els * (els+1))**2. /4. )
            clf()
            ax = subplot(111, yscale = 'log')
            plot(els, Cl_phiphi * phi_kappa_fac)
            ylabel(r'$C_{L}^{\kappa \kappa}$', fontsize = 14)
            ylim(1e-9,1e-4);
            show()
            sys.exit()

    else:
        #Tcmb factor
        if pars.OutputNormalization == 1:
            powers[which_spectra] = param_dict['T_cmb']**2. *  powers[which_spectra]

        #K or uK
        if param_dict['uK'] == 1:
            powers[which_spectra] *= 1e12

        Cl_TT, Cl_EE, Cl_BB, Cl_TE = powers[which_spectra].T

        Cl_dic = {}
        Cl_dic['TT'] = Cl_TT
        Cl_dic['EE'] = Cl_EE
        Cl_dic['BB'] = Cl_BB
        Cl_dic['TE'] = Cl_TE

    if (0):
        #from IPython import embed; embed()  
        ax = subplot(111, yscale = 'log');
        dls_fac = (els * (els+1)) /(2. * np.pi)
        plot(Cl_TT * dls_fac, 'k-'); 
        plot(Cl_EE * dls_fac, 'r-'); plot(Cl_TE * dls_fac, 'g-'); 
        plot(Cl_BB * dls_fac, 'b-'); 
        show()
        sys.exit()

    if add_lensing:

        Cl_phiphi, Cl_Tphi, Cl_Ephi = powers['lens_potential'].T

        #K or uK
        if param_dict['uK'] == 1:
            Cl_Tphi *= 1e6##1e12
            Cl_Ephi *= 1e6##1e12

        '''
        if (0):
            from IPython import embed; embed()
            powers_v2 = results.get_cmb_power_spectra(pars, lmax = param_dict['max_l_limit'], raw_cl = 0)

            Cl_phiphi_v2, Cl_Tphi_v2, Cl_Ephi_v2 = powers_v2['lens_potential'].T

            nlfile = 'data/DRAFT/results/20200601/lensing_noise_curves/20200601/s4like_mask/TT-EE-TE/baseline/ilc/S4_ilc_galaxy1_27-39-93-145-225-278_TT-EE-TE_lf2-mf12-hf5_galmask0_AZ_lmin100_lmax5000_lmaxtt3000.npy'
            result_dic = np.load(nlfile, allow_pickle = 1, encoding = 'latin1').item()
            lmin, lmax, lmax_tt = result_dic['lmin'], result_dic['lmax'], result_dic['lmax_tt']
            el, cl_kk = result_dic['els'], result_dic['cl_kk']

            inds = np.arange(10, 2800)
            el = el[inds] 
            cl_kk = cl_kk[inds] 
            Cl_phiphi = Cl_phiphi[inds] 
            Cl_phiphi_v2 = Cl_phiphi_v2[inds] 
            els = els[inds]

            #phi_kappa_fac = 1.#( (els * (els+1))**2. /4. )
            phi_kappa_fac = (els * (els+1))**2. /(2. * np.pi)
            clf()
            ax = subplot(111, yscale = 'log')
            plot(els, Cl_phiphi * phi_kappa_fac)
            plot(els, Cl_phiphi_v2)
            plot(el, cl_kk)
            ylabel(r'$C_{L}^{\kappa \kappa}$', fontsize = 14)
            xlim(1,3000);#lmax);
            ylim(1e-10,1e-4);
            show()
            sys.exit()
        '''

        if (1):
            Cl_phiphi = Cl_phiphi * (els * (els+1))**2. /(2. * np.pi)
            Cl_Tphi = Cl_Tphi * (els * (els+1))**1.5 /(2. * np.pi)
            Cl_Ephi = Cl_Ephi * (els * (els+1))**1.5 /(2. * np.pi)

        
        Cl_dic['PP'] = Cl_phiphi
        Cl_dic['Tphi'] = Cl_Tphi
        Cl_dic['Ephi'] = Cl_Ephi

    if derived_param is not None:
        return pars, els, Cl_dic, derived_param_val
    else:
        return pars, els, Cl_dic

    #return pars, els, Cl_TT, Cl_EE, Cl_BB, Cl_EE
    #for name in powers: print name

########################################################################################################################

def fn_el_binning(els, Cl, delta_l = 1):
    if delta_l == 1:
        return els, Cl
    binned_el = np.arange(min(els), max(els)+delta_l, delta_l)
    binned_Cl = np.asarray( [np.mean(Cl[el:el+delta_l]) for el in binned_el] )
    binned_Cl[np.isnan(binned_Cl)] = 1e10
    #binned_el = binned_el + int(delta_l/2.)

    return binned_el, binned_Cl

########################################################################################################################

def fn_get_dervatives_Aphiphi(Cl_dic, Aphiphi = 1., delta_Aphiphi = 1e-3):

    cl_phiphi = Cl_dic['PP']

    cl_phiphi_mod_lower = cl_phiphi * (Aphiphi - delta_Aphiphi)
    cl_phiphi_mod_higher = cl_phiphi * (Aphiphi + delta_Aphiphi)

    Cl_deriv = {}
    keyname = 'Aphiphi'
    Cl_deriv[keyname] = {}
    Cl_deriv[keyname]['PP'] = (cl_phiphi_mod_higher - cl_phiphi_mod_lower) / (2 * delta_Aphiphi)

    if (0):
        from IPython import embed; embed()
        phi_kappa_fac =1. #( (els * (els+1))**2. /4. )
        cl_deriv_Aphiphi = Cl_deriv[keyname]['PP']
        clf()
        ax = subplot(111, yscale = 'log')
        #plot(cl_phiphi_mod_lower * phi_kappa_fac);plot(cl_phiphi * phi_kappa_fac);plot(cl_phiphi_mod_higher * phi_kappa_fac)
        plot(cl_phiphi_mod_higher - cl_phiphi_mod_lower, color = 'black')
        plot(cl_deriv_Aphiphi,  color = 'orangered')
        ylabel(r'$C_{L}^{\kappa \kappa}$', fontsize = 14)
        #ylim(1e-9,1e-4);
        show()
        sys.exit()

    return Cl_deriv

########################################################################################################################

def get_derivatives_finite_difference(cl, fid, delta_fid):

    cl_mod_lower = cl * (fid - delta_fid)
    cl_mod_higher = cl * (fid + delta_fid)

    cl_deriv = (cl_mod_higher - cl_mod_lower) / (2 * delta_fid)

    return np.nan_to_num(cl_deriv)

########################################################################################################################

def fn_get_dervatives(Cl_dic, param_dict, param_dict_derivatives, which_spectra = 'lensed_scalar', delta_l = 1., add_lensing = 0, use_thetastar = 0, use_cosmomc_theta = 0, two_sided_der = 1, derived_param = None, both_Cl_param = 0):#, fix_params = None, plot_now = 0):

    if derived_param is None:
        Cl_deriv = {}
    else:
        if both_Cl_param:
            Cl_deriv = {}
            derived_param_deriv = {}
        else:
            derived_param_deriv = {}
    for keyname in sorted(param_dict_derivatives):
        tmpdic = {}
        tmpdic[keyname] = param_dict_derivatives[keyname][0]

        '''
        if keyname in fix_params:
            Cl_mod_dic = Cl_dic
        else:
            dummypars, els, Cl_mod_dic = fn_set_CAMB_como(param_dict, param_dict_derivatives = tmpdic, add_lensing = add_lensing, use_thetastar = use_thetastar)
        '''

        if not two_sided_der:
            if derived_param is None:
                dummypars, els, Cl_mod_dic = fn_set_CAMB_como(param_dict, which_spectra, param_dict_derivatives = tmpdic, add_lensing = add_lensing, use_thetastar = use_thetastar, use_cosmomc_theta = use_cosmomc_theta, derived_param = derived_param)
            else:
                derived_param_val = fn_set_CAMB_como(param_dict, which_spectra, param_dict_derivatives = tmpdic, add_lensing = add_lensing, use_thetastar = use_thetastar, use_cosmomc_theta = use_cosmomc_theta, derived_param = derived_param)

            if keyname == 'As':
                #tmpdic[keyname] *= 1e9
                tmpdic[keyname] = np.log( ( param_dict[keyname] + tmpdic[keyname] ) * 1e10 ) - np.log( ( param_dict[keyname] - tmpdic[keyname] ) * 1e10 ) 

            if derived_param is not None:
                print('\n\t\tcannot calculated derivative of derived param with onesided difference since the fiducial value is not passed. aboritng script now\n\n')
                sys.exit()
                #return derived_param_deriv
            else:
                binned_Cl_mod_dic = {}
                for spec in Cl_dic:
                    Cl_mod = Cl_mod_dic[spec]
                    binned_el, binned_Cl_mod = fn_el_binning(els, Cl_mod, delta_l = delta_l)
                    binned_Cl_mod_dic[spec] = binned_Cl_mod        
                Cl_mod_dic = binned_Cl_mod_dic

                Cl_deriv[keyname] = {}
                for XX in Cl_dic: #loop over TT,EE,BB,TE
                    #Cl_deriv[keyname][XX] = (Cl_mod_dic[XX] - Cl_dic[XX]) / tmpdic[keyname]
                    Cl_deriv[keyname][XX] = (Cl_mod_dic[XX] - Cl_dic[XX]) / tmpdic[keyname]
        else:

            if derived_param is not None and both_Cl_param:
                dummypars, els, Cl_mod_dic_1, derived_param_val_1 = fn_set_CAMB_como(param_dict, which_spectra, param_dict_derivatives = tmpdic, high_low = 0, add_lensing = add_lensing, use_thetastar = use_thetastar, use_cosmomc_theta = use_cosmomc_theta, derived_param = derived_param, both_Cl_param = both_Cl_param)
                dummypars, els, Cl_mod_dic_2, derived_param_val_2 = fn_set_CAMB_como(param_dict, which_spectra, param_dict_derivatives = tmpdic, high_low = 1, add_lensing = add_lensing, use_thetastar = use_thetastar, use_cosmomc_theta = use_cosmomc_theta, derived_param = derived_param, both_Cl_param = both_Cl_param)                

            else:

                if derived_param is None:
                    dummypars, els, Cl_mod_dic_1 = fn_set_CAMB_como(param_dict, which_spectra, param_dict_derivatives = tmpdic, high_low = 0, add_lensing = add_lensing, use_thetastar = use_thetastar, use_cosmomc_theta = use_cosmomc_theta, derived_param = derived_param)
                    dummypars, els, Cl_mod_dic_2 = fn_set_CAMB_como(param_dict, which_spectra, param_dict_derivatives = tmpdic, high_low = 1, add_lensing = add_lensing, use_thetastar = use_thetastar, use_cosmomc_theta = use_cosmomc_theta, derived_param = derived_param)                
                    derived_param_val_1, derived_param_val_2 = None, None
                else:
                    derived_param_val_1 = fn_set_CAMB_como(param_dict, which_spectra, param_dict_derivatives = tmpdic, high_low = 0, add_lensing = add_lensing, use_thetastar = use_thetastar, use_cosmomc_theta = use_cosmomc_theta, derived_param = derived_param)
                    derived_param_val_2 = fn_set_CAMB_como(param_dict, which_spectra, param_dict_derivatives = tmpdic, high_low = 1, add_lensing = add_lensing, use_thetastar = use_thetastar, use_cosmomc_theta = use_cosmomc_theta, derived_param = derived_param)
                    Cl_mod_dic_1, Cl_mod_dic_2 = None, None

            if (0):#keyname == 'As': #20210108 - np.linalg.inv fixes matirx inversion with large numbers. pinv is bad. So this extra factor is no longer needed.
                if param_dict['As']>3.:
                    modify_As_step = 1 ###1e9
                else:
                    modify_As_step = 1e9
                tmpdic[keyname] *= modify_As_step
                #print('\n\n\n\n\n\n\t\tCheck As derviative step size. Aborting here. \n\n\n\n\n'); #sys.exit()
                #####tmpdic[keyname] = np.log( ( param_dict[keyname] + tmpdic[keyname] ) * 1e10 ) - np.log( ( param_dict[keyname] - tmpdic[keyname] ) * 1e10 ) 

            if derived_param_val_1 is not None and derived_param_val_2 is not None:
                if keyname == 'As': 
                    tmpdic[keyname] = tmpdic[keyname] / modify_As_step
                derived_param_deriv[keyname] = (derived_param_val_1 - derived_param_val_2) / (2*tmpdic[keyname])
                print(derived_param, keyname, derived_param_deriv[keyname], derived_param_val_1, derived_param_val_2, 2*tmpdic[keyname])
            
            if Cl_mod_dic_1 is not None and Cl_mod_dic_2 is not None:
                binned_Cl_mod_dic = {}
                for spec in Cl_mod_dic_1:
                    binned_el, binned_Cl_mod_1 = fn_el_binning(els, Cl_mod_dic_1[spec], delta_l = delta_l)
                    binned_el, binned_Cl_mod_2 = fn_el_binning(els, Cl_mod_dic_2[spec], delta_l = delta_l)
                    binned_Cl_mod_dic[spec] = [binned_Cl_mod_1, binned_Cl_mod_2]
                Cl_mod_dic = binned_Cl_mod_dic

                Cl_deriv[keyname] = {}
                for XX in Cl_dic: #loop over TT,EE,BB,TE
                    #Cl_deriv[keyname][XX] = (Cl_mod_dic[XX] - Cl_dic[XX]) / tmpdic[keyname]
                    Cl_deriv[keyname][XX] = (Cl_mod_dic[XX][0] - Cl_mod_dic[XX][1]) / (2*tmpdic[keyname])

    if (0):
        from IPython import embed; embed()

        els = binned_el
        keyname = 'ombh2'#tau' #As' #'tau' ##ombh2' #'ns'
        Dls_fac = els * (els+1)/2/np.pi
        ax  =subplot(111, xscale = 'log', yscale = 'log')
        colorarr = ['k', 'orangered', 'limegreen']
        for cntr, XX in enumerate( ['TT', 'EE', 'TE'] ):
            colorval = colorarr[cntr]
            neginds = np.where(Cl_deriv[keyname][XX]<0.)[0]
            plot(els[neginds], Dls_fac[neginds] * abs(Cl_deriv[keyname][XX][neginds]), ',', alpha = 0.5, ls = 'None', color = colorval)
            plot(els, Dls_fac * Cl_deriv[keyname][XX], '.', ms = 1., alpha = 1., ls = 'None', label = XX, color = colorval)
        title(r'%s' %(keyname))
        legend(loc = 3)
        ylim(1e-4, 1e5)
        xlim(10,2500)
        show();sys.exit()

    if derived_param is None:
        return Cl_deriv
    else:
        if both_Cl_param:
            return Cl_deriv, derived_param_deriv
        else:
            return derived_param_deriv

########################################################################################################################

def fn_get_Nl(els, fwhm, rms_map_T, rms_map_P = None, CMB = 1):
    """
    compute Nl - white noise + beam
    """

    if rms_map_P == None:
        rms_map_P = rms_map_T * 1.414

    fwhm_radians = np.radians(fwhm/60.)
    #Bl = np.exp((-fwhm_radians**2.) * els * (els+1) /2.35)
    sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
    sigma2 = sigma ** 2
    Bl = np.exp(els * (els+1) * sigma2)

    rms_map_T_radians = rms_map_T * np.radians(1/60.)
    rms_map_P_radians = rms_map_P * np.radians(1/60.)

    Nl_TT = (rms_map_T_radians)**2. * Bl
    Nl_PP = (rms_map_P_radians)**2. * Bl

    return Bl, Nl_TT, Nl_PP

########################################################################################################################
#20210423 - get galactic emission / residuals / derivatives
sys.path.append('/Users/sraghunathan/Research/SPTPol/analysis/git/DRAFT/modules/')
import foregrounds as fg, misc, ilc 

def get_gal_dust_residuals(ell, expname, freqarr, galmask, param_dict, bl_dic, ilc_weights_dic, debug = False):
    cl_gal_folder = '/Volumes/data_PHD_WD_babbloo/s4/cmbs4/map_based_simulations/202102_design_tool_input/4096/'
    if expname == 's4' or expname =='s4wide':
        #s4like_mask_v2 = 1
        cl_gal_dic_dust_fname = 'dust/0000/s4like_mask_v2/cls_galactic_sims_dust_nside2048_lmax5000_cos_el_40.npy' #dust
        cl_gal_dic_sync_fname = 'synchrotron/0000/s4like_mask_v2/cls_galactic_sims_sync_nside2048_lmax5000_cos_el_40.npy' #sync
        which_gal_mask = galmask
    elif expname == 's4deepv3r025':
        cl_gal_dic_dust_fname = 'dust/0000/s4delensing_mask/cls_galactic_sims_dust_nside2048_lmax5000_delensing.npy' #dust
        cl_gal_dic_sync_fname = 'synchrotron/0000/s4delensing_mask/cls_galactic_sims_sync_nside2048_lmax5000_delensing.npy' #sync

    param_dict['cl_gal_folder'] = cl_gal_folder
    param_dict['cl_gal_dic_dust_fname'] = cl_gal_dic_dust_fname
    param_dict['cl_gal_dic_sync_fname'] = cl_gal_dic_sync_fname

    which_spec_arr = ['TT', 'EE']
    dust_freq0 = 278
    cl_gal_dust_dic = {}
    for which_spec in which_spec_arr:
        if debug:
            ax = subplot(111, yscale = 'log')#, xscale = 'log')
            #colorarr = [cm.jet(int(d) + 40) for d in np.linspace(0, 255, len(freqarr))]
            colorarr = [cm.jet(int(d)) for d in np.linspace(0, 255, len(freqarr)**2)]
        cl_gal_dust_dic[which_spec]={}
        el_, cl_dust_freq0 = fg.get_cl_galactic(param_dict, 'dust', dust_freq0, dust_freq0, which_spec = which_spec, which_gal_mask = galmask, bl_dic = bl_dic, el = ell)
        for freq1cntr, freq1 in enumerate( freqarr ):
            for freq2cntr, freq2 in enumerate( freqarr ):
                if (freq2, freq1) in cl_gal_dust_dic: continue
                cl_dust_scaled = fg.scale_cl_dust_galactic(cl_dust_freq0, freq1, freq2 = freq2, freq0 = dust_freq0, Tdust = param_dict['Tdust'], beta_dust = param_dict['betadust'])
                if (0):
                    ##cl_dust_scaled = cl_dust_scaled / (bl_dic[freq1] * bl_dic[freq2])
                    el_, cl_dust_scaled = fg.get_cl_galactic(param_dict, 'dust', freq1, freq2, which_spec = which_spec, which_gal_mask = galmask, bl_dic = bl_dic, el = ell)
                    pass

                cl_gal_dust_dic[which_spec][(freq1, freq2)] = cl_gal_dust_dic[which_spec][(freq2, freq1)] = cl_dust_scaled
                if debug and freq1 == freq2 and freq1 == dust_freq0:
                    plot(ell, cl_dust_scaled, label = '(%s, %s)' %(freq1, freq2), color = colorarr[freq1cntr + freq2cntr])
                    el_, cl_dust = fg.get_cl_galactic(param_dict, 'dust', freq1, freq2, which_spec = which_spec, which_gal_mask = galmask, bl_dic = bl_dic, el = ell)
                    plot(el_, cl_dust, alpha = 0.5, color = 'k', lw = 0.5)
                    xlim(-100, 5000); ylim(1e-10, 1e4)
                    show(); #sys.exit()
        if debug:
            show(); #sys.exit()

    #from IPython import embed; embed()
    debug = False #True
    cl_gal_dust_res_dic = {}
    for which_spec in which_spec_arr:
        cl_gal_dust_res_dic[which_spec] = []
        for elcnt, currel in enumerate(ell):
            #if (elcnt%2500) == 0: print(which_spec, elcnt)
            clmat = np.mat( ilc.create_clmat(freqarr, elcnt, cl_gal_dust_dic[which_spec]) )
            currw_ilc = np.mat( ilc_weights_dic[which_spec][:, elcnt] )
            
            curr_res_ilc = np.asarray(np.dot(currw_ilc, np.dot(clmat, currw_ilc.T)))[0][0]
            if (0):#elcnt %500 == 0:
                print(curr_res_ilc, elcnt, which_spec)
            cl_gal_dust_res_dic[which_spec].append( curr_res_ilc )
        
        cl_gal_dust_res_dic[which_spec] = np.asarray(cl_gal_dust_res_dic[which_spec])
        if debug:
            clf()
            ax = subplot(111, yscale = 'log')#, xscale = 'log')
            plot(cl_gal_dust_res_dic[which_spec]); title(r'%s' %(which_spec))
            show(); sys.exit()

    return cl_gal_dust_res_dic

def get_gal_dust_derivatives(ell, expname, freqarr, galmask, params_for_galdust_fitting, param_dict, param_dict_derivatives, bl_dic, ilc_weights_dic, which_spec_arr = ['TT', 'EE', 'TE']):

    cl_gal_params_deriv_dic = {}
    for p in params_for_galdust_fitting:
        store_for_derivs = {}
        for high_low in range(2):
            param_dict_mod = param_dict.copy()
            step = param_dict_derivatives[p][0]
            if high_low == 0:
                param_dict_mod[p] = param_dict_mod[p] + step
            else:
                param_dict_mod[p] = param_dict_mod[p] - step
            cl_gal_dust_res_dic = get_gal_dust_residuals(ell, expname, freqarr, galmask, param_dict_mod, bl_dic, ilc_weights_dic)
            store_for_derivs[high_low] = cl_gal_dust_res_dic
            #print(p, high_low, param_dict_mod[p], cl_gal_dust_res_dic['TT'].max(), cl_gal_dust_res_dic['EE'].max())

        #now get derivatives
        cl_gal_params_deriv_dic[p] = {}
        for which_spec in which_spec_arr:
            if which_spec not in store_for_derivs[0]:
                cl_gal_params_deriv_dic[p][which_spec] = np.zeros(len(ell))
            else:
                cl_gal_params_deriv_dic[p][which_spec] = (store_for_derivs[1][which_spec] - store_for_derivs[0][which_spec]) / (2*step)
            #print(p, which_spec, cl_gal_params_deriv_dic[p][which_spec].max())

    return cl_gal_params_deriv_dic

########################################################################################################################

def fn_delta_Cl(els, Cl_dic, exp_dic, delta_l = 1., Nlfile = None, Nldic = None, scale_noise_curves = 1.):
    def fn_pad(el, nl):
        el.astype(int)
        reclen_padding_zeros = int( max(el) - len(nl) + 1 )
        nl = np.concatenate( (nl, np.zeros(reclen_padding_zeros)) )
        nl[nl == 0] = max(nl) * 1e6 #some large number        
        return nl
    if Nldic is not None:
        if 'T' in Nldic:
            Nl_TT = Nldic['T'][els]
        if 'P' in Nldic:
            Nl_PP = Nldic['P'][els]
        if 'TT' in Nldic:
            Nl_TT = fn_pad(els, Nldic['TT'])
            Nl_TT = Nl_TT[els]

            Nl_TT = Nldic['TT']

        if 'EE' in Nldic:
            Nl_EE = fn_pad(els, Nldic['EE'])
            Nl_EE = Nl_EE[els]

            Nl_EE = Nldic['EE']

        if 'TE' in Nldic:
            Nl_TE = Nldic['TE']
            if Nldic['TE'] is not None:
                Nl_TE = fn_pad(els, Nl_TE)
                Nl_TE = Nl_TE[els]

            Nl_TE = Nldic['TE']

        if 'TB' in Nldic:
            Nl_TB = fn_pad(els, Nldic['TB'])
            Nl_TB = Nl_TB[els] 

            Nl_TB = Nldic['TB']           

        if 'BB' in Nldic:
            Nl_BB = fn_pad(els, Nldic['BB'])
            Nl_BB = Nl_BB[els]            

            Nl_BB = Nldic['BB']

        if 'EB' in Nldic:
            Nl_EB = fn_pad(els, Nldic['EB'])
            Nl_EB = Nl_EB[els]            

            Nl_EB = Nldic['EB']

        if 'PP' in Nldic: #lensing phiphi N0
            Nl_PP = fn_pad(els, Nldic['PP'])            
            Nl_PP = Nl_PP[els] 
            ##Nl_PP = Nldic['PP']


        if (0):
            loglog(Nl_TT)
            loglog(Nl_EE)
            ylim(1e-6, 1e3)
            show();sys.exit()

    else:
        if Nlfile is not None:
            #from IPython import embed; embed()        
            Nl = np.load(Nlfile)

            if isinstance(Nl, dict):
                Nl_TT = Nl['TT']
                Nl_PP = Nl['PP']
            else:
                Nl_TT = Nl
                Nl_PP = Nl_TT * np.sqrt(2.)

            #adjust lengths
            reclen_padding_zeros = max(els) + 1 - len(Nl_TT)
            Nl_TT = np.concatenate( (Nl_TT, np.zeros(reclen_padding_zeros)) )
            Nl_PP = np.concatenate( (Nl_PP, np.zeros(reclen_padding_zeros)) )

            #pick the desired ell
            Nl_TT = Nl_TT[els]
            Nl_PP = Nl_PP[els]
        else:
            exp = [exp_dic['beam'], exp_dic['deltaT']] #beam, deltaT, deltaP = None --> 1.414 * deltaT
            Bl, Nl_TT, Nl_EE = fn_get_Nl(els, *exp)
            Nl_TE = np.zeros( len(els) )
            Nl_PP = np.zeros( len(els) )

    #exp = [exp_dic['beam'], exp_dic['deltaT']] #beam, deltaT, deltaP = None --> 1.414 * deltaT
    #Bl, Nl_TT_white, Nl_EE_white = fn_get_Nl(els, *exp)

    delta_Cl_dic = {}
    for XX in Cl_dic:
        #print(XX)       
        if XX == 'TT':
            Nl = Nl_TT
        elif XX == 'EE' or XX == 'BB':
            Nl = Nl_EE ##Nl_PP
        elif XX == 'TE':
            Nl = Nl_TE

            if Nl is not None:
                print('\n\t\t\t\tnulling galaxy TE\n')
                Nl = np.copy(Nl) * 0.

        elif XX == 'PP':            
            Nl = Nl_PP
        else:
            Nl = np.zeros( len( els ) )

        #print(XX, Nl)
        if Nl is None:
            Nl = np.zeros( len( els ) )
        #print(XX, Nl)

        #20211208 - scale noise curves
        if scale_noise_curves != 1.0:
            Nl = Nl/scale_noise_curves
        Nl[np.isnan(Nl)] = 0.

        Cl = Cl_dic[XX]
        if (0): #20200602 - this is wrong - being done twice. Nl was already picked at binned el centres.
            els_2, Nl = fn_el_binning(np.arange(min(els),max(els)+1), Nl, delta_l = delta_l)
            #els_2, Nl = fn_el_binning(np.arange(min(els),len(Nl)+2), Nl, delta_l = delta_l)
        if len(Nl)<len(Cl):
            Nl = fn_pad(els, Nl)
            Nl = Nl[els]
        ##delta_Cl_dic[XX] = np.sqrt(2./ (2*els + 1) / exp_dic['fsky'] / (1.*delta_l) ) * (Cl + Nl)
        delta_Cl_dic[XX] = np.sqrt(2./ (2.*els + 1.) / exp_dic['fsky'] ) * (Cl + Nl)

        #from IPython import embed; embed()

        if (0):
            ax = subplot(111, yscale = 'log')
            plot(Nl);
            errorbar(els, Cl, yerr = delta_Cl_dic[XX]);
            title(XX);
            ylim(1e-8, 1e3)
            show(); #sys.exit()

    if (0):
        ax = subplot(111, yscale = 'log')
        dls_fac = 1. ##(els * (els+1))/2/np.pi
        plot(Cl_dic['TT'] * dls_fac, 'k-'); plot(Cl_dic['EE'] * dls_fac, 'r-'); #plot(Cl_dic['TE'] * dls_fac, 'g-')
        plot(Nl_TT * dls_fac, 'k--'); plot(Nl_EE * dls_fac, 'r--'); #plot(Nl_TE * dls_fac, 'g--')
        plot(Nl_TT_white * dls_fac, 'k:'); plot(Nl_EE_white * dls_fac, 'r:'); #plot(Nl_TE * dls_fac, 'g:')
        #plot(delta_Cl_dic['TT'] * dls_fac, 'k--'); plot(delta_Cl_dic['EE'] * dls_fac, 'r--'); #plot(delta_Cl_dic['TE'] * dls_fac, 'g--')
        show()
        sys.exit()

    #print(delta_Cl_dic); sys.exit()
    return delta_Cl_dic

########################################################################################################################

def fn_delta_Cl_simple(els, cl_dic, nl_dic, fsky):
    delta_cl_dic = {}
    for XX in cl_dic:
        if XX == 'TT':
            nl = nl_dic['TT']
        elif XX == 'EE' or XX == 'BB':
            nl = nl_dic['EE']
        elif XX == 'TE':
            nl = nl_dic['TE']
            if (1):
                #print('\n\n\n\t\t\t\tnulling galaxy TE\n\n\n')
                nl = np.copy(nl) * 0.

        cl = cl_dic[XX]
        delta_cl_dic[XX] = np.sqrt(2./ (2.*els + 1.) / fsky ) * (cl + nl)
        delta_cl_dic[XX] = np.nan_to_num(delta_cl_dic[XX])

    return delta_cl_dic

########################################################################################################################

def fn_fix_params(F_mat, param_names, fix_params):

    #remove parameters that must be fixed    
    F_mat_refined = []
    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):
            if p1 in fix_params or p2 in fix_params: continue
            F_mat_refined.append( (F_mat[pcntr2, pcntr1]) )

    totparamsafterfixing = int( np.sqrt( len(F_mat_refined) ) )
    F_mat_refined = np.asarray( F_mat_refined ).reshape( (totparamsafterfixing, totparamsafterfixing) )

    param_names_refined = []
    for p in param_names:
        if p in fix_params: continue
        param_names_refined.append(p)


    return F_mat_refined, param_names_refined

########################################################################################################################

def get_jacobian(param_dict, param_dict_derivatives, param_names, derived_param_names_dic, derived_param_names = None, der_params_derv_fname = None):

    if (0):
        print('\n\n\tremoving non linear lensing\n')
        param_dict['lens_potential_accuracy'] = 0

    if (1):
        #print('\n\n\tusing cosmomc_theta now\n')
        param_dict['cosmomc_theta'] = 0.010406810462970357

    param_dict_derivatives_refined = {}
    for p in param_names:
        param_dict_derivatives_refined[p] = param_dict_derivatives[p]

    if os.path.exists(der_params_derv_fname):
        der_params_derv_dic = np.load(der_params_derv_fname, allow_pickle = 1).item()
    else:
        der_params_derv_dic = {}

    if derived_param_names is None:
        derived_param_names = sorted( derived_param_names_dic.keys() )
        derived_param_names = np.concatenate( (param_names, derived_param_names) )

    Cl_dic = None
    nparams = len( param_names )
    nderparams = len( derived_param_names )
    J_mat = np.zeros( (nparams, nderparams) )
    for pcntr, der_p in enumerate( derived_param_names ):

        if 'thetastar' in param_dict_derivatives:
            use_thetastar = 1
        else:
            use_thetastar = 0

        if 'cosmomc_theta' in param_dict_derivatives:
            use_cosmomc_theta = 1
        else:
            use_cosmomc_theta = 0

        #check if this is fine for all parameters
        if (0):##der_p.find('omega')>-1 or der_p.find('sigma8')>-1 or der_p == 'cosmomc_theta':
            use_thetastar = 0 ###1
            #use_cosmomc_theta = 1

        print(der_p, use_thetastar, use_cosmomc_theta)

        if der_p not in der_params_derv_dic:
            if der_p in param_names:
                derived_param_deriv = {}
                for tmp_p in param_names:
                    if tmp_p == der_p:
                        derived_param_deriv[tmp_p] = 1.
                    else:
                        derived_param_deriv[tmp_p] = 0.
            else:
                der_p_cmd = derived_param_names_dic[der_p]
                derived_param_deriv = fn_get_dervatives(Cl_dic, param_dict, param_dict_derivatives_refined, use_thetastar = use_thetastar, use_cosmomc_theta = use_cosmomc_theta, derived_param = [der_p, der_p_cmd])
            der_params_derv_dic[der_p] = [derived_param_deriv, use_thetastar]
            np.save(der_params_derv_fname, der_params_derv_dic)
        else:
            derived_param_deriv = der_params_derv_dic[der_p][0]

        for pcntr2, p2 in enumerate(param_names):
            if derived_param_deriv[p2] == 0.:
                J_mat[pcntr2, pcntr] = derived_param_deriv[p2]
            else:
                J_mat[pcntr2, pcntr] = 1./derived_param_deriv[p2]

        '''
        for pcntr2, p2 in enumerate(param_names):
            J_mat[pcntr2, pcntr] = derived_param_deriv[p2]
        J_mat = sc.linalg.pinv2(J_mat)
        J_mat = np.linalg.inv(J_mat)
        '''

    from IPython import embed; embed()
    return J_mat

########################################################################################################################

def fisher_transform(F_mat, param_dict, param_dict_derivatives, param_names, derived_param_names_dic, J_mat = None):

    F_mat = np.mat( F_mat )
    if J_mat is None:
        J_mat = get_jacobian(param_dict, param_dict_derivatives, param_names, derived_param_names_dic)
    J_mat = np.mat( J_mat )

    F_mat_updated = np.dot(J_mat.T, np.dot( F_mat, J_mat ) )

    #from IPython import embed; embed()
    
    return F_mat_updated

########################################################################################################################

def fn_add_sys_errors(F_mat, param_names, sys_error_dic):

    C_mat = np.linalg.inv(F_mat)
    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):
            if p1 == p2 and p2 in sys_error_dic:
                stat_errr_val = np.diag(C_mat)[pcntr2]                
                sys_error_val = sys_error_dic[p2]
                if sys_error_val == 0.: continue
                scale_fac = sys_error_val/stat_errr_val
                F_mat[pcntr2, :] = F_mat[pcntr2, :]/scale_fac
                F_mat[:, pcntr2] = F_mat[:, pcntr2]/scale_fac
    return F_mat

def fn_add_prior(F_mat, param_names, prior_dic):

    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):
            if p1 == p2 and p1 in prior_dic:
                prior_val = prior_dic[p1]
                F_mat[pcntr2, pcntr1] += 1./prior_val**2.

    return F_mat

########################################################################################################################

def get_planck_prior(param_names, fix_params = [], add_BAO = 1):

    if add_BAO:
        planck_prior_dic_stored = { 
                'ombh2' : 0.00014, 'omch2' : 0.00091, 
                'h': 0.0042, 'tau' : 0.0071, 'ns' : 0.0038,
                'As' : 0.2917e-10 * 1e9, 
                #'ws' : (-1e-1,(0.5, 0.0001)), 'neff': (1e-3,(0.5,0.01)), 'mnu': (1e-3,(1.,0.0001)), 'YHe': (1e-3,(1.,0.0001)),\
                #'Alens': (1e-3, (1., 0.3)), 
                }
    else:
        planck_prior_dic_stored = { 
                'ombh2' : 0.00015, 'omch2' : 0.0012, 
                'h': 0.0054, 'tau' : 0.0073, 'ns' : 0.0042,
                'As' : 0.2917e-10 * 1e9, 
                }

    planck_prior_dic = {}
    for p1 in param_names:
        if p1 in fix_params: continue
        if p1 not in planck_prior_dic_stored: continue
        planck_prior_dic[p1] = planck_prior_dic_stored[p1]

    F_mat_planck = np.zeros( ( len(param_names), len(param_names) ) )
    F_mat_planck = fn_add_prior(F_mat_planck, param_names, planck_prior_dic)

    return planck_prior_dic, F_mat_planck

########################################################################################################################
def get_dl_fac(el, which_spectra = 'cmb'):

    if which_spectra == 'cmb':

        return el * (el+1.)/2./np.pi

    else:

        return None


########################################################################################################################


def fn_get_ellipse_specs(COV, howmanysigma = 1):
    """
    Refer https://arxiv.org/pdf/0906.4123.pdf
    """
    assert COV.shape == (2,2)
    confsigma_dic = {1:2.3, 2:6.17, 3: 11.8}

    sig_x2, sig_y2 = COV[0,0], COV[1,1]
    sig_xy = COV[0,1]
    
    t1 = (sig_x2 + sig_y2)/2.
    t2 = np.sqrt( (sig_x2 - sig_y2)**2. /4. + sig_xy**2. )
    
    a2 = t1 + t2
    b2 = t1 - t2

    a = np.sqrt(a2)
    b = np.sqrt(b2)

    t1 = 2 * sig_xy
    t2 = sig_x2 - sig_y2
    theta = np.arctan2(t1,t2) / 2.
    
    alpha = np.sqrt(confsigma_dic[howmanysigma])
    
    #return (a*alpha, b*alpha, theta)
    return (a*alpha, b*alpha, theta, alpha*(sig_x2**0.5), alpha*(sig_y2**0.5))

########################################################################################################################

def get_Gaussian(mean, sigma, minx, maxx, delx = None):

    if delx is None: delx = (maxx - minx)/100000.

    x = np.arange(minx, maxx, delx)

    #return x, 1./(2*np.pi*sigma)**0.5 * np.exp( -(x - mean)**2. / (2 * sigma**2.)  )
    return x, np.exp( -(x - mean)**2. / (2 * sigma**2.)  )

########################################################################################################################

def fn_fisher_forecast_Aphiphi(els, Cl_deriv_dic, delta_Cl_dic, params, pspectra_to_use, min_l = 0, max_l = 6000):

    npar = len(params)
    F = np.zeros([npar,npar])
    #els = np.arange( len( delta_Cl_dic.values()[0] ) )

    pspectra_to_use_full = np.asarray( ['PP'] )

    for lcntr, l in enumerate( els ):

        if l<min_l or l>max_l:
            continue

        PP = delta_Cl_dic['PP'][lcntr]
        COV_mat_l = PP**2.
        COV_mat_l = np.mat( COV_mat_l )
        #Cinv_l = sc.linalg.pinv2(COV_mat_l) #made sure that COV_mat_l * Cinv_l ~= I
        Cinv_l = np.linalg.inv(COV_mat_l) #made sure that COV_mat_l * Cinv_l ~= I
        #print l, p, p2, fprime1_l_vec, fprime2_l_vec, COV_mat_l

        pspec_combinations = []
        for X in pspectra_to_use:
            for Y in pspectra_to_use:
                xind = np.where(pspectra_to_use_full == X)[0][0]
                yind = np.where(pspectra_to_use_full == Y)[0][0]
                if [Y,X, yind, xind] in pspec_combinations: continue
                pspec_combinations.append([X, Y, xind, yind])

        param_combinations = []
        for pcnt,p in enumerate(params):
            for pcnt2,p2 in enumerate(params):
                ##if [p2,p,pcnt2,pcnt] in param_combinations: continue
                param_combinations.append([p,p2, pcnt, pcnt2])

        for (p,p2, pcnt, pcnt2) in param_combinations:
            for (X,Y, xind, yind) in pspec_combinations:

                der1 = np.asarray( [Cl_deriv_dic[p]['PP'][lcntr]] )
                der2 = np.asarray( [Cl_deriv_dic[p2]['PP'][lcntr]] )

                fprime1_l_vec = np.zeros(len(der1))
                fprime2_l_vec = np.zeros(len(der2))

                fprime1_l_vec[xind] = der1[xind]
                fprime2_l_vec[yind] = der2[yind]

                #if l > 100:
                #    from IPython import embed; embed()

                curr_val = np.dot(fprime1_l_vec, np.dot( Cinv_l, fprime2_l_vec ))

                F[pcnt2,pcnt] += curr_val

    return F    

########################################################################################################################
########################################################################################################################
########################################################################################################################

def get_cov_for_fisher_with_trace(TT, EE, TE, PP, TP, EP):

    C = np.zeros( (3,3) ) #TT, EE, PP
    C[0,0] = TT
    C[1,1] = EE
    C[0,1] = C[1,0] = TE

    C[2,2] = PP
    C[0,2] = C[2,0] = TP
    C[1,2] = C[2,1] = EP ##0. ##EP

    return np.mat( C )

def get_cov_for_fisher(TT, EE, TE, with_lensing = 0):

    if not with_lensing:
        C = np.zeros( (3,3) ) #TT, EE, TE
    else:
        C = np.zeros( (4,4) ) #TT, EE, TE

    C[0,0] = TT**2.
    C[1,1] = EE**2.
    C[2,2] = 0.5 * (TE**2. + TT * EE )

    C[0,1] = C[1,0] = TE**2.
    C[0,2] = C[2,0] = TT*TE
    C[1,2] = C[2,1] = EE*TE

    if with_lensing:
        C[3,3] = np.copy(PP_ori)**2.
        C[0,3] = C[3,0] = np.copy(Tphi_ori)**2.
        C[1,3] = C[3,1] = np.copy(Ephi_ori)**2.

    return np.mat( C )

#def fn_fisher_forecast(els, Cl_dic, Cl_deriv_dic, delta_Cl_dic, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None):
def fn_fisher_forecast_with_trace(els, Cl_deriv_dic, delta_Cl_dic, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None):

    if min_l_temp is None: min_l_temp = 0
    if max_l_temp is None: max_l_temp = 10000

    if min_l_pol is None: min_l_pol = 0
    if max_l_pol is None: max_l_pol = 10000

    npar = len(params)
    F = np.zeros([npar,npar])
    #els = np.arange( len( delta_Cl_dic.values()[0] ) )

    with_lensing = 0
    if 'PP' in pspectra_to_use:
        with_lensing = 1

    all_pspectra_to_use = []
    for tmp in pspectra_to_use:
        if isinstance(tmp, list):      
            all_pspectra_to_use.extend(tmp)
        else:
            all_pspectra_to_use.append(tmp)

    print('!!Using trace now!!')

    for lcntr, l in enumerate( els ):

        '''
        #creating the covariance matrix for this multipole
        TT, EE, TE = delta_Cl_dic['TT'][lcntr], delta_Cl_dic['EE'][lcntr], delta_Cl_dic['TE'][lcntr]
        if with_lensing:
            Tphi, Ephi, PP = delta_Cl_dic['Tphi'][lcntr], delta_Cl_dic['Ephi'][lcntr], delta_Cl_dic['PP'][lcntr]
        else:
            Tphi = Ephi = PP = 0.
        '''

        TT, EE, TE = 0., 0., 0.
        Tphi = Ephi = PP = 0.
        if 'TT' in delta_Cl_dic:
            TT = delta_Cl_dic['TT'][lcntr]
        if 'EE' in delta_Cl_dic:
            EE = delta_Cl_dic['EE'][lcntr]
        if 'TE' in delta_Cl_dic:
            TE = delta_Cl_dic['TE'][lcntr]
        if with_lensing:
            Tphi, Ephi, PP = delta_Cl_dic['Tphi'][lcntr], delta_Cl_dic['Ephi'][lcntr], delta_Cl_dic['PP'][lcntr]

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

        #20200621 - nulling unwanted fields
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
        #20200621 - nulling unwanted fields

        COV_mat_l = get_cov(TT, EE, TE, PP, Tphi, Ephi)
        inv_COV_mat_l = sc.linalg.pinv2(COV_mat_l)
        #inv_COV_mat_l = np.linalg.inv(COV_mat_l)

        #print(COV_mat_l, inv_COV_mat_l)
        #print(l, null_TT, null_EE, null_TE); sys.exit()
        #print(l, TT, EE, TE)#; sys.exit()

        if (0):##l%500 == 0: 
            from IPython import embed; embed()
            print(l, null_TT, null_EE, null_TE, null_PP)
            print(COV_mat_l)

        for (p,p2, pcnt, pcnt2) in param_combinations:

            '''
            TT_der1, EE_der1, TE_der1 = Cl_deriv_dic[p]['TT'][lcntr], Cl_deriv_dic[p]['EE'][lcntr], Cl_deriv_dic[p]['TE'][lcntr]
            TT_der2, EE_der2, TE_der2 = Cl_deriv_dic[p2]['TT'][lcntr], Cl_deriv_dic[p2]['EE'][lcntr], Cl_deriv_dic[p2]['TE'][lcntr]
            '''

            TT_der1, EE_der1, TE_der1 = 0., 0., 0.
            TT_der2, EE_der2, TE_der2 = 0., 0., 0.

            if 'TT' in Cl_deriv_dic[p]:
                TT_der1 = Cl_deriv_dic[p]['TT'][lcntr]
                TT_der2 = Cl_deriv_dic[p2]['TT'][lcntr]
            if 'EE' in Cl_deriv_dic[p]:
                EE_der1 = Cl_deriv_dic[p]['EE'][lcntr]
                EE_der2 = Cl_deriv_dic[p2]['EE'][lcntr]
            if 'TE' in Cl_deriv_dic[p]:
                TE_der1 = Cl_deriv_dic[p]['TE'][lcntr]
                TE_der2 = Cl_deriv_dic[p2]['TE'][lcntr]


            if with_lensing:
                PP_der1, TPhi_der1, EPhi_der1 = Cl_deriv_dic[p]['PP'][lcntr], Cl_deriv_dic[p]['Tphi'][lcntr], Cl_deriv_dic[p]['Ephi'][lcntr]
                PP_der2, TPhi_der2, EPhi_der2 = Cl_deriv_dic[p2]['PP'][lcntr], Cl_deriv_dic[p2]['Tphi'][lcntr], Cl_deriv_dic[p2]['Ephi'][lcntr]
            else:
                PP_der1 = PP_der2 = 0.
                TPhi_der1 = TPhi_der2 = 0. 
                EPhi_der1 = EPhi_der2 = 0.

            '''
            if null_TT: TT_der1 = TT_der2 = 0.
            if null_EE: EE_der1 = EE_der2 = 0.
            if null_TE: TE_der1 = TE_der2 = 0
            if null_PP: PP_der1 = PP_der2 = 0
            '''

            if null_TT: TT_der1 = TT_der2 = TPhi_der1 = TPhi_der2 = 0
            if null_EE: EE_der1 = EE_der2 = EPhi_der1 = EPhi_der2 = 0
            if null_TE: TE_der1 = TE_der2 = 0
            if null_PP: PP_der1 = PP_der2 = 0

            fprime1_l_vec = get_cov(TT_der1, EE_der1, TE_der1, PP_der1, TPhi_der1, EPhi_der1)
            fprime2_l_vec = get_cov(TT_der2, EE_der2, TE_der2, PP_der2, TPhi_der2, EPhi_der2)

            if (0):#l==4000:#l % 500 == 0:
                #from IPython import embed; embed()
                print('\n')
                print(null_TT, null_EE, null_TE, null_PP, with_lensing)
                print(fprime1_l_vec)
                print(fprime2_l_vec)
                print(COV_mat_l) 

            #curr_val = np.trace( inv_COV_mat_l * fprime1_l_vec * inv_COV_mat_l * fprime2_l_vec)
            curr_val = np.trace( np.dot( np.dot(inv_COV_mat_l, fprime1_l_vec), np.dot(inv_COV_mat_l, fprime2_l_vec) ) )

            F[pcnt2,pcnt] += curr_val

    if (0):
        from IPython import embed; embed()
        F = np.mat(F)

    if (0):
        from IPython import embed; embed()
        F = np.mat(F)
        #C = sc.linalg.pinv2(F) #made sure that COV_mat_l * Cinv_l ~= I
        C = np.linalg.inv(F) #made sure that COV_mat_l * Cinv_l ~= I

    return F   




def fn_fisher_forecast_with_trace_spt(els, Cl_deriv_dic, delta_Cl_dic, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None):

    if min_l_temp is None: min_l_temp = 0
    if max_l_temp is None: max_l_temp = 10000

    if min_l_pol is None: min_l_pol = 0
    if max_l_pol is None: max_l_pol = 10000

    npar = len(params)
    F = np.zeros([npar,npar])
    #els = np.arange( len( delta_Cl_dic.values()[0] ) )

    with_lensing = 0
    if 'PP' in pspectra_to_use:
        with_lensing = 1

    all_pspectra_to_use = []
    for tmp in pspectra_to_use:
        if isinstance(tmp, list):      
            all_pspectra_to_use.extend(tmp)
        else:
            all_pspectra_to_use.append(tmp)

    print('!!Using trace now!!')

    for lcntr, l in enumerate( els ):

        TT, EE, TE = 0., 0., 0.
        Tphi = Ephi = PP = 0.
        if 'TT' in delta_Cl_dic:
            TT = delta_Cl_dic['TT'][lcntr]
        if 'EE' in delta_Cl_dic:
            EE = delta_Cl_dic['EE'][lcntr]
        if 'TE' in delta_Cl_dic:
            TE = delta_Cl_dic['TE'][lcntr]
        if with_lensing:
            Tphi, Ephi, PP = delta_Cl_dic['Tphi'][lcntr], delta_Cl_dic['Ephi'][lcntr], delta_Cl_dic['PP'][lcntr]

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
            if 'TT' not in pspectra_to_use and 'EE' not in pspectra_to_use:
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
            C[1,2] = C[2,1] = 0. ##EP

            return np.mat( C )

        #20200621 - nulling unwanted fields
        if null_TT and null_TE: TT = 0
        if null_EE and null_TE: EE = 0
        if null_TE and (null_TT and null_EE): TE = 0
        if null_PP: PP = Tphi = EPhi = 0
        if null_TT: Tphi = 0
        if null_EE: Ephi = 0
        #20200621 - nulling unwanted fields


        COV_mat_l = get_cov(TT, EE, TE, PP, Tphi, Ephi)
        inv_COV_mat_l = sc.linalg.pinv2(COV_mat_l)
        #inv_COV_mat_l = np.linalg.inv(COV_mat_l)

        #print(l, null_TT, null_EE, null_EE)

        ##if l%500 == 0: print(l, COV_mat_l)

        for (p,p2, pcnt, pcnt2) in param_combinations:

            '''
            TT_der1, EE_der1, TE_der1 = Cl_deriv_dic[p]['TT'][lcntr], Cl_deriv_dic[p]['EE'][lcntr], Cl_deriv_dic[p]['TE'][lcntr]
            TT_der2, EE_der2, TE_der2 = Cl_deriv_dic[p2]['TT'][lcntr], Cl_deriv_dic[p2]['EE'][lcntr], Cl_deriv_dic[p2]['TE'][lcntr]
            '''

            TT_der1, EE_der1, TE_der1 = 0., 0., 0.
            TT_der2, EE_der2, TE_der2 = 0., 0., 0.

            if 'TT' in Cl_deriv_dic[p]:
                TT_der1 = Cl_deriv_dic[p]['TT'][lcntr]
                TT_der2 = Cl_deriv_dic[p2]['TT'][lcntr]
            if 'EE' in Cl_deriv_dic[p]:
                EE_der1 = Cl_deriv_dic[p]['EE'][lcntr]
                EE_der2 = Cl_deriv_dic[p2]['EE'][lcntr]
            if 'TE' in Cl_deriv_dic[p]:
                TE_der1 = Cl_deriv_dic[p]['TE'][lcntr]
                TE_der2 = Cl_deriv_dic[p2]['TE'][lcntr]


            if with_lensing:
                PP_der1, TPhi_der1, EPhi_der1 = Cl_deriv_dic[p]['PP'][lcntr], Cl_deriv_dic[p]['Tphi'][lcntr], Cl_deriv_dic[p]['Ephi'][lcntr]
                PP_der2, TPhi_der2, EPhi_der2 = Cl_deriv_dic[p2]['PP'][lcntr], Cl_deriv_dic[p2]['Tphi'][lcntr], Cl_deriv_dic[p2]['Ephi'][lcntr]
            else:
                PP_der1 = PP_der2 = 0.
                TPhi_der1 = TPhi_der2 = 0. 
                EPhi_der1 = EPhi_der2 = 0.

            '''
            if null_TT: TT_der1 = TT_der2 = 0.
            if null_EE: EE_der1 = EE_der2 = 0.
            if null_TE: TE_der1 = TE_der2 = 0
            if null_PP: PP_der1 = PP_der2 = 0
            '''

            if null_TT: TT_der1 = TT_der2 = TPhi_der1 = TPhi_der2 = 0
            if null_EE: EE_der1 = EE_der2 = EPhi_der1 = EPhi_der2 = 0
            if null_TE: TE_der1 = TE_der2 = 0
            if null_PP: PP_der1 = PP_der2 = 0


            fprime1_l_vec = get_cov(TT_der1, EE_der1, TE_der1, PP_der1, TPhi_der1, EPhi_der1)
            fprime2_l_vec = get_cov(TT_der2, EE_der2, TE_der2, PP_der2, TPhi_der2, EPhi_der2)

            if (0):#l % 500 == 0:
                from IPython import embed; embed()
                print(null_TT, null_EE, null_TE, null_PP, with_lensing)
                print(fprime1_l_vec)
                print(fprime2_l_vec)
                print(COV_mat_l) 

            #curr_val = np.trace( inv_COV_mat_l * fprime1_l_vec * inv_COV_mat_l * fprime2_l_vec)
            curr_val = np.trace( np.dot( np.dot(inv_COV_mat_l, fprime1_l_vec), np.dot(inv_COV_mat_l, fprime2_l_vec) ) )

            F[pcnt2,pcnt] += curr_val

    if (0):
        from IPython import embed; embed()
        F = np.mat(F)

    if (0):
        from IPython import embed; embed()
        F = np.mat(F)
        #C = sc.linalg.pinv2(F) #made sure that COV_mat_l * Cinv_l ~= I
        C = np.linalg.inv(F) #made sure that COV_mat_l * Cinv_l ~= I

    return F  


#def fn_fisher_forecast(els, Cl_dic, Cl_deriv_dic, delta_Cl_dic, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None):
def fn_fisher_forecast(els, Cl_deriv_dic, delta_Cl_dic, params, pspectra_to_use, min_l_temp = None, max_l_temp = None, min_l_pol = None, max_l_pol = None):

    if min_l_temp is None: min_l_temp = 0
    if max_l_temp is None: max_l_temp = 10000

    if min_l_pol is None: min_l_pol = 0
    if max_l_pol is None: max_l_pol = 10000

    npar = len(params)
    F = np.zeros([npar,npar])
    #els = np.arange( len( delta_Cl_dic.values()[0] ) )

    pspectra_to_use_full = np.asarray( ['TT', 'EE', 'TE', 'EE-TE', 'TT-EE-TE'] )
    with_lensing = 0
    if 'PP' in pspectra_to_use:
        with_lensing = 1
        pspectra_to_use_full = np.asarray( ['TT', 'EE', 'TE', 'PP', 'EE-TE', 'TT-EE-TE', 'TT-EE-TE-PP'] )

    for lcntr, l in enumerate( els ):

        #creating the covariance matrix for this multipole
        TT_ori, EE_ori, TE_ori = delta_Cl_dic['TT'][lcntr], delta_Cl_dic['EE'][lcntr], delta_Cl_dic['TE'][lcntr]
        if with_lensing:
            Tphi_ori, Ephi_ori, PP_ori = delta_Cl_dic['Tphi'][lcntr], delta_Cl_dic['Ephi'][lcntr], delta_Cl_dic['PP'][lcntr]


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
        if 'TT' not in pspectra_to_use:
            null_TT = 1
        if 'EE' not in pspectra_to_use:
            null_EE = 1
        if 'TE' not in pspectra_to_use and (pspectra_to_use is not ['TT','EE'] or pspectra_to_use is not ['EE','TE']):
            null_TE = 1
        #20200611

        if (null_TT and null_EE and null_TE): continue# and null_PP): continue

        if l%100 == 0: print(l)

        #populate the covariance matrix
        TT, EE, TE = np.copy(TT_ori), np.copy(EE_ori), np.copy(TE_ori)


        if not with_lensing:
            COV_mat_l = np.zeros( (3,3) ) #TT, EE, TE
        else:
            COV_mat_l = np.zeros( (4,4) ) #TT, EE, TE

        COV_mat_l[0,0] = TT**2.
        COV_mat_l[1,1] = EE**2.
        COV_mat_l[2,2] = 0.5 * (TE**2. + TT * EE )

        COV_mat_l[0,1] = COV_mat_l[1,0] = TE**2.
        COV_mat_l[0,2] = COV_mat_l[2,0] = TT*TE
        COV_mat_l[1,2] = COV_mat_l[2,1] = EE*TE

        if with_lensing:
            COV_mat_l[3,3] = np.copy(PP_ori)**2.
            COV_mat_l[0,3] = COV_mat_l[3,0] = np.copy(Tphi_ori)**2.
            COV_mat_l[1,3] = COV_mat_l[3,1] = np.copy(Ephi_ori)**2.

        COV_mat_l_ori = np.copy(COV_mat_l)
        inv_COV_mat_l = sc.linalg.pinv2(COV_mat_l)
        #inv_COV_mat_l = np.linalg.inv(COV_mat_l)

        param_combinations = []
        for pcnt,p in enumerate(params):
            for pcnt2,p2 in enumerate(params):
                ##if [p2,p,pcnt2,pcnt] in param_combinations: continue
                param_combinations.append([p,p2, pcnt, pcnt2])


        for (p,p2, pcnt, pcnt2) in param_combinations:

            der1 = [Cl_deriv_dic[p]['TT'][lcntr], Cl_deriv_dic[p]['EE'][lcntr], Cl_deriv_dic[p]['TE'][lcntr]]
            der2 = [Cl_deriv_dic[p2]['TT'][lcntr], Cl_deriv_dic[p2]['EE'][lcntr], Cl_deriv_dic[p2]['TE'][lcntr]]

            if with_lensing:
                der1.append(Cl_deriv_dic[p]['PP'][lcntr])
                der2.append(Cl_deriv_dic[p2]['PP'][lcntr])

            der1 = np.asarray( der1 )
            der2 = np.asarray( der2 )

            if null_TT: der1[0] = der2[0] = 0.
            if null_EE: der1[1] = der2[1] = 0.
            if null_TE: der1[2] = der2[2] = 0.
            if null_PP: der1[3] = der2[3] = 0.

            fprime1_l_vec = np.copy(der1)
            fprime2_l_vec = np.copy(der2)

            curr_val = np.dot(fprime1_l_vec, np.dot( inv_COV_mat_l, fprime2_l_vec ))

            #from IPython import embed; embed()
            #print(X, Y, fprime1_l_vec, fprime2_l_vec, cov_extract, curr_val)

            ##if l > 3000: from IPython import embed; embed()

            F[pcnt2,pcnt] += curr_val

    if (0):
        from IPython import embed; embed()
        F = np.mat(F)
        #C = sc.linalg.pinv2(F) #made sure that COV_mat_l * Cinv_l ~= I
        C = np.linalg.inv(F) #made sure that COV_mat_l * Cinv_l ~= I

    return F 

#############################################################################################################################
def get_nldic(nlfile, els, tsznull = 0):
    dic = np.load(nlfile, allow_pickle = 1, encoding = 'latin1').item()
    el_nl, cl_residual = dic['el'], dic['cl_residual']
    if tsznull:
        if 'cl_residual_tsz_nulled' in dic:
            cl_residual = dic['cl_residual_tsz_nulled']
        elif 'cl_residual_nulled' in dic:
            cl_residual = dic['cl_residual_nulled']

    if 'T' in cl_residual:
        nl_TT, nl_EE = cl_residual['T'], cl_residual['P']
        nl_TT = np.interp(els, el_nl, nl_TT)
        nl_EE = np.interp(els, el_nl, nl_EE)
        nl_TE = None
    else:
        nl_TT = cl_residual['TT']
        if 'EE' in cl_residual:
            nl_EE = cl_residual['EE']
        else:
            nl_EE = None
        if 'TE' in cl_residual:
            nl_TE = cl_residual['TE']
        else:
            nl_TE = None
        el_nl = np.arange(len(nl_TT))
        nl_TT = np.interp(els, el_nl, nl_TT)

        if nl_EE is not None:
            nl_EE = np.interp(els, el_nl, nl_EE)
        else:
            nl_EE = np.zeros(len(els))

        if nl_TE is not None:
            nl_TE = np.interp(els, el_nl, nl_TE)
        else:
            nl_TE = np.zeros(len(els))

    nldic = {}
    nldic['TT'] = nl_TT
    nldic['EE'] = nl_EE
    nldic['TE'] = nl_TE

    try:
        fsky_val = dic['fsky_val']
    except:
        fsky_val = None

    return nldic, fsky_val

##########################################################################################
##########################################################################################

def get_latex_param_str(param):
    params_str_dic= {\
    'norm_YszM': r'${\rm log}(Y_{\ast})$', 'alpha_YszM': r'$\alpha_{_{Y}}$',\
    'beta_YszM': r'$\beta_{_{Y}}$', 'gamma_YszM': r'$\gamma_{_{Y}}$', \
    'alpha': r'$\eta_{\rm v}$', 'sigma_8': r'$\sigma_{\rm 8}$', \
    'one_minus_hse_bias': r'$1-b_{\rm SZ}$', 'omega_m': r'$\Omega_{\rm m}$', \
    'h0':r'$h$', 'm_nu':r'$\sum m_{\nu}$', 'ombh2': r'$\Omega_{b}h^{2}$', 'omch2': r'$\Omega_{c}h^{2}$', 'w0': r'$w_{0}$', 'wa': r'$w_{a}$', \
    'tau': r'$\tau_{\rm re}$', 'As': r'$A_{\rm s}$', 'ns': r'$n_{\rm s}$', 'neff': r'N$_{\rm eff}$', \
    'mnu': r'$\sum m_{\nu}$', 'thetastar': r'$\theta_{\ast}$', \
    }

    return params_str_dic[param]

def get_param_names_to_plot(desired_params_to_plot, param_names, fix_params = None):
    param_names_to_plot = []
    for pp in desired_params_to_plot:
        if pp in param_names and pp not in fix_params:
            param_names_to_plot.append(pp)
    return param_names_to_plot

def make_triangle_plot(exparr, F_dic, param_dict, cosmo_param_dict, tr, tc, param_names, desired_params_to_plot, fix_params, color_dic, one_or_two_sigma = 1, fsval = 12, noofticks = 4, exp_dic = None, use_percent = 0, ls_dic = None, show_one_sigma_lab = 1, bias_dic = None):

    #print('\n')
    import matplotlib.patches as patches
    import warnings, matplotlib.cbook
    warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)

    param_names_to_plot = []
    cosmo_param_pl_chars_dict = {}
    pcntr_for_plotting = 1
    for pp in sorted( desired_params_to_plot ):
        if pp in param_names and pp not in fix_params:
            param_names_to_plot.append(pp)
            cosmo_param_pl_chars_dict[pp] = pcntr_for_plotting
            pcntr_for_plotting += 1

    totparamstoplot = len(param_names_to_plot)
    diag_matrix = np.arange( totparamstoplot**2 ).reshape((totparamstoplot, totparamstoplot)) + 1

    sbpl_locs_dic = {}
    for p1 in param_names_to_plot:
        for p2 in param_names_to_plot:
            sbpl_locs_dic[(p1,p2)] = cosmo_param_pl_chars_dict[p1] + ((cosmo_param_pl_chars_dict[p2]-1) * totparamstoplot)

    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):        

            if p1 not in desired_params_to_plot or p2 not in desired_params_to_plot: continue
            if p1 in fix_params or p2 in fix_params: continue

            sbpl = sbpl_locs_dic[(p1,p2)]
            if sbpl not in np.tril(diag_matrix): continue

            cov_inds_to_extract = [(pcntr1, pcntr1), (pcntr1, pcntr2), (pcntr2, pcntr1), (pcntr2, pcntr2)]

            x = param_dict[p1]
            y = param_dict[p2]

            deltax, epsilon_x = cosmo_param_dict[p1][1]
            deltay, epsilon_y = cosmo_param_dict[p2][1]

            p1str = get_latex_param_str(p1)
            p2str = get_latex_param_str(p2)

            '''
            if p1 == 'As': x*=1e9
            if p2 == 'As':  y*=1e9

            #if p1 == 'thetastar': x*=100.
            #if p2 == 'thetastar': y*=100.
            '''

            x1, x2 = x - deltax, x + deltax
            y1, y2 = y - deltay, y + deltay

            ax = subplot(tr, tc, sbpl)#, aspect = 'equal')

            if sbpl<=(tr*(tc-1)):
                setp(ax.get_xticklabels(), visible=False)
            else:
                xlabel(p1str, fontsize = fsval);

            if ((sbpl-1)%tc == 0) and totparamstoplot>1 and sbpl!= 1:
                ylabel(p2str, fontsize = fsval);
            else:
                setp(ax.get_yticklabels(), visible=False)

            #print(p1, p2, sbpl)
            for expcntr, exp in enumerate( exparr ):

                F_mat = F_dic[exp]
                #exp_COV = sc.linalg.pinv(F_mat)
                exp_COV = np.linalg.inv(F_mat)

                #cov_extract = np.asarray( [exp_COV[ii] for ii in cov_inds_to_extract] ).reshape((2,2))
                cov_extract = []
                for ii in cov_inds_to_extract:
                    cov_extract.append(exp_COV[ii])
                cov_extract = np.asarray( cov_extract ).reshape((2,2))
                if (0):#p1 == p2 and p1 == 'alpha':
                    F_ex = []
                    for ii in cov_inds_to_extract:
                        F_ex.append(F_mat[ii])
                    F_ex = np.asarray( F_ex ).reshape((2,2))
                    print(cov_inds_to_extract, cov_extract, F_ex); #sys.exit()

                #if np.sum(cov_extract)<=1e-20: print(p1,p2, cov_extract); continue
                #print(p1, p2, cov_extract)

                colorarr = color_dic[exp]
                alphaarr = [1., 0.5]
                for ss in range(one_or_two_sigma):

                    if exp.find('spt4')>-1:
                        lwval = 1.
                        lsval = '-'
                    else:
                        lwval = .5
                        lsval = '-'
                    lwval = 1.
                    if ls_dic is not None:
                        lsval = ls_dic[exp][ss]
                    if p1 == p2:

                        widthval = cov_extract[0,0]**0.5##/2.35
                        #print(p1, widthval)
                        hor, ver = get_Gaussian(x, widthval, x1, x2)#, epsilon_x)
                        #labval = r'%.4f' %(widthval)
                        labval = None
                        if show_one_sigma_lab:
                            if bias_dic is not None:
                                labval = r'%.2f$\sigma$' %(bias_dic[p1]/widthval)
                                #print(p1, labval, bias_dic[p1], widthval)
                            else:
                                if abs(x)>0. and use_percent:
                                    labval = r'%.3f\%%' %(100. * abs(widthval/x))
                                else:
                                    labval = r'%g' %(widthval)
                        #print(labval)
                        if totparamstoplot==1 and exp_dic is not None:
                            labval = r'%s: %s' %(exp_dic[exp][0], labval)
                        plot(hor, ver, color = colorarr[ss], lw = lwval, label = labval, ls = lsval)
                        legend(loc = 4, framealpha = 1, fontsize = fsval-6, ncol = 1, edgecolor = 'None', handletextpad=0.8, handlelength = 0.8, numpoints = 1, columnspacing = 1)#, handlelength = 2.)

                        xlim(x1, x2)
                        ylim(0., 1.)
                        setp(ax.get_yticklabels(), visible=False); tick_params(axis='y',left='off')
                        title(p1str, fontsize = fsval);

                    else:

                        Ep = fn_get_ellipse_specs(cov_extract, howmanysigma = ss + 1)
                        widthval, heightval = Ep[0], Ep[1]
                        #if widthval<=1e-10 or heightval<=1e-10: continue
                        #print(widthval, heightval, p1, p2)
                        ellipse = patches.Ellipse(xy=[x,y], width=2.*widthval, height=2.*heightval, angle=np.degrees(Ep[2]))

                        ax.add_artist(ellipse)
                        ellipse.set_clip_box(ax.bbox)
                        ellipse.set_facecolor('None')#colorarr[ss])
                        ellipse.set_edgecolor(colorarr[ss])
                        ellipse.set_linewidth(lwval)
                        ellipse.set_linestyle(lsval)
                        #ellipse.set_alpha(alphaarr[ss])

                        xlim(x1, x2)
                        ylim(y1, y2)

                        if exp.find('bias') == -1:
                            axhline(y, lw = 0.1);axvline(x, lw = 0.1)

            if noofticks is not None:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=noofticks))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=noofticks))

            for label in ax.get_xticklabels(): label.set_fontsize(fsval-3.)
            for label in ax.get_yticklabels(): label.set_fontsize(fsval-3.)

            if (0):
                grid(True, which='major', axis = 'x', lw = 0.5, alpha = 0.1)
                grid(True, which='major', axis = 'y', lw = 0.5, alpha = 0.1)
    return

################################################################################
################################################################################
################################################################################

def get_JM_fisher_derivatives(els, which_spectra, gal_mask = 2, get_fisher = 1, get_derivatives = 1, get_pspectra = 1, get_sys_spectra = 1):
    import pickle 
    #jobName = 'checks/fisher_bias_checks/fisher_biasSr_covs_noPlanck'
    #jobName = '/Users/sraghunathan/Research/SPTpol/analysis/git/fisher_cosmo/checks/fisher_bias_checks/fisher_biasSr_covs_noPlanck_0.1'
    jobName = '/Users/sraghunathan/Research/SPTpol/analysis/git/fisher_cosmo/checks/fisher_bias_checks/fisher_biasSr'
    myf = open(jobName + '.pkl', 'rb')
    data = pickle.load(myf, encoding = 'latin1')
    myf.close()
    fisherAllExp=data['fisherGaussian']
    if gal_mask == 2:
        which_ind = 1
        fksy_JM = 0.57
    elif gal_mask == 5:
        which_ind = 2
        fksy_JM = 0.11

    if which_spectra == 'lensed_scalar':
        JM_spec_name = 'lensed'
    elif which_spectra == 'lensed_scalar_Alens0.30':
        JM_spec_name = 'delensed'
    elif which_spectra == 'unlensed_scalar':
        JM_spec_name = 'unlensed'

    sysSpectrum = data['sysSpectrum'][which_ind]
    F_mat_JM = fisherAllExp[which_ind][JM_spec_name] * fksy_JM
    biasVector_JM = data['biasVector'][which_ind][JM_spec_name]
    powersFid=data['powersFid'][which_ind][JM_spec_name]
    paramDerivs=data['paramDerivs'][which_ind]

    cosmoParams=data['cosmoParams']
    params_mapper_dic = {'tau':'tau', 'omega_c_h2':'omch2', 'A_s':'As', 'theta_s':'thetastar', 'N_eff':'neff', 'omega_b_h2':'ombh2', 'n_s': 'ns', 'mnu':'mnu'}#, 'YHe':}
    param_names_JM = []
    for p in cosmoParams:
        pmod = params_mapper_dic[p]
        param_names_JM.append( pmod )
    param_names_JM = np.asarray( param_names_JM )

    ret_dic = {}
    if get_fisher:
        F_mat = np.copy( F_mat_JM )
        param_names = np.copy( param_names_JM )
        ret_dic['F_mat'] = F_mat
        ret_dic['param_names'] = param_names
    if get_derivatives:
        cl_deriv_dic_JM = {}
        for p in cosmoParams:
            pmod = params_mapper_dic[p]
            tmpderivdic = paramDerivs[p][JM_spec_name]
            tt_der, ee_der, te_der = tmpderivdic['cl_TT'], tmpderivdic['cl_EE'], tmpderivdic['cl_TE']
            tt_der = np.interp(els, np.arange(len(tt_der)), tt_der)                
            ee_der = np.interp(els, np.arange(len(ee_der)), ee_der)
            te_der = np.interp(els, np.arange(len(te_der)), te_der)
            cl_deriv_dic_JM[pmod] = {}
            cl_deriv_dic_JM[pmod]['TT'] = tt_der
            cl_deriv_dic_JM[pmod]['EE'] = ee_der
            cl_deriv_dic_JM[pmod]['TE'] = te_der
        ret_dic['cl_deriv_dic'] = cl_deriv_dic_JM

    if get_pspectra:
        cl_dic = {}
        cl_dic['TT'] = powersFid['cl_TT']
        cl_dic['EE'] = powersFid['cl_EE']
        cl_dic['TE'] = powersFid['cl_TE']
        ret_dic['cl_dic'] = cl_dic

    if get_sys_spectra:
        cl_sysdic = {}
        cl_sysdic['TT'] = sysSpectrum['cl_TT']
        cl_sysdic['EE'] = sysSpectrum['cl_EE']
        ret_dic['cl_sysdic'] = cl_sysdic

    return ret_dic


def get_likelihood(data, model, C):

    """
    function to calculate the likelihood given data, model, covariance matrix
    """
    import scipy.linalg as linalg    
    C = np.mat(C)
    Cinv = linalg.pinv2(C)

    #sign, logdetval = np.linalg.slogdet(C)
    #logdetval = logdetval * sign

    d = data.flatten()
    dp = model.flatten()
    d = d-dp

    logLval =  -0.5 * np.asarray( np.dot(d.T, np.dot( Cinv, d ))).squeeze()

    return logLval


def random_sampler(x, y, howmanysamples = 100000, burn_in = 5000):

    import scipy.integrate as integrate
    import scipy.interpolate as interpolate

    #plot(x, y);show()
    norm = integrate.simps(y, x) #area under curve for norm
    y = y/norm #normalise dn/dM here

    cdf = np.asarray([integrate.simps(y[:i+1], x[:i+1]) for i in range(len(x))])
    #print(len(cdf));sys.exit()
    #plot(cdf);show()
    cdf_inv = interpolate.interp1d(cdf, x)

    random_sample = cdf_inv(np.random.rand(howmanysamples))
    #plot(random_sample);show()#;sys.exit()

    return random_sample[burn_in:]  

def get_width_from_sampling(x, likelihood_curve, which_percentile_1 = 16., which_percentile_2 = None):#, sigma_value = [1.]):

    if which_percentile_2 == None:
        which_percentile_2 = 100. - which_percentile_1
    randoms = random_sampler(x, likelihood_curve)
    mean_mass = x[np.argmax(likelihood_curve)]
    low_err = mean_mass - np.percentile(randoms, which_percentile_1)
    high_err = np.percentile(randoms, which_percentile_2) - mean_mass
    return mean_mass, low_err, high_err