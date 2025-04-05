import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kl_sample.io as io
import kl_sample.reshape as rsh
import kl_sample.settings as set
import kl_sample.cosmo as cosmo_tools
import kl_sample.likelihood as lkl


def plots(args):
    """ Generate plots for the papers.

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to fits files file the output.

    """

    fields = set.FIELDS_CFHTLENS

    # Define absolute paths
    path = {}
    path['params'] = io.path_exists_or_error(args.params_file)
    path['mcm'] = io.path_exists_or_error(io.read_param(args.params_file, 'mcm', type='path'))+'/'
    path['fourier'] = io.path_exists_or_error('{}/data/data_fourier.fits'.format(sys.path[0]))
    path['real'] = io.path_exists_or_error('{}/data/data_real.fits'.format(sys.path[0]))
    path['output'] = io.path_exists_or_create(os.path.abspath(args.output_path))+'/'

    # Settings
    settings = {
            'kl_scale_dep' : True,
            'method' : 'kl_diag',
            'n_kl' : 7
        }


    # Read data
    ell = io.read_from_fits(path['fourier'], 'ELL')
    cl_EE = io.read_from_fits(path['fourier'], 'CL_EE')
    cl_BB = io.read_from_fits(path['fourier'], 'CL_BB')
    noise_EE = io.read_from_fits(path['fourier'], 'CL_EE_NOISE')
    noise_BB = io.read_from_fits(path['fourier'], 'CL_BB_NOISE')
    sims_EE = io.read_from_fits(path['fourier'], 'CL_SIM_EE')
    sims_BB = io.read_from_fits(path['fourier'], 'CL_SIM_BB')
    pz = io.read_from_fits(path['fourier'], 'PHOTO_Z')
    n_eff = io.read_from_fits(path['fourier'], 'N_EFF')
    sigma_g = io.read_from_fits(path['fourier'], 'SIGMA_G')
    pz_r = io.read_from_fits(path['real'], 'PHOTO_Z')
    n_eff_r = io.read_from_fits(path['real'], 'N_EFF')
    sigma_g_r = io.read_from_fits(path['real'], 'SIGMA_G')
    kl_t = io.read_from_fits(path['fourier'], 'KL_T_ELL')
#    io.print_info_fits(path['fourier'])


    # Clean data from noise
    cl_EE = rsh.clean_cl(cl_EE, noise_EE)
    sims_EE = rsh.clean_cl(sims_EE, noise_EE)
    cl_BB = rsh.clean_cl(cl_BB, noise_BB)
    sims_BB = rsh.clean_cl(sims_BB, noise_BB)

    cl_EE_kl = lkl.apply_kl(kl_t, cl_EE, settings)
    sims_EE_kl = lkl.apply_kl(kl_t, sims_EE, settings)
    cov_pf_kl = rsh.get_covmat_cl(sims_EE_kl,is_diag=True)
    cl_EE_kl = rsh.unify_fields_cl(cl_EE_kl, cov_pf_kl, is_diag=True)
    sims_EE_kl = rsh.unify_fields_cl(sims_EE_kl, cov_pf_kl, is_diag=True)

    # Mask observed Cl's
#    cl_EE_kl = rsh.mask_cl(cl_EE_kl, is_diag=True)

    # Calculate covmat Cl's
#    sims_EE_kl = rsh.mask_cl(sims_EE_kl, is_diag=True)
    cov_kl = rsh.get_covmat_cl(sims_EE_kl, is_diag=True)
    factor = (2000.-35.-2.)/(2000.-1.)
    cov_kl = cov_kl/factor


    # Create array with cosmo parameters
    params_name = ['h', 'omega_c', 'omega_b', 'ln10_A_s', 'n_s', 'w_0', 'w_A']
    params_val = io.read_cosmo_array(path['params'], params_name)[:,1]

    # Get theory Cl's
    bp = set.BANDPOWERS
    n_bins = len(bp)
    cosmo = cosmo_tools.get_cosmo_ccl(params_val)
    th_cl = cosmo_tools.get_cls_ccl(params_val, cosmo, pz, bp[-1,-1])

    from astropy.io import fits
    d=fits.open("/users/groups/damongebellini/data/preliminary/7bins/cls_bf.fits")
    cl_ee=d[2].data[:len(th_cl)]

    # th_cl = rsh.bin_cl(th_cl, bp)
    tot_ell = np.arange(bp[-1,-1]+1)
    th_cl,th_cl_BB = rsh.couple_decouple_cl(tot_ell, th_cl, path['mcm'], len(fields), n_bins, len(bp),
                                            return_BB=True)
    th_clb,th_cl_BBb = rsh.couple_decouple_cl(tot_ell, cl_ee, path['mcm'], len(fields), n_bins, len(bp),
                                              return_BB=True)

    th_cl_kl = lkl.apply_kl(kl_t, th_cl, settings)
    th_cl_kl = rsh.unify_fields_cl(th_cl_kl, cov_pf_kl, is_diag=True)

    io.write_to_fits(fname=path['output']+'obs.fits', array=ell, name='ell')
    io.write_to_fits(fname=path['output']+'obs.fits', array=cl_EE_kl, name='cl_EE_obs')
    io.write_to_fits(fname=path['output']+'obs.fits', array=cov_kl, name='cov_obs')
    io.write_to_fits(fname=path['output']+'obs.fits', array=th_cl_kl, name='cl_EE_th')
    plt.figure()
    x = ell
    for b1 in range(3):
        y1 = th_cl_kl[:,b1]
        y2 = cl_EE_kl[:,b1]
        err = np.sqrt(np.diag(cov_kl[:,:,b1,b1]))
#        plt.plot(x, y1, label = '$\\sigma_g^2/n_{eff}$')
        plt.errorbar(x, y2, yerr=err, label = 'Bin {}'.format(b1+1), fmt='o')
        plt.errorbar(x, -y2, yerr=err, label = 'Bin {}'.format(b1+1), fmt='^')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\\ell$')
    plt.ylabel('$\\hat{C}_\\ell^{EE}$')
    plt.legend(loc='best')
    plt.savefig('{}cl_kl_data.pdf'.format(path['output']))
    plt.close()

    exit(1)

    cov_pf = rsh.get_covmat_cl(sims_EE)
    cov_pf_BB = rsh.get_covmat_cl(sims_BB)
    th_cl = rsh.unify_fields_cl(th_cl, cov_pf)
    th_cl_BB = rsh.unify_fields_cl(th_cl_BB, cov_pf_BB)
    th_clb = rsh.unify_fields_cl(th_clb, cov_pf)
    th_cl_BBb = rsh.unify_fields_cl(th_cl_BBb, cov_pf_BB)

    # Unify fields
    cl_EE = rsh.unify_fields_cl(cl_EE, cov_pf)
    noise_EE = rsh.unify_fields_cl(noise_EE, cov_pf)
    sims_EE = rsh.unify_fields_cl(sims_EE, cov_pf)
    cl_BB = rsh.unify_fields_cl(cl_BB, cov_pf_BB)
    noise_BB = rsh.unify_fields_cl(noise_BB, cov_pf_BB)
    sims_BB = rsh.unify_fields_cl(sims_BB, cov_pf_BB)

    # Average simulations
    sims_EE_avg = np.average(sims_EE, axis=0)
    sims_BB_avg = np.average(sims_BB, axis=0)

    # Calculate covmat
    covmat_EE = rsh.get_covmat_cl(sims_EE)
    covmat_BB = rsh.get_covmat_cl(sims_BB)
    np.savez("cls_all",dd_EE=cl_EE,dd_BB=cl_BB,
             tt_EE=th_cl,tt_BB=th_cl_BB,tb_EE=th_clb,tb_BB=th_cl_BBb,
             ss_EE=sims_EE,ss_BB=sims_BB,cv_EE=covmat_EE,cv_BB=covmat_BB)


    # Noise based on n_eff and sigma_g
    n_eff = n_eff*(180.*60./np.pi)**2. #converted in stedrad^-1
    noise_ns = np.array([np.diag(sigma_g**2/n_eff) for x in ell])
    # Noise based on n_eff and sigma_g
    n_eff_r = n_eff_r*(180.*60./np.pi)**2. #converted in stedrad^-1
    noise_ns_r = np.array([np.diag(sigma_g_r**2/n_eff_r) for x in ell])
    # Average noise
    noise_EE_avg = rsh.debin_cl(noise_EE, bp)
    noise_EE_avg = np.average(noise_EE_avg[bp[0,0]:bp[-1,-1]],axis=0)
    noise_EE_avg = np.array([noise_EE_avg for x in ell])

    # Plot noise
    x = ell
    for b1 in range(n_bins):
        for b2 in range(b1,n_bins):
            y1 = noise_ns[:,b1,b2]
            y2 = noise_EE_avg[:,b1,b2]
            y3 = noise_EE[:,b1,b2]
            y4 = noise_ns_r[:,b1,b2]
            plt.figure()
            plt.plot(x, y1, label = '$\\sigma_g^2/n_{eff}$')
            plt.plot(x, y2, label = 'Average')
            plt.plot(x, y3, 'o', label = 'Noise')
            plt.plot(x, y4, label = '$\\sigma_g^2/n_{eff}$ real')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.title('Bins {} {}'.format(b1+1,b2+1))
            plt.xlabel('$\\ell$')
            plt.ylabel('$N_\\ell^{EE}$')
            plt.savefig('{}noise_bin{}{}.pdf'.format(path['output'],b1+1,b2+1))
            plt.close()



    # Plot Cl
    x = ell
    for b1 in range(n_bins):
        for b2 in range(b1,n_bins):
            y1 = cl_EE[:,b1,b2]
            y2 = th_cl[:,b1,b2]
            y3 = sims_EE_avg[:,b1,b2]
            err1 = np.sqrt(np.diag(covmat_EE[:,:,b1,b1,b2,b2]))
            plt.figure()
            plt.errorbar(x, y1, yerr=err1, fmt='-o', label='data')
            plt.plot(x, y2, label='theory')
            plt.plot(x, y3, label='simulations')
            plt.xscale('log')
            plt.legend(loc='best')
            plt.title('Bins {} {}'.format(b1+1,b2+1))
            plt.xlabel('$\\ell$')
            plt.ylabel('$C_\\ell^{EE}$')
            plt.savefig('{}cl_bin{}{}.pdf'.format(path['output'],b1+1,b2+1))
            plt.close()

    return
