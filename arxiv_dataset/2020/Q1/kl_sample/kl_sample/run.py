"""

This module contains the main function run, from where
it is possible to run an MCMC (emcee), or evaluate the
likelihood at one single point (single_point).

"""

import numpy as np
import kl_sample.io as io
import kl_sample.cosmo as cosmo_tools
import kl_sample.checks as checks
import kl_sample.likelihood as lkl
import kl_sample.reshape as rsh
import kl_sample.settings as set
import kl_sample.sampler as sampler


def run(args):
    """ Run with different samplers: emcee, single_point

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to file the output (emcee) or just
        print on the screen the likelihood (single_point)

    """

    # ----------- Initialize -------------------------------------------------#

    # Define absolute paths and check the existence of each required file
    path = {
        'params': io.path_exists_or_error(args.params_file),
        'data':   io.read_param(args.params_file, 'data', type='path'),
        'output': io.read_param(args.params_file, 'output', type='path')
    }
    io.path_exists_or_error(path['data'])
    io.path_exists_or_create(path['output'])

    # Create array with cosmo parameters
    add_ia = io.read_param(path['params'], 'add_ia', type='bool')
    cosmo = {
        'names': ['h', 'omega_c', 'omega_b', 'ln10_A_s', 'n_s', 'w_0', 'w_A']
    }
    if add_ia:
        cosmo['names'].append('A_IA')
        cosmo['names'].append('beta_IA')
    cosmo['params'] = io.read_cosmo_array(path['params'], cosmo['names'])
    cosmo['mask'] = cosmo_tools.get_cosmo_mask(cosmo['params'])

    # Read and store the remaining parameters
    settings = {
        'sampler': io.read_param(path['params'], 'sampler'),
        'space': io.read_param(path['params'], 'space'),
        'method': io.read_param(path['params'], 'method'),
        'n_sims': io.read_param(path['params'], 'n_sims'),
        'add_ia': add_ia
    }
    # Sampler settings
    if settings['sampler'] == 'emcee':
        settings['n_walkers'] = \
            io.read_param(path['params'], 'n_walkers', type='int')
        settings['n_steps'] = \
            io.read_param(path['params'], 'n_steps', type='int')
        settings['n_threads'] = \
            io.read_param(path['params'], 'n_threads', type='int')
    # KL settings
    if settings['method'] in ['kl_diag', 'kl_off_diag']:
        settings['n_kl'] = io.read_param(path['params'], 'n_kl', type='int')
        settings['kl_scale_dep'] = \
            io.read_param(path['params'], 'kl_scale_dep', type='bool')
    if settings['method'] == 'kl_diag':
        is_diag = True
    else:
        is_diag = False
    # Real/Fourier space settings and mcm path
    if settings['space'] == 'real':
        settings['ell_max'] = \
            io.read_param(path['params'], 'ell_max', type='int')
    elif settings['space'] == 'fourier':
        settings['bp_ell'] = set.BANDPOWERS
        settings['ell_max'] = settings['bp_ell'][-1, -1]
        settings['mcm'] = io.read_param(args.params_file, 'mcm', type='path')
        io.path_exists_or_error(settings['mcm'])

    # Check if there are unused parameters.
    checks.unused_params(cosmo, settings, path)

    # Perform sanity checks on the parameters and data file
    checks.sanity_checks(cosmo, settings, path)

    # Read data
    data = {
        'photo_z': io.read_from_fits(path['data'], 'photo_z')
    }
    if settings['space'] == 'real':
        data['theta_ell'] = np.array(set.THETA_ARCMIN)/60.  # theta in degrees
        data['mask_theta_ell'] = set.MASK_THETA
        data['corr_obs'] = io.read_from_fits(path['data'], 'xipm_obs')
        data['corr_sim'] = io.read_from_fits(path['data'], 'xipm_sim')
    elif settings['space'] == 'fourier':
        data['theta_ell'] = io.read_from_fits(path['data'], 'ELL')
        data['mask_theta_ell'] = set.MASK_ELL
        cl_EE = io.read_from_fits(path['data'], 'CL_EE')
        noise_EE = io.read_from_fits(path['data'], 'CL_EE_NOISE')
        sims_EE = io.read_from_fits(path['data'], 'CL_SIM_EE')
        data['corr_obs'] = rsh.clean_cl(cl_EE, noise_EE)
        data['corr_sim'] = rsh.clean_cl(sims_EE, noise_EE)
    if settings['method'] in ['kl_diag', 'kl_off_diag']:
        if settings['kl_scale_dep']:
            data['kl_t'] = io.read_from_fits(path['data'], 'kl_t_ell')
        else:
            data['kl_t'] = io.read_from_fits(path['data'], 'kl_t')

    # Add some dimension to settings (n_bins, n_x_var)
    settings['n_fields'] = data['corr_sim'].shape[0]
    settings['n_sims_tot'] = data['corr_sim'].shape[1]
    settings['n_bins'] = len(data['photo_z']) - 1
    settings['n_theta_ell'] = len(data['theta_ell'])

    # Calculate number of elements in data vector
    settings['n_data'] = \
        len(data['mask_theta_ell'].flatten()[data['mask_theta_ell'].flatten()])
    settings['n_data_tot'] = \
        settings['n_data']*settings['n_bins']*(settings['n_bins']+1)/2
    if settings['method'] == 'kl_diag':
        settings['n_data'] = settings['n_data']*settings['n_kl']
    elif settings['method'] == 'kl_off_diag':
        settings['n_data'] = \
            settings['n_data']*settings['n_kl']*(settings['n_kl']+1)/2
    else:
        settings['n_data'] = \
            settings['n_data']*settings['n_bins']*(settings['n_bins']+1)/2

    # ------------------- Preliminary computations ---------------------------#

    if set.BNT:
        data['photo_z'] = io.read_from_fits(path['data'], 'photo_z')
        bnt = cosmo_tools.BNT(cosmo['params'], data['photo_z'])
        data['bnt_mat'] = bnt.get_matrix()

    # Apply KL
    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        data['corr_obs'] = \
            lkl.apply_kl(data['kl_t'], data['corr_obs'], settings)
        data['corr_sim'] = \
            lkl.apply_kl(data['kl_t'], data['corr_sim'], settings)

    # Compute how many simulations have to be used
    settings['n_sims'] = \
        lkl.how_many_sims(settings['n_sims'], settings['n_sims_tot'],
                          settings['n_data'], settings['n_data_tot'])
    data['corr_sim'] = lkl.select_sims(data, settings)

    # Prepare data if real
    if settings['space'] == 'real':
        # Reshape observed correlation function
        data['corr_obs'] = rsh.flatten_xipm(data['corr_obs'], settings)
        # Reshape simulated correlation functions
        cs = np.empty((settings['n_fields'],
                      settings['n_sims'])+data['corr_obs'].shape)
        for nf in range(settings['n_fields']):
            for ns in range(settings['n_sims']):
                cs[nf][ns] = \
                    rsh.flatten_xipm(data['corr_sim'][nf][ns], settings)
        data['corr_sim'] = cs
        # Mask observed correlation function
        data['corr_obs'] = rsh.mask_xipm(data['corr_obs'],
                                         data['mask_theta_ell'], settings)
        # Compute inverse covariance matrix
        data['inv_cov_mat'] = lkl.compute_inv_covmat(data, settings)

    # Prepare data if fourier
    else:
        # Mask Cl's
        data['corr_obs'] = rsh.mask_cl(data['corr_obs'], is_diag=is_diag)
        data['corr_sim'] = rsh.mask_cl(data['corr_sim'], is_diag=is_diag)
        # Unify fields
        data['cov_pf'] = rsh.get_covmat_cl(data['corr_sim'], is_diag=is_diag)
        data['corr_obs'] = rsh.unify_fields_cl(data['corr_obs'],
                                               data['cov_pf'], is_diag=is_diag,
                                               pinv=set.PINV)
        data['corr_sim'] = rsh.unify_fields_cl(data['corr_sim'],
                                               data['cov_pf'], is_diag=is_diag,
                                               pinv=set.PINV)
        # Apply BNT if required
        if set.BNT:
            data['corr_obs'] = \
                cosmo_tools.apply_bnt(data['corr_obs'], data['bnt_mat'])
            data['corr_sim'] = \
                cosmo_tools.apply_bnt(data['corr_sim'], data['bnt_mat'])
        # Reshape observed Cl's
        data['corr_obs'] = rsh.flatten_cl(data['corr_obs'], is_diag=is_diag)
        # Calculate covmat Cl's
        cov = rsh.get_covmat_cl(data['corr_sim'], is_diag=is_diag)
        cov = rsh.flatten_covmat(cov, is_diag=is_diag)
        factor = \
            (settings['n_sims']-settings['n_data']-2.)/(settings['n_sims']-1.)
        if set.PINV:
            data['inv_cov_mat'] = factor*np.linalg.pinv(cov)
        else:
            data['inv_cov_mat'] = factor*np.linalg.inv(cov)

    # Import Camera's template
    if set.THEORY == 'Camera':
        settings['cls_template'] = \
            io.import_template_Camera(set.CLS_TEMPLATE, settings)

    # ------------------- Run ------------------------------------------------#

    if settings['sampler'] == 'emcee':
        sampler.run_emcee(args, cosmo, data, settings, path)
    elif settings['sampler'] == 'single_point':
        sampler.run_single_point(cosmo, data, settings)

    print('Success!!')

    return
