import os, sys, warnings, argparse, h5py, numpy as np, time, itertools, random, shutil, datetime, tensorflow as tf
from laueotx.utils import logging as utils_logging
LOGGER = utils_logging.get_logger(__file__)

def get_sinkhorn_optimization_control_params(conf, n_target, n_model, max_nn_dist, eps_decrease=0.97, pixel_size=0.1, test=False):

    get_niter = lambda e, b, d : int(np.ceil((np.log(e)-np.log(b))/np.log(d)))
        
    # get annealing parameters

    noise_sig = float(conf['noise_sigma'])
    eps_init = 2*(max_nn_dist/3.)**2  # median fartherst neighbour is 3 sigma away 
    eps_min = 2*max(noise_sig, pixel_size)**2
    n_iter_annealing = get_niter(eps_min, eps_init, eps_decrease)

    if test:
        n_iter_annealing = 100
        LOGGER.warning('test!')

    method_ot = conf['solver']['method_ot']
    conf_method = conf['solver']['params_ot']

    LOGGER.info(f'running global optimization with method_ot={method_ot}, noise_sig={noise_sig:2.4f}') 
    LOGGER.info(f'--> annealing parameters: n_iter_annealing={n_iter_annealing} eps_init={eps_init:4.2e} eps_min={eps_min:4.2e} eps_decrease={eps_decrease:4.3f}')

    if 'unbalanced' in method_ot:

        kappa_obs = conf_method['unbalanced_kappa_obs'] # 0.1
        lambda_mod = conf_method['unbalanced_lambda_mod'] # 0.1
        n_iter_sinkhorn = 1000
        control_params = eps_init, eps_decrease, eps_min, kappa_obs, lambda_mod, n_iter_sinkhorn
        LOGGER.info(f'--> n_iter_sinkhorn={n_iter_sinkhorn}, outlier paramters: kappa_obs={kappa_obs:2.4f} lambda_mod={lambda_mod:2.4f}')

    elif 'balanced' in method_ot:

        n_iter_sinkhorn = 1000
        control_params = eps_init, eps_decrease, eps_min, n_iter_sinkhorn
        LOGGER.info(f'--> n_iter_sinkhorn={n_iter_sinkhorn}')

    elif 'partial' in method_ot:

        frac_mass = float(conf_method['partial_fracmass']) # 0.75
        n_iter_sinkhorn = 1000
        control_params = eps_init, eps_decrease, eps_min, frac_mass, n_iter_sinkhorn
        LOGGER.info(f'--> n_iter_sinkhorn={n_iter_sinkhorn}, outlier paramters: frac_mass={frac_mass:2.4f}')

    elif 'softslacks' in method_ot:

        n_iter_sinkhorn = 1000
        n_sig_outlier = float(conf_method['softslacks_nsig_outliers']) #3
        c_outlier = 0.5*(n_sig_outlier)**2 # n_sigmas
        n_max_unmatched = int(n_model * float(conf_method['softslacks_frac_unmatched'])) # 0.4
        n_max_outliers =  int(n_target * float(conf_method['softslacks_frac_outliers'])) # 0.05
        control_params = eps_init, eps_decrease, eps_min, c_outlier, n_max_unmatched, n_max_outliers, n_iter_sinkhorn
        LOGGER.info(f'--> n_iter_sinkhorn={n_iter_sinkhorn}, outlier paramters: outlier cost={c_outlier:2.4f} n_max_unmatched={n_max_unmatched} n_max_outliers={n_max_outliers}')


    return n_iter_annealing, control_params


def get_single_optimization_control_params(conf, max_nn_dist, eps_decrease=0.97, min_n_iter=32, pixel_size=0.1):


    get_niter = lambda e, b, d : int(np.ceil((np.log(e)-np.log(b))/np.log(d)))

    # get annealing parameters

    noise_sig = float(conf['noise_sigma'])
    eps_init = 2*(max_nn_dist/3.)**2  # median fartherst neighbour is 3 sigma away 
    eps_final = 2*max(noise_sig, pixel_size)**2
    n_iter = get_niter(eps_final, eps_init, eps_decrease)
    n_iter = max(n_iter, min_n_iter) # do not go below min_n_iter

    conf_opt = conf['solver']

    if conf_opt['method_single'] == 'softassign' or conf_opt['method_single'] == 'softlowmem':

        control_params = (eps_init, eps_final, eps_decrease)
        LOGGER.info(f'n_iter={n_iter} beta_init={eps_init:4.2e} beta_min={eps_final:4.2e} beta_decrease={eps_decrease:4.2f}')
    
    elif conf_opt['method_single'] == 'softent':

        tau_mod = float(conf_opt['outliers_tau'])
        control_params = (eps_init, tau_mod, eps_decrease)
        LOGGER.info(f'n_iter={n_iter} eps_init={eps_init:4.2e} eps_min={eps_final:4.2e} eps_decrease={eps_decrease:4.4f} tau_mod={tau_mod:4.2e}')

    elif conf_opt['method_single'] == 'softpart':

        outliers_m = float(conf_opt['outliers_m'])
        control_params = (eps_init, eps_decrease, outliers_m)
        LOGGER.info(f'n_iter={n_iter} eps_init={eps_init:4.2e} eps_min={eps_final:4.2e} eps_decrease={eps_decrease:4.4f} ot_m={outliers_m:4.2e}')


    elif conf_opt['method_single'] == 'softslacks':
        
        n_sig_outlier = 3.
        control_params = (eps_init, eps_decrease, n_sig_outlier)
        LOGGER.info(f'n_iter={n_iter} eps_init={eps_init:4.2e} eps_min={eps_final:4.2e} eps_decrease={eps_decrease:4.4f} n_sig_outlier={n_sig_outlier:4.2e}')

    else:
        control_params = None

    return n_iter, control_params


def get_neighbour_stats(sample_trials, inds_trials, lookup_data, s_target, n_grains_use=100):

    from laueotx.spot_neighbor_lookup import nn_lookup_all
    from laueotx.laue_math_tensorised import fast_full_forward_lab
    from laueotx.polycrystalline_sample import apply_selections

    nn_lookup_ind, lookup_pixel_size, lookup_n_pix = lookup_data
    i_grn_trials, i_ang_trials, i_det_trials = inds_trials
    n_grains_use = min(len(i_grn_trials), n_grains_use)

    s_lab, p_lab, p_lam, select = fast_full_forward_lab(sample_trials.U[:n_grains_use], sample_trials.x0[:n_grains_use], sample_trials.Gamma, sample_trials.v, sample_trials.dn[:n_grains_use], sample_trials.d0[:n_grains_use], sample_trials.dr[:n_grains_use], sample_trials.dl[:n_grains_use], sample_trials.dh[:n_grains_use], sample_trials.lam, I=tf.eye(3, dtype=tf.float64))
    s_lab_select, i_grn_select, i_ang_select, i_det_select = apply_selections(select, s_lab, i_grn_trials[:n_grains_use], i_ang_trials[:n_grains_use], i_det_trials[:n_grains_use])
    spotind_nn = nn_lookup_all(nn_lookup_ind, s_target, s_lab_select, i_ang_select[:,tf.newaxis], i_det_select[:,tf.newaxis], i_grn_select[:,tf.newaxis], lookup_pixel_size, lookup_n_pix)
    s_fnn = tf.gather(s_target, spotind_nn[:,-1]) # farthest nearest neighbour
    max_l2 = np.median(tf.math.sqrt(tf.reduce_sum((s_lab_select-s_fnn)**2, axis=-1))) # median farthest nearest neighbours
    ui, uc = np.unique(i_grn_select, return_counts=True)
    n_spots_mean = np.median(uc)

    LOGGER.info(f'median distance to the farthest neighbour {max_l2:6.2f} n_spots_mean={n_spots_mean:4.2f}')

    return max_l2, n_spots_mean