# key packages imports
import os, sys, warnings, argparse, h5py, numpy as np, time, itertools, random, shutil, datetime, click
from copy import deepcopy
from collections import OrderedDict
from tqdm.auto import trange, tqdm
from pathlib import Path 


# tensorflow imports and settings
import tensorflow as tf
tf.config.set_soft_device_placement(False)
tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(False)

# laueotx imports
from laueotx.utils import logging as utils_logging
from laueotx.utils import io as utils_io
from laueotx.utils import config as utils_config
from laueotx import laue_math, laue_math_tensorised, laue_math_graph, assignments
from laueotx.detector import Detector
from laueotx.beamline import Beamline
from laueotx.filenames import get_filename_realdata_part, get_filename_realdata_merged
from laueotx.grain import Grain
from laueotx.laue_rotation import Rotation

# warnings and logger
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

# constants
FLAG_OUTLIER = -999999
astr = lambda x: np.array2string(np.atleast_1d(x), max_line_width=1000, precision=5, formatter={'all': lambda x: '{: 2.6f}'.format(x)})
astr3 = lambda x: np.array2string(np.atleast_1d(x), max_line_width=1000, precision=3, formatter={'all': lambda x: '{: 6.3f}'.format(x)})
mse = lambda x, y: np.mean((x - y)**2)
rms = lambda x, y: np.sqrt(np.mean((x - y)**2))
# sinkhorn_method = {'softslacks': 'sinkhorn_slacks_sparse', 'softent': 'sinkhorn_unbalanced_sparse'}


import functools
def common_options(f):
    @click.option('--conf', "-c", required=True, type=str,  show_default=True,help='Configuration yaml file') # type=click.File(),
    @click.option('output_dir','--output-dir', "-o", required=True,type=str, show_default=True,help='Directory to store the produced files') # type=click.Path(), 
    @click.option('--verbosity','-v', type=click.Choice( ['critical', 'error', 'warning', 'info', 'debug']), default="info", show_default=True,help='Logging level')
    @click.option('--n-grid', default=2000, show_default=True,help='number of grid points from which to initialize coordinate descent')
    # @click.option("--calibrate-coniga/--no-calibrate-coninga",default=False, help="Calibrate the coniga sample")
    # @click.option("--calibrate-fenimn/--no-calibrate-fenimn", default=False, help="Calibrate the fenimn sample")
    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options

@click.group("realdata")
def main():
    pass

@main.command()
@common_options
@click.argument("tasks", nargs=-1)
def singlegrain(conf: str | Path, output_dir: str, verbosity: bool, n_grid: int, tasks: list):
    """Perform single-grain fitting on part (or full) initialization grid.
    This will follow these steps:
        1. Create a grid of initial points in the grain position-orientation space
        2. For each point, it will run a single-grain coordinate descent to find the best values of these parameters, jointly with spot assignments
    

    Parameters
    ----------
    conf : str | Path
        Path to the configuration file
    output_dir : str
        Path to the output directory to store the (intermediate) results
    verbosity : bool
        Verbosity level for printing to console
    n_grid : int
        Number of single-grain initialial guesses per task. They will be distributed using a Sobol sequence inside the grain position and orientation space.
    tasks : list
        List of integer task ids to use. Each task will compute n_grid single-grain fitting solutions. The starting points will be created using Sobol points with indices `(task_id * n_grid):((task_id+1) * n_grid)`.

    Returns
    -------
    output


    """

    # type-proff indices
    indices = [int(i) for i in tasks]

    # read args
    args = {} # TODO: implement additional arguments from commandline using click
    args = argparse.Namespace(**args)
    utils_logging.set_all_loggers_level(verbosity)

    # read config and make output dirs
    conf = utils_io.get_abs_path(conf)
    output_dir = utils_io.get_abs_path(output_dir)
    utils_io.robust_makedirs(output_dir)
    conf = utils_io.read_config(conf, args)

    # find experiment rotation (projection) angles
    omegas = utils_config.get_angles(conf['angles'])
    LOGGER.info(f'got {len(omegas)} angles')

    # set up detectors and beamline
    dt = get_detectors_from_config(conf)
    bl = Beamline(omegas=omegas, 
                  detectors=dt,
                  lambda_lim=[conf['beam']['lambda_min'], conf['beam']['lambda_max']])

    # read input peaks data
    mpd = get_peaskdata_from_config(conf, bl)

    # loop over tasks
    for index in indices:

        LOGGER.info(f'=================> running on index {index} / {str(indices)}')
        
        # main analysis
        dict_out = analyse(n_grid, deepcopy(conf), mpd, index, test=False)

        # store output
        filename_out = get_filename_realdata_part(output_dir, tag=conf['tag'], index=index)
        utils_io.write_arrays(filename_out, **dict_out)



def analyse(n_grid, conf, mpd, index=0, test=False):
    """Run single-grain fitting function for a given index. Output the optimized parameters of grains.
    
    Parameters
    ----------
    n_grid : int
        Number of starting points to use.
    conf : TYPE
        Configuration dictionary.
    mpd : from laueotx.peaks_data.MultidetectorPeaksData
        Multi-detector peaks data containing the detected peaks positions.
    index : int, optional
        Index of the task to process. Each task will compute n_grid single-grain fitting solutions. The starting points will be created using Sobol points with indices `(index * n_grid):((index+1) * n_grid)`.
    test : bool, optional
        Test mode.
    
    Returns
    -------
    TYPE
        Description
    """
    from laueotx import laue_coordinate_descent
    from laueotx.utils import inversion as utils_inversion
    from laueotx.polycrystalline_sample import polycrystalline_sample, get_batch_indices_full, select_indices, merge_duplicated_spots, get_sobol_grid_laue
    from laueotx.spot_neighbor_lookup import spot_neighbor_lookup, nn_lookup_all
    from laueotx.laue_rotation import get_rotation_constraint_rf, get_max_rf
    from laueotx.spot_prototype_selection import remove_candidates_with_bad_fits, prune_similar_solutions
    from laueotx.deterministic_annealing import get_neighbour_stats, get_single_optimization_control_params

    # get observed data
    s_obs = mpd.to_lab_coords()
    n_trials = int(n_grid)

    # re-case indices of angles and detectors
    def prep_inds(x):
        x = np.array(x).reshape(-1,1)
        x = np.unique(x, return_inverse=True)[1][:,np.newaxis]
        return x 
        
    # read indices of rotation angles and detectors
    i_det_obs = prep_inds(mpd.peaks['id_detector'])
    i_ang_obs = prep_inds(mpd.peaks['id_angle'])
    # frac_outliers = 0

    base_a, base_x, max_grain_pos, max_grain_rot = get_experimental_sample_params(deepcopy(conf))

    # # use a single grain
    # select_grain = 0
    # base_a = base_a[[select_grain]]
    # base_x = base_x[[select_grain]]

    # input-proof
    # n_grains_obs = len(base_a)
    # n_grn_sample = len(base_a)

    # render spots from the previously measured grain parameters
    # sample = polycrystalline_sample(deepcopy(conf), hkl_sign_redundancy=True, rotation_type='mrp')
    # sample.set_tensor_variables(base_a, base_x)
    # i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=n_grn_sample)
    # s_lab, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='laboratory')
    # # s_lab, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='detector')
    # s_lab, i_grn_sample, i_ang_sample, i_hkl_sample, i_det_sample, i_all_sample = select_indices(select_sample, s_lab, i_grn, i_ang, i_hkl, i_det, i_all)
    # s_lab, i_grn_sample, i_ang_sample, i_hkl_sample, i_det_sample, i_all_sample, select_merged = merge_duplicated_spots(s_lab, i_grn_sample, i_ang_sample, i_hkl_sample, i_det_sample, i_all_sample, decimals=4, return_index=True, split_by_grain=True)

    # p_lam_sample = np.tile(np.expand_dims(p_lam, axis=-1), [1, 1, 1, n_detectors])[select_sample]
    # p_lam_merged = p_lam_sample[select_merged]

    # if index == 0:

    #     try:
    #         fname_out = 'coord_test_{}.h5'.format(conf['tag'])
    #         utils_io.write_arrays(fname_out, 'w', mpd.peaks, s_obs, s_lab, i_grn_sample, i_ang_sample, i_hkl_sample, i_det_sample, p_sample, p_lam_sample, p_lam_merged)
    #     except Exception as err:
    #         LOGGER.warning(f'failed to write {fname_out} err={err}')
    
    # for i in range(n_detectors):
    #     select = i_det_sample==i
    #     LOGGER.info(f'n_peaks detector {i} mod={np.count_nonzero(i_det_sample==i):>8d} obs={np.count_nonzero(i_det_obs==i):>8d} lam_min={np.min(p_lam_merged[select[:,0]]):2.4f} lam_max={np.max(p_lam_merged[select[:,0]]):2.4f}')

    # solver params
    opt_fun = conf['solver']['method_single']
    lookup_n_pix = int(conf['solver']['lookup_n_pix']) # pix
    n_neighbours = int(conf['solver']['n_neighbours'])  # pix
    det_side_length =  conf['detectors'][0]['side_length'] # mm
    sig_s = float(conf['noise_sigma'])

    if test:
        LOGGER.warning('test!')
        n_trials = 100
        batch_size = 10
        lookup_n_pix = 100
        n_neighbours = 3

    # get lookup
    snl = spot_neighbor_lookup(conf=deepcopy(conf), 
                               s_spot=s_obs, 
                               i_ang=i_ang_obs, 
                               i_det=i_det_obs, 
                               lookup_n_pix=lookup_n_pix, 
                               detector_side_length=det_side_length, 
                               n_neighbours=n_neighbours)
    nn_lookup_ind = tf.constant(snl.nn_lookup_ind.numpy(), dtype=tf.int32)
    lookup_pixel_size=snl.detector_side_length/snl.lookup_n_pix
    lookup_n_pix=snl.lookup_n_pix
    # i_nn = nn_lookup_all(nn_lookup_ind, s_obs, s_lab, i_ang_sample, i_det_sample, i_grn_sample, lookup_pixel_size, lookup_n_pix)

    # get trials grid
    sobol_samples = tf.math.sobol_sample(dim=6, num_results=n_trials*2, skip=index*n_trials*2, dtype=tf.float64).numpy()*2 -1 # numbers between -1 and 1, start with more trials to make sure there is enough samples after rotation constraints is satisfied
    rot_grid = sobol_samples[:,0:3] * get_max_rf(laue_group=conf['sample']['laue_group'])
    pos_grid = sobol_samples[:,3:6] * float(conf['sample']['size'])
    select_rot_constraint = get_rotation_constraint_rf(rot_grid, laue_group=conf['sample']['laue_group']) # TODO: make this more efficient, with respect to the Sobol sampling above
    rot_grid = rot_grid[select_rot_constraint][:n_trials]
    pos_grid = pos_grid[select_rot_constraint][:n_trials]
    rot_grid = Rotation.from_gibbs(rot_grid).as_mrp()
    LOGGER.info(f'created rot_grid {rot_grid.shape}')
    LOGGER.info(f'created pos_grid {pos_grid.shape}')

    # create the sample
    sample_trials = polycrystalline_sample(deepcopy(conf), hkl_sign_redundancy=True, rotation_type='mrp', restrict_fitting_area=True)
    sample_trials.set_tensor_variables(rot_grid, pos_grid)

    # get constants
    batch_size = int(conf['solver']['batch_size'])
    i_grn_trials, i_ang_trials, i_hkl_trials, i_det_trials, i_all_trials = get_batch_indices_full(n_grn=batch_size, 
                                                                                                  n_ang=len(sample_trials.Gamma),
                                                                                                  n_hkl=len(sample_trials.v[...,0]),
                                                                                                  n_det=len(sample_trials.dt))

    # find neighbours params and run configuration
    max_nn_dist, _ = get_neighbour_stats(sample_trials, 
                                         inds_trials=(i_grn_trials, i_ang_trials, i_det_trials), 
                                         lookup_data=(nn_lookup_ind, lookup_pixel_size, lookup_n_pix), 
                                         s_target=s_obs)
    n_iter, control_params = get_single_optimization_control_params(conf, max_nn_dist)
    LOGGER.info(f'running optimization {opt_fun} with n_steps_annealing={n_iter} batch_size={batch_size}')

    # run analysis
    time_start = time.time()
    a_fit, x_fit, loss, nspot, fracin = laue_math_graph.batch_optimize_grain(n_per_batch=batch_size,
                                                                             s_target=s_obs,
                                                                             nn_lookup_data=(nn_lookup_ind, lookup_pixel_size, lookup_n_pix),
                                                                             indices_mesh=(i_grn_trials, i_ang_trials, i_hkl_trials, i_det_trials),
                                                                             params_grain=(sample_trials.U, sample_trials.x0, sample_trials.v),
                                                                             params_experiment=(sample_trials.Gamma, sample_trials.dn, sample_trials.d0, sample_trials.dr, sample_trials.dl, sample_trials.dh, sample_trials.lam, sig_s),
                                                                             opt_fun=opt_fun, 
                                                                             n_iter=n_iter,
                                                                             control_params=control_params)

    # process output
    a_fit = Rotation.from_matrix(a_fit.numpy()).as_mrp()
    time_elapsed = time.time() - time_start
    LOGGER.info(f'finished fitting {n_trials} trials')
    LOGGER.info(f'time taken {time_elapsed:2.2f} sec, time per trial {time_elapsed/len(pos_grid):2.6f} sec')

    # store output
    dict_out = dict(x_tru=base_x,
                    x_est=x_fit,
                    a_tru=base_a,
                    a_est=a_fit,
                    loss=loss,
                    nspot=nspot,
                    fracin=fracin,
                    rot_grid=rot_grid,
                    pos_grid=pos_grid,
                    time_elapsed=time_elapsed)

    # for zero-index, also store additional variables
    if index==0:
        dict_out.update(dict(s_lab_noisy=s_obs,
                             s_lab=s_obs,
                             i_det_obs=i_det_obs,
                             i_ang_obs=i_ang_obs,
                             nn_lookup_ind=nn_lookup_ind,
                             lookup_pixel_size=lookup_pixel_size,
                             lookup_n_pix=lookup_n_pix,
                             sig_s=sig_s))

    # output
    return dict_out


@main.command()
@common_options
@click.argument("tasks", nargs=-1)
# TODO: create a CLI parameter for test, dict_merged if necessary or remove
def multigrain(conf, output_dir, verbosity, n_grid, calibrate_coniga, calibrate_fenimn,tasks,test=False,dict_merged = None): 
    """Perform multi-grain fitting with Optimal Transport. 
    This function performs the following steps:

        1. collect and merges results from parallelized single-grain fitting runs
        2. perform prototype selection using OT framework
        3. perform multi-grain fitting with OT

    The optput is a file .h5 with datasets.


    Parameters
    ----------
    conf : str | Path
        Path to the configuration file
    output_dir : str
        Path to the output directory where the single-grain fitting results are stored
    verbosity : bool
        Verbosity level for printing to console
    n_grid : int
        _description_
    tasks : list
        _description_
    """


    from laueotx import laue_coordinate_descent
    from laueotx.utils import inversion as utils_inversion
    from laueotx.polycrystalline_sample import polycrystalline_sample, get_batch_indices_full, merge_duplicated_spots, apply_selections
    from laueotx.laue_rotation import get_rotation_constraint_rf
    from laueotx.spot_neighbor_lookup import spot_neighbor_lookup, nn_lookup
    from laueotx import spot_prototype_selection as prototype_selection
    from laueotx.laue_rotation import Rotation

    indices = [int(task) for task in tasks]

    print(indices)

    args = {} # TODO: implement additional arguments from commandline using click
    args = argparse.Namespace(**args)

    # args = setup(args)
    conf = utils_io.read_config(conf, args)
    sig_s = float(conf['noise_sigma'])
    opt_fun = conf['solver']['method_ot']
    opt_fun_single = conf['solver']['method_single']


    base_a, base_x, max_grain_pos, max_grain_rot = get_experimental_sample_params(conf)
    n_grains_obs = len(base_a)

    fname_merged = get_filename_realdata_merged(output_dir, tag=conf['tag'])
    if dict_merged is None:
    
        dict_merged = {}
                    
        LOGGER.info(f'=================> merging {len(indices)} parts')
        for index in LOGGER.progressbar(indices, desc='merging..', at_level='info'):
        
            filename_out = get_filename_realdata_part(output_dir, tag=conf['tag'], index=index)
            LOGGER.debug(f'reading {index+1}/{len(indices)} {filename_out}')

            if not os.path.isfile(filename_out):

                LOGGER.warning(f'failed to read {index+1}/{len(indices)} {filename_out}, no file')

            else:

                dict_single = utils_io.read_arrays(filename_out)
                for key in dict_single.keys():
                    dict_merged.setdefault(key, [])
                    dict_merged[key].append(np.array(dict_single[key]))

        for key in dict_merged.keys():
            dict_merged[key] = np.concatenate(dict_merged[key], axis=0)

    keys_store =  ['a_tru', 'x_tru', 'i_grn_tru', 'i_ang_tru', 'i_det_tru', 'i_hkl_tru', 'i_all_tru', 's_lab_noisy', 'i_det_obs', 'i_ang_obs']
    # keys_store =  ['a_tru', 'x_tru', 'i_grn_tru', 'i_ang_tru', 'i_det_tru', 'i_hkl_tru', 'i_all_tru', 's_lab_noisy']
    dict_store = {k:dict_merged[k] for k in keys_store}
    utils_io.write_arrays(fname_merged, open_mode='w', **dict_store)

    a_est = np.array(dict_merged['a_est'])
    x_est = np.array(dict_merged['x_est'])
    chi2_red = np.array(dict_merged['loss'])[:,-1]
    nspot = np.array(dict_merged['nspot'])
    fracin = np.array(dict_merged['fracin'])[:,-1]
    n_trials = len(chi2_red)
    time_per_trial = np.sum(np.array(dict_merged['time_elapsed']))/n_trials
    LOGGER.info(f'checking {n_trials} grains, time_per_trial={time_per_trial:8.4f}, median_chi2={np.median(chi2_red):2.4f}, median_fracin={np.median(fracin):2.4f}')

    # mark bad solutions
    chi2_red = np.where(chi2_red==0, 1e6, chi2_red)
    chi2_red, fracin, a_est, x_est = prototype_selection.remove_candidates_with_bad_fits(chi2_red=chi2_red, 
                                                                                         frac_inliers=fracin, 
                                                                                         apply_to=(chi2_red, fracin, a_est, x_est), 
                                                                                         max_chi2_red=conf['prototype_selection']['max_chi2_red'], 
                                                                                         min_frac_inliers=max(0, 1-conf['prototype_selection']['max_frac_outliers']))


    chi2_red, fracin, a_est, x_est = prototype_selection.prune_similar_solutions(a=a_est, 
                                                                                 x=x_est, 
                                                                                 apply_to=(chi2_red, fracin, a_est, x_est),
                                                                                 precision_rot=2e-4,
                                                                                 precision_pos=2e-3)

    # limit the number of candidates
    sorting = np.argsort(fracin)[::-1]
    chi2_red = chi2_red[sorting]
    a_est = a_est[sorting]
    x_est = x_est[sorting]
    n_candidates = len(chi2_red)
    n_max_grains_select=min(n_candidates, conf['n_grains_max'])
    LOGGER.info(f'using up to {n_max_grains_select} prototype grains')

    # get the target spots and the lookup
    s_target = tf.constant(dict_merged['s_lab_noisy'])
    i_grn_target = tf.constant(dict_merged['i_grn_tru'])
    nn_lookup_ind = tf.constant(dict_merged['nn_lookup_ind'], dtype=tf.int32)
    lookup_pixel_size = np.array(dict_merged['lookup_pixel_size'])[...]
    lookup_n_pix = np.array(dict_merged['lookup_n_pix'])[...]

    # remove duplicated candidates
    a_accept, x_accept, l_accept, grain_accept, spot_obs_assign, spot_mod_assign, spot_loss = assignments.prototype_selection_spot(conf=deepcopy(conf), 
                                                                                                                                   s_target=s_target, 
                                                                                                                                   a_est=a_est, 
                                                                                                                                   x_est=x_est, 
                                                                                                                                   l_est=chi2_red,
                                                                                                                                   lookup_data=(nn_lookup_ind, lookup_pixel_size, lookup_n_pix), 
                                                                                                                                   noise_sig=sig_s,
                                                                                                                                   n_max_grains_select=n_max_grains_select)
    LOGGER.info('finished init optimization')
    n_accept = len(a_accept)
    for i in range(n_grains_obs):
        print(f'previously found grains {i+1:>4d}/{n_grains_obs}   a={astr(base_a[i])}   x={astr(base_x[i])}')
    for i in range(n_accept):
        print(f'found model grain {i+1:>4d}/{n_accept}   a={astr(a_accept[i])}   x={astr(x_accept[i])}')


    # get the sample

    LOGGER.info(f'getting spots for {n_accept} best candidate grains')
    sample = polycrystalline_sample(conf=deepcopy(conf), hkl_sign_redundancy=True, rotation_type='mrp', restrict_fitting_area=False)
    sample.set_tensor_variables(a_accept, x_accept) 
    i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=n_accept)
    s_sample, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='laboratory', n_per_batch=n_accept)

    s_accept, p_lam, i_grn_accept, i_ang_accept, i_hkl_accept, i_det_accept, i_all_accept = apply_selections(select_sample, s_sample, p_lam, i_grn, i_ang, i_hkl, i_det, i_all)
    s_accept, i_grn_accept, i_ang_accept, i_hkl_accept, i_det_accept, i_all_accept, select_merged = merge_duplicated_spots(s_accept, i_grn_accept, i_ang_accept, i_hkl_accept, i_det_accept, i_all_accept, decimals=4, return_index=True, split_by_grain=True)
    p_lam = tf.gather(p_lam, select_merged)
    spotind_nn = nn_lookup(nn_lookup_ind, s_target, s_accept, i_ang_accept, i_det_accept, i_grn_accept, lookup_pixel_size, lookup_n_pix)

    root_key = 'assignments/prototypes'
    dict_out = {f'{root_key}/a': a_accept,
                f'{root_key}/x': x_accept,
                f'{root_key}/spot_obs_assign': spot_obs_assign,                  
                f'{root_key}/spot_mod_assign': spot_mod_assign,                 
                f'{root_key}/i_grn': i_grn_accept,             
                f'{root_key}/i_ang': i_ang_accept,                    
                f'{root_key}/i_hkl': i_hkl_accept,                    
                f'{root_key}/i_det': i_det_accept,                    
                f'{root_key}/i_all': i_all_accept,                    
                f'{root_key}/loss': l_accept,
                f'{root_key}/s': s_accept,
                f'{root_key}/p_lam': p_lam,
                f'{root_key}/spotind_nn': spotind_nn,
                f'{root_key}/grain_accept': grain_accept,
                f'{root_key}/s_accept': s_accept,
                'v': np.array(sample.v),
                'spot_loss': spot_loss}
    utils_io.write_arrays(fname_merged, open_mode='a', **dict_out)
    LOGGER.info(f'stored prototype selection data in file {fname_merged}')

    # run sinkhorn
    # spotind_nn_accept = nn_lookup(nn_lookup_ind, s_target, s_accept, i_ang_accept, i_det_accept, i_grn_accept, lookup_pixel_size, lookup_n_pix)
    # dict_out_sinkhorn = assignments.assign_refine_sinkhorn(args, a_accept, x_accept, s_target, nn_lookup_data=(nn_lookup_ind, lookup_pixel_size, lookup_n_pix), noise_sig=sig_s, annealing_range=100.)
    dict_out_sinkhorn = assignments.assign_refine_sinkhorn(conf, a_accept, x_accept, s_target, 
                                                           nn_lookup_data=(nn_lookup_ind, lookup_pixel_size, lookup_n_pix), 
                                                           noise_sig=sig_s, 
                                                           method=opt_fun,
                                                           test=test)


    i_grn_global = dict_out_sinkhorn['inds_model'][0]
    ind_max = np.argmax(dict_out_sinkhorn['q_fit'], axis=1)
    s_global_2d = np.array(dict_out_sinkhorn['s_global_2d'])
    s_global_2d = np.array([s[i,:] for i, s in zip(ind_max, s_global_2d)])
    l2_fit = np.array(dict_out_sinkhorn['l2_fit'])

    # get sparse matrix for assignment
    Q_sparse = assignments.array_neighbours_to_sparse(dict_out_sinkhorn['q_fit'], spot_ind=dict_out_sinkhorn['i_target'], n_target=len(s_target))
    l2_sparse = assignments.array_neighbours_to_sparse(dict_out_sinkhorn['l2_fit'], spot_ind=dict_out_sinkhorn['i_target'], n_target=len(s_target))
    spot_obs_assign, spot_mod_assign, s2s_obs_assign, s2s_mod_assign = assignments.get_softassign_spot_matching(Q_sparse, np.array(i_grn_global).ravel(), l2=l2_sparse, noise_sig=sig_s)
            
    a_global = np.array(dict_out_sinkhorn['a_fit_global'])
    x_global = np.array(dict_out_sinkhorn['x_fit_global'])
    n_accept = len(a_global)

    root_key = 'assignments/global'
    dict_out = {f'{root_key}/a': a_global,
                f'{root_key}/x': x_global,
                f'{root_key}/spot_obs_assign': spot_obs_assign,                  
                f'{root_key}/spot_mod_assign': spot_mod_assign,                 
                f'{root_key}/s2s_obs_assign': s2s_obs_assign, 
                f'{root_key}/s2s_mod_assign': s2s_mod_assign,
                f'{root_key}/i_grn': i_grn_accept,             
                f'{root_key}/i_ang': i_ang_accept,                    
                f'{root_key}/i_hkl': i_hkl_accept,                    
                f'{root_key}/i_det': i_det_accept,                    
                f'{root_key}/i_all': i_all_accept,                    
                f'{root_key}/i_target': dict_out_sinkhorn['i_target'],                    
                f'{root_key}/loss': dict_out_sinkhorn['loss_global'],
                f'{root_key}/q_sparse': dict_out_sinkhorn['q_fit'],
                f'{root_key}/l2_fit': l2_fit,
                f'{root_key}/s': s_global_2d,
                f'{root_key}/s_global': np.array(dict_out_sinkhorn['s_global']),
                f'{root_key}/p_lam': p_lam}

    dict_out.update(sparse_matrix_to_dict(data=Q_sparse, prefix=f'{root_key}/Qs'))
    utils_io.write_arrays(fname_merged, open_mode='a', **dict_out)
    LOGGER.info(f'stored global assignmnet data in file {fname_merged}')

    print('Model sample in the MRP units:')
    for i in range(n_accept):
        print(f'found model grain {i+1:>4d}/{n_accept}   a={astr(a_global[i])}   x={astr(x_global[i])}')

    R_global = Rotation.from_mrp(a_global)
    a_rf = Rotation.as_rodrigues_frank(R_global)

    print('Model sample in the Rodrigues-Frank units:')
    for i in range(n_accept):
        print(f'found model grain {i+1:>4d}/{n_accept}   a={astr(a_rf[i])}   x={astr(x_global[i])}')

    def print_nspots_per_det(i_det, assign, tag=''):
        print('spots={} total      n_spots={:>6d} (assigned {})'.format(tag, len(i_det), np.count_nonzero((assign>=-1))))
        for i in [0,1]:
            print('spots={} detector={} n_spots={:>6d} (assigned {})'.format(tag, i, np.count_nonzero(i_det == i), np.count_nonzero((i_det == i) & (assign>=-1))))

    print('Spot assignment stats:')
    print_nspots_per_det(dict_store['i_det_obs'].ravel(), spot_obs_assign, tag='detected ')
    print_nspots_per_det(i_det_accept.numpy().ravel(), spot_mod_assign, tag='model    ')

    # print('Candidate grains:')
    # print_grain_candidates(a_est, chi2_red, fracin)
    # import pudb; pudb.set_trace();
    # pass


def print_grain_candidates(a_est, chi2_red, fracin):

    sorting = np.argsort(a_est[:,0])
    for i, (a, c, f) in enumerate(zip(a_est[sorting], chi2_red[sorting], fracin[sorting])): 
        print(f"{i:>5d}", astr(a), astr(c), astr(f))

def sparse_matrix_to_dict(data, prefix):

    out =   {f'{prefix}/data': data.data,
             f'{prefix}/row': data.row,
             f'{prefix}/col': data.col,
             f'{prefix}/shape': data.shape}

    return out

def get_detectors_from_config(conf):

    detectors = []
    for d in conf['detectors']:
        det = Detector(detector_type=d['type'],
                       detector_id=d['id'],
                       side_length=d['side_length'],
                       num_pixels=d['num_pixels'],
                       position=d['position'],
                       tilt=d['tilt'],
                       tol_position=d['position_err'],
                       tol_tilt=d['tilt_err'])
        detectors.append(det)
    
    return detectors

def get_peaskdata_from_config(conf, beamline):

    from laueotx.peaks_data import PeaksData, MultidetectorPeaksData

    omegas = utils_config.get_angles(conf['angles'])

    list_pd = []
    for peaks in conf['data']['peaks']:

        fpath = peaks['filepath'] if os.path.isabs(peaks['filepath']) else os.path.join(conf['data']['path_root'], peaks['filepath'])

        pd = PeaksData(detector_type=peaks['detector'],
                        beamline=beamline,
                        filepath=fpath,
                        apply_coord_shift=peaks['apply_coord_shift'],
                        flip_x_coord=peaks['flip_x_coord'],
                        omegas_use=omegas)
        list_pd.append(pd)

    mpd = MultidetectorPeaksData(list_pd, beamline=beamline)

    return mpd



def get_indices(tasks):
    """
    Parses the jobids from the tasks string.

    :param tasks: The task string, which will get parsed into the job indices
    :return: A list of the jobids that should be executed
    """
    # parsing a list of indices from the tasks argument

    if '>' in tasks:
        tasks = tasks.split('>')
        start = tasks[0].replace(' ', '')
        stop = tasks[1].replace(' ', '')
        indices = list(range(int(start), int(stop)))
    elif ',' in tasks:
        indices = tasks.split(',')
        indices = list(map(int, indices))
    else:
        try:
            indices = [int(tasks)]
        except ValueError:
            raise ValueError("Tasks argument is not in the correct format!")

    return indices




def get_experimental_sample_params(conf):

    if 'ruby' in conf['tag']:
        # ruby input
        base_x = np.atleast_2d(np.array([-0.2449,    0.2495,   -0.7436]))
        base_a = Rotation.from_gibbs(np.array([0.1376,    0.1535,    0.2704])).as_mrp()
        max_grain_pos=6./2.  # 6 mm
        # max_grain_rot=0.18568445
        max_grain_rot = 0.414 # Rodriguez-Frank (not mrp)
        sig_s = 0.8
        n_grains_obs = len(base_a)
        n_trials = 10000

    elif 'coniga' in conf['tag']:

        # from experimental_data_230227/coniga_oligocrystal/20200503_084102_CoNiGa_ExtrHT.mat "all"
        base_a = [[0.249974984983611,   0.387394791655793,   0.292030922315102],
                  [-0.349467852241861,  0.147359359261315,   0.376151811405052],
                  [0.298932214021211,   -0.126119804303904,  0.0669088133675702],
                  [0.0109076926773256,  0.244302853225442,   -0.320339764054170],
                  [-0.260110009777407,  0.173117900718748,   -0.308602425250214],
                  [0.362971092596224,   0.181554790458536,   -0.243766664693416],
                  [0.202541719150792,   0.406892084264865,   0.0643182265858323],
                  [-0.0319159189235867, -0.250187025571050,  -0.0737122635780873],
                  [-0.349098742839504,  0.144580556514080,   0.194355760579785],
                  [-0.0466625863598419, -0.384395314029808,  0.186856449451044],
                  [0.153908228116895,   0.274319259806297,   -0.0449973584144624],
                  [-0.159554489291583,  0.112032709868792,   0.183053253385828],
                  [-0.311620828939660,  0.233264966884310,   0.0455811962482679],
                  [0.181097726348299,   -0.0391779656074941, -0.175117051982365],
                  [0.126697735967340,   -0.138699999381848,  -0.0912802589424690],
                  [0.0384948670395929,  0.342260578406194,   0.131282408364991],
                  [0.349344356000426,   0.330779821558213,   0.0393629984514759],
                  [-0.148333812488533,  -0.285965558400905,  -0.0285525293389816],
                  [-0.156940698675776,  0.0437648980274408,  0.0304640983640349],
                  [0.114939844845496,   -0.370005942855764,  0.102303706355135],
                  [0.251080731579256,   0.387177835416426,   0.296089074361782]]


        base_x = [[0.0701807512073583, 2.35881874526016,    0.492694766053906], 
                   [1.07586717959680,    1.44136541112850,    -2.50792490092970], 
                   [1.20490432473608,    0.0685842031900762,  -4.59341543373832], 
                   [-0.932731523730208,  0.642300146314609,   6.58754100771537], 
                   [1.15560691244364,    1.88739774545844,    -5.17429063461656], 
                   [1.62732490347044,    0.394636324269765,   0.875171932438993], 
                   [2.02644989031317,    0.453162778808494,   3.59062953214833], 
                   [0.858109583361683,   -0.418281188449186,  -2.94730435202928], 
                   [1.95212806058845,    1.65026524647153,    4.75115250045785], 
                   [0.0647505435313007,  2.24003377763175,    6.33192239497657], 
                   [0.292573538709847,   0.0908528828303694,  4.93760161729973], 
                   [0.308485140561124,   1.95975203921147,    4.17832152787860], 
                   [-0.686963024111653,  1.26920530964844,    -5.47519919800038], 
                   [-1.11433647511425,   1.33813689614714,    3.51029165642460], 
                   [0.575865669542714,   0.464436342866357,   -6.79929414983671], 
                   [1.94804295807138,    0.758228493983753,   5.83027699776625], 
                   [-0.833953163914242,  0.116199741379371,   -3.38653475892078], 
                   [0.286908931432729,   0.138333159011729,   2.04644748393284], 
                   [0.214803273699547,   -0.114365173980439,  -1.68261690526988], 
                   [2.26930358610734,    0.197925897634415,   5.69965963906363], 
                   [0.350158624762042,   1.00808498728128,    0.613719477310155]] 

        base_x = np.array(base_x)
        base_a = Rotation.from_gibbs(base_a).as_mrp()
        max_grain_pos = np.array([6./2., 6./2., 14./2.])  # mm
        # max_grain_rot=0.18568445
        max_grain_rot = 0.414 # Rodriguez-Frank (not mrp)
        sig_s = 8.
        n_grains_obs = len(base_a)

    elif 'fenimn' in conf['tag']:

        base_x = np.array([[0,0,0]])
        base_a = Rotation.from_gibbs([[0,0,0]]).as_mrp()
        max_grain_pos = np.array([6./2., 6./2., 6/2.])  # mm
        # max_grain_rot=0.18568445
        max_grain_rot = 0.414 # Rodriguez-Frank (not mrp)
        sig_s = 0.4
        n_grains_obs = len(base_a)

    elif 'cng' in conf['tag']:

        base_x = np.array([[0,0,0]])
        base_a = Rotation.from_gibbs([[0,0,0]]).as_mrp()
        max_grain_pos = np.array([7./2., 4./2., 4/2.])  # mm
        # max_grain_rot=0.18568445
        max_grain_rot = 0.414 # Rodriguez-Frank (not mrp)
        n_grains_obs = len(base_a)

    elif 'fega' in conf['tag']:

        base_x = np.array([[0,0,0]])
        base_a = Rotation.from_gibbs([[0,0,0]]).as_mrp()
        max_grain_pos = np.array([5./2., 5./2., 5/2.])  # mm
        # max_grain_rot=0.18568445
        max_grain_rot = 0.414 # Rodriguez-Frank (not mrp)
        n_grains_obs = len(base_a)

    return base_a, base_x, max_grain_pos, max_grain_rot


# def fenimn_calibration(indices, output_dir, conf):

#     LOGGER.critical('running fenimn sample calibration')
    
#     grid_shift = np.linspace(-10,10,11)
#     n_det = 2
#     dir_out_root = output_dir

#     for index in indices:
#         conf = conf.copy()
#         # conf = utils_io.read_config(args)

#         # get detector shift

#         id_det = index % n_det
#         id_shift = index // n_det
#         id_det_remove = 0 if id_det==1 else 1

#         del(conf['detectors'][id_det_remove])
#         del(conf['data']['peaks'][id_det_remove])

#         conf['detectors'][0]['position'][0] += grid_shift[id_shift]

#         LOGGER.info(f'=================> running on index {index} / {str(len(indices))} id_det={id_det} id_shift={id_shift}')
#         LOGGER.info(f"shifted detector {id_det} {conf['detectors'][0]['position']}")

#         # get measurements

#         omegas = utils_config.get_angles(conf['angles'])
#         LOGGER.info(f'got {len(omegas)} angles')

#         dt = get_detectors_from_config(conf)

#         bl = Beamline(omegas=omegas, 
#                       detectors=dt,
#                       lambda_lim=[conf['beam']['lambda_min'], conf['beam']['lambda_max']])

#         mpd = get_peaskdata_from_config(conf, bl)

#         # get output dir

#         current_output_dir = os.path.join(dir_out_root, f"det{id_det}_shift{id_shift:02d}")
#         utils_io.robust_makedirs(current_output_dir)

#         # main magic

#         dict_out = analyse(n_grid, conf, mpd, index=0, test=False)
#         multigrain(indices=[0], args=args, dict_merged=dict_out)



# def coniga_calibration(indices, n_grid, conf):

#     from laueotx import laue_coordinate_descent, utils_inversion
#     from laueotx.polycrystalline_sample import polycrystalline_sample, get_batch_indices_full, select_indices
#     from laueotx.spot_neighbor_lookup import spot_neighbor_lookup, nn_lookup, nn_distance

#     # conf = utils_io.read_config(args)
#     conf = conf.copy()

#     omegas = utils_config.get_angles(conf['angles'])
#     LOGGER.info(f'got {len(omegas)} angles')

#     dt = get_detectors_from_config(conf)

#     bl = Beamline(omegas=omegas, 
#                   detectors=dt,
#                   lambda_lim=[conf['beam']['lambda_min'], conf['beam']['lambda_max']])


#     mpd = get_peaskdata_from_config(conf, bl)

#     s_obs = np.array(mpd.to_lab_coords())
#     s_obs[:,0] = 0
#     i_det_obs = np.array(mpd.peaks['id_detector']).reshape(-1,1)
#     i_ang_obs = np.array(mpd.peaks['id_angle']).reshape(-1,1)

#     base_a_all, base_x_all, max_grain_pos, max_grain_rot = get_experimental_sample_params(conf)

#     # get lookup

#     snl = spot_neighbor_lookup(conf=conf, 
#                                s_spot=s_obs, 
#                                i_ang=i_ang_obs, 
#                                i_det=i_det_obs, 
#                                lookup_n_pix=int(conf['solver']['lookup_n_pix']), # pix, 
#                                detector_side_length=conf['detectors'][0]['side_length'], 
#                                n_neighbours=int(conf['solver']['n_neighbours']))
#     nn_lookup_ind = tf.constant(snl.nn_lookup_ind.numpy(), dtype=tf.int32)
#     lookup_pixel_size=snl.detector_side_length/snl.lookup_n_pix
#     lookup_n_pix=snl.lookup_n_pix


#     for gi in indices:

#         LOGGER.info(f'=================== grain {gi}')

#         base_a = base_a_all[[gi]]
#         base_x = base_x_all[[gi]]

#         n_grains_obs = len(base_a)

#         # get search grid

#         max_rms_inlier = 20
#         n_trials = int(n_grid)
#         max_detector_delta_rot = 0.01
#         max_detector_delta_pos = 10
#         max_grain_delta_rot = 0.01 
#         max_grain_delta_pos = 3
#         sobol_samples = tf.math.sobol_sample(dim=18, num_results=n_trials, skip=0, dtype=tf.float64).numpy()*2 -1 # numbers between -1 and 1
#         dr_grid_det0 = sobol_samples[:,0:3]  * max_detector_delta_rot
#         dr_grid_det1 = sobol_samples[:,3:6]  * max_detector_delta_rot
#         dx_grid_det0 = sobol_samples[:,6:9]  * max_detector_delta_pos
#         dx_grid_det1 = sobol_samples[:,9:12] * max_detector_delta_pos
#         dr_grid = np.concatenate([np.expand_dims(dr_grid_det0, axis=1),np.expand_dims(dr_grid_det1, axis=1)], axis=1)
#         dx_grid = np.concatenate([np.expand_dims(dx_grid_det0, axis=1),np.expand_dims(dx_grid_det1, axis=1)], axis=1)
#         dr_grid[0,...] = 0
#         dx_grid[0,...] = 0
#         dx_grid[:,0,0] -= 160
#         dx_grid[:,1,0] += 160

#         gr_grid = sobol_samples[:,12:15] * max_grain_delta_rot
#         gx_grid = sobol_samples[:,15:18] * max_grain_delta_pos
#         gr_grid = base_a + gr_grid
#         gx_grid = base_x + gx_grid
#         gr_grid[0] = 0
#         gx_grid[0] = 0

#         loss = np.ones([n_trials])*1e6

#         n_trials_per_batch = 1
#         n_batches = n_trials//n_trials_per_batch
#         sample = polycrystalline_sample(conf, hkl_sign_redundancy=True, rotation_type='mrp', restrict_fitting_area=True)
#         sample.set_tensor_variables(rot=gr_grid[:n_trials_per_batch], pos=gx_grid[:n_trials_per_batch], dr=dr_grid[:n_trials_per_batch], d0=dx_grid[:n_trials_per_batch]) # this is just so the next line runs
#         i_grn, i_ang, i_hkl, i_det, i_all = sample.get_batch_indices_full(n_grn=n_trials_per_batch)
#         for j in LOGGER.progressbar(range(n_batches), at_level='info'):

#             si, ei = j*n_trials_per_batch, (j+1)*n_trials_per_batch

#             sample.set_tensor_variables(rot=gr_grid[si:ei], pos=gx_grid[si:ei], dr=dr_grid[si:ei], d0=dx_grid[si:ei])
#             s_lab, p_sample, p_lam, select_sample = sample.get_spots_batch(reference_frame='detector', n_per_batch=n_trials_per_batch)

#             s_sample, i_grn_sample, i_ang_sample, i_hkl_sample, i_det_sample, i_all_sample = select_indices(select_sample, s_lab, i_grn, i_ang, i_hkl, i_det, i_all)
#             i_nn = nn_lookup(nn_lookup_ind, s_obs, s_sample, i_ang_sample, i_det_sample, i_grn_sample, lookup_pixel_size, lookup_n_pix)
#             s_target = tf.gather(s_obs, i_nn)

#             # remove stuff that went outside the screen

#             i_ang_target = tf.gather(i_ang_obs, i_nn)
#             select_ok = np.array(i_ang_target)==np.array(i_ang_sample)

#             s_sample, i_grn_sample, i_ang_sample, i_hkl_sample, i_det_sample, i_all_sample = select_indices(select_ok[:,0], s_sample, i_grn_sample, i_ang_sample, i_hkl_sample, i_det_sample, i_all_sample)
#             i_nn = nn_lookup(nn_lookup_ind, s_obs, s_sample, i_ang_sample, i_det_sample, i_grn_sample, lookup_pixel_size, lookup_n_pix)
#             s_target = tf.gather(s_obs, i_nn)
#             l2 = np.sqrt(tf.reduce_sum((s_sample-s_target)**2, axis=1))

#             # assert np.all(select_ok), f'bound errors in lookup {np.count_nonzero(select_ok)}/{len(select_ok)}'

#             select0 = i_det_sample==0
#             select1 = i_det_sample==1

#             from scipy.stats import sigmaclip
#             clipped, clip_lower, clip_upper0 = sigmaclip(l2[select0[:,0]], low=100, high=2)
#             clipped, clip_lower, clip_upper1 = sigmaclip(l2[select1[:,0]], low=100, high=2)
#             select_inliers = ((l2<clip_upper0) & select0[:,0]) | ((l2<clip_upper1) & select1[:,0])

#             l2_seg = tf.math.segment_mean(l2[select_inliers], np.array(i_grn_sample)[select_inliers,0])
#             loss[si:ei] = l2_seg
            
#             LOGGER.debug(f'batch {j:>6d}/{n_batches} best_loss={astr3(np.min(l2_seg)):>10s} n_ok={astr3(np.count_nonzero(select_ok)/len(select_ok)):>6s} selected inliers {astr3(np.count_nonzero(select_inliers)/len(select_inliers)):>6s} clip_upper_l2={astr3([clip_upper0, clip_upper1])}')



#         best_id = np.argmin(loss)
#         LOGGER.critical(f'===============> grain {gi} {loss[0]} -> {loss[best_id]}')
#         LOGGER.critical(f'gx{i}={astr(gx_grid[best_id])} {astr(base_x)}')
#         LOGGER.critical(f'gr{i}={astr(gr_grid[best_id])} {astr(base_a)}')
        
#         for i in range(n_detectors):
#             LOGGER.critical(f'calibration detector {i}:')
#             LOGGER.critical(f'dx{i}={astr(dx_grid[best_id,i])}')
#             LOGGER.critical(f'dr{i}={astr(dr_grid[best_id,i])}')

#         utils_io.write_arrays(f'coniga_calibr_{gi}.h5', 'w', gx_grid=gx_grid, gr_grid=gr_grid, dx_grid=dx_grid, dr_grid=dr_grid, loss=loss)







# import functools
# def common_options(f):
#     @click.option('--conf', "-c", required=True, type=click.File(), show_default=True,help='Configuration yaml file')
#     @click.option('output_dir','--output-dir', "-o", required=True, type=click.Path(), show_default=True,help='Directory to store the produced files')
#     @click.option('--n-grid', default=2000, show_default=True,help='number of grid points from which to initialize coordinate descent')
#     @click.option("--calibrate-coniga/--no-calibrate-coninga",default=False, help="Calibrate the coniga sample")
#     @click.option("--calibrate-fenimn/--no-calibrate-fenimn", default=False, help="Calibrate the fenimn sample")
#     @functools.wraps(f)
#     def wrapper_common_options(*args, **kwargs):
#         return f(*args, **kwargs)

#     return wrapper_common_options

# if __name__ == '__main__':
#     pass






# def resources(args):

#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--queue', type=str, default='gwen_short', choices=('gwen_short', 'gwen', 'gpu'))
#     argk, _ = parser.parse_known_args(args)
    
#     res = dict(main_memory=4000,
#                main_time_per_index=1, # hours
#                main_nproc=4,
#                main_scratch=6500,
#                main_ngpu=1,
#                merge_ngpu=2,
#                merge_time=1) # hours

#     if argk.queue == 'gwen_short':

#         res['main_time_per_index']=2 # hours
#         res['main_ngpu']=1
#         res['main_nproc']=16*res['main_ngpu']
#         res['main_memory']=3900*res['main_nproc']
#         res['pass'] = {'cluster':'gmerlin6', 'account':'gwendolen', 'partition':'gwendolen', 'gpus-per-task':res['main_ngpu'], 'cpus-per-gpu':16}
#         res['main_nsimult'] = 5
#         res['merge_ngpu']=1
#         res['merge_nproc']=16*res['main_ngpu']
#         res['merge_memory']=3900*res['main_nproc']

#     elif argk.queue == 'gwen':

#         res['main_time_per_index']=8 # hours
#         res['main_ngpu']=4
#         res['main_nproc']=16*res['main_ngpu']
#         res['main_memory']=3900*res['main_nproc']
#         res['pass'] = {'cluster':'gmerlin6', 'account':'gwendolen', 'partition':'gwendolen-long', 'gpus-per-task':res['main_ngpu'], 'cpus-per-gpu':16}

#     elif argk.queue == 'gpu':

#         res['main_time_per_index']=24 # hours
#         res['main_ngpu']=1
#         res['main_nproc']=4
#         res['main_memory']=4000*res['main_nproc']
#         res['pass'] = {'cluster':'gmerlin6', 'partition':'gpu', 'gpus-per-task':res['main_ngpu'], 'cpus-per-gpu':4}


#     return res


# def setup(args):

#     if type(args) is not list: 
#         return args

#     description = 'Run compression benchmarks for climate simulations'
#     parser = argparse.ArgumentParser(description=description, add_help=True)
#     parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
#                         help='logging level')
#     parser.add_argument('--conf', type=str, required=True, 
#                         help='configuration yaml file')
#     parser.add_argument('--dir_out', type=str, required=True, 
#                         help='output dir')
#     parser.add_argument('--params_ot', type=str, default=None,
#                         help='parameters for the OT, string will be parsed, format: --params_ot=key1=value1,key2=value2')
#     parser.add_argument('--test', action='store_true',
#                         help='test mode')
#     parser.add_argument('--n_grid', type=int, default=2000,
#                         help='number of grid points from which to initialize coordinate descent')
#     parser.add_argument('--calibrate_coniga', action='store_true',
#                         help='if to calibrate the coniga sample')
#     parser.add_argument('--calibrate_fenimn', action='store_true',
#                         help='if to calibrate the fenimn sample')

#     args, _ = parser.parse_known_args(args)
#     utils_logging.set_all_loggers_level(args.verbosity)
    
#     args.conf = utils_io.get_abs_path(args.conf)
#     args.dir_out = utils_io.get_abs_path(args.dir_out)
#     utils_io.robust_makedirs(args.dir_out)

#     return args

