from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np

from lsstools import parameters
from lsstools import parameters_ms_gadget
from lsstools.model_spec import *
import model_error
from nbodykit import CurrentMPIComm, logging, setup_logging


def main():
    """
    Combine source fields to get proxy of a target field. This is stage 0 of 
    reconstruction, and can be used to quantify Perror of a bias model.

    For example:

      - Combine delta_m, delta_m^2, s^2 to get proxy of target=delta_halo.

      - Combine different mass-weighted delta_h fields to get proxy of
        target=delta_m.

    Usage examples:
      ./run.sh python main_calc_Perr_test.py
    or 
      ./run.sh mpiexec -n 4 python main_calc_Perr_test.py --SimSeed 403
    """

    #####################################
    # PARSE COMMAND LINE ARGS
    #####################################
    ap = ArgumentParser()

    ap.add_argument('--SimSeed',
                    type=int,
                    default=403,
                    help='Simulation seed to load.')

    ap.add_argument('--HaloMassString',
                    default='13.8_15.1',
                    help="Halo mass string, for example '13.8_15.1'.")

    cmd_args = ap.parse_args()

    #####################################
    # OPTIONS
    #####################################
    opts = OrderedDict()

    # Bump this when changing code without changing options. Otherwise pickle
    # loading might wrongly read old pickles.
    opts['main_calc_Perr_test_version'] = '1.5'

    # Simulation options. Will be used by path_utils to get input path, and
    # to compute deltalin at the right redshift.
    seed = cmd_args.SimSeed
    opts['sim_opts'] = parameters_ms_gadget.MSGadgetSimOpts.load_default_opts(
        sim_name='ms_gadget_test_data',
        sim_seed=seed,
        halo_mass_string=cmd_args.HaloMassString)

    # Grid options.
    Ngrid = 64
    opts['grid_opts'] = parameters.GridOpts(
        Ngrid=Ngrid,
        kmax=2.0*np.pi/opts['sim_opts'].boxsize * float(Ngrid)/2.0,
        grid_ptcle2grid_deconvolution=None
        )

    # Options for measuring power spectrum. Use defaults.
    opts['power_opts'] = parameters.PowerOpts()

    # Transfer function options. See lsstools.parameters.py for details.
    opts['trf_fcn_opts'] = parameters.TrfFcnOpts(
        Rsmooth_for_quadratic_sources=0.1,
        Rsmooth_for_quadratic_sources2=0.1,
        N_ortho_iter=1,
        orth_method='CholeskyDecomp',
        interp_kind='manual_Pk_k_bins'
        )

    # External grids to load: deltalin, delta_m, shifted grids
    opts['ext_grids_to_load'] = opts['sim_opts'].get_default_ext_grids_to_load(
        Ngrid=opts['grid_opts'].Ngrid)

    # Catalogs to read
    opts['cats'] = opts['sim_opts'].get_default_catalogs()

    # Specify bias expansions to test
    opts['trf_specs'] = []

    # Quadratic Lagrangian bias: delta_Z + b1 deltalin(q+Psi) + b2 
    # [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi)
    opts['trf_specs'].append(
        TrfSpec(linear_sources=[
            'deltalin_SHIFTEDBY_deltalin',
            'deltalin_growth-mean_SHIFTEDBY_deltalin',
            'deltalin_G2_SHIFTEDBY_deltalin'
        ],
                fixed_linear_sources=['1_SHIFTEDBY_deltalin'],
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='delta_h',
                save_bestfit_field=
                'hat_delta_h_from_1_Tdeltalin2G2_SHIFTEDBY_PsiZ'))

    # Save results
    opts['keep_pickle'] = False
    opts['pickle_file_format'] = 'dill'
    opts['pickle_path'] = '$SCRATCH/perr/pickle/'

    # Save some additional power spectra that are useful for plotting later
    opts['Pkmeas_helper_columns'] = [
        'delta_h', 'delta_m', '1_SHIFTEDBY_deltalin', 'deltalin'
    ]

    # Save grids for 2d slice plots and histograms
    opts['save_grids4plots'] = False
    opts['grids4plots_base_path'] = '$SCRATCH/perr/grids4plots/'
    opts['grids4plots_R'] = 0.0  # Gaussian smoothing applied to grids4plots

    # Cache path
    opts['cache_base_path'] = '$SCRATCH/perr/cache/'


    # Run the program given the above opts.
    outdict = model_error.calculate_model_error(opts)


    # Compare vs expected result.
    residual_key = '[hat_delta_h_from_1_Tdeltalin2G2_SHIFTEDBY_PsiZ]_MINUS_[delta_h]'
    Perr = outdict['Pkmeas'][(residual_key, residual_key)].P
    Perr_expected = np.array([
        9965.6, 17175.8, 22744.4, 19472.3, 19081.2, 19503.4, 19564.9,
        18582.9, 19200.1, 16911.3, 16587.4, 16931.9, 15051.0, 13835.1,
        13683.8, 13109.9, 12353.5, 11900.2, 11085.1, 11018.4, 10154.0,
        9840.7, 8960.6, 8484.1, 7942.2, 7426.7, 6987.8, 6578.1, 6269.1,
        5810.7, 5511.7
    ])

    setup_logging()
    comm = CurrentMPIComm.get()
    logger = logging.getLogger('TestPerrCalc')

    if comm.rank == 0:
        Perr_lst = ['%.1f' % a for a in list(Perr)]
        Perr_expected_lst = ['%.1f' % a for a in list(Perr)]
        logger.info('Perr:\n%s' % str(','.join(Perr_lst)))
        logger.info('Expected Perr:\n%s' % str(','.join(Perr_expected_lst)))
        if np.allclose(Perr, Perr_expected, rtol=1e-3):
            logger.info('TEST Perr: OK')
        else:
            logger.info('TEST Perr: FAILED')
            raise Exception('Test failed')


if __name__ == '__main__':
    main()
