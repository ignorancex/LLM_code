from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np

from lsstools import parameters
from lsstools import parameters_ms_gadget
from lsstools.model_spec import *
import model_error


def main():
    """
    Combine source fields to get proxy of a target field. This is stage 0 of 
    reconstruction, and can be used to quantify Perror of a bias model.

    For example:

      - Combine delta_m, delta_m^2, s^2 to get proxy of target=delta_halo.

      - Combine different mass-weighted delta_h fields to get proxy of
        target=delta_m.

    Usage examples:
      ./run.sh python main_calc_Perr.py
    or 
      ./run.sh mpiexec -n 4 python main_calc_Perr.py --SimSeed 403
    """

    #####################################
    # PARSE COMMAND LINE ARGS
    #####################################
    ap = ArgumentParser()

    ap.add_argument('--SimSeed',
                    type=int,
                    default=400,
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
    #opts['main_calc_Perr_version'] = '0.2'
    opts['code_version_for_pickles'] = '0.3'


    # Simulation options. Will be used by path_utils to get input path, and
    # to compute deltalin at the right redshift.
    seed = cmd_args.SimSeed
    opts['sim_opts'] = parameters_ms_gadget.MSGadgetSimOpts.load_default_opts(
        sim_name='ms_gadget',
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

    # Specify bias expansions to test
    opts['trf_specs'] = []

    if False:
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

    if True:
        # Try out cubic operators
        # Cubic Lagrangian bias with delta^3, G2*delta, G3 and Gamma3: 
        # delta_Z + b1 deltalin(q+Psi) + b2 [deltalin^2-<deltalin^2>](q+Psi) + bG2 [G2](q+Psi)
        # + b3 [delta^3](q+Psi) + ...
        # With RSD, cannot orthogonalize deltalin^3 and deltalin at low k b/c too correlated?
        RSDstring = ''
        opts['trf_specs'].append(
            TrfSpec(
                linear_sources=[
                    'deltalin_SHIFTEDBY_deltalin%s' % RSDstring,
                    'deltalin_growth-mean_SHIFTEDBY_deltalin%s' % RSDstring,
                    'deltalin_G2_SHIFTEDBY_deltalin%s' % RSDstring,
                    'deltalin_cube-mean_SHIFTEDBY_deltalin%s' % RSDstring,
                    'deltalin_G2_delta_SHIFTEDBY_deltalin%s' % RSDstring,
                    'deltalin_G3_SHIFTEDBY_deltalin%s' % RSDstring,
                    'deltalin_Gamma3_SHIFTEDBY_deltalin%s' % RSDstring
                ],
                fixed_linear_sources=['1_SHIFTEDBY_deltalin%s' % RSDstring],
                field_to_smoothen_and_square=None,
                quadratic_sources=[],
                target_field='delta_h',
                save_bestfit_field=
                'hat_delta_h_from_1_Tdeltalin2G23dG2G3Gamma3_SHIFTEDBY_PsiZ'))


    # Save results
    opts['keep_pickle'] = True
    opts['pickle_file_format'] = 'dill'
    opts['pickle_path'] = '$SCRATCH/perr/pickle/'

    # Save some additional power spectra that are useful for plotting later
    opts['Pkmeas_helper_columns'] = [
        #'delta_h', 'delta_m', '1_SHIFTEDBY_deltalin', 'deltalin'
        '1_SHIFTEDBY_deltalin',
    ]

    # Save grids for 2d slice plots and histograms
    opts['save_grids4plots'] = False
    opts['grids4plots_base_path'] = '$SCRATCH/perr/grids4plots/'
    opts['grids4plots_R'] = 0.0  # Gaussian smoothing applied to grids4plots

    # Cache path
    opts['cache_base_path'] = '$SCRATCH/perr/cache/'

    # what to return
    opts['return_fields'] = ['residual']

    # Run the program given the above opts, save results in pickle file.
    residual_fields, outdict = model_error.calculate_model_error(**opts)

    #for rf in residual_fields:
    #    print(rf.compute(mode='real'))


if __name__ == '__main__':
    main()
