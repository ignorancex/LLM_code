from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np

from lsstools import parameters
from lsstools import parameters_ms_gadget
from lsstools.model_spec import *
from model_error_at_cat_pos import calc_and_save_model_errors_at_cat_pos


def main():
    """
    Calculate Perror for velocity at halo positions.
    """

    #####################################
    # PARSE COMMAND LINE ARGS
    #####################################
    ap = ArgumentParser()

    ap.add_argument('--SimSeed',
                    type=int,
                    default=400,
                    help='Simulation seed to load.')

    ap.add_argument('--Mmin',
                    default=10.8,
                    type=float,
                    help="Minimum log10M, e.g. 10.8.")

    ap.add_argument('--Mmax',
                    default=11.8,
                    type=float,
                    help="Minimum log10M, e.g. 11.8.")

    ap.add_argument('--ShiftedFieldsNp',
                    default=64,
                    type=int,
                    help='Number of particles used to shift fields.')

    ap.add_argument('--ShiftedFieldsNmesh',
                    default=64,
                    type=int,
                    help='Nmesh used to shift fields.')

    ap.add_argument('--Ngrid',
                    default=64,
                    type=int,
                    help='Number of grid points used to compute power spectra.')

    cmd_args = ap.parse_args()

    #####################################
    # OPTIONS
    #####################################
    opts = OrderedDict()

    # Bump this when changing code without changing options. Otherwise pickle
    # loading might wrongly read old pickles.
    # 31/03/2020: v0.1: Start implementation for velocity at halo pos Perr.
    opts['code_version_for_pickles'] = '0.1'

    # Simulation options. Will be used by path_utils to get input path, and
    # to compute deltalin at the right redshift.
    seed = cmd_args.SimSeed

    halo_mass_string = '%.1f_%.1f' % (cmd_args.Mmin, cmd_args.Mmax)

    opts['sim_opts'] = parameters_ms_gadget.MSGadgetSimOpts.load_default_opts(
        sim_name='ms_gadget',
        sim_seed=seed,
        halo_mass_string=halo_mass_string)

    # Grid options.
    Ngrid = cmd_args.Ngrid
    opts['shifted_fields_Np'] = cmd_args.ShiftedFieldsNp
    opts['shifted_fields_Nmesh'] = cmd_args.ShiftedFieldsNmesh

    opts['grid_opts'] = parameters.GridOpts(
        Ngrid=Ngrid,
        kmax=2.0*np.pi/opts['sim_opts'].boxsize * float(Ngrid)/2.0,
        grid_ptcle2grid_deconvolution=None
        )

    # Options for measuring power spectrum. 
    opts['power_opts'] = parameters.PowerOpts(
        k_bin_width=1.0,
        Pk_1d_2d_mode='1d'
        )

    # Not using trf fcns here b/c do everything at halo pos, not on grid.
    opts['trf_fcn_opts'] = None


    # External grids to load: deltalin, delta_m, shifted grids
    opts['ext_grids_to_load'] = opts['sim_opts'].get_default_ext_grids_to_load(
        Ngrid=opts['grid_opts'].Ngrid,
        shifted_fields_Np=opts['shifted_fields_Np'],
        shifted_fields_Nmesh=opts['shifted_fields_Nmesh'],
        include_shifted_fields=False
        )

    # Catalogs to read
    opts['cat_specs'] = OrderedDict()
    fof_fname = 'nbkit_fof_%.4f/ll_0.200_nmin25' % (
        opts['sim_opts'].sim_scale_factor)

    for direction in range(3):
        opts['cat_specs']['v_h_%d' % direction] = {
            'in_fname': fof_fname,
            'position_column': 'CMPosition',
            'val_column': 'CMVelocity',
            'val_component': direction,
            'rescale_factor': 'RSDFactor',
            'cuts': {
                'log10M': ('min', cmd_args.Mmin),
                'log10M': ('max', cmd_args.Mmax)
            }
        }

    # Specify models to test. Call this trf_specs for historic reasons,
    # actually not using any trf fcns. (Should rename to model_specs or similar.)
    opts['trf_specs'] = []


    # ######################################################################
    # Specify sources and targets of the models
    # ######################################################################

    opts['trf_specs'] = []


    ## Compare simulated halo velocity components vs model, in real space

    ## Define models for halo velocity components
    #for direction in [0,1,2]:
    for direction in [0]:

        # target = halo velocity components
        target = 'v_h_%d' % direction

        if True:
            # model = \tilde\PsiDot^{(1)} 
            s = ('IC_LinearMesh_PsiDot1_%d_intR0.00_extR0.00_SHIFTEDBY_'
                 'IC_LinearMeshR0.23_a%.4f_Np%d_Nm%d_Ng%d_CICavg' % (
                     direction, 
                     opts['sim_opts'].sim_scale_factor,
                     opts['shifted_fields_Np'], 
                     opts['shifted_fields_Nmesh'],
                     opts['grid_opts'].Ngrid))
            opts['trf_specs'].append(TrfSpec(
                fixed_linear_sources=[s],
                target_field=target,
                save_bestfit_field=s,
                field_opts={
                    s: {'additional_rescale_factor': 'f_log_growth',
                        'read_mode': 'velocity', # don't divide out mean
                        'readout_window': 'cic'
                    }
                }))


    # Save results
    opts['keep_pickle'] = True
    opts['pickle_file_format'] = 'dill'
    opts['pickle_path'] = '$SCRATCH/perr/pickle/'

    # Save some additional power spectra that are useful for plotting later
    opts['Pkmeas_helper_columns'] = [
        'deltalin',
        'v_h_0', 'v_h_1', 'v_h_2'
    ]

    opts['Pkmeas_helper_columns_calc_crosses'] = False

    # Cache path
    opts['cache_base_path'] = '$SCRATCH/perr/cache/'

    # Run the program given the above opts, save results in pickle file.
    outdict = calc_and_save_model_errors_at_cat_pos(**opts)


if __name__ == '__main__':
    main()
