from __future__ import print_function, division
import os

from lsstools import combine_fields_to_match_target as combine_fields
from lsstools.cosmo_model import CosmoModel
from lsstools.gen_cosmo_fcns import calc_f_log_growth_rate, generate_calc_Da
from lsstools.mesh_collections import ComplexGrid
from lsstools import model_spec
from lsstools.results_db.io import Pickler
from nbodykit import CurrentMPIComm, logging, setup_logging
import path_utils
import utils


#def calculate_model_error(opts):
def calculate_model_error(
    sim_opts=None,
    grid_opts=None,
    power_opts=None,
    trf_fcn_opts=None,
    ext_grids_to_load=None,
    xgrids_in_memory=None,
    kgrids_in_memory=None,
    cats=None,
    trf_specs=None,
    keep_pickle=False,
    pickle_file_format='dill',
    pickle_path='$SCRATCH/perr/pickle/',
    Pkmeas_helper_columns=None,
    Pkmeas_helper_columns_calc_crosses=False,
    store_Pkmeas_in_trf_results=False,
    save_grids4plots=False,
    grids4plots_base_path=None,
    grids4plots_R=None,
    cache_base_path=None,
    RSDstrings=None,
    code_version_for_pickles=None,
    return_fields=None,
    shifted_fields_Np=None,
    shifted_fields_Nmesh=None,
    shifted_fields_RPsi=None
    ):
    """
    Calculate the model error for all models specified by trf_specs.

    Use return_fields=['bestfit'] or ['residual'] or ['bestfit','residual']
    to return the fields as well, as lists in same order as trf_specs.
    """

    # store opts in dict so we can save in pickle later
    opts = dict(    
        sim_opts=sim_opts,
        grid_opts=grid_opts,
        power_opts=power_opts,
        trf_fcn_opts=trf_fcn_opts,
        ext_grids_to_load=ext_grids_to_load,
        #xgrids_in_memory=xgrids_in_memory,
        #kgrids_in_memory=kgrids_in_memory,        
        cats=cats,
        trf_specs=trf_specs,
        keep_pickle=keep_pickle,
        pickle_file_format=pickle_file_format,
        pickle_path=pickle_path,
        Pkmeas_helper_columns=Pkmeas_helper_columns,
        Pkmeas_helper_columns_calc_crosses=Pkmeas_helper_columns_calc_crosses,
        store_Pkmeas_in_trf_results=store_Pkmeas_in_trf_results,
        save_grids4plots=save_grids4plots,
        grids4plots_base_path=grids4plots_base_path,
        grids4plots_R=grids4plots_R,
        cache_base_path=cache_base_path,
        code_version_for_pickles=code_version_for_pickles
        )

    #####################################
    # Initialize
    #####################################

    # make sure we keep the pickle if it is a big run and do not plot
    if grid_opts.Ngrid > 256:
        keep_pickle = True

    # load defaults if not set
    if ext_grids_to_load is None:
        ext_grids_to_load = sim_opts.get_default_ext_grids_to_load(
            Ngrid=grid_opts.Ngrid)

    if cats is None:
        cats = sim_opts.get_default_catalogs()

    

    ### derived options (do not move above b/c command line args might
    ### overwrite some options!)
    opts['in_path'] = path_utils.get_in_path(opts)
    # for output densities
    opts['out_rho_path'] = os.path.join(
        opts['in_path'],
        'out_rho_Ng%d' % grid_opts.Ngrid
    )

    # expand environment names in paths
    paths = {}
    for key in [
            'in_path', 'in_fname', 'in_fname_PTsim_psi_calibration',
            'in_fname_halos_to_displace_by_mchi', 'pickle_path',
            'cache_base_path', 'grids4plots_base_path', 'out_rho_path'
    ]:
        if opts.has_key(key):
            if opts[key] is None:
                paths[key] = None
            else:
                paths[key] = os.path.expandvars(opts[key])

    setup_logging()
    comm = CurrentMPIComm.get()
    logger = logging.getLogger('PerrCalc')

    model_spec.check_trf_specs_consistency(trf_specs)

  
    # Init Pickler instance to save pickle later (this will init pickle fname)
    pickler = None
    if comm.rank == 0:
        pickler = Pickler(path=paths['pickle_path'],
                          base_fname='main_calc_Perr',
                          file_format=pickle_file_format,
                          rand_sleep=(grid_opts.Ngrid > 128))
        print("Pickler: ", pickler.full_fname)
    pickler = comm.bcast(pickler, root=0)

    # where to save grids for slice and scatter plots
    if save_grids4plots:
        paths['grids4plots_path'] = os.path.join(
            paths['grids4plots_base_path'],
            os.path.basename(pickler.full_fname))
        if comm.rank == 0:
            if not os.path.exists(paths['grids4plots_path']):
                os.makedirs(paths['grids4plots_path'])
            print("grids4plots_path:", paths['grids4plots_path'])

    paths['cache_path'] = utils.make_cache_path(paths['cache_base_path'], comm)

    # Get list of all densities actually needed for trf fcns.
    densities_needed_for_trf_fcns = utils.get_densities_needed_for_trf_fcns(
        trf_specs)
    #opts['densities_needed_for_trf_fcns'] = densities_needed_for_trf_fcns


    # ##########################################################################
    # Run program.
    # ##########################################################################

    #if opts.get('RSDstrings', ['']) != ['']:
    if True or RSDstrings not in [ None, [''] ]:
        # calculate D and f
        cosmo = CosmoModel(**sim_opts.cosmo_params)
        calc_Da = generate_calc_Da(cosmo=cosmo)
        f_log_growth = calc_f_log_growth_rate(
            a=sim_opts.sim_scale_factor,
            calc_Da=calc_Da,
            cosmo=cosmo,
            do_test=True
            )
        # save in opts so we can easily access it throughout code (although strictly
        # speaking it is not a free option but derived from cosmo_params)
        opts['f_log_growth'] = f_log_growth
    else:
        opts['f_log_growth'] = None

    # Compute best-fit model and power spectra
    # TODO: maybe split into method computing field and separate method to compute power spectra.
    # For now, load fields from cache as workaround (see below)
    pickle_dict = combine_fields.paint_combine_and_calc_power(
        trf_specs=trf_specs,
        paths=paths,
        catalogs=cats,
        needed_densities=densities_needed_for_trf_fcns,
        ext_grids_to_load=ext_grids_to_load,
        xgrids_in_memory=xgrids_in_memory,
        kgrids_in_memory=kgrids_in_memory,
        trf_fcn_opts=trf_fcn_opts,
        grid_opts=grid_opts,
        sim_opts=sim_opts,
        power_opts=power_opts,
        save_grids4plots=save_grids4plots,
        grids4plots_R=grids4plots_R,
        Pkmeas_helper_columns=Pkmeas_helper_columns,
        Pkmeas_helper_columns_calc_crosses=Pkmeas_helper_columns_calc_crosses,
        store_Pkmeas_in_trf_results=store_Pkmeas_in_trf_results,
        f_log_growth=opts['f_log_growth']
        )

    # Load fields from cache if they shall be returned
    # (actually not used anywhere, could delete)
    if return_fields is not None:
        if 'bestfit' in return_fields:
            bestfit_fields = []
            for trf_spec in trf_specs:
                # load bestfit fields from cache
                gridk = ComplexGrid(
                    fname=pickle_dict['gridk_cache_fname'],
                    read_columns=[trf_spec.save_bestfit_field])
                bestfit_fields.append(gridk.G[trf_spec.save_bestfit_field])
                del gridk

        if 'residual' in return_fields:
            residual_fields = []
            for trf_spec in trf_specs:
                # load residual field from cache
                residual_key = '[%s]_MINUS_[%s]' % (
                    trf_spec.save_bestfit_field,
                    trf_spec.target_field)

                gridk = ComplexGrid(
                    fname=pickle_dict['gridk_cache_fname'],
                    read_columns=[residual_key])

                residual_fields.append(gridk.G[residual_key])
                del gridk

    # copy over opts so they are saved
    assert not pickle_dict.has_key('opts')
    pickle_dict['opts'] = opts.copy()

    # save all resutls to pickle
    if comm.rank == 0:
        pickler.write_pickle(pickle_dict)

    # print path with grids for slice and scatter plotting
    if save_grids4plots:
        print("grids4plots_path: %s" % paths['grids4plots_path'])

    # print save_bestfit_fields
    save_bestfit_fields = [ t.save_bestfit_field for t in opts['trf_specs'] ]
    print('\nsave_bestfit_fields:\n' + '\n'.join(save_bestfit_fields))


    # delete pickle if not wanted any more
    if comm.rank == 0:
        if keep_pickle:
            print("Pickle: %s" % pickler.full_fname)
        else:
            pickler.delete_pickle_file()

        # delete cache dir
        from shutil import rmtree
        rmtree(paths['cache_path'])

    if return_fields in [False, None]:
        return pickle_dict
    elif return_fields == ['bestfit']:
        return bestfit_fields, pickle_dict
    elif return_fields == ['residual']:
        return residual_fields, pickle_dict
    elif return_fields == ['bestfit','residual']:
        return bestfit_fields, residual_fields, pickle_dict

if __name__ == '__main__':
    main()
