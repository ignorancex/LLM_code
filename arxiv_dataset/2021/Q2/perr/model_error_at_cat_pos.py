from __future__ import print_function, division
import os
from collections import OrderedDict
import numpy as np

from lsstools import combine_fields_to_match_target as combine_fields
from lsstools.cosmo_model import CosmoModel
from lsstools.gen_cosmo_fcns import calc_f_log_growth_rate, generate_calc_Da
from lsstools.mesh_collections import ComplexGrid
from lsstools import model_spec
from lsstools.nbkit03_utils import catalog_persist, get_crms, get_cstats_string
from lsstools.paint_utils import mass_avg_weighted_paint_cat_to_rho
from lsstools.results_db.io import Pickler
from nbodykit import CurrentMPIComm, logging, setup_logging
from nbodykit.algorithms.fftpower import FFTPower
from nbodykit.lab import ArrayCatalog, BigFileMesh, BigFileCatalog, FieldMesh
from perr import path_utils
from perr import utils
from read_utils import read_delta_from_bigfile, read_vel_from_bigfile, readout_mesh_at_cat_pos


def calc_and_save_model_errors_at_cat_pos(
    sim_opts=None,
    grid_opts=None,
    power_opts=None,
    trf_fcn_opts=None,
    ext_grids_to_load=None,
    cat_specs=None,
    trf_specs=None,
    keep_pickle=False,
    pickle_file_format='dill',
    pickle_path='$SCRATCH/perr/pickle/',
    Pkmeas_helper_columns=None,
    Pkmeas_helper_columns_calc_crosses=False,
    cache_base_path=None,
    code_version_for_pickles=None,
    shifted_fields_Np=None,
    shifted_fields_Nmesh=None
    ):
    """
    Calculate the model error for all models specified by trf_specs.

    Do this by reading out the model at the positions of objects in the catalog.
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
        cat_specs=cat_specs,
        trf_specs=trf_specs,
        keep_pickle=keep_pickle,
        pickle_file_format=pickle_file_format,
        pickle_path=pickle_path,
        Pkmeas_helper_columns=Pkmeas_helper_columns,
        cache_base_path=cache_base_path,
        code_version_for_pickles=code_version_for_pickles,
        shifted_fields_Np=shifted_fields_Np,
        shifted_fields_Nmesh=shifted_fields_Nmesh
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

    if cat_specs is None:
        cat_specs = {}


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

    # make sure there are no duplicate save_bestfit_field entries
    model_spec.check_trf_specs_consistency(trf_specs)

  
    # Init Pickler instance to save pickle later (this will init pickle fname)
    pickler = None
    if comm.rank == 0:
        pickler = Pickler(path=paths['pickle_path'],
                          base_fname='main_calc_vel_at_halopos_Perr',
                          file_format=pickle_file_format,
                          rand_sleep=(grid_opts.Ngrid > 128))
        print("Pickler: ", pickler.full_fname)
    pickler = comm.bcast(pickler, root=0)

    paths['cache_path'] = utils.make_cache_path(paths['cache_base_path'], comm)

    # Get list of all densities actually needed for trf fcns.
    #densities_needed_for_trf_fcns = utils.get_densities_needed_for_trf_fcns(
    #    trf_specs)


    # ##########################################################################
    # Run program.
    # ##########################################################################

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


    # Compute model at catalog positions, and residual to target.
    pickle_dict = calc_model_errors_at_cat_pos(
        trf_specs=trf_specs,
        paths=paths,
        cat_specs=cat_specs,
        ext_grids_to_load=ext_grids_to_load,
        trf_fcn_opts=trf_fcn_opts,
        grid_opts=grid_opts,
        sim_opts=sim_opts,
        power_opts=power_opts,
        Pkmeas_helper_columns=Pkmeas_helper_columns,
        Pkmeas_helper_columns_calc_crosses=Pkmeas_helper_columns_calc_crosses,
        f_log_growth=opts['f_log_growth']
        )


    # copy over opts so they are saved
    assert not pickle_dict.has_key('opts')
    pickle_dict['opts'] = opts.copy()

    # save all resutls to pickle
    if comm.rank == 0:
        pickler.write_pickle(pickle_dict)

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

    return pickle_dict


def calc_model_errors_at_cat_pos(
        trf_specs=None,
        paths=None,
        cat_specs=None,
        ext_grids_to_load=None,
        trf_fcn_opts=None,
        grid_opts=None,
        sim_opts=None,
        power_opts=None,
        Pkmeas_helper_columns=None,
        Pkmeas_helper_columns_calc_crosses=False,
        f_log_growth=None,
        debug=True
        ):

    """ 
    Compute models at catalog positions, and residual to target, for each
    trf_spec.
    """
    comm = CurrentMPIComm.get()

    result_of_trf_spec = OrderedDict()

    for trf_spec in trf_specs:

        cats = OrderedDict()
    
        if True:
            ## Load target catalog
            cat_id = trf_spec.target_field
            cat_opts = cat_specs[cat_id]

            # read file
            fname = os.path.join(paths['in_path'], cat_opts['in_fname'])
            if comm.rank == 0:
                print('Read %s' % fname)

            # Read rho, not dividing or subtracting mean, to get velocity correctly
            cat = BigFileCatalog(fname, dataset='./', header='Header')

            # cuts
            for cut_column, cut_instruction in cat_opts['cuts'].items():
                cut_op, cut_value = cut_instruction
                if cut_op == 'min':
                    cat = cat[ cat[cut_column] >= cut_value ]
                elif cut_op == 'max':
                    cat = cat[ cat[cut_column] < cut_value ]
                else:
                    raise Exception('Invalid cut operation %s' % str(cut_op))
                
            # Set Position column
            cat['Position'] = cat[cat_opts['position_column']]

            # compute the value we are interested in, save in 'val' column
            component = cat_opts['val_component']
            if component is None:
                cat['val'] = cat[cat_opts['val_column']][:]
            else:
                cat['val'] = cat[cat_opts['val_column']][:, component]

            # catalog rescale factor
            if cat_opts.get('rescale_factor', None) is not None:
                if cat_opts['rescale_factor'] == 'RSDFactor':
                    cat['val'] *= cat.attrs['RSDFactor'][0]
                else:
                    raise Exception('Invalid rescale_factor %s' % 
                        cat_opts['rescale_factor'])

            # additional rescale factors or post processing options for catalog
            if hasattr(trf_spec, 'field_opts'):
                if cat_id in trf_spec.field_opts:
                    this_field_opts = trf_spec.field_opts[cat_id]
                    if this_field_opts.get('additional_rescale_factor', 1.0) != 1.0:
                        resc_fac = this_field_opts['additional_rescale_factor']
                        if type(resc_fac) == float:
                            cat['val'] *= resc_fac
                        else:
                            raise Exception("Invalid rescale factor: %s" % resc_fac)

            # keep only Position and val columns, delete all other columns
            cat2 = catalog_persist(cat, columns=['Position','val'])
            del cat

            cstats = get_cstats_string(cat2['val'].compute())
            if comm.rank == 0:
                print('CATALOG %s: %s\n' % (cat_id, cstats))

            # Store in cats dict
            cats[cat_id] = cat2
            del cat2


        target_cat = cats[trf_spec.target_field]


        # Make sure we have no linear source (trf fcns not implemented here)
        if trf_spec.linear_sources is not None:
            raise Exception('Trf fcns not implemented for model error at cat pos')

        ## Load remaining fields, assuming they are bigfiles meshs, and get
        # residual at target positions.
        residual_cat = target_cat.copy()

        for mesh_id in trf_spec.fixed_linear_sources:

            fname = os.path.join(paths['in_path'], mesh_id)
            if comm.rank == 0:
                print('Read %s' % fname)

            if hasattr(trf_spec, 'field_opts'):
                fopts = trf_spec.field_opts[mesh_id]
            else:
                # default options for meshs
                fopts = {
                    'read_mode': 'velocity',
                    'readout_window': 'cic'}

            # read mesh
            if fopts['read_mode'] == 'velocity':
                # get rho (don't divide by mean)
                mesh = read_vel_from_bigfile(fname)
            elif fopts['read_mode'] == 'density':
                # compute fractional delta (taking out mean)
                mesh = read_delta_from_bigfile(fname)
            else:
                raise Exception('Invalid read_mode: %s' % fopts['read_mode'])

            # more post processing of mesh
            if fopts.get('additional_rescale_factor', 1.0) != 1.0:
                resc_fac = fopts['additional_rescale_factor']
                if comm.rank == 0:
                    print('Apply rescale fac to %s: %s' % (mesh_id, str(resc_fac)))
                if type(resc_fac) == float:
                    mesh = FieldMesh(mesh.compute(mode='real')*resc_fac)
                elif resc_fac == 'f_log_growth':
                    mesh = FieldMesh(
                        mesh.compute(mode='real')*f_log_growth)
                else:
                    raise Exception("Invalid rescale factor: %s" % resc_fac)

            cstats = get_cstats_string(mesh.compute())
            if comm.rank == 0:
                print('MESH %s: %s\n' % (mesh_id, cstats))
       
            # Read out mesh at Position of target catalog
            if comm.rank == 0:
                print('Read out mesh at target pos: %s' % mesh_id)
            mesh_at_target_pos = readout_mesh_at_cat_pos(
                mesh=mesh,
                cat=target_cat,
                readout_window=fopts['readout_window']
                )

            ## Compute residual = target - fixed_linear_sources at target position
            residual_cat['val'] -= mesh_at_target_pos

        cstats = get_cstats_string(residual_cat['val'].compute())
        if comm.rank == 0:
            print('RESIDUAL %s - %s: %s\n' % (
                trf_spec.target_field, trf_spec.save_bestfit_field, cstats))


        # plot histogram
        if debug:
            import matplotlib.pyplot as plt
            plt.hist(residual_cat['val'].compute(), bins=50, range=(-10,10), alpha=0.5)
            plt.hist(target_cat['val'].compute(), bins=50, range=(-10,10), alpha=0.5)
            plt.hist(mesh_at_target_pos, bins=50, range=(-10,10), alpha=0.5)


            plt.show()


        ## paint to mesh
        to_mesh_kwargs = {
            #'Nmesh': grid_opts.Ngrid,
            'window': 'cic',
            'compensated': False,
            'interlaced': False,
            #'BoxSize': residual_cat.attrs['BoxSize'],
            'dtype': 'f8'
        }


        # rho mesh of target, taking avg value instead of summing
        target_mesh = FieldMesh(
            mass_avg_weighted_paint_cat_to_rho(
                target_cat,
                weight='val',
                Nmesh=grid_opts.Ngrid,
                to_mesh_kwargs=to_mesh_kwargs,
                rho_of_empty_cells=0.0,
                verbose=True)[0])
       

        # rho mesh of residual, taking avg value instead of summing
        residual_mesh = FieldMesh(
            mass_avg_weighted_paint_cat_to_rho(
                residual_cat,
                weight='val',
                Nmesh=grid_opts.Ngrid,
                to_mesh_kwargs=to_mesh_kwargs,
                rho_of_empty_cells=0.0,
                verbose=True)[0])

        ## Compute power spectra
        pow_kwargs = {
            'mode': power_opts.Pk_1d_2d_mode,
            'k_bin_width': power_opts.k_bin_width
        }


        ## Save results of this trf_spec in dict
        trf_res = OrderedDict()

        # Power spectra
        trf_res['power'] = OrderedDict()
        trf_res['power']['Target-Model'] = calc_power(
            residual_mesh, **pow_kwargs)
        trf_res['power']['Target'] = calc_power(
            target_mesh, **pow_kwargs)


        if debug:
            # plot power spectra
            import matplotlib.pyplot as plt
            print('plot power')
            k = trf_res['power']['Target'].power['k']
            plt.loglog(
                k,
                k**2*trf_res['power']['Target'].power['power'])
            k = trf_res['power']['Target-Model'].power['k']
            plt.loglog(
                k,
                k**2*trf_res['power']['Target-Model'].power['power'])
            plt.show()

        # target catalog results
        trf_res['target_cat_attrs'] = target_cat.attrs
        trf_res['target_cat_csize'] = target_cat.csize

        # todo: save rms of fields and target
        result_of_trf_spec[trf_spec.save_bestfit_field] = trf_res

    # save all results in a big pickle_dict
    pickle_dict = OrderedDict()
    pickle_dict['result_of_trf_spec'] = result_of_trf_spec

    print('pickle_dict:', pickle_dict)

    return pickle_dict



# def paint_cat_avg_value_to_mesh(
#     cat, 
#     value='Value',
#     additional_to_mesh_kwargs=None):
#     """Paint catalog to mesh, using value. Average values, don't sum.
#     """
#     if additional_to_mesh_kwargs is None:
#         additional_to_mesh_kwargs = {}
# mesh = cat.to_mesh(value=value, **additional_to_mesh_kwargs)

# #if mesh.comm.rank == 0:
# #    print("mesh attrs: %s" % str(mesh.attrs))

# # Paint. If normalize=True, outfield = 1+delta; if normalize=False: outfield=rho
# outfield = mesh.to_real_field(normalize=normalize)
# mesh = FieldMesh(outfield)
# return mesh


def calc_power(mesh, second=None, mode='1d', k_bin_width=1.0, verbose=False):
    BoxSize = mesh.attrs['BoxSize']
    assert BoxSize[0] == BoxSize[1]
    assert BoxSize[0] == BoxSize[2]
    boxsize = BoxSize[0]
    dk = 2.0 * np.pi / boxsize * k_bin_width
    kmin = 2.0 * np.pi / boxsize / 2.0

    if mode == '1d':
        res = FFTPower(first=mesh,
                        second=second,
                        mode=mode,
                        dk=dk,
                        kmin=kmin)
        if verbose and mesh.comm.rank == 0:
            print('power: ', res.power['power'])
            #print(res.attrs)
        return res
    else:
        raise Exception("Mode not implemented: %s" % mode)


