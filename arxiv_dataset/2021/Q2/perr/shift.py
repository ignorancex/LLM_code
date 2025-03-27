from __future__ import print_function, division
from nbodykit.lab import *
import numpy as np
import os
from pmesh.pm import ParticleMesh
from shutil import rmtree

from lsstools import nbkit03_utils, paint_utils
from lsstools.cosmo_model import CosmoModel
from lsstools.gen_cosmo_fcns import calc_f_log_growth_rate, generate_calc_Da
from nbodykit import CurrentMPIComm, logging


def weigh_and_shift_uni_cats(
    Nptcles_per_dim,
    out_Ngrid,
    densities_to_shift,
    displacement_source=None,
    RSD_displacement_source=None,
    PsiOrder=1,
    RSD=False,
    RSD_method=None,
    RSD_line_of_sight=None,
    basepath=None,
    out_scale_factor=None,
    internal_scale_factor_for_weights=None,
    weighted_CIC_mode=None,
    boxsize=None,
    Nmesh_orig=None,
    cosmo_params=None,
    save_result=False,
    verbose=False,
    plot_slices=False,
    ):
    """
    Do the following: 
    - create uniform catalog
    - load deltalin or other density delta on grid (given by densities_to_shift)
    - weigh ptcles in uniform catalog by 1+delta
    - compute smoothed Psi_lin on grid (=evaluated at ptcle positions)
    - shift (displace) particles by Psi
    - compute density of shifted catalog using weighted CIC
    - save shifted catalog

    Parameters
    ----------
    Nptcles_per_dim : int
        Number of particles per dimension to use for uniform catalog

    out_Ngrid : int
        Number of cells per dimension for the output grid to which particles are
        CIC interpolated.

    densities_to_shift : dict
        Dicts specifying which fields to load from disk. Specify file name etc.
    
    Nmesh_orig : int
        Only used for output file name. Number of cells per dimension of the 
        original field that is shifted and of the displacement field.

    displacement_source : dict
        Dict specifying how to compute displacement field.
    """
    # init MPI
    comm = CurrentMPIComm.get()
    print("%d: Greetings from rank %d" % (comm.rank, comm.rank))
    logger = logging.getLogger("shift")

    outmesh_dict = dict()
    outfiles_dict = dict()

    if RSD:
        # calculate f
        cosmo = CosmoModel(**cosmo_params)
        calc_Da = generate_calc_Da(cosmo=cosmo)
        f_log_growth = calc_f_log_growth_rate(a=out_scale_factor,
                                              calc_Da=calc_Da,
                                              cosmo=cosmo,
                                              do_test=True)
    else:
        f_log_growth = None

    # loop over all densities to be shifted (load from disk)
    for specs_of_density_to_shift in densities_to_shift:

        # ######################################################################
        # Load density to be shifted and weigh particles by delta
        # ######################################################################

        # todo: could create linearmesh on the fly rather than reading from disk.
        # but also fine to load from disk for now.

        # get field from bigmesh file; rescale to delta at internal_scale_factor_for_weights.
        # note that we get delta, not 1+delta.
        rfield_density_to_shift = nbkit03_utils.get_rfield_from_bigfilemesh_file(
            specs_of_density_to_shift['in_fname'],
            file_scale_factor=specs_of_density_to_shift['file_scale_factor'],
            desired_scale_factor=internal_scale_factor_for_weights,
            cosmo_params=cosmo_params)

        nbkit03_utils.rfield_print_info(rfield_density_to_shift, comm,
                                        'rfield_density_to_shift: ')

        # compute quadratic field if desired
        if specs_of_density_to_shift.get('calc_quadratic_field',
                                         None) is not None:
            # get quadratic field
            rfield_density_to_shift = nbkit03_utils.calc_quadratic_field(
                base_field_mesh=FieldMesh(rfield_density_to_shift),
                quadfield=specs_of_density_to_shift['calc_quadratic_field'],
                smoothing_of_base_field=specs_of_density_to_shift.get(
                    'smoothing_quadratic_source', None),
                verbose=verbose).compute(mode='real')

            # print info
            nbkit03_utils.rfield_print_info(
                rfield_density_to_shift, comm,
                '%s: ' % specs_of_density_to_shift['calc_quadratic_field'])

        if specs_of_density_to_shift.get('calc_trf_of_field', None) is not None:
            # calculate some transformation of the field
            if specs_of_density_to_shift['calc_trf_of_field'] == '1':
                # replace field by 1
                rfield_density_to_shift = 0 * rfield_density_to_shift + 1.0
            else:
                raise Exception(
                    "Invalid calc_trf_of_field %s" %
                    str(specs_of_density_to_shift['calc_trf_of_field']))


        if specs_of_density_to_shift.get('external_smoothing', None) is not None:
            rfield_density_to_shift = nbkit03_utils.apply_smoothing(
                mesh_source=FieldMesh(rfield_density_to_shift),
                **specs_of_density_to_shift['external_smoothing']
                ).compute(mode='real')


        # print("%d: rfield_density_to_shift: type=%s, shape=%s, rms=%g"% (
        #     comm.rank, str(type(rfield_density_to_shift)), str(rfield_density_to_shift.shape),
        #     np.sqrt(np.mean(rfield_density_to_shift**2))))


        # ######################################################################
        # Compute displacement Psi on grid, interpolate to particle positions, 
        # and shift them. 
        # ######################################################################

        # get displacement source field from bigmesh file; scale to out_scale_factor
        # to make sure we use full displacement up to low z of output.
        rfield_displacement_source = nbkit03_utils.get_rfield_from_bigfilemesh_file(
            displacement_source['in_fname'],
            file_scale_factor=displacement_source['file_scale_factor'],
            desired_scale_factor=out_scale_factor,
            cosmo_params=cosmo_params)

        RSD_method = specs_of_density_to_shift['RSD_method']

        if (RSD==False) or (RSD==True and RSD_method == 'LPT'):

            Psi_rfields = [None, None, None]
            for direction in range(3):
                # Get Psi_i = k_i delta_smoothed/k^2 following http://rainwoodman.github.io/pmesh/intro.html
                if comm.rank == 0:
                    print("%d: Get Psi_%d: " % (comm.rank, direction))

                # Compute displacement
                Psi_rfields[
                    direction] = nbkit03_utils.get_displacement_from_density_rfield(
                        rfield_displacement_source,
                        component=direction,
                        Psi_type=displacement_source['Psi_type'],
                        smoothing=displacement_source['smoothing'],
                        smoothing_Psi3LPT=displacement_source.get('smoothing_Psi3LPT', None),
                        RSD=RSD,
                        RSD_line_of_sight=RSD_line_of_sight,
                        RSD_f_log_growth=f_log_growth)

            # ######################################################################
            # Shift uniform catalog weighted by delta_for_weights and get shifted
            # density.
            # ######################################################################

            # get delta_shifted  (get 1+delta)
            delta_shifted, attrs = weigh_and_shift_uni_cat(
                delta_for_weights=rfield_density_to_shift,
                displacements=Psi_rfields,
                Nptcles_per_dim=Nptcles_per_dim,
                out_Ngrid=out_Ngrid,
                boxsize=boxsize,
                internal_scale_factor_for_weights=internal_scale_factor_for_weights,
                out_scale_factor=out_scale_factor,
                cosmo_params=cosmo_params,
                weighted_CIC_mode=weighted_CIC_mode,
                plot_slices=plot_slices,
                verbose=verbose
            )


        elif RSD_method == 'PotentialOfRSDDisplSource_xPtcles':
            ## Step 1: Move operators without any RSD displacement. 
            ## Step 2: Apply RSD displacement in EUlerian space by displacing by
            # k/k^2*delta_RSD_displ_source, where delta_RSD_displ_source=delta_ZA.
            
            assert RSD_displacement_source is not None

            ## Step 1: Move operators without any RSD displacement. 
            Psi_rfields = [None, None, None]
            for direction in range(3):
                # Get Psi_i = k_i delta_smoothed/k^2 following http://rainwoodman.github.io/pmesh/intro.html
                if comm.rank == 0:
                    print("%d: Get Psi_%d: " % (comm.rank, direction))

                # Compute displacement used to displace operators from q to x
                Psi_rfields[
                    direction] = nbkit03_utils.get_displacement_from_density_rfield(
                        rfield_displacement_source,
                        component=direction,
                        Psi_type=displacement_source['Psi_type'],
                        smoothing=displacement_source['smoothing'],
                        smoothing_Psi3LPT=displacement_source.get('smoothing_Psi3LPT', None),
                        RSD=False,  # no RSD displacement in step 1
                        RSD_line_of_sight=RSD_line_of_sight,
                        RSD_f_log_growth=f_log_growth)

            del rfield_displacement_source

            # Shift uniform catalog weighted by rfield_density_to_shift.
            # get delta_shifted  (get 1+delta)
            delta_shifted_noRSD, attrs = weigh_and_shift_uni_cat(
                delta_for_weights=rfield_density_to_shift,
                displacements=Psi_rfields,
                Nptcles_per_dim=Nptcles_per_dim,
                out_Ngrid=out_Ngrid,
                boxsize=boxsize,
                internal_scale_factor_for_weights=internal_scale_factor_for_weights,
                out_scale_factor=out_scale_factor,
                cosmo_params=cosmo_params,
                weighted_CIC_mode=weighted_CIC_mode,
                plot_slices=plot_slices,
                verbose=verbose
            )


            ## Step 2: Apply RSD displacement in Eulerian x space by displacing 
            # by f*k/k^2*deltaZA.

            # Generate new uniform catalog in x space, with particle weights
            # given by delta_shifted(x). Then shift by fk/k^2 delta_ZA(x).
            # Note k/k^2 delta_ZA != k/k^2 delta_lin = Psi_ZA.
            # In general, use delta_RSD_displacement_source.
            del Psi_rfields

            # get RSD displacement source field from bigmesh file; scale to out_scale_factor
            rfield_RSD_displacement_source = nbkit03_utils.get_rfield_from_bigfilemesh_file(
                RSD_displacement_source['in_fname'],
                file_scale_factor=RSD_displacement_source['file_scale_factor'],
                desired_scale_factor=out_scale_factor,
                cosmo_params=cosmo_params)

            Psi_RSD_rfields = [None, None, None]
            if RSD_line_of_sight == [0,0,1]:
                # get k_z / k^2 delta_ZA
                direction = 2
                Psi_RSD_rfields[direction] = nbkit03_utils.get_displacement_from_density_rfield(
                        in_density_rfield=rfield_RSD_displacement_source,
                        component=direction,
                        Psi_type=RSD_displacement_source['Psi_type'], # to get k/k^2*in_density_rfield
                        smoothing=RSD_displacement_source['smoothing'])

                # 23/03/2020: flipping sign of displacement and using - here
                # gives worse Perr.
                Psi_RSD_rfields[direction] *= f_log_growth

            else:
                raise Exception('RSD_line_of_sight not implemented: ',
                    RSD_line_of_sight)

            delta_shifted, attrs = weigh_and_shift_uni_cat(
                delta_for_weights=delta_shifted_noRSD,
                displacements=Psi_RSD_rfields,
                Nptcles_per_dim=Nptcles_per_dim,
                out_Ngrid=out_Ngrid,
                boxsize=boxsize,
                internal_scale_factor_for_weights=internal_scale_factor_for_weights,
                out_scale_factor=out_scale_factor,
                cosmo_params=cosmo_params,
                weighted_CIC_mode=weighted_CIC_mode,
                plot_slices=plot_slices,
                verbose=verbose
            )


        else:
            raise Exception('Invalid RSD_method %s' % RSD_method)

        # print cmean
        shifted_cmean = delta_shifted.cmean()
        if comm.rank == 0:
            print("%d: delta_shifted cmean:" % comm.rank, shifted_cmean)


        # ######################################################################
        # save to bigfile
        # ######################################################################
        if save_result:

            if not RSD:
                RSDstring = ''
            else:
                if RSD_line_of_sight in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]:
                    RSDstring = '_RSD%d%d%d' % (RSD_line_of_sight[0],
                                                RSD_line_of_sight[1],
                                                RSD_line_of_sight[2])
                else:
                    # actually not implemented above
                    RSDstring = '_RSD_%.2f_%.2f_%.2f' % (
                        RSD_line_of_sight[0], RSD_line_of_sight[1],
                        RSD_line_of_sight[2])

                if RSD_method == 'LPT':
                    pass
                elif RSD_method == 'PotentialOfRSDDisplSource_xPtcles':
                    RSDstring += "_%s" % RSD_displacement_source['id_for_out_fname']
                else:
                    raise Exception('Invalid RSD_method %s' % RSD_method)

            out_fname = os.path.join(
                basepath,
                '%s_int%s_ext%s_SHIFTEDBY_%s%s_a%.4f_Np%d_Nm%d_Ng%d_CIC%s%s' %
                (specs_of_density_to_shift['id_for_out_fname'],
                 smoothing_str(
                     specs_of_density_to_shift.get('smoothing_quadratic_source',
                                                   None)),
                 smoothing_str(specs_of_density_to_shift['external_smoothing']),
                 displacement_source['id_for_out_fname'],
                 smoothing_str(displacement_source['smoothing']),
                 out_scale_factor, Nptcles_per_dim, Nmesh_orig,
                 out_Ngrid, weighted_CIC_mode, RSDstring))

            if comm.rank == 0:
                print("Writing to %s" % out_fname)

            # delta_shifted contains 1+delta_shifted
            # Wrongly used FieldMesh(1+delta_shifted) until 31/3/2020.
            outmesh = FieldMesh(delta_shifted)

            # copy MeshSource attrs
            for k, v in attrs.items():
                outmesh.attrs['MeshSource_%s' % k] = v
            if comm.rank == 0:
                print("outmesh.attrs:\n", outmesh.attrs)

            outmesh.save(out_fname, mode='real')
            if comm.rank == 0:
                print("Wrote %s" % out_fname)

        else:            
            if comm.rank == 0:
                print("Not writing result to disk")

        # return mesh
        if specs_of_density_to_shift.get('return_mesh', False):
            mykey = specs_of_density_to_shift['id_for_out_fname']
            if mykey in outmesh_dict:
                raise Exception('Do not overwrite outmesh_dict key')
            else:
                outmesh_dict[mykey] = outmesh

            outfiles_dict[mykey] = out_fname


        # optionally plot slice
        if plot_slices:
            plt.imshow(outmesh.preview(Nmesh=32, axes=(0, 1)))
            if comm.rank == 0:
                plt_fname = 'outmesh_Np%d_Nm%d_Ng%d.pdf' % (
                    Nptcles_per_dim, Nmesh_orig, out_Ngrid)
                plt.savefig(plt_fname)
                print("Made %s" % plt_fname)

    return outmesh_dict, outfiles_dict

def weigh_and_shift_uni_cat(
    delta_for_weights,
    displacements,
    Nptcles_per_dim,
    out_Ngrid,
    boxsize,
    internal_scale_factor_for_weights=None,
    out_scale_factor=None,
    cosmo_params=None,
    weighted_CIC_mode=None,
    uni_cat_generator='pmesh',
    plot_slices=False,
    verbose=False,
    return_catalog=False
    ):
    """
    Make uniform catalog, weigh by delta_for_weights, displace by displacements
    and interpolate to grid which we return.

    Parameters
    ----------
    delta_for_weights : None or pmesh.pm.RealField object
        Particles are weighted by 1+delta_for_weights. If None, use weight=1.

    displacements : list
        [Psi_x, Psi_y, Psi_z] where Psi_i are pmesh.pm.RealField objects,
        holding the displacement field in different directions (on the grid).
        If None, do not shift.

    uni_cat_generator : string
        If 'pmesh', use pmesh for generating uniform catalog.
        If 'manual', use an old serial code.

    Returns
    -------
    delta_shifted : FieldMesh object
        Density delta_shifted of shifted weighted particles (normalized to mean
        of 1 i.e. returning 1+delta). Returned if return_catalog=False.

    attrs : meshsource attrs
        Attrs of delta_shifted. Returned if return_catalog=False.

    catalog_shifted : ArrayCatalog object
        Shifted catalog, returned if return_catalog=True.
    """
    comm = CurrentMPIComm.get()

    # ######################################################################
    # Generate uniform catalog with Nptcles_per_dim^3 particles on regular 
    # grid
    # ######################################################################

    if uni_cat_generator == 'pmesh':
        # use pmesh generate_uniform_particle_grid
        # http://rainwoodman.github.io/pmesh/pmesh.pm.html?highlight=
        # readout#pmesh.pm.ParticleMesh.generate_uniform_particle_grid
        pmesh = ParticleMesh(BoxSize=boxsize,
                             Nmesh=[
                                 Nptcles_per_dim,
                                 Nptcles_per_dim,
                                 Nptcles_per_dim
                             ])
        ptcles = pmesh.generate_uniform_particle_grid(shift=0.0, dtype='f8')
        #print("type ptcles", type(ptcles), ptcles.shape)
        #print("head ptcles:", ptcles[:5,:])

        dtype = np.dtype([('Position', ('f8', 3))])

        # number of rows is given by number of ptcles on this rank
        uni_cat_array = np.empty((ptcles.shape[0],), dtype=dtype)
        uni_cat_array['Position'] = ptcles

        uni_cat = ArrayCatalog(uni_cat_array,
                               comm=None,
                               BoxSize=boxsize * np.ones(3),
                               Nmesh=[
                                   Nptcles_per_dim,
                                   Nptcles_per_dim,
                                   Nptcles_per_dim
                               ])

        print("%d: local Nptcles=%d, global Nptcles=%d" %
              (comm.rank, uni_cat.size, uni_cat.csize))

        del ptcles
        del uni_cat_array


    elif uni_cat_generator == 'manual':

        # Old serial code to generate regular grid and scatter across ranks.
        if comm.rank == 0:
            # Code copied from do_rec_v1.py and adopted

            # Note that nbkit UniformCatalog is random catalog, but we want a catalog
            # where each ptcle sits at grid points of a regular grid.
            # This is what we call 'regular uniform' catalog.
            Np = Nptcles_per_dim
            dtype = np.dtype([('Position', ('f8', 3))])
            # Have Np**3 particles, and each particle has position x,y,z and weight 'Weight'
            uni_cat_array = np.empty((Np**3,), dtype=dtype)

            # x components in units such that box ranges from 0 to 1. Note dx=1/Np.
            #x_components_1d = np.linspace(0.0, (Np-1)*(L/float(Np)), num=Np, endpoint=True)/L
            x_components_1d = np.linspace(0.0, (Np - 1) / float(Np),
                                          num=Np,
                                          endpoint=True)
            ones_1d = np.ones(x_components_1d.shape)

            # Put particles on the regular grid
            print("%d: Fill regular uniform catalog" % comm.rank)
            uni_cat_array['Position'][:, 0] = np.einsum(
                'a,b,c->abc', x_components_1d, ones_1d, ones_1d).reshape(
                    (Np**3,))
            uni_cat_array['Position'][:, 1] = np.einsum(
                'a,b,c->abc', ones_1d, x_components_1d, ones_1d).reshape(
                    (Np**3,))
            uni_cat_array['Position'][:, 2] = np.einsum(
                'a,b,c->abc', ones_1d, ones_1d, x_components_1d).reshape(
                    (Np**3,))
            print("%d: Done filling regular uniform catalog" % comm.rank)

            # in nbkit0.3 units must be in Mpc/h
            uni_cat_array['Position'] *= boxsize

        else:
            uni_cat_array = None

        # Scatter across all ranks
        print("%d: Scatter array" % comm.rank)
        from nbodykit.utils import ScatterArray
        uni_cat_array = ScatterArray(uni_cat_array,
                                     comm,
                                     root=0,
                                     counts=None)
        print("%d: Scatter array done. Shape: %s" %
              (comm.rank, str(uni_cat_array.shape)))

        # Save in ArrayCatalog object
        uni_cat = ArrayCatalog(uni_cat_array)
        uni_cat.attrs['BoxSize'] = np.ones(3) * boxsize
        uni_cat.attrs['Nmesh'] = np.ones(3) * Nptcles_per_dim
        uni_cat.attrs['Nmesh_internal'] = np.ones(3) * Nmesh_orig

    else:
        raise Exception('Invalid uni_cat_generator %s' % uni_cat_generator)


    ########################################################################
    # Set weight of particles in uni_cat to delta (interpolated to ptcle 
    # positions)
    ########################################################################
    if delta_for_weights is None:
        # set all weights to 1
        uni_cat['Mass'] = np.ones(uni_cat['Position'].shape[0])
    else:
        # weight by delta_for_weights
        nbkit03_utils.interpolate_pm_rfield_to_catalog(
            delta_for_weights, uni_cat, catalog_column_to_save_to='Mass')

    print("%d: rms Mass: %g" %
          (comm.rank, np.sqrt(np.mean(np.array(uni_cat['Mass'])**2))))

    # optionally plot weighted uniform cat before shifting
    if plot_slices:
        # paint the original uni_cat to a grid and plot slice
        import matplotlib.pyplot as plt

        tmp_meshsource = uni_cat.to_mesh(Nmesh=out_Ngrid,
                                         value='Mass',
                                         window='cic',
                                         compensated=False,
                                         interlaced=False)
        # paint to get delta(a_internal)
        tmp_outfield = tmp_meshsource.paint(mode='real')
        # linear rescale factor from internal_scale_factor_for_weights to 
        # out_scale_factor
        rescalefac = nbkit03_utils.linear_rescale_fac(
            internal_scale_factor_for_weights,
            out_scale_factor,
            cosmo_params=cosmo_params)
        tmp_outfield = 1.0 + rescalefac * (tmp_outfield - 1.0)
        tmp_mesh = FieldMesh(tmp_outfield)
        plt.imshow(tmp_mesh.preview(Nmesh=32, axes=(0, 1)))
        if comm.rank == 0:
            plt_fname = 'inmesh_Np%d_Nm%d_Ng%d.pdf' % (
                Nptcles_per_dim, Nmesh_orig, out_Ngrid)
            plt.savefig(plt_fname)
            print("Made %s" % plt_fname)
        del tmp_meshsource, rescalefac, tmp_outfield, tmp_mesh

    # ######################################################################
    # Shift uniform catalog particles by Psi (changes uni_cat)
    # ######################################################################
    nbkit03_utils.shift_catalog_by_psi_grid(
        cat=uni_cat,
        in_displacement_rfields=displacements,
        pos_column='Position',
        pos_units='Mpc/h',
        displacement_units='Mpc/h',
        boxsize=boxsize,
        verbose=verbose)
    #del Psi_rfields

    if return_catalog:
        # return shifted catalog
        return uni_cat

    else:
        # return density of shifted catalog, delta_shifted

        # ######################################################################
        # paint shifted catalog to grid, using field_to_shift as weights
        # ######################################################################

        print("%d: paint shifted catalog to grid using mass weights" %
              comm.rank)

        # this gets 1+delta
        if weighted_CIC_mode == 'sum':
            delta_shifted, attrs = paint_utils.weighted_paint_cat_to_delta(
                uni_cat,
                weight='Mass',
                Nmesh=out_Ngrid,
                weighted_paint_mode=weighted_CIC_mode,
                normalize=True, # compute 1+delta
                verbose=verbose,
                to_mesh_kwargs={
                    'window': 'cic',
                    'compensated': False,
                    'interlaced': False
                })

        # this get rho
        elif weighted_CIC_mode == 'avg':
            delta_shifted, attrs = paint_utils.mass_avg_weighted_paint_cat_to_rho(
                uni_cat,
                weight='Mass',
                Nmesh=out_Ngrid,
                verbose=verbose,
                to_mesh_kwargs={
                    'window': 'cic',
                    'compensated': False,
                    'interlaced': False
                })

        else:
            raise Exception('Invalid weighted_CIC_mode %s' % weighted_CIC_mode)

        # ######################################################################
        # rescale to output redshift
        # ######################################################################

        if internal_scale_factor_for_weights != out_scale_factor:
            # linear rescale factor from internal_scale_factor_for_weights to 
            # out_scale_factor
            rescalefac = nbkit03_utils.linear_rescale_fac(
                internal_scale_factor_for_weights,
                out_scale_factor,
                cosmo_params=cosmo_params)

            delta_shifted *= rescalefac

            # print some info:
            if comm.rank == 0:
                print("%d: Linear rescalefac from a=%g to a=%g, rescalefac=%g" %
                      (comm.rank, internal_scale_factor_for_weights,
                       out_scale_factor, rescalefac))

            raise Exception('Check if rescaling of delta_shifted is correct. Looks like 1+delta.')


        if verbose:
            print("%d: delta_shifted: min, mean, max, rms(x-1):" % comm.rank,
                  np.min(delta_shifted), np.mean(delta_shifted),
                  np.max(delta_shifted), np.mean((delta_shifted - 1.)**2)**0.5)

        # get 1+deta mesh from field
        #outmesh = FieldMesh(1 + out_delta)

        # print some info: this makes code never finish (race condition maybe?)
        #nbkit03_utils.rfield_print_info(outfield, comm, 'outfield: ')

        return delta_shifted, attrs


def smoothing_str(smoothing_dict):
    if smoothing_dict is None:
        return 'R0.00'
    else:
        if smoothing_dict['mode'] == 'Gaussian':
            if smoothing_dict.get('kmax', 0.0) == 0.0:
                return 'R%.2f' % smoothing_dict['R']
            else:
                return 'R%.2f_%.2f' % (smoothing_dict['R'],
                                       smoothing_dict['kmax'])
        else:
            raise Exception("invalid smoothing mode %s" %
                            str(smoothing['mode']))


if __name__ == '__main__':
    main()
