from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import os

from nbodykit.algorithms.fftpower import FFTPower
from nbodykit.source.mesh.field import FieldMesh
from nbodykit import CurrentMPIComm

from perr_private.model_target_pair import ModelTargetPair, Model, Target
from skewspec import smoothing
from skewspec.skew_spectrum import SkewSpectrum
from utils import catalog_persist, linear_rescale_fac


def main():
    """
    Script to calculate skew spectra of an input density.
    """

    #####################################
    # PARSE COMMAND LINE ARGS
    #####################################
    ap = ArgumentParser()

    ap.add_argument('--SimSeed',
                    type=int,
                    default=400,
                    help='Simulation seed to load.')

    ap.add_argument('--boxsize',
                    default=1500.0,
                    type=float,
                    help='Boxsize in Mpc/h.')

    ap.add_argument('--ApplyRSD',
                    type=int,
                    default=0,
                    help='0: No RSD. 1: Include RSD in catalog.')

    ap.add_argument('--Rsmooth',
                    default=20.0,
                    type=float,
                    help='Smoothing of quad field.')

    ap.add_argument('--Ngrid', 
                    default=64, 
                    type=int,
                    help='Ngrid used to compute skew spectra.')

    ap.add_argument('--SubsampleRatio',
                    default=0.0015,
                    type=float,
                    help='Subsample ratio of DM snapshot to use as input.')

    ap.add_argument('--MaxDisplacement',
                default=100.0,
                type=float,
                help='Maximum RSD displacement in Mpc/h.')

    ap.add_argument('--DensitySource', default='catalog', type=str,
        help='Source from which to compute the density. catalog or delta_2SPT')

    ap.add_argument('--b1', default=1.0, type=float,
        help='b1 bias. Only used if DensitySource=delta_2SPT.')

    ap.add_argument('--b2', default=0.0, type=float,
        help='b2 bias. Only used if DensitySource=delta_2SPT.')

    ap.add_argument('--bG2', default=0.0, type=float,
        help='bG2 bias. Only used if DensitySource=delta_2SPT.')

    ap.add_argument('--fLogGrowth', default=0.786295, type=float,
        help='Logarithmic growth factor f. Only used if DensitySource=delta_2SPT.')


    cmd_args = ap.parse_args()

    #####################################
    # OPTIONS
    #####################################
    opts = OrderedDict()

    opts['code_version_for_outputs'] = '0.2'

    # Parse options
    opts['sim_seed'] = cmd_args.SimSeed
    opts['boxsize'] = cmd_args.boxsize
    basedir = os.path.expandvars('$SCRATCH/lss/ms_gadget/run4/00000%d-01536-%.1f-wig/' % (
        opts['sim_seed'], opts['boxsize']))
    opts['sim_scale_factor'] = 0.625
    opts['Rsmooth'] = cmd_args.Rsmooth
    opts['Ngrid'] = cmd_args.Ngrid
    opts['LOS'] = np.array([0,0,1])
    opts['APPLY_RSD'] = bool(cmd_args.ApplyRSD)
    opts['subsample_ratio'] = cmd_args.SubsampleRatio
    opts['max_displacement'] = cmd_args.MaxDisplacement

    # which multipoles (ell) to compute
    opts['poles'] = [0,2]

    # Source from which to compute density: 'catalog' or 'delta_2SPT'
    opts['density_source'] = cmd_args.DensitySource

    # more options if source is catalog
    if opts['density_source'] == 'catalog':
        # Catalog with particle positions: 'DM_subsample' or 'gal_ptchall_with_RSD'
        opts['positions_catalog'] = 'DM_subsample'
        # Velocity source: DM_sim, deltalin_D2, deltalin_D2_2SPT
        opts['velocity_source'] = 'DM_sim'

    elif opts['density_source'] == 'delta_2SPT':
        opts['positions_catalog'] = None
        opts['velocity_source'] = None
    else:
        raise Exception('Invalid density_source %s' % opts['density_source'])

    # cosmology of ms_gadget sims (to compute D_lin(z))
    # omega_m = 0.307494
    # omega_bh2 = 0.022300
    # omega_ch2 = 0.118800
    # h = math.sqrt((omega_bh2 + omega_ch2) / omega_m) = 0.6774
    opts['cosmo_params'] = dict(Om_m=0.307494,
                       Om_L=1.0 - 0.307494,
                       Om_K=0.0,
                       Om_r=0.0,
                       h0=0.6774)

    #opts['f_log_growth'] = 0.7862951  # np.sqrt(0.61826)
    opts['f_log_growth'] = cmd_args.fLogGrowth


    # where to save output (todo: move code to a function )
    DS_string = '_DS%s' % opts['density_source']
    if opts['density_source'] == 'catalog':
        sr_string = '_sr%g' % opts['subsample_ratio']
        vel_string = '_v%s' % opts['velocity_source']
        MD_string = '_MD%g' % opts['max_displacement']
    else:
        sr_string, vel_string, MD_string = '', '', ''

    b1, b2, bG2 = cmd_args.b1, cmd_args.b2, cmd_args.bG2
    if opts['density_source'] == 'catalog':
        b_string, f_string = '', ''
    elif opts['density_source'] == 'delta_2SPT':
        b_string = '' if (b1==1. and b2==0. and bG2==0.) else ('_b%g_%g_%g' % (b1,b2,bG2))
        f_string = '' if opts['f_log_growth']==0.786295 else ('_f%g' % opts['f_log_growth'])

    opts['outdir'] = 'data/Pskew_sims/00000%d-01536-%.1f-wig/R%.1f_Ng%d_RSD%d%s%s%s%s%s%s/' % (
        opts['sim_seed'], opts['boxsize'], opts['Rsmooth'], opts['Ngrid'],
        int(opts['APPLY_RSD']), 
        DS_string, sr_string, vel_string, MD_string, b_string, f_string)



    # ###########################
    # START PROGRAM
    # ###########################
    Nmesh = opts['Ngrid']
    BoxSize = np.array([opts['boxsize'], opts['boxsize'], opts['boxsize']])
    comm = CurrentMPIComm.get()
    LOS = opts['LOS']
    LOS_string = 'LOS%d%d%d' % (LOS[0], LOS[1], LOS[2])


    # Below, 'D' stands for RSD displacement in Mpc/h: D=v/(aH)=f*PsiDot.


    ### Catalogs

    # DM subsample
    if opts['subsample_ratio'] != 1.0:
        DM_subsample = Target(
            name='DM_subsample',
            in_fname=os.path.join(basedir, 'snap_%.4f_sub_sr%g_ssseed40%d.bigfile' % (
                opts['sim_scale_factor'], opts['subsample_ratio'], opts['sim_seed'])),
            position_column='Position'
        )
    else:
        # full DM sample
        DM_subsample = Target(
            name='DM_subsample',
            in_fname=os.path.join(basedir, 'snap_%.4f' % (
                opts['sim_scale_factor'])),
            position_column='Position'
        )


    # DM_D2 = Target(
    #     name='DM_D2',
    #     in_fname=os.path.join(basedir, 'snap_%.4f_sub_sr0.0015_ssseed40%d.bigfile' % (
    #         opts['sim_scale_factor'], opts['sim_seed'])),
    #     position_column='Position',
    #     val_column='Velocity',
    #     val_component=2,
    #     rescale_factor='RSDFactor'
    # )

    if False:
        # PT Challenge galaxies from rockstar halos. Rockstar gives core positions and velocities.
        # Units: 1/(aH) = 1./(a * H0*np.sqrt(Om_m/a**3+Om_L)) * (H0/100.) in Mpc/h / (km/s).
        # For ms_gadget, get 1/(aH) = 0.01145196 Mpc/h/(km/s) = 0.0183231*0.6250 Mpc/h/(km/s).
        # Note that MP-Gadget files have RSDFactor=1/(a^2H)=0.0183231 for a=0.6250 b/c they use a^2\dot x for Velocity.
        from lsstools.sim_galaxy_catalog_creator import PTChallengeGalaxiesFromRockstarHalos

        assert opts['sim_scale_factor'] == 0.625

        # PT challenge galaxies, apply RSD to position (TEST)
        assert opts['sim_scale_factor'] == 0.625
        gal_ptchall_with_RSD = Target(
            name='gal_ptchall_with_RSD',
            in_fname=os.path.join(basedir, 'snap_%.4f.gadget3/rockstar_out_0.list.parents.bigfile' % opts['sim_scale_factor']),
            position_column='Position',
            velocity_column='Velocity', 
            apply_RSD_to_position=True,
            RSD_los=LOS,
            RSDFactor=0.01145196,
            #val_column='Velocity', # This is rockstar velocity, which is v=a\dot x in km/s ("Velocities in km / s (physical, peculiar)")
            #val_component=0,
            #rescale_factor=0.01145196, # RSD displacement in Mpc/h is D=v/(aH)=0.01145196*v. 
            cuts=[PTChallengeGalaxiesFromRockstarHalos(
                    log10M_column='log10Mvir', log10Mmin=12.97, sigma_log10M=0.35, RSD=False),
                  #('Position', 'max', [100.,100.,20.])
                 ]
            )


    ### Densities

    # Linear density
    z_rescalefac = linear_rescale_fac(current_scale_factor=1.0,
                                      desired_scale_factor=opts['sim_scale_factor'],
                                      cosmo_params=opts['cosmo_params'])
    print('z_rescalefac:', z_rescalefac)
    deltalin = Model(
        name='deltalin',
        in_fname=os.path.join(basedir, 'IC_LinearMesh_z0_Ng%d' % opts['Ngrid']),
        rescale_factor=z_rescalefac,
        read_mode='delta from 1+delta',
        filters=None,
        readout_window='cic')

    # Linear RSD displacement
    def k2ovksq_filter_fcn(k, v, d=2):
        ksq = sum(ki**2 for ki in k)
        return np.where(ksq == 0.0, 0*v, 1j*k[d] * v / (ksq))
    deltalin_D2 = Model(
        name='deltalin_D2',
        in_fname=os.path.join(basedir, 'IC_LinearMesh_z0_Ng%d' % opts['Ngrid']),
        rescale_factor=opts['f_log_growth']*z_rescalefac,
        read_mode='delta from 1+delta',
        filters=[k2ovksq_filter_fcn],
        readout_window='cic')

    # 2nd order SPT displacement: ik/k^2*G2
    deltalin_D2_2ndorderSPTcontri = Model(
        name='deltalin_D2_2ndorderSPTcontri',
        in_fname=os.path.join(basedir, 'IC_LinearMesh_z0_Ng%d' % opts['Ngrid']),
        rescale_factor=opts['f_log_growth']*z_rescalefac,
        read_mode='delta from 1+delta',
        filters=[k2ovksq_filter_fcn],
        readout_window='cic')
        

    if opts['density_source'] == 'delta_2SPT':
        # compute 2nd order density from linear mesh
        if comm.rank == 0:
            print('Warning: Not smoothing when generating 2nd order field')

        if not opts['APPLY_RSD']:
            # Compute delta1 + F2[delta1]
            delta_mesh = FieldMesh(
                deltalin.get_mesh().compute()
                + QuadField(composite='F2').compute_from_mesh(deltalin.get_mesh()).compute(mode='real')
                )
        else:
            # compute 2nd order density in redshift space
            f = opts['f_log_growth']
            delta_mesh = FieldMesh(
                b1 * deltalin.get_mesh().compute()
                + f * LinField(m=2*LOS, n=-2).compute_from_mesh(deltalin.get_mesh()).compute()
                + b1 * QuadField(composite='F2').compute_from_mesh(deltalin.get_mesh()).compute()
                + f * QuadField(composite=('velocity_G2_par_%s' % LOS_string)).compute_from_mesh(deltalin.get_mesh()).compute(mode='real')
                + b1 * f * QuadField(nprime=-2, mprime=LOS, mprimeprime=LOS).compute_from_mesh(deltalin.get_mesh()).compute()
                + f**2 * QuadField(n=-2, m=2*LOS, nprime=-2, mprime=LOS, mprimeprime=LOS).compute_from_mesh(deltalin.get_mesh()).compute()
            )

            if b2 != 0.:
                delta_mesh = FieldMesh(
                    delta_mesh.compute()
                    + b2/2.0 * QuadField().compute_from_mesh(deltalin.get_mesh()).compute())

            if bG2 != 0.0:
                delta_mesh = FieldMesh(
                    delta_mesh.compute()
                    + bG2 * QuadField(composite='tidal_G2').compute_from_mesh(deltalin.get_mesh()).compute())



    elif opts['density_source'] == 'catalog':

        ##########################################################################
        # Get DM catalog in redshift space (if APPLY_RSD==True) 
        ##########################################################################

        # get the catalog
        if opts['positions_catalog'] == 'gal_ptchall_with_RSD':
            target = gal_ptchall_with_RSD
        elif opts['positions_catalog'] == 'DM_subsample':
            target = DM_subsample
        cat = target.get_catalog(keep_all_columns=True)
        
        if comm.rank == 0:
            print('Positions catalog:')
            print(cat.attrs)

        # add redshift space positions, assuming LOS is in z direction
        if opts['velocity_source'] == 'DM_sim':
            # use DM velocity
            displacement = cat['Velocity']*cat.attrs['RSDFactor'] * opts['LOS']
            displacement = displacement.compute()
            if opts['max_displacement'] is not None:
                ww = np.where(np.linalg.norm(displacement, axis=1)>opts['max_displacement'])[0]
                print(r'%d: max_displacement=%g. Set velocity to 0 for %g percent of particles' % (
                    comm.rank, opts['max_displacement'], 100.*float(len(ww)/displacement.shape[0])))
                displacement[ww,0] *= 0
                displacement[ww,1] *= 0
                displacement[ww,2] *= 0
            cat['RSDPosition'] = cat['Position'] + displacement


        elif opts['velocity_source'] in ['deltalin_D2', 'deltalin_D2_2SPT']:

            if opts['velocity_source'] == 'deltalin_D2':
                print('Warning: linear velocity does not have 2nd order G2 velocity, so expect wrong bispectrum')
        
            assert np.all(opts['LOS'] == np.array([0,0,1]))
            mtp = ModelTargetPair(model=deltalin_D2, target=DM_subsample)
            cat['RSDPosition'] = cat['Position'].compute()

            # read out model at each catalog position on current rank.
            model_at_target_pos = mtp.readout_model_at_target_pos()
            mat = np.zeros((model_at_target_pos.size,3))
            assert np.all(opts['LOS'] == np.array([0,0,1]))
            mat[:,2] = model_at_target_pos
            cat['RSDPosition'] += mat


            if opts['velocity_source'] == 'deltalin_D2_2SPT':
                # add 2nd order G2 contribution to velocity
                raise Exception('todo')

        else:
            raise Exception('Invalid velocity_source: %s' % str(opts['velocity_source']))
            
        if comm.rank == 0:
            print('rms RSD displacement: %g Mpc/h' % np.mean((cat['Position'].compute()-cat['RSDPosition'].compute())**2)**0.5)
            print('max RSD displacement: %g Mpc/h' % np.max(np.abs(cat['Position'].compute()-cat['RSDPosition'].compute())))
        

        # Get redshift space catalog
        RSDcat = catalog_persist(cat, columns=['ID','PID','Position','RSDPosition','Velocity', 'log10Mvir'])
        del cat
        if opts['APPLY_RSD']:
            if comm.rank == 0:
                print('Applying RSD')
            RSDcat['Position'] = RSDcat['RSDPosition']
        else:
            if comm.rank == 0:
                print('Not applying RSD')

        # paint to mesh
        delta_mesh = FieldMesh(RSDcat.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, 
                    window='cic', interlaced=False, compensated=False).compute()-1)

        if comm.rank == 0:
           print('# objects: Original: %d' % (RSDcat.csize))


    else:
        raise Exception('Invalid density_source %s' % opts['density_source'])



    ##########################################################################
    # Calculate power spectrum multipoles 
    ##########################################################################
    def calc_power(mesh, second=None, mode='1d', k_bin_width=1.0, verbose=False, los=None, poles=None):
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
        elif mode == '2d':
            if poles is None:
                poles = [0,2,4]
            res = FFTPower(first=mesh,
                                second=second,
                                mode=mode,
                                dk=dk,
                                kmin=kmin,
                                poles=poles,
                                Nmu=5,
                                los=los)
        else:
            raise Exception("Mode not implemented: %s" % mode)

        return res


    ## Compute density power spectrum 
    Pdd = calc_power(delta_mesh, los=opts['LOS'], mode='2d', poles=opts['poles'])


    ##########################################################################
    # Get RSD skew spectra
    ##########################################################################
    power_kwargs={'mode': '2d', 'poles': opts['poles']}

    # Apply smoothing
    smoothers = [smoothing.GaussianSmoother(R=opts['Rsmooth'])]
    delta_mesh_smoothed = FieldMesh(delta_mesh.compute(mode='real'))
    for smoother in smoothers:
        delta_mesh_smoothed = smoother.apply_smoothing(delta_mesh_smoothed)

    # Compute skew spectra
    skew_spectra = SkewSpectrum.get_list_of_standard_skew_spectra(
        LOS=LOS, redshift_space_spectra=True)
    for skew_spec in skew_spectra:
        # compute and store in skew_spec.Pskew
        skew_spec.compute_from_mesh(mesh=delta_mesh_smoothed, 
            third_mesh=delta_mesh, power_kwargs=power_kwargs,
            store_key='default_key')


    # Store results in files
    if comm.rank == 0:
        if not os.path.exists(opts['outdir']):
            os.makedirs(opts['outdir'])

        if False:
            for skew_spec in skew_spectra:
                # store as json
                fname = os.path.join(opts['outdir'], skew_spec.name+'.json')
                skew_spec.Pskew.save(fname)
                print('Wrote %s' % fname)

                # store as plain text
                fname = os.path.join(opts['outdir'], skew_spec.name+'.txt')
                skew_spec.save_plaintext(fname)
                print('Wrote %s' % fname)
            
        # store all in one file for each multipole
        if skew_spectra != []:
            for ell in skew_spec.Pskew['default_key'].attrs['poles']:
                mydtype = [('k', 'f8')]
                for skew_spec in skew_spectra:
                    mydtype.append((skew_spec.name, 'f8'))
                arr = np.empty(shape=skew_spec.Pskew['default_key'].poles['k'].shape, dtype=mydtype)
                arr['k'] = skew_spec.Pskew['default_key'].poles['k']
                for skew_spec in skew_spectra:
                    arr[skew_spec.name] = skew_spec.Pskew['default_key'].poles['power_%d'%ell].real
                fname = os.path.join(opts['outdir'], 'Sn_ell%d.txt'%ell)
                header = 'Columns: ' + str(arr.dtype.names)
                np.savetxt(fname, arr, header=header)
                print('Wrote %s' % fname)

        # also store density power
        for ell in Pdd.attrs['poles']:
            mydtype = [('k', 'f8'), ('P', 'f8')]
            arr = np.empty(shape=Pdd.poles['k'].shape, dtype=mydtype)
            arr['k'] = Pdd.poles['k']
            arr['P'] = Pdd.poles['power_%d'%ell].real
            fname = os.path.join(opts['outdir'], 'P_ell%d.txt'%ell)
            header = 'Columns: ' + str(arr.dtype.names)
            np.savetxt(fname, arr, header=header)
            print('Wrote %s' % fname)




if __name__ == '__main__':
    main()
