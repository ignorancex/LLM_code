from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
from nbodykit.lab import *
import numpy as np
import os

from lsstools.nbkit03_utils import calc_divergence_of_3_meshs
from perr.shift import weigh_and_shift_uni_cats

def main():
    """
    Do the following: 
    - create uniform catalog
    - load deltalin or other density delta on grid
    - weigh ptcles in uniform catalog by 1+delta
    - compute smoothed Psi_lin on grid (interpolating to ptcle positions)
    - shift (displace) particles by Psi
    - save shifted catalog

    Example:
    ./run.sh mpiexec -n 3 python main_shift_catalog_by_Psi_grid.py 64 64

    Should make sure that PYTHONPATH="".
    And load environment with nbodykit 0.3, e.g. using run.sh script.
    """

    # command line args
    ap = ArgumentParser()
    ap.add_argument('Nptcles_per_dim',
                    type=int,
                    default=0,
                    help='Number of particles per dimension used in internal'
                    'catalog')
    ap.add_argument('out_Ngrid',
                    type=int,
                    default=0,
                    help='Ngrid for output density')

    # optional arguments
    ap.add_argument('--Nmesh',
                    type=int,
                    default=0,
                    help='Internal Nmesh used to compute density to be shifted '
                    ' and displacement. Should be >= Nptcles_per_dim.')

    ap.add_argument('--SimSeed',
                    type=int,
                    default=400,
                    help='Simulation seed to load.')

    ap.add_argument('--PsiOrder',
                    type=str,
                    default="1",
                    help='Order of displacement field (1 for Zeldovich, 2 for '
                         '2LPT displacement).')
    ap.add_argument('-v',
                    '--verbose',
                    action='store_const',
                    dest='verbose',
                    const=1,
                    default=0,
                    help='run in verbose mode, with increased logging output')
    
    ap.add_argument('--RSD',
                    type=int,
                    default=0,
                    help='0: No RSD. 1: Include RSD.')

    help = 'Smoothing applied to the field to be shifted in Lagrangian space.'
    ap.add_argument('--RDensityToShift',
                    type=float,
                    default=0.0,
                    help=help)

    ap.add_argument('--RDisplacementSource',
                    type=float,
                    default=0.23,
                    help='Smoothing applied to the displacement source.')


    # copy args
    cmd_args = ap.parse_args()
    opts = OrderedDict()
    opts['Nptcles_per_dim'] = cmd_args.Nptcles_per_dim
    opts['out_Ngrid'] = cmd_args.out_Ngrid
    #opts['plot_slices'] = cmd_args.plot_slices
    sim_seed = cmd_args.SimSeed

    if cmd_args.PsiOrder in ['1','2','3','-3']:
        opts['PsiOrder'] = int(cmd_args.PsiOrder)
    else:
        opts['PsiOrder'] = cmd_args.PsiOrder

    opts['RSD'] = bool(cmd_args.RSD)
    opts['RSD_line_of_sight'] = [0, 0, 1]
    Rsmooth_density_to_shift = cmd_args.RDensityToShift
    Rsmooth_displacement_source = cmd_args.RDisplacementSource



    opts['verbose'] = cmd_args.verbose
    if cmd_args.Nmesh == 0:
        opts['Nmesh'] = opts['Nptcles_per_dim']
    else:
        opts['Nmesh'] = cmd_args.Nmesh

    # ################################################################################
    # OPTIONS (not from command line)
    # ################################################################################

    # plot slices
    opts['plot_slices'] = False
    if opts['Nptcles_per_dim'] >= 512:
        opts['plot_slices'] = False

    if True:
        # ms_gadget L=500 or L=1500 sim
        sim_name= 'ms_gadget'
        opts['boxsize'] = 500.
        opts['basepath'] = os.path.expandvars(
            '$SCRATCH/lss/ms_gadget/run4/00000%d-01536-%.1f-wig/' %
            (sim_seed, opts['boxsize']))

        # Get deltalin at internal_scale_factor_for_weights, shift it by Psi(out_scale_factor),
        # and then rescale the result from internal_scale_factor_for_weights to out_scale_factor.
        # (We must use Psi at out_scale_factor to get correct displacement, which is larger at lower z.)
        opts['out_scale_factor'] = 0.6250
        deltalin_file_name = os.path.join(opts['basepath'],
                         'IC_LinearMesh_z0_Ng%d/' % opts['Nmesh'])
        deltalin_file_scale_factor = 1.0


    if False:
        # IllustrisTNG L=205 sim
        sim_name = 'IllustrisTNG_L205n2500TNG'
        opts['boxsize'] = 205.0
        opts['basepath'] = os.path.expandvars(
            '$DATA/lss/IllustrisTNG/L%dn2500TNG/output/' % int(opts['boxsize']))
        #opts['out_scale_factor'] = 0.66531496 # snap 67
        opts['out_scale_factor'] = 0.33310814 # snap 33
        # deltalin
        deltalin_file_name = os.path.join(opts['basepath'],
                         'snap_ics.hdf5_PtcleDensity_z127_Ng%d/' % opts['Nmesh'])
        deltalin_file_scale_factor = 1.0/(1.0+127.0)


    # Internal scale factor used for 1+delta weights. Initially used high z for that, but it should
    # be same as out_scale_factor b/c otherwise weights will be too small to matter (would get
    # almost the same as when shifting from uniform grid without weights).
    opts['internal_scale_factor_for_weights'] = opts['out_scale_factor']

    # Mode for painting shifted particles (with masses given by density) to grid
    # - 'avg': average contributions (divide by number of CIC contributions)
    # - 'sum': sum up contributions (like Zeldovich; increases density if ptcles move to same location;
    #          accounts for volume factor along the flow/Jacobian enforcing mass conservation)
    # Use 'avg' to get shifted PsiDot, and 'sum' to get shifted bias params.
    opts['weighted_CIC_mode'] = 'sum'

    # kmax for internal grid (Nptcles_per_dim grid points per dim)
    #opts['kmax'] = 2.0*np.pi/opts['boxsize'] * float(opts['Nptcles_per_dim'])/2.0


    opts['densities_to_shift'] = []


    ## RSD displacement source.
    # Used for RSD_method=PotentialOfRSDDisplSource_xPtcles.
    # Not used for RSD_method=LPT.

    if False:
        # Use k/k^2*delta_ZA for RSD displacement if RSD_method=PotentialOfRSDDisplSource_xPtcles.
        # (Code multiplies by f internally)
        opts['RSD_displacement_source'] = {
                'id_for_out_fname':
                'PotdZx',
                'Psi_type': 'Zeldovich', # do k/k^2
                'in_fname': os.path.join(opts['basepath'], 
                    '1_intR0.00_extR0.00_SHIFTEDBY_IC_LinearMeshR0.23_a%.4f_Np%d_Nm%d_Ng%d_CICsum' % (
                    opts['out_scale_factor'], opts['Nptcles_per_dim'],
                    opts['Nmesh'], opts['out_Ngrid'])),
                'file_scale_factor': opts['out_scale_factor'],
                'smoothing': {
                    'mode': 'Gaussian',
                    'R': Rsmooth_displacement_source
                }
            }

    if False:
        # Use k/k^2*theta_sims for RSD displacement if RSD_method=PotentialOfRSDDisplSource_xPtcles.
        # (Code multiplies by f internally)
        sim_mass_string = 'log10M_10.8_11.8'
        opts['RSD_displacement_source'] = {
                'id_for_out_fname':
                'Potthetasimx_%s' % sim_mass_string,
                'Psi_type': 'Zeldovich', # do k/k^2
                'in_fname': os.path.join(opts['basepath'], 
                    'nbkit_fof_0.6250/ll_0.200_nmin25/fof_nbkfmt.hdf5_BOUNDS_%s.hdf5_thetav' % sim_mass_string),
                'file_scale_factor': opts['out_scale_factor'],
                'smoothing': {
                    'mode': 'Gaussian',
                    'R': Rsmooth_displacement_source
                }
            }





    ## RSD_method: 
    # - 'LPT': Include RSD using standard LPT integral, displacing
    # by Psi(q)+ (e_LOS.v(q)) e_LOS, where e_LOS is unit vector along
    # line of sight direction, and v(q)=\dot\vPsi(q) is time  derivative
    # of displacement of order ``PsiOrder``. This is standard LPT in
    # redshift space. 
    # - 'PotentialOfRSDDisplSource_xPtcles': Include RSD by moving by
    # \vk/k^2 deltaZA(k) along the LOS, i.e. move by the velocity one
    # would get if true Eulerian density were deltaZA. Do this by generating
    # new uniform catalog in x space weighted by shifted operator \tilde O(x),
    # then shit those x particles by \vk/k^2 deltaZA(k). In general, use
    # RSD_displacement_source instead of deltaZA.
    
    for RSD_method in [
        'LPT',
        #'PotentialOfRSDDisplSource_xPtcles'
    ]:

        ## densities to shift

        if True:
            # shift the field 1 (this gives delta_ZA, delta_2LPT, etc)
            opts['densities_to_shift'].append({
                'id_for_out_fname':
                '1',
                'RSD_method': RSD_method,
                'in_fname': deltalin_file_name,
                'file_scale_factor': deltalin_file_scale_factor,
                'external_smoothing': {
                    'mode': 'Gaussian',
                    'R': Rsmooth_density_to_shift
                },
                'calc_trf_of_field':
                '1'
            })

        if True:
            # Shift linear density, quadratic operators and delta^3

            if True:
                # shift delta_lin
                opts['densities_to_shift'].append({
                    'id_for_out_fname':
                    'IC_LinearMesh',
                    'RSD_method': RSD_method,
                    'in_fname': deltalin_file_name,
                    'file_scale_factor': deltalin_file_scale_factor,
                    'external_smoothing':
                    None
                })
            if True:
                # shift delta_lin^2-<delta_lin^2>
                opts['densities_to_shift'].append({
                    'id_for_out_fname':
                    'IC_LinearMesh_growth-mean',
                    'RSD_method': RSD_method,
                    'in_fname': deltalin_file_name,
                    'file_scale_factor': deltalin_file_scale_factor,
                    'external_smoothing':
                    None,  # external smoothingn of delta^2(x)
                    'smoothing_quadratic_source': {
                        'mode': 'Gaussian',
                        'R': 0.0
                    },
                    'calc_quadratic_field':
                    'growth-mean'
                })
            if True:
                # shift G2[delta_lin]
                opts['densities_to_shift'].append({
                    'id_for_out_fname':
                    'IC_LinearMesh_tidal_G2',
                    'RSD_method': RSD_method,
                    'in_fname': deltalin_file_name,
                    'file_scale_factor': deltalin_file_scale_factor,
                    'external_smoothing':
                    None,  # external smoothingn of delta^2(x)
                    'smoothing_quadratic_source': {
                        'mode': 'Gaussian',
                        'R': 0.0
                    },  # 'kmax': opts['kmax']},
                    'calc_quadratic_field':
                    'tidal_G2'
                })

            if True:
                # shift the shift term Psi.nabla delta
                opts['densities_to_shift'].append({
                    'id_for_out_fname':
                    'IC_LinearMesh_PsiNablaDelta',
                    'RSD_method': RSD_method,
                    'in_fname': deltalin_file_name,
                    'file_scale_factor': deltalin_file_scale_factor,
                    'external_smoothing':
                    None,  # external smoothingn of delta^2(x)
                    'smoothing_quadratic_source': {
                        'mode': 'Gaussian',
                        'R': 0.0
                    },  # 'kmax': opts['kmax']},
                    'calc_quadratic_field':
                    'PsiNablaDelta'
                })

            if True:
                # shift deltalin^3
                opts['densities_to_shift'].append({
                    'id_for_out_fname':
                    'IC_LinearMesh_cube-mean',
                    'RSD_method': RSD_method,
                    'in_fname': deltalin_file_name,
                    'file_scale_factor': deltalin_file_scale_factor,
                    'external_smoothing':
                    None,  # external smoothingn of delta^2(x)
                    #'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0},
                    'smoothing_quadratic_source': {
                        'mode': 'Gaussian',
                        'R': 0.0,
                        'kmax': 0.5
                    },
                    'calc_quadratic_field':
                    'cube-mean'
                })

            if False:
                # shift the field delta_short = (1-W_R)delta
                # todo: implement
                opts['densities_to_shift'].append({
                    'id_for_out_fname':
                    'IC_LinearMesh_short',
                    'RSD_method': RSD_method,
                    'in_fname': deltalin_file_name,
                    'file_scale_factor': deltalin_file_scale_factor,
                    'external_smoothing': {
                        'mode': '1-Gaussian',
                        'R': 10.0
                    },
                })



        if True:
            #### SHIFT NEW RSD OPERATORS

            if True:
                # shift G2_parallel[delta_lin]
                LOS_string = '%d%d%d' % (
                    opts['RSD_line_of_sight'][0], opts['RSD_line_of_sight'][1], 
                    opts['RSD_line_of_sight'][2])
                opts['densities_to_shift'].append({
                    'id_for_out_fname':
                    'IC_LinearMesh_tidal_G2_par_LOS%s'%LOS_string,
                    'RSD_method': RSD_method,
                    'in_fname': deltalin_file_name,
                    'file_scale_factor': deltalin_file_scale_factor,
                    'external_smoothing':
                    None,  # external smoothingn of delta^2(x)
                    'smoothing_quadratic_source': {
                        'mode': 'Gaussian',
                        'R': 0.0
                    },  # 'kmax': opts['kmax']},
                    'calc_quadratic_field':
                    'tidal_G2_par_LOS%s' % LOS_string
                })


        if False:
            #### SHIFT NEW CUBIC OPERATORS G3, Gamma3, etc

            # TODO: maybe apply more aggressive smoothing to avoid unwanted Dirac
            # delta images.

            if True:
                # shift G2*delta
                opts['densities_to_shift'].append({
                    'id_for_out_fname':
                    'IC_LinearMesh_G2_delta',
                    'RSD_method': RSD_method,
                    'in_fname': deltalin_file_name,
                    'file_scale_factor': deltalin_file_scale_factor,
                    'external_smoothing':
                    None,  # external smoothingn of delta^2(x)
                    #'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0},
                    'smoothing_quadratic_source': {
                        'mode': 'Gaussian',
                        'R': 0.0,
                        'kmax': 0.5
                    },
                    'calc_quadratic_field':
                    'G2_delta'
                })


            if True:
                # shift G3
                opts['densities_to_shift'].append({
                    'id_for_out_fname':
                    'IC_LinearMesh_tidal_G3',
                    'RSD_method': RSD_method,
                    'in_fname': deltalin_file_name,
                    'file_scale_factor': deltalin_file_scale_factor,
                    'external_smoothing':
                    None,  # external smoothingn of delta^2(x)
                    #'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0},
                    'smoothing_quadratic_source': {
                        'mode': 'Gaussian',
                        'R': 0.0,
                        'kmax': 0.5
                    },
                    'calc_quadratic_field':
                    'tidal_G3'
                })

            if True:
                # shift Gamma3
                opts['densities_to_shift'].append({
                    'id_for_out_fname':
                    'IC_LinearMesh_Gamma3',
                    'RSD_method': RSD_method,
                    'in_fname': deltalin_file_name,
                    'file_scale_factor': deltalin_file_scale_factor,
                    'external_smoothing':
                    None,  # external smoothingn of delta^2(x)
                    #'smoothing_quadratic_source': {'mode': 'Gaussian', 'R': 0.0},
                    'smoothing_quadratic_source': {
                        'mode': 'Gaussian',
                        'R': 0.0,
                        'kmax': 0.5
                    },
                    'calc_quadratic_field':
                    'Gamma3'
                })



        if False:
            # Shift nth order Psidot.
            # Can optionally compute divergence, see below.
            # Should use CIC avg instead of CIC sum mode.

            # We shift \dot Psi^{[n_max]} = sum_{n=1}^{n_max} \dot Psi^{(n)}, i.e.
            # the displacement up to order n_max.

            if True:
                # Shift nth order displacement by ZA
                for order in [1]:
                    for direction in [0,1,2]:
                        opts['densities_to_shift'].append({
                            'id_for_out_fname':
                            'IC_LinearMesh_PsiDot%d_%d' % (
                                order, direction),
                            'RSD_method': RSD_method,
                            'in_fname': deltalin_file_name,
                            'file_scale_factor': deltalin_file_scale_factor,
                            'external_smoothing': {
                                'mode': 'Gaussian',
                                'R': Rsmooth_density_to_shift
                            },  # external smoothingn of PsiDot
                            'smoothing_quadratic_source': {
                                'mode': 'Gaussian',
                                'R': 0.0
                            },
                            'calc_quadratic_field':
                            'PsiDot%d_%d' % (order, direction),
                            'return_mesh': True
                        })


    ## Displacement field by which operators are shifted.
    # Note: MP-Gadget libgenic/zeldovich.c applies smoothing by a gaussian kernel of 1 mesh grid
    # (i.e. R=boxsize/Nmesh*sqrt(2) b/c they use Gaussian w/o 0.5 in exponent), and they might do read-out
    # of displacements to ptcle positions using nearest neighbor instead of CIC we use. For L=500,
    # Np=1536, get Nmesh=3072 and thus R=500/3072*1.42==0.23
    if opts['PsiOrder'] == 1:
        # displace by Psi_ZA[delta_lin] = ik/k^2 delta_lin
        opts['displacement_source'] = {
            'id_for_out_fname':
            'IC_LinearMesh',
            'Psi_type':
            'Zeldovich',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'smoothing': {
                'mode': 'Gaussian',
                'R': Rsmooth_displacement_source
            }
        }
    elif opts['PsiOrder'] == 2:
        # displace by Psi_2LPT[delta_lin] = ik/k^2 delta_lin(k) - 3/14 ik/k^2 
        #                                   G2[delta_lin](k)
        opts['displacement_source'] = {
            'id_for_out_fname':
            'Psi2LPT_IC_LinearMesh',
            'Psi_type':
            '2LPT',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'smoothing': {
                'mode': 'Gaussian',
                'R': Rsmooth_displacement_source
            }
        }
    # elif opts['PsiOrder'] == -2:
    #     # just to check sign of Psi^[2] is correct
    #     # displace by -Psi_2LPT[delta_lin] = ik/k^2 delta_lin(k) + 3/14 ik/k^2
    #                                          G2[delta_lin](k)
    #     opts['displacement_source'] = {
    #         'id_for_out_fname': '-Psi2LPT_IC_LinearMesh',
    #         'Psi_type': '-2LPT',
    #        'in_fname': deltalin_file_name,
    #        'file_scale_factor': deltalin_file_scale_factor,
    #         'smoothing': {'mode': 'Gaussian', 'R': 0.23}
    #         }
    elif opts['PsiOrder'] == 3:
        # displace by Psi_3LPT[delta_lin]
        opts['displacement_source'] = {
            'id_for_out_fname':
            'Psi3LPT_IC_LinearMesh',
            'Psi_type':
            '3LPT',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'smoothing': {
                'mode': 'Gaussian',
                'R': Rsmooth_displacement_source
            },
            'smoothing_Psi3LPT': {
                'mode': 'Gaussian',
                'R': Rsmooth_displacement_source,
                'kmax': 0.5
            }
        }
    elif opts['PsiOrder'] == -3:
        # displace by -Psi_3LPT[delta_lin]
        opts['displacement_source'] = {
            'id_for_out_fname':
            'Psi-3LPT_IC_LinearMesh',
            'Psi_type':
            '-3LPT',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'smoothing': {
                'mode': 'Gaussian',
                'R': Rsmooth_displacement_source
            },
            'smoothing_Psi3LPT': {
                'mode': 'Gaussian',
                'R': Rsmooth_displacement_source,
                'kmax': 0.5
            }
        }

    elif opts['PsiOrder'] == '3_v2':
        # displace by Psi_3LPT[delta_lin] with sign of G3 switched.
        # gives correct power on large scales, but still no improvement over 2LPT...
        opts['displacement_source'] = {
            'id_for_out_fname':
            'Psi3LPT_v2_IC_LinearMesh',
            'Psi_type':
            '3LPT_v2',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'smoothing': {
                'mode': 'Gaussian',
                'R': Rsmooth_displacement_source
            },
            'smoothing_Psi3LPT': {
                'mode': 'Gaussian',
                'R': Rsmooth_displacement_source,
                'kmax': 0.1 # 24 april 2020: When comparing with DM, 0.5 better than None, using Np=512,
                #but always worse than rcc(2LPT,DM), so probably 3lpt kernel wrong..
            }
        }

    elif opts['PsiOrder'] == '3_v3':
        # displace by Psi_3LPT[delta_lin] with sign of G3 switched
        opts['displacement_source'] = {
            'id_for_out_fname':
            'Psi3LPT_v3_IC_LinearMesh',
            'Psi_type':
            '3LPT_v3',
            'in_fname': deltalin_file_name,
            'file_scale_factor': deltalin_file_scale_factor,
            'smoothing': {
                'mode': 'Gaussian',
                'R': Rsmooth_displacement_source
            },
            'smoothing_Psi3LPT': {
                'mode': 'Gaussian',
                'R': Rsmooth_displacement_source,
                'kmax': 0.5
            }
        }


    else:
        raise Exception("Invalid PsiOrder %s" % str(opts['PsiOrder']))


    # ADDITIONALLY COMPUTE DIVERGENCE of shifted fields.
    # Entries are 3-tuples containing the fields whose divergence to compute.
    compute_divergence_of_shifted_fields = []

    if False:
        # compute divergence of PsiDot1
        compute_divergence_of_shifted_fields.append(
            ('IC_LinearMesh_PsiDot2_0', 'IC_LinearMesh_PsiDot2_1',
                'IC_LinearMesh_PsiDot2_2')
            )
    

    # COSMOLOGY OPTIONS
    if sim_name in ['jerryou_baoshift', 'ms_gadget']:
        opts['cosmo_params'] = dict(Om_m=0.307494,
                                    Om_L=1.0 - 0.307494,
                                    Om_K=0.0,
                                    Om_r=0.0,
                                    h0=0.6774)

    elif sim_name in ['IllustrisTNG_L205n2500TNG']:
        opts['cosmo_params'] = dict(Om_m=0.3089,
                                    Om_L=0.6911,
                                    Om_K=0.0,
                                    Om_r=0.0,
                                    h0=0.6774)

    # save result to bigfile on disk
    opts['save_result'] = True

    opts['Nmesh_orig'] = opts['Nmesh']
    del opts['Nmesh']


    # ##########################################################################
    # START PROGRAM
    # ##########################################################################

    # compute shifted densities and save to disk
    outmesh_dict, outfiles_dict = weigh_and_shift_uni_cats(**opts)


    # compute divergence of fields (to get velocity divergence)
    for div_field_tuple in compute_divergence_of_shifted_fields:
        # compute divergence of the three fields in div_field_tuple
        for i in [0,1,2]:
            if div_field_tuple[i] not in outmesh_dict.keys():
                raise Exception('Cannot take divergence of %s' % div_field_tuple[i])

        div_mesh = calc_divergence_of_3_meshs(
            (outmesh_dict[div_field_tuple[0]],
             outmesh_dict[div_field_tuple[1]],
             outmesh_dict[div_field_tuple[2]]))

        # save to disk
        head, tail = os.path.split(outfiles_dict[div_field_tuple[0]])
        div_fname = os.path.join(head, 'div_' + tail)
        div_mesh.save(div_fname, mode='real')
        if div_mesh.comm.rank == 0:
            print('Wrote %s' % div_fname)


if __name__ == '__main__':
    main()
