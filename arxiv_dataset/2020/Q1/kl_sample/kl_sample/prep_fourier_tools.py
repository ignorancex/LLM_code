"""

This module contains the tools to prepare
data in fourier space.

Functions:
 - get_map(w, mask, cat)

"""

import sys
import os
import re
import numpy as np
import pymaster as nmt
import kl_sample.io as io


def get_map(w, mask, cat, pos_in=None):
    """ Generate a map from a catalogue, a mask
        and a WCS object.

    Args:
        w: WCS object.
        mask: array with mask.
        cat: catalogue of objects.
        pos_in: to save cpu time it is possible to provide pixel positions.

    Returns:
        map_1, map_2: array with maps for each polarization.
        pos: pixel positions.

    """

    # Create arrays for the two shears
    map_1 = np.zeros(mask.shape)
    map_2 = np.zeros(mask.shape)

    # Get World position of each galaxy
    if pos_in is None:
        pos = np.vstack(
            (cat['ALPHA_J2000'], cat['DELTA_J2000'])).T
        # Calculate Pixel position of each galaxy
        pos = w.wcs_world2pix(pos, 0).astype(int)
        pos = np.flip(pos, axis=1)  # Need to invert the columns
    else:
        pos = pos_in.copy()

    # Perform lex sort and get the sorted indices
    sorted_idx = np.lexsort(pos.T)
    sorted_pos = pos[sorted_idx, :]
    # Differentiation along rows for sorted array
    diff_pos = np.diff(sorted_pos, axis=0)
    diff_pos = np.append([True], np.any(diff_pos != 0, 1), 0)
    # Get unique sorted labels
    sorted_labels = diff_pos.cumsum(0)-1
    # Get labels
    labels = np.zeros_like(sorted_idx)
    labels[sorted_idx] = sorted_labels
    # Get unique indices
    unq_idx = sorted_idx[diff_pos]
    # Get unique pos's and ellipticities
    pos_unique = pos[unq_idx, :]
    w_at_pos = np.bincount(labels, weights=cat['weight'])
    g1_at_pos = np.bincount(labels, weights=cat['e1']*cat['weight'])/w_at_pos
    g2_at_pos = np.bincount(labels, weights=cat['e2']*cat['weight'])/w_at_pos
    # Create the maps
    map_1[pos_unique[:, 0], pos_unique[:, 1]] = g1_at_pos
    map_2[pos_unique[:, 0], pos_unique[:, 1]] = g2_at_pos

    # empty = 1.-np.array(
    #     [mask[tuple(x)] for x in pos_unique]).sum()/mask.flatten().sum()
    # print('----> Empty pixels: {0:5.2%}'.format(empty))
    # sys.stdout.flush()

    return np.array([map_1, map_2]), pos


def get_cl(field, bp, hd, mask, map, coupled_cell, tmp_path=None):
    """ Generate cl's from a mask and a map.

    Args:
        field: field.
        bp: bandpowers for ell.
        hd: header with infos about the mask and maps.
        mask: array with mask.
        coupled_cell: do not decouple cls with mcm (but calculated and stored)
        map: maps for each bin and polarization.

    Returns:
        cl: array with cl (E/B, ell, bins).
        mcm_path: path to the mode coupling matrix.

    """

    # Initialize Cls
    n_bins = map.shape[1]
    n_ells = len(bp)
    cl = np.zeros((2, 2, n_ells, n_bins, n_bins))

    # Dimensions
    Nx = hd['NAXIS1']
    Ny = hd['NAXIS2']
    Lx = Nx*abs(hd['CDELT1'])*np.pi/180  # Mask dimension in radians
    Ly = Ny*abs(hd['CDELT2'])*np.pi/180  # Mask dimension in radians

    # Fields definition
    fd = np.array([nmt.NmtFieldFlat(
        Lx, Ly, mask[x], [map[0, x], -map[1, x]]) for x in range(n_bins)])
    # Bins for flat sky fields
    b = nmt.NmtBinFlat(bp[:, 0], bp[:, 1])
    # Effective ells
    ell = b.get_effective_ells()

    # Iterate over redshift bins to compute Cl's
    mcm_paths = []
    for nb1 in range(n_bins):
        for nb2 in range(nb1, n_bins):
            # Temporary path for mode coupling matrix
            if tmp_path is None:
                mcm_p = os.path.expanduser('~')
            else:
                mcm_p = tmp_path
            mcm_p = mcm_p+'/mcm_{}_Z{}{}.dat'.format(field, nb1+1, nb2+1)
            mcm_paths.append(mcm_p)
            # Define workspace for mode coupling matrix
            wf = nmt.NmtWorkspaceFlat()
            try:
                wf.read_from(mcm_p)
            except RuntimeError:
                wf.compute_coupling_matrix(fd[nb1], fd[nb2], b)
                wf.write_to(mcm_p)
                print('Calculated mode coupling matrix for bins {}{}'
                      ''.format(nb1+1, nb2+1))
                sys.stdout.flush()
            # Calculate Cl's
            cl_c = nmt.compute_coupled_cell_flat(fd[nb1], fd[nb2], b)
            if coupled_cell is True:
                cl_d = cl_c
            else:
                cl_d = wf.decouple_cell(cl_c)
            cl_d = np.reshape(cl_d, (2, 2, n_ells))
            cl[:, :, :, nb1, nb2] = cl_d
            cl[:, :, :, nb2, nb1] = cl_d

    return ell, cl, mcm_paths


def get_io_paths(args, fields):
    """ Get paths for input and output.

    Args:
        args: the arguments read by the parser.
        fields: list of the observed fields.

    Returns:
        path: dictionary with all the necessary paths.

    """

    # Define local variables
    path = {}
    create = io.path_exists_or_create
    join = os.path.join

    path['input'] = os.path.abspath(args.input_path)
    io.path_exists_or_error(path['input'])
    if args.output_path:
        path['output'] = create(os.path.abspath(args.output_path))
    else:
        path['output'] = create(join(path['input'], 'output'))
    if args.badfields_path:
        path['badfields'] = create(os.path.abspath(args.badfields_path))
    else:
        path['badfields'] = create(join(path['input'], 'badfields'))
    path['final'] = join(path['output'], 'data_fourier.fits')
    if args.want_plots:
        path['plots'] = create(join(path['output'], 'plots'))
    path['cat_full'] = join(path['input'], 'cat_full.fits')
    path['mask_url'] = join(path['input'], 'mask_url.txt')
    path['photo_z'] = join(path['output'], 'photo_z.fits')
    path['mcm'] = create(join(path['output'], 'mcm'))
    if args.cat_sims_path:
        path['cat_sims'] = create(os.path.abspath(args.cat_sims_path))
    else:
        path['cat_sims'] = join(path['output'], 'cat_sims')
    for f in fields:
        path['mask_sec_'+f] = \
            join(path['input'], 'mask_arcsec_{}.fits.gz'.format(f))
        path['mask_'+f] = join(path['output'], 'mask_{}.fits'.format(f))
        path['m_'+f] = join(path['output'], 'mult_corr_{}.fits'.format(f))
        path['cat_'+f] = join(path['output'], 'cat_{}.fits'.format(f))
        path['map_'+f] = join(path['output'], 'map_{}.fits'.format(f))
        path['cl_'+f] = join(path['output'], 'cl_{}.fits'.format(f))
        path['cl_sims_'+f] = join(path['output'], 'cl_sims_{}.fits'.format(f))

    return path


def is_run_and_check(args, fields, path, n_sims_cov):
    """ Determine which modules to run and do some preliminary check.

    Args:
        args: the arguments read by the parser.
        fields: list of the observed fields.
        path: dictionary with all the necessary paths.

    Returns:
        is_run: dictionary with the modules to be run.
        warning: true if it generated some warning.

    """

    # Local variables
    warning = False
    is_run = {}
    ex = os.path.exists

    # Determine which modules have to be run, by checking the existence of the
    # output files and arguments passed by the user
    is_run['mask'] = np.array([not(ex(path['mask_'+f])) for f in fields]).any()
    if args.run_mask or args.run_all:
        is_run['mask'] = True
    is_run['mult'] = np.array([not(ex(path['m_'+f])) for f in fields]).any()
    if args.run_mult or args.run_all:
        is_run['mult'] = True
    is_run['pz'] = not(ex(path['photo_z']))
    if args.run_pz or args.run_all:
        is_run['pz'] = True
    is_run['cat'] = np.array([not(ex(path['cat_'+f])) for f in fields]).any()
    if args.run_cat or args.run_all:
        is_run['cat'] = True
    is_run['map'] = np.array([not(ex(path['map_'+f])) for f in fields]).any()
    if args.run_map or args.run_all:
        is_run['map'] = True
    is_run['cl'] = np.array([not(ex(path['cl_'+f])) for f in fields]).any()
    if args.run_cl or args.run_all:
        is_run['cl'] = True
    is_run['cat_sims'] = not(ex(path['cat_sims']))
    # Check that all the files are present as well
    if ex(path['cat_sims']):
        is_files = True
        for f in fields:
            sims = [x for x in os.listdir(path['cat_sims'])
                    if re.match('.+{}\.fits'.format(f), x)]  # noqa:W605
            if len(sims) != n_sims_cov:
                is_files = False
        is_run['cat_sims'] = not(is_files)
    if args.run_cat_sims or args.run_all:
        is_run['cat_sims'] = True
    is_run['cl_sims'] = \
        np.array([not(ex(path['cl_sims_'+f])) for f in fields]).any()
    if args.run_cl_sims or args.run_all:
        is_run['cl_sims'] = True

    # Check the existence of the required input files
    if is_run['mask']:
        nofile1 = not(ex(path['mask_url']))
        nofile2 = \
            np.array([not(ex(path['mask_sec_'+f])) for f in fields]).any()
        nofile3 = not(ex(path['cat_full']))
        if nofile1 or nofile2 or nofile3:
            print(
                'WARNING: I will skip the MASK module. Input files not found!')
            is_run['mask'] = False
            warning = True
    else:
        print('I will skip the MASK module. Output files already there!')
        sys.stdout.flush()
    if is_run['mult']:
        nofile1 = np.array([not(ex(path['mask_'+f])) for f in fields]).any()
        nofile2 = not(ex(path['cat_full']))
        if (not(is_run['mask']) and nofile1) or nofile2:
            print('WARNING: I will skip the MULT_CORR module. Input file not '
                  'found!')
            sys.stdout.flush()
            is_run['mult'] = False
            warning = True
    else:
        print('I will skip the MULT_CORR module. Output files already there!')
        sys.stdout.flush()
    if is_run['pz']:
        nofile1 = np.array([not(ex(path['mask_'+f])) for f in fields]).any()
        nofile2 = not(ex(path['cat_full']))
        nofile3 = np.array([not(ex(path['m_'+f])) for f in fields]).any()
        test1 = not(is_run['mask']) and nofile1
        test3 = not(is_run['mult']) and nofile3
        if test1 or nofile2 or test3:
            print('WARNING: I will skip the PHOTO_Z module. Input files not '
                  'found!')
            sys.stdout.flush()
            is_run['pz'] = False
            warning = True
    else:
        print('I will skip the PHOTO_Z module. Output file already there!')
        sys.stdout.flush()
    if is_run['cat']:
        nofile1 = not(ex(path['cat_full']))
        nofile2 = np.array([not(ex(path['m_'+f])) for f in fields]).any()
        if nofile1 or (not(is_run['mult']) and nofile2):
            print('WARNING: I will skip the CATALOGUE module. Input file not '
                  'found!')
            sys.stdout.flush()
            is_run['cat'] = False
            warning = True
    else:
        print('I will skip the CATALOGUE module. Output files already there!')
        sys.stdout.flush()
    if is_run['map']:
        nofile1 = np.array([not(ex(path['mask_'+f])) for f in fields]).any()
        nofile2 = np.array([not(ex(path['cat_'+f])) for f in fields]).any()
        test1 = not(is_run['mask']) and nofile1
        test2 = not(is_run['cat']) and nofile2
        if test1 or test2:
            print('WARNING: I will skip the MAP module. Input files not '
                  'found!')
            sys.stdout.flush()
            is_run['map'] = False
            warning = True
    else:
        print('I will skip the MAP module. Output files already there!')
        sys.stdout.flush()
    if is_run['cl']:
        nofile1 = np.array([not(ex(path['mask_'+f])) for f in fields]).any()
        nofile2 = np.array([not(ex(path['cat_'+f])) for f in fields]).any()
        test1 = not(is_run['mask']) and nofile1
        test2 = not(is_run['cat']) and nofile2
        if test1 or test2:
            print('WARNING: I will skip the CL module. Input files not found!')
            sys.stdout.flush()
            is_run['cl'] = False
            warning = True
    else:
        print('I will skip the CL module. Output files already there!')
        sys.stdout.flush()
    if is_run['cat_sims']:
        nofile1 = np.array([not(ex(path['mask_'+f])) for f in fields]).any()
        nofile2 = np.array([not(ex(path['cat_'+f])) for f in fields]).any()
        test1 = not(is_run['mask']) and nofile1
        test2 = not(is_run['cat']) and nofile2
        if test1 or test2:
            print('WARNING: I will skip the CAT_SIMS module. Input files not '
                  'found!')
            sys.stdout.flush()
            is_run['cat_sims'] = False
            warning = True
    else:
        print('I will skip the CAT_SIMS module. Output files already there!')
        sys.stdout.flush()
    if is_run['cl_sims']:
        nofile1 = np.array([not(ex(path['mask_'+f])) for f in fields]).any()
        nofile2 = np.array([not(ex(path['cl_'+f])) for f in fields]).any()
        nofile3 = not(ex(path['cat_sims']))
        if ex(path['cat_sims']):
            is_files = True
            for f in fields:
                sims = [x for x in os.listdir(path['cat_sims'])
                        if re.match('.+{}\.fits'.format(f), x)]  # noqa:W605
                if len(sims) != n_sims_cov:
                    is_files = False
            nofile3 = not(is_files)
        test1 = not(is_run['mask']) and nofile1
        test2 = not(is_run['cl']) and nofile2
        test3 = not(is_run['cat_sims']) and nofile3
        if test1 or test2 or test3:
            print('WARNING: I will skip the CL_SIMS module. Input files not '
                  'found!')
            sys.stdout.flush()
            is_run['cl_sims'] = False
            warning = True
    else:
        print('I will skip the CL_SIMS module. Output files already there!')
        sys.stdout.flush()

    return is_run, warning
