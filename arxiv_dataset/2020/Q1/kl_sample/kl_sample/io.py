"""

Module containing all the input/output related functions.

Functions:
 - argument_parser()
 - path_exists_or_error(path)
 - path_exists_or_create(path)
 - read_param(fname, par, type)
 - read_cosmo_array(fname, pars)
 - read_from_fits(fname, name)
 - read_header_from_fits(fname, name)
 - write_to_fits(fname, array, name)
 - print_info_fits(fname)
 - unpack_simulated_xipm(fname)
 - read_photo_z_data(fname)

"""

import argparse
import os
import sys
import re
import numpy as np
from astropy.io import fits
import kl_sample.settings as set


# ------------------- Parser -------------------------------------------------#

def argument_parser():
    """ Call the parser to read command line arguments.

    Args:
        None.

    Returns:
        args: the arguments read by the parser

    """

    parser = argparse.ArgumentParser(
        'Sample the cosmological parameter space using lensing data.')

    # Add supbarser to select between run and prep modes.
    subparsers = parser.add_subparsers(
        dest='mode',
        help='Options are: '
        '(i) prep_real: prepare data in real space. '
        '(ii) prep_fourier: prepare data in fourier space. '
        '(iii) run: do the actual run. '
        '(iv) get_kl: calculate the kl transformation '
        'Options (i) and (ii) are usually not necessary since '
        'the data are are already stored in this repository.')

    run_parser = subparsers.add_parser('run')
    prep_real_parser = subparsers.add_parser('prep_real')
    prep_fourier_parser = subparsers.add_parser('prep_fourier')
    plots_parser = subparsers.add_parser('plots')
    get_kl_parser = subparsers.add_parser('get_kl')

    # Arguments for 'run'
    run_parser.add_argument('params_file', type=str, help='Parameters file')
    run_parser.add_argument(
        '--restart', '-r', help='Restart the chains from the last point '
        'of the output file (only for emcee)', action='store_true')

    # Arguments for 'prep_real'
    prep_real_parser.add_argument(
        'input_folder', type=str, help='Input folder.')

    # Arguments for 'prep_fourier'
    prep_fourier_parser.add_argument(
        'input_path', type=str, help='Input folder. Files that should contain:'
        ' cat_full.fits, mask_arcsec_N.fits.gz (N=1,..,4), mask_url.txt. '
        'See description in kl_sample/prep_fourier.py for more details.')
    prep_fourier_parser.add_argument(
        '--output_path', '-o', type=str, help='Output folder.')
    prep_fourier_parser.add_argument(
        '--badfields_path', '-bp', type=str, help='Folder where the bad fields'
        ' mask are stored, or where they well be downloaded.')
    prep_fourier_parser.add_argument(
        '--cat_sims_path', '-cp', type=str, help='Folder where the catalogues'
        ' of the simulations are stored, or where they well be downloaded.')
    prep_fourier_parser.add_argument(
        '--run_all', '-a', help='Run all routines even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_mask', '-mk', help='Run mask routine even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_mult', '-m', help='Run multiplicative correction routine even '
        'if the files are already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_pz', '-pz', help='Run photo_z routine even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_cat', '-c', help='Run catalogue routine even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_map', '-mp', help='Run map routine even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_cl', '-cl', help='Run Cl routine even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_cat_sims', '-cats', help='Run Cat sims routine even if the '
        'files are already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_cl_sims', '-cls', help='Run Cl sims routine even if the '
        'files are already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--want_plots', '-p', help='Generate plots for the images',
        action='store_true')
    prep_fourier_parser.add_argument(
        '--remove_files', '-rp', help='Remove downloaded files',
        action='store_true')

    # Arguments for 'plots'
    plots_parser.add_argument(
        'output_path', type=str, help='Path to output files')
    plots_parser.add_argument(
        '--params_file', '-p', type=str, help='Path to parameter file')

    # Arguments for 'get_kl'
    get_kl_parser.add_argument('params_file', type=str, help='Parameters file')

    return parser.parse_args()


# ------------------- Check existence ----------------------------------------#

def path_exists_or_error(path):
    """ Check if a path exists, otherwise it returns error.

    Args:
        path: path to check.

    Returns:
        abspath: if the file exists it returns its absolute path

    """
    abspath = os.path.abspath(path)
    if os.path.exists(abspath):
        return abspath
    raise IOError('Path {} not found!'.format(abspath))


def path_exists_or_create(path):
    """ Check if a path exists, otherwise it creates it.

    Args:
        path: path to check. If path contains a file
        name, it does create only the folders containing it.

    Returns:
        abspath: return the absolute path of path.

    """
    abspath = os.path.abspath(path)
    folder, name = os.path.split(abspath)
    cond1 = not bool(re.fullmatch('.+_', name, re.IGNORECASE))
    cond2 = not bool(re.fullmatch(r'.+\..{3}', name, re.IGNORECASE))
    if cond1 and cond2:
        folder = abspath
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


# ------------------- Read ini -----------------------------------------------#

def read_param(fname, par, type='string'):
    """ Return the value of a parameter, either from the
        input file or from the default settings.

    Args:
        fname: path of the input file.
        par: string containing the name of the parameter
        type: output type for the value of the parameter

    Returns:
        value: value of the parameter par

    """

    # Read the file looking for the parameter
    value = None
    n_par = 0
    with open(fname) as fn:
        for line in fn:
            line = re.sub('#.+', '', line)
            if '=' in line:
                name, _ = line.split('=')
                name = name.strip()
                if name == par:
                    n_par = n_par + 1
                    _, value = line.split('=')
                    value = value.strip()

    # If there are duplicated parameters raise an error
    if n_par > 1:
        raise IOError('Found duplicated parameter: ' + par)

    # If par was not in the file use the default value
    if value is None:
        value = set.default_params[par]
        if type == 'bool':
            return value
        print('Default value used for ' + par + ' = ' + str(value))
        sys.stdout.flush()

    # Convert the parameter to the desired type
    if type == 'float':
        return float(value)
    elif type == 'int':
        return int(value)
    # Path type returns the absolute path
    elif type == 'path':
        return os.path.abspath(value)
    # Boolean type considers only the first letter (case insensitive)
    elif type == 'bool':
        if re.match('y.*', value, re.IGNORECASE):
            return True
        elif re.match('n.*', value, re.IGNORECASE):
            return False
        else:
            raise IOError('Boolean type for ' + par + ' not recognized!')
    # Cosmo type has to be returned as a three dimensional array
    # The check that it has been converted correctly is done in
    # read_cosmo_array.
    elif type == 'cosmo':
        try:
            return np.array([float(value), float(value), float(value)])
        except ValueError:
            try:
                array = value.split(',')
                array = [x.strip() for x in array]
                return [None if x == 'None' else float(x) for x in array]
            except ValueError:
                return value
    # All other types (such as strings) will be returned as strings
    else:
        return value


def read_cosmo_array(fname, pars):
    """ Read from the parameter file the cosmological
        parameters and store them in an array.

    Args:
        fname: path of the input file.
        pars: list of the cosmological parameters. Used
        to determine the order in which they are stored

    Returns:
        cosmo_params: array containing the cosmological
        parameters. Each parameter is a row as
        [left_bound, central, right_bound].

    """

    # Initialize the array
    cosmo_params = []
    # Run over the parameters and append them
    # to the array
    for n, par in enumerate(pars):
        # Get the values of the parameter
        value = read_param(fname, par, type='cosmo')
        # Check that the parameter has the correct shape and
        # it is not a string
        if len(value) == 3 and type(value) is not str:
            cosmo_params.append(value)
        else:
            raise IOError('Check the value of ' + par + '!')

    return np.array(cosmo_params)


# ------------------- FITS files ---------------------------------------------#

def read_from_fits(fname, name):
    """ Open a fits file and read data from it.

    Args:
        fname: path of the data file.
        name: name of the data we want to extract.

    Returns:
        array with data for name.

    """
    with fits.open(fname) as fn:
        return fn[name].data


def read_header_from_fits(fname, name):
    """ Open a fits file and read header from it.

    Args:
        fname: path of the data file.
        name: name of the data we want to extract.

    Returns:
        header.

    """
    with fits.open(fname) as fn:
        return fn[name].header


def write_to_fits(fname, array, name, type='image', header=None):
    """ Write an array to a fits file.

    Args:
        fname: path of the input file.
        array: array to save.
        name: name of the image.

    Returns:
        None

    """

    warning = False

    # If file does not exist, create it
    if not os.path.exists(fname):
        hdul = fits.HDUList([fits.PrimaryHDU()])
        hdul.writeto(fname)
    # Open the file
    with fits.open(fname, mode='update') as hdul:
        try:
            hdul.__delitem__(name)
        except KeyError:
            pass
        if type == 'image':
            hdul.append(fits.ImageHDU(array, name=name, header=header))
        elif type == 'table':
            hdul.append(array)
        else:
            print('Type '+type+' not recognized! Data not saved to file!')
            return True
    print('Appended ' + name.upper() + ' to ' + os.path.relpath(fname))
    sys.stdout.flush()
    return warning


def print_info_fits(fname):
    """ Print on screen fits file info.

    Args:
        fname: path of the input file.

    Returns:
        None

    """

    with fits.open(fname) as hdul:
        print(hdul.info())
        sys.stdout.flush()
    return


def get_keys_from_fits(fname):
    """ Get keys from fits file.

    Args:
        fname: path of the data file.

    Returns:
        list of keys.

    """
    with fits.open(fname) as fn:
        return [x.name for x in fn]


# ------------------- On preliminary data ------------------------------------#

def unpack_simulated_xipm(fname):
    """ Unpack a tar file containing the simulated
        correlation functions and write them into
        a single array.

    Args:
        fname: path of the input file.

    Returns:
        array with correlation functions.

    """

    # Import local variables from settings
    n_bins = len(set.Z_BINS)
    n_theta = len(set.THETA_ARCMIN)
    n_fields = len(set.A_CFHTlens)

    # Base name of each file inside the compressed tar
    base_name = '/xipm_cfhtlens_sub2real0001_maskCLW1_blind1_z1_z1_athena.dat'

    # Calculate how many simulations were run based on the number of files
    nfiles = len(os.listdir(fname))
    n_sims, mod = np.divmod(nfiles, n_fields*n_bins*(n_bins+1)/2)
    if mod != 0:
        raise IOError('The number of files in ' + fname + ' is not correct!')

    # Initialize array
    xipm_sims = np.zeros((n_fields, n_sims, 2*n_theta*n_bins*(n_bins+1)/2))

    # Main loop: scroll over each file and import data
    for nf in range(n_fields):
        for ns in range(n_sims):
            for nb1 in range(n_bins):
                for nb2 in range(nb1, n_bins):
                    # Modify the base name to get the actual one
                    new_name = base_name.replace('maskCLW1', 'maskCLW{0:01d}'
                                                 ''.format(nf+1))
                    new_name = new_name.replace('real0001', 'real{0:04d}'
                                                ''.format(ns+1))
                    new_name = new_name.replace('z1_athena', 'z{0:01d}_athena'
                                                ''.format(nb1+1))
                    new_name = new_name.replace('blind1_z1', 'blind1_z{0:01d}'
                                                ''.format(nb2+1))
                    # For each bin pair calculate position on the final array
                    pos = np.flip(np.arange(n_bins+1), 0)[:nb1].sum()
                    pos = (pos + nb2 - nb1)*2*n_theta
                    # Extract file and read it only if it is not None
                    fn = np.loadtxt(fname+new_name)
                    # Read xi_plus and xi_minus and stack them
                    xi = np.hstack((fn[:, 1], fn[:, 2]))
                    # Write imported data on final array
                    for i, xi_val in enumerate(xi):
                        xipm_sims[nf, ns, pos+i] = xi_val

    return xipm_sims


def read_photo_z_data(fname):
    """ Read CFHTlens data and calculate photo_z,
        n_eff and sigma_g.

    Args:
        fname: path of the input file.

    Returns:
        arrays with photo_z, n_eff and sigma_g.

    """

    # Read from fits
    hdul = fits.open(fname, memmap=True)
    table = hdul['data'].data
    image = hdul['PZ_full'].data
    hdul.close()

    # Local variables
    z_bins = set.Z_BINS
    sel_bins = np.array([set.filter_galaxies(table, z_bins[n][0], z_bins[n][1])
                        for n in range(len(z_bins))])
    photo_z = np.zeros((len(z_bins)+1, len(image[0])))
    n_eff = np.zeros(len(z_bins))
    sigma_g = np.zeros(len(z_bins))
    photo_z[0] = (np.arange(len(image[0]))+1./2.)*set.dZ_CFHTlens

    # Main loop: for each bin calculate photo_z, n_eff and sigma_g
    for n in range(len(z_bins)):
        # Useful quantities TODO: Correct ellipticities
        w_sum = table['weight'][sel_bins[n]].sum()
        w2_sum = (table['weight'][sel_bins[n]]**2.).sum()
        m = np.average(table['e1'][sel_bins[n]])
        e1 = table['e1'][sel_bins[n]]/(1+m)
        e2 = table['e2'][sel_bins[n]]-table['c2'][sel_bins[n]]/(1+m)

        # photo_z
        photo_z[n+1] = np.dot(table['weight'][sel_bins[n]],
                              image[sel_bins[n]])/w_sum
        # n_eff
        n_eff[n] = w_sum**2/w2_sum/set.A_CFHTlens.sum()
        # sigma_g
        sigma_g[n] = np.dot(table['weight'][sel_bins[n]]**2.,
                            (e1**2. + e2**2.)/2.)/w2_sum
        sigma_g[n] = sigma_g[n]**0.5

        # Print progress message
        print('----> Completed bin {}/{}'.format(n+1, len(z_bins)))
        sys.stdout.flush()

    return photo_z, n_eff, sigma_g


# ------------------- Import template Camers ---------------------------------#

def import_template_Camera(path, settings):
    ell_max = settings['ell_max']
    nb = settings['n_bins']
    file = np.genfromtxt(path, unpack=True)
    rell = int(file[0].min()), int(file[0].max())
    corr = np.zeros((ell_max+1, nb, nb))
    triu_r, triu_c = np.triu_indices(nb)
    for n, _ in enumerate(range(int(nb*(nb+1)/2))):
        corr[rell[0]:rell[1]+1, triu_r[n], triu_c[n]] = file[n+1]
        corr[rell[0]:rell[1]+1, triu_c[n], triu_r[n]] = file[n+1]
    return corr
