# author: Koustav Konar <n.koustav.konar@gmail.com>
# Peripheral tools for FRB_SBI


import configparser
import collections
import healpy as hp
import numpy as np
import pandas as pd


def frb_in_bin(z_edges_for_glass, z_frb, phi_radians, theta_radians, NSIDE):
    '''
    Returns bin information based on FRB redshift and the redshift boundaries of shells
    ------------
    PARAMETERS:
        z_edges_for_glass (array): redshift boundaries of GLASS shells
        z_frb (array): redshift of FRBs
    ------------
    RETURNS:
    shell_num, redshifts, weight, theta, phi, pixel_position

        shell_num (array): shell numbers occupied by FRBs in increasing redshift
        redshifts (array): redshift of FRBs (same as input)
        weight (array): weight based on fractional position of the FRB wrt to the shell boundary
        theta, phi (array): angular position of FRBs (same as input catalogue)
        pixel_position (array): Healpix pixel position for a patch of 1 degree 
    ------------
    '''
    z_frb_sorted = np.sort(z_frb)
    nbins = len(z_edges_for_glass) - 1
    patch_size_in_degree = 1
    bins = [(z_edges_for_glass[i], z_edges_for_glass[i+1]) for i in range(nbins)]
    bin_edges = [(z_edges_for_glass[i], z_edges_for_glass[i+1]) for i in range(nbins)]
    bin_edges = np.array(bin_edges, dtype=float)
    bin_lower_edge = bin_edges[:,0]
    bin_upper_edge = bin_edges[:,1]
    z_frb_bins = np.zeros(nbins)
    z_in_bins = []
    theta_in_bins = []
    phi_in_bins = []
    vec_in_bins = []
    pixel_in_bins = []
    info_in_bins = []
    info_in_empty_bins = []
    for i, bin in enumerate(bins):
        indices_ = (z_frb_sorted >= bin[0]) & (z_frb_sorted < bin[1])
        z_in_bins.append(z_frb_sorted[indices_])
        theta_in_bins.append(theta_radians[indices_])
        phi_in_bins.append(phi_radians[indices_])
        theta_shell_radians = abs(theta_in_bins[i])
        phi_shell_radians = phi_in_bins[i]
        vec = hp.ang2vec(theta_shell_radians, phi_shell_radians)
        vec_in_bins.append(vec)
    count = 0
    for i in range(nbins):
        if len(vec_in_bins[i]) == 0:
            info_in_empty_bins.append(f'No FRB in Shell {i} for redshift boundary {z_edges_for_glass[i], z_edges_for_glass[i+1]}')
        else:
            for j in range(len(vec_in_bins[i])):
                ipix_disc = hp.query_disc(nside=NSIDE, vec=vec_in_bins[i][j], radius=np.radians(patch_size_in_degree))
                pixel_in_bins.append(ipix_disc)
                bin_edges_for_frb = bin_edges[i]
                low_edge = bin_edges_for_frb[0]
                up_edge = bin_edges_for_frb[1]
                weight = (z_frb_sorted[count] - low_edge)/(up_edge - low_edge)
                info_in_bins.append([i, z_in_bins[i][j], weight, theta_in_bins[i][j], phi_in_bins[i][j], ipix_disc])
                count+=1
    result1 = np.array(info_in_bins, dtype=object)
    result = result1.T
    return result


def auto_correlation_index(n=2):
    '''
    Returns the indices of Cl_gg from Levin for the 'Auto' correlation.
    ------------
    n = no. of redshift bins being used in Levin such that len(z_edges) = n+1
    Levin returns the spctra in the following manner
    00(0) 01(1) 02(2)
            11(3) 12(4)
                    22(5)
    Here, n = 3 and the indices for Cl_gg are mentioned in brackets.
    For this case the function returns an array [0,3,5]
    ------------
    '''
    start = 0
    n = n
    result = []
    for i in range(n):
        result.append(start + (n - i))
    new_lst = np.cumsum(result)
    new_lst = np.insert(new_lst, 0, 0)
    return new_lst[:n]

def cross_correlation_index(n=2):
    '''
    Returns the indices of Cl_gg from Levin for the 'Cross' correlation.
    ------------
    n = no. of redshift bins being used in Levin such that len(z_edges) = n+1
    Levin returns the spctra in the following manner
    00(0) 01(1) 02(2)
            11(3) 12(4)
                    22(5)
    Here, n = 3 and the indices for Cl_gg are mentioned in brackets.
    For this case the function returns an array [1,2,4]
    ------------
    '''
    #triangular number to calculate number of spctra
    a = 0
    n = n
    for i in range(n+1):
        a+=i
    auto_indices = auto_correlation_index(n)
    all_indices = np.arange(a)
    cross_correlation_index = np.delete(all_indices, auto_indices)
    return cross_correlation_index

def levin_to_glass(cls, z_array, ell_array = np.arange(2, 1003, 1)):
    '''
    Returns correctly ordered Angular power spectra ($C_\ell$) for GLASS
    ------------
    PARAMETERS:
        cls (ndarray): Angular power spectra from Levin (Cl_gg)
        z_array (array): redshift boundaries of GLASS shells
        ell_array (array): multipole as defined in Levin code
    ------------
    RETURNS:
        cls (ndarray): cls following GLASS ordering scheme
        X (array): index of ordering Glass
    ------------
    gls = [gl_00,
            gl_11, gl_10,
            gl_22, gl_21, gl_20,
            ...]
    https://glass.readthedocs.io/en/stable/reference/fields.html
    ------------
    '''
    cells = cls
    zb = z_array
    nbins = len(zb) - 1
    ells = ell_array

    if nbins*(nbins+1)/2 != np.shape(cells)[0]:
        print('Wrong z_edge input')
    else:
        cells = np.insert(cells, 0, np.arange(len(cells)), axis=1)

        counter = 0
        nbins = len(zb)-1

        matrix_cells = np.zeros((len(ells)+1,nbins,nbins))
        for i in range(nbins):
            for j in range(i,nbins):
                matrix_cells[:,i,j] = cells[counter,:]
                matrix_cells[:,j,i] = cells[counter,:]
                counter +=1
        
        counter = 0
        for i in range(nbins):
            for j in reversed(range(i+1)): 
                index_i = i
                index_j = j
                cells[counter,:] = matrix_cells[:,index_i, index_j]
                counter += 1
        correct_sequencce = np.array(cells[:,0])
        correct_cls = np.delete(cells, 0, axis=1)
        return(correct_cls, correct_sequencce)



def read_catalogue(loc):
    '''
    Read catalogue and returns arrays
    ---------
    PARAMETERS:
        loc (string): catalogue location
    ---------
    RETURNS:
    z_frb, dm_from_catalogue, dm_milky_way, frb_name, frb_name_label, phi_radians, theta_radians
        z_frb (array): FRB redshifts
        dm_from_catalogue (array): Observed DM
        dm_milky_way (array): Milky Way DM
        frb_name (array): FRB names 
        frb_name_label (array): FRB detection date 
        phi_radians, theta_radians (array): FRB angular position in radians
    ------------
    '''
    data = pd.read_csv(loc)

    data = data.sort_values(by='redshift', ascending=True)
    data = data[data['redshift']>0.1]
    # data = data[data['dm']<1200]
    
    z_frb = np.array(data['redshift'])
    dm_from_catalogue = np.array(data['dm'])
    dm_milky_way = np.array(data['dm_milky_way'])
    frb_name = np.array(data['frb'])
    frb_name_label = np.array([s.replace('FRB', '').strip() for s in frb_name])
    
    ra_degrees2 = data['ra (deg)']
    phi_degrees2 = np.copy(ra_degrees2)
    phi_radians = np.deg2rad(phi_degrees2)
    
    dec_degrees2 = data['dec (deg)']
    theta_degrees2 = np.copy(dec_degrees2)
    theta_radians = np.deg2rad(theta_degrees2)

    return z_frb, dm_from_catalogue, dm_milky_way, frb_name, frb_name_label, phi_radians, theta_radians


def assign_variable(file_path='config_SBI_DM_FRB.ini', verbose=False):
    '''
    Assigns variables from the config file to their numerical values (as global parameters) and returns a dictionary.
    ------------
    PARAMETERS:
        file_path (string): config file path
        verbose (boolean): True: prints variables and numerical values
    ---------
    RETURNS:
        variables_dict (dictionary): dictionary from the config file, 'keys': variables, 'value': values
    ---------
    '''
    config = configparser.ConfigParser()
    config.read(file_path)

    variables_dict = {}  # Dictionary to store variables

    for section in config.sections():
        named_tuple = collections.namedtuple(section, config.options(section))
        values = [config.get(section, option) for option in config.options(section)]

        # Convert each value individually
        converted_values = []
        for value in values:
            try:
                int_value = int(value)  # Try converting to integer
                converted_values.append(int_value)
            except ValueError:
                try:
                    float_value = float(value)  # Try converting to float
                    converted_values.append(float_value)
                except ValueError:
                    converted_values.append(value)  # Keep original string value

        # Create an instance of the namedtuple with the converted values
        section_instance = named_tuple(*converted_values)

        # Assign variables as globals
        for field in section_instance._fields:
            value = getattr(section_instance, field)
            variables_dict[field] = value
            if verbose:
                print(f'{field} = {value}')
            globals()[field] = value

    return variables_dict


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        minutes, remaining_seconds = divmod(seconds, 60)
        return f"{int(minutes)} min{'s' if minutes > 1 else ''} {int(remaining_seconds)} sec{'s' if remaining_seconds > 1 else ''}"
    else:
        hours, remaining_minutes = divmod(seconds, 3600)
        minutes, remaining_seconds = divmod(remaining_minutes, 60)
        formatted_time = f"{int(hours)} hour{'s' if hours > 1 else ''}"
        if minutes > 0:
            formatted_time += f" {int(minutes)} min{'s' if minutes > 1 else ''}"
        if remaining_seconds > 0:
            formatted_time += f" {int(remaining_seconds)} sec{'s' if remaining_seconds > 1 else ''}"
        return formatted_time
