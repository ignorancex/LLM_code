import numpy as np
import time
import logging 
logging.basicConfig(level=logging.INFO)
import src.utilities.constants as const


def rho(r, l, b):
    """ Function to calculate the distance from the Galactic center to the star/ a point in the Galaxy from Earth-centered coordinates (r, l, b)

    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        l: Galactic longitude, radians
        b: Galactic latitude, radians
    Returns:
        distance from the Galactic center to the star/ a point in the Galaxy
    """
    return np.sqrt(np.power(r * np.cos(b), 2) + np.power(const.r_s, 2) - 2 * const.r_s * r * np.cos(b) * np.cos(l)) # kpc, distance from the Sun to the star/ spacepoint


def theta(r, l, b):
    """ Function to calculate the angle from the Galactic centre to the star/ a point in the Galaxy from Earth-centered coordinates (r, l, b)

    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        l: Galactic longitude, radians
        b: Galactic latitude, radians
    Returns:
        angle from the Sun to the star/ a point in the Galaxy
    """
    return np.arctan2(const.r_s - r*np.cos(b)*np.cos(l), r * np.cos(b) * np.sin(l))


def z(r, b):
    """ Function to calculate the z-coordinate of the star/ a point in the Galaxy from Earth-centered coordinates (r, l, b)

    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        b: Galactic latitude, radians
    Returns:
        z-coordinate of the star/ a point in the Galaxy
    """
    return r * np.sin(b)


def xy_to_long(x, y):
    """ Function to calculate the Galactic longitude from the x and y coordinates

    Args:
        x: x-coordinate
        y: y-coordinate
    Returns:
        Galactic longitude in radians
    """
    return np.arctan2(y - const.r_s, x) + np.pi/2


def axisymmetric_disk_population(rho, h):
    """ Function describing the density of the disk at a distance rho from the Galactic center

    Args:
        rho: distance from the Galactic center
        h: scale length of the disk
    Returns:
        density of the disk at a distance rho from the Galactic center
    """
    return np.exp(-rho/h)


def height_distribution(z, sigma):
    """ Function describing the density of the disk at a height z above the Galactic plane

    Args:
        z: height above the Galactic plane
        sigma: 
    Returns:
        height distribution of the disk at a height z above the Galactic plane
    """
    return np.exp(-0.5 * z**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)


def running_average(data, window_size):
    """ Calculates the running average of the data

    Args: 
            data: 1D np.array with data
            window_size: int, the size of the window used to calculate the running average. Denotes the number of points for each window
    Returns:
            1D np.array with the running average of the data
    """
    array_running_averaged = []
    delta = int((window_size)//2)
    for i in range(len(data)):
        if i-delta < 0:
            val = np.sum(data[-delta + i:]) + np.sum(data[:delta + i + 1])
            array_running_averaged.append(val)
        elif i+delta >= len(data):
            val = np.sum(data[i-delta:]) + np.sum(data[:delta + i - len(data) + 1])
            array_running_averaged.append(val)
        else:
            array_running_averaged.append(np.sum(data[i-delta:i+delta + 1]))
    return np.array(array_running_averaged)


def sum_pairwise(data):
    """ Sums up the elements of an array pairwise. Array must contain even number of points

    Args:
        a: even 1D np.array
    Returns:
        1D np.array with the summed up values. Half the size of the input array
    """
    if not len(data) % 2 == 0:
        print("The array must contain an even number of points")
        return None
    paired_data = data.reshape(-1, 2)
    result = np.sum(paired_data, axis=1)  # Sum along the specified axis (axis=1 sums up each row)
    return result


def rearange_data(data):
    """ Rearanges data to be plotted in desired format. E.g. instead of data going from 0 to 360 degrees, the returned data will go from 180 -> 0/ 360 -> 180 degrees, the format used by FIXEN et al. 1999 and Higdon and Lingenfelter 2013
    
    Args:
        data: 1D np.array with data. Must contain even number of points
    Returns:
        1D np.array with the rearanged data. Half the size of the input array
    """
    if not len(data) % 2 == 0:
        raise ValueError("The array must contain an even number of points")
    middle = int(len(data)/2)
    data_centre_left = data[0]
    data_left = sum_pairwise(data[1:middle-1])
    data_left_edge = data[middle-1]
    data_right_edge = data[middle]
    data_edge = (data_right_edge + data_left_edge)
    data_right = sum_pairwise(data[middle+1:-1])
    data_centre_right = data[-1]
    data_centre = (data_centre_left + data_centre_right)
    rearanged_data = np.concatenate(([data_edge], data_left[::-1], [data_centre], data_right[::-1], [data_edge]))
    return rearanged_data


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"{func.__name__} took {elapsed_time:.6f} seconds to run.")
        return result
    return wrapper


def imf_0(mass):
    """ Function for the modified Kroupa initial mass function in the range 0.01 <= M/M_sun < 0.08
    From: https://ui.adsabs.harvard.edu/abs/2002Sci...295...82K/abstract
    Args:
        mass: numpy array with mass values in units of solar mass for which to calculate the initial mass function
    
    Returns:
        numpy array with the initial mass function for the given mass values
    """
    from src.utilities.constants import m_lim_imf_powerlaw as m
    if not np.all((mass >= m[0]) & (mass < m[1])):
        print("The mass values must be in the range 0.01 <= M/M_sun < 0.08")
        return
    
    return np.power(mass / m[1], -const.alpha[0]) # the modified Kroupa initial mass function for 0.01 <= M/M_sun < 0.08


def imf_1(mass):
    """ Function for the modified Kroupa initial mass function in the range 0.08 <= M/M_sun < 0.5
    From: https://ui.adsabs.harvard.edu/abs/2002Sci...295...82K/abstract
    Args:
        mass: numpy array with mass values in units of solar mass for which to calculate the initial mass function
    
    Returns:
        numpy array with the initial mass function for the given mass values
    """
    from src.utilities.constants import m_lim_imf_powerlaw as m
    if not np.all((mass >= m[1]) & (mass < m[2])):
        print("The mass values must be in the range 0.08 <= M/M_sun < 0.5")
        return
    
    return np.power(mass / m[1], -const.alpha[1]) # the modified Kroupa initial mass function for 0.01 <= M/M_sun < 0.08


def imf_2(mass):
    """ Function for the modified Kroupa initial mass function in the range 0.5 <= M/M_sun < 1
    From: https://ui.adsabs.harvard.edu/abs/2002Sci...295...82K/abstract
    Args:
        mass: numpy array with mass values in units of solar mass for which to calculate the initial mass function
    
    Returns:
        numpy array with the initial mass function for the given mass values
    """
    from src.utilities.constants import m_lim_imf_powerlaw as m
    if not np.all((mass >= m[2]) & (mass < m[3])):
        print("The mass values must be in the range 0.5 <= M/M_sun < 1")
        return
    
    return np.power(m[2]/m[1], -const.alpha[1]) * np.power(mass / m[2], -const.alpha[2]) # the modified Kroupa initial mass function for 0.5 <= M/M_sun < 1.0


def imf_3(mass):
    """ Function for the modified Kroupa initial mass function in the range 1 <= M/M_sun < 120
    From: https://ui.adsabs.harvard.edu/abs/2002Sci...295...82K/abstract
    Args:
        mass: numpy array with mass values in units of solar mass for which to calculate the initial mass function
    
    Returns:
        numpy array with the initial mass function for the given mass values
    """
    from src.utilities.constants import m_lim_imf_powerlaw as m
    if not np.all((mass >= m[3]) & (mass <= m[4])):
        print("The mass values must be in the range 1 <= M/M_sun <= 120")
    
    return np.power(m[2]/m[1], -const.alpha[1]) * np.power(m[3]/m[2], -const.alpha[2]) * (mass/m[3])**(-const.alpha[3]) # the modified Kroupa initial mass function for M/M_sun > 1


def imf():
    """ Function for the modified Kroupa initial mass function in the range 0.01 <= M/M_sun < 120

    Returns: 
        m: numpy array with mass values in units of solar mass in the range 0.01 <= M/M_sun < 120
        imf: numpy array with the initial mass function for the given mass values
    """
    m0 = np.linspace(0.01, 0.08, 100, endpoint=False)
    m1 = np.linspace(0.08, 0.5, 100, endpoint=False)
    m2 = np.linspace(0.5, 1, 100, endpoint=False)
    m3 = np.linspace(1, 120, 700, endpoint=False)
    imf0 = imf_0(m0)
    imf1 = imf_1(m1)
    imf2 = imf_2(m2)
    imf3 = imf_3(m3)
    m = np.concatenate((m0, m1, m2, m3))
    imf = np.concatenate((imf0, imf1, imf2, imf3))
    return m, imf


def lifetime_as_func_of_initial_mass(mass):
    """ Function for lifetime as function of initial stellar mass

    Args: 
        mass: stellar mass

    Returns:
        lifetime: stellar age in Myr
    """
    lifetime = const.tau_0 * np.power(mass, const.beta) / 1e6 # Lifetime as function of initial mass
    return lifetime
