# imports and initialization
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from . import jpl
from . import openorb
from . import orbfit

integ_functions = {}
integ_functions["JPL Horizons"] = jpl.get_ephem_jpl
integ_functions["OpenOrb"] = openorb.get_ephem_OpenOrb
integ_functions["OrbFit"] = orbfit.get_ephem_OrbFit


# Takes in object id and calls all integrators
def get_ephems(obj_id, start, stop, obs):
    '''
    Parameters
    ----------
    obj_id : `str`
        Number of asteroid, defined by MPC.
    start : `str`
        Beginning time of integration, UTC.   
    stop : `str`
        End time of integration, UTC.   
    obs : `str`
        Observatory code.
        
    Returns
    -------
    el_jpl : `~numpy.ndarray` (N, 18)
        Orbital elements as determined by JPL Horizons.
    results : `dict` (N, size(integ_functions), 3)
        (RA, DEC) units deg + (mag) units mags as determined by all integrators in integ_functions
        '''
    
    el_jpl = jpl.get_elems(obj_id, start)
    print("Elements done")
    results = {}
    
    for name, func in integ_functions.items():
        #if name in ["OpenOrb", "JPL Horizons"]:
            #continue
        results[name] = func(el_jpl, start, stop, obs)
        print(f"{name} done")
        
    return el_jpl, results


# Maximum circle difference function. Calculates great circle difference and other difference metrics (future)
def gc_dist(results, reference="JPL Horizons"):
    '''
    Parameters
    ----------
    results : `dict` (N, size(integ_functions), 3)
        (RA, DEC) units deg + (mag) units mags as determined by all integrators in integ_functions
    reference : `str` kwarg
        Keyword argument with reference to which integrator all computations are made. 
        Current default is set as JPL Horizons.
        
    Returns
    -------
    rsep : `dict` (N, size(integ_functions))
        Difference of coordinates as calculated by SkyCoord from astropy.
    rmag : `dict` (N, size(integ_functions))
        Difference of magnitudes.
        '''

    ref = results[reference]
    coord_ref = SkyCoord(ref[0][0], ref[0][1], frame='icrs', unit="deg")
    
    rsep, rmag = {}, {}
    for integ_name, result in results.items():
        if integ_name == reference:
                continue

        # compute the coordinate difference relative to a reference integrator
        coord = SkyCoord(result[0][0], result[0][1], frame='icrs', unit="deg")
        sep = coord_ref.separation(coord).arcsec
        rsep[integ_name] = sep
    
        # compute the magnitude difference
        dmag = result[1] - ref[1]
        rmag[integ_name] = dmag

    return rsep, rmag


def table_stats(dictionary):
    '''
    Parameters
    ----------
    dictionary : `dict`
        Dictionary of separations or magnitude differences to compute statistics on
    
    Returns
    -------
    statistics.stack() : `pd.Dataframe`
        pandas DataFrame as a series generated via the function, contains mean, median and maximuma from gc_diff 
    '''
    
    statistics = pd.DataFrame(data= None, columns=['integrator', 'mean', 'median', 'max'])

    for integ_name, result in dictionary.items():
        statistics = statistics.append({'integrator' : integ_name, 'mean': np.mean(result), 'median': np.median(result), 
                              'max': np.max(result)}, ignore_index = True)
        
    statistics.set_index('integrator', inplace=True)  
    
    return statistics.stack()


# Creates default colour map for tables
def default_map(s):
    ret = []
    for val in s:
        if val < 0.05:
            style = ['background-color: green']
        elif val < 0.2:
            style = ['background-color: yellow']
        elif val < 0.6:
            style = ['background-color: orange']
        else:
            style = ['background-color: red']
        
        ret += style
    return ret

# Creates red/green colourblind-friendly colour map for tables
def rg_friendly_map(s):
    ret = []
    for val in s:
        if val < 0.05:
            style = ['background-color: #F7F7F7']
        elif val < 0.2:
            style = ['background-color: #92C5DE']
        elif val < 0.6:
            style = ['background-color: #F4A552']
        else:
            style = ['background-color: #CA0020']
        
        ret += style
    return ret

default_asteroids = [('MBA','202930'), ('MBA', '110'), ('MBA', '887'), ('MBA', '6489'), ('MBA', '176014'),
             ('KBO','136199'), ('KBO', '15760'), 
             ('PHA', '2018 VP1'), ('PHA', '101955'), ('PHA', '99942'), 
             ('Mars Trojan', '5261'),
             ('Jupiter Trojan', '624'), ('Jupiter Trojan', '588'),
             ('Centaur', '10199'),
             ('TNO', '136472'), ('TNO', '136199'),
             ('Impactor', '2008 TC3'),
             ('NEO', '2020 VT4'), ('NEO', '433'), ('NEO', '367943'), ('NEO', '2019 OK'),
             ('a Vatira', '2020 AV2'),
             ('Interstellar', 'A/2017 U1')
            ]

def calc_all(asteroids=default_asteroids, start='2010-01-01T00:00:00', stop='2020-01-01T00:00:01', obs='I11'):
    '''
    Parameters
    ----------
    *currently all are hardcoded, but override is possible to pick and choose any subset of asteroids*
    asteroids: `str`, `tuple` (N, 2)
        Tuple list of type and asteroid id, both in string format
    start: `str`
        Start time of integration epoch, ISOT
    stop: `str`
        End time of integration epoch, ISOT
    obs: `str`
        Observatory code
    
    Returns
    -------
    result: `dict` (N, size(integ_functions))
        median, mean, max for every value in ephemerides in comparison to JPL ephem outputs.
    '''
    
    result = {}
    for kind, obj_id in asteroids:
        el_jpl, results = get_ephems(obj_id, start, stop, obs)
        rsep, rmag = gc_dist(results, reference="JPL Horizons")
        stats = table_stats(rsep)
        result[(kind, obj_id)] = stats
    return result

def plot(data, cmap):
    '''
    Parameters
    ----------
    data: `dict`
        Dictionary with data that you would like to plot up via the provided colourmaps
    cmap: `str`
        Specifies which colour map to use, default or red/green colourblind friendly. Possible arguments: 'default', 'rg_freindly'
    Returns
    -------
    pandas.DataFrame() object
        Pandas Data Frame object with colouration
    '''
    if cmap=='default':
         display(pd.DataFrame(data).T.style.apply(default_map))
    elif cmap=='rg_friendly':
        display(pd.DataFrame(data).T.style.apply(rg_friendly_map))
    return