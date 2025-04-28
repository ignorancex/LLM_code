# imports and initialization
import astropy as ap
import numpy as np
from astroquery.jplhorizons import Horizons
import astropy.units as u

# calculates elements
def get_elems(obj_id, start):
    '''
    Parameters
    ----------
    obj_id : `str`
        Number of asteroid, defined by MPC.
    start : `str`
        Beginning time of integration, UTC.
        
    Returns
    -------
    el_jpl : `~numpy.ndarray` (N, 12)
        Orbital elements as returned by jpl, first column is obj_id.   
    '''
    
    epochs = ap.time.Time(start).jd
    
    el_obj = Horizons(id=obj_id, location= '500@10',
               epochs=epochs)
    el_jpl = el_obj.elements()    
    
    el_jpl['targetname'] = obj_id
    
    return el_jpl


# calculates ephemerides for object via JPL Horizons
def get_ephem_jpl(el_jpl, start, stop, obs):
    '''
    Parameters
    ----------
    el_jpl : `~numpy.ndarray` (N, 12)
        Orbital elements as returned by jpl.    
    start : `str`
        Beginning time of integration, UTC.   
    stop : `str`
        End time of integration, UTC.    
    obs : `str`
        Observatory code.
        
    Returns
    -------
    coord_jpl : `~numpy.ndarray` (N, 2)
        RA and DEC coordinates of ephemderides as determined by JPL, units deg
            RA : first column of coord_jpl
            DEC : second column of coord_jpl     
    mag_jpl : `~numpy.ndarray` (N, 1)
        Contains magnitudes of object as determined by JPL, units mags
    '''
    
    obj_id = el_jpl['targetname'][0]

    ephem_obj = Horizons(id= obj_id, location= obs,
               epochs={'start': start, 'stop':stop,
                      'step':'1d'})
    ephem_jpl = ephem_obj.ephemerides()

    coord_jpl = np.array([ephem_jpl['RA'], ephem_jpl['DEC']]) * u.deg
    mag_jpl = np.array([ephem_jpl['V']]) * u.mag

    return coord_jpl, mag_jpl