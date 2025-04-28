#imports and initialization
import numpy as np
import astropy as ap
import astropy.units as u
import pyoorb as oo
oo.pyoorb.oorb_init()

# Reorganizes JPL Horizons elements output into pyoorb-acceptable input. (Expand for multiple orbits next?)
def pyoorb_input(orbits, epoch):
    '''
    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 18)
        Orbital elements as determined by JPL Horizons.
    epoch : `~numpy.ndarray` (3652, 2)
        Constrained to cometary format.
        
    Returns
    -------
    new_array : `~numpy.ndarray` (N, 12)
        Orbits formatted in the format expected by PYOORB. 
            orbit_id : index of input orbits
            elements x6: orbital elements of propagated orbits
            orbit_type : orbit type
            epoch_mjd : epoch of the propagate orbit
            time_scale : time scale of output epochs
            H/M1 : absolute magnitude
            G/K1 : photometric slope parameter
    '''
    
    temp = orbits.copy()
    temp = temp.as_array().data
    if temp.shape == (6,):
        num_orbits = 1
    else:
        num_orbits = temp.shape[0]
        
    for i in range(num_orbits):
        ids = i
        orbit_type = 2
        time_scale = 1
        
    # elements x6
    q = temp[0][6]
    e = temp[0][5]
    incl = np.deg2rad(temp[0][7])
    longnode = np.deg2rad(temp[0][8])
    argper = np.deg2rad(temp[0][9])
    peri_epoch = ap.time.Time(temp[0][10], format='jd').mjd

    mag = temp[0][3]
    slope = temp[0][4]
    
    if num_orbits > 1:
        new_array = np.array(
            np.array([
                ids, 
                q,
                e,
                incl,
                longnode,
                argper,
                peri_epoch,
                orbit_type,
                epoch,
                time_scale,
                mag,
                slope
            ]), 
            dtype=np.double, 
            order='F')
    else:
        new_array = np.array(
            [[
                ids, 
                q,
                e,
                incl,
                longnode,
                argper,
                peri_epoch,
                orbit_type,
                epoch,
                time_scale,
                mag,
                slope
            ]], 
            dtype=np.double,
            order='F')
    
    return new_array


# Calculates ephemerides using PYOORB
def get_ephem_OpenOrb(el_jpl, start, stop, obs):
    '''
    Parameters
    ----------
    el_jpl : `~numpy.ndarray` (N, 18)
        Orbital elements as determined by JPL Horizons.
    start : `str`
        Beginning time of integration, UTC.   
    stop : `str`
        End time of integration, UTC.
    obs : `str`
        Observatory code.
        
    Returns
    ------- 
    coord_OpenOrb : `~numpy.ndarray` (N, 2)
        RA and DEC coordinates of ephemderides as determined by PYOORB, units deg
            RA : first column of coord_OpenOrb
            DEC : second column of coord_OpenOrb 
            
    mag_OpenOrb :  `~numpy.ndarray` (N, 1)
        Contains magnitudes of object as determined by PYOORB, units mags
    '''
    
    # time conversions, epochs for pyoorb to work
    element_time_pyoorb = ap.time.Time(start).mjd
    start_pyoorb = ap.time.Time(start).mjd
    stop_pyoorb = ap.time.Time(stop).mjd
    peri_time = ap.time.Time(el_jpl['Tp_jd'][0], format='jd').mjd
    
    #conversion and implementation
    pyoorb_formatted = pyoorb_input(el_jpl, start_pyoorb)
    t0 = np.array([element_time_pyoorb, 1], dtype=np.double, order='F')
    mjds = np.arange(start_pyoorb, stop_pyoorb, 1)
    epochs = np.array(list(zip(mjds, [1]*len(mjds))), dtype=np.double, order='F')
    ephem_pyoorb, err = oo.pyoorb.oorb_ephemeris_basic(in_orbits=pyoorb_formatted,
                                         in_obscode=obs,
                                         in_date_ephems=epochs,
                                         in_dynmodel='N')
    if err != 0:
        print(err)
        
    coord_OpenOrb = np.array([ephem_pyoorb[0][:,1],ephem_pyoorb[0][:,2]]) * u.deg
    mag_OpenOrb = np.array([ephem_pyoorb[0][:,9]]) * u.mag
    
    return coord_OpenOrb, mag_OpenOrb