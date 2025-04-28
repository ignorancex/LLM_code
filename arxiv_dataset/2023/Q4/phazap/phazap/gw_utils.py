import numpy as np
import lal
import bilby

DETECTORS = {'H1': lal.CachedDetectors[lal.LHO_4K_DETECTOR],
             'L1': lal.CachedDetectors[lal.LLO_4K_DETECTOR],
             'V1': lal.CachedDetectors[lal.VIRGO_DETECTOR]}

def time_delay_det(detec,ra, dec, geocent_time):
    """
    Compute time delay from Earth center to detector. 

    Wrap to LAL function TimeDelayFromEarthCenter. 
    See L83 in https://docs.ligo.org/lscsoft/lalsuite/lal/_time_delay_8c_source.html
    
    Parameters
    ----------
    detec: str
        detector
    ra: float
        right ascension
    dec: float
        declination
    geocent_time: float
        geocentric time

    Returns
    -------
    float
        time delay in seconds
    """
    #det: detector
    #ra: right ascension
    #dec: declination
    #geocent_time: geocentric time
    return lal.TimeDelayFromEarthCenter(DETECTORS[detec].location, ra, dec, geocent_time)

def FpFx(det,ra, dec, psi, geocent_time):
    """
    Compute antenna pattern functions.

    See Eq. A.1 in https://arxiv.org/abs/2308.06616.

    Wrap to LAL function ComputeDetAMResponse. See https://docs.ligo.org/lscsoft/lalsuite/lal/antenna_8py_source.html

    Parameters
    ----------
    det: str
        detector
    ra: np.ndarray
        right ascension
    dec: np.ndarray
        declination
    psi: np.ndarray
        polarization
    geocent_time: np.ndarray
        geocentric time

    Returns
    -------
    np.ndarray
        Fp  antenna pattern function            
    np.ndarray
        Fx  antenna pattern function
    """

    #det: detector
    #ra: right ascension
    #dec: declination
    #psi: polarization
    #geocent_time: geocentric time
    gmst = [lal.GreenwichMeanSiderealTime(t) for t in geocent_time]      
    return np.transpose([lal.ComputeDetAMResponse(DETECTORS[det].response, r, d, p, g)
        for r, d, p, g in np.broadcast(ra, dec, psi, gmst)])

#https://docs.ligo.org/lscsoft/lalsuite/lal/group___detector_constants.html
#Position of the detectors vertex in meters with respect to geocenter
H1_vertex_meters = np.array([lal.LHO_4K_VERTEX_LOCATION_X_SI,
                            lal.LHO_4K_VERTEX_LOCATION_Y_SI,
                            lal.LHO_4K_VERTEX_LOCATION_Z_SI])

L1_vertex_meters = np.array([lal.LLO_4K_VERTEX_LOCATION_X_SI,
                            lal.LLO_4K_VERTEX_LOCATION_Y_SI,
                            lal.LLO_4K_VERTEX_LOCATION_Z_SI])

V1_vertex_meters = np.array([lal.VIRGO_VERTEX_LOCATION_X_SI,
                            lal.VIRGO_VERTEX_LOCATION_Y_SI,
                            lal.VIRGO_VERTEX_LOCATION_Z_SI])

def Nvector(source_right_ascension_radians,source_declination_radians,gpstime):
    """
    Compute the unit vector pointing from the source to geocenter.

    For details see "Source position unit vector" section in page 28 of the PDF in 
    https://dcc.ligo.org/LIGO-T010110/public

    For LAL implementation see L53 https://docs.ligo.org/lscsoft/lalsuite/lal/_time_delay_8c_source.html

    Parameters
    ----------
    source_right_ascension_radians: float
        right ascension
    source_declination_radians: float
        declination
    gpstime: float  
        geocentric time

    Returns
    -------
    np.ndarray
        unit vector pointing from the source to geocenter
    """

    #compute the unit vector pointing from the the source to geocenter
    #in LAL they compute -N
    
    gmst = lal.GreenwichMeanSiderealTime(gpstime)
    greenwich_hour_angle = gmst - source_right_ascension_radians
    
    mN_x = np.cos(source_declination_radians) * np.cos(greenwich_hour_angle)
    mN_y = np.cos(source_declination_radians) * (-np.sin(greenwich_hour_angle))
    mN_z = np.sin(source_declination_radians)
    return np.array([-mN_x,-mN_y,-mN_z])

def N_dot_cross_d123(ra,dec,gpstime,d1,d2,d3):
    """
    Compute the dot product between the unit vector pointing from the source to geocenter
    and the cross product between the unit vector pointing from detector 1 to detector 2
    and the unit vector pointing from detector 1 to detector 3

    See Eq. C1 in https://arxiv.org/abs/2308.06616

    Parameters
    ----------
    ra: float
        right ascension
    dec: float
        declination
    gpstime: float
        geocentric time
    d1: np.ndarray
        detector 1 position
    d2: np.ndarray
        detector 2 position
    d3: np.ndarray
        detector 3 position

    Returns
    -------
    float
        dot product
    """
    N = Nvector(ra,dec,gpstime)
    d12 = d1 - d2
    d13 = d1 - d3
    return np.dot(N, np.cross(d12,d13))

def N_dot_d12(ra,dec,gpstime,d1,d2):
    """
    Compute the dot product between the unit vector pointing from the source to geocenter
    and the unit vector pointing from detector 1 to detector 2

    This is useful to compute the time delay between detector 1 and detector 2
    See Eq. 3 in https://arxiv.org/abs/2308.06616

    Parameters
    ----------
    ra: float
        right ascension
    dec: float
        declination
    gpstime: float
        geocentric time
    d1: np.ndarray
        detector 1 position
    d2: np.ndarray
        detector 2 position

    Returns
    -------
    float
        dot product
    """
    N = Nvector(ra,dec,gpstime)
    d12 = d1 - d2
    return np.dot(N, d12) / lal.C_SI

"""Generate waveform"""
def hp_hx(binary_parameters,waveform_generator):
    """
    Generate waveform plus and cross polarizations

    Parameters
    ----------
    binary_parameters: dict
        binary parameters
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        waveform generator

    Returns
    -------
    hp: np.ndarray
        waveform plus polarization
    hx: np.ndarray
        waveform cross polarization
    """
    generate_h = waveform_generator.frequency_domain_strain(binary_parameters)

    hp = generate_h['plus']
    hx = generate_h['cross']
    
    return hp, hx
