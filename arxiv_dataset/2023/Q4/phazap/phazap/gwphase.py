import numpy as np
from . import gw_utils as gwutils

"""
Phase at new frequency
"""
def phases_f_ev(waveform_generator,rang_fs,mass_1=30,mass_2=30,a_1=0,a_2=0,tilt_1=0,tilt_2=0,phi_12=0,phi_jl=0,luminosity_distance=500,theta_jn=0.1,psi = 0.1,phase_ref=0.0,geocent_time=0.,ra=0.,dec=0.):
    """
    Compute the phase at a new frequencies
    For details see Appendix B in https://arxiv.org/pdf/2308.06616.pdf

    Parameters
    ----------
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        waveform generator
    rang_fs: np.ndarray
        range of frequencies
    mass_1: float
        primary mass
    mass_2: float
        secondary mass
    a_1: float
        primary aligned spin
    a_2: float
        secondary aligned spin
    tilt_1: float
        primary spin tilt
    tilt_2: float
        secondary spin tilt
    phi_12: float
        relative spin phase
    phi_jl: float
        angle between J and L
    luminosity_distance: float
        luminosity distance
    theta_jn: float
        angle between J and N
    psi: float
        polarization angle
    phase_ref: float
        phase at reference frequency
    geocent_time: float
        geocentric time
    ra: float
        right ascension
    dec: float
        declination

    Returns
    -------
    phase: np.ndarray
        phase at new frequency.
        See Eq. B3 of https://arxiv.org/pdf/2308.06616.pdf 
    a22: np.ndarray
        Tangent of beta, the ratio of the semi-major and minor axes.
        It is a function of the inclination at new frequency. 
        See Eqs. B4 and B14 of https://arxiv.org/pdf/2308.06616.pdf
    zeta: np.ndarray
        polarization rotation at new frequency.
        See Eq. B5 of https://arxiv.org/pdf/2308.06616.pdf
    r: np.ndarray
        ratio of amplitudes L and R at new frequency.
        See below Eq. B4 of https://arxiv.org/pdf/2308.06616.pdf 
    """
    binary_parameters = dict(
        mass_1=mass_1,
        mass_2=mass_2,
        a_1=a_1,
        a_2=a_2,
        tilt_1=tilt_1,
        tilt_2=tilt_2,
        phi_12=phi_12,
        phi_jl=phi_jl,
        luminosity_distance=luminosity_distance,
        theta_jn=theta_jn,
        psi=psi,
        phase=phase_ref,
        geocent_time=geocent_time,
        ra=ra,
        dec=dec,
    )
    
    hp, hx = gwutils.hp_hx(binary_parameters,waveform_generator)
    
    #Left and right
    thL_all = (hp - 1.0j*hx)/np.sqrt(2.)
    thR_all = (hp + 1.0j*hx)/np.sqrt(2.)
    
    thL = thL_all[rang_fs]
    thR = thR_all[rang_fs]
    
    phase_L = -np.angle(thL)
    phase_R = -np.angle(thR)
   
    #Global phase    
    phase= (phase_L + phase_R) / 2.

    #Polarization rotation
    zeta = (phase_L - phase_R)/4.
    
    #Ratio of amplitudes
    r = np.abs(thL/thR)
    
    #Inclination
    a22 = (1. - r) / (1. + r)
    
    return phase, a22, zeta, r


def phases_fs(waveform_generator,detec,rang_fs,i_best,mass_1=30,mass_2=30,a_1=0,a_2=0,tilt_1=0,tilt_2=0,phi_12=0,phi_jl=0,luminosity_distance=500,theta_jn=0.1,psi = 0.1,phase_ref=0.0,geocent_time=0.,ra=0.,dec=0.,duration = 4.0,sampling_frequency = 1024.0):
    """
    Compute the phase at a new frequency.
    For details see Appendix B in https://arxiv.org/pdf/2308.06616.pdf 

    Parameters
    ----------
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        waveform generator
    detec: str
        detector name
    rang_fs: np.ndarray
        range of frequencies
    i_best: int
        index of new frequency
    mass_1: float
        primary mass
    mass_2: float     
        secondary mass
    a_1: float
        primary aligned spin
    a_2: float
        secondary aligned spin
    tilt_1: float
        primary spin tilt
    tilt_2: float
        secondary spin tilt
    phi_12: float
        relative spin phase
    phi_jl: float
        angle between J and L
    luminosity_distance: float
        luminosity distance
    theta_jn: float
        angle between J and N
    psi: float
        polarization angle
    phase_ref: float
        phase at reference frequency
    geocent_time: float
        geocentric time
    ra: float
        right ascension
    dec: float
        declination
    duration: float
        duration of the signal
    sampling_frequency: float
        sampling frequency of the signal

    Returns
    -------
    phase: float
        phase at new frequency.
        See Eq. B3 of https://arxiv.org/pdf/2308.06616.pdf 
    a22: float
        Tangent of beta, the ratio of the semi-major and minor axes.
        It is a function of the inclination at new frequency. 
        See Eqs. B4 and B14 of https://arxiv.org/pdf/2308.06616.pdf
    zeta: float
        polarization rotation at new frequency.
        See Eq. B5 of https://arxiv.org/pdf/2308.06616.pdf
    r: float
        ratio of amplitudes L and R at new frequency.
        See below Eq. B4 of https://arxiv.org/pdf/2308.06616.pdf
    phase_D_unwrap[-1]: float
        phase at detector D at highest frequency.
        See Eq. B11 of https://arxiv.org/pdf/2308.06616.pdf 
    phase_D_unwrap[0]: float
        phase at detector D at lowest frequency.
        See Eq. B11 of https://arxiv.org/pdf/2308.06616.pdf 
    """
    phase, a22, zeta, r = phases_f_ev(waveform_generator,rang_fs,mass_1=mass_1,mass_2=mass_2,a_1=a_1,a_2=a_2,tilt_1=tilt_1,tilt_2=tilt_2,phi_12=phi_12,phi_jl=phi_jl,luminosity_distance=luminosity_distance,theta_jn=theta_jn,psi = psi,phase_ref=phase_ref,geocent_time=geocent_time,ra=ra,dec=dec)
    
    #Phase at detector    
    one = np.ones(1)
    Fp, Fx = gwutils.FpFx(detec,ra*one, dec*one, psi*one, geocent_time*one)
    
    phase_D = phase_d(phase,a22,zeta,Fp,Fx)

    #Unwrapped phase
    phase_D_unwrap = phases_unwrap(phase_D)
    
    return phase[i_best], a22[i_best], zeta[i_best], r[i_best], phase_D_unwrap[-1], phase_D_unwrap[0]

"""
Phase at detector
"""
def chi22_det(Fp,Fx,a22,zeta):
    """
    Chi_22 at detector D.
    See Eq. B10 and B11 of https://arxiv.org/pdf/2308.06616.pdf

    Parameters
    ----------
    Fp : np.ndarray
        plus polarization of antenna pattern function
    Fx : np.ndarray
        cross polarization of antenna pattern function
    a22 : np.ndarray
        Tangent of beta, the ratio of the semi-major and minor axes.
        It is a function of the inclination at new frequency.
        See Eqs. B4 and B14 of https://arxiv.org/pdf/2308.06616.pdf
    zeta : np.ndarray
        polarization rotation at new frequency.
        See Eq. B5 of https://arxiv.org/pdf/2308.06616.pdf

    Returns
    -------
    chi22_det : np.ndarray
        Chi_22 at detector D.
        See Eq. B10 and B11 of https://arxiv.org/pdf/2308.06616.pdf
    """
    Fp_rot = +np.cos(2*zeta)*Fp + np.sin(2*zeta)*Fx
    Fx_rot = -np.sin(2*zeta)*Fp + np.cos(2*zeta)*Fx

    return np.arctan2(a22*Fx_rot,Fp_rot)

def phase_d(phase,a22,zeta,Fp,Fx):
    """
    Compute the phase at a detector.
    For details see Appendix B in https://arxiv.org/pdf/2308.06616.pdf

    Parameters
    ----------
    phase : np.ndarray
        phase at new frequency
    a22 : np.ndarray
        Tangent of beta, the ratio of the semi-major and minor axes.
        It is a function of the inclination at new frequency.
        See Eqs. B4 and B14 of https://arxiv.org/pdf/2308.06616.pdf
    zeta : np.ndarray
        polarization rotation at new frequency.
        See Eq. B5 of https://arxiv.org/pdf/2308.06616.pdf
    Fp : np.ndarray
        plus polarization of antenna pattern function
    Fx : np.ndarray
        cross polarization of antenna pattern function

    Returns
    -------
    phase_D : np.ndarray
        phase at detector D at new frequency.
        See Eq. B11 of https://arxiv.org/pdf/2308.06616.pdf
    """
    #phase : phase at f
    #a22 : inclination at f
    #zeta : polarization rotation at f
    #Fp, Fx: antenna pattern functions
    
    chi_d = chi22_det(Fp,Fx,a22,zeta)
    
    return np.mod(phase + chi_d,2.*np.pi)

"""
Phase unwrap
"""
def phases_unwrap(phase_fs):
    """
    Unwrap the phase over 2pi period.

    Parameters
    ----------
    phase_fs : np.ndarray
        phase at new frequencies

    Returns
    -------
    phase_new_fs : np.ndarray
        unwrapped phase at new frequencies
    """
    phase_new_fs = phase_fs*1.
    phase_diff = np.diff(phase_fs,n=1)
    for i in range(len(phase_diff)):
        if (phase_diff[i]<-1.*np.pi):
            phase_new_fs[i+1:] += 2*np.pi 
        if (phase_diff[i]>1.*np.pi):
            phase_new_fs[i+1:] -= 2*np.pi
    return phase_new_fs