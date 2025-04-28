"""Library containing useful functions."""

import numpy as np
from numpy import pi, sqrt, exp, sin, cos, sinh, tanh, arccos
from scipy.fft import fft

# Some constants and parameters:
c = 299792.458 # Speed of light (in km/s)
v_0 = 220 # Local circular rotation speed (in km/s)
v_lab = 233 # Laboratory speed relative to the galactic rest frame (in km/s)


def get_random_distributions(v_0, v_lab, nu_a, N_a):
    """Generate random velocities, frequencies, and phases of axionlike particles."""
    v_x = np.random.normal(0, v_0/sqrt(2), N_a) # X components of velocities
    v_y = np.random.normal(0, v_0/sqrt(2), N_a) # Y components of velocities
    v_z = np.random.normal(0, v_0/sqrt(2), N_a) # Z components of velocities
    
    # Galilean transformation between the galactic frame and the lab frame:
    v_z = v_z - v_lab
    
    # Speeds of axionlike particles:
    v = sqrt(v_x**2 + v_y**2 + v_z**2)
    
    # Frequencies of axionlike particles, Eq. (2):
    nu = (1 + v**2/(2*c**2))*nu_a
    
    # Phases of axionlike particles:
    phi = np.random.uniform(0, 2*pi, N_a)
    
    return v_x, v_y, v_z, nu, phi


def get_v_components(v_x, v_y, v_z, alpha):
    """Calculate the parallel and perpendicular components of the velocities."""
    v = sqrt(v_x**2 + v_y**2 + v_z**2) # Speeds of axionlike particles
    
    # Components parallel and perpendicular to B_0, Eq. (17):
    v_par = v_x*sin(alpha) + v_z*cos(alpha) # Parallel components
    v_perp = sqrt(v**2 - v_par**2) # Perpendicular components
    return v_par, v_perp


def get_signal(nu, phi, time, v=None):
    """Generate the time-domain signal."""
    N_a = len(nu) # Number of axionlike particles
    s = np.zeros(time.shape)
    
    if v is not None: # The case of gradient coupling, Eq. (4)
        for i in range(N_a):
            s += sqrt(2)*v[i]*sin(2*pi*nu[i]*time + phi[i])/c
    else: # The case of non-gradient couplings, Eq. (3)
        for i in range(N_a):
            s += cos(2*pi*nu[i]*time + phi[i])
    return s/sqrt(N_a)


def calculate_psd(signal, T):
    """Calculate the PSD from the time-domain signal."""
    # We assume a single-sided PSD normalized as in Eq. (5)
    N = len(signal)
    return 2*T*(np.abs(fft(signal)[0:N//2])/N)**2


def get_lineshape(v_0, v_lab, nu_a, nu, case="non-grad", alpha=0):
    """Calculate analytical lineshapes."""
    assert case in ["non-grad", "grad_par", "grad_perp"], "Case should be 'non-grad', 'grad_par', or 'grad_perp'!"
    
    beta = 2*c*v_lab*sqrt(2*(nu - nu_a)/nu_a)/v_0**2 # Eq. (13)
    
    if case == "non-grad": # Non-gradient case, Eq. (12)
        return 2*c**2*exp(-(0.5*beta*v_0/v_lab)**2 - (v_lab/v_0)**2)*sinh(beta)/(sqrt(pi)*v_0*v_lab*nu_a)
    elif case == "grad_par": # Parallel gradient case, Eq. (19)
        factor = cos(alpha)**2 - (1/tanh(beta) - 1/beta)*(2 - 3*sin(alpha)**2)/beta
        return (4*c**2/(v_0**2 + 2*(v_lab*cos(alpha))**2))*(nu/nu_a - 1)*factor*get_lineshape(v_0, v_lab, nu_a, nu)
    elif case == "grad_perp": # Perpendicular gradient case, Eq. (20)
        factor = sin(alpha)**2 + (1/tanh(beta) - 1/beta)*(2 - 3*sin(alpha)**2)/beta
        return (2*c**2/(v_0**2 + (v_lab*sin(alpha))**2))*(nu/nu_a - 1)*factor*get_lineshape(v_0, v_lab, nu_a, nu)


def get_psd(v_0, v_lab, nu_a, nu, case="non-grad", alpha=0):
    """Calculate analytical PSDs."""
    assert case in ["non-grad", "grad_par", "grad_perp"], "Case should be 'non-grad', 'grad_par', or 'grad_perp'!"
    
    # Calculate the total signal power:
    if case == "non-grad": # Non-gradient case, Eq. (14)
        P = 0.5
    elif case == "grad_par": # Parallel gradient case, Eqs. (21) and (23)
        P = (v_0**2 + 2*(v_lab*cos(alpha))**2)/(2*c**2)
    elif case == "grad_perp": # Perpendicular gradient case, Eqs. (22) and (24)
        P = (v_0**2 + (v_lab*sin(alpha))**2)/c**2
    
    return P*get_lineshape(v_0, v_lab, nu_a, nu, case, alpha)


def get_vlab(lambda_lab, phi_lab, time):
    """Calculate the vector v_lab for specific location and time."""
    
    # Values of b0, b1, and psi on January 1:
    b0 = 0.7589
    b1 = 0.6512
    psi = -3.5336
    
    # Time offset (in hours):
    tau = (78.2 + 72.4)*24
    
    # Speeds in km/s:
    v_sun = 233 # Speed of the Sun in the galactic rest frame
    v_earth = 29.8 # Orbital speed of the Earth
    
    # Angular speeds (in radian/hour):
    omega_d = 2*pi/(0.9973*24) # Earth's rotational angular speed
    omega_y = 2*pi/(365*24) # Earth's orbital angular speed
    
    # Lab-frame velocity relative to the galactic frame, Eq. (33):
    v_lab = sqrt(v_sun**2 + v_earth**2 + 0.982*v_sun*v_earth*cos(omega_y*(time - tau)))
    
    # Angles alpha_N, alpha_W, and alpha_Z given by Eqs. (34)-(36):
    alphas = np.zeros((3, len(time)))
    alphas[0] = arccos(b0*cos(lambda_lab) - b1*sin(lambda_lab)*cos(omega_d*time + phi_lab + psi))
    alphas[1] = arccos(b1*sin(omega_d*time + phi_lab + psi))
    alphas[2] = arccos(b0*sin(lambda_lab) + b1*cos(lambda_lab)*cos(omega_d*time + phi_lab + psi))
    
    return v_lab, alphas


def get_power(v_lab, alphas, case):
    """Calculate the normalized total signal power."""
    assert case in ["par", "perp"], "Case should be 'par' or 'perp'!"
    
    if case == "par": # Parallel gradient case, Eq. (23)
        P = (v_0**2 + 2*(v_lab*cos(alphas))**2)/(2*c**2)
    elif case == "perp": # Perpendicular gradient case, Eq. (24)
        P = (v_0**2 + (v_lab*sin(alphas))**2)/c**2
    
    # Normalization:
    P /= np.max([P[0], P[1], P[2]])
    
    return P

