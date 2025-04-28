from numpy import array, sqrt, pi, exp, interp, loadtxt, zeros, shape, ones
from numpy import logspace, linspace, log10
from scipy.special import erf

# Directory structure:
data_dir = '../data/'
limit_dir = '../data/limit_data/'
pltdir = '../plots/'
pltdir_png = pltdir+'plots_png/'

# Constants
m_p = 0.9315*1e6
m_p_keV = 0.9315*1e6
m_e = 511.0 # keV
c_m = 2.99792458e8 # speed of light in m/s
c_cm = c_m*100.0 # speed of light in cm/s
c_km = c_m/1000.0 # speed of light in km/s
GeV_2_kg = 1.0e6*1.783e-33 # convert GeV to kg
alph = 1.0/137.0 # fine structure constant
m_p_kg = 1.660538782e-27 # amu in kg
a0 = 0.268173 # Bohr radius keV^-1
N_A = 6.02214e23 # Avocado's constant
sinTheta_Wsq = 0.2387e0 # sin^2(Theta_W) weinberg angle
G_F_GeV = 1.16637e-5 # GeV**-2 ! Fermi constan in GeV
Jan1 = 2458849.5 # January 1st 2020
seconds2year = 365.25*3600*24
eV2J = 1.6e-19
AstronomicalUnit = 1.49597892e11 # Astronomical Unit
EarthRadius = 6371e3 # Earth Radius
Msun = 2.0e30 # Solar mass (kg)
bigG = 6.67e-11*(1.0e3)**(-3)

K_2_keV = 8.617333262e-5/1e3 # convert Kelvin to keV
cm_2_keV = 1.0/(1e6*1.9732e-14) # convert cm to keV^-1
fsc = 1.0/137.036 # fine structure constant
amu_grams = 1.66054e-24 # amu in grams
AU_cm = 100*1.495978707e11 # cm
keV_2_s = 1000/6.5821e-16 # keV to s^-1
cm_2_keV = 1.0/(1e6*1.9732e-14) # keV to cm^-1
Rsol_keV = 696340*1000*100*cm_2_keV # solar radius to keV^-1

Gauss_2_keV = 1.95e-20*(1e6)**2.0
Tesla_2_keV = 1e4*Gauss_2_keV
