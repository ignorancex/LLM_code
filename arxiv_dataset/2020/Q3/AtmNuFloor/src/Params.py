from __future__ import print_function
from numpy import array, sqrt, pi, exp, interp, loadtxt, zeros, shape, ones
from numpy import logspace, linspace, log10
from scipy.special import erf

data_dir = '../data/'
recoil_dir = '../data/recoils/'
nufile_dir = "../data/neutrinos/"
mylimit_dir = '../data/WIMPlimits/mylimits/'


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


#==============================================================================#
# Set WIMP paramters, currently:
# Mass
# SI cross section
class WIMP:
    def __init__(self,m_chi,sigma_p):
        self.Mass = m_chi
        self.SICrossSection = sigma_p

    def RecoilDistribution(self,RD):
        self.RD = RD
#==============================================================================#



#==============================================================================#
# Set Nucleus params
class Nucleus:
    def __init__(self,xi,N,Z,J,Sp,Sn,name):
        self.IsotopicFraction = xi
        self.NumberOfNeutrons = N
        self.NumberOfProtons = Z
        self.MassNumber = N+Z
        self.NuclearSpin = J
        self.ExpProtonSpin = Sp
        self.ExpNeutronSpin = Sp
        if J>0.0:
            self.SDEnhancement = (4.0/3.0)*((J+1.0)/J)*(Sp-Sn)**2.0
        self.Name = name

# Some nuclei:
#              (xi,      N,   Z,    J,     Sp,      Sn, name)
He4 =   Nucleus(1.0,     2,   2, 0.01,  0.000,   0.000,'He')
F19 =   Nucleus(1.0,    10,   9,  0.5,  0.421,   0.045,'F')
Ar40 =  Nucleus(1.0,    21,  18,  0.00,    0.0,    0.0,'Ar')
S32 =   Nucleus(1.0,    16,  16,  0.01,    0.0,    0.0,'S')
Xe129 = Nucleus(0.265,  75,  54,  0.5,  0.046,   0.293,'Xe')
Xe131 = Nucleus(0.212,  75,  54,  1.5, -0.038,   0.242,'Xe')
Ge74 =  Nucleus(1.0,    41.0,  32,  0.0,   0.00,   0.000,'Ge')
Si28 =  Nucleus(1.0,  14.0,  14,  0.0,   0.00,   0.000,'Si')
Au197 = Nucleus(1.0,   118,  79,  1.5,    0.0,     0.0,'Au') # find sp and sn for this
#==============================================================================#









#==============================================================================#
# Set parameters of halo models
class Halo:
    def __init__(self,rho_0,v_LSR,sig_v,v_esc,v_pec):
        self.Density = rho_0
        self.RotationSpeed = v_LSR
        self.Dispersion = sig_v
        self.EscapeSpeed =  v_esc
        self.PeculiarVelocity = v_pec
        self.Normalisation = erf(v_esc/(sqrt(2.0)*sig_v))-\
                            sqrt(2.0/pi)*(v_esc/sig_v)*\
                            exp(-v_esc**2.0/(2.0*sig_v**2.0))

# Standard Halo Model (old parameters)
SHM = Halo(0.3,
        220.0,
        156.0,
        544.0,
        array([11.1,12.2,7.3]))

# Standard Halo Model (Gaia parameters)
SHMpp = Halo(0.55,
        233.0,
        164.8,
        580.0,
        array([11.1,12.2,7.3]))
#==============================================================================#










#=============================================================================#
# Current number of neutrinos sources for full neutrino floor calculation:
n_nu_tot = 15
# Neutrino files names:
nufile_root = "-1000.txt"
nuname = ["" for x in range(0,n_nu_tot)]
nuname[0] = "pp"
nuname[1] = "pep"
nuname[2] = "hep"
nuname[3] = "7Be1"
nuname[4] = "7Be2"
nuname[5] = "8B"
nuname[6] = "13N"
nuname[7] = "15O"
nuname[8] = "17F"
nuname[9] = "DSNB"
nuname[10] = "Atm"
nuname[11] = "GeoU"
nuname[12] = "GeoTh"
nuname[13] = "GeoK"
nuname[14] = "Reactor"
n_Enu_vals = 1000
# Mark which neutrinos are monochromatic
mono = zeros(n_nu_tot,dtype=bool)
mono[[1,3,4]] = True

# Set which neutrinos are Solar
whichsolar = zeros(n_nu_tot,dtype=bool)
whichsolar[0:8] = True

# Neutrino max energies (MeV):
NuMaxEnergy = array([0.42341,1.44,18.765,0.3843,0.8613,16.34,1.193,\
                    1.7285,1.7365,91.201,1.0e4,
                    4.54,2.33,1.3572,\
                    1.1418e1])

# Neutrino fluxes (cm-2 s-1 MeV-1) and uncertainties (%):
# (from Vinyoles et al (2017) Barcelona GS98 SSM) + Bergstrom global analysis
# Geo-nu from https://arxiv.org/pdf/1301.0365.pdf
NuFlux = array([5.98e10,1.44e8,7.98e3,4.93e8,4.50e9,5.16e6,\
                        2.78e8,2.05e8,5.29e6,85.7,10.7,\
                        4.34e6,4.23e6,20.54e6,\
                        3.06e6])
NuUnc = array([0.006, 0.01, 0.3,0.06, 0.06, 0.02, 0.15 ,\
                        0.17 ,0.2 ,0.5, 0.25,\
                        0.2,0.257,0.168,\
                        0.08])

# The GS98 SSM 8B flux and uncertainty just in case:
B8Flux_GS98 = 5.45e6
B8Unc_GS98 = 0.12
#=============================================================================#



#==============================================================================#
NuclearReactors = {'Hartlepool': [1310.0,54.640268, -1.191826],
                        'Heynsham': [2610.0,54.030124, -2.915303],
                        'Hunterston': [1288.0,55.722970, -4.891329],
                        'Torness': [1364.0,55.969103, -2.406619],
                        'Sizewell': [1250.0,52.213361, 1.619662],
                        'Dungeness': [1230.0,50.913796, 0.959705],
                        'Vandellos': [1087.0,40.951389, 0.866667],
                        'Asco':[4120.0,41.200000, 0.569444],
                        'Cofrentes': [1102.0,39.216667, -1.050000],
                        'Trillo': [1003.0,40.701111, -2.621944],
                        'Krsko': [730.0,45.938333, 15.515556],
                        'Tricastin': [3660.0,44.329722, 4.732222],
                        'Cruas': [4824.0,44.633056, 4.756667],
                        'Saint-Alban': [7634.0,45.404444, 4.755278],
                        'Bugey': [540.0,45.798333, 5.270833],
                        'Blayais': [3640.0,45.255833, -0.693056],
                        'Belleville': [2620.0,47.509722, 2.875000],
                        'Fessenheim': [1840.0,47.903056, 7.563056],
                        'Dampierre': [3748.0,47.733056, 2.516667],
                        'Gravelines':[5460.0,51.015278, 2.136111],
                        'Penly':[2764.0,49.976667, 1.211944],
                        'Beznau':[2*365,47.552107, 8.228492],
                        'Gosgen':[970.0,47.366111, 7.966667],
                        'Leibstadt':[1220.0,47.603056, 8.184722],
                        'Isar':[1485.0,48.605606, 12.293150],
                        'Neckarwestheim':[1400.0,49.041667, 9.175000],
                        'Emsland':[1400.0,52.474231, 7.317858],
                        'Grohnde': [1430.0,52.035278, 9.413333],
                        'Paks':[2000.0,46.572500, 18.854167]}

# Location class only has latitude and longitude at the moment
class Location:
    def __init__(self,lat,lon):
        self.Latitude = lat
        self.Longitude = lon

Boulby = Location(54.5561,-0.8216)
GranSasso = Location(42.4691, 13.5654)
#------------------------------------------------------------------------------#
