"""Units and constants

Standard units: [cm] [MHz] [g] [K] [rad]
SNR : [kpc] [yr]
"""

from astropy.coordinates import SkyCoord
from astropy import units as u


#===== constants =====

kb = 1.38065e-28 # [cm^2 MHz^2 g / K]
c0 = 29979.2 # [cm MHz]
c0_kpc_yr = 0.000306601 # [kpc/yr]
hbar = 1.05457e-33 # [cm^2 MHz g]
PI = 3.141592653589793
EPSILON = 1e-30


#===== derived =====

hour = 3.6e9 # [MHz^-1]
sec = 1e6 # [MHz^-1]
GeV = 1.60218e-15 # [cm^2 MHz^2 g]
invGeV = 1/GeV # [cm^-2 MHz^-2 g^-1]
eV = GeV * 1e-9 # [cm^2 MHz^2 g]
Jy = 1e-35 # [MHz^2 g]
kpc = 3.08568e21 # [cm]
deg = PI/180 # [rad]
arcmin = deg/60 # [rad]


#===== astrophysical =====

r_Sun = 8.22 # [kpc]
sigmad_over_c = 116/300000 # [1] | (km/s) / (km/s)

c_NGP = SkyCoord(l=0*u.deg, b=90*u.deg, frame='galactic') # north galactic pole
c_NCP = SkyCoord(ra=0*u.deg, dec=90*u.deg, frame='icrs') # north celestial pole
ra_NGP = float(c_NGP.icrs.ra/u.rad)
dec_NGP = float(c_NGP.icrs.dec/u.rad)
l_NCP = float(c_NCP.galactic.l/u.rad)
b_NCP = float(c_NCP.galactic.b/u.rad)


#===== this project =====

gagg_CAST = 6.6e-11 * invGeV # [cm^-2 MHz^-2 g^-1]
nu_haslam = 408 # [MHz]