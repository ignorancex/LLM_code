'''
NAME:
-----
constants

PURPOSE:
--------
Define a bunch of physical constants (cgs) and unit conversions.
================================================================================
'''
import numpy as np

# Physical constants (cgs)
c       = 2.99792458e10        # Speed of light [cm s^-1]
G       = 6.67384e-8           # Gravitational constant [cm^3 g^-1 s^-2]
kB      = 1.3806488e-16        # Boltzmann constant [erg K-1]
h       = 6.62606957e-27       # Planck's constant [erg s]
hbar    = 1.054571726e-27      # Planck's constant / (2 pi) [erg s]
fsc     = 7.2973525698e-3      # Fine structure constant = e^2/(hbar c)
mp      = 1.672621777e-24      # Proton mass [g]
me      = 9.10938291e-28       # Electron mass [g]
mH      = 1.673534e-24         # Hydrogen mass [g]
re      = 2.8179403267e-13     # Classical electron radius = e^2/(m c^2) [cm]
sigmaSB = 5.670373e-5          # Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]
sigmaT  = 0.665245854533e-24   # Thomson cross section [cm^2]
lambC   = 2.4263102175e-10     # Compton wavelength for electrons = h/(m c) [cm]
kR      = sigmaT / mp          # Kramers's opacity (e- scattering) [cm^2 g^-1]
mu      = 0.615                # Fully ionized gas with cosmic abundance
a       = 7.5657e-15           # Radiation constant [erg cm^-3 K^-4]
e       = 4.80320425e-10       # Elementary charge [statC = g^1/2 cm^3/2 s^-1 = erg^1/2 cm^1/2]
mu_B    = e * hbar / (2.0 * me * c)  # Bohr magneton [erg Gauss^-1]
Ry      = 13.60569253          # Rydberg [eV]

# Unit conversions
sun2g      = 1.9891e33         # g Msun^-1
g2sun      = 1.0 / sun2g       # Msun g^-1
s2yr       = (3.1536e7)**-1.0  # yr s^-1
erg2eV     = 6.2415e11         # eV erg^-1
erg2keV    = 6.2415e8          # keV erg^-1
eV2erg     = 1.602176565e-12   # erg eV^-1
keV2erg    = 1.602176565e-9    # erg keV^-1
keV2eV     = 1.0e3             # eV keV^-1
eV2keV     = 1.0e-3            # keV eV^-1
K2keV      = kB * erg2keV      # keV K^-1
keV2K      = 1.0 / K2keV       # K keV^-1
Mbarn2cm2  = 1.0e-18           # cm^2 megabarn^-1
deg2rad    = np.pi / 180.0     # radians degrees^-1
rad2deg    = 180.0 / np.pi     # degrees radians^-1
kpc2cm     = 3.0857e21         # cm kpc^-1
cm2kpc     = 1.0 / kpc2cm      # kpc cm^-1
Jy2cgs     = 1.0e-23           # (erg s^-1 cm^-2 Hz^-1) Jy^-1
GHz2Hz     = 1.0e9             # Hz GHz^-1
cm2km      = 1.0e-5            # km cm^-1
km2cm      = 1.0e5             # cm km^-1
sec2hr     = 1.0 / 3600.0      # hr sec^-1
sec2day    = sec2hr / 24.0     # day sec^-1
sec2yr     = sec2day / 365.25  # yr sec^-1
yr2sec     = 1.0 / sec2yr      # sec yr^-1
arcsec2deg = 1.0 / 3600.0      # deg arcsec^-1
arcsec2rad = arcsec2deg * deg2rad  # rad arcsec^-1
Ry2keV     = 1.360569253e-2    # keV Rydberg^-1
keV2Ry     = 1.0 / Ry2keV      # Rydberg keV^-1
Kflux2cgs  = km2cm**2 / (10.0 * kpc2cm)**2
cgs2Kflux  = 1.0 / Kflux2cgs

