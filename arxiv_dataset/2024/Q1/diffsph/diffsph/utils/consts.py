import numpy as np

# Fundamental physics constants
c0 = 29979245800  # cm/s
hbar = 1.05457182e-27  # gr cm^2/s
kB = 1.380649e-16  # erg/K

# More constants of nature
e0 = 4.80326e-10  # gr^(1/2)cm^(3/2)/s
me = 9.1093837e-28  # gr
TCMB = 2.725  # K
GNw = 113.285885  # km^2/s^2 kpc^-2 (GeV/c^2/cm^3)^-1

# Unit conversions
Gyr_to_sec = 3.1556926e16
erg_to_GeV = 624.150907
cm_to_kpc = 3.24077929e-22
rad_to_arcmin = 180 * 60 / np.pi

# Further useful constants
Bc = np.sqrt(8 * np.pi**3 * (kB * TCMB)**4 / c0**3 / 15 / hbar**3) * 1e6  # muG
b0 = 32 * np.pi**3 * e0**4 * (kB * TCMB)**4 / 135 / hbar**3 / me**4 / c0**10 / erg_to_GeV  # GeV^-1 s^-1
eps0 = np.sqrt(2 * np.pi * me**3 * c0**5 * 1e9 / 3 / e0 / 1e-6) * erg_to_GeV  # GeV muG^1/2 GHz^-1/2
X0 = 1e9 * 4 * np.sqrt(3) * e0**3 * 1e-6 / me / c0**2 / b0  # erg GHz^-1 GeV muG^-1
