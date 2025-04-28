from astropy import constants, units

G = constants.G
sigma_sb = constants.sigma_sb
k = constants.k_B
c = constants.c
Ms = constants.M_sun
h = constants.h

rg = G*Ms/c**2
pi = 3.14159265359

norm = sigma_sb / (pi * k**4) * (rg/(units.kpc))**2 * (units.keV)**4 * 1/units.keV 
print(norm.to(1/units.cm**2 * 1/units.s))