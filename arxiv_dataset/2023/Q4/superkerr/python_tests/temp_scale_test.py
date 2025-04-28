from astropy import constants, units

G = constants.G
sigma_sb = constants.sigma_sb
k = constants.k_B
c = constants.c
Ms = constants.M_sun
h = constants.h

rg = G*Ms/c**2
pi = 3.14159265359
Mdot_scale = 1e15 * units.kg/units.s

kT_scale = k * (3 * G * Ms * Mdot_scale / (8 * pi * rg**3 * sigma_sb))**0.25 
print(kT_scale.to(units.keV))