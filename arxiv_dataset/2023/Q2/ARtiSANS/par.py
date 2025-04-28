import math
import numpy as np

#constants
gf = 1.1663787e-11 #Fermi Coupling constant [MeV^(-2)]
sinW2 = 0.23122 #Sin Weinberg angle squared
AlphaEM = 1/137.035 # EM fine structure constant
mN = 1.6726219e-27 #nucleon mass [kg]

#conversions
g2kg = 0.001
kg2g = 1e+3
erg2MeV = 624150.648
kpc2km  = 3.08567758e+16
cm2km = 1e-5
km2cm = 1e+5
J2eV = 6.24150913e+18
kBoltz = 1.38064852e-23
Msol = 1.9891e+30

#natural units
m2eV = 1.9732705e-7
kg2eV = 0.561e+36
s2eV = 0.658e-15
K2eV = 0.862e-4
MeV2erg = 1.60218e-6
erg2MeV = 624151
CMeVQ = 1.9773106606045893e+68 #conversion MeV^-5 to MeV km^-3 s^-1