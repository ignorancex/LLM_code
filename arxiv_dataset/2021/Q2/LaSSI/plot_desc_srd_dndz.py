# Lens and source dn/dz from the LSST DESC SRD

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

plt.figure(0)
#
z = np.linspace(0., 4., 101)
#
# lens sample
z0 = 0.28
alpha = 0.9
ngal = 48.  # per sq arcmin
zMin = 0.2
zMax = 1.2
f = lambda z: z**2 * np.exp(-(z/z0)**alpha) * (z>=zMin) * (z<=zMax)
lenses = f(z)
lenses /= integrate.quad(f, zMin, zMax, epsabs=0., epsrel=1.e-3)[0]
lenses *= ngal
print "lenses: ngal=", np.trapz(lenses, z), "arcmin^-2"
plt.plot(z, lenses, label=r'lenses, $n_\text{gal}=48$arcmin$^{-2}$')
#
# source sample
z0 = 0.13
alpha = 0.78
zMin = 0.
zMax = 4.
ngal = 27.  # per square arcmin
f = lambda z: z**2 * np.exp(-(z/z0)**alpha) * (z>=zMin) * (z<=zMax)
sources = f(z)
sources /= integrate.quad(f, zMin, zMax, epsabs=0., epsrel=1.e-3)[0]
sources *= ngal
print "sources: ngal=", np.trapz(sources, z), "arcmin^-2"
print "source within lens redshift range: ngal", np.trapz(sources*(z>=0.2)*(z<=1.2), z), "arcmin^-2"
plt.plot(z, sources, label=r'sources, $n_\text{gal}=27$arcmin$^{-2}$')
#
plt.legend(fontsize=18)
plt.xlabel(r'$z$')
plt.ylabel(r'$dN/dz/d\Omega$ [arcmin$^{-2}$]')

plt.show()
