from fit_lin import pofk_enhancement_linear
import numpy as np

# fR0 and redshift
fofr0 = 1e-5
z = 0.0

# k-range to output 
nk = 30
kmin = 0.01
kmax = 10.0

print "For fR0 = ", fofr0, " and z = ", z, " we have the following (k, ratio(k)):"
for i in range(nk):
  k = np.exp(np.log(kmin) + np.log(kmax/kmin) * i /(nk-1.0))
  print k,  pofk_enhancement_linear(z, fofr0, k)


print pofk_enhancement_linear(z, fofr0, 1.0)
