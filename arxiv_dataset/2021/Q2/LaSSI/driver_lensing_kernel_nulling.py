import universe
reload(universe)
from universe import *

import projection_kernel
reload(projection_kernel)
from projection_kernel import *


##################################################################################

u = UnivPlanck15()

##################################################################################
# Nulling low-z signal for single-source planes

# single source planes
z = np.array([0.2, 0.4, 0.6])
w0 = WeightLensSingle(u, z_source=z[0], name="gallens0")
w1 = WeightLensSingle(u, z_source=z[1], name="gallens1")
w2 = WeightLensSingle(u, z_source=z[2], name="gallens2")

# linear combination to null the low-z signal
w_combined = WeightLensSingle(u, z_source=z[2], name="combined")
chi = u.bg.comoving_distance(z)
alpha = (1./chi[2] - 1./chi[0]) / (1./chi[0] - 1./chi[1])
w_combined.f = lambda a:  w2.f(a) + alpha * w1.f(a) - (1.+alpha) * w0.f(a)
w_combined.plotW()


##################################################################################
# What about source bins with wider redshift distribution?





