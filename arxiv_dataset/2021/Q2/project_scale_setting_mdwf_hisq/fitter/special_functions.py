import gvar as gv
import numpy as np
import scipy.special as ss


### volume corrections
# gvar version of modified Bessel function of the second kind
def fcn_Kn(n, g):
    # input is a gvar scalar
    if isinstance(g, gv._gvarcore.GVar):
        f = ss.kn(n, gv.mean(g))
        dfdg = ss.kvp(n, gv.mean(g), 1)
        return gv.gvar_function(g, f, dfdg)

    # input is a gvar vector
    elif hasattr(g, "__len__") and isinstance(g[0], gv._gvarcore.GVar):
        f = ss.kn(n, gv.mean(g))
        dfdg = ss.kvp(n, gv.mean(g), 1)
        return np.array([gv.gvar_function(g[j], f[j], dfdg[j]) for j in range(len(g))])

    # input is not a gvar variable
    else:
        return ss.kn(n, gv.mean(g))

# I(m) in notes: FV correction to tadpole integral
def fcn_I_m(xi, L, mu, order):
    c = [None, 6, 12, 8, 6, 24, 24, 0, 12, 30, 24]
    m = np.sqrt(xi *mu**2)

    output = np.log(xi)

    for n in range(1, np.min((order+1, 11))):
        output = output + (4 *c[n]/(m *L *np.sqrt(n))) *fcn_Kn(1, m *L *np.sqrt(n))
    return output