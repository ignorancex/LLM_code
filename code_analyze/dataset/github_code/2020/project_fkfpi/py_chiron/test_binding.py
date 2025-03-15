#!/usr/bin/env python3

import chiron
import gvar as gv


def FF(x):
    if isinstance(x, gv.GVar):
        f = chiron.FF(x.mean)
        stepSize = 1e-7
        dfdx = 0.5*(chiron.FF(x.mean+stepSize) - chiron.FF(x.mean-stepSize))/stepSize
        return gv.gvar_function(x, f, dfdx)

    else:
        return chiron.FF(x)

print('FF(0.7**2) = ',FF(0.7**2))

print('FF(0.2**2) = ',FF(0.2**2))
print('FF(gvar(0.2,0.1)**2) = ',FF(gv.gvar(0.2, 0.1) ** 2))
print('FF(gvar(0.8,0.1)**2) = ',FF(gv.gvar(0.8, 0.1) ** 2))
