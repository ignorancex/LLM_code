# Copyright 2019 by Ben Granett, granett@gmail.com
# All rights reserved.
# This file is part of the VIPERS-SHAM project,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import os
import logging
import numpy as N

def lowerwater(z, plevels,alpha=0.99):
    """ """
    if N.max(z)==0:
        return [0]*len(plevels)
    tot = N.sum(z)
    levels = []
    for l in plevels:
        h = z.max()
        t = tot*l
        ll = 0
        if z.sum()<t:
            levels.append(0)
        while z[z>h].sum()<t:
            h*=alpha
            ll+=1
            if ll>1e6:
                print "uhoh!",ll,h,t
                break

        levels.append(h)

    return levels
