
from numerics.backend_mpmath import MPMathBackend
import numerics as bd

from nlft import NonLinearFourierSequence

import riemann_hilbert, weiss

#set_backend(MPMathBackend(mp.mp)) # default is numpy

with bd.workdps(90):
    nlft = NonLinearFourierSequence([1, 2, 3])
    _, b = nlft.transform()

    a, c = weiss.ratio(b)

    new_nlft = riemann_hilbert.inlft_hc(b, c)

    _, b2 = new_nlft.transform()

    print((b - b2).l2_norm())