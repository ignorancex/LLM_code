"""User-facing functions"""

import numpy as np

from .core.bndfun import Bndfun
from .core.chebfun import Chebfun
from .core.utilities import Domain
from .core.settings import _preferences
from .core.chebtech import Chebtech2
from .core.settings import DefaultPreferences


def chebfun(f = None, domain = _preferences.domain, n = None, prefs = _preferences, initcoeffs = False):
    """Chebfun constructor"""
    # chebfun()
    _preferences = prefs

    if f is None:
        cf = Chebfun.initempty()
        _preferences.reset()
        return cf

    # chebfun(lambda x: f(x), ... )
    if hasattr(f, "__call__"):
        cf = Chebfun.initfun(f, domain, n, prefs = prefs)
        _preferences.reset()
        return cf

    # chebfun('x', ... )
    if isinstance(f, str) and len(f) == 1 and f.isalpha():
        if n:
            cf = Chebfun.initfun(lambda x: x, domain, n, prefs = prefs)
            _preferences.reset()
            return cf
        else:
            cf = Chebfun.initidentity(domain, prefs = prefs)
            _preferences.reset()
            return cf

    if initcoeffs:
        if isinstance(f, np.ndarray) and len(f.shape) == 2:
            if f.shape[1] == 1:
                cf = Chebfun.initcoeffs(f, domain, prefs = prefs)
                _preferences.reset()
                return cf
            else:
                cf = np.array([Chebfun.initcoeffs(f[:,i], domain, prefs = prefs) for i in range(f.shape[1])])
                _preferences.reset()
                return cf
            
    if isinstance(f, np.ndarray) and len(f.shape) == 2:
        if f.shape[1] == 1:
            cf = Chebfun.initvalues(f, domain, prefs = prefs)
            _preferences.reset()
            return cf
        else:
            cf = np.array([Chebfun.initvalues(f[:,i], domain, prefs = prefs) for i in range(f.shape[1])])
            _preferences.reset()
            return cf
        
    try:
        # chebfun(3.14, ... ), chebfun('3.14', ... )
        cf = Chebfun.initconst(float(f), domain, prefs = prefs)
        _preferences.reset()
        return cf
    except (OverflowError, ValueError):
        raise ValueError(f"Unable to construct const function from {{{f}}}")


def pwc(domain=[-1, 0, 1], values=[0, 1], prefs = _preferences):
    """Initialise a piecewise-constant Chebfun"""
    _preferences = prefs
    funs = []
    intervals = [x for x in Domain(domain).intervals]
    for interval, value in zip(intervals, values):
        funs.append(Bndfun.initconst(value, interval, prefs = prefs))
    cf = Chebfun(funs)
    _preferences.reset()
    return cf

def scaleNodes(x, dom):
    # SCALENODES   Scale the Chebyshev nodes X from [-1,1] to DOM.
    if dom[0] == -1 and dom[1] == 1:
        # Nodes are already on [-1, 1]
        return x

    # Scale the nodes:
    return dom[1]*(x + 1)/2 + dom[0]*(1 - x)/2

def chebpts(n, dom = DefaultPreferences.domain, type = 2):
    #chebpts    Chebyshev points.
    #   chebpts(N) returns N Chebyshev points of the 2nd-kind in [-1,1].
    #
    #   chebpts(N, D), where D is vector of length 2 and N is a scalar integer,
    #   scales the nodes and weights for the interval [D(1),D(2)].

    ##################################################################################
    #   [Mathematical reference]:
    #   Jarg Waldvogel, "Fast construction of the Fejer and Clenshaw-Curtis
    #   quadrature rules", BIT Numerical Mathematics, 46, (2006), pp 195-202.
    ##################################################################################

    # Create a dummy CHEBTECH of appropriate type to access static methods.
    if type == 2:
        f = Chebtech2
    else:
        raise Exception('CHEBFUN:chebpts:type, Unknown point type.') 

    if np.size(n) == 1:         # Single grid
        # Call the static CHEBTECH.CHEBPTS() method:
        x = f._chebpts(n)
        # Scale the domain:
        x = scaleNodes(x, dom)
    else:                  # Piecewise grid.
        raise NotImplementedError
    return x