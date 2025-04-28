"""
Classes containing method for fitting channels
==============================================
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from methods import crystal, loglike, jac_loglike
from data import background, bins


class Channels():
    def __init__(self, observeds, resolutions):
        self.channels = [Channel(o, r) for o, r in zip(observeds, resolutions)]
        self.background_only = sum(c.background_only for c in self.channels)

    def fixed_mass_fit(self, mass, **kwargs):
        return sum(c.fixed_mass_fit(mass, **kwargs) for c in self.channels)

    def scan(self, masses, **kwargs):
        return np.array([self.fixed_mass_fit(m, **kwargs) for m in masses])

    def fit(self, masses, tol=1e-8, **kwargs):
        # find best fit from scan
        results = self.scan(masses, **kwargs)
        mass = masses[np.argmax(results)]

        # search inside region around it
        delta = masses[1] - masses[0]
        bounds = (mass - delta, mass + delta)

        r = minimize_scalar(lambda x: -self.fixed_mass_fit(x),
                            bounds=bounds,
                            method="bounded",
                            options={'xatol': tol, 'maxiter': np.inf},
                            **kwargs)

        assert bounds[0] <= r.x <= bounds[1]

        return -r.fun

class Channel():
    def __init__(self, observed, resolution):
        """
        @observed Observed counts
        @resolution Resolution in this channel
        """
        self.observed = observed
        self.resolution = resolution
        self.background_only = -2. * loglike(self.observed, background)

    def events(self, mass, signal_strength):
        shape = crystal(bins, mass, self.resolution)
        return signal_strength * shape / shape.sum() + background

    def fixed_mass_fit(self, mass, method="L-BFGS-B", tol=1e-8, **kwargs):
        signal = self.events(mass, 1.) - background
        nearest = np.argmin(np.abs(bins - mass))
        signal_strength = (self.observed[nearest] - background[nearest]) / signal[nearest]
        signal_strength = max(signal_strength, 0.)

        def fun(x):
            total = background + x[0] * signal
            return -2. * loglike(self.observed, total)

        def jac(x):
            total = background + x[0] * signal
            return -2. * jac_loglike(self.observed, signal, total)

        r = minimize(fun, signal_strength,
                     jac=jac,
                     tol=tol, method=method, bounds=[(0, np.inf)],
                     options={'ftol': tol, 'gtol': 1e-12, 'maxfun': np.inf, 'maxiter': np.inf}, **kwargs)

        # validate result - either gradient vanishes or at boundary

        assert np.isclose(r.jac, 0., atol=1e-3) or np.isclose(r.x[0], 0., atol=1e-3)
        assert r.x[0] >= 0.

        return self.background_only - r.fun
