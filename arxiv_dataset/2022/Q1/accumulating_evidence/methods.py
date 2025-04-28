"""
Pythranized functions for signal and log-like computation
=========================================================
"""

import numpy as np


# signal. crystal ball settings
alow = ahigh = 1.5
ealow = np.exp(-0.5 * alow**2)
eahigh = np.exp(-0.5 * ahigh**2)
nlow = 5
nhigh = 9
flow = alow / nlow
clow = nlow / alow - alow
fhigh = ahigh / nhigh
chigh = nhigh / ahigh - ahigh

#pythran export crystal(float64[], float, float)
def crystal(energy, mass, sigma):
    t = (energy  - mass) / sigma
    y = np.empty_like(t)
    mask = t < -alow
    y[mask] = ealow / (flow * (clow - t[mask]))**nlow
    mask = (t > -alow) & (t < ahigh)
    y[mask] = np.exp(-0.5 * t[mask]**2)
    mask = t > ahigh
    y[mask] = eahigh / (fhigh * (chigh + t[mask]))**nhigh
    return y

#pythran export loglike(int64[], float64[])
def loglike(observed, events):
    log_events = np.log(events / observed)
    log_events[observed == 0] = 0.
    return np.dot(log_events, observed) - (events - observed).sum()

#pythran export jac_loglike(int64[], float64[], float64[])
def jac_loglike(observed, signal, total):
    return (observed * signal / total - signal).sum()
