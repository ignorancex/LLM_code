import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

sys.path.append('/Users/belugawhale/Documents/GitHub/Llamaradas-Estelares/') #Edit this to your own file path
from Flare_model import flare_model

__all__ = ['llamarades_model', 'pietras_double', 'run_fit']


def llamarades_model(t, tpeak, fwhm, ampl):
    # Fits the single peak flare model from Tovar Mendoza et al. (2022)
    fm = flare_model(t, tpeak, fwhm, ampl)
    return fm[np.isfinite(fm)==True]

def pietras_double(t, A1, B1, C1, D1, A2, B2, C2, D2):
    # Fits the double peak flare model from Pietras et al. (2022)

    def exp1(x, b, c):
        num = -(x-b)**2
        denom = c**2
        return np.exp(num/denom)

    def exp2(x, t, d):
        return np.exp(-d*(t-x))

    def f(x, time, A1, B1, C1, D1, A2, B2, C2, D2, t):

        term1 = A1*exp1(x, B1, C1)
        term2 = exp2(x, t, D1)

        term3 = A2*exp1(x, B2, C2)
        term4 = exp2(x, t, D2)

        f_t = (term1*term2) + (term3*term4)
        return f_t

    fm = np.zeros(len(t))
    for i in range(len(t)):
        fm[i] = quad(f, 0, t[i], args=(t, A1, B1, C1, D1, A2, B2, C2, D2, t[i]))[0]

    return fm

def run_fit(time, flux, err, tpeak, ampl, model='TM22'):
    # Runs scipy.optimize.curve_fit for a given flare model

    if model == 'TM22':
        # run fit for Tovar Mendoza et al. (2022) model
        init_vals = [tpeak, 0.005, np.abs(ampl)]
        bounds = ((tpeak-0.05, 1e-4, 1e-4),
                  (tpeak+0.05, 0.09, 2000.0))
        popt, pcov = curve_fit(llamarades_model, time, flux, p0=init_vals,
                               maxfev=15000, bounds=bounds)
        results = {'tpeak':popt[0][0], 'fwhm':popt[0][1],
                   'amp':popt[2]}

    elif model == 'P22':
        # run fit for Pietras et al. (2022) double-peak model

        # initialize values for fit
        A1, A2 = ampl*400, ampl/100.0
        B1, B2 = tpeak-time[0]-0.002, tpeak-time[0]+0.005
        C1, C2 = 0.002, 0.001
        D1, D2 = 200.0, 10000.0

        # set the bounds for the fit
        Alow, Aupp = ampl/1000.0, ampl*1000.0
        Blow, Bupp = tpeak-time[0]-0.01, tpeak-time[0]+0.1
        Clow, Cupp = 0.0001, 0.005
        Dlow, Dupp = 10.0, 1e6

        init_vals = [A1, B1, C1, D1, A2, B2, C2, D2]
        bounds = ((Alow, Blow, Clow, Dlow, Alow, Blow, Clow, Dlow),
                  (Aupp, Bupp, Cupp, Dupp, Aupp, Bupp, Cupp, Dupp))

        popt, pcov = curve_fit(pietras_double, time-time[0], flux, p0=init_vals,
                               maxfev=1500, bounds=bounds)

        parameters = ['A1', 'B1', 'C1', 'D1', 'A1', 'B2', 'C2', 'D2']
        results = {parameters[x]:popt[0][x] for x in range(len(parameters))}

    else:
        return('Model not implemented. Please use model = "TM22" or "P22".')

    perr = np.sqrt(np.diag(pcov))
    return results, perr
