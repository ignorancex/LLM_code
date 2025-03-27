import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from data.utils import load_milanko, variance_explained

def dX(t,X,tau,P,orbital_funs):
    eps,bet,rho = orbital_funs
    O,I = X
    dO = eps(t) - O
    dI = P[0]*O + P[1]*eps(t) + P[2]*bet(t) + P[3]*np.cos(rho(t)) + P[4] - I
    return dO/tau, dI/tau

def solve_wheen(t_span,tau,P,orbital_funs,X0=None):
    if X0:
        tmin = t_span[0]
    else:
        tmin = t_span[0]-500
        X0 = (0,0)
    I_sol = solve_ivp(dX, (tmin,t_span[-1]), X0, args=(tau,P,orbital_funs),
                      t_eval=t_span, rtol=1e-4, atol=1e-7).y.squeeze()
    return I_sol[1,:]

if __name__ == '__main__':
    ice_vol = np.load('data/bin_isolated_ice_vol_km3')
    ice_vol_fun = interp1d(*ice_vol.T)

    mil_t, ecc, obliq, prec = load_milanko(direction='backward')
    eps = interp1d(mil_t,ecc,'cubic')
    bet = interp1d(mil_t,obliq,'cubic')
    rho = interp1d(mil_t,prec)
    orbital_funs = eps,bet,rho

    tmin = -800
    tmax = 0
    t_n = (tmax-tmin)*10+1
    t_span = np.linspace(tmin, tmax, t_n)

    # These can be used to calculate p4 as a function of p0-p3 and the variables 
    # This is useful for reducing parameter precision (but cant reduce p4 as
    # has to be very precise to align with data) and also for parameter
    # perturbations which may be done later.
    mu_dat = 55251815.01272642
    mu_eps,mu_bet,mu_rho = 0.02707292,0.40738971,-0.00211251
    mu_O = 0.02702285963589531
    '''
    p0 = 1.884e+09
    p1 = -1.942e+09
    p2 = -1.540e+09
    p3 = -1.897e+07
    #p4 = 6.816e+08 # No longer accurate since the other parameters are rounded
    p4 = mu_dat - (p0*mu_O + p1*mu_eps + p2*mu_bet + p3*mu_rho)
    P = p0,p1,p2,p3,p4
    tau = 14.76
    I_0 = solve_wheen(t_span, tau, P, orbital_funs)
    plt.plot(t_span,I_0,label='Hi Precision')
    print('var exp: ',variance_explained(I_0, ice_vol_fun(t_span)))
    '''
    p0 = 1.9e+09
    p1 = -1.9e+09
    p2 = -1.5e+09
    p3 = -1.9e+07
    p4 = mu_dat - (p0*mu_O + p1*mu_eps + p2*mu_bet + p3*mu_rho)
    P = p0,p1,p2,p3,p4
    tau = 15
    I_0 = solve_wheen(t_span, tau, P, orbital_funs)
    print('var exp: ',variance_explained(I_0, ice_vol_fun(t_span)))
    plt.plot(t_span,I_0,label='Low Precision',linewidth=2)

    plt.plot(t_span,ice_vol_fun(t_span),label='Data',linewidth=2)
    plt.legend()
    plt.show()
