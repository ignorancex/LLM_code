import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from data.utils import load_milanko, variance_explained

def dX(t,X,params,orbital_funs):
    eps,bet,rho = orbital_funs
    a,b,c,d,e,f,alpha_I,alpha_O,alpha_S,tau = params
    O,I = X
    S = d*eps(t) + e*bet(t) + f*np.cos(rho(t)) + alpha_S
    dO = c*eps(t) - O + alpha_O
    dI = a*O - b*S - I + alpha_I
    return dO/tau, dI/tau

def solve_wheen(t_span, params, orbital_funs, X0=None):
    if X0:
        tmin = t_span[0]
    else:
        tmin = t_span[0]-500
        X0 = (0,0)
    sol = solve_ivp(dX, (tmin,t_span[-1]), X0, args=(params, orbital_funs),
                    t_eval=t_span, rtol=1e-4, atol=1e-7).y.T
    return sol

if __name__ == '__main__':
    ice_vol = np.load('data/bin_isolated_ice_vol_km3')
    ice_vol_fun = interp1d(*ice_vol.T)

    mil_t, ecc, obliq, prec = load_milanko(direction='backward')
    eps = interp1d(mil_t,ecc,'cubic')
    bet = interp1d(mil_t,obliq,'cubic')
    rho = interp1d(mil_t,prec)
    orbital_funs = eps,bet,rho

    tmin = -800
    tmax = -0
    t_n = (tmax-tmin)*10+1
    t_span = np.linspace(tmin, tmax, t_n)

    # This only performs better when we can specify the X0 as we are able to
    # hit the slope exactly right. This is sort of cheating so should really be
    # run with no X0 and with the pre_run which is handled in solve_wheen.
    #X0,params = (5.969,7.999e7),(2.099e7,3.892e7,87.34,50.03,37.89,0.4686,
    #                             2.609e8,6.457,-6.700,14.29)

    # These are the exact fit parameter values from extract_from_reduced_I,
    # they have alpha_I omitted since we calculate it equivalently from the
    # other parameters and p4. This is not needed here but useful for when
    # reducing the precision of all by the alpha_I parameters.
    a,b,c,d,e,f,alpha_O,alpha_S,tau = (22234418.8909247, 27671371.9710858,
                                       84.73349401520272, 70.18083534090209,
                                       55.65318559629106, 0.6855460589683019,
                                       7.5422979381739115, -14.471086592929344, 14.76)
    # Calculating alpha_I from other two so that precision can be reduced for
    # the others and this will absorb the necessary change.
    p4 = 6.816e+08
    alpha_I = -a*alpha_O + b*alpha_S + p4
    print(f'{alpha_I=}')
    params = a,b,c,d,e,f,alpha_I,alpha_O,alpha_S,tau

    sol = solve_wheen(t_span, params, orbital_funs)[:,1]
    print(variance_explained(sol,ice_vol_fun(t_span)))
    plt.plot(t_span,sol,label='Hi Precision')
    # These are the rounded versions of above but need to calculate alpha_I
    # separately to account for the small change in offset that occurs from
    # rounding. This is done through first calculating what the effective p4 is
    # for these rounded parameters, then calculating alpha_I from there.
    a,b,c,d,e,f,alpha_O,alpha_S,tau = (2.2e7,2.8e7,8.5e1,7.0e1,5.6e1,6.9e-01,7.5e0,-1.4e1,1.5e1)

    mu_dat = 55300000
    mu_eps,mu_bet,mu_rho = 0.0271,0.407,-0.00211
    mu_O = 0.0270
    p4 = mu_dat - (a*c*mu_O - b*d*mu_eps - b*e*mu_bet - b*f*mu_rho)
    alpha_I = round(-a*alpha_O + b*alpha_S + p4,-7)
    print(f'{alpha_I=}')

    params = a,b,c,d,e,f,alpha_I,alpha_O,alpha_S,tau

    sol = solve_wheen(t_span, params, orbital_funs)[:,1]
    print(variance_explained(sol,ice_vol_fun(t_span)))
    plt.plot(t_span,sol,label='Low Precision')
    plt.plot(t_span,ice_vol_fun(t_span),label='Data')
    plt.legend()
    plt.show()
