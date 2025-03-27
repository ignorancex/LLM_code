import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from data.utils import load_milanko, variance_explained
from scipy.optimize import root
from wheen import solve_wheen

def integ(f):
    inte = np.zeros(len(f))
    inte[1:] = np.cumsum(f)[:-1]
    return inte

def optim_fun(X,t,orbital_funs,tau,P,mu_O,R_O,mu_S,R_S):
    dt = t[1]-t[0]
    a,b,c,d,e,f,alpha_I,alpha_O,alpha_S = X
    E = lambda x: np.mean(x)
    R = lambda x: np.ptp(x)
    O = c*np.exp(-t/tau)/tau*integ(eps(t)*np.exp(t/tau)*dt) + alpha_O
    S = d*eps(t) + e*bet(t) + f*np.cos(rho(t)) + alpha_S
    eq1 = a*c-P[0]
    eq2 = -b*d-P[1]
    eq3 = -b*e-P[2]
    eq4 = -b*f-P[3]
    eq5 = a*alpha_O-b*alpha_S+alpha_I-P[4]
    eq6 = E(O) - mu_O
    eq7 = R(O) - R_O
    eq8 = E(S) - mu_S
    eq9 = R(S) - R_S
    return eq1,eq2,eq3,eq4,eq5,eq6,eq7,eq8,eq9

def extract_params(t_span,orbital_funs,tau,P,mu_O,R_O,mu_S,R_S):
    a = 2e7
    b = 4e7
    c = 90
    d = 50
    e = 40
    f = 0.5
    alpha_I = 3e8
    alpha_O = 7
    alpha_S = -6
    X0 = (a,b,c,d,e,f,alpha_I,alpha_O,alpha_S)
    optim = root(optim_fun, X0, (t_span, orbital_funs,
                    tau, P, mu_O, R_O, mu_S, R_S)).x
    return optim

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

    p0 = 1.884e+09
    p1 = -1.942e+09
    p2 = -1.540e+09
    p3 = -1.897e+07
    p4 = 6.816e+08
    P = p0,p1,p2,p3,p4
    tau = 14.76

    mu_O = 9.8
    R_O = 4
    mu_S = 10.1
    R_S = 5.8

    a,b,c,d,e,f,alpha_I,alpha_O,alpha_S = extract_params(t_span,orbital_funs,tau,P,mu_O,R_O,mu_S,R_S)
    [print(val,' = ',eval(val)) for val in ('a','b','c','d','e','f','alpha_I','alpha_O','alpha_S')]
    sol = solve_wheen(t_span,(a,b,c,d,e,f,alpha_I,alpha_O,alpha_S,tau),orbital_funs)

    plt.plot(t_span,sol[:,1])
    plt.plot(t_span,ice_vol_fun(t_span))

    plt.figure()
    plt.plot(t_span,sol[:,0])
    plt.figure()
    plt.plot(t_span,d*eps(t_span) + e*bet(t_span) + f*np.cos(rho(t_span)) + alpha_S)

    print('var exp: ', variance_explained(sol[:,1], ice_vol_fun(t_span)))

    plt.show()
