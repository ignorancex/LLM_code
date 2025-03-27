import numpy as np
import sys
sys.path.append('../')
import os
from numba import njit

#K is calculated as 4pi*Q0*a**2
K = 3.8284e+26 # Total sun power
au = 149597870700 # metres
year_length = 365.256363

wd = os.path.dirname(os.path.realpath(__file__))

@njit
def rk4(F, t_ind, y, h, args, dy):
    'F is the input (q65 or orbital) and t_ind is the index for time t'
    k1 =  h*dy(F[t_ind], y, args)
    k2 =  h*dy(F[t_ind+1], y + k1/2, args)
    k3 =  h*dy(F[t_ind+1], y + k2/2, args)
    k4 =  h*dy(F[t_ind+2], y + k3, args)
    return (k1 + 2*k2 + 2*k3 + k4)/6


def linear_fit(fit,data):
    m,c = np.linalg.lstsq(np.c_[fit,np.ones_like(fit)],data,rcond=None)[0]
    return m*fit+c

def variance_explained(fit,data,best_fit=False):
    if fit.ndim>1:
        data = data.reshape(-1,1)
    if best_fit:
        fit = linear_fit(fit,data)
    tot_vari = np.var(data,0)
    resid_vari = np.var(data-fit,0)
    expl_vari = tot_vari - resid_vari
    return expl_vari/tot_vari

def norm(x):
    y = x - np.mean(x,0)
    return y/np.std(y,0)

def get_Q_year(year, span, year_res):
    year_span = np.linspace(0,year_length,year_res+1)[:-1]
    if max(span)==1:
        span = np.arcsin(span)
    elif max(span)!=np.pi/2:
        raise ValueError('Latitude domain not y or phi')

    steps = len(span)
    f_name = wd+f'/.Q_year_cache/{int(np.ptp(span))}_{year_res}_{steps}_{int(year)}'
    try:
        Q_year = np.fromfile(f_name).reshape(year_res,steps)

    except (FileNotFoundError, ValueError):
        eps, beta, rho = milanko_update(year)
        _,theta = polar_pos(year_span, eps)
        Q_year = calc_Q(rho, beta, theta, span, eps)
        Q_year.tofile(f_name)
    return Q_year

def milanko_update(t,old_laskar=False):
    # t (years)
    t/=1000
    if np.any(abs(t)>3000):
        long=True
        interp=False
    else:
        long=False
        interp=True
    t_, ecc, obliq, rho = load_milanko(long=long,interp=interp,old_laskar=old_laskar)
    return np.interp(t,t_,ecc), np.interp(t,t_,obliq), np.interp(t,t_,rho)%(2*np.pi)

def load_milanko(long=False,interp=True,direction=None,old_laskar=False):
    if old_laskar: return np.load(wd+'/bin_laskar_90')
    if long and interp: raise ValueError('No hi-res long milanko file')
    data_file = wd+'/bin_milanko'+long*'_long'+interp*'_cub_interp'
    t, ecc, obliq, rho = np.load(data_file)
    if direction is None: return t, ecc, obliq, rho
    res = 10 if interp else 1
    present = 51000 if long else 3000*res 
    if direction == 'backward':
        return t[:present+1], ecc[:present+1], obliq[:present+1], rho[:present+1]
    if direction == 'forward':
        return t[present:], ecc[present:], obliq[present:], rho[present:]
    raise ValueError('Invalid direction specified')

def calc_E(M, eps):
    # Two iterations of Newton works well
    E = M + eps*np.sin(M)/(1-eps*np.cos(M))
    E -= (M+eps*np.sin(E)-E)/(eps*np.cos(E)-1)
    return E

def calc_theta(E, eps):
    sign = np.ones(np.shape(E))
    sign[(E>np.pi)] = -1
    return np.pi+2*sign*np.arctan(np.sqrt((1+eps)/(1-eps)*(np.tan(E/2))**2))

def polar_pos(t, eps, grid=True):
    """Using Kepler's law: https://bit.ly/3jI5orE
       Allows for vector of either t or eps or both
       grid=True gives outer product of vectors 
       whilst False aligns them into one vector"""
    t = np.asarray(t)
    eps = np.asarray(eps)
    if grid and eps.size>1: eps = eps.reshape(-1,1)
    t = (t+year_length/2)%year_length # Equations assume theta=0 at perihelion
    M = t*2*np.pi/year_length
    E = calc_E(M, eps)
    r = au*(1 - eps*np.cos(E))
    theta = calc_theta(E,eps)
    return r, theta

def calc_Q(rho, beta, theta, phi, eps):
    """Shift dimensions to allow for iteration over orbital parameters,
       time of year (theta), and latitude. Here theta (as output by polar_pos
       in utils) is of shape (kyrs,year_res).

       returns Q of shape (kyrs,lat_span(s-n),year_res)"""

    theta = theta[:,np.newaxis,:]
    rho = rho[:,np.newaxis,np.newaxis]
    beta = beta[:,np.newaxis,np.newaxis]
    eps = eps[:,np.newaxis,np.newaxis]
    phi = phi[np.newaxis,:,np.newaxis]

    x0 = rho - theta
    x1 = np.sin(x0)
    x2 = np.sin(beta)
    x3 = np.cos(x0)
    x4 = x3**2
    x5 = x2**2*x4
    x6 = np.sin(phi)
    x7 = x2*x3*x6
    x8 = -x1*x7/(x5 - 1)
    x9 = x1**2
    x10 = np.cos(beta)
    x11 = x10**2*x4
    x12 = 2*beta
    x13 = 2*rho
    x14 = 2*theta
    x15 = x13 - x14
    x16 = x10*x3
    x17 = x16*np.sqrt(abs(2*np.cos(2*phi) + np.cos(x12) - np.cos(x15) + np.cos(x12 + x15)/2 +
        np.cos(x12 - x13 + x14)/2 + 1))/(2*(x11 + x9))
    x18 = x17 + x8
    x19 = -x6*np.tan(beta)
    x20 = np.tan(x0)/x10
    x21 = np.arctan2(x18, x18*x20 + x19)
    x22 = np.cos(phi)
    x23 = x22**2
    x24 = x11*x23 + x23*x9 - x5*x6**2 > 0
    x25 = np.select([x24,True], [x21,0], default=np.nan)
    x26 = 2*np.pi
    x27 = -x17 + x8
    x28 = np.arctan2(x27, x19 + x20*x27)
    x29 = np.select([x24,np.greater(x3*np.cos(beta - phi), 0),True], [np.select([np.less(x26,
        -x21 + x28),np.greater(x21, x28),True], [-x26 + x28,x26 + x28,x28],
        default=np.nan),0,x26], default=np.nan)
    return (K*(-eps*np.cos(theta) + 1)**2*(-x22*(-x1*np.cos(x25) - x16*np.sin(x25)) +
            x22*(-x1*np.cos(x29) - x16*np.sin(x29)) + x25*x7 -
            x29*x7)/(8*np.pi**2*au**2*(1 - eps**2)**2)).squeeze().T

def calc_Q_point(t, phi, rho, beta, eps):
    'returns Q at single point in time and latitude'
    _,theta = polar_pos(t,eps,grid=False)
    x0 = rho - theta
    x1 = np.sin(x0)
    x2 = np.sin(beta)
    x3 = np.cos(x0)
    x4 = x3**2
    x5 = x2**2*x4
    x6 = np.sin(phi)
    x7 = x2*x3*x6
    x8 = -x1*x7/(x5 - 1)
    x9 = x1**2
    x10 = np.cos(beta)
    x11 = x10**2*x4
    x12 = 2*beta
    x13 = 2*rho
    x14 = 2*theta
    x15 = x13 - x14
    x16 = x10*x3
    x17 = x16*np.sqrt(abs(2*np.cos(2*phi) + np.cos(x12) - np.cos(x15) + np.cos(x12 + x15)/2 +
        np.cos(x12 - x13 + x14)/2 + 1))/(2*(x11 + x9))
    x18 = x17 + x8
    x19 = -x6*np.tan(beta)
    x20 = np.tan(x0)/x10
    x21 = np.arctan2(x18, x18*x20 + x19)
    x22 = np.cos(phi)
    x23 = x22**2
    x24 = x11*x23 + x23*x9 - x5*x6**2 > 0
    x25 = np.select([x24,True], [x21,0], default=np.nan)
    x26 = 2*np.pi
    x27 = -x17 + x8
    x28 = np.arctan2(x27, x19 + x20*x27)
    x29 = np.select([x24,np.greater(x3*np.cos(beta - phi), 0),True], [np.select([np.less(x26,
        -x21 + x28),np.greater(x21, x28),True], [-x26 + x28,x26 + x28,x28],
        default=np.nan),0,x26], default=np.nan)
    return (K*(-eps*np.cos(theta) + 1)**2*(-x22*(-x1*np.cos(x25) - x16*np.sin(x25)) +
            x22*(-x1*np.cos(x29) - x16*np.sin(x29)) + x25*x7 -
            x29*x7)/(8*np.pi**2*au**2*(1 - eps**2)**2))

