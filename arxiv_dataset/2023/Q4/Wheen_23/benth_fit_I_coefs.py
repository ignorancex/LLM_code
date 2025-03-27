import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from data.utils import load_milanko, variance_explained
from scipy.linalg import solve_triangular
from numba import njit

mil_t, ecc, obliq, prec = load_milanko(long=True,interp=False,direction='backward')
eps = interp1d(mil_t,ecc,'cubic')
bet = interp1d(mil_t,obliq,'cubic')
rho = interp1d(mil_t,prec)

@njit
def zeta(x, etau, dt):
    integral = np.zeros_like(x)
    for i in range(1, len(x)):
        integral[i] = integral[i-1] + x[i]*etau[i]*dt
    return integral/etau

def I_args(t,dt,tau):
    long_t = extend_time(t)
    etau = np.exp((long_t-long_t[0])/tau)

    zeta_eps = zeta(eps(long_t), etau, dt)

    return np.array((1/tau*zeta(zeta_eps,etau, dt),
                     #1/tau*zeta(zeta(bet(long_t), etau, dt),etau,dt),
                     zeta_eps, zeta(bet(long_t), etau, dt),
                     zeta(np.cos(rho(long_t)), etau, dt),
                     tau*np.ones_like(long_t)))[:,-len(t):]/tau

def lstsqr(A,b):
    q, r = np.linalg.qr(A)
    y = np.dot(q.T, b)
    return solve_triangular(r, y)

def fit_lstsqrs(t,tau,ice_fun):
    A = I_args(t,t[1]-t[0],tau)
    b = ice_fun(t)
    P = lstsqr(A.T,b)
    I_sol = A.T.dot(P)
    return I_sol,P

def sweep_tau(range_n, tau_range, t_span, ice_fun, stats=True, plots=True):
    tau_range = np.linspace(*tau_range,range_n)
    var_explained_vals = np.empty_like(tau_range)
    for i,tau in enumerate(tau_range):
        I = fit_lstsqrs(t_span,tau,ice_fun)[0]
        var_explained_vals[i] = variance_explained(I,ice_fun(t_span))

    if stats:print_stats(t_span, tau_range, ice_fun, var_explained_vals)
    if plots:show_plots(t_span, tau_range, ice_fun, var_explained_vals)

    return var_explained_vals

def optim_sol(t_span, ice_fun, range_n=30, tau_range=(10,20), stats=False, plots=False):
    var_explained_vals = sweep_tau(range_n,tau_range,t_span,ice_fun,stats=stats,plots=plots)
    i = np.unravel_index(np.argmax(var_explained_vals), var_explained_vals.shape)
    tau_range = np.linspace(*tau_range,range_n)
    tau = tau_range[i]
    I = fit_lstsqrs(t_span,tau,ice_fun)[0]
    return I, tau

def print_stats(t_span, tau_range, ice_fun, var_explained_vals):
    i = np.unravel_index(np.argmax(var_explained_vals), var_explained_vals.shape)
    tau = tau_range[i]
    P = fit_lstsqrs(t_span,tau,ice_fun)[1]
    print('Coupled Params:')
    [print(f'p{i}: {val:.3e}') for i,val in enumerate(P)]
    print('tau: ',tau)
    print('Var explained: ',var_explained_vals[i])

def show_plots(t_span, tau_range, ice_fun, var_explained_vals):
    i = np.unravel_index(np.argmax(var_explained_vals), var_explained_vals.shape)
    tau = tau_range[i]
    I = fit_lstsqrs(t_span,tau,ice_fun)[0]
    plt.figure()
    plt.plot(t_span,ice_fun(t_span),'C1',label='Data')
    plt.plot(t_span,I,'C0',label='Fit')
    plt.xlabel('Time (kyr)')
    plt.ylabel('Ice Vol (Gt)')
    plt.legend()
    plt.figure()
    plt.plot(tau_range,var_explained_vals)
    plt.gca().set_xlabel(r'$\tau$')
    plt.show()

def extend_time(t, extend=400):
    dt = t[1]-t[0]
    return np.r_[np.arange(t[0]-extend//dt*dt,t[0],dt),t]

if __name__ == '__main__':
    ice_vol = np.load('data/bin_isolated_ice_vol_km3')
    ice_vol_fun = interp1d(*ice_vol[:,:2].T)

    tmin = -800
    tmax = -0
    t_n = (tmax-tmin)*10+1
    t_span = np.linspace(tmin, tmax, t_n)

    tau_range = (12,20)
    grid_n = 51
    from scipy.optimize import curve_fit
    def calc_conf(params):
    # Calculate the 95% confidence interval for params
        sol = fun(t_span,*params)
        resid = ice_vol_fun(t_span)-sol
        jacobian = np.zeros((len(t_span),len(params)))
        for i in range(len(params)):
            pert = np.zeros(len(params))
            pert[i] = 0.1
            pert_sol = fun(t_span,*(params+pert))
            jacobian[:,i] = (pert_sol - sol)/0.1

        resid_var = np.var(resid)
        co_var = np.linalg.inv(jacobian.T@jacobian)*resid_var
        return 1.96*np.sqrt(np.diag(co_var))

    def fun(t,p1,p2,p3,p4,p5,tau):
        args = I_args(t,t[1]-t[0],tau)
        return args.T.dot(np.array((p1,p2,p3,p4,p5)))
    popt, pcov = curve_fit(fun, t_span, ice_vol_fun(t_span),
                           p0=(2e9,-2e9,-2e9,-2e7,2e8,15))
    print(popt)
    print(1.96*np.sqrt(np.diag(pcov)))
    print(calc_conf(popt))

    I,tau = optim_sol(t_span,ice_vol_fun,grid_n,tau_range,True,True)

