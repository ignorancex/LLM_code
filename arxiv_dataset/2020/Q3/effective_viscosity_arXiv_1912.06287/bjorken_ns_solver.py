import numpy as np
from scipy import interpolate
from scipy.integrate import ode
from scipy import integrate

##################################################################
############# Numerical solution to NS Bjorken ###################
##################################################################

# Right-hand-side of Navier-Stokes Bjorken 

# Args are T0, tau0, the function for c_s^2 and 
# the function for the combined viscosity (4/3*eta/s+zeta/s)
def rhs(tau, Tvec, args):
   
    T_in_fm=Tvec[0]

    cs2_fct=args[0]
    viscosity_fct=args[1]

    result=-1*T_in_fm/tau*cs2_fct(T_in_fm)*(1.-viscosity_fct(T_in_fm)/(tau*T_in_fm))

    return result

# Set-up ODE solver

def init_bjorken_ns_solver(T0_in_fm, tau0, cs2_fct, combined_visc_fct_param=None):

    if (combined_visc_fct_param is None):
        combined_visc_fct=lambda T: 0.0
    else:
        combined_visc_fct=combined_visc_fct_param

    sol=ode(rhs).set_integrator('dopri5')
    sol.set_initial_value(T0_in_fm, tau0).set_f_params([cs2_fct, combined_visc_fct])

    return sol


def approx_effective_viscosity_ns(T0_in_fm, Tf_in_fm, cs2_fct, combined_visc_fct_param):

    def weight(T,T0_in_fm):
        return np.power(T/T0_in_fm,1./cs2_fct(np.sqrt(T0_in_fm*T))-2)

    num=integrate.quad(lambda T : weight(T,T0_in_fm)*combined_visc_fct_param(T), Tf_in_fm, T0_in_fm,limit=1000, epsabs=1e-5, epsrel=1e-5)
    denum=integrate.quad(lambda T : weight(T,T0_in_fm), Tf_in_fm, T0_in_fm,limit=1000, epsabs=1e-5, epsrel=1e-5)

    #print(num, denum)

    return num[0]/denum[0]
    
# Exact for constant speed of sound
def better_effective_viscosity_ns(tau0, tauf, ns_solution, cs2_fct, combined_visc_fct_param):

    def num_integrand(taup):
        T_in_fm=ns_solution.integrate(taup)
        return cs2_fct(T_in_fm)/(taup*taup*T_in_fm)*combined_visc_fct_param(T_in_fm)

    def denom_integrand(taup):
        T_in_fm=ns_solution.integrate(taup)
        return cs2_fct(T_in_fm)/(taup*taup*T_in_fm)

    num=integrate.quad(lambda taup: num_integrand(taup), tau0, tauf, limit=1000, epsabs=1e-5, epsrel=1e-5)
    denom=integrate.quad(lambda taup: denom_integrand(taup), tau0, tauf, limit=1000, epsabs=1e-5, epsrel=1e-5)

    #print(num, denum)

    return num[0]/denom[0]


# Usage:
#hbarc=0.1973
#T0_in_fm=0.4/hbarc
#tau0=0.4
#sol=init_bjorken_ns_solver(T0_in_fm,tau0,lambda T: 1/3)
#for tau in np.arange(tau0,2.0,.2):
#    print(tau,sol.integrate(tau),T0_in_fm*np.power(tau0/tau,1./3.))

