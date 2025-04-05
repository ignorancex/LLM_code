import numpy as np
from scipy import interpolate
from scipy.integrate import ode
from scipy import integrate

#from bjorken_ns_solver import approx_effective_viscosity_ns

##################################################################
############# Numerical solution to IS Bjorken ###################
##################################################################

# Right-hand-side of Israel-Stewart Bjorken 

# Args are [T(tau_0), pi(tau_0)/(sT), Pi(tau_0)/(sT)], tau0, the function for c_s^2(T_in_fm),
# the function of eta/s(T_in_fm) and zeta/s(T_in_fm)
def rhs(tau, varvec, args):

    T_in_fm=varvec[0]
    piHat=varvec[1]
    PiHat=varvec[2]

    cs2_fct=args[0]
    eta_over_s_fct=args[1]
    zeta_over_s_fct=args[2]
    tau_pi_fct=args[3]
    tau_Pi_fct=args[4]

    # Helper variables
    piHat_NS=4./3.*eta_over_s_fct(T_in_fm)/(T_in_fm*tau)
    PiHat_NS=-1*zeta_over_s_fct(T_in_fm)/(T_in_fm*tau)
    #print(tau,PiHat_NS)

    # Relaxation times
    tau_pi=tau_pi_fct(T_in_fm)
    #tau_Pi=C_Pi*zeta_over_s_fct(T_in_fm)/T_in_fm
    tau_Pi=tau_Pi_fct(T_in_fm)

    # Can't allow the relaxation time to be zero...
    #small_num=1e-20
    small_num=3*0.005
    tau_pi=np.max([tau_pi,small_num])
    tau_Pi=np.max([tau_Pi,small_num])

    Teq_rhs=-1*T_in_fm/tau*cs2_fct(T_in_fm)*(1.-piHat+PiHat)
    pi_rhs=-(piHat-piHat_NS)/tau_pi-0.5*piHat*piHat
    #Pi_rhs=-(PiHat+PiHat_NS)/tau_Pi-PiHat*PiHat
    delta_PiPi=2./3.
    Pi_rhs=-(PiHat-PiHat_NS)/tau_Pi-PiHat/tau*(delta_PiPi-cs2_fct(T_in_fm)-1)+PiHat*PiHat*(cs2_fct(T_in_fm)+1)/tau

    return [Teq_rhs, pi_rhs, Pi_rhs]

# Set-up ODE solver
def init_bjorken_is_solver(initial_values, tau0, cs2_fct, eta_over_s_fct_param=None, zeta_over_s_fct_param=None, tau_pi_fct_param=None, tau_Pi_fct_param=None):

    T0_in_fm, piHat0, PiHat0 = initial_values

    # Deal with the ideal case
    if (eta_over_s_fct_param is None):
        eta_over_s_fct=lambda T: 0.0
        tau_pi_fct=lambda T: 0.0
    else:
        assert (tau_pi_fct_param is not None), 'Must pass a function for the shear relaxation time'
        eta_over_s_fct=eta_over_s_fct_param
        tau_pi_fct=tau_pi_fct_param

    if (zeta_over_s_fct_param is None):
        zeta_over_s_fct=lambda T: 0.0
        tau_Pi_fct=lambda T: 0.0
    else:
        assert (tau_Pi_fct_param is not None), 'Must pass a function for the bulk relaxation time'
        zeta_over_s_fct=zeta_over_s_fct_param
        tau_Pi_fct=tau_Pi_fct_param

    # 
    sol=ode(rhs).set_integrator('dopri5', nsteps=1000)
    sol.set_initial_value([T0_in_fm, piHat0, PiHat0], tau0).set_f_params([cs2_fct, eta_over_s_fct, zeta_over_s_fct,tau_pi_fct,tau_Pi_fct])

    return sol


#
def Y_factor(tau0, T0_in_fm, Tf_in_fm, cs2_fct, tau_Pi_fct):

    cs2bar=1./4.

    def num_integrand(Tp):
        tau_pi=tau_Pi_fct(Tp)
        csm2=1./cs2_fct(np.sqrt(T0_in_fm*Tp))
        return -1./Tp*np.exp(-tau0/tau_pi*(np.power(T0_in_fm/Tp,csm2)-1))*np.power(T0_in_fm/Tp,(1/3.+cs2bar)*csm2) 
        #return -1./Tp*np.exp(-tau0/tau_pi*(np.power(T0_in_fm/Tp,csm2)-1))*np.power(T0_in_fm/Tp,(csm2/3.+1)) 

    def denom_integrand(Tp):
        return np.power(Tp/T0_in_fm,1./cs2_fct(np.sqrt(T0_in_fm*Tp))-1)/(Tp*T0_in_fm*tau0)

    num=integrate.quad(num_integrand, Tf_in_fm, T0_in_fm,limit=1000, epsabs=1e-5, epsrel=1e-5)
    denom=integrate.quad(denom_integrand, Tf_in_fm, T0_in_fm,limit=1000, epsabs=1e-5, epsrel=1e-5)

    #print(num, denum)

    return num[0]/denom[0]


## Usage:
#hbarc=0.1973
#T0_in_fm=0.4/hbarc
#initial_values=[T0_in_fm,0.0,0.0]
#tau0=0.4
#sol=init_bjorken_is_solver(initial_values,tau0,lambda T: 1/3)
#for tau in np.arange(tau0,15.0,.2):
#    print(tau,sol.integrate(tau),T0_in_fm*np.power(tau0/tau,1./3.))

