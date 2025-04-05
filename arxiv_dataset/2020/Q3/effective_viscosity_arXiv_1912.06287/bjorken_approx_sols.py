import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as mtick

from bjorken_ns_solver import *
from eos import *

hbarc=0.1973

#########################################################################
############# Approximation solution to Bjorken hydro ###################
#########################################################################

def T_ideal_bjorken_approx(T0_in_fm, tau0, tau, csm2_bar, cs2_fct):

    #\bar{c}_s^-2, the average speed of sound around which the QCD ideal solution is constructed
    #csm2_bar=5

    T0=T0_in_fm

    Tbar=T0*np.power(tau0/tau,1./csm2_bar)

    exponent=1.-(1./cs2_fct(np.sqrt(T0*Tbar))-csm2_bar)*cs2_fct(Tbar)

    return T0*np.power(Tbar/T0,exponent)



def T_NS_bjorken_approx(T0_in_fm, tau0, tau, cs2_fct, viscosity_fct):

    T0=T0_in_fm

    Tid=T_ideal_bjorken_approx(T0, tau0, tau, cs2_fct)

    def integrand(Tp):
        return 1/Tp*np.power(Tp/T0,1/cs2_fct(np.sqrt(T0*Tp))-1)*viscosity_fct(Tp)

    integral=integrate.quad(integrand, Tid, T0)

    return Tid*(1+integral[0]/(tau0*T0_in_fm))


def PiHat_approx(tau, tau0, PiHat0, T0_in_fm, T_in_fm, zeta_over_s_fct_param, tau_Pi_fct_param):

    tau_Pi=tau_Pi_fct_param(tau)

    PiHat0_NS=-zeta_over_s_fct_param(T0_in_fm)/T0_in_fm/tau0

    first_term=(PiHat0-PiHat0_NS)*np.exp(-(tau-tau0)/tau_Pi)*np.power(tau/tau0,2./3.)

    PiHat_NS=-zeta_over_s_fct_param(T_in_fm)/T_in_fm/tau


#    def integrand(taup):
#        T_in_fm=T_fct_param(taup)
#        tau_Pi=tau_Pi_fct_param(taup)
#
#        PiHatNS=-zeta_over_s_fct_param(T_in_fm)/(taup*T_in_fm)
#
#        return np.exp(-(tau-taup)/tau_Pi)*PiHatNS/tau_Pi
#
#    integral=integrate.quad(integrand, tau0, tau, epsabs=1e-10, epsrel=1e-3)

    return first_term+PiHat_NS

    #tau_Pi=tau_Pi_fct_param(

    #return PiHat0*np.exp(()



#def T_visc_approx2_qcd(T0_in_fm, tau0, tau, viscosity_fct, ideal_ode):
#
#    T0=T0_in_fm
#
#    Tid=T0_in_fm*np.exp(-1*ideal_ode.integrate(np.log(tau/tau0))[0]) #Tid_approx_qcd(T0, tau0, tau)
#
#    def integrand(Tp):
#        return 1/Tp*np.power(Tp/T0,1/cs2_qcd(np.sqrt(T0*Tp))-1)*viscosity_fct(Tp)
#
#    integral=integrate.quad(integrand, Tid, T0)
#
#    return Tid*(1+integral[0]/(tau0*T0_in_fm))
#
