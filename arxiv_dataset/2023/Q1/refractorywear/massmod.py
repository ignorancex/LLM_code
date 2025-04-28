import numpy as np
import math


def mass_step(mass, mfr_in, mfr_out, dt):
    retval = mass + (mfr_in-mfr_out)*dt
    return retval


# The feed and tap will not fill or empty the ladle at this point.
def feed_steel(time):
    if (time <= 110):
        retval = 0000.0
    else:
        retval = 0.0
    return retval


def tap_steel(time):
    if (time >= 3700 and time <= 7300):
        retval = 00.0
    else:
        retval = 0.0
    return retval


def feed_slag(time):
    if (time >= 110 and time < 2000):
        retval = 00.0
    else:
        retval = 0.0
    return retval


def tap_slag(time):
    if (time >= 7300 and time <= 7400):
        retval = 00.0
    else:
        retval = 0.0
    return retval


def diff_C_in_steel(Xc):
    return 1.1*(1.0 + Xc/0.053)*1.0e-8

def k_Cb(Xc, u_tau):
    v_steel = 1.0e-6  # is v_steel (stj), nærmere 1.0e-7 ?
    Sc = v_steel/diff_C_in_steel(Xc)
    return 0.09*u_tau*Sc**(-0.7)

def k_Mgob(u_tau,diff_Mgo_in_slag):  # ny (stj)
    v_slag = 1.0e-4  / 5.0
    Sc = v_slag/diff_Mgo_in_slag
    return 0.09*u_tau*Sc**(-0.7)


def k_cpore(Xc,s_pore):
    return diff_C_in_steel(Xc)*s_pore
#   s_pore is diffusion lengt of carbon in between grains of MgO 
#   longer s_pore will give lower erosion


def k_eff_C(alpha_carbon,Xc, s_pore, u_tau):
    return k_Cb(Xc, u_tau)*diff_C_in_steel(Xc)*alpha_carbon/(s_pore*k_Cb(Xc, u_tau) + diff_C_in_steel(Xc))


def k_eff_mgo(qg, diff_Mgo_in_slag, delta_slag,utau):
    tune = 1.0
    v_slag = 1.0e-4/5.0
    Uwave = 0.0121*0.999925**qg*qg**0.39233
    awave = 0.2
    default = diff_Mgo_in_slag/0.0005 # 0.001
    default = 0.0
    Sc = v_slag/diff_Mgo_in_slag
    # The next line does not make sense, as awave goes to zero, the mass transfer coeff goes to inf
  # if qg > 0:
  #     print("Uwave,Sc,delta_slag",Uwave,Sc,delta_slag)
    retval = default + tune*0.678*(v_slag/(2.0*awave+delta_slag))*Sc**(-2.0/3.0)*np.sqrt(Uwave*awave/v_slag)
    retval = retval + k_Mgob(utau,diff_Mgo_in_slag)
  # retval = k_Mgob(utau,diff_Mgo_in_slag)

    return retval

def MgOFeO(wtFe2O3, T):
    return (8.814-0.5481*wtFe2O3)*math.exp((T - 2273.15)/(258.2))

def MgOSiO2(wtSiO2, T_):
    T = T_ - 273.15 # Need T in C
    return (5.5 + 0.15357*wtSiO2)*((-434000.0 + 514.3*T)/(1.0 + 100.74*T - 0.041*T**2))/(5.5 + 0.15357*10.2)

def MgOCaO(wtCaO, T_):
    T = min(2400.0, T_-273.15)
    return max(min(0.3*(50 - wtCaO) + (-4.34e5 + 514.3*T)/(1.0 + 100.74*T - 0.041*T**2),\
            wtCaO*(9.025 - 4.427e-3*T - 7.7e6/T**2) + (-598.7 + 0.2927*T + 5.015e8/T**2)), 0.0)

def MgOsol(T_):
    T = min(2400.0, max(T_, 1200.0))
    retval = MgOCaO(50.0, T)*MgOFeO(2.5, 1700.0+273.15)/(MgOFeO(2.5, 1700.0+273.15))*MgOSiO2(10.0,1700.0+273.15)/(MgOSiO2(10.0,1700.0+273.15))/100.0
    retval = min(0.4, retval)
    return retval
