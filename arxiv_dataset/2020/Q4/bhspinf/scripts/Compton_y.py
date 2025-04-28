import numpy as np
import constants as c
from bhtools import radiative_efficiency

# Parameters
alpha = 1
mdot  = 0.1
x     = 20.0
m     = 10.0
print("")
print("  alpha = ", alpha)
print("   mdot = ", mdot)
print("      x = ", x)
print("      m = ", m)

# Include the radiative efficiency in the definition of the Eddington-scaled mass accretion rate
a     = 0  # non-spinning black hole
eta   = radiative_efficiency(a)
mdot /= eta

# Additional parameters / constants
M     = m * c.sun2g
mu    = 0.6 * c.mp
kappa = c.sigmaT / c.mp
J     = 1.0#1.0 - np.sqrt(6.0 / x)
xi    = 1.0
f     = 0.0


kappa_es = 0.4

def calc_kappa_bf(T, rho):
    kappa_bf = 1.6e24 * rho * T**(-3.5)
    return kappa_bf

def calc_tau_eff(T, rho, tau_es):
    kappa_th = calc_kappa_bf(T=T, rho=rho)
    tau_eff  = tau_es * np.sqrt(kappa_th / kappa_es)
    return tau_eff

def calc_fcol_min(T, rho, tau_es):
    kappa_th = calc_kappa_bf(T=T, rho=rho)
    fcol_min = tau_es**(-7.0/36.0) * (kappa_th / kappa_es)**(-2.0/9.0)
    return fcol_min

def calc_y_min(T, rho, tau_es):
    kappa_th = calc_kappa_bf(T=T, rho=rho)
    y_min    = 4.0 * c.kB * T / (c.me * c.c**(2.0)) \
             * tau_es**(-2.0) * (kappa_th / kappa_es)**(-2.0)
    return y_min


# Begelman & Pringle (2007)
T_BP   = (3.0 * c.c**(5.0) / (c.a * kappa * c.G) * np.sqrt(mu/c.kB))**(2.0/9.0) \
       * alpha**(-2.0/9.0) * mdot**(4.0/9.0) * x**(-8.0/9.0) * M**(-2.0/9.0)
rho_BP = (2.0/3.0) * c.c**(3.5) / (kappa * c.G) * (mu/c.kB)**(0.75) \
       * (3.0 * c.c**(5.0) / (c.a * kappa * c.G) * np.sqrt(mu/c.kB))**(-1.0/6.0) \
       * alpha**(-5.0/6.0) * mdot**(2.0/3.0) * x**(-19.0/12.0) * M**(-5.0/6.0)
tau_BP = (4.0/3.0) * c.c * np.sqrt(mu/c.kB) \
       * (3.0 * c.c**(5.0) / (c.a * kappa * c.G) * np.sqrt(mu/c.kB))**(-1.0/9.0) \
       * alpha**(-8.0/9.0) * mdot**(7.0/9.0) * x**(-5.0/9.0) * M**(1.0/9.0)
m0_BP  = 0.5 * tau_BP / kappa
tau_eff_BP  = calc_tau_eff(T=T_BP, rho=rho_BP, tau_es=tau_BP)
fcol_min_BP = calc_fcol_min(T=T_BP, rho=rho_BP, tau_es=tau_BP)
y_min_BP    = calc_y_min(T=T_BP, rho=rho_BP, tau_es=tau_BP)
print("")
print("Begelman & Pringle (2007):")
print("      T = ", T_BP)
print("    rho = ", rho_BP)
print(" tau_es = ", tau_BP)
print("tau_eff = ", tau_eff_BP)
print("     m0 = ", m0_BP)
print("  f_col > ", fcol_min_BP)
print("      y > ", y_min_BP)

# Shakura & Sunyaev (1973)
T_SS   = c.me * c.c**(2.0) / c.kB * (45.0 / (2.0 * np.sqrt(2.0) * np.pi**3))**(0.25) \
       * c.fsc**(-0.75) * (c.mp/c.me)**0.25 \
       * (2.0 * c.G * c.sun2g / (c.re * c.c**2))**(-0.25) * 2.0**(3.0/8.0) \
       * alpha**(-0.25) * x**(-0.375) * m**(-0.25) \
       * (xi * (1.0 - f))**(-0.25)
rho_SS = c.mp / c.sigmaT * (2.0 * c.G * c.sun2g / c.c**2)**(-1.0) * (128.0/27.0) \
       * alpha**(-1.0) * x**(1.5) * m**(-1.0) * (mdot * J)**(-2.0) \
       * (xi * (1.0 - f))**(-3)
tau_SS = (16.0/9.0) * alpha**(-1.0) * x**(1.5) * (mdot * J)**(-1.0) \
       * (xi * (1.0 - f))**(-2.0)
m0_SS  = 0.5 * tau_SS / kappa
tau_eff_SS  = calc_tau_eff(T=T_SS, rho=rho_SS, tau_es=tau_SS)
fcol_min_SS = calc_fcol_min(T=T_SS, rho=rho_SS, tau_es=tau_SS)
y_min_SS    = calc_y_min(T=T_SS, rho=rho_SS, tau_es=tau_SS)
print("")
print("Shakura & Sunyaev (1973):")
print("      T = ", T_SS)
print("    rho = ", rho_SS)
print("tau_es = ", tau_SS)
print("tau_eff = ", tau_eff_SS)
print("     m0 = ", m0_SS)
print("  f_col > ", fcol_min_SS)
print("      y > ", y_min_SS)


print("")
print("     T = ", T_BP / T_SS)
print("   rho = ", rho_BP / rho_SS)
print("tau_es = ", tau_BP / tau_SS)
print("    m0 = ", m0_BP / m0_SS)
print(" f_col > ", fcol_min_BP / fcol_min_SS)
print("     y > ", y_min_BP / y_min_SS)
print("")

