# Initializing parameters (like cosmos.h)

import numpy as np
from scipy.special import zeta
from scipy import integrate


    
# Physical constants, use these to convert between natural units
hbar = 6.58e-25                                     # [GeV*s]
c = 2.9979e8                                        # [m/s]
kb = 8.617e-14                                      # [GeV/K]

# More constants in natural units
G = 6.795e-39                                       # [GeV^-2]
H0_km_s_Mpc = 67.76                                 # [km/s/Mpc]
H0 = H0_km_s_Mpc*c*hbar*1.081e-28                   # [GeV]
m_e = 511e-6                                        # [GeV] electron mass
m_mu = 0.106                                        # [GeV] muon mass
m_p = 0.938                                         # [GeV] proton mass
m_n = 0.939                                         # [GeV] neutron mass
Omega_r = 5.38e-5                                   # [0] Radiation density
Omega_m = 0.315                                     # [0] (Total) matter density
Omega_dm = 0.265                                    # [0] DM density
Omega_b = 0.0493                                    # [0] Baryon density
Omega_lambda = 0.685                                # [0] Cosmological const
T0 = 2.728*kb                                       # [GeV] Temperature today
Msun = 1.125e57                                     # [GeV] Solar mass
Mpc = 1.56332e38                                    # [GeV^-1] Megaparsec
Mpl = G**(-1/2)                                     # [GeV] Planck mass
Mpl_r = (8*np.pi*G)**(-1/2)                         # [GeV] reduced Planck mass
eta = 6e-10                                         # [0] Baryon to photon ratio
n_b = (2.515e-1)*(c*hbar)**3                        # [GeV^3] Baryon density today
z_eq = 3402.0                                       # [0] redshift of matter radiation eq
Y_p = 0.245                                         # [0] Primordial helium fraction
f_HE = Y_p/(4*(1-Y_p))                              # [0] Primordial helium fraction by # of nuclei
sigma_T = 0.665246e-28*(c*hbar)**(-2)               # [GeV^-2] Thomson cross section
alpha_em = 1.0/137                                  # [0] EM FST (no running, valid til GeV scales maybe)

# Other useful things
g_p = 2                                             # [0] Internal (spin) D.o.f. for a proton
g_e = 2                                             # [0] D.o.f. for an electron
g_H = 4                                             # [0] D.o.f. for Hydrogen

# Recombination and Recfast++ functions
Rec_F = 1.14                                        # [0] Recfast++ fudge factor
Rec_a1, Rec_a2 = 4.309,  -0.6166                    # [0] Fitting function params
Rec_a3, Rec_a4 = 0.6703,  0.5300                    # [0] Fitting function params
Lambda_2s = 8.22458*hbar                            # [GeV] Hydrogen 2S decay rate 
E_ion2s = 3.39856e-9                                # [GeV] Ionization energy of hydrogen (2s state)
E_ion = 13.6e-9                                     # [GeV] Ionization energy of hydrogen (1s state)
E_Lya = 10.28e-9                                    # [GeV] Lyman-alpha energy
L_Lya = 2*np.pi/E_Lya                               # [GeV^-1] Lyman-alpha wavelength


# Functions and redshift dependent quantities

# [GeV] Temperature scaling 
def T_CMB(z):
    return T0*(1+z)

# [GeV] Hubble scaling 
def H_z(z):
    return H0*(Omega_lambda + Omega_m*(1+z)**3 + Omega_r*(1+z)**4)**(1/2)

# [GeV^4] Critical density 
def rho_c(z):
    return (3/(8*np.pi*G))*(H_z(z)**2)

# [GeV^3] Number density of photons 
def N_CMB(z):
    return (2*zeta(3))/(np.pi**2)*(T0**3)*(1+z)**3

# [GeV^4] Energy density of photons 
def rho_CMB(z):
    return (np.pi**2)/(15)*(T0**4)*(1+z)**4

# [GeV^3] Number density of baryons
def N_b(z):
    return n_b*(1+z)**3

# [0] Occupation number of a blackbody in x = h\nu/kT = E/T (natural units for last equality)
def n_BB(x):
    return 1/(np.exp(x)-1)

# [0] y-spectrum
def n_Y(x):
    Coeff = (x*np.exp(x))/((np.exp(x)-1)**2)
    return Coeff*(x*(np.exp(x)+1)/(np.exp(x)-1)-4)

# [GeV^-2] RecFast++ alpha parameter
# Note that T_M is the matter temperature in units of GeV
def Rec_alpha(T_M):
    t = T_M/((1e4)*(kb))
    return (c**(-3))*(hbar**-2)*(1e-19)*(Rec_F*Rec_a1*t**Rec_a2)/(1+Rec_a3*t**Rec_a4)

# [GeV^1] RecFast++ beta parameter
# Note that T is the CMB temperature in units of GeV
# Note also that old RecFast++ implementations use T = T_M which is incorrect
def Rec_beta(T):
    mu_red = 1/(1+5.44617e-4)                       # [0] reduced mass of electron-proton system
    return Rec_alpha(T)*((m_e*mu_red*T)/(2*np.pi))**(3/2)*np.exp(-E_ion2s/T)
    
# [0] RecFast++ BH parameter
# Note that T is the CMB temperature in units of GeV
# Note also that old RecFast++ implementations use T = T_M which is incorrect
def Rec_BH(T):
    return np.exp(-E_Lya/T)

# [GeV^-4] KH parameter (redshift dependent)
def Rec_KH(z):
    return (L_Lya**3)/(8*np.pi*H_z(z))

# Gaunt factor
def gaunt_ff(x, z, T_m):
    x_e = x*(T_CMB(z)/T_m)
    return (1+np.log(1+np.exp(np.sqrt(3)/np.pi*(np.log(2.25/x_e)+np.log(T_m/(2*m_e)))+1.425)))

# BR emissivity coefficieent
def Lambda_BR(x, z, T_m, X_e):
    lambda_e = 2*np.pi/m_e
    Lambda_Brem = (alpha_em*lambda_e**3)/(2*np.pi*np.sqrt(6*np.pi))*(T_m/m_e)**(-7/2)* \
                  (1-Y_p)*X_e*N_b(z)*gaunt_ff(x,z,T_m)
    return Lambda_Brem

# Free-free optical depth
def Delta_tau_ff(x, z_start, z_end, T_m, X_e):
    # Note: z_start > z_end
    # Note: evaluate the various functions at z_end, because the evaluation of the Dn equation
    # is below that of the Xe and Tm equations, otherwise get unphysical absorption
    z_step = z_start - z_end
    x_e = x*(T_CMB(z_end)/T_m)
    D_tau_ff = Lambda_BR(x, z_end, T_m, X_e)*(sigma_T*X_e*(1-Y_p)*N_b(z_end))/(H_z(z_end)*(1+z_end))* \
               (1-np.exp(-x_e))/(x_e**3)*z_step
    return D_tau_ff

# 'Tilde' source term from arxiv 2404.xxxx
def Delta_Src_tilde(x, z_start, z_end, T_m, X_e, S_inj, inc_CS, inc_Src):
    # Note: z_start > z_end
    # Note: evaluate the various functions at z_end, because the evaluation of the Dn equation
    # is below that of the Xe and Tm equations, otherwise get unphysical absorption
    z_step = z_start - z_end
    x_e = x*(T_CMB(z_end)/T_m)
    Src_coeff = (x_e**3)/(Lambda_BR(x, z_end, T_m, X_e)*(1-np.exp(-x_e)))
    S_inj_format = ((H_z(z_end)*(1+z_end))/(sigma_T*X_e*(1-Y_p)*N_b(z_end)))*S_inj/z_step
    Src_tilde = Src_coeff*(inc_Src*S_inj_format+inc_CS*(1/m_e)*(T_m-T_CMB(z_end))*n_Y(x))+(1/(np.exp(x_e)-1)-1/(np.exp(x)-1))
    return Src_tilde

# Free-free heating source for matter temperature evolution
def Compute_heating(Adv_Dist, x, z_start, z_end, T_m, X_e):
    x_e = x*(T_CMB(z_end)/T_m)
    # Computes heating integral
    SPH_Coeff = ((3*X_e)/(2*(1+X_e+f_HE)*(np.pi**2)))*(T_CMB(z_end)*(T_m**3))*((sigma_T)/(H_z(z_end)*(1+z_end)))
    # Integrand vector to easily integrate over later
    SPH_Integrand = Lambda_BR(x[:], z_end, T_m, X_e)*(1-np.exp(-x_e[:]))* \
                    (n_BB(x_e[:])-n_BB(x[:])-Adv_Dist[:])
    SPH_Integral = integrate.simps(SPH_Integrand,x, even='last')
    SPH_term = SPH_Coeff*SPH_Integral
    return SPH_term
    
# -------------------------------------------------- # 
# Coupled differential equation for X_e and T_M with soft photon heating
def rhs(z,Y,SPH_heat):
    dY = np.zeros_like(Y)
 
    
    # Peebles coefficient for X_e equation
    C_H = (1+Rec_KH(z)*Lambda_2s*(1-Y[0])*(1-Y_p)*N_b(z))/ \
              (1+Rec_KH(z)*(Lambda_2s+Rec_beta(T_CMB(z)))*(1-Y[0])*(1-Y_p)*N_b(z));
    
    # Thomson coefficient term for T_M equation
    ThomCoeff = (Y[0])/(1+Y[0]+f_HE)*(8*sigma_T)/(3*m_e);

        
    # Xe equation
    dY[0] = (C_H/(H_z(z)*(1+z)))*(Rec_alpha(Y[1])*(1-Y_p)*N_b(z)*(Y[0])**2-Rec_beta(T_CMB(z))*(1-Y[0])*Rec_BH(T_CMB(z)))
    # Tm equation
    dY[1] = 2*Y[1]/(1+z) + ThomCoeff*rho_CMB(z)/((1+z)*H_z(z))*(Y[1]-T_CMB(z))+SPH_heat
    return dY
# -------------------------------------------------- # 
    




