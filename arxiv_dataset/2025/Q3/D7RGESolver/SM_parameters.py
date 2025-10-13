import numpy as np
from math import  sqrt, pi 
import ckmutil
import ckmutil.ckm

#################################################################################################
#                                                                                               #
#            Default values for SM parameters: MSbar parameters at M_Z (except GF)              #
#            All the dimensionful parameters are in units of GeV to proper power                #
#                                                                                               #
#################################################################################################

## CKM paramters
Vus = 0.2243
Vub = 3.62e-3
Vcb = 4.221e-2
deltaCP = 1.27

## the mass of the standard model particles
m_e = 0.000511
m_mu = 0.1057
m_tau = 1.777
m_u = 0.00127    
m_c = 0.635      
m_d = 0.00270    
m_s = 0.0551     
m_b = 2.85       
m_t = 169.0
m_h = 130.6
MW = 80.20

## gauge couplings and Higgs potential parameters
GF = 1.1663787e-5
v=sqrt(1 / sqrt(2) / GF)
alpha_e = 1 / 127.9
alpha_s = 0.1185
e=sqrt(4 * pi * alpha_e)
g = 2 * MW / v
gp = e * g / sqrt(g**2 - e**2)
gs = sqrt(4 * pi * alpha_s)
Lambda = m_h**2 / (2 * v**2)
muh = -Lambda * v**2

# Convert into the standard model parameters in the unbroken phase
CKM = ckmutil.ckm.ckm_tree(Vus, Vub, Vcb, deltaCP)

### down-quark flavor basis
C_in_SM_down = {}
Me = np.diag([m_e, m_mu, m_tau])
Mu_down = CKM.conj().T @ np.diag([m_u, m_c, m_t])
Md_down = np.diag([m_d, m_s, m_b])
Yd_down = Md_down / (v / sqrt(2))
Yu_down = Mu_down / (v / sqrt(2))
Yl = Me / (v / sqrt(2))

# Yu_downbasis
for i in range(3):
    for j in range(3):
        C_in_SM_down[f"Yu{i+1}{j+1}"] = Yu_down[i][j]

# Yd_downbasis
for i in range(3):
    for j in range(3):
        C_in_SM_down[f"Yd{i+1}{j+1}"] = Yd_down[i][j]

# Yl
for i in range(3):
    for j in range(3):
        C_in_SM_down[f"Yl{i+1}{j+1}"] = Yl[i][j]

C_in_SM_down["gp"] = gp
C_in_SM_down["g"] = g
C_in_SM_down["gs"] = gs
C_in_SM_down["Lambda"] = Lambda
C_in_SM_down["muh"] = muh


### up-quark flavor basis
C_in_SM_up = {}
Mu_up = np.diag([m_u, m_c, m_t])
Md_up = CKM @ np.diag([m_d, m_s, m_b])
Yd_up = Md_up / (v / sqrt(2))
Yu_up = Mu_up / (v / sqrt(2))

# Yu_upbasis
for i in range(3):
    for j in range(3):
        C_in_SM_up[f"Yu{i+1}{j+1}"] = Yu_up[i][j]

# Yd_upbasis
for i in range(3):
    for j in range(3):
        C_in_SM_up[f"Yd{i+1}{j+1}"] = Yd_up[i][j]

# Yl
for i in range(3):
    for j in range(3):
        C_in_SM_up[f"Yl{i+1}{j+1}"] = Yl[i][j]

C_in_SM_up["gp"] = gp
C_in_SM_up["g"] = g
C_in_SM_up["gs"] = gs
C_in_SM_up["Lambda"] = Lambda
C_in_SM_up["muh"] = muh