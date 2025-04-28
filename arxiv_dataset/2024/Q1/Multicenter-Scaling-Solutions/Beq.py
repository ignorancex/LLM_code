import sympy as sy
from parameters import nn
from parameters import g_inv_sq

'''
Eq.s references refer to 1709.03985
CONVENTIONS: indices i,j = 1,2,3 refer to fluxes, a,b = 0, ..., nn-1 label
             the centers
'''

k = sy.IndexedBase('k')  # Fluxes, indices (a,i)
q = sy.IndexedBase('q')  # Charges, index (a)
lz = sy.IndexedBase('lz')  # Asymptotic value of L_i, index (i)
pcent = sy.IndexedBase('pcent')  # Position of the centers, indices (a,i)
j = sy.symbols('j', cls=sy.Idx)
mu = sy.symbols('mu')  # Scaling parameter

'''omog_var: auxiliary variable.
   omog_var = 0 : AdS_2 bubble equation (homogeneous)
   omog_var = 1 : Asymp flat bubble equation (non-homogeneous)
'''
omog_var = sy.symbols('omog_var')


# Define distances between the centers
dcent = sy.zeros(nn, nn)

for a in range(nn):
    for b in range(nn):
        dcent[a, b] = sy.sqrt(sy.summation((pcent[a, j]-pcent[b, j])**2, [j, 0, 2]))
                                

# Define P and T tensors as in eq.(2.5)
P0 = sy.zeros(nn, nn)
P1 = sy.zeros(nn, nn)
P2 = sy.zeros(nn, nn)
T = sy.zeros(nn, nn)

for a in range(nn):
    for b in range(nn):
        T[a, b] = q[a]**(-2) + q[b]**(-2)
        P0[a, b] = k[b, 0]/q[b]-k[a, 0]/q[a]
        P1[a, b] = k[b, 1]/q[b]-k[a, 1]/q[a]
        P2[a, b] = k[b, 2]/q[b]-k[a, 2]/q[a]


# Define ALPHABAR, ALPHADOT, BETABAR, BETADOT eq.s(3.21),(3.22)

alphabar = sy.zeros(nn, nn)
alphadot = sy.zeros(nn, nn)
betabar = []
betadot = []

for a in range(nn):
    for b in range(nn):
        if a != b:
            alphabar[a, b] = q[a]*q[b]*P0[a, b]*P1[a, b]/dcent[a, b]
            alphadot[a, b] = - omog_var * q[a]*q[b]*lz[2]
        else:
            alphabar[a, b] = 0
            alphadot[a, b] = 0
            
for a in range(nn):
    tempbetabar = 0
    tempbetadot = 0
    for b in range(nn):
        if a != b:
            tempbetabar = tempbetabar + \
                g_inv_sq * q[a]*q[b]/dcent[a, b]*T[a, b]/2*P0[a, b]
            tempbetadot = tempbetadot + \
                omog_var * q[a]*q[b]*(lz[0]*P0[a, b]+lz[1]*P1[a, b])
                          
    betabar.append(tempbetabar)
    betadot.append(tempbetadot)
       

# Define X as in eq.(3.3)
            
X = []

for ab in range(nn-1):
    X.append(P2[0, ab+1])

          
# Define MBAR, MDOT, BBAR, BDOT as in eq.s (3.24), (3.25) 

Mbar = sy.zeros(nn-1, nn-1)
Mdot = sy.zeros(nn-1, nn-1)
Bbar = []
Bdot = []

for ab in range(nn-1):
    Bbar.append(betabar[ab+1])
    Bdot.append(betadot[ab+1])
    for bb in range(nn-1):
        tempc1 = 0
        tempc2 = 0
        for c in range(nn):
            tempc1 = tempc1 + alphabar[ab+1, c]
            tempc2 = tempc2 + alphadot[ab+1, c]
        Mbar[ab, bb] = alphabar[ab+1, bb+1] - sy.eye(nn-1)[bb, ab] * tempc1
        Mdot[ab, bb] = alphadot[ab+1, bb+1] - sy.eye(nn-1)[bb, ab] * tempc2


# Define bubble equations as in (3,23)

bubbleeq = []

for ab in range(nn-1):
    temp = 0
    for bb in range(nn-1):
        temp = temp + (Mbar[ab, bb] * X[bb] + mu * Mdot[ab, bb] * X[bb])
    bubbleeq.append(temp-Bbar[ab]-mu*Bdot[ab])


# Necessary condition for AdS_2 equation

nec_cond = []

for i in range(nn-1):
    nec_cond.append(q[i] * q[nn-1]* P0[i, nn-1] * (P1[i, nn-1] * P2[i, nn-1] - T[i, nn-1]/2))
    
    
# Asymptotic charges as in (A.41)-(A.45)

asympt_charges = {
    "Q0": 0,
    "Q1": 0,
    "Q2": 0,
    "JR": 0,
    "JL1": 0,
    "JL2": 0,
    "JL3": 0}


for a in range(nn):
    asympt_charges["Q0"] = asympt_charges["Q0"] + g_inv_sq/q[a]/2
    for b in range(nn):
        asympt_charges["JR"] = asympt_charges["JR"] + g_inv_sq/4 * q[b] * P0[a, b]/q[a]
        if a != b:
            asympt_charges["JL1"] = asympt_charges["JL1"] -\
                1/4 * q[a] * q[b] * P0[a, b] * (P1[a, b] * P2[a, b] - g_inv_sq * T[a, b]/2)\
                    * (pcent[a, 0]-pcent[b, 0])/dcent[a, b]
            asympt_charges["JL2"] = asympt_charges["JL2"] -\
                1/4 * q[a] * q[b] * P0[a, b] * (P1[a, b] * P2[a, b] - g_inv_sq * T[a, b]/2)\
                    * (pcent[a, 1]-pcent[b, 1])/dcent[a, b]
            asympt_charges["JL3"] = asympt_charges["JL3"] -\
                1/4 * q[a] * q[b] * P0[a, b] * (P1[a, b] * P2[a, b] - g_inv_sq * T[a, b]/2)\
                    * (pcent[a, 2] - pcent[b, 2])/dcent[a, b]
        for c in range(nn):
            asympt_charges["Q0"] = asympt_charges["Q0"] -\
                q[a] * q[b] * q[c] * P1[a, b] * P2[a, c]
            asympt_charges["Q1"] = asympt_charges["Q1"] -\
                q[a] * q[b] * q[c] * P0[a, b] * P2[a, c]
            asympt_charges["Q2"] = asympt_charges["Q2"] -\
                q[a] * q[b] * q[c] * P0[a, b] * P1[a, c]
            for d in range(nn):
                asympt_charges["JR"] = asympt_charges["JR"] -\
                    q[a] * q[b] * q[c] * q[d] * P0[a, b] * P1[a, c] * P2[a, d]/2

