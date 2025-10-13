import numpy as np
from math import pi, log, sqrt
from scipy.integrate import odeint
from copy import deepcopy
from itertools import permutations
from itertools import product
from sympy.combinatorics import Permutation

# define the Beta function 
def beta_function(C, scale):

# Yukawa couplings matrix for up, down quarks and charged leptons in SM 
    yukawa_names = ["Yd", "Yu", "Yl"]

    for name in yukawa_names:
        globals()[name] = np.array([[C.get(f"{name}{i+1}{j+1}", 0) for j in range(3)] for i in range(3)])

    gp=C.get("gp", 0)  #gauge coupling of U(1)_Y in SM 
    g=C.get("g", 0)    #gauge coupling of SU(2) in SM
    gs=C.get("gs", 0)  #gauge coupling of SU(3) in SM
    Lambda=C.get("Lambda", 0)  #quartic couplings of Higgs potential
    muh=C.get("muh", 0)  #square couplings of Higgs potential

#################################################################################################
#                                                                                               #
#                   Represent the dim-5 and dim-7 WCs in matrix form                            #
#                                                                                               #
#################################################################################################

    #2FS
    matrix_names = ["LH5", "LH", "DLDH1", "DLDH2"]

    indices = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]

    for name in matrix_names:
        globals()[f"C_{name}"] = np.zeros((3, 3), dtype=np.complex128)  

    
        for i, j in indices:
            key = f"{name}_{i}{j}" 
            globals()[f"C_{name}"][i-1, j-1] = C.get(key, 0 + 0j)
            globals()[f"C_{name}"][j-1, i-1] = C.get(key, 0 + 0j) 
     
#################################################################################################

    #2F
    matrix_names = ["LeDH", "LHW"]

    for name in matrix_names:
        globals()[f"C_{name}"] = np.zeros((3, 3), dtype=np.complex128) 
        for i in range(3):
            for j in range(3):
                
                key = f"{name}_{i+1}{j+1}"  
                globals()[f"C_{name}"][i, j] = C.get(key, 0 + 0j)  

#################################################################################################

    #2FA
    matrix_names = ["LHB"]
    indices = [(1,2), (1,3), (2,3)]
    
    for name in matrix_names:
        C_matrix = np.zeros((3, 3), dtype=np.complex128)  # Initialize with zeros
    
        for i, j in indices:
            if i != j:  
               key = f"{name}_{i}{j}"
               value = C.get(key, 0 + 0j)
               C_matrix[i-1, j-1] = value
               C_matrix[j-1, i-1] = -value  
    
    
        for i in range(3):
            C_matrix[i, i] = 0

        globals()[f"C_{name}"] = C_matrix
    
#################################################################################################
   
    #4F3S
    matrix_names = ["eLLLHS", "eddDd"]

    indices = [
    (1,1,1,1), (1,1,1,2), (1,1,1,3), (1,1,2,2), (1,1,2,3),
    (1,1,3,3), (1,2,2,2), (1,2,2,3), (1,2,3,3), (1,3,3,3),
    (2,1,1,1), (2,1,1,2), (2,1,1,3), (2,1,2,2), (2,1,2,3),
    (2,1,3,3), (2,2,2,2), (2,2,2,3), (2,2,3,3), (2,3,3,3),
    (3,1,1,1), (3,1,1,2), (3,1,1,3), (3,1,2,2), (3,1,2,3),
    (3,1,3,3), (3,2,2,2), (3,2,2,3), (3,2,3,3), (3,3,3,3),  
    ]

    for name in matrix_names:
        C_matrix = np.zeros((3, 3, 3, 3), dtype=np.complex128)  
        for i, j, k, l in indices:
            key = f"{name}_{i}{j}{k}{l}"
            value = C.get(key, 0 + 0j)          
            for perm in set(permutations([j, k, l])):  
                C_matrix[i-1, perm[0]-1, perm[1]-1, perm[2]-1] = value

        globals()[f"C_{name}"] = C_matrix
     
#################################################################################################

    #4F3A
    matrix_names = ["eLLLHA"]
    indices = [(1,1,2,3), (2,1,2,3), (3,1,2,3)] 

    for name in matrix_names:
        C_matrix = np.zeros((3, 3, 3, 3), dtype=np.complex128)  
    
        for i, j, k, l in indices:
            key = f"{name}_{i}{j}{k}{l}"
            value = C.get(key, 0 + 0j)
        
            if len({j, k, l}) == 3:  
                original = [j, k, l]
                for perm in set(permutations(original)):
                    sign = Permutation([original.index(p) for p in perm]).signature()
                    C_matrix[i-1, perm[0]-1, perm[1]-1, perm[2]-1] = sign * value

        globals()[f"C_{name}"] = C_matrix

#################################################################################################
    
    #4F     
    matrix_names = ["C_dLQLH1", "C_dLQLH2", "C_dLueH", "C_QuLLH", "C_LdudH", "C_LdQQH"]


    for name in matrix_names:
        globals()[name] = np.zeros((3, 3, 3, 3), dtype=np.complex128)

        for i in range(3):
           for j in range(3):
               for k in range(3):
                    for l in range(3):
                        key = f"{name[2:]}_{i+1}{j+1}{k+1}{l+1}"  
                        globals()[name][i, j, k, l] = C.get(key, 0 + 0j)  

#################################################################################################

    #4F3M1
    matrix_names = ["eLLLHM"]

    indices = [
    (1,1,1,2), (1,1,2,2), (1,1,3,2), (1,1,1,3), (1,1,2,3), (1,1,3,3), (1,2,2,3), (1,2,3,3), 
    (2,1,1,2), (2,1,2,2), (2,1,3,2), (2,1,1,3), (2,1,2,3), (2,1,3,3), (2,2,2,3), (2,2,3,3), 
    (3,1,1,2), (3,1,2,2), (3,1,3,2), (3,1,1,3), (3,1,2,3), (3,1,3,3), (3,2,2,3), (3,2,3,3)  
    ]

    for name in matrix_names:
        C_matrix = np.zeros((3, 3, 3, 3), dtype=np.complex128)  # Initialize with zeros
    
        for i, j, k, l in indices:  
               key = f"{name}_{i}{j}{k}{l}"
               value = C.get(key, 0 + 0j)
               C_matrix[i-1, j-1,k-1,l-1] = value
               C_matrix[i-1, l-1,k-1,j-1] = -value  

        for i in range(3):
            C_matrix[i, 1, 0, 2]=C_matrix[i, 0, 1, 2]-C_matrix[i, 0, 2, 1]
            C_matrix[i, 2, 0, 1]=-C_matrix[i, 1, 0, 2]
    
    
        for i in range(3):
            for j in range(3):
                for k in range(3):
                   C_matrix[i, j, k ,j] = 0

        globals()[f"C_{name}"] = C_matrix
  
#################################################################################################
    
    #4F3M2
    matrix_names = ["LdddHM"]

    indices = [
       (1,1,1,2), (1,2,1,2), (1,3,1,2), (1,1,1,3), (1,2,1,3), (1,3,1,3), (1,2,2,3), (1,3,2,3),
       (2,1,1,2), (2,2,1,2), (2,3,1,2), (2,1,1,3), (2,2,1,3), (2,3,1,3), (2,2,2,3), (2,3,2,3),
       (3,1,1,2), (3,2,1,2), (3,3,1,2), (3,1,1,3), (3,2,1,3), (3,3,1,3), (3,2,2,3), (3,3,2,3) 
        ]

    for name in matrix_names:
        C_matrix = np.zeros((3, 3, 3, 3), dtype=np.complex128)  # Initialize with zeros
    
        for i, j, k, l in indices:  
               key = f"{name}_{i}{j}{k}{l}"
               value = C.get(key, 0 + 0j)
               C_matrix[i-1, j-1,k-1,l-1] = value
               C_matrix[i-1, j-1,l-1,k-1] = -value  

        for i in range(3):
            C_matrix[i, 0, 1, 2]=C_matrix[i, 1, 0, 2]-C_matrix[i, 2, 0, 1]
            C_matrix[i, 0, 2, 1]=-C_matrix[i, 0, 1, 2]
    
    
        for i in range(3):
            for j in range(3):
                for k in range(3):
                   C_matrix[i, j, k ,k] = 0

        globals()[f"C_{name}"] = C_matrix
    
#################################################################################################   

    #4F2A
    matrix_names = ["eQddH"]

    indices = [
    (1,1,1,2), (1,1,1,3), (1,1,2,3),
    (1,2,1,2), (1,2,1,3), (1,2,2,3),
    (1,3,1,2), (1,3,1,3), (1,3,2,3),
    (2,1,1,2), (2,1,1,3), (2,1,2,3),
    (2,2,1,2), (2,2,1,3), (2,2,2,3),
    (2,3,1,2), (2,3,1,3), (2,3,2,3),
    (3,1,1,2), (3,1,1,3), (3,1,2,3),
    (3,2,1,2), (3,2,1,3), (3,2,2,3),
    (3,3,1,2), (3,3,1,3), (3,3,2,3)
    ]

    for name in matrix_names:
        globals()[f"C_{name}"] = np.zeros((3, 3, 3, 3), dtype=np.complex128)

        for i, j, k, l in indices:
            key = f"{name}_{i}{j}{k}{l}"
            value = C.get(key, 0 + 0j)

        
            if k != l:
                globals()[f"C_{name}"][i-1, j-1, k-1, l-1] = value
                globals()[f"C_{name}"][i-1, j-1, l-1, k-1] = -value
            else:
                globals()[f"C_{name}"][i-1, j-1, k-1, l-1] = 0 

#################################################################################################    
       
    #4F2S     
    matrix_names = ["duLDL", "LQdDd"]

    indices = [
    (1,1,1,1), (1,1,1,2), (1,1,1,3), (1,1,2,2), (1,1,2,3), (1,1,3,3),
    (1,2,1,1), (1,2,1,2), (1,2,1,3), (1,2,2,2), (1,2,2,3), (1,2,3,3),
    (1,3,1,1), (1,3,1,2), (1,3,1,3), (1,3,2,2), (1,3,2,3), (1,3,3,3),
    (2,1,1,1), (2,1,1,2), (2,1,1,3), (2,1,2,2), (2,1,2,3), (2,1,3,3),
    (2,2,1,1), (2,2,1,2), (2,2,1,3), (2,2,2,2), (2,2,2,3), (2,2,3,3),
    (2,3,1,1), (2,3,1,2), (2,3,1,3), (2,3,2,2), (2,3,2,3), (2,3,3,3),
    (3,1,1,1), (3,1,1,2), (3,1,1,3), (3,1,2,2), (3,1,2,3), (3,1,3,3),
    (3,2,1,1), (3,2,1,2), (3,2,1,3), (3,2,2,2), (3,2,2,3), (3,2,3,3),
    (3,3,1,1), (3,3,1,2), (3,3,1,3), (3,3,2,2), (3,3,2,3), (3,3,3,3),
    ]

    for name in matrix_names:
        globals()[f"C_{name}"] = np.zeros((3, 3, 3, 3), dtype=np.complex128)

        for i, j, k, l in indices:
            key = f"{name}_{i}{j}{k}{l}"
            value = C.get(key, 0 + 0j)

        
            globals()[f"C_{name}"][i-1, j-1, k-1, l-1] = value
            globals()[f"C_{name}"][i-1, j-1, l-1, k-1] = value  

#################################################################################################
    
    # conventional quantities in dim-7 SMEFT RGEs

    WH = np.trace(3*Yu.conj().T @ Yu + 3*Yd.conj().T @ Yd + Yl.conj().T @ Yl)
    TH=(3*np.trace(Yd @ Yd.conj().T @ Yd @ Yd.conj().T) 
           +3*np.trace(Yu @ Yu.conj().T @ Yu @ Yu.conj().T) 
           +np.trace(Yl @ Yl.conj().T @ Yl @ Yl.conj().T)) 
    
#################################################################################################

    Beta= {}  

#################################################################################################
#                                                                                               #
#                              beta function for SM parameters                                  #
#                                                                                               #
#################################################################################################

    Beta["gp"] = 41/6*gp**3
    Beta["g"] = -19/6*g**3
    Beta["gs"] = -7*gs**3
    Beta["Lambda"] = 3/8*gp**4 + 9/8*g**4+ 3/4*g**2*gp**2- 2*TH + (-3*gp**2-9*g**2+24*Lambda+4*WH)*Lambda
    Beta["muh"] = muh*(- 3/2*gp**2- 9/2*g**2  + 12*Lambda + 2*WH)
 
    indices = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]  
   
    for p, r in indices:
        Beta[f"Yu{p}{r}"] = (- 17/12*gp**2 - 9/4*g**2 - 8*gs**2 + WH )*Yu[p-1,r-1]\
                             +3/2*((Yu @ Yu.conj().T - Yd @ Yd.conj().T)@ Yu)[p-1,r-1] 
                                  
    for p, r in indices:
        Beta[f"Yd{p}{r}"] = (-5/12*gp**2 -9/4*g**2- 8*gs**2+WH)*Yd[p-1,r-1]\
                            -3/2*((Yu @ Yu.conj().T -Yd @ Yd.conj().T)@ Yd)[p-1,r-1]
                                  
    for p, r in indices:
        Beta[f"Yl{p}{r}"] = (- 15/4*gp**2-9/4*g**2 +WH)*Yl[p-1,r-1]\
                            +3/2*(Yl @ Yl.conj().T @ Yl)[p-1,r-1] 
                            
#################################################################################################
#                                                                                               #
#                        Beta function for dim-5 + dim-7 SMEFT operators                        #
#                                                                                               #
#################################################################################################

    # 2FS: LH5 LH DLDH1 DLDH2
    indices = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)] 

    """Dimension-5 operator"""
    
    for p, r in indices:
        Beta[f"LH5_{p}{r}"] = (
            ###### dim-5 contribution to LH5
            1/2*(-3*g**2 + 4*Lambda + 2*WH)*C_LH5[p-1,r-1]
            +1/2*(-3*g**2 + 4*Lambda + 2*WH)*C_LH5[r-1,p-1]
            -3/2*(C_LH5 @ Yl @ Yl.conj().T)[p-1,r-1]
            -3/2*(C_LH5 @ Yl @ Yl.conj().T)[r-1,p-1]
            ###### dim-7 contribution to LH5
            +muh*(
                8*C_LH[p-1,r-1]
                +8*C_LH[r-1,p-1]
                +2*(C_LeDH @ Yl.conj().T)[p-1,r-1]
                +2*(C_LeDH @ Yl.conj().T)[r-1,p-1]
                +3/2*g**2*(2*C_DLDH1+C_DLDH2)[p-1,r-1]
                +3/2*g**2*(2*C_DLDH1+C_DLDH2)[r-1,p-1]
                +(C_DLDH1 @ Yl @ Yl.conj().T)[p-1,r-1]
                +(C_DLDH1 @ Yl @ Yl.conj().T)[r-1,p-1]
                -1/2*(C_DLDH2 @ Yl @ Yl.conj().T)[p-1,r-1]
                -1/2*(C_DLDH2 @ Yl @ Yl.conj().T)[r-1,p-1]
                -np.tensordot(Yl[:, :], (3*C_eLLLHS[:,:,p-1,r-1]+2*C_eLLLHM[:,:,p-1,r-1]), axes=([0, 1], [1, 0]))
                -np.tensordot(Yl[:, :], (3*C_eLLLHS[:,:,r-1,p-1]+2*C_eLLLHM[:,:,r-1,p-1]), axes=([0, 1], [1, 0]))
                -3*np.tensordot(Yd[:, :], C_dLQLH1[:,p-1,:,r-1], axes=([0, 1], [1, 0]))
                -3*np.tensordot(Yd[:, :], C_dLQLH1[:,r-1,:,p-1], axes=([0, 1], [1, 0]))
                +6*np.tensordot(Yu.conj().T[:, :], C_QuLLH[:,:,p-1,r-1], axes=([0, 1], [1, 0]))
                +6*np.tensordot(Yu.conj().T[:, :], C_QuLLH[:,:,r-1,p-1], axes=([0, 1], [1, 0]))           
                )
            )

    for p, r in indices:
        Beta[f"LH_{p}{r}"] = (
            - 1 / 4 * (3 * gp**2 + 15*g**2 -80 * Lambda - 8 * WH) * C_LH[p-1, r-1]
            - 1 / 4 * (3 * gp**2 + 15*g**2 -80 * Lambda - 8 * WH) * C_LH[r-1, p-1]
            - 3 / 2 * (C_LH @ Yl @ Yl.conj().T)[p-1, r-1]
            - 3 / 2 * (C_LH @ Yl @ Yl.conj().T)[r-1, p-1]
            +(2*Lambda-3/2*g**2)*(C_LeDH@Yl.conj().T)[p-1, r-1]
            +(2*Lambda-3/2*g**2)*(C_LeDH@Yl.conj().T)[r-1, p-1]
            +(C_LeDH@Yl.conj().T@Yl@Yl.conj().T)[p-1, r-1]
            +(C_LeDH@Yl.conj().T@Yl@Yl.conj().T)[r-1, p-1]
            - 3 / 4 *g**2*(g**2-4*Lambda)* C_DLDH1[p-1, r-1]
            - 3 / 4 *g**2*(g**2-4*Lambda)* C_DLDH1[r-1, p-1]
            + Lambda* (C_DLDH1 @ Yl @ Yl.conj().T)[p-1, r-1]
            + Lambda* (C_DLDH1 @ Yl @ Yl.conj().T)[r-1, p-1]
            - (C_DLDH1 @ Yl @ Yl.conj().T @ Yl @ Yl.conj().T)[p-1, r-1]
            - (C_DLDH1 @ Yl @ Yl.conj().T @ Yl @ Yl.conj().T)[r-1, p-1]
            - 3 / 8 *(gp**4+2*gp**2*g**2+3*g**4-4*g**2*Lambda)*C_DLDH2[p-1, r-1]
            - 3 / 8 *(gp**4+2*gp**2*g**2+3*g**4-4*g**2*Lambda)*C_DLDH2[r-1, p-1]
            - 1 / 2 *Lambda* (C_DLDH2 @ Yl @ Yl.conj().T)[p-1, r-1]
            - 1 / 2 *Lambda* (C_DLDH2 @ Yl @ Yl.conj().T)[r-1, p-1]
            - (C_DLDH2 @ Yl @ Yl.conj().T @ Yl @ Yl.conj().T)[p-1, r-1]
            - (C_DLDH2 @ Yl @ Yl.conj().T @ Yl @ Yl.conj().T)[r-1, p-1]
            - 3*g**3*C_LHW[p-1, r-1]
            - 3*g**3*C_LHW[r-1, p-1]
            -6*g*(C_LHW @ Yl @ Yl.conj().T)[p-1, r-1]
            -6*g*(C_LHW @ Yl @ Yl.conj().T)[r-1, p-1]
            -3*np.tensordot(C_eLLLHS[:, :, p-1, r-1], (Lambda*Yl-Yl@ Yl.conj().T@ Yl)[:, :], axes=([1, 0], [0, 1]))
            -3*np.tensordot(C_eLLLHS[:, :, r-1, p-1], (Lambda*Yl-Yl@ Yl.conj().T@ Yl)[:, :], axes=([1, 0], [0, 1]))
            -2*np.tensordot(C_eLLLHM[:, :, p-1, r-1], (Lambda*Yl-Yl@ Yl.conj().T@ Yl)[:, :], axes=([1, 0], [0, 1]))
            -2*np.tensordot(C_eLLLHM[:, :, r-1, p-1], (Lambda*Yl-Yl@ Yl.conj().T@ Yl)[:, :], axes=([1, 0], [0, 1]))
            -3*np.tensordot(C_dLQLH1[:, p-1, :, r-1], (Lambda*Yd-Yd@ Yd.conj().T@ Yd)[:, :], axes=([1, 0], [0, 1]))
            -3*np.tensordot(C_dLQLH1[:, r-1, :, p-1], (Lambda*Yd-Yd@ Yd.conj().T@ Yd)[:, :], axes=([1, 0], [0, 1]))
            +6*np.tensordot(C_QuLLH[:, :, p-1, r-1], (Lambda*Yu.conj().T-Yu.conj().T@ Yu@ Yu.conj().T)[:, :], axes=([1, 0], [0, 1]))
            +6*np.tensordot(C_QuLLH[:, :, r-1, p-1], (Lambda*Yu.conj().T-Yu.conj().T@ Yu@ Yu.conj().T)[:, :], axes=([1, 0], [0, 1]))
          )
     
    for p, r in indices:
        Beta[f"DLDH1_{p}{r}"] = (
            - 1 / 4 * (3 * gp**2 - 11 * g**2 - 4 * WH) * C_DLDH1[p-1, r-1]  
            - 1 / 4 * (3 * gp**2 - 11 * g**2 - 4 * WH) * C_DLDH1[r-1, p-1]
            + 7 / 2 * (C_DLDH1 @ Yl @ Yl.conj().T)[p-1, r-1]
            + 7 / 2 * (C_DLDH1@ Yl @ Yl.conj().T)[r-1, p-1]
            - 1 / 8 * (11 * gp**2 + 11 * g**2 + 8 * Lambda) * C_DLDH2[p-1, r-1]
            - 1 / 8 * (11 * gp**2 + 11 * g**2 + 8 * Lambda) * C_DLDH2[r-1, p-1]
            +6*np.tensordot(C_duLDL[:, :, p-1, r-1], (Yu.conj().T @ Yd)[:, :], axes=([0, 1], [1, 0]))
            +6*np.tensordot(C_duLDL[:, :, r-1, p-1], (Yu.conj().T @ Yd)[:, :], axes=([0, 1], [1, 0]))
        )

    for p, r in indices:
        Beta[f"DLDH2_{p}{r}"] = (
            - 4 * g**2 * C_DLDH1[p-1, r-1]
            - 4 * g**2 * C_DLDH1[r-1, p-1]
            - 4 * (C_DLDH1 @ Yl @ Yl.conj().T)[p-1, r-1]
            - 4 * (C_DLDH1 @ Yl @ Yl.conj().T)[r-1, p-1]
            + 1 / 2 * (4 * gp**2 + g**2 + 4 * Lambda + 2 * WH) * C_DLDH2[p-1, r-1]
            + 1 / 2 * (4 * gp**2 + g**2 + 4 * Lambda + 2 * WH) * C_DLDH2[r-1, p-1]
            - 3 / 2 * (C_DLDH2 @ Yl @ Yl.conj().T)[p-1, r-1]
            - 3 / 2 * (C_DLDH2 @ Yl @ Yl.conj().T)[r-1, p-1]
          )
        
####################################################################################################    
        
    # 2F: LHW LeDH 
    indices = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]  
       
    for p, r in indices:
        Beta[f"LHW_{p}{r}"] = (
             1 / 2 * g**3 * C_DLDH1[p-1, r-1]  
             - 1 / 4 * g * (C_DLDH1 @ Yl @ Yl.conj().T)[r-1, p-1]
             + 1 / 2 * g * (C_DLDH1 @ Yl @ Yl.conj().T)[p-1, r-1]
             + 5 / 8 * g**3 * C_DLDH2[p-1, r-1]
             + 3 / 4 * g * (C_DLDH2 @ Yl @ Yl.conj().T)[p-1, r-1]
             + 1 / 8 * g * (C_DLDH2 @ Yl @ Yl.conj().T)[r-1, p-1]
             -1/2*(4*gp**2-9*g**2-8*Lambda-4*WH)*C_LHW[p-1, r-1]
             +7/2*g**2*C_LHW[r-1, p-1]
             + 9 / 2 * (C_LHW @ Yl @ Yl.conj().T)[p-1, r-1]
             + 2 * (C_LHW @ Yl @ Yl.conj().T)[r-1, p-1]
             -3/2*(C_LHW.T @ Yl @ Yl.conj().T)[r-1, p-1]  
             +3*gp*g*C_LHB[p-1, r-1]     
             -1/4*g*np.tensordot((3*C_eLLLHS+C_eLLLHA-2*C_eLLLHM)[:,:,p-1,r-1], Yl[:, :], axes=([0, 1], [1, 0]))
             -3/4*g*np.tensordot(C_dLQLH1[:,r-1,:,p-1], Yd[:, :], axes=([0, 1], [1, 0]))
             -3/4*g*np.tensordot(C_dLQLH2[:,p-1,:,r-1]+C_dLQLH2[:,r-1,:,p-1], Yd[:, :], axes=([0, 1], [1, 0]))           
        )
     
    for p, r in indices:
        Beta[f"LeDH_{p}{r}"] = (
             -3/2*(3*gp**2-4*Lambda-2*WH)*C_LeDH[p-1, r-1]
             +(Yl.T @ C_LeDH @ Yl.conj().T)[r-1, p-1]
             +4*(C_LeDH @ Yl.conj().T @ Yl)[p-1, r-1]
             +1/2*(C_LeDH.T @ Yl @ Yl.conj().T)[r-1, p-1]
             +(3*gp**2-g**2)*(C_DLDH1 @ Yl)[p-1, r-1]
             -2*(C_DLDH1 @ Yl @ Yl.conj().T @ Yl)[p-1, r-1]
             +1/8*(7*gp**2-17*g**2-8*Lambda)*(C_DLDH2 @ Yl)[p-1, r-1]
             -2*(C_DLDH2 @ Yl @ Yl.conj().T @ Yl)[p-1, r-1]
             -1/2*(Yl.T @ C_DLDH2 @ Yl @ Yl.conj().T)[r-1, p-1]
             -6*np.tensordot(C_dLueH[:,p-1,:,r-1], Yu.conj().T @ Yd[:, :], axes=([0, 1], [1, 0]))
        )
        
####################################################################################################

    # 2FA：LHB 
    indices = [(1,2), (1,3), (2,3)]  
       
    for p, r in indices:
        Beta[f"LHB_{p}{r}"] = (
             -1 / 8 * gp * (C_DLDH1 @ Yl @ Yl.conj().T)[p-1, r-1]
             +1 / 8 * gp * (C_DLDH1 @ Yl @ Yl.conj().T)[r-1, p-1]
             -3 / 16 * gp * (C_DLDH2 @ Yl @ Yl.conj().T)[p-1, r-1]
             +3 / 16 * gp * (C_DLDH2 @ Yl @ Yl.conj().T)[r-1, p-1]
             +3 * gp * g * C_LHW[p-1, r-1]
             -3 * gp * g * C_LHW[r-1, p-1]
             -3 / 2 * (C_LHB @ Yl @ Yl.conj().T)[p-1, r-1]
             +3 / 2 * (C_LHB @ Yl @ Yl.conj().T)[r-1, p-1]
             +1/12*(47*gp**2-30*g**2+24*Lambda+12*WH)*C_LHB[p-1, r-1]
             -1/12*(47*gp**2-30*g**2+24*Lambda+12*WH)*C_LHB[r-1, p-1]
             +3/8*gp*np.tensordot((C_eLLLHA-2*C_eLLLHM)[:,:,p-1,r-1], Yl[:, :], axes=([0, 1], [1, 0]))
             -3/8*gp*np.tensordot((C_eLLLHA-2*C_eLLLHM)[:,:,r-1,p-1], Yl[:, :], axes=([0, 1], [1, 0]))
             -1/8*gp*np.tensordot(C_dLQLH1[:,p-1,:,r-1], Yd[:, :], axes=([0, 1], [1, 0]))
             +1/8*gp*np.tensordot(C_dLQLH1[:,r-1,:,p-1], Yd[:, :], axes=([0, 1], [1, 0]))        
        )

####################################################################################################

    # 4F2S: duLDL LQdDd 
    indices = [
    (1,1,1,1), (1,1,1,2), (1,1,1,3), (1,1,2,2), (1,1,2,3), (1,1,3,3),
    (1,2,1,1), (1,2,1,2), (1,2,1,3), (1,2,2,2), (1,2,2,3), (1,2,3,3),
    (1,3,1,1), (1,3,1,2), (1,3,1,3), (1,3,2,2), (1,3,2,3), (1,3,3,3),
    (2,1,1,1), (2,1,1,2), (2,1,1,3), (2,1,2,2), (2,1,2,3), (2,1,3,3),
    (2,2,1,1), (2,2,1,2), (2,2,1,3), (2,2,2,2), (2,2,2,3), (2,2,3,3),
    (2,3,1,1), (2,3,1,2), (2,3,1,3), (2,3,2,2), (2,3,2,3), (2,3,3,3),
    (3,1,1,1), (3,1,1,2), (3,1,1,3), (3,1,2,2), (3,1,2,3), (3,1,3,3),
    (3,2,1,1), (3,2,1,2), (3,2,1,3), (3,2,2,2), (3,2,2,3), (3,2,3,3),
    (3,3,1,1), (3,3,1,2), (3,3,1,3), (3,3,2,2), (3,3,2,3), (3,3,3,3),
      ] 

    for p, r, s, t in indices:
        Beta[f"duLDL_{p}{r}{s}{t}"] = (
            (2*C_DLDH1+C_DLDH2)[s-1, t-1] * (Yd.conj().T @ Yu)[p-1, r-1]  
            + 1/6*(gp**2+9*g**2)*C_duLDL[p-1, r-1, s-1, t-1]   
            + C_duLDL[:, r-1, s-1, t-1] @ (Yd.conj().T @ Yd)[p-1, :]
            + C_duLDL[p-1, :, s-1, t-1] @ (Yu.conj().T @ Yu)[:, r-1]
            + 1/2*C_duLDL[p-1, r-1, :, t-1] @ (Yl @ Yl.conj().T)[:, s-1]
            + 1/2*C_duLDL[p-1, r-1, s-1, :] @ (Yl @ Yl.conj().T)[:, t-1]
        )

    for p, r, s, t in indices:
        Beta[f"LQdDd_{p}{r}{s}{t}"] = (
           - 3 * np.tensordot(C_eddDd[:, :, s-1, t-1], Yl[p-1, :], axes=([0], [0])) @ (Yd.conj().T)[:, r-1]
           + 4/9*(gp**2+3*gs**2)*C_LQdDd[p-1, r-1, s-1, t-1]
           - np.tensordot(C_LQdDd[p-1, :, :, t-1], (Yd.conj().T)[:, r-1], axes=([1], [0])) @ Yd[:, s-1]
           - np.tensordot(C_LQdDd[p-1, :, s-1, :], (Yd.conj().T)[:, r-1], axes=([1], [0])) @ Yd[:, t-1]
           + 1/2*C_LQdDd[:, r-1, s-1, t-1] @ (Yl @ Yl.conj().T)[p-1, :]
           + 1/2*C_LQdDd[p-1, :, s-1, t-1] @ (Yd @ Yd.conj().T+Yu @ Yu.conj().T)[:, r-1]
           + C_LQdDd[p-1, r-1, :, t-1] @ (Yd @ Yd.conj().T)[:, s-1]
           + C_LQdDd[p-1, r-1, s-1, :] @ (Yd @ Yd.conj().T)[:, t-1]
        )

####################################################################################################

    # 4F: LdudH QuLLH LdQQH dLQLH1 dLQLH2 dLueH   
    indices = [
    (1,1,1,1), (1,1,1,2), (1,1,1,3), (1,1,2,1), (1,1,2,2), (1,1,2,3), (1,1,3,1), (1,1,3,2), (1,1,3,3),
    (1,2,1,1), (1,2,1,2), (1,2,1,3), (1,2,2,1), (1,2,2,2), (1,2,2,3), (1,2,3,1), (1,2,3,2), (1,2,3,3),
    (1,3,1,1), (1,3,1,2), (1,3,1,3), (1,3,2,1), (1,3,2,2), (1,3,2,3), (1,3,3,1), (1,3,3,2), (1,3,3,3),
    (2,1,1,1), (2,1,1,2), (2,1,1,3), (2,1,2,1), (2,1,2,2), (2,1,2,3), (2,1,3,1), (2,1,3,2), (2,1,3,3),
    (2,2,1,1), (2,2,1,2), (2,2,1,3), (2,2,2,1), (2,2,2,2), (2,2,2,3), (2,2,3,1), (2,2,3,2), (2,2,3,3),
    (2,3,1,1), (2,3,1,2), (2,3,1,3), (2,3,2,1), (2,3,2,2), (2,3,2,3), (2,3,3,1), (2,3,3,2), (2,3,3,3),
    (3,1,1,1), (3,1,1,2), (3,1,1,3), (3,1,2,1), (3,1,2,2), (3,1,2,3), (3,1,3,1), (3,1,3,2), (3,1,3,3),
    (3,2,1,1), (3,2,1,2), (3,2,1,3), (3,2,2,1), (3,2,2,2), (3,2,2,3), (3,2,3,1), (3,2,3,2), (3,2,3,3),
    (3,3,1,1), (3,3,1,2), (3,3,1,3), (3,3,2,1), (3,3,2,2), (3,3,2,3), (3,3,3,1), (3,3,3,2), (3,3,3,3)
    ]  
      
    for p, r, s, t in indices:
        Beta[f"LdudH_{p}{r}{s}{t}"] = (
             -1/12*(17*gp**2+27*g**2+48*gs**2-12*WH)*C_LdudH[p-1, r-1, s-1, t-1]
             -10/3*gp**2*C_LdudH[p-1, t-1, s-1, r-1]
             +3*C_LdudH[p-1, :, s-1, t-1] @ (Yd.conj().T @ Yd)[:, r-1]
             +3*C_LdudH[p-1, r-1, s-1, :] @ (Yd.conj().T @ Yd)[:, t-1]
             -3/2*C_LdudH[:, r-1, s-1, t-1] @ (Yl @ Yl.conj().T)[p-1, :]
             +2*C_LdudH[p-1, r-1, :, t-1] @ (Yu.conj().T @ Yu)[:, s-1]
             -2*(C_LdddHM[p-1,r-1,:,t-1]+C_LdddHM[p-1,:,r-1,t-1]) @ (Yd.conj().T @ Yu)[:, s-1]
             +4*np.tensordot(C_eQddH[:, :, r-1, t-1], Yl[p-1, :], axes=([0], [0])) @ Yu[:, s-1]
             -2*np.tensordot(C_LdQQH[p-1, r-1, :, :], Yu[:, s-1], axes=([0], [0])) @ Yd[:, t-1]
             -2*np.tensordot(C_LdQQH[p-1, r-1, :, :], Yu[:, s-1], axes=([1], [0])) @ Yd[:, t-1]
             +np.tensordot(C_eddDd[:, r-1, t-1, :], (Yd.conj().T @ Yu)[:, s-1], axes=([1], [0])) @ Yl[p-1, :]
             +1/18*(29*gp**2+27*g**2+96*gs**2)*C_LQdDd[p-1, :, r-1, t-1] @ Yu[:, s-1]
             - 2*np.tensordot(C_LQdDd[p-1, :, r-1, :], (Yd.conj().T @ Yd)[:, t-1], axes=([1], [0])) @ Yu[:, s-1]
             - np.tensordot(C_LQdDd[p-1, :, r-1, :], (Yd.conj().T @ Yu)[:, s-1], axes=([1], [0])) @ Yd[:, t-1]
             + 2*np.tensordot(C_LQdDd[p-1, :, :, t-1], (Yd.conj().T @ Yd)[:, r-1], axes=([1], [0])) @ Yu[:, s-1]         
        )


    for p, r, s, t in indices:
        Beta[f"QuLLH_{p}{r}{s}{t}"] = (
             Yu[p-1, r-1] * (3*g**2*C_DLDH1[s-1, t-1]
             +(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, s-1]
             + 2* (C_DLDH1 @ Yl @ Yl.conj().T)[s-1, t-1])
             +1/2*Yu[p-1, r-1] *(3*g**2*C_DLDH2[s-1, t-1]+4*(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, s-1]-(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, t-1])
             -Yu[p-1,r-1]*(3*np.tensordot(Yl[:, :], C_eLLLHS[:,:,s-1,t-1], axes=([0,1], [1,0]))
                            +3*np.tensordot(Yl[:, :], C_eLLLHA[:,:,s-1,t-1], axes=([0,1], [1,0]))
                            +4*np.tensordot(Yl[:, :], C_eLLLHM[:,:,s-1,t-1], axes=([0,1], [1,0]))
                            -2*np.tensordot(Yl[:, :], C_eLLLHM[:,:,t-1,s-1], axes=([0,1], [1,0])))           
            +np.tensordot((C_dLQLH1[:,s-1,:,t-1]+C_dLQLH2[:,s-1,:,t-1]-C_dLQLH2[:,t-1,:,s-1]), Yd[p-1,:], axes=([0], [0]))@Yu[:,r-1]
            -3*np.tensordot((C_dLQLH1[:,s-1,:,t-1]+C_dLQLH2[:,s-1,:,t-1]-C_dLQLH2[:,t-1,:,s-1]), Yd[:,:], axes=([0,1], [1,0]))*Yu[p-1,r-1]
            +np.tensordot(C_dLueH[:,t-1,r-1,:], Yd[p-1,:], axes=([0], [0]))@Yl.conj().T[:,s-1]            
             + 1/12*(gp**2-45*g**2-96*gs**2+12*WH)*C_QuLLH[p-1, r-1, s-1, t-1]
             + 3*g**2*C_QuLLH[p-1, r-1, t-1,  s-1]
             + 5/2*C_QuLLH[p-1, r-1, :, t-1] @ (Yl @ Yl.conj().T)[:, s-1]
             - 1/2*(3*C_QuLLH[p-1, r-1, s-1, :]+ 4*C_QuLLH[p-1, r-1, :, s-1]) @ (Yl @ Yl.conj().T)[:, t-1]          
             - 1/2*C_QuLLH[:, r-1, s-1, t-1] @ (Yd @ Yd.conj().T- Yu @ Yu.conj().T)[p-1, :]
             + C_QuLLH[:, r-1, t-1, s-1] @ (2*Yu @ Yu.conj().T+Yd @ Yd.conj().T)[p-1, :]
             + 3*C_QuLLH[p-1, :, s-1, t-1] @ (Yu.conj().T @ Yu)[:, r-1]
             + 6*np.tensordot(C_QuLLH[:, :, s-1, t-1], Yu.conj().T[:, :], axes=([0,1],[1,0]))*Yu[p-1, r-1]
             +3*g**2*C_duLDL[:, r-1, s-1, t-1] @ Yd[p-1, :]
             + 2*np.tensordot(C_duLDL[:, r-1, t-1, :], Yd[p-1, :], axes=([0], [0])) @ (Yl @ Yl.conj().T)[:, s-1]
        )
        
    for p, r, s, t in indices:
        Beta[f"LdQQH_{p}{r}{s}{t}"] = (
             -2*np.tensordot(C_LdudH[p-1, r-1, :, :], Yu.conj().T[:, s-1], axes=([0], [0])) @ Yd.conj().T[:, t-1]
             -2*np.tensordot(C_LdudH[p-1, r-1, :, :], Yu.conj().T[:, t-1], axes=([0], [0])) @ Yd.conj().T[:, s-1]
             -np.tensordot(C_LdudH[p-1, :, :,r-1], Yu.conj().T[:, s-1], axes=([1], [0])) @ Yd.conj().T[:, t-1]
             -np.tensordot(C_LdudH[p-1, :, :,r-1], Yu.conj().T[:, t-1], axes=([1], [0])) @ Yd.conj().T[:, s-1]
             -2*np.tensordot(C_eQddH[:, t-1, :, r-1], Yl[p-1,:], axes=([0], [0])) @ Yd.conj().T[:, s-1]
             - 1/12*(19*gp**2+45*g**2+48*gs**2-12*WH)*C_LdQQH[p-1, r-1, s-1, t-1]
             -3*g**2*C_LdQQH[p-1, r-1, t-1, s-1]
             -np.tensordot(C_LdQQH[p-1, :, :, t-1], Yd.conj().T[:, s-1], axes=([0], [0])) @ Yd[:, r-1]
             -np.tensordot(C_LdQQH[p-1, :, s-1, :], Yd.conj().T[:, t-1], axes=([0], [0])) @ Yd[:, r-1]
             +1/2*(C_LdQQH[:,r-1,s-1,t-1]-4*C_LdQQH[:,r-1,t-1,s-1])@(Yl @ Yl.conj().T)[p-1, :]
             +3*C_LdQQH[p-1, :, s-1, t-1] @ (Yd.conj().T @ Yd)[:, r-1]
             +1/2*C_LdQQH[p-1, r-1, :, t-1] @ (Yd @ Yd.conj().T+5*Yu @ Yu.conj().T)[:, s-1]
             +1/2*(4*C_LdQQH[p-1, r-1, :, s-1]-3*C_LdQQH[p-1, r-1, s-1, :]) @ (Yu @ Yu.conj().T)[:, t-1]
             +1/2*(5*C_LdQQH[p-1, r-1, s-1, :]-2*C_LdQQH[p-1, r-1, :, s-1]) @ (Yd @ Yd.conj().T)[:, t-1]
             +3*np.tensordot(np.tensordot(C_eddDd[:, :, :, r-1], Yl[p-1, :], axes=([0], [0])), Yd.conj().T[:, s-1], axes=([0], [0]))@Yd.conj().T[:, t-1]
             -2/9*(gp**2-24*gs**2)*C_LQdDd[p-1, s-1, r-1, :] @ Yd.conj().T[:, t-1]
             +2*np.tensordot(C_LQdDd[p-1, s-1, :, :], (Yd.conj().T @ Yd)[:, r-1], axes=([0], [0])) @ Yd.conj().T[:, t-1]
             -np.tensordot(C_LQdDd[:, t-1, r-1, :], (Yl @ Yl.conj().T)[p-1, :], axes=([0], [0])) @ Yd.conj().T[:, s-1]
             +np.tensordot(C_LQdDd[p-1, :, r-1, :], (Yu @ Yu.conj().T)[:, s-1], axes=([0], [0])) @ Yd.conj().T[:, t-1]
             +np.tensordot(C_LQdDd[p-1, :, r-1, :], (Yu @ Yu.conj().T)[:, t-1], axes=([0], [0])) @ Yd.conj().T[:, s-1]
        )

    for p, r, s, t in indices:
        Beta[f"dLQLH1_{p}{r}{s}{t}"] = (
            -3*g**2*Yd.conj().T[p-1, s-1]*(2*C_DLDH1+C_DLDH2)[r-1, t-1]
            -3*Yd.conj().T[p-1, s-1]*((C_DLDH1 @ Yl @ Yl.conj().T)[r-1, t-1]
                                      +(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, r-1])
            -3/2*Yd.conj().T[p-1, s-1]*((C_DLDH2 @ Yl @ Yl.conj().T)[r-1, t-1]
                                      +(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, r-1])
            +8*g*Yd.conj().T[p-1, s-1]*(C_LHW[r-1, t-1]-C_LHW[t-1, r-1])
            -8/3*gp*Yd.conj().T[p-1, s-1]*C_LHB[r-1, t-1]
            +2*Yd.conj().T[p-1, s-1]*np.tensordot(Yl[:,:], (3*C_eLLLHS+C_eLLLHM)[:, :,r-1, t-1]+C_eLLLHM[:, :,t-1, r-1], axes=([0,1], [1,0])) 
            - 1/36*(41*gp**2+63*g**2+96*gs**2-36*WH)*C_dLQLH1[p-1, r-1, s-1, t-1]
            + 4/9*(5*gp**2+9*g**2-12*gs**2)*C_dLQLH1[p-1, t-1, s-1, r-1]
            +1/2*C_dLQLH1[p-1, r-1, :, t-1] @ (5*Yd @ Yd.conj().T+Yu @ Yu.conj().T)[:, s-1]
            +3*C_dLQLH1[:, r-1, s-1, t-1] @ (Yd.conj().T @ Yd)[p-1, :]
            +1/2*C_dLQLH1[p-1, :, s-1, t-1] @ (Yl @ Yl.conj().T)[:, r-1]
            -3/2*C_dLQLH1[p-1, r-1, s-1, :] @ (Yl @ Yl.conj().T)[:, t-1]
            +3*Yd.conj().T[p-1, s-1]*np.tensordot(Yd[:, :], C_dLQLH1[:, r-1, :, t-1]+C_dLQLH1[:, t-1, :, r-1], axes=([0, 1], [1, 0]))
            +2*g**2*(2*C_dLQLH2[p-1,r-1,s-1,t-1]+C_dLQLH2[p-1,t-1,s-1,r-1])
            +2*C_dLQLH2[p-1,:,s-1,t-1]@(Yl @ Yl.conj().T)[:, r-1]
            -2*C_dLQLH2[p-1,r-1,s-1,:]@(Yl @ Yl.conj().T)[:, t-1]
            -2*np.tensordot(C_dLueH[p-1,r-1,:,:], Yu.conj().T[:,s-1], axes=([0], [0])) @ Yl.conj().T[:,t-1]
            +2*np.tensordot((C_QuLLH[:,:,r-1,t-1]+C_QuLLH[:,:,t-1,r-1]), Yd.conj().T[p-1,:], axes=([0], [0])) @ Yu.conj().T[:,s-1]
            -6*np.tensordot((C_QuLLH[:,:,r-1,t-1]+C_QuLLH[:,:,t-1,r-1]), Yu.conj().T[:,:], axes=([0,1], [1,0])) * Yd.conj().T[p-1,s-1]
            -6*g**2*C_duLDL[p-1,:,r-1,t-1]@Yu.conj().T[:,s-1]
            -2*np.tensordot(C_duLDL[p-1,:,r-1,:], Yu.conj().T[:,s-1], axes=([0], [0])) @ (Yl @ Yl.conj().T)[:,t-1]
            -2*np.tensordot(C_duLDL[p-1,:,t-1,:], Yu.conj().T[:,s-1], axes=([0], [0])) @ (Yl @ Yl.conj().T)[:,r-1]     
        )
        
    for p, r, s, t in indices:
        Beta[f"dLQLH2_{p}{r}{s}{t}"] = (
            1/3*(gp**2+9*g**2)*Yd.conj().T[p-1, s-1]*C_DLDH1[r-1, t-1]
            +Yd.conj().T[p-1, s-1]*((C_DLDH1 @ Yl @ Yl.conj().T)[r-1, t-1]
                                      +(2*C_DLDH1 @ Yl @ Yl.conj().T)[t-1, r-1])
            +1/6*gp**2*Yd.conj().T[p-1, s-1]*C_DLDH2[r-1, t-1]
            +1/2*Yd.conj().T[p-1, s-1]*(4*(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, t-1]
                                      -(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, r-1])
            -2*g*Yd.conj().T[p-1, s-1]*(5*C_LHW[r-1, t-1]+C_LHW[t-1, r-1])
            +4/3*gp*Yd.conj().T[p-1, s-1]*C_LHB[r-1, t-1]
            -Yd.conj().T[p-1,s-1]*np.tensordot(Yl[:, :], (3*C_eLLLHS-3*C_eLLLHA-2*C_eLLLHM)[:, :, r-1, t-1]+4*C_eLLLHM[:, :, t-1, r-1], axes=([0, 1], [1, 0]))
            +2*g**2*C_dLQLH1[p-1, r-1, s-1, t-1]
            -2/9*(10*gp**2+9*g**2-24*gs**2)*C_dLQLH1[p-1, t-1, s-1, r-1]
            -C_dLQLH1[p-1, r-1, :, t-1] @ (2*Yd @ Yd.conj().T-Yu @ Yu.conj().T)[:, s-1]
            +2*C_dLQLH1[p-1, :, s-1, t-1] @ (Yl @ Yl.conj().T)[:, r-1]
            -3*Yd.conj().T[p-1, s-1]*np.tensordot(Yd[:, :], C_dLQLH1[:, t-1, :, r-1], axes=([0, 1], [1, 0]))
            -1/36*(41*gp**2+207*g**2+96*gs**2-36*WH)*C_dLQLH2[p-1, r-1, s-1, t-1]
            -2/9*(10*gp**2-9*g**2-24*gs**2)*C_dLQLH2[p-1, t-1, s-1, r-1]
            -1/2*C_dLQLH2[p-1, r-1, :, t-1] @ (3*Yd @ Yd.conj().T-5*Yu @ Yu.conj().T)[:, s-1]
            +3*C_dLQLH2[:, r-1, s-1, t-1] @ (Yd.conj().T @ Yd)[p-1, :]
            +5/2*C_dLQLH2[p-1, r-1, s-1, :] @ (Yl @ Yl.conj().T)[:, t-1]
            +1/2*C_dLQLH2[p-1, :, s-1, t-1] @ (Yl @ Yl.conj().T)[:, r-1]
            +3*Yd.conj().T[p-1, s-1]*np.tensordot(Yd[:, :], C_dLQLH2[:, r-1, :, t-1]-C_dLQLH2[:, t-1, :, r-1], axes=([0, 1], [1, 0]))          
            +2*np.tensordot(C_dLueH[p-1,r-1,:, :], Yu.conj().T[:, s-1], axes=([0], [0]))@Yl.conj().T[:, t-1]            
            +6*Yd.conj().T[p-1, s-1]*np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, t-1, r-1], axes=([0, 1], [1, 0]))
            -2*np.tensordot(Yd.conj().T[p-1, :], C_QuLLH[:, :, t-1, r-1], axes=([0], [0])) @ Yu.conj().T[:, s-1]
            +1/3*(gp**2+9*g**2)*C_duLDL[p-1, :, r-1, t-1] @ Yu.conj().T[:, s-1]
            +2*np.tensordot(C_duLDL[p-1, :, r-1, :], Yu.conj().T[:, s-1], axes=([0], [0])) @ (Yl @ Yl.conj().T)[:, t-1]
        )
        
    for p, r, s, t in indices:
        Beta[f"dLueH_{p}{r}{s}{t}"] = (
            2*(Yd.conj().T @ Yu)[p-1, s-1]*(-3*C_LeDH+C_DLDH1 @ Yl +2*C_DLDH2 @ Yl)[r-1, t-1]
            -2*np.tensordot(Yu[:, s-1], C_dLQLH1[p-1, r-1, :, :],  axes=([0], [0])) @ Yl[:, t-1]
            +2*np.tensordot(Yu[:, s-1], C_dLQLH1[p-1, :, :, r-1],  axes=([0], [1])) @ Yl[:, t-1]
            +2*np.tensordot(Yu[:, s-1], C_dLQLH2[p-1, r-1, :, :],  axes=([0], [0])) @ Yl[:, t-1]
            +np.tensordot(Yu[:, s-1], C_dLQLH2[p-1, :, :, r-1],  axes=([0], [1])) @ Yl[:, t-1]
            -1/4*(23*gp**2+9*g**2-4*WH)*C_dLueH[p-1, r-1, s-1, t-1]
            +3*(Yd.conj().T @ Yd)[p-1, :] @ C_dLueH[:, r-1, s-1, t-1]
            +np.tensordot(C_dLueH[p-1, :, s-1, :], Yl[:, t-1],  axes=([0], [0])) @ Yl.conj().T[:, r-1]
            -3/2*C_dLueH[p-1, :, s-1, t-1] @ (Yl @ Yl.conj().T)[:, r-1]
            +3*C_dLueH[p-1, r-1, :, t-1] @ (Yu.conj().T @ Yu)[:, s-1]
            +2*C_dLueH[p-1, r-1, s-1, :] @ (Yl.conj().T @ Yl)[:, t-1]
            +np.tensordot(C_QuLLH[:, s-1, r-1, :]+2*C_QuLLH[:, s-1, :, r-1], Yd.conj().T[p-1, :],  axes=([0], [0])) @ Yl[:, t-1]           
            -1/3*(19*gp**2+9*g**2)*C_duLDL[p-1, s-1, r-1, :] @ Yl[:, t-1]
            -2*np.tensordot(C_duLDL[p-1, s-1, :, :], (Yl @ Yl.conj().T)[:, r-1],  axes=([0], [0])) @ Yl[:, t-1]
            +np.tensordot(C_duLDL[:, s-1, r-1, :], (Yd.conj().T @ Yd)[p-1, :],  axes=([0], [0])) @ Yl[:, t-1]
            +np.tensordot(C_duLDL[p-1, :, r-1, :], (Yu.conj().T @ Yu)[:, s-1],  axes=([0], [0])) @ Yl[:, t-1]   
           )
        
####################################################################################################

    # 3F2A: eQddH 
    indices = [
    (1,1,1,2), (1,1,1,3), (1,1,2,3), 
    (1,2,1,2), (1,2,1,3), (1,2,2,3), 
    (1,3,1,2), (1,3,1,3), (1,3,2,3), 
    (2,1,1,2), (2,1,1,3), (2,1,2,3), 
    (2,2,1,2), (2,2,1,3), (2,2,2,3), 
    (2,3,1,2), (2,3,1,3), (2,3,2,3), 
    (3,1,1,2), (3,1,1,3), (3,1,2,3), 
    (3,2,1,2), (3,2,1,3), (3,2,2,3), 
    (3,3,1,2), (3,3,1,3), (3,3,2,3), 
    ]
 
    for p, r, s, t in indices:
        Beta[f"eQddH_{p}{r}{s}{t}"] = (  
            1/2*np.tensordot(C_LdudH[:, s-1, :, t-1], Yl.conj().T[p-1, :],  axes=([0], [0])) @ Yu.conj().T[:, r-1]
            -1/2*np.tensordot(C_LdudH[:, t-1, :, s-1], Yl.conj().T[p-1, :],  axes=([0], [0])) @ Yu.conj().T[:, r-1]
            +1/24*(11*gp**2-27*g**2-48*gs**2+12*WH)*C_eQddH[p-1, r-1, s-1, t-1]
            -1/24*(11*gp**2-27*g**2-48*gs**2+12*WH)*C_eQddH[p-1, r-1, t-1, s-1]
            +np.tensordot(C_eQddH[p-1, :, :, s-1], Yd[:, t-1],  axes=([0], [0])) @ Yd.conj().T[:, r-1]
            -np.tensordot(C_eQddH[p-1, :, :, t-1], Yd[:, s-1],  axes=([0], [0])) @ Yd.conj().T[:, r-1]
            -1/4*C_eQddH[p-1, :, s-1, t-1] @ (3*Yu @ Yu.conj().T-5*Yd @ Yd.conj().T)[:, r-1]
            +1/4*C_eQddH[p-1, :, t-1, s-1] @ (3*Yu @ Yu.conj().T-5*Yd @ Yd.conj().T)[:, r-1]
            +3*C_eQddH[p-1, r-1, :, t-1] @ (Yd.conj().T @ Yd)[:, s-1]
            -3*C_eQddH[p-1, r-1, :, s-1] @ (Yd.conj().T @ Yd)[:, t-1]
            +C_eQddH[:, r-1, s-1, t-1] @ (Yl.conj().T @ Yl)[p-1, :]
            -C_eQddH[:, r-1, t-1, s-1] @ (Yl.conj().T @ Yl)[p-1, :]
            -1/2*np.tensordot(C_LdQQH[:, s-1, r-1, :]-2*C_LdQQH[:, s-1, :, r-1], Yl.conj().T[p-1, :],  axes=([0], [0])) @ Yd[:, t-1]
            +1/2*np.tensordot(C_LdQQH[:, t-1, r-1, :]-2*C_LdQQH[:, t-1, :, r-1], Yl.conj().T[p-1, :],  axes=([0], [0])) @ Yd[:, s-1]          
            -gp**2*C_eddDd[p-1, s-1, t-1, :] @ Yd.conj().T[:, r-1]
            +gp**2*C_eddDd[p-1, t-1, s-1, :] @ Yd.conj().T[:, r-1]
            +3/2*np.tensordot(C_eddDd[p-1, :, s-1, :], (Yd.conj().T @ Yd)[:, t-1],  axes=([1], [0])) @ Yd.conj().T[:, r-1]
            -3/2*np.tensordot(C_eddDd[p-1, :, t-1, :], (Yd.conj().T @ Yd)[:, s-1],  axes=([1], [0])) @ Yd.conj().T[:, r-1]
            -np.tensordot(C_LQdDd[:, r-1, s-1, :], (Yd.conj().T @ Yd)[:, t-1],  axes=([1], [0])) @ Yl.conj().T[p-1, :]
            +np.tensordot(C_LQdDd[:, r-1, t-1, :], (Yd.conj().T @ Yd)[:, s-1],  axes=([1], [0])) @ Yl.conj().T[p-1, :]
            +np.tensordot((np.tensordot(C_LQdDd[:, :, s-1, :], Yl.conj().T [p-1, :],  axes=([0], [0]))), Yd.conj().T [:, r-1],  axes=([1], [0])) @ Yd[:, t-1]
            -np.tensordot((np.tensordot(C_LQdDd[:, :, t-1, :], Yl.conj().T [p-1, :],  axes=([0], [0]))), Yd.conj().T [:, r-1],  axes=([1], [0])) @ Yd[:, s-1]
           )
        
####################################################################################################

    # 4F3S: eLLLHS eddDd 
    indices = [
    (1,1,1,1), (1,1,1,2), (1,1,1,3), (1,1,2,2), (1,1,2,3),
    (1,1,3,3), (1,2,2,2), (1,2,2,3), (1,2,3,3), (1,3,3,3),
    (2,1,1,1), (2,1,1,2), (2,1,1,3), (2,1,2,2), (2,1,2,3),
    (2,1,3,3), (2,2,2,2), (2,2,2,3), (2,2,3,3), (2,3,3,3),
    (3,1,1,1), (3,1,1,2), (3,1,1,3), (3,1,2,2), (3,1,2,3),
    (3,1,3,3), (3,2,2,2), (3,2,2,3), (3,2,3,3), (3,3,3,3),  
    ]
     
    for p, r, s, t in indices:
        Beta[f"eLLLHS_{p}{r}{s}{t}"] = (
             (gp**2-g**2)*(Yl.conj().T[p-1, r-1]*C_DLDH1[s-1, t-1]
                           +Yl.conj().T[p-1, s-1]*C_DLDH1[r-1, t-1]
                           +Yl.conj().T[p-1, t-1]*C_DLDH1[r-1, s-1])
            -1/2*((C_DLDH1 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1])
            +1/2*(gp**2-2*g**2)*(Yl.conj().T[p-1, r-1]*C_DLDH2[s-1, t-1]
                           +Yl.conj().T[p-1, s-1]*C_DLDH2[r-1, t-1]
                           +Yl.conj().T[p-1, t-1]*C_DLDH2[r-1, s-1])
            -1/4*((C_DLDH2 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1])
            -2*g*(Yl.conj().T[p-1, r-1]*(C_LHW[s-1, t-1]+C_LHW[t-1, s-1])
                  +Yl.conj().T[p-1, s-1]*(C_LHW[r-1, t-1]+C_LHW[t-1, r-1])
                  +Yl.conj().T[p-1, t-1]*(C_LHW[r-1, s-1]+C_LHW[s-1, r-1]))
            -1/4*(9*gp**2-9*g**2-4*WH)*C_eLLLHS[p-1, r-1, s-1, t-1]
            +3*(Yl.conj().T @ Yl)[p-1, :] @ C_eLLLHS[:, r-1, s-1, t-1]
            +np.tensordot(Yl[:, :], C_eLLLHS[:, :, s-1, t-1], axes=([0, 1], [0, 1]))*Yl.conj().T[p-1, r-1]
            +np.tensordot(Yl[:, :], C_eLLLHS[:, :, t-1, r-1], axes=([0, 1], [0, 1]))*Yl.conj().T[p-1, s-1]
            +np.tensordot(Yl[:, :], C_eLLLHS[:, :, r-1, s-1], axes=([0, 1], [0, 1]))*Yl.conj().T[p-1, t-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHS[p-1, :, s-1, t-1], axes=([0], [0])) @ Yl.conj().T[:, r-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHS[p-1, r-1, :, t-1], axes=([0], [0])) @ Yl.conj().T[:, s-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHS[p-1, r-1, s-1, :], axes=([0], [0])) @ Yl.conj().T[:, t-1]
            +2/3*((Yl@Yl.conj().T)[:,r-1]@(C_eLLLHM[p-1,:,s-1,t-1]+C_eLLLHM[p-1,:,t-1,s-1])
                +(Yl@Yl.conj().T)[:,s-1]@(2*C_eLLLHM[p-1,:,t-1,r-1]+C_eLLLHM[p-1,r-1,:,t-1])
                +(Yl@Yl.conj().T)[:,t-1]@(2*C_eLLLHM[p-1,:,s-1,r-1]+C_eLLLHM[p-1,r-1,:,s-1]))
            +1/3*np.tensordot(Yl[:,:], (C_eLLLHM[:,:,s-1,t-1]+C_eLLLHM[:,:,t-1,s-1])*Yl.conj().T[p-1, r-1]
            +(C_eLLLHM[:,:,r-1,t-1]+C_eLLLHM[:,:,t-1,r-1])*Yl.conj().T[p-1, s-1]
            +(C_eLLLHM[:,:,r-1,s-1]+C_eLLLHM[:,:,s-1,r-1])*Yl.conj().T[p-1, t-1], axes=([0,1], [1,0]))
            +1/2*np.tensordot(Yd[:,:], (C_dLQLH1[:,s-1,:,t-1]+C_dLQLH1[:,t-1,:,s-1])*Yl.conj().T[p-1, r-1]
            +(C_dLQLH1[:,r-1,:,t-1]+C_dLQLH1[:,t-1,:,r-1])*Yl.conj().T[p-1, s-1]
            +(C_dLQLH1[:,s-1,:,r-1]+C_dLQLH1[:,r-1,:,s-1])*Yl.conj().T[p-1, t-1], axes=([0,1], [1,0]))
            -np.tensordot(Yu.conj().T[:,:], (C_QuLLH[:,:,s-1,t-1]+C_QuLLH[:,:,t-1,s-1])*Yl.conj().T[p-1, r-1]
            +(C_QuLLH[:,:,r-1,t-1]+C_QuLLH[:,:,t-1,r-1])*Yl.conj().T[p-1, s-1]
            +(C_QuLLH[:,:,s-1,r-1]+C_QuLLH[:,:,r-1,s-1])*Yl.conj().T[p-1, t-1], axes=([0,1], [1,0]))         
        )

    for p, r, s, t in indices:
        Beta[f"eddDd_{p}{r}{s}{t}"] = (
            - 2 / 3 * (gp**2 - 6 * gs**2) * C_eddDd[p-1, r-1, s-1, t-1]  
            + (C_eddDd[:, r-1, s-1, t-1] @ (Yl.conj().T @ Yl)[p-1, :])
            + (C_eddDd[p-1, :, s-1, t-1] @ (Yd.conj().T @ Yd)[:, r-1])
            + (C_eddDd[p-1, r-1, :, t-1] @ (Yd.conj().T @ Yd)[:, s-1])
            + (C_eddDd[p-1, r-1, s-1, :] @ (Yd.conj().T @ Yd)[:, t-1])   
            -2/3*Yl.conj().T[p-1,:] @ (np.tensordot(C_LQdDd[:,:,s-1,t-1],Yd[:,r-1],axes=([1], [0]))
                                       +np.tensordot(C_LQdDd[:,:,r-1,t-1],Yd[:,s-1],axes=([1], [0]))
                                       +np.tensordot(C_LQdDd[:,:,r-1,s-1],Yd[:,t-1],axes=([1], [0])))
        )

####################################################################################################

    # 4F3A: eLLLHA 
    indices = [
    (1,1,2,3), (2,1,2,3), (3,1,2,3)  
    ]
    
    for p, r, s, t in indices:
        Beta[f"eLLLHA_{p}{r}{s}{t}"] = (
            1/6*((C_DLDH1 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  -(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  -(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1]
                  -(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1])
            -5/12*((C_DLDH2 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  -(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  -(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1]
                  -(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1])
            -4*g*(Yl.conj().T[p-1, r-1]*C_LHW[s-1, t-1]
                  -Yl.conj().T[p-1, r-1]*C_LHW[t-1, s-1]
                  -Yl.conj().T[p-1, s-1]*C_LHW[r-1, t-1]
                  +Yl.conj().T[p-1, s-1]*C_LHW[t-1, r-1]
                  +Yl.conj().T[p-1, t-1]*C_LHW[r-1, s-1]
                  -Yl.conj().T[p-1, t-1]*C_LHW[s-1, r-1])
            +12*gp*(Yl.conj().T[p-1, r-1]*C_LHB[s-1, t-1]
                  +Yl.conj().T[p-1, s-1]*C_LHB[t-1, r-1]
                  +Yl.conj().T[p-1, t-1]*C_LHB[r-1, s-1])
            -1/4*(9*gp**2+39*g**2-4*WH)*C_eLLLHA[p-1, r-1, s-1, t-1]
            +np.tensordot(Yl[:, :], C_eLLLHA[:, :, s-1, t-1], axes=([0, 1], [1, 0]))*Yl.conj().T[p-1, r-1]
            -np.tensordot(Yl[:, :], C_eLLLHA[:, :, r-1, t-1], axes=([0, 1], [1, 0]))*Yl.conj().T[p-1, s-1]
            +np.tensordot(Yl[:, :], C_eLLLHA[:, :, r-1, s-1], axes=([0, 1], [1, 0]))*Yl.conj().T[p-1, t-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHA[p-1, :, s-1, t-1], axes=([0], [0])) @ Yl.conj().T[:, r-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHA[p-1, r-1, :, t-1], axes=([0], [0])) @ Yl.conj().T[:, s-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHA[p-1, r-1, s-1, :], axes=([0], [0])) @ Yl.conj().T[:, t-1]
            +3*(Yl.conj().T @ Yl)[p-1, :] @ C_eLLLHA[:, r-1, s-1, t-1]
            -2*(Yl @ Yl.conj().T)[:, r-1] @ C_eLLLHM[p-1, :, s-1, t-1]
            +2*(Yl @ Yl.conj().T)[:, r-1] @ C_eLLLHM[p-1, :, t-1, s-1]
            +2*(Yl @ Yl.conj().T)[:, s-1] @ C_eLLLHM[p-1, r-1, :, t-1]
            -2*(Yl @ Yl.conj().T)[:, t-1] @ C_eLLLHM[p-1, r-1, :, s-1]
            +np.tensordot(Yl[:,:], Yl.conj().T[p-1,r-1]*(C_eLLLHM[:,:,s-1,t-1]-C_eLLLHM[:,:,t-1,s-1]),axes=([0, 1], [1, 0]))
            -np.tensordot(Yl[:,:], Yl.conj().T[p-1,s-1]*(C_eLLLHM[:,:,r-1,t-1]-C_eLLLHM[:,:,t-1,r-1]),axes=([0, 1], [1, 0]))
            +np.tensordot(Yl[:,:], Yl.conj().T[p-1,t-1]*(C_eLLLHM[:,:,r-1,s-1]-C_eLLLHM[:,:,s-1,r-1]),axes=([0, 1], [1, 0]))
            +1/2*np.tensordot(Yd[:,:], Yl.conj().T[p-1,r-1]*(C_dLQLH1[:,s-1,:,t-1]-C_dLQLH1[:,t-1,:,s-1]),axes=([0, 1], [1, 0]))
            -1/2*np.tensordot(Yd[:,:], Yl.conj().T[p-1,s-1]*(C_dLQLH1[:,r-1,:,t-1]-C_dLQLH1[:,t-1,:,r-1]),axes=([0, 1], [1, 0]))
            +1/2*np.tensordot(Yd[:,:], Yl.conj().T[p-1,t-1]*(C_dLQLH1[:,r-1,:,s-1]-C_dLQLH1[:,s-1,:,r-1]),axes=([0, 1], [1, 0]))
            +np.tensordot(Yd[:,:], Yl.conj().T[p-1,r-1]*(C_dLQLH2[:,s-1,:,t-1]-C_dLQLH2[:,t-1,:,s-1]),axes=([0, 1], [1, 0]))
            -np.tensordot(Yd[:,:], Yl.conj().T[p-1,s-1]*(C_dLQLH2[:,r-1,:,t-1]-C_dLQLH2[:,t-1,:,r-1]),axes=([0, 1], [1, 0]))
            +np.tensordot(Yd[:,:], Yl.conj().T[p-1,t-1]*(C_dLQLH2[:,r-1,:,s-1]-C_dLQLH2[:,s-1,:,r-1]),axes=([0, 1], [1, 0]))     
            -np.tensordot(Yu.conj().T[:,:], Yl.conj().T[p-1,r-1]*(C_QuLLH[:,:,s-1,t-1]-C_QuLLH[:,:,t-1,s-1]),axes=([0, 1], [1, 0]))
            +np.tensordot(Yu.conj().T[:,:], Yl.conj().T[p-1,s-1]*C_QuLLH[:,:,r-1,t-1],axes=([0, 1], [1, 0]))
            -np.tensordot(Yu.conj().T[:,:], Yl.conj().T[p-1,s-1]*C_QuLLH[:,:,t-1,r-1],axes=([0, 1], [1, 0]))
            -np.tensordot(Yu.conj().T[:,:], Yl.conj().T[p-1,t-1]*C_QuLLH[:,:,r-1,s-1],axes=([0, 1], [1, 0]))
            +np.tensordot(Yu.conj().T[:,:], Yl.conj().T[p-1,t-1]*C_QuLLH[:,:,s-1,r-1],axes=([0, 1], [1, 0]))                  
        )

####################################################################################################

    # 4F3M1: eLLLHM 
    indices = [
    (1,1,1,2), (1,1,2,2), (1,1,3,2), (1,1,1,3), (1,1,2,3), (1,1,3,3), (1,2,2,3), (1,2,3,3),
    (2,1,1,2), (2,1,2,2), (2,1,3,2), (2,1,1,3), (2,1,2,3), (2,1,3,3), (2,2,2,3), (2,2,3,3),
    (3,1,1,2), (3,1,2,2), (3,1,3,2), (3,1,1,3), (3,1,2,3), (3,1,3,3), (3,2,2,3), (3,2,3,3)  
    ]   

    for p, r, s, t in indices:
        Beta[f"eLLLHM_{p}{r}{s}{t}"] = (
            -3/2*(gp**2+g**2)*(Yl.conj().T[p-1, r-1]*C_DLDH1[s-1, t-1]-Yl.conj().T[p-1, t-1]*C_DLDH1[r-1, s-1])
            -1/6*(4*(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  -(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  -5*(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  +5*(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1]
                  -4*(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1])
            -3/4*gp**2*((Yl.conj().T)[p-1, r-1]*C_DLDH2[s-1, t-1]
                        -(Yl.conj().T)[p-1, t-1]*C_DLDH2[r-1, s-1])
            -1/12*(7*(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  +5*(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  -2*(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  +2*(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  -5*(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1]
                  -7*(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1])
            +g*(5*Yl.conj().T[p-1, r-1]*C_LHW[s-1, t-1]
                  +Yl.conj().T[p-1, r-1]*C_LHW[t-1, s-1]
                  +4*Yl.conj().T[p-1, s-1]*C_LHW[r-1, t-1]
                  -4*Yl.conj().T[p-1, s-1]*C_LHW[t-1, r-1]
                  -Yl.conj().T[p-1, t-1]*C_LHW[r-1, s-1]
                  -5*Yl.conj().T[p-1, t-1]*C_LHW[s-1, r-1])
            -6*gp*(Yl.conj().T[p-1, r-1]*C_LHB[s-1, t-1]
                  +2*Yl.conj().T[p-1, s-1]*C_LHB[r-1, t-1]
                  +Yl.conj().T[p-1, t-1]*C_LHB[r-1, s-1])
            +3*(C_eLLLHS[p-1,:,s-1,t-1]@(Yl @ Yl.conj().T)[:, r-1]
               -C_eLLLHS[p-1,r-1,s-1,:]@(Yl @ Yl.conj().T)[:, t-1])
            +3/2*np.tensordot(Yl[:, :], C_eLLLHS[:, :, s-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            -3/2*np.tensordot(Yl[:, :], C_eLLLHS[:, :, r-1, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]            
            -C_eLLLHA[p-1, :, s-1, t-1] @ (Yl @ Yl.conj().T)[:, r-1]
            +2*C_eLLLHA[p-1, r-1, :, t-1] @ (Yl @ Yl.conj().T)[:, s-1]
            -C_eLLLHA[p-1, r-1, s-1, :] @ (Yl @ Yl.conj().T)[:, t-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHA[:, :, s-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            +np.tensordot(Yl[:, :], C_eLLLHA[:, :, r-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHA[:, :, r-1, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]
            +3*(Yl.conj().T @ Yl)[p-1, :] @ C_eLLLHM[:, r-1, s-1, t-1]
            -1/4*(9*gp**2+15*g**2-4*WH)*C_eLLLHM[p-1, r-1, s-1, t-1]
            +np.tensordot(Yl[:, :], C_eLLLHM[:, :, s-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            +np.tensordot(Yl[:, :], C_eLLLHM[:, :, r-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            -np.tensordot(Yl[:, :], C_eLLLHM[:, :, t-1, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            -np.tensordot(Yl[:, :], C_eLLLHM[:, :, s-1, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHM[p-1, :, s-1, t-1], axes=([0], [0])) @ Yl.conj().T[:, r-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHM[p-1, r-1, :, t-1], axes=([0], [0])) @ Yl.conj().T[:, s-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHM[p-1, r-1, s-1, :], axes=([0], [0])) @ Yl.conj().T[:, t-1]
            +np.tensordot(Yd[:, :], C_dLQLH1[:, s-1, :, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            +1/2*np.tensordot(Yd[:, :], C_dLQLH1[:, t-1, :, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            +1/2*np.tensordot(Yd[:, :], C_dLQLH1[:, r-1, :, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            -1/2*np.tensordot(Yd[:, :], C_dLQLH1[:, t-1, :, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            -1/2*np.tensordot(Yd[:, :], C_dLQLH1[:, r-1, :, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]
            -np.tensordot(Yd[:, :], C_dLQLH1[:, s-1, :, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]            
            +1/2*np.tensordot(Yd[:, :], C_dLQLH2[:, s-1, :, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            -1/2*np.tensordot(Yd[:, :], C_dLQLH2[:, t-1, :, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            +np.tensordot(Yd[:, :], C_dLQLH2[:, r-1, :, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            -1/2*np.tensordot(Yd[:, :], C_dLQLH2[:, t-1, :, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            +1/2*np.tensordot(Yd[:, :], C_dLQLH2[:, r-1, :, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]
            -1/2*np.tensordot(Yd[:, :], C_dLQLH2[:, s-1, :, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]           
            -2*np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, s-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            -np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, t-1, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            -np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, r-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            +np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, t-1, r-1], axes=([0,1], [0,1])) * Yl.conj().T[p-1, s-1]
            +np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, r-1, s-1], axes=([0,1], [0,1])) * Yl.conj().T[p-1, t-1]
            +2*np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, s-1, r-1], axes=([0,1], [0,1])) * Yl.conj().T[p-1, t-1]
        )

####################################################################################################

    #4F3M2: LdddHM
    indices = [
       (1,1,1,2), (1,2,1,2), (1,3,1,2), (1,1,1,3), (1,2,1,3), (1,3,1,3), (1,2,2,3), (1,3,2,3),
       (2,1,1,2), (2,2,1,2), (2,3,1,2), (2,1,1,3), (2,2,1,3), (2,3,1,3), (2,2,2,3), (2,3,2,3),
       (3,1,1,2), (3,2,1,2), (3,3,1,2), (3,1,1,3), (3,2,1,3), (3,3,1,3), (3,2,2,3), (3,3,2,3) 
        ]
    
    for p, r, s, t in indices:
        Beta[f"LdddHM_{p}{r}{s}{t}"] = (
            1/6*(Yu.conj().T @ Yd)[:, r-1] @ C_LdudH[p-1,t-1,:,s-1]
            -1/6*(Yu.conj().T @ Yd)[:, r-1] @ C_LdudH[p-1,s-1,:,t-1]
            -1/6*(Yu.conj().T @ Yd)[:, s-1] @ C_LdudH[p-1,t-1,:,r-1]
            -1/3*(Yu.conj().T @ Yd)[:, s-1] @ C_LdudH[p-1,r-1,:,t-1]
            +1/6*(Yu.conj().T @ Yd)[:, t-1] @ C_LdudH[p-1,s-1,:,r-1]
            +1/3*(Yu.conj().T @ Yd)[:, t-1] @ C_LdudH[p-1,r-1,:,s-1]            
            -1/12*(13*gp**2+27*g**2+48*gs**2-12*WH)*C_LdddHM[p-1,r-1,s-1,t-1]
            +2*C_LdddHM[p-1,:,s-1,t-1] @ (Yd.conj().T @ Yd)[:, r-1]
            +2*C_LdddHM[p-1,r-1,:,t-1] @ (Yd.conj().T @ Yd)[:, s-1]
            +2*C_LdddHM[p-1,r-1,s-1,:] @ (Yd.conj().T @ Yd)[:, t-1]
            +5/2*C_LdddHM[:,r-1,s-1,t-1] @ (Yl @ Yl.conj().T)[p-1, :]
            +1/2*np.tensordot(Yl[p-1, :], C_eddDd[:, :, r-1, s-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, t-1]
            -1/2*np.tensordot(Yl[p-1, :], C_eddDd[:, :, r-1, t-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, s-1]           
            -1/9*(5*gp**2-12*gs**2)*(C_LQdDd[p-1, :, t-1, r-1] @ Yd[:, s-1]-C_LQdDd[p-1, :, s-1, r-1] @ Yd[:, t-1])            
            +1/2*np.tensordot(C_LQdDd[p-1, :, :, t-1], Yd[:, s-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, r-1]
            +1/2*np.tensordot(C_LQdDd[p-1, :, :, t-1], Yd[:, r-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, s-1]            
            -1/2*np.tensordot(C_LQdDd[p-1, :, :, s-1], Yd[:, t-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, r-1]
            -1/2*np.tensordot(C_LQdDd[p-1, :, :, s-1], Yd[:, r-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, t-1]
            +1/3*np.tensordot((Yl @ Yl.conj().T)[p-1, :], C_LQdDd[:, :, r-1, s-1], axes=([0], [0])) @ Yd[:, t-1]
            -1/3*np.tensordot((Yl @ Yl.conj().T)[p-1, :], C_LQdDd[:, :, r-1, t-1], axes=([0], [0])) @ Yd[:, s-1]
        )    
####################################################################################################    
    return Beta


#################################################################################################### 
# define the Beta function with two-loop SM RGEs
def beta_function_twoloopSM(C, scale):

# Yukawa couplings matrix for up, down quarks and charged leptons in SM 
    yukawa_names = ["Yd", "Yu", "Yl"]

    for name in yukawa_names:
        globals()[name] = np.array([[C.get(f"{name}{i+1}{j+1}", 0) for j in range(3)] for i in range(3)])

    gp=C.get("gp", 0)  #gauge coupling of U(1)_Y in SM 
    g=C.get("g", 0)    #gauge coupling of SU(2) in SM
    gs=C.get("gs", 0)  #gauge coupling of SU(3) in SM
    Lambda=C.get("Lambda", 0)  #quartic couplings of Higgs potential
    muh=C.get("muh", 0)  #square couplings of Higgs potential

#################################################################################################
#                                                                                               #
#                   Represent the dim-5 and dim-7 WCs in matrix form                            #
#                                                                                               #
#################################################################################################

    #2FS
    matrix_names = ["LH5", "LH", "DLDH1", "DLDH2"]

    indices = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]

    for name in matrix_names:
        globals()[f"C_{name}"] = np.zeros((3, 3), dtype=np.complex128)  

    
        for i, j in indices:
            key = f"{name}_{i}{j}" 
            globals()[f"C_{name}"][i-1, j-1] = C.get(key, 0 + 0j)
            globals()[f"C_{name}"][j-1, i-1] = C.get(key, 0 + 0j) 
     
#################################################################################################

    #2F
    matrix_names = ["LeDH", "LHW"]

    for name in matrix_names:
        globals()[f"C_{name}"] = np.zeros((3, 3), dtype=np.complex128) 
        for i in range(3):
            for j in range(3):
                
                key = f"{name}_{i+1}{j+1}"  
                globals()[f"C_{name}"][i, j] = C.get(key, 0 + 0j)  

#################################################################################################

    #2FA
    matrix_names = ["LHB"]
    indices = [(1,2), (1,3), (2,3)]
    
    for name in matrix_names:
        C_matrix = np.zeros((3, 3), dtype=np.complex128)  # Initialize with zeros
    
        for i, j in indices:
            if i != j:  
               key = f"{name}_{i}{j}"
               value = C.get(key, 0 + 0j)
               C_matrix[i-1, j-1] = value
               C_matrix[j-1, i-1] = -value  
    
    
        for i in range(3):
            C_matrix[i, i] = 0

        globals()[f"C_{name}"] = C_matrix
    
#################################################################################################
   
    #4F3S
    matrix_names = ["eLLLHS", "eddDd"]

    indices = [
    (1,1,1,1), (1,1,1,2), (1,1,1,3), (1,1,2,2), (1,1,2,3),
    (1,1,3,3), (1,2,2,2), (1,2,2,3), (1,2,3,3), (1,3,3,3),
    (2,1,1,1), (2,1,1,2), (2,1,1,3), (2,1,2,2), (2,1,2,3),
    (2,1,3,3), (2,2,2,2), (2,2,2,3), (2,2,3,3), (2,3,3,3),
    (3,1,1,1), (3,1,1,2), (3,1,1,3), (3,1,2,2), (3,1,2,3),
    (3,1,3,3), (3,2,2,2), (3,2,2,3), (3,2,3,3), (3,3,3,3),  
    ]

    for name in matrix_names:
        C_matrix = np.zeros((3, 3, 3, 3), dtype=np.complex128)  
        for i, j, k, l in indices:
            key = f"{name}_{i}{j}{k}{l}"
            value = C.get(key, 0 + 0j)          
            for perm in set(permutations([j, k, l])):  
                C_matrix[i-1, perm[0]-1, perm[1]-1, perm[2]-1] = value

        globals()[f"C_{name}"] = C_matrix
     
#################################################################################################

    #4F3A
    matrix_names = ["eLLLHA"]
    indices = [(1,1,2,3), (2,1,2,3), (3,1,2,3)] 

    for name in matrix_names:
        C_matrix = np.zeros((3, 3, 3, 3), dtype=np.complex128)  
    
        for i, j, k, l in indices:
            key = f"{name}_{i}{j}{k}{l}"
            value = C.get(key, 0 + 0j)
        
            if len({j, k, l}) == 3:  
                original = [j, k, l]
                for perm in set(permutations(original)):
                    sign = Permutation([original.index(p) for p in perm]).signature()
                    C_matrix[i-1, perm[0]-1, perm[1]-1, perm[2]-1] = sign * value

        globals()[f"C_{name}"] = C_matrix

#################################################################################################
    
    #4F     
    matrix_names = ["C_dLQLH1", "C_dLQLH2", "C_dLueH", "C_QuLLH", "C_LdudH", "C_LdQQH"]


    for name in matrix_names:
        globals()[name] = np.zeros((3, 3, 3, 3), dtype=np.complex128)

        for i in range(3):
           for j in range(3):
               for k in range(3):
                    for l in range(3):
                        key = f"{name[2:]}_{i+1}{j+1}{k+1}{l+1}"  
                        globals()[name][i, j, k, l] = C.get(key, 0 + 0j)  

#################################################################################################

    #4F3M1
    matrix_names = ["eLLLHM"]

    indices = [
    (1,1,1,2), (1,1,2,2), (1,1,3,2), (1,1,1,3), (1,1,2,3), (1,1,3,3), (1,2,2,3), (1,2,3,3), 
    (2,1,1,2), (2,1,2,2), (2,1,3,2), (2,1,1,3), (2,1,2,3), (2,1,3,3), (2,2,2,3), (2,2,3,3), 
    (3,1,1,2), (3,1,2,2), (3,1,3,2), (3,1,1,3), (3,1,2,3), (3,1,3,3), (3,2,2,3), (3,2,3,3)  
    ]

    for name in matrix_names:
        C_matrix = np.zeros((3, 3, 3, 3), dtype=np.complex128)  # Initialize with zeros
    
        for i, j, k, l in indices:  
               key = f"{name}_{i}{j}{k}{l}"
               value = C.get(key, 0 + 0j)
               C_matrix[i-1, j-1,k-1,l-1] = value
               C_matrix[i-1, l-1,k-1,j-1] = -value  

        for i in range(3):
            C_matrix[i, 1, 0, 2]=C_matrix[i, 0, 1, 2]-C_matrix[i, 0, 2, 1]
            C_matrix[i, 2, 0, 1]=-C_matrix[i, 1, 0, 2]
    
    
        for i in range(3):
            for j in range(3):
                for k in range(3):
                   C_matrix[i, j, k ,j] = 0

        globals()[f"C_{name}"] = C_matrix
  
#################################################################################################
    
    #4F3M2
    matrix_names = ["LdddHM"]

    indices = [
       (1,1,1,2), (1,2,1,2), (1,3,1,2), (1,1,1,3), (1,2,1,3), (1,3,1,3), (1,2,2,3), (1,3,2,3),
       (2,1,1,2), (2,2,1,2), (2,3,1,2), (2,1,1,3), (2,2,1,3), (2,3,1,3), (2,2,2,3), (2,3,2,3),
       (3,1,1,2), (3,2,1,2), (3,3,1,2), (3,1,1,3), (3,2,1,3), (3,3,1,3), (3,2,2,3), (3,3,2,3) 
        ]

    for name in matrix_names:
        C_matrix = np.zeros((3, 3, 3, 3), dtype=np.complex128)  # Initialize with zeros
    
        for i, j, k, l in indices:  
               key = f"{name}_{i}{j}{k}{l}"
               value = C.get(key, 0 + 0j)
               C_matrix[i-1, j-1,k-1,l-1] = value
               C_matrix[i-1, j-1,l-1,k-1] = -value  

        for i in range(3):
            C_matrix[i, 0, 1, 2]=C_matrix[i, 1, 0, 2]-C_matrix[i, 2, 0, 1]
            C_matrix[i, 0, 2, 1]=-C_matrix[i, 0, 1, 2]
    
    
        for i in range(3):
            for j in range(3):
                for k in range(3):
                   C_matrix[i, j, k ,k] = 0

        globals()[f"C_{name}"] = C_matrix
    
#################################################################################################   

    #4F2A
    matrix_names = ["eQddH"]

    indices = [
    (1,1,1,2), (1,1,1,3), (1,1,2,3),
    (1,2,1,2), (1,2,1,3), (1,2,2,3),
    (1,3,1,2), (1,3,1,3), (1,3,2,3),
    (2,1,1,2), (2,1,1,3), (2,1,2,3),
    (2,2,1,2), (2,2,1,3), (2,2,2,3),
    (2,3,1,2), (2,3,1,3), (2,3,2,3),
    (3,1,1,2), (3,1,1,3), (3,1,2,3),
    (3,2,1,2), (3,2,1,3), (3,2,2,3),
    (3,3,1,2), (3,3,1,3), (3,3,2,3)
    ]

    for name in matrix_names:
        globals()[f"C_{name}"] = np.zeros((3, 3, 3, 3), dtype=np.complex128)

        for i, j, k, l in indices:
            key = f"{name}_{i}{j}{k}{l}"
            value = C.get(key, 0 + 0j)

        
            if k != l:
                globals()[f"C_{name}"][i-1, j-1, k-1, l-1] = value
                globals()[f"C_{name}"][i-1, j-1, l-1, k-1] = -value
            else:
                globals()[f"C_{name}"][i-1, j-1, k-1, l-1] = 0 

#################################################################################################    
       
    #4F2S     
    matrix_names = ["duLDL", "LQdDd"]

    indices = [
    (1,1,1,1), (1,1,1,2), (1,1,1,3), (1,1,2,2), (1,1,2,3), (1,1,3,3),
    (1,2,1,1), (1,2,1,2), (1,2,1,3), (1,2,2,2), (1,2,2,3), (1,2,3,3),
    (1,3,1,1), (1,3,1,2), (1,3,1,3), (1,3,2,2), (1,3,2,3), (1,3,3,3),
    (2,1,1,1), (2,1,1,2), (2,1,1,3), (2,1,2,2), (2,1,2,3), (2,1,3,3),
    (2,2,1,1), (2,2,1,2), (2,2,1,3), (2,2,2,2), (2,2,2,3), (2,2,3,3),
    (2,3,1,1), (2,3,1,2), (2,3,1,3), (2,3,2,2), (2,3,2,3), (2,3,3,3),
    (3,1,1,1), (3,1,1,2), (3,1,1,3), (3,1,2,2), (3,1,2,3), (3,1,3,3),
    (3,2,1,1), (3,2,1,2), (3,2,1,3), (3,2,2,2), (3,2,2,3), (3,2,3,3),
    (3,3,1,1), (3,3,1,2), (3,3,1,3), (3,3,2,2), (3,3,2,3), (3,3,3,3),
    ]

    for name in matrix_names:
        globals()[f"C_{name}"] = np.zeros((3, 3, 3, 3), dtype=np.complex128)

        for i, j, k, l in indices:
            key = f"{name}_{i}{j}{k}{l}"
            value = C.get(key, 0 + 0j)

        
            globals()[f"C_{name}"][i-1, j-1, k-1, l-1] = value
            globals()[f"C_{name}"][i-1, j-1, l-1, k-1] = value  

#################################################################################################
    
    # conventional quantities in dim-7 SMEFT RGEs

    WH = np.trace(3*Yu.conj().T @ Yu + 3*Yd.conj().T @ Yd + Yl.conj().T @ Yl)
    TH=(3*np.trace(Yd @ Yd.conj().T @ Yd @ Yd.conj().T) 
           +3*np.trace(Yu @ Yu.conj().T @ Yu @ Yu.conj().T) 
           +np.trace(Yl @ Yl.conj().T @ Yl @ Yl.conj().T)) 
    loop=1/(16*pi**2)
    gmuhT=np.trace(15*Yl.conj().T @ Yl+17*Yu.conj().T @ Yu + 5*Yd.conj().T @ Yd)
    
#################################################################################################

    Beta= {}  

#################################################################################################
#                                                                                               #
#                              beta function for SM parameters                                  #
#                                                                                               #
#################################################################################################

    Beta["gp"] = gp**3*(41/6+loop*(199/18*gp**2+9/2*g**2+44/3*gs**2-1/6*gmuhT) )
    Beta["g"] = -g**3*(19/6-loop*(3/2*gp**2+35/6*g**2+12*gs**2-1/2*WH))
    Beta["gs"] = -gs**3*(7-loop*(11/6*gp**2+9/2*g**2-26*gs**2-2*np.trace(Yu.conj().T @ Yu + Yd.conj().T @ Yd)))
    Beta["Lambda"] = 3/8*gp**4 + 9/8*g**4+ 3/4*g**2*gp**2- 2*TH + (-3*gp**2-9*g**2+24*Lambda+4*WH)*Lambda\
                     +loop*(-379/48*gp**6+305/16*g**6-289/48*gp**2*g**4-559/48*gp**4*g**2
                            -1/4*gp**4*np.trace(25*Yl @ Yl.conj().T+19*Yu @ Yu.conj().T-5*Yd @ Yd.conj().T)-3/4*g**4*WH
                            +1/2*gp**2*g**2*np.trace(11*Yl @ Yl.conj().T+21*Yu @ Yu.conj().T+9*Yd @ Yd.conj().T)
                            -4/3*gp**2*np.trace(3*Yl @ Yl.conj().T@Yl @ Yl.conj().T
                                                +2*Yu @ Yu.conj().T@Yu @ Yu.conj().T  
                                                -Yd @ Yd.conj().T@Yd @ Yd.conj().T 
                                                )
                            -32*gs**2*np.trace(Yu @ Yu.conj().T@Yu @ Yu.conj().T+Yd @ Yd.conj().T@Yd @ Yd.conj().T)
                            +Lambda*(629/24*gp**4-73/8*g**4+39/4*gp**2*g**2
                                     +5/6*gp**2*gmuhT+15/2*g**2*WH+80*gs**2*np.trace(Yu.conj().T @ Yu + Yd.conj().T @ Yd)-TH
                                     -42*np.trace(Yu @ Yu.conj().T@Yd @ Yd.conj().T)                         
                                    )
                            +2*Lambda**2*(18*gp**2+54*g**2-156*Lambda-24*WH)
                            +2*np.trace(5*Yl @ Yl.conj().T@Yl @ Yl.conj().T@Yl @ Yl.conj().T
                                        +15*Yu @ Yu.conj().T@Yu @ Yu.conj().T@Yu @ Yu.conj().T
                                        +15*Yd @ Yd.conj().T@Yd @ Yd.conj().T@Yd @ Yd.conj().T
                                        -3*Yd @ Yd.conj().T@Yd @ Yd.conj().T@Yu @ Yu.conj().T
                                        -3*Yd @ Yd.conj().T@Yu @ Yu.conj().T@Yu @ Yu.conj().T
                                         )
                             )
    Beta["muh"] = muh*(- 3/2*gp**2- 9/2*g**2  + 12*Lambda + 2*WH
                        + loop*(557/48*gp**4-145/16*g**4+15/8*gp**2*g**2+5/12*gp**2*gmuhT 
                                 +15/4*g**2*WH+40*gs**2*np.trace(Yu.conj().T @ Yu + Yd.conj().T @ Yd)-9/2*TH
                                 -21*np.trace(Yu @ Yu.conj().T @ Yd @ Yd.conj().T) 
                                 +2*Lambda*(12*gp**2+36*g**2-30*Lambda-12*WH)
                                )
                        )
 
    indices = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]  
   
    for p, r in indices:
        Beta[f"Yu{p}{r}"] = (-17/12*gp**2 -9/4*g**2- 8*gs**2+WH
                             +loop*(1187/216*gp**4-23/4*g**4-108*gs**4-3/4*gp**2*g**2+19/9*gp**2*gs**2+9*g**2*gs**2
                                    +5/24*gp**2*gmuhT+15/8*g**2*WH+20*gs**2*np.trace(Yu.conj().T @ Yu + Yd.conj().T @ Yd)+6*Lambda**2-9/4*TH
                                    +3/2*np.trace(Yu @ Yu.conj().T @ Yd @ Yd.conj().T)))*Yu[p-1,r-1]\
                            +((3/2*(Yu @ Yu.conj().T -Yd @ Yd.conj().T)
                               +loop*((223/48*gp**2+135/16*g**2+16*gs**2-12*Lambda-9/4*WH)* Yu @ Yu.conj().T
                                      +(-43/48*gp**2+9/16*g**2-16*gs**2+5/4*WH)* Yd @ Yd.conj().T
                                      +3/2*Yu @ Yu.conj().T @ Yu @ Yu.conj().T
                                      +11/4*Yd @ Yd.conj().T @ Yd @ Yd.conj().T
                                      -Yd @ Yd.conj().T @Yu @ Yu.conj().T
                                      -1/4*Yu @ Yu.conj().T @ Yd @ Yd.conj().T
                                     )
                               )@Yu
                              )[p-1,r-1]
                                  
    for p, r in indices:
        Beta[f"Yd{p}{r}"] = (-5/12*gp**2 -9/4*g**2- 8*gs**2+WH
                             +loop*(-127/216*gp**4-23/4*g**4-108*gs**4-9/4*gp**2*g**2+31/9*gp**2*gs**2+9*g**2*gs**2
                                    +5/24*gp**2*gmuhT+15/8*g**2*WH+20*gs**2*np.trace(Yu.conj().T @ Yu + Yd.conj().T @ Yd)+6*Lambda**2-9/4*TH
                                    +3/2*np.trace(Yu @ Yu.conj().T @ Yd @ Yd.conj().T)))*Yd[p-1,r-1]\
                            +((-3/2*(Yu @ Yu.conj().T -Yd @ Yd.conj().T)
                               +loop*((187/48*gp**2+135/16*g**2+16*gs**2-12*Lambda-9/4*WH)* Yd @ Yd.conj().T
                                      +(-79/48*gp**2+9/16*g**2-16*gs**2+5/4*WH)* Yu @ Yu.conj().T
                                      +3/2*Yd @ Yd.conj().T @ Yd @ Yd.conj().T
                                      +11/4*Yu @ Yu.conj().T @ Yu @ Yu.conj().T
                                      -Yu @ Yu.conj().T @Yd @ Yd.conj().T
                                      -1/4*Yd @ Yd.conj().T @ Yu @ Yu.conj().T
                                     )
                               )@Yd
                              )[p-1,r-1]
                                  
    for p, r in indices:
        Beta[f"Yl{p}{r}"] = (- 15/4*gp**2-9/4*g**2 +WH
                               +loop*(457/24*gp**4+9/4*gp**2*g**2-23/4*g**4
                               +6*Lambda**2+5/24*gp**2*gmuhT +15/8*g**2*WH
                               +20*gs**2*np.trace(Yu.conj().T @ Yu + Yd.conj().T @ Yd)-9/4*TH 
                               +3/2*np.trace(Yu@Yu.conj().T@Yd@Yd.conj().T))
                            )*Yl[p-1,r-1]\
                            +((3/2*Yl @ Yl.conj().T
                             +loop*((129/16*gp**2+135/16*g**2-12*Lambda-9/4*WH)*Yl@Yl.conj().T
                                    +3/2*Yl @ Yl.conj().T @ Yl @ Yl.conj().T
                                    )
                            ) @ Yl                                
                            )[p-1,r-1] 
                            
#################################################################################################
#                                                                                               #
#                        Beta function for dim-5 + dim-7 SMEFT operators                        #
#                                                                                               #
#################################################################################################

    # 2FS: LH5 LH DLDH1 DLDH2
    indices = [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)] 

    """Dimension-5 operator"""
    
    for p, r in indices:
        Beta[f"LH5_{p}{r}"] = (
            ###### dim-5 contribution to LH5
            1/2*(-3*g**2 + 4*Lambda + 2*WH)*C_LH5[p-1,r-1]
            +1/2*(-3*g**2 + 4*Lambda + 2*WH)*C_LH5[r-1,p-1]
            -3/2*(C_LH5 @ Yl @ Yl.conj().T)[p-1,r-1]
            -3/2*(C_LH5 @ Yl @ Yl.conj().T)[r-1,p-1]
            ###### dim-7 contribution to LH5
            +muh*(
                8*C_LH[p-1,r-1]
                +8*C_LH[r-1,p-1]
                +2*(C_LeDH @ Yl.conj().T)[p-1,r-1]
                +2*(C_LeDH @ Yl.conj().T)[r-1,p-1]
                +3/2*g**2*(2*C_DLDH1+C_DLDH2)[p-1,r-1]
                +3/2*g**2*(2*C_DLDH1+C_DLDH2)[r-1,p-1]
                +(C_DLDH1 @ Yl @ Yl.conj().T)[p-1,r-1]
                +(C_DLDH1 @ Yl @ Yl.conj().T)[r-1,p-1]
                -1/2*(C_DLDH2 @ Yl @ Yl.conj().T)[p-1,r-1]
                -1/2*(C_DLDH2 @ Yl @ Yl.conj().T)[r-1,p-1]
                -np.tensordot(Yl[:, :], (3*C_eLLLHS[:,:,p-1,r-1]+2*C_eLLLHM[:,:,p-1,r-1]), axes=([0, 1], [1, 0]))
                -np.tensordot(Yl[:, :], (3*C_eLLLHS[:,:,r-1,p-1]+2*C_eLLLHM[:,:,r-1,p-1]), axes=([0, 1], [1, 0]))
                -3*np.tensordot(Yd[:, :], C_dLQLH1[:,p-1,:,r-1], axes=([0, 1], [1, 0]))
                -3*np.tensordot(Yd[:, :], C_dLQLH1[:,r-1,:,p-1], axes=([0, 1], [1, 0]))
                +6*np.tensordot(Yu.conj().T[:, :], C_QuLLH[:,:,p-1,r-1], axes=([0, 1], [1, 0]))
                +6*np.tensordot(Yu.conj().T[:, :], C_QuLLH[:,:,r-1,p-1], axes=([0, 1], [1, 0]))           
                )
            )

    for p, r in indices:
        Beta[f"LH_{p}{r}"] = (
            - 1 / 4 * (3 * gp**2 + 15*g**2 -80 * Lambda - 8 * WH) * C_LH[p-1, r-1]
            - 1 / 4 * (3 * gp**2 + 15*g**2 -80 * Lambda - 8 * WH) * C_LH[r-1, p-1]
            - 3 / 2 * (C_LH @ Yl @ Yl.conj().T)[p-1, r-1]
            - 3 / 2 * (C_LH @ Yl @ Yl.conj().T)[r-1, p-1]
            +(2*Lambda-3/2*g**2)*(C_LeDH@Yl.conj().T)[p-1, r-1]
            +(2*Lambda-3/2*g**2)*(C_LeDH@Yl.conj().T)[r-1, p-1]
            +(C_LeDH@Yl.conj().T@Yl@Yl.conj().T)[p-1, r-1]
            +(C_LeDH@Yl.conj().T@Yl@Yl.conj().T)[r-1, p-1]
            - 3 / 4 *g**2*(g**2-4*Lambda)* C_DLDH1[p-1, r-1]
            - 3 / 4 *g**2*(g**2-4*Lambda)* C_DLDH1[r-1, p-1]
            + Lambda* (C_DLDH1 @ Yl @ Yl.conj().T)[p-1, r-1]
            + Lambda* (C_DLDH1 @ Yl @ Yl.conj().T)[r-1, p-1]
            - (C_DLDH1 @ Yl @ Yl.conj().T @ Yl @ Yl.conj().T)[p-1, r-1]
            - (C_DLDH1 @ Yl @ Yl.conj().T @ Yl @ Yl.conj().T)[r-1, p-1]
            - 3 / 8 *(gp**4+2*gp**2*g**2+3*g**4-4*g**2*Lambda)*C_DLDH2[p-1, r-1]
            - 3 / 8 *(gp**4+2*gp**2*g**2+3*g**4-4*g**2*Lambda)*C_DLDH2[r-1, p-1]
            - 1 / 2 *Lambda* (C_DLDH2 @ Yl @ Yl.conj().T)[p-1, r-1]
            - 1 / 2 *Lambda* (C_DLDH2 @ Yl @ Yl.conj().T)[r-1, p-1]
            - (C_DLDH2 @ Yl @ Yl.conj().T @ Yl @ Yl.conj().T)[p-1, r-1]
            - (C_DLDH2 @ Yl @ Yl.conj().T @ Yl @ Yl.conj().T)[r-1, p-1]
            - 3*g**3*C_LHW[p-1, r-1]
            - 3*g**3*C_LHW[r-1, p-1]
            -6*g*(C_LHW @ Yl @ Yl.conj().T)[p-1, r-1]
            -6*g*(C_LHW @ Yl @ Yl.conj().T)[r-1, p-1]
            -3*np.tensordot(C_eLLLHS[:, :, p-1, r-1], (Lambda*Yl-Yl@ Yl.conj().T@ Yl)[:, :], axes=([1, 0], [0, 1]))
            -3*np.tensordot(C_eLLLHS[:, :, r-1, p-1], (Lambda*Yl-Yl@ Yl.conj().T@ Yl)[:, :], axes=([1, 0], [0, 1]))
            -2*np.tensordot(C_eLLLHM[:, :, p-1, r-1], (Lambda*Yl-Yl@ Yl.conj().T@ Yl)[:, :], axes=([1, 0], [0, 1]))
            -2*np.tensordot(C_eLLLHM[:, :, r-1, p-1], (Lambda*Yl-Yl@ Yl.conj().T@ Yl)[:, :], axes=([1, 0], [0, 1]))
            -3*np.tensordot(C_dLQLH1[:, p-1, :, r-1], (Lambda*Yd-Yd@ Yd.conj().T@ Yd)[:, :], axes=([1, 0], [0, 1]))
            -3*np.tensordot(C_dLQLH1[:, r-1, :, p-1], (Lambda*Yd-Yd@ Yd.conj().T@ Yd)[:, :], axes=([1, 0], [0, 1]))
            +6*np.tensordot(C_QuLLH[:, :, p-1, r-1], (Lambda*Yu.conj().T-Yu.conj().T@ Yu@ Yu.conj().T)[:, :], axes=([1, 0], [0, 1]))
            +6*np.tensordot(C_QuLLH[:, :, r-1, p-1], (Lambda*Yu.conj().T-Yu.conj().T@ Yu@ Yu.conj().T)[:, :], axes=([1, 0], [0, 1]))
          )
     
    for p, r in indices:
        Beta[f"DLDH1_{p}{r}"] = (
            - 1 / 4 * (3 * gp**2 - 11 * g**2 - 4 * WH) * C_DLDH1[p-1, r-1]  
            - 1 / 4 * (3 * gp**2 - 11 * g**2 - 4 * WH) * C_DLDH1[r-1, p-1]
            + 7 / 2 * (C_DLDH1 @ Yl @ Yl.conj().T)[p-1, r-1]
            + 7 / 2 * (C_DLDH1@ Yl @ Yl.conj().T)[r-1, p-1]
            - 1 / 8 * (11 * gp**2 + 11 * g**2 + 8 * Lambda) * C_DLDH2[p-1, r-1]
            - 1 / 8 * (11 * gp**2 + 11 * g**2 + 8 * Lambda) * C_DLDH2[r-1, p-1]
            +6*np.tensordot(C_duLDL[:, :, p-1, r-1], (Yu.conj().T @ Yd)[:, :], axes=([0, 1], [1, 0]))
            +6*np.tensordot(C_duLDL[:, :, r-1, p-1], (Yu.conj().T @ Yd)[:, :], axes=([0, 1], [1, 0]))
        )

    for p, r in indices:
        Beta[f"DLDH2_{p}{r}"] = (
            - 4 * g**2 * C_DLDH1[p-1, r-1]
            - 4 * g**2 * C_DLDH1[r-1, p-1]
            - 4 * (C_DLDH1 @ Yl @ Yl.conj().T)[p-1, r-1]
            - 4 * (C_DLDH1 @ Yl @ Yl.conj().T)[r-1, p-1]
            + 1 / 2 * (4 * gp**2 + g**2 + 4 * Lambda + 2 * WH) * C_DLDH2[p-1, r-1]
            + 1 / 2 * (4 * gp**2 + g**2 + 4 * Lambda + 2 * WH) * C_DLDH2[r-1, p-1]
            - 3 / 2 * (C_DLDH2 @ Yl @ Yl.conj().T)[p-1, r-1]
            - 3 / 2 * (C_DLDH2 @ Yl @ Yl.conj().T)[r-1, p-1]
          )
        
####################################################################################################    
        
    # 2F: LHW LeDH 
    indices = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]  
       
    for p, r in indices:
        Beta[f"LHW_{p}{r}"] = (
             1 / 2 * g**3 * C_DLDH1[p-1, r-1]  
             - 1 / 4 * g * (C_DLDH1 @ Yl @ Yl.conj().T)[r-1, p-1]
             + 1 / 2 * g * (C_DLDH1 @ Yl @ Yl.conj().T)[p-1, r-1]
             + 5 / 8 * g**3 * C_DLDH2[p-1, r-1]
             + 3 / 4 * g * (C_DLDH2 @ Yl @ Yl.conj().T)[p-1, r-1]
             + 1 / 8 * g * (C_DLDH2 @ Yl @ Yl.conj().T)[r-1, p-1]
             -1/2*(4*gp**2-9*g**2-8*Lambda-4*WH)*C_LHW[p-1, r-1]
             +7/2*g**2*C_LHW[r-1, p-1]
             + 9 / 2 * (C_LHW @ Yl @ Yl.conj().T)[p-1, r-1]
             + 2 * (C_LHW @ Yl @ Yl.conj().T)[r-1, p-1]
             -3/2*(C_LHW.T @ Yl @ Yl.conj().T)[r-1, p-1]  
             +3*gp*g*C_LHB[p-1, r-1]     
             -1/4*g*np.tensordot((3*C_eLLLHS+C_eLLLHA-2*C_eLLLHM)[:,:,p-1,r-1], Yl[:, :], axes=([0, 1], [1, 0]))
             -3/4*g*np.tensordot(C_dLQLH1[:,r-1,:,p-1], Yd[:, :], axes=([0, 1], [1, 0]))
             -3/4*g*np.tensordot(C_dLQLH2[:,p-1,:,r-1]+C_dLQLH2[:,r-1,:,p-1], Yd[:, :], axes=([0, 1], [1, 0]))           
        )
     
    for p, r in indices:
        Beta[f"LeDH_{p}{r}"] = (
             -3/2*(3*gp**2-4*Lambda-2*WH)*C_LeDH[p-1, r-1]
             +(Yl.T @ C_LeDH @ Yl.conj().T)[r-1, p-1]
             +4*(C_LeDH @ Yl.conj().T @ Yl)[p-1, r-1]
             +1/2*(C_LeDH.T @ Yl @ Yl.conj().T)[r-1, p-1]
             +(3*gp**2-g**2)*(C_DLDH1 @ Yl)[p-1, r-1]
             -2*(C_DLDH1 @ Yl @ Yl.conj().T @ Yl)[p-1, r-1]
             +1/8*(7*gp**2-17*g**2-8*Lambda)*(C_DLDH2 @ Yl)[p-1, r-1]
             -2*(C_DLDH2 @ Yl @ Yl.conj().T @ Yl)[p-1, r-1]
             -1/2*(Yl.T @ C_DLDH2 @ Yl @ Yl.conj().T)[r-1, p-1]
             -6*np.tensordot(C_dLueH[:,p-1,:,r-1], Yu.conj().T @ Yd[:, :], axes=([0, 1], [1, 0]))
        )
        
####################################################################################################

    # 2FA：LHB 
    indices = [(1,2), (1,3), (2,3)]  
       
    for p, r in indices:
        Beta[f"LHB_{p}{r}"] = (
             -1 / 8 * gp * (C_DLDH1 @ Yl @ Yl.conj().T)[p-1, r-1]
             +1 / 8 * gp * (C_DLDH1 @ Yl @ Yl.conj().T)[r-1, p-1]
             -3 / 16 * gp * (C_DLDH2 @ Yl @ Yl.conj().T)[p-1, r-1]
             +3 / 16 * gp * (C_DLDH2 @ Yl @ Yl.conj().T)[r-1, p-1]
             +3 * gp * g * C_LHW[p-1, r-1]
             -3 * gp * g * C_LHW[r-1, p-1]
             -3 / 2 * (C_LHB @ Yl @ Yl.conj().T)[p-1, r-1]
             +3 / 2 * (C_LHB @ Yl @ Yl.conj().T)[r-1, p-1]
             +1/12*(47*gp**2-30*g**2+24*Lambda+12*WH)*C_LHB[p-1, r-1]
             -1/12*(47*gp**2-30*g**2+24*Lambda+12*WH)*C_LHB[r-1, p-1]
             +3/8*gp*np.tensordot((C_eLLLHA-2*C_eLLLHM)[:,:,p-1,r-1], Yl[:, :], axes=([0, 1], [1, 0]))
             -3/8*gp*np.tensordot((C_eLLLHA-2*C_eLLLHM)[:,:,r-1,p-1], Yl[:, :], axes=([0, 1], [1, 0]))
             -1/8*gp*np.tensordot(C_dLQLH1[:,p-1,:,r-1], Yd[:, :], axes=([0, 1], [1, 0]))
             +1/8*gp*np.tensordot(C_dLQLH1[:,r-1,:,p-1], Yd[:, :], axes=([0, 1], [1, 0]))        
        )

####################################################################################################

    # 4F2S: duLDL LQdDd 
    indices = [
    (1,1,1,1), (1,1,1,2), (1,1,1,3), (1,1,2,2), (1,1,2,3), (1,1,3,3),
    (1,2,1,1), (1,2,1,2), (1,2,1,3), (1,2,2,2), (1,2,2,3), (1,2,3,3),
    (1,3,1,1), (1,3,1,2), (1,3,1,3), (1,3,2,2), (1,3,2,3), (1,3,3,3),
    (2,1,1,1), (2,1,1,2), (2,1,1,3), (2,1,2,2), (2,1,2,3), (2,1,3,3),
    (2,2,1,1), (2,2,1,2), (2,2,1,3), (2,2,2,2), (2,2,2,3), (2,2,3,3),
    (2,3,1,1), (2,3,1,2), (2,3,1,3), (2,3,2,2), (2,3,2,3), (2,3,3,3),
    (3,1,1,1), (3,1,1,2), (3,1,1,3), (3,1,2,2), (3,1,2,3), (3,1,3,3),
    (3,2,1,1), (3,2,1,2), (3,2,1,3), (3,2,2,2), (3,2,2,3), (3,2,3,3),
    (3,3,1,1), (3,3,1,2), (3,3,1,3), (3,3,2,2), (3,3,2,3), (3,3,3,3),
      ] 

    for p, r, s, t in indices:
        Beta[f"duLDL_{p}{r}{s}{t}"] = (
            (2*C_DLDH1+C_DLDH2)[s-1, t-1] * (Yd.conj().T @ Yu)[p-1, r-1]  
            + 1/6*(gp**2+9*g**2)*C_duLDL[p-1, r-1, s-1, t-1]   
            + C_duLDL[:, r-1, s-1, t-1] @ (Yd.conj().T @ Yd)[p-1, :]
            + C_duLDL[p-1, :, s-1, t-1] @ (Yu.conj().T @ Yu)[:, r-1]
            + 1/2*C_duLDL[p-1, r-1, :, t-1] @ (Yl @ Yl.conj().T)[:, s-1]
            + 1/2*C_duLDL[p-1, r-1, s-1, :] @ (Yl @ Yl.conj().T)[:, t-1]
        )

    for p, r, s, t in indices:
        Beta[f"LQdDd_{p}{r}{s}{t}"] = (
           - 3 * np.tensordot(C_eddDd[:, :, s-1, t-1], Yl[p-1, :], axes=([0], [0])) @ (Yd.conj().T)[:, r-1]
           + 4/9*(gp**2+3*gs**2)*C_LQdDd[p-1, r-1, s-1, t-1]
           - np.tensordot(C_LQdDd[p-1, :, :, t-1], (Yd.conj().T)[:, r-1], axes=([1], [0])) @ Yd[:, s-1]
           - np.tensordot(C_LQdDd[p-1, :, s-1, :], (Yd.conj().T)[:, r-1], axes=([1], [0])) @ Yd[:, t-1]
           + 1/2*C_LQdDd[:, r-1, s-1, t-1] @ (Yl @ Yl.conj().T)[p-1, :]
           + 1/2*C_LQdDd[p-1, :, s-1, t-1] @ (Yd @ Yd.conj().T+Yu @ Yu.conj().T)[:, r-1]
           + C_LQdDd[p-1, r-1, :, t-1] @ (Yd @ Yd.conj().T)[:, s-1]
           + C_LQdDd[p-1, r-1, s-1, :] @ (Yd @ Yd.conj().T)[:, t-1]
        )

####################################################################################################

    # 4F: LdudH QuLLH LdQQH dLQLH1 dLQLH2 dLueH   
    indices = [
    (1,1,1,1), (1,1,1,2), (1,1,1,3), (1,1,2,1), (1,1,2,2), (1,1,2,3), (1,1,3,1), (1,1,3,2), (1,1,3,3),
    (1,2,1,1), (1,2,1,2), (1,2,1,3), (1,2,2,1), (1,2,2,2), (1,2,2,3), (1,2,3,1), (1,2,3,2), (1,2,3,3),
    (1,3,1,1), (1,3,1,2), (1,3,1,3), (1,3,2,1), (1,3,2,2), (1,3,2,3), (1,3,3,1), (1,3,3,2), (1,3,3,3),
    (2,1,1,1), (2,1,1,2), (2,1,1,3), (2,1,2,1), (2,1,2,2), (2,1,2,3), (2,1,3,1), (2,1,3,2), (2,1,3,3),
    (2,2,1,1), (2,2,1,2), (2,2,1,3), (2,2,2,1), (2,2,2,2), (2,2,2,3), (2,2,3,1), (2,2,3,2), (2,2,3,3),
    (2,3,1,1), (2,3,1,2), (2,3,1,3), (2,3,2,1), (2,3,2,2), (2,3,2,3), (2,3,3,1), (2,3,3,2), (2,3,3,3),
    (3,1,1,1), (3,1,1,2), (3,1,1,3), (3,1,2,1), (3,1,2,2), (3,1,2,3), (3,1,3,1), (3,1,3,2), (3,1,3,3),
    (3,2,1,1), (3,2,1,2), (3,2,1,3), (3,2,2,1), (3,2,2,2), (3,2,2,3), (3,2,3,1), (3,2,3,2), (3,2,3,3),
    (3,3,1,1), (3,3,1,2), (3,3,1,3), (3,3,2,1), (3,3,2,2), (3,3,2,3), (3,3,3,1), (3,3,3,2), (3,3,3,3)
    ]  
      
    for p, r, s, t in indices:
        Beta[f"LdudH_{p}{r}{s}{t}"] = (
             -1/12*(17*gp**2+27*g**2+48*gs**2-12*WH)*C_LdudH[p-1, r-1, s-1, t-1]
             -10/3*gp**2*C_LdudH[p-1, t-1, s-1, r-1]
             +3*C_LdudH[p-1, :, s-1, t-1] @ (Yd.conj().T @ Yd)[:, r-1]
             +3*C_LdudH[p-1, r-1, s-1, :] @ (Yd.conj().T @ Yd)[:, t-1]
             -3/2*C_LdudH[:, r-1, s-1, t-1] @ (Yl @ Yl.conj().T)[p-1, :]
             +2*C_LdudH[p-1, r-1, :, t-1] @ (Yu.conj().T @ Yu)[:, s-1]
             -2*(C_LdddHM[p-1,r-1,:,t-1]+C_LdddHM[p-1,:,r-1,t-1]) @ (Yd.conj().T @ Yu)[:, s-1]
             +4*np.tensordot(C_eQddH[:, :, r-1, t-1], Yl[p-1, :], axes=([0], [0])) @ Yu[:, s-1]
             -2*np.tensordot(C_LdQQH[p-1, r-1, :, :], Yu[:, s-1], axes=([0], [0])) @ Yd[:, t-1]
             -2*np.tensordot(C_LdQQH[p-1, r-1, :, :], Yu[:, s-1], axes=([1], [0])) @ Yd[:, t-1]
             +np.tensordot(C_eddDd[:, r-1, t-1, :], (Yd.conj().T @ Yu)[:, s-1], axes=([1], [0])) @ Yl[p-1, :]
             +1/18*(29*gp**2+27*g**2+96*gs**2)*C_LQdDd[p-1, :, r-1, t-1] @ Yu[:, s-1]
             - 2*np.tensordot(C_LQdDd[p-1, :, r-1, :], (Yd.conj().T @ Yd)[:, t-1], axes=([1], [0])) @ Yu[:, s-1]
             - np.tensordot(C_LQdDd[p-1, :, r-1, :], (Yd.conj().T @ Yu)[:, s-1], axes=([1], [0])) @ Yd[:, t-1]
             + 2*np.tensordot(C_LQdDd[p-1, :, :, t-1], (Yd.conj().T @ Yd)[:, r-1], axes=([1], [0])) @ Yu[:, s-1]         
        )


    for p, r, s, t in indices:
        Beta[f"QuLLH_{p}{r}{s}{t}"] = (
             Yu[p-1, r-1] * (3*g**2*C_DLDH1[s-1, t-1]
             +(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, s-1]
             + 2* (C_DLDH1 @ Yl @ Yl.conj().T)[s-1, t-1])
             +1/2*Yu[p-1, r-1] *(3*g**2*C_DLDH2[s-1, t-1]+4*(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, s-1]-(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, t-1])
             -Yu[p-1,r-1]*(3*np.tensordot(Yl[:, :], C_eLLLHS[:,:,s-1,t-1], axes=([0,1], [1,0]))
                            +3*np.tensordot(Yl[:, :], C_eLLLHA[:,:,s-1,t-1], axes=([0,1], [1,0]))
                            +4*np.tensordot(Yl[:, :], C_eLLLHM[:,:,s-1,t-1], axes=([0,1], [1,0]))
                            -2*np.tensordot(Yl[:, :], C_eLLLHM[:,:,t-1,s-1], axes=([0,1], [1,0])))           
            +np.tensordot((C_dLQLH1[:,s-1,:,t-1]+C_dLQLH2[:,s-1,:,t-1]-C_dLQLH2[:,t-1,:,s-1]), Yd[p-1,:], axes=([0], [0]))@Yu[:,r-1]
            -3*np.tensordot((C_dLQLH1[:,s-1,:,t-1]+C_dLQLH2[:,s-1,:,t-1]-C_dLQLH2[:,t-1,:,s-1]), Yd[:,:], axes=([0,1], [1,0]))*Yu[p-1,r-1]
            +np.tensordot(C_dLueH[:,t-1,r-1,:], Yd[p-1,:], axes=([0], [0]))@Yl.conj().T[:,s-1]            
             + 1/12*(gp**2-45*g**2-96*gs**2+12*WH)*C_QuLLH[p-1, r-1, s-1, t-1]
             + 3*g**2*C_QuLLH[p-1, r-1, t-1,  s-1]
             + 5/2*C_QuLLH[p-1, r-1, :, t-1] @ (Yl @ Yl.conj().T)[:, s-1]
             - 1/2*(3*C_QuLLH[p-1, r-1, s-1, :]+ 4*C_QuLLH[p-1, r-1, :, s-1]) @ (Yl @ Yl.conj().T)[:, t-1]          
             - 1/2*C_QuLLH[:, r-1, s-1, t-1] @ (Yd @ Yd.conj().T- Yu @ Yu.conj().T)[p-1, :]
             + C_QuLLH[:, r-1, t-1, s-1] @ (2*Yu @ Yu.conj().T+Yd @ Yd.conj().T)[p-1, :]
             + 3*C_QuLLH[p-1, :, s-1, t-1] @ (Yu.conj().T @ Yu)[:, r-1]
             + 6*np.tensordot(C_QuLLH[:, :, s-1, t-1], Yu.conj().T[:, :], axes=([0,1],[1,0]))*Yu[p-1, r-1]
             +3*g**2*C_duLDL[:, r-1, s-1, t-1] @ Yd[p-1, :]
             + 2*np.tensordot(C_duLDL[:, r-1, t-1, :], Yd[p-1, :], axes=([0], [0])) @ (Yl @ Yl.conj().T)[:, s-1]
        )
        
    for p, r, s, t in indices:
        Beta[f"LdQQH_{p}{r}{s}{t}"] = (
             -2*np.tensordot(C_LdudH[p-1, r-1, :, :], Yu.conj().T[:, s-1], axes=([0], [0])) @ Yd.conj().T[:, t-1]
             -2*np.tensordot(C_LdudH[p-1, r-1, :, :], Yu.conj().T[:, t-1], axes=([0], [0])) @ Yd.conj().T[:, s-1]
             -np.tensordot(C_LdudH[p-1, :, :,r-1], Yu.conj().T[:, s-1], axes=([1], [0])) @ Yd.conj().T[:, t-1]
             -np.tensordot(C_LdudH[p-1, :, :,r-1], Yu.conj().T[:, t-1], axes=([1], [0])) @ Yd.conj().T[:, s-1]
             -2*np.tensordot(C_eQddH[:, t-1, :, r-1], Yl[p-1,:], axes=([0], [0])) @ Yd.conj().T[:, s-1]
             - 1/12*(19*gp**2+45*g**2+48*gs**2-12*WH)*C_LdQQH[p-1, r-1, s-1, t-1]
             -3*g**2*C_LdQQH[p-1, r-1, t-1, s-1]
             -np.tensordot(C_LdQQH[p-1, :, :, t-1], Yd.conj().T[:, s-1], axes=([0], [0])) @ Yd[:, r-1]
             -np.tensordot(C_LdQQH[p-1, :, s-1, :], Yd.conj().T[:, t-1], axes=([0], [0])) @ Yd[:, r-1]
             +1/2*(C_LdQQH[:,r-1,s-1,t-1]-4*C_LdQQH[:,r-1,t-1,s-1])@(Yl @ Yl.conj().T)[p-1, :]
             +3*C_LdQQH[p-1, :, s-1, t-1] @ (Yd.conj().T @ Yd)[:, r-1]
             +1/2*C_LdQQH[p-1, r-1, :, t-1] @ (Yd @ Yd.conj().T+5*Yu @ Yu.conj().T)[:, s-1]
             +1/2*(4*C_LdQQH[p-1, r-1, :, s-1]-3*C_LdQQH[p-1, r-1, s-1, :]) @ (Yu @ Yu.conj().T)[:, t-1]
             +1/2*(5*C_LdQQH[p-1, r-1, s-1, :]-2*C_LdQQH[p-1, r-1, :, s-1]) @ (Yd @ Yd.conj().T)[:, t-1]
             +3*np.tensordot(np.tensordot(C_eddDd[:, :, :, r-1], Yl[p-1, :], axes=([0], [0])), Yd.conj().T[:, s-1], axes=([0], [0]))@Yd.conj().T[:, t-1]
             -2/9*(gp**2-24*gs**2)*C_LQdDd[p-1, s-1, r-1, :] @ Yd.conj().T[:, t-1]
             +2*np.tensordot(C_LQdDd[p-1, s-1, :, :], (Yd.conj().T @ Yd)[:, r-1], axes=([0], [0])) @ Yd.conj().T[:, t-1]
             -np.tensordot(C_LQdDd[:, t-1, r-1, :], (Yl @ Yl.conj().T)[p-1, :], axes=([0], [0])) @ Yd.conj().T[:, s-1]
             +np.tensordot(C_LQdDd[p-1, :, r-1, :], (Yu @ Yu.conj().T)[:, s-1], axes=([0], [0])) @ Yd.conj().T[:, t-1]
             +np.tensordot(C_LQdDd[p-1, :, r-1, :], (Yu @ Yu.conj().T)[:, t-1], axes=([0], [0])) @ Yd.conj().T[:, s-1]
        )

    for p, r, s, t in indices:
        Beta[f"dLQLH1_{p}{r}{s}{t}"] = (
            -3*g**2*Yd.conj().T[p-1, s-1]*(2*C_DLDH1+C_DLDH2)[r-1, t-1]
            -3*Yd.conj().T[p-1, s-1]*((C_DLDH1 @ Yl @ Yl.conj().T)[r-1, t-1]
                                      +(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, r-1])
            -3/2*Yd.conj().T[p-1, s-1]*((C_DLDH2 @ Yl @ Yl.conj().T)[r-1, t-1]
                                      +(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, r-1])
            +8*g*Yd.conj().T[p-1, s-1]*(C_LHW[r-1, t-1]-C_LHW[t-1, r-1])
            -8/3*gp*Yd.conj().T[p-1, s-1]*C_LHB[r-1, t-1]
            +2*Yd.conj().T[p-1, s-1]*np.tensordot(Yl[:,:], (3*C_eLLLHS+C_eLLLHM)[:, :,r-1, t-1]+C_eLLLHM[:, :,t-1, r-1], axes=([0,1], [1,0])) 
            - 1/36*(41*gp**2+63*g**2+96*gs**2-36*WH)*C_dLQLH1[p-1, r-1, s-1, t-1]
            + 4/9*(5*gp**2+9*g**2-12*gs**2)*C_dLQLH1[p-1, t-1, s-1, r-1]
            +1/2*C_dLQLH1[p-1, r-1, :, t-1] @ (5*Yd @ Yd.conj().T+Yu @ Yu.conj().T)[:, s-1]
            +3*C_dLQLH1[:, r-1, s-1, t-1] @ (Yd.conj().T @ Yd)[p-1, :]
            +1/2*C_dLQLH1[p-1, :, s-1, t-1] @ (Yl @ Yl.conj().T)[:, r-1]
            -3/2*C_dLQLH1[p-1, r-1, s-1, :] @ (Yl @ Yl.conj().T)[:, t-1]
            +3*Yd.conj().T[p-1, s-1]*np.tensordot(Yd[:, :], C_dLQLH1[:, r-1, :, t-1]+C_dLQLH1[:, t-1, :, r-1], axes=([0, 1], [1, 0]))
            +2*g**2*(2*C_dLQLH2[p-1,r-1,s-1,t-1]+C_dLQLH2[p-1,t-1,s-1,r-1])
            +2*C_dLQLH2[p-1,:,s-1,t-1]@(Yl @ Yl.conj().T)[:, r-1]
            -2*C_dLQLH2[p-1,r-1,s-1,:]@(Yl @ Yl.conj().T)[:, t-1]
            -2*np.tensordot(C_dLueH[p-1,r-1,:,:], Yu.conj().T[:,s-1], axes=([0], [0])) @ Yl.conj().T[:,t-1]
            +2*np.tensordot((C_QuLLH[:,:,r-1,t-1]+C_QuLLH[:,:,t-1,r-1]), Yd.conj().T[p-1,:], axes=([0], [0])) @ Yu.conj().T[:,s-1]
            -6*np.tensordot((C_QuLLH[:,:,r-1,t-1]+C_QuLLH[:,:,t-1,r-1]), Yu.conj().T[:,:], axes=([0,1], [1,0])) * Yd.conj().T[p-1,s-1]
            -6*g**2*C_duLDL[p-1,:,r-1,t-1]@Yu.conj().T[:,s-1]
            -2*np.tensordot(C_duLDL[p-1,:,r-1,:], Yu.conj().T[:,s-1], axes=([0], [0])) @ (Yl @ Yl.conj().T)[:,t-1]
            -2*np.tensordot(C_duLDL[p-1,:,t-1,:], Yu.conj().T[:,s-1], axes=([0], [0])) @ (Yl @ Yl.conj().T)[:,r-1]     
        )
        
    for p, r, s, t in indices:
        Beta[f"dLQLH2_{p}{r}{s}{t}"] = (
            1/3*(gp**2+9*g**2)*Yd.conj().T[p-1, s-1]*C_DLDH1[r-1, t-1]
            +Yd.conj().T[p-1, s-1]*((C_DLDH1 @ Yl @ Yl.conj().T)[r-1, t-1]
                                      +(2*C_DLDH1 @ Yl @ Yl.conj().T)[t-1, r-1])
            +1/6*gp**2*Yd.conj().T[p-1, s-1]*C_DLDH2[r-1, t-1]
            +1/2*Yd.conj().T[p-1, s-1]*(4*(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, t-1]
                                      -(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, r-1])
            -2*g*Yd.conj().T[p-1, s-1]*(5*C_LHW[r-1, t-1]+C_LHW[t-1, r-1])
            +4/3*gp*Yd.conj().T[p-1, s-1]*C_LHB[r-1, t-1]
            -Yd.conj().T[p-1,s-1]*np.tensordot(Yl[:, :], (3*C_eLLLHS-3*C_eLLLHA-2*C_eLLLHM)[:, :, r-1, t-1]+4*C_eLLLHM[:, :, t-1, r-1], axes=([0, 1], [1, 0]))
            +2*g**2*C_dLQLH1[p-1, r-1, s-1, t-1]
            -2/9*(10*gp**2+9*g**2-24*gs**2)*C_dLQLH1[p-1, t-1, s-1, r-1]
            -C_dLQLH1[p-1, r-1, :, t-1] @ (2*Yd @ Yd.conj().T-Yu @ Yu.conj().T)[:, s-1]
            +2*C_dLQLH1[p-1, :, s-1, t-1] @ (Yl @ Yl.conj().T)[:, r-1]
            -3*Yd.conj().T[p-1, s-1]*np.tensordot(Yd[:, :], C_dLQLH1[:, t-1, :, r-1], axes=([0, 1], [1, 0]))
            -1/36*(41*gp**2+207*g**2+96*gs**2-36*WH)*C_dLQLH2[p-1, r-1, s-1, t-1]
            -2/9*(10*gp**2-9*g**2-24*gs**2)*C_dLQLH2[p-1, t-1, s-1, r-1]
            -1/2*C_dLQLH2[p-1, r-1, :, t-1] @ (3*Yd @ Yd.conj().T-5*Yu @ Yu.conj().T)[:, s-1]
            +3*C_dLQLH2[:, r-1, s-1, t-1] @ (Yd.conj().T @ Yd)[p-1, :]
            +5/2*C_dLQLH2[p-1, r-1, s-1, :] @ (Yl @ Yl.conj().T)[:, t-1]
            +1/2*C_dLQLH2[p-1, :, s-1, t-1] @ (Yl @ Yl.conj().T)[:, r-1]
            +3*Yd.conj().T[p-1, s-1]*np.tensordot(Yd[:, :], C_dLQLH2[:, r-1, :, t-1]-C_dLQLH2[:, t-1, :, r-1], axes=([0, 1], [1, 0]))          
            +2*np.tensordot(C_dLueH[p-1,r-1,:, :], Yu.conj().T[:, s-1], axes=([0], [0]))@Yl.conj().T[:, t-1]            
            +6*Yd.conj().T[p-1, s-1]*np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, t-1, r-1], axes=([0, 1], [1, 0]))
            -2*np.tensordot(Yd.conj().T[p-1, :], C_QuLLH[:, :, t-1, r-1], axes=([0], [0])) @ Yu.conj().T[:, s-1]
            +1/3*(gp**2+9*g**2)*C_duLDL[p-1, :, r-1, t-1] @ Yu.conj().T[:, s-1]
            +2*np.tensordot(C_duLDL[p-1, :, r-1, :], Yu.conj().T[:, s-1], axes=([0], [0])) @ (Yl @ Yl.conj().T)[:, t-1]
        )
        
    for p, r, s, t in indices:
        Beta[f"dLueH_{p}{r}{s}{t}"] = (
            2*(Yd.conj().T @ Yu)[p-1, s-1]*(-3*C_LeDH+C_DLDH1 @ Yl +2*C_DLDH2 @ Yl)[r-1, t-1]
            -2*np.tensordot(Yu[:, s-1], C_dLQLH1[p-1, r-1, :, :],  axes=([0], [0])) @ Yl[:, t-1]
            +2*np.tensordot(Yu[:, s-1], C_dLQLH1[p-1, :, :, r-1],  axes=([0], [1])) @ Yl[:, t-1]
            +2*np.tensordot(Yu[:, s-1], C_dLQLH2[p-1, r-1, :, :],  axes=([0], [0])) @ Yl[:, t-1]
            +np.tensordot(Yu[:, s-1], C_dLQLH2[p-1, :, :, r-1],  axes=([0], [1])) @ Yl[:, t-1]
            -1/4*(23*gp**2+9*g**2-4*WH)*C_dLueH[p-1, r-1, s-1, t-1]
            +3*(Yd.conj().T @ Yd)[p-1, :] @ C_dLueH[:, r-1, s-1, t-1]
            +np.tensordot(C_dLueH[p-1, :, s-1, :], Yl[:, t-1],  axes=([0], [0])) @ Yl.conj().T[:, r-1]
            -3/2*C_dLueH[p-1, :, s-1, t-1] @ (Yl @ Yl.conj().T)[:, r-1]
            +3*C_dLueH[p-1, r-1, :, t-1] @ (Yu.conj().T @ Yu)[:, s-1]
            +2*C_dLueH[p-1, r-1, s-1, :] @ (Yl.conj().T @ Yl)[:, t-1]
            +np.tensordot(C_QuLLH[:, s-1, r-1, :]+2*C_QuLLH[:, s-1, :, r-1], Yd.conj().T[p-1, :],  axes=([0], [0])) @ Yl[:, t-1]           
            -1/3*(19*gp**2+9*g**2)*C_duLDL[p-1, s-1, r-1, :] @ Yl[:, t-1]
            -2*np.tensordot(C_duLDL[p-1, s-1, :, :], (Yl @ Yl.conj().T)[:, r-1],  axes=([0], [0])) @ Yl[:, t-1]
            +np.tensordot(C_duLDL[:, s-1, r-1, :], (Yd.conj().T @ Yd)[p-1, :],  axes=([0], [0])) @ Yl[:, t-1]
            +np.tensordot(C_duLDL[p-1, :, r-1, :], (Yu.conj().T @ Yu)[:, s-1],  axes=([0], [0])) @ Yl[:, t-1]   
           )
        
####################################################################################################

    # 3F2A: eQddH 
    indices = [
    (1,1,1,2), (1,1,1,3), (1,1,2,3), 
    (1,2,1,2), (1,2,1,3), (1,2,2,3), 
    (1,3,1,2), (1,3,1,3), (1,3,2,3), 
    (2,1,1,2), (2,1,1,3), (2,1,2,3), 
    (2,2,1,2), (2,2,1,3), (2,2,2,3), 
    (2,3,1,2), (2,3,1,3), (2,3,2,3), 
    (3,1,1,2), (3,1,1,3), (3,1,2,3), 
    (3,2,1,2), (3,2,1,3), (3,2,2,3), 
    (3,3,1,2), (3,3,1,3), (3,3,2,3), 
    ]
 
    for p, r, s, t in indices:
        Beta[f"eQddH_{p}{r}{s}{t}"] = (  
            1/2*np.tensordot(C_LdudH[:, s-1, :, t-1], Yl.conj().T[p-1, :],  axes=([0], [0])) @ Yu.conj().T[:, r-1]
            -1/2*np.tensordot(C_LdudH[:, t-1, :, s-1], Yl.conj().T[p-1, :],  axes=([0], [0])) @ Yu.conj().T[:, r-1]
            +1/24*(11*gp**2-27*g**2-48*gs**2+12*WH)*C_eQddH[p-1, r-1, s-1, t-1]
            -1/24*(11*gp**2-27*g**2-48*gs**2+12*WH)*C_eQddH[p-1, r-1, t-1, s-1]
            +np.tensordot(C_eQddH[p-1, :, :, s-1], Yd[:, t-1],  axes=([0], [0])) @ Yd.conj().T[:, r-1]
            -np.tensordot(C_eQddH[p-1, :, :, t-1], Yd[:, s-1],  axes=([0], [0])) @ Yd.conj().T[:, r-1]
            -1/4*C_eQddH[p-1, :, s-1, t-1] @ (3*Yu @ Yu.conj().T-5*Yd @ Yd.conj().T)[:, r-1]
            +1/4*C_eQddH[p-1, :, t-1, s-1] @ (3*Yu @ Yu.conj().T-5*Yd @ Yd.conj().T)[:, r-1]
            +3*C_eQddH[p-1, r-1, :, t-1] @ (Yd.conj().T @ Yd)[:, s-1]
            -3*C_eQddH[p-1, r-1, :, s-1] @ (Yd.conj().T @ Yd)[:, t-1]
            +C_eQddH[:, r-1, s-1, t-1] @ (Yl.conj().T @ Yl)[p-1, :]
            -C_eQddH[:, r-1, t-1, s-1] @ (Yl.conj().T @ Yl)[p-1, :]
            -1/2*np.tensordot(C_LdQQH[:, s-1, r-1, :]-2*C_LdQQH[:, s-1, :, r-1], Yl.conj().T[p-1, :],  axes=([0], [0])) @ Yd[:, t-1]
            +1/2*np.tensordot(C_LdQQH[:, t-1, r-1, :]-2*C_LdQQH[:, t-1, :, r-1], Yl.conj().T[p-1, :],  axes=([0], [0])) @ Yd[:, s-1]          
            -gp**2*C_eddDd[p-1, s-1, t-1, :] @ Yd.conj().T[:, r-1]
            +gp**2*C_eddDd[p-1, t-1, s-1, :] @ Yd.conj().T[:, r-1]
            +3/2*np.tensordot(C_eddDd[p-1, :, s-1, :], (Yd.conj().T @ Yd)[:, t-1],  axes=([1], [0])) @ Yd.conj().T[:, r-1]
            -3/2*np.tensordot(C_eddDd[p-1, :, t-1, :], (Yd.conj().T @ Yd)[:, s-1],  axes=([1], [0])) @ Yd.conj().T[:, r-1]
            -np.tensordot(C_LQdDd[:, r-1, s-1, :], (Yd.conj().T @ Yd)[:, t-1],  axes=([1], [0])) @ Yl.conj().T[p-1, :]
            +np.tensordot(C_LQdDd[:, r-1, t-1, :], (Yd.conj().T @ Yd)[:, s-1],  axes=([1], [0])) @ Yl.conj().T[p-1, :]
            +np.tensordot((np.tensordot(C_LQdDd[:, :, s-1, :], Yl.conj().T [p-1, :],  axes=([0], [0]))), Yd.conj().T [:, r-1],  axes=([1], [0])) @ Yd[:, t-1]
            -np.tensordot((np.tensordot(C_LQdDd[:, :, t-1, :], Yl.conj().T [p-1, :],  axes=([0], [0]))), Yd.conj().T [:, r-1],  axes=([1], [0])) @ Yd[:, s-1]
           )
        
####################################################################################################

    # 4F3S: eLLLHS eddDd 
    indices = [
    (1,1,1,1), (1,1,1,2), (1,1,1,3), (1,1,2,2), (1,1,2,3),
    (1,1,3,3), (1,2,2,2), (1,2,2,3), (1,2,3,3), (1,3,3,3),
    (2,1,1,1), (2,1,1,2), (2,1,1,3), (2,1,2,2), (2,1,2,3),
    (2,1,3,3), (2,2,2,2), (2,2,2,3), (2,2,3,3), (2,3,3,3),
    (3,1,1,1), (3,1,1,2), (3,1,1,3), (3,1,2,2), (3,1,2,3),
    (3,1,3,3), (3,2,2,2), (3,2,2,3), (3,2,3,3), (3,3,3,3),  
    ]
     
    for p, r, s, t in indices:
        Beta[f"eLLLHS_{p}{r}{s}{t}"] = (
             (gp**2-g**2)*(Yl.conj().T[p-1, r-1]*C_DLDH1[s-1, t-1]
                           +Yl.conj().T[p-1, s-1]*C_DLDH1[r-1, t-1]
                           +Yl.conj().T[p-1, t-1]*C_DLDH1[r-1, s-1])
            -1/2*((C_DLDH1 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1])
            +1/2*(gp**2-2*g**2)*(Yl.conj().T[p-1, r-1]*C_DLDH2[s-1, t-1]
                           +Yl.conj().T[p-1, s-1]*C_DLDH2[r-1, t-1]
                           +Yl.conj().T[p-1, t-1]*C_DLDH2[r-1, s-1])
            -1/4*((C_DLDH2 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1])
            -2*g*(Yl.conj().T[p-1, r-1]*(C_LHW[s-1, t-1]+C_LHW[t-1, s-1])
                  +Yl.conj().T[p-1, s-1]*(C_LHW[r-1, t-1]+C_LHW[t-1, r-1])
                  +Yl.conj().T[p-1, t-1]*(C_LHW[r-1, s-1]+C_LHW[s-1, r-1]))
            -1/4*(9*gp**2-9*g**2-4*WH)*C_eLLLHS[p-1, r-1, s-1, t-1]
            +3*(Yl.conj().T @ Yl)[p-1, :] @ C_eLLLHS[:, r-1, s-1, t-1]
            +np.tensordot(Yl[:, :], C_eLLLHS[:, :, s-1, t-1], axes=([0, 1], [0, 1]))*Yl.conj().T[p-1, r-1]
            +np.tensordot(Yl[:, :], C_eLLLHS[:, :, t-1, r-1], axes=([0, 1], [0, 1]))*Yl.conj().T[p-1, s-1]
            +np.tensordot(Yl[:, :], C_eLLLHS[:, :, r-1, s-1], axes=([0, 1], [0, 1]))*Yl.conj().T[p-1, t-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHS[p-1, :, s-1, t-1], axes=([0], [0])) @ Yl.conj().T[:, r-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHS[p-1, r-1, :, t-1], axes=([0], [0])) @ Yl.conj().T[:, s-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHS[p-1, r-1, s-1, :], axes=([0], [0])) @ Yl.conj().T[:, t-1]
            +2/3*((Yl@Yl.conj().T)[:,r-1]@(C_eLLLHM[p-1,:,s-1,t-1]+C_eLLLHM[p-1,:,t-1,s-1])
                +(Yl@Yl.conj().T)[:,s-1]@(2*C_eLLLHM[p-1,:,t-1,r-1]+C_eLLLHM[p-1,r-1,:,t-1])
                +(Yl@Yl.conj().T)[:,t-1]@(2*C_eLLLHM[p-1,:,s-1,r-1]+C_eLLLHM[p-1,r-1,:,s-1]))
            +1/3*np.tensordot(Yl[:,:], (C_eLLLHM[:,:,s-1,t-1]+C_eLLLHM[:,:,t-1,s-1])*Yl.conj().T[p-1, r-1]
            +(C_eLLLHM[:,:,r-1,t-1]+C_eLLLHM[:,:,t-1,r-1])*Yl.conj().T[p-1, s-1]
            +(C_eLLLHM[:,:,r-1,s-1]+C_eLLLHM[:,:,s-1,r-1])*Yl.conj().T[p-1, t-1], axes=([0,1], [1,0]))
            +1/2*np.tensordot(Yd[:,:], (C_dLQLH1[:,s-1,:,t-1]+C_dLQLH1[:,t-1,:,s-1])*Yl.conj().T[p-1, r-1]
            +(C_dLQLH1[:,r-1,:,t-1]+C_dLQLH1[:,t-1,:,r-1])*Yl.conj().T[p-1, s-1]
            +(C_dLQLH1[:,s-1,:,r-1]+C_dLQLH1[:,r-1,:,s-1])*Yl.conj().T[p-1, t-1], axes=([0,1], [1,0]))
            -np.tensordot(Yu.conj().T[:,:], (C_QuLLH[:,:,s-1,t-1]+C_QuLLH[:,:,t-1,s-1])*Yl.conj().T[p-1, r-1]
            +(C_QuLLH[:,:,r-1,t-1]+C_QuLLH[:,:,t-1,r-1])*Yl.conj().T[p-1, s-1]
            +(C_QuLLH[:,:,s-1,r-1]+C_QuLLH[:,:,r-1,s-1])*Yl.conj().T[p-1, t-1], axes=([0,1], [1,0]))         
        )

    for p, r, s, t in indices:
        Beta[f"eddDd_{p}{r}{s}{t}"] = (
            - 2 / 3 * (gp**2 - 6 * gs**2) * C_eddDd[p-1, r-1, s-1, t-1]  
            + (C_eddDd[:, r-1, s-1, t-1] @ (Yl.conj().T @ Yl)[p-1, :])
            + (C_eddDd[p-1, :, s-1, t-1] @ (Yd.conj().T @ Yd)[:, r-1])
            + (C_eddDd[p-1, r-1, :, t-1] @ (Yd.conj().T @ Yd)[:, s-1])
            + (C_eddDd[p-1, r-1, s-1, :] @ (Yd.conj().T @ Yd)[:, t-1])   
            -2/3*Yl.conj().T[p-1,:] @ (np.tensordot(C_LQdDd[:,:,s-1,t-1],Yd[:,r-1],axes=([1], [0]))
                                       +np.tensordot(C_LQdDd[:,:,r-1,t-1],Yd[:,s-1],axes=([1], [0]))
                                       +np.tensordot(C_LQdDd[:,:,r-1,s-1],Yd[:,t-1],axes=([1], [0])))
        )

####################################################################################################

    # 4F3A: eLLLHA 
    indices = [
    (1,1,2,3), (2,1,2,3), (3,1,2,3)  
    ]
    
    for p, r, s, t in indices:
        Beta[f"eLLLHA_{p}{r}{s}{t}"] = (
            1/6*((C_DLDH1 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  -(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  -(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1]
                  -(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1])
            -5/12*((C_DLDH2 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  -(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  -(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1]
                  -(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1])
            -4*g*(Yl.conj().T[p-1, r-1]*C_LHW[s-1, t-1]
                  -Yl.conj().T[p-1, r-1]*C_LHW[t-1, s-1]
                  -Yl.conj().T[p-1, s-1]*C_LHW[r-1, t-1]
                  +Yl.conj().T[p-1, s-1]*C_LHW[t-1, r-1]
                  +Yl.conj().T[p-1, t-1]*C_LHW[r-1, s-1]
                  -Yl.conj().T[p-1, t-1]*C_LHW[s-1, r-1])
            +12*gp*(Yl.conj().T[p-1, r-1]*C_LHB[s-1, t-1]
                  +Yl.conj().T[p-1, s-1]*C_LHB[t-1, r-1]
                  +Yl.conj().T[p-1, t-1]*C_LHB[r-1, s-1])
            -1/4*(9*gp**2+39*g**2-4*WH)*C_eLLLHA[p-1, r-1, s-1, t-1]
            +np.tensordot(Yl[:, :], C_eLLLHA[:, :, s-1, t-1], axes=([0, 1], [1, 0]))*Yl.conj().T[p-1, r-1]
            -np.tensordot(Yl[:, :], C_eLLLHA[:, :, r-1, t-1], axes=([0, 1], [1, 0]))*Yl.conj().T[p-1, s-1]
            +np.tensordot(Yl[:, :], C_eLLLHA[:, :, r-1, s-1], axes=([0, 1], [1, 0]))*Yl.conj().T[p-1, t-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHA[p-1, :, s-1, t-1], axes=([0], [0])) @ Yl.conj().T[:, r-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHA[p-1, r-1, :, t-1], axes=([0], [0])) @ Yl.conj().T[:, s-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHA[p-1, r-1, s-1, :], axes=([0], [0])) @ Yl.conj().T[:, t-1]
            +3*(Yl.conj().T @ Yl)[p-1, :] @ C_eLLLHA[:, r-1, s-1, t-1]
            -2*(Yl @ Yl.conj().T)[:, r-1] @ C_eLLLHM[p-1, :, s-1, t-1]
            +2*(Yl @ Yl.conj().T)[:, r-1] @ C_eLLLHM[p-1, :, t-1, s-1]
            +2*(Yl @ Yl.conj().T)[:, s-1] @ C_eLLLHM[p-1, r-1, :, t-1]
            -2*(Yl @ Yl.conj().T)[:, t-1] @ C_eLLLHM[p-1, r-1, :, s-1]
            +np.tensordot(Yl[:,:], Yl.conj().T[p-1,r-1]*(C_eLLLHM[:,:,s-1,t-1]-C_eLLLHM[:,:,t-1,s-1]),axes=([0, 1], [1, 0]))
            -np.tensordot(Yl[:,:], Yl.conj().T[p-1,s-1]*(C_eLLLHM[:,:,r-1,t-1]-C_eLLLHM[:,:,t-1,r-1]),axes=([0, 1], [1, 0]))
            +np.tensordot(Yl[:,:], Yl.conj().T[p-1,t-1]*(C_eLLLHM[:,:,r-1,s-1]-C_eLLLHM[:,:,s-1,r-1]),axes=([0, 1], [1, 0]))
            +1/2*np.tensordot(Yd[:,:], Yl.conj().T[p-1,r-1]*(C_dLQLH1[:,s-1,:,t-1]-C_dLQLH1[:,t-1,:,s-1]),axes=([0, 1], [1, 0]))
            -1/2*np.tensordot(Yd[:,:], Yl.conj().T[p-1,s-1]*(C_dLQLH1[:,r-1,:,t-1]-C_dLQLH1[:,t-1,:,r-1]),axes=([0, 1], [1, 0]))
            +1/2*np.tensordot(Yd[:,:], Yl.conj().T[p-1,t-1]*(C_dLQLH1[:,r-1,:,s-1]-C_dLQLH1[:,s-1,:,r-1]),axes=([0, 1], [1, 0]))
            +np.tensordot(Yd[:,:], Yl.conj().T[p-1,r-1]*(C_dLQLH2[:,s-1,:,t-1]-C_dLQLH2[:,t-1,:,s-1]),axes=([0, 1], [1, 0]))
            -np.tensordot(Yd[:,:], Yl.conj().T[p-1,s-1]*(C_dLQLH2[:,r-1,:,t-1]-C_dLQLH2[:,t-1,:,r-1]),axes=([0, 1], [1, 0]))
            +np.tensordot(Yd[:,:], Yl.conj().T[p-1,t-1]*(C_dLQLH2[:,r-1,:,s-1]-C_dLQLH2[:,s-1,:,r-1]),axes=([0, 1], [1, 0]))     
            -np.tensordot(Yu.conj().T[:,:], Yl.conj().T[p-1,r-1]*(C_QuLLH[:,:,s-1,t-1]-C_QuLLH[:,:,t-1,s-1]),axes=([0, 1], [1, 0]))
            +np.tensordot(Yu.conj().T[:,:], Yl.conj().T[p-1,s-1]*C_QuLLH[:,:,r-1,t-1],axes=([0, 1], [1, 0]))
            -np.tensordot(Yu.conj().T[:,:], Yl.conj().T[p-1,s-1]*C_QuLLH[:,:,t-1,r-1],axes=([0, 1], [1, 0]))
            -np.tensordot(Yu.conj().T[:,:], Yl.conj().T[p-1,t-1]*C_QuLLH[:,:,r-1,s-1],axes=([0, 1], [1, 0]))
            +np.tensordot(Yu.conj().T[:,:], Yl.conj().T[p-1,t-1]*C_QuLLH[:,:,s-1,r-1],axes=([0, 1], [1, 0]))                  
        )

####################################################################################################

    # 4F3M1: eLLLHM 
    indices = [
    (1,1,1,2), (1,1,2,2), (1,1,3,2), (1,1,1,3), (1,1,2,3), (1,1,3,3), (1,2,2,3), (1,2,3,3),
    (2,1,1,2), (2,1,2,2), (2,1,3,2), (2,1,1,3), (2,1,2,3), (2,1,3,3), (2,2,2,3), (2,2,3,3),
    (3,1,1,2), (3,1,2,2), (3,1,3,2), (3,1,1,3), (3,1,2,3), (3,1,3,3), (3,2,2,3), (3,2,3,3)  
    ]   

    for p, r, s, t in indices:
        Beta[f"eLLLHM_{p}{r}{s}{t}"] = (
            -3/2*(gp**2+g**2)*(Yl.conj().T[p-1, r-1]*C_DLDH1[s-1, t-1]-Yl.conj().T[p-1, t-1]*C_DLDH1[r-1, s-1])
            -1/6*(4*(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  -(C_DLDH1 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  -5*(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  +5*(C_DLDH1 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  +(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1]
                  -4*(C_DLDH1 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1])
            -3/4*gp**2*((Yl.conj().T)[p-1, r-1]*C_DLDH2[s-1, t-1]
                        -(Yl.conj().T)[p-1, t-1]*C_DLDH2[r-1, s-1])
            -1/12*(7*(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, s-1]*Yl.conj().T[p-1, r-1]
                  +5*(C_DLDH2 @ Yl @ Yl.conj().T)[t-1, r-1]*Yl.conj().T[p-1, s-1]
                  -2*(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, r-1]*Yl.conj().T[p-1, t-1]
                  +2*(C_DLDH2 @ Yl @ Yl.conj().T)[s-1, t-1]*Yl.conj().T[p-1, r-1]
                  -5*(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, t-1]*Yl.conj().T[p-1, s-1]
                  -7*(C_DLDH2 @ Yl @ Yl.conj().T)[r-1, s-1]*Yl.conj().T[p-1, t-1])
            +g*(5*Yl.conj().T[p-1, r-1]*C_LHW[s-1, t-1]
                  +Yl.conj().T[p-1, r-1]*C_LHW[t-1, s-1]
                  +4*Yl.conj().T[p-1, s-1]*C_LHW[r-1, t-1]
                  -4*Yl.conj().T[p-1, s-1]*C_LHW[t-1, r-1]
                  -Yl.conj().T[p-1, t-1]*C_LHW[r-1, s-1]
                  -5*Yl.conj().T[p-1, t-1]*C_LHW[s-1, r-1])
            -6*gp*(Yl.conj().T[p-1, r-1]*C_LHB[s-1, t-1]
                  +2*Yl.conj().T[p-1, s-1]*C_LHB[r-1, t-1]
                  +Yl.conj().T[p-1, t-1]*C_LHB[r-1, s-1])
            +3*(C_eLLLHS[p-1,:,s-1,t-1]@(Yl @ Yl.conj().T)[:, r-1]
               -C_eLLLHS[p-1,r-1,s-1,:]@(Yl @ Yl.conj().T)[:, t-1])
            +3/2*np.tensordot(Yl[:, :], C_eLLLHS[:, :, s-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            -3/2*np.tensordot(Yl[:, :], C_eLLLHS[:, :, r-1, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]            
            -C_eLLLHA[p-1, :, s-1, t-1] @ (Yl @ Yl.conj().T)[:, r-1]
            +2*C_eLLLHA[p-1, r-1, :, t-1] @ (Yl @ Yl.conj().T)[:, s-1]
            -C_eLLLHA[p-1, r-1, s-1, :] @ (Yl @ Yl.conj().T)[:, t-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHA[:, :, s-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            +np.tensordot(Yl[:, :], C_eLLLHA[:, :, r-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHA[:, :, r-1, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]
            +3*(Yl.conj().T @ Yl)[p-1, :] @ C_eLLLHM[:, r-1, s-1, t-1]
            -1/4*(9*gp**2+15*g**2-4*WH)*C_eLLLHM[p-1, r-1, s-1, t-1]
            +np.tensordot(Yl[:, :], C_eLLLHM[:, :, s-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            +np.tensordot(Yl[:, :], C_eLLLHM[:, :, r-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            -np.tensordot(Yl[:, :], C_eLLLHM[:, :, t-1, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            -np.tensordot(Yl[:, :], C_eLLLHM[:, :, s-1, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHM[p-1, :, s-1, t-1], axes=([0], [0])) @ Yl.conj().T[:, r-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHM[p-1, r-1, :, t-1], axes=([0], [0])) @ Yl.conj().T[:, s-1]
            +1/2*np.tensordot(Yl[:, :], C_eLLLHM[p-1, r-1, s-1, :], axes=([0], [0])) @ Yl.conj().T[:, t-1]
            +np.tensordot(Yd[:, :], C_dLQLH1[:, s-1, :, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            +1/2*np.tensordot(Yd[:, :], C_dLQLH1[:, t-1, :, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            +1/2*np.tensordot(Yd[:, :], C_dLQLH1[:, r-1, :, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            -1/2*np.tensordot(Yd[:, :], C_dLQLH1[:, t-1, :, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            -1/2*np.tensordot(Yd[:, :], C_dLQLH1[:, r-1, :, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]
            -np.tensordot(Yd[:, :], C_dLQLH1[:, s-1, :, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]            
            +1/2*np.tensordot(Yd[:, :], C_dLQLH2[:, s-1, :, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            -1/2*np.tensordot(Yd[:, :], C_dLQLH2[:, t-1, :, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            +np.tensordot(Yd[:, :], C_dLQLH2[:, r-1, :, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            -1/2*np.tensordot(Yd[:, :], C_dLQLH2[:, t-1, :, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            +1/2*np.tensordot(Yd[:, :], C_dLQLH2[:, r-1, :, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]
            -1/2*np.tensordot(Yd[:, :], C_dLQLH2[:, s-1, :, r-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, t-1]           
            -2*np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, s-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            -np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, t-1, s-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, r-1]
            -np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, r-1, t-1], axes=([0,1], [1,0])) * Yl.conj().T[p-1, s-1]
            +np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, t-1, r-1], axes=([0,1], [0,1])) * Yl.conj().T[p-1, s-1]
            +np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, r-1, s-1], axes=([0,1], [0,1])) * Yl.conj().T[p-1, t-1]
            +2*np.tensordot(Yu.conj().T[:, :], C_QuLLH[:, :, s-1, r-1], axes=([0,1], [0,1])) * Yl.conj().T[p-1, t-1]
        )

####################################################################################################

    #4F3M2: LdddHM
    indices = [
       (1,1,1,2), (1,2,1,2), (1,3,1,2), (1,1,1,3), (1,2,1,3), (1,3,1,3), (1,2,2,3), (1,3,2,3),
       (2,1,1,2), (2,2,1,2), (2,3,1,2), (2,1,1,3), (2,2,1,3), (2,3,1,3), (2,2,2,3), (2,3,2,3),
       (3,1,1,2), (3,2,1,2), (3,3,1,2), (3,1,1,3), (3,2,1,3), (3,3,1,3), (3,2,2,3), (3,3,2,3) 
        ]
    
    for p, r, s, t in indices:
        Beta[f"LdddHM_{p}{r}{s}{t}"] = (
            1/6*(Yu.conj().T @ Yd)[:, r-1] @ C_LdudH[p-1,t-1,:,s-1]
            -1/6*(Yu.conj().T @ Yd)[:, r-1] @ C_LdudH[p-1,s-1,:,t-1]
            -1/6*(Yu.conj().T @ Yd)[:, s-1] @ C_LdudH[p-1,t-1,:,r-1]
            -1/3*(Yu.conj().T @ Yd)[:, s-1] @ C_LdudH[p-1,r-1,:,t-1]
            +1/6*(Yu.conj().T @ Yd)[:, t-1] @ C_LdudH[p-1,s-1,:,r-1]
            +1/3*(Yu.conj().T @ Yd)[:, t-1] @ C_LdudH[p-1,r-1,:,s-1]            
            -1/12*(13*gp**2+27*g**2+48*gs**2-12*WH)*C_LdddHM[p-1,r-1,s-1,t-1]
            +2*C_LdddHM[p-1,:,s-1,t-1] @ (Yd.conj().T @ Yd)[:, r-1]
            +2*C_LdddHM[p-1,r-1,:,t-1] @ (Yd.conj().T @ Yd)[:, s-1]
            +2*C_LdddHM[p-1,r-1,s-1,:] @ (Yd.conj().T @ Yd)[:, t-1]
            +5/2*C_LdddHM[:,r-1,s-1,t-1] @ (Yl @ Yl.conj().T)[p-1, :]
            +1/2*np.tensordot(Yl[p-1, :], C_eddDd[:, :, r-1, s-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, t-1]
            -1/2*np.tensordot(Yl[p-1, :], C_eddDd[:, :, r-1, t-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, s-1]           
            -1/9*(5*gp**2-12*gs**2)*(C_LQdDd[p-1, :, t-1, r-1] @ Yd[:, s-1]-C_LQdDd[p-1, :, s-1, r-1] @ Yd[:, t-1])            
            +1/2*np.tensordot(C_LQdDd[p-1, :, :, t-1], Yd[:, s-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, r-1]
            +1/2*np.tensordot(C_LQdDd[p-1, :, :, t-1], Yd[:, r-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, s-1]            
            -1/2*np.tensordot(C_LQdDd[p-1, :, :, s-1], Yd[:, t-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, r-1]
            -1/2*np.tensordot(C_LQdDd[p-1, :, :, s-1], Yd[:, r-1], axes=([0], [0])) @ (Yd.conj().T @ Yd)[:, t-1]
            +1/3*np.tensordot((Yl @ Yl.conj().T)[p-1, :], C_LQdDd[:, :, r-1, s-1], axes=([0], [0])) @ Yd[:, t-1]
            -1/3*np.tensordot((Yl @ Yl.conj().T)[p-1, :], C_LQdDd[:, :, r-1, t-1], axes=([0], [0])) @ Yd[:, s-1]
        )    
####################################################################################################    
    return Beta































































