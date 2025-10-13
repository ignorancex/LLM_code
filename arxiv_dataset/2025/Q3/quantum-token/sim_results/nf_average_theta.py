"""
L.Tsunaki et al., 'Ensemble-Based Quantum-Token Protocol Benchmarked on IBM Quantum Processors', arXiv:2412.08530 (2024).

This script calculates the average number of qubits that the bank from a forged token as function of theta in IBMQ Kyiv, Sherbrooke Brisbane.
"""

import os
os.environ["OMP_NUM_THREADS"] = "40"

import numpy as np

def na_analytical(theta_1, theta_2, phi_1, phi_2, c):
    """
    Analytical expression of the probability of the faker to measure the token in the |0> state, as defined in the paper.

    Parameters:
    theta_1 (float): The angle of the first qubit
    theta_2 (float): The angle of the second qubit
    phi_1 (float): The angle of the first qubit
    phi_2 (float): The angle of the second qubit

    Returns:
    float: The probability of the faker to measure the token in the |0> state
    """
    return 1/2 + c/2*(np.cos(theta_1)*np.cos(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1 - phi_2))

def calculate_phi_f(na, theta_a, phi_a, c):
    """
    Calculated the forged angles based on the input parameters

    Parameters:
    na (float): Fraction of qubits measured in the |0> state by the attacker
    theta_a (float): Attacker measurement polar angle
    phi_a (float): Attacker measurement azimuthal angle
    c (float): Contrast of the hardware

    Returns:
    float: The polar angle of the forged qubit
    float: The azimuthal angle of the forged qubit    
    """

    alpha = (2*na-1)/c

    if theta_a == 0 or theta_a == np.pi:
        phi_f = np.random.uniform(0, 2*np.pi)
        if alpha/np.cos(theta_a) <-1:
            theta_f = np.pi
        elif alpha/np.cos(theta_a) > 1:
            theta_f = 0
        else:
            theta_f = np.arccos(alpha/np.cos(theta_a))

    else:
        delta = alpha**2*np.cos(theta_a)**2 - alpha**2 - np.cos(2*theta_a)/2 + 1/2

        if delta < 0:
            # print(f"No solution")
            return np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)

        zf_max = alpha*np.cos(theta_a) + np.sqrt(delta)
        if zf_max > 1:
            theta_fmax = np.pi
        else:
            theta_fmax = np.arccos(zf_max)
                
        zf_min = alpha*np.cos(theta_a) - np.sqrt(delta)
        if zf_min < -1:
            theta_fmin = 0
        else:
            theta_fmin = np.arccos(zf_min)
        
        theta_f = np.random.uniform(theta_fmin, theta_fmax)

        arg = (alpha - np.cos(theta_a)*np.cos(theta_f))/np.sin(theta_a)/np.sin(theta_f)
        if np.abs(arg) > 1:
            phi_f = np.random.uniform(0, 2*np.pi)
            # print('No solution')
        else:
            phi_f = np.random.choice([phi_a - np.arccos(arg), phi_a + np.arccos(arg)])

    return theta_f, phi_f

thetab = np.linspace(1e-3, np.pi-1e-3, 100)
c_kyiv = 0.9503942133692314
c_sherbrooke = 0.9859975714946958
c_brisbane = 0.843234615325578
nf = np.empty((3, len(thetab)), dtype=float)
avg=100000

for itr_zb in range(len(thetab)):

    nf_phib_calc_k = 0
    nf_phib_calc_s = 0
    nf_phib_calc_b = 0      
    for itr_phib in range(avg):
        phi_b = np.random.uniform(0, 2*np.pi)
        theta_a = np.random.uniform(0, np.pi)
        phi_a = np.random.uniform(0, 2*np.pi)

        na_k = na_analytical(thetab[itr_zb], theta_a, phi_b, phi_a, c_kyiv)
        na_s = na_analytical(thetab[itr_zb], theta_a, phi_b, phi_a, c_sherbrooke)
        na_b = na_analytical(thetab[itr_zb], theta_a, phi_b, phi_a, c_brisbane)

        theta_f_k, phi_f_k = calculate_phi_f(na_k, theta_a, phi_a, c_kyiv)
        theta_f_s, phi_f_s = calculate_phi_f(na_s, theta_a, phi_a, c_sherbrooke)
        theta_f_b, phi_f_b = calculate_phi_f(na_b, theta_a, phi_a, c_brisbane)
        nf_phib_calc_k += na_analytical(theta_f_k, thetab[itr_zb], phi_f_k, phi_b, c_kyiv)
        nf_phib_calc_s += na_analytical(theta_f_s, thetab[itr_zb], phi_f_s, phi_b, c_sherbrooke)
        nf_phib_calc_b += na_analytical(theta_f_s, thetab[itr_zb], phi_f_s, phi_b, c_brisbane)

    nf[0, itr_zb] = nf_phib_calc_k/avg
    nf[1, itr_zb] = nf_phib_calc_s/avg
    nf[2, itr_zb] = nf_phib_calc_b/avg

    print(f'{itr_zb+1}/{len(thetab)}')

np.savetxt('./nf_average_kyiv.txt', nf[0], delimiter='\t')
np.savetxt('./nf_average_sherbrooke.txt', nf[1], delimiter='\t')
np.savetxt('./nf_average_brisbane.txt', nf[2], delimiter='\t')