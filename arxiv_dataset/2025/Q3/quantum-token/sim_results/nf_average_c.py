"""
L.Tsunaki et al., 'Ensemble-Based Quantum-Token Protocol Benchmarked on IBM Quantum Processors', arXiv:2412.08530 (2024).

This script calculates the average number of qubits that the bank from a forged token as a function of the IBMQ contrast.
"""

import os

os.environ["OMP_NUM_THREADS"] = "32"

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
c = np.arange(.1, 1.1, .1)
nf = np.empty((len(c), len(thetab)), dtype=float)

for itr_c in range(len(c)):
    for itr_zb in range(len(thetab)):

        nf_phib = 0       
        for itr_phib in range(100000):
            phi_b = np.random.uniform(0, 2*np.pi)
            theta_a = np.random.uniform(0, np.pi)
            phi_a = np.random.uniform(0, 2*np.pi)

            na = na_analytical(thetab[itr_zb], theta_a, phi_b, phi_a, c[itr_c])
            theta_f, phi_f = calculate_phi_f(na, theta_a, phi_a, c[itr_c])
            nf_phib += na_analytical(theta_f, thetab[itr_zb], phi_f, phi_b, c[itr_c])

        nf[itr_c, itr_zb] = nf_phib/100000
        print(f"c = {c[itr_c]}, theta_b = {thetab[itr_zb]}, nf = {nf[itr_c, itr_zb]}")

np.savetxt('./nf_average_c.txt', nf, delimiter='\t')