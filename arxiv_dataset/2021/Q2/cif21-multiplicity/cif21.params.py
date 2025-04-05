# %%
# MULTIPLICITY PARAMETERS
# C. Cifuentes San Roman (2021)

# Absolute magnitudes
# Physical separation (s)
# theta
# µ ratio
# Positional angle difference (PA)
# Distance difference (∆d)
# Mass from Lbol, MG, MJ (Pecaut et al. 2013)

# 'B' denotes any component different from the primary.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from __future__ import print_function, division
from PyAstronomy import pyasl
from matplotlib import rc
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator

rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
rc('text', usetex=False)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


# Data

input_name = 'cif21.multiplicity'
output_name = input_name + '_out' + '.csv'
pec13_name = 'Pec13.csv'

df = pd.read_csv(input_name+'.csv')
pec13 = pd.read_csv('Data/'+pec13_name)

# Variables

ID = df['ID_star']
ra_A = df['ra_A']
ra_B = df['ra_B']
ra_C = df['ra_C']
dec_A = df['dec_A']
dec_B = df['dec_B']
dec_C = df['dec_C']
pmra_A = df['pmra_A']
pmra_B = df['pmra_B']
pmra_C = df['pmra_C']
pmdec_A = df['pmdec_A']
pmdec_B = df['pmdec_B']
pmdec_C = df['pmdec_C']
d_A = df['d_A']
d_B = df['d_B']
d_C = df['d_C']
rho_AB = df['rho_AB']
rho_AC = df['rho_AC']
Mass_A = df['Mass_A']
Mass_B = df['Mass_B']
Mass_C = df['Mass_C']
Lbol = df['Lbol']
G_mag = df['phot_g_mean_mag']
J_mag = df['Jmag']
parallax = df['parallax']

# %%
# Function definitions


def Mabs(mag, parallax):
    Mabs = mag - 5*np.log10(1000/parallax) + 5
    return Mabs


def s(d_A, rho_AB):
    s_AB = d_A * rho_AB
    return s_AB


def theta(ra_A, dec_A, ra_B, dec_B):
    theta_AB = pyasl.positionAngle(ra_A, dec_A, ra_B, dec_B)
    return theta_AB


def muratio(pmra_A, pmra_B, pmdec_A, pmdec_B):
    muratio_AB = np.sqrt(
        ((pmra_A-pmra_B)**2 + (pmdec_A-pmdec_B)**2)/(pmra_A**2 + pmdec_A**2))
    return muratio_AB


def deltaPA(pmra_A, pmdec_A, pmra_B, pmdec_B):
    deltaPA = np.rad2deg(
        np.abs(np.arctan(pmdec_A/pmra_A) - np.arctan(pmdec_B/pmra_B)))
    return deltaPA


def deltad(d_A, d_B):
    deltad = np.abs((d_A - d_B)/d_A)
    return deltad


def Mass_L(Lbol, plot=0):
    """
    Creates a model to derive the mass from the bolometric luminosity in solar units.
    plot = 1 shows statistics and plotting.
    """
    SpT_min, SpT_max = [25, 83]  # Set range of SpT
    y = pec13[(pec13['SpTnum'] >= SpT_min) & (pec13['SpTnum'] < SpT_max)]['M']
    x = pec13[(pec13['SpTnum'] >= SpT_min) & (
        pec13['SpTnum'] < SpT_max)]['logL']
    # Fit
    k = 10
    p, cov = np.polyfit(x, y, k, cov=True)
    model = np.poly1d(p)
    xp = np.linspace(min(x), max(x), 1000)
    yp = np.polyval(p, xp)
    # Standard-deviation estimates for each coefficient
    perr = np.sqrt(np.diag(cov))
    # Coefficient of correlation between x and y
    R2 = np.corrcoef(y, model(x))[0, 1]**2
    residuals = y - model(x)
    n = len(y)
    m = p.size
    dof = n - m
    chi_squared = np.sum((residuals/model(x))**2)
    chi_squared_red = chi_squared/(dof)
    sdev_err = np.sqrt(np.sum(residuals**2)/(dof))
    #
    if plot == 1:
        # Statistics
        print('Polynomial fitting:\n')
        print('degree =', k, '\nR2 =', np.round(R2, 5), '\ncoeffs a-c =',
              p, '\nerr_coeffs a-c =', perr)
        print('---')
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(x, y, c='r')
        plt.plot(xp, yp, 'b-')
        ax.set_ylabel('Msol')
        ax.set_xlabel('Lsol')
        plt.show()
    #
    logL = np.log10(Lbol)
    Mass = model(logL)
    return Mass


def Mass_G(G_mag, parallax, plot=0):
    """
    Creates a model to derive the mass from the abdolute magnitude in G, in solar units.
    plot = 1 shows statistics and plotting.
    """
    SpT_min, SpT_max = [29, 83]  # Set range of SpT
    y = pec13[(pec13['SpTnum'] >= SpT_min) & (pec13['SpTnum'] < SpT_max)]['M']
    x = pec13[(pec13['SpTnum'] >= SpT_min)
              & (pec13['SpTnum'] < SpT_max)]['M_G']
    # Fit
    k = 14
    p, cov = np.polyfit(x, y, k, cov=True)
    model = np.poly1d(p)
    xp = np.linspace(min(x), max(x), 1000)
    yp = np.polyval(p, xp)
    # Standard-deviation estimates for each coefficient
    perr = np.sqrt(np.diag(cov))
    # Coefficient of correlation between x and y
    R2 = np.corrcoef(y, model(x))[0, 1]**2
    residuals = y - model(x)
    n = len(y)
    m = p.size
    dof = n - m
    chi_squared = np.sum((residuals/model(x))**2)
    chi_squared_red = chi_squared/(dof)
    sdev_err = np.sqrt(np.sum(residuals**2)/(dof))
    #
    if plot == 1:
        # Statistics
        print('Polynomial fitting:\n')
        print('degree =', k, '\nR2 =', np.round(R2, 5), '\ncoeffs a-c =',
              p, '\nerr_coeffs a-c =', perr)
        print('---')
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(x, y, c='r')
        plt.plot(xp, yp, 'b-')
        ax.set_ylabel('Msol')
        ax.set_xlabel('Lsol')
        plt.show()
    #
    MG = G_mag - 5*np.log10(1000/parallax) + 5
    Mass = model(MG)
    return Mass


def Ug(M1, M2, rho, d_pc):
    """  Calculates the binding energy in joules (See Sec. 3.2 in Caballero et al. 2009).
    The '*' in Ug* denotes the approximation r ~ s for small angles.

    Args:
        M1 (float): Mass of the primary in solar units.
        M2 (float): Mass of the secondary in solar units.
        rho (float): angular separation in arcsec. 
        d_pc (float): distance in parsec.

    Returns:
        float: Binding energy in joules.

    Nominal solar values from the IAU B2 (AU) and B3 (GM) resolutions.
    Gravitational constant value from 2018 CODATA recommended values.
    """
    G = 6.67408*1E-11  # m3 kg−1 s−2
    Msol = 1.98847*1E30  # kg
    GM = 1.3271244*1E20  # m3 s-2
    s = rho * d_pc  # AU
    AU = 149597870700  # m/AU
    Ug = -G * (M1*M2 * Msol**2)/(s*AU)  # kg2 m2 s-2 = kg J (1 J = kg m2 s-2)
    return Ug


def P(M1, rho, d_pc):
    """Calculates the orbital period in years.

    Args:
        M1 (float): Mass of the primary in solar units.
        rho (float): angular separation in arcsec. 
        d_pc (float): distance in parsec.

    Returns:
        float: orbital period in years.

    Nominal solar values from the IAU B2 (AU) and B3 (GM) resolutions.
    Gravitational constant value from 2018 CODATA recommended values.
    """
    G = 6.67408*1E-11  # m3 kg−1 s−2
    Msol = 1.98847*1E30  # kg
    GM = 1.3271244*1E20  # m3 s-2
    s = rho * d_pc  # AU
    AU = 149597870700  # m/AU
    year = 1/(31.5576*1E6)  # 1 year = 365.25 days (31.5576 million seconds)
    Porb = 2*np.pi * np.sqrt((s*AU)**3 / (GM * M1))*year  # a
    return Porb


# %%
# Parameter calculation

MG = []
MJ = []
s_AB = []
s_AC = []
theta_AB = []
theta_AC = []
muratio_AB = []
muratio_AC = []
deltaPA_AB = []
deltaPA_AC = []
deltad_AB = []
deltad_AC = []
Mass_Lbol = []
Mass_MG = []
Ug_AB = []
Ug_AC = []
Porb_AB = []
Porb_AC = []

#

for i in range(len(G_mag)):
    MG.append(Mabs(G_mag[i], parallax[i]))

for i in range(len(J_mag)):
    MJ.append(Mabs(J_mag[i], parallax[i]))

for i in range(len(d_A)):
    s_AB.append(s(d_A[i], rho_AB[i]))
    s_AC.append(s(d_A[i], rho_AC[i]))

for i in range(len(ra_A)):
    theta_AB.append(theta(ra_A[i], dec_A[i], ra_B[i], dec_B[i]))
    theta_AC.append(theta(ra_A[i], dec_A[i], ra_C[i], dec_C[i]))

for i in range(len(ra_A)):
    muratio_AB.append(
        muratio(pmra_A[i], pmra_B[i], pmdec_A[i], pmdec_B[i]))
for i in range(len(ra_A)):
    muratio_AC.append(
        muratio(pmra_A[i], pmra_C[i], pmdec_A[i], pmdec_C[i]))

for i in range(len(pmra_A)):
    deltaPA_AB.append(deltaPA(pmra_A[i], pmdec_A[i], pmra_B[i], pmdec_B[i]))
    deltaPA_AC.append(deltaPA(pmra_A[i], pmdec_A[i], pmra_C[i], pmdec_C[i]))

for i in range(len(d_A)):
    deltad_AB.append(deltad(d_A[i], d_B[i]))
    deltad_AC.append(deltad(d_A[i], d_C[i]))

for i in range(len(Lbol)):
    Mass_Lbol.append(Mass_L(Lbol[i]))

for i in range(len(G_mag)):
    Mass_MG.append(Mass_G(G_mag[i], parallax[i]))

for i in range(len(Mass_A)):
    Ug_AB.append(Ug(Mass_A[i], Mass_B[i], rho_AB[i], d_A[i]))
    Ug_AC.append(Ug(Mass_A[i], Mass_C[i], rho_AC[i], d_A[i]))

for i in range(len(Mass_A)):
    Porb_AB.append(P(Mass_A[i], rho_AB[i], d_A[i]))
    Porb_AC.append(P(Mass_A[i], rho_AC[i], d_A[i]))

# %%
# Write out

parameters = pd.DataFrame({'ID': ID, 'theta_AB': theta_AB, 'theta_AC': theta_AC, 'muratio_AB': muratio_AB, 'muratio_AC': muratio_AC, 'deltaPA_AB': deltaPA_AB,
                           'deltaPA_AC': deltaPA_AC, 'deltad_AB': deltad_AB, 'deltad_AC': deltad_AC, 's_AB': s_AB, 's_AC': s_AC, 'MG': MG, 'MJ': MJ,
                           'Mass_Lbol': Mass_Lbol, 'Mass_MG': Mass_MG, 'Ug_AB': Ug_AB, 'Ug_AC': Ug_AC, 'Porb_AB': Porb_AB, 'Porb_AC': Porb_AC})
output = pd.concat([df, parameters], axis=1)
output.to_csv(output_name, sep=',', encoding='utf-8')
