# %%

# RADII AND MASSES CALCULATOR
# Stefan-Boltzmann and Schweitzer et al. 2019.
# Cifuentes et al. (2021)

import numpy as np
import pandas as pd

# Data

input_name = 'cif21.multiplicity'
output_name = input_name + '_MR.csv'

df = pd.read_csv(input_name+'.csv')

ID = df['ID_star']
Lbol = df['Lbol']
e_Lbol = df['Lberr']
Teff = df['Teff']
e_Teff = df['e_Teff']

# Radius (Stefan-Boltzmann) and Mass (Schweitzer+19)

def Radius_SB(Lbol, Lberr, Teff, eTeff):
    """Stellar radius and its error from the Stefan–Boltzmann law under the black body approximation.

    Args:
        Lbol (float): Bolometric luminosity in solar units.
        Lberr (float): Bolometric luminosity uncertainty in solar units.
        Teff (float): Effective temperature in Kelvin.
        eTeff (float): Effective temperature uncertainty in Kelvin.

    Returns:
        float: Stellar radius in solar units.
        float: Stellar radius error in solar units.

    Nominal solar values from the IAU B.3 resolution
    on recommended nominal conversion constants for selected solar and planetary properties:
    https://www.iau.org/static/resolutions/IAU2015_English.pdf

    Nominal solar luminosity: 3.828 x 10+26 W (exact)
    Nominal solar radius: 6.957 x 10+8 m (exact)

    Stefan-Boltzmann constant value from 2018 CODATA recommended values:
    https://physics.nist.gov/cuu/pdf/wall_2018.pdf

    Stefan-Boltzman constant, sigma: 5.670 374 419 x 10-8 W m-2 K-4 (exact)

    """
    Lsun = 3.828*1e26
    Rsun = 6.957*1e8
    sigma = 5.670374419*1e-8
    R = 1/Rsun * np.sqrt(Lbol*Lsun/(4*np.pi*sigma*Teff**4))
    eR = R * np.sqrt((Lberr/(2*Lbol))**2 + (-2*eTeff/Teff)**2)
    return(R, eR)


def Mass_sch19(Radius, eRadius):
    """Stellar mass and its error from the empirical relation by Schweitzer et al. 2019 
    (2019A&A...625A..68S), based on masses and radii of eclipsing binaries.

    Args:
        Radius (float): Stellar radius in solar units.
        eRadius (float): Stellar radius uncertainty in solar units.

    Returns:
        float: Stellar mass in solar units.
        float: Stellar mass error in solar units.

    (See Equation 6 in Schweitzer et al. 2019 and references therein).
    """
    a = -0.024048024
    b = 1.0552427
    a_err = 0.007592668
    b_err = 0.017044148
    M = a + b * Radius
    eM = np.sqrt((a_err)**2 + (Radius * b_err)**2 + (b * eRadius)**2)
    return(M, eM)

# Results

Radius = Radius_SB(Lbol, e_Lbol, Teff, e_Teff)
Mass = Mass_sch19(Radius[0], Radius[1])

# Write out

MR = pd.DataFrame(
    {'ID': ID, 'R': Radius[0], 'eR': Radius[1], 'M': Mass[0], 'eM': Mass[1]})
output = pd.concat([df, MR], axis=1)
output.to_csv(output_name, sep=',', encoding='utf-8')
