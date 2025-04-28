import numpy as np
import constants as c
from scipy.optimize import bisect
from scipy.interpolate import interp1d
from scipy.misc import derivative

'''
Various functions for probability density function transformations:
'''

#====================================================================================================
# GENERAL FUNCTIONS

#----------------------------------------------------------------------------------------------------
# Check for normalization
def check_norm(x, f_x, atol=1.0e-4):
    chknorm = np.trapz(y=f_x, x=x)
    if (np.fabs(chknorm - 1.0) > atol):
        print("\nWARNING: f_x(x) normalization = ", chknorm)
        return False
    else:
        return True

#----------------------------------------------------------------------------------------------------
# Class for generating a Python object f_x with a .pdf attribute...
# ...given the interpolation function f_x_interp
class generate_f_x():
    def __init__(self, f_x_interp):
        self.f_x_interp = f_x_interp
    def pdf(self, x):
        f_x = self.f_x_interp(x)
        return f_x

#====================================================================================================
# FUNCTIONS: SPIN <--> ISCO

#----------------------------------------------------------------------------------------------------
# Convert a black hole spin (a [-]) to an ISCO location (risco [Rg])
# NOTE: Input "a" must be a single value and -1 <= a <= 1
def spin2risco(a):
    
    # Check that -1 <= a <= 1
    if not np.logical_and(a>=-1.0, a<=1.0):
        print("\nWARNING: Input black hole spin must satisfy -1 <= a <= 1\n")
        if   (a >=  1.0): risco = 1.0; return risco
        elif (a <= -1.0): risco = 9.0; return risco
        else:             quit()
        
    # Calculate risco
    Z1 = 1.0 + (1.0 - a**2.0)**(1.0/3.0) * ((1.0 + a)**(1.0/3.0) + (1.0 - a)**(1.0/3.0))
    Z2 = (3.0 * a**2.0 + Z1**2.0)**(0.5)
    if (a >= 0.0):
        risco = 3.0 + Z2 - ((3.0 - Z1)*(3.0 + Z1 + 2.0*Z2))**(0.5)
    if (a < 0.0):
        risco = 3.0 + Z2 + ((3.0 - Z1)*(3.0 + Z1 + 2.0*Z2))**(0.5)
    return risco

#----------------------------------------------------------------------------------------------------
# Convert an ISCO location (risco [Rg]) to a black hole spin (a [-])
# NOTE: Input "risco" must be a single value and 1 <= risco <= 9
def risco2spin(risco):

    # Check that 1 <= risco <= 9
    if not np.logical_and(risco>=1.0, risco<=9.0):
        print("\nWARNING: Input ISCO location must satisfy 1 <= risco <= 9\n")
        if   (risco >= 9.0): a = -1.0; return a
        elif (risco <= 1.0): a =  1.0; return a
        else:                quit()
        
    # Function needed by bisect()
    a_min, a_max = -1.0, 1.0
    def func_risco2spin(a, *args):
        risco = args[0]
        zero  = risco - spin2risco(a=a)
        return zero
    
    # Calculate a
    a = bisect(f=func_risco2spin, a=a_min, b=a_max, args=(risco,))
    return a

#----------------------------------------------------------------------------------------------------
# Calculate the Jacobian of the transformation: risco --> a
# NOTE: Input "a" must be a single value and -1 <= a <= 1
def calc_J_risco2spin(a):

    # Check that -1 <= a <= 1
    if not np.logical_and(a>=-1.0, a<=1.0):
        print("\nERROR: Input black hole spin must satisfy -1 <= a <= 1\n"); quit()

    # Calculate J = drisco/da
    Z1     = 1.0 + (1.0 - a**2)**(1.0/3.0) * ((1.0 + a)**(1.0/3.0) + (1.0 - a)**(1.0/3.0))
    Z2     = np.sqrt(3.0 * a**2 + Z1**2)
    dZ1_da = (1.0/3.0) * ((1.0 + a)**(-2.0/3.0) - (1.0 - a)**(-2.0/3.0)) * (1.0 - a**2)**(1.0/3.0) \
           - (2.0/3.0) * a * ((1.0 + a)**(1.0/3.0) + (1.0 - a)**(1.0/3.0)) * (1.0 - a**2)**(-2.0/3.0)
    dZ2_da = (3.0 * a + Z1 * dZ1_da) / np.sqrt(3.0 * a**2 + Z1**2)
    if (a >= 0.0):
        J = dZ2_da - ((3.0 - Z1) * dZ2_da - (Z1 + Z2) * dZ1_da) \
          / np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    if (a < 0.0):
        J = dZ2_da + ((3.0 - Z1) * dZ2_da - (Z1 + Z2) * dZ1_da) \
          / np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    return J

#----------------------------------------------------------------------------------------------------
# Convert f_o(risco) to f_a(a)
def transform_frisco2fspin(f_risco, Na=1000):
    '''
    f_risco - Marginal density for ISCO <-- Must be an object with a .pdf attribute
    '''
    # Construct the black hole spin array
    a_min, a_max = -1.0, 1.0
    a = np.linspace(a_min, a_max, Na+2)[1:-1]
    
    # Transform f_risco to f_a
    f_a = np.zeros(Na)
    for i in np.arange(Na):
        risco  = spin2risco(a=a[i])              # Inverse transformation function
        J      = calc_J_risco2spin(a=a[i])       # Jacobian of the transformation
        f_a[i] = f_risco.pdf(risco) * np.abs(J)  # Black hole spin marginal density

    # Normalization check
    chknorm = check_norm(x=a, f_x=f_a)
    if (chknorm is False):
        print("\n...renormalizing the black hole spin marginal density.\n")
        f_a /= np.trapz(y=f_a, x=a)

    # Create a function that interpolates f_a
    f_a_interp = interp1d(x=a, y=f_a, kind='linear', fill_value=(0.0,0.0), bounds_error=False)

    # Using the interpolated f_a, generate the f_a object with a .pdf attribute
    f_a = generate_f_x(f_x_interp=f_a_interp)

    return f_a
    
#----------------------------------------------------------------------------------------------------
# Convert f_a(a) to f_o(risco)
def transform_fspin2frisco(f_a, Nrisco=1000):
    '''
    f_a - Marginal density for the black hole spin <-- Must be an object with a .pdf attribute
    '''
    # Construct the ISCO array
    risco_min, risco_max = 1.0, 9.0
    risco = np.linspace(risco_min, risco_max, Nrisco+2)[1:-1]

    # Transform f_a to f_risco
    f_risco = np.zeros(Nrisco)
    for i in np.arange(Nrisco):
        a          = risco2spin(risco=risco[i])                         # Inverse transformation function
        J          = derivative(func=risco2spin, x0=risco[i], dx=1e-6)  # Jacobian of the transformation (numerical derivative)
        f_risco[i] = f_a.pdf(a) * np.abs(J)                             # Black hole spin marginal density

    # Normalization check
    chknorm = check_norm(x=risco, f_x=f_risco)
    if (chknorm is False):
        print("\n...renormalizing the ISCO marginal density.\n")
        f_risco /= np.trapz(y=f_risco, x=risco)

    # Create a function that interpolates f_risco
    f_risco_interp = interp1d(x=risco, y=f_risco, kind='linear', fill_value=(0.0,0.0), bounds_error=False)

    # Using the interpolated f_risco, generate the f_risco object with a .pdf attribute
    f_risco = generate_f_x(f_x_interp=f_risco_interp)

    return f_risco
    
#====================================================================================================
'''
# Calculate the limb-darkening parameter
def calc_Yld(inc):
    Yld = 0.5 + 0.75 * np.cos(inc)
    return Yld

# Calculate the disk flux normalization [-] (all inputs and output in cgs units)
def calc_K(r, fcol, M, D, i, Yld=1.0, gGR=1.0):
    K = r**2.0 / fcol**4.0 * (c.G * M / c.c**2.0)**2.0 / D**2.0 * np.cos(i) * Yld * gGR
    return K
'''
#====================================================================================================

