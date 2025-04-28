import numpy as np
import constants as c
import asciitable
import h5py
import pickle
from scipy.stats import truncnorm, expon
from scipy.interpolate import interp1d, RegularGridInterpolator, RectBivariateSpline #interp2d
from scipy import ndimage

'''
Various functions for collecting results from published disk continnum fitting measurements.
'''

#====================================================================================================
# Class for generating a Python object f_x with a .pdf attribute...
# ...given the interpolation function f_x_interp
class generate_f_x():
    def __init__(self, f_x_interp):
        self.f_x_interp = f_x_interp
    def pdf(self, x):
        f_x = self.f_x_interp(x)
        return f_x

#====================================================================================================
# Check for normalization
def check_norm(x, f_x, atol=1.0e-4):
    chknorm = np.trapz(y=f_x, x=x)
    if (np.fabs(chknorm - 1.0) > atol):
        print("\nWARNING: f_x(x) normalization = ", chknorm)
        return False
    else:
        return True

#====================================================================================================
# Return the min/max x-values located (Nstd * x_err) away from x_val
def get_minmax(x_val, x_err, Nstd):
    x_min = x_val - Nstd * x_err
    x_max = x_val + Nstd * x_err
    return x_min, x_max

#====================================================================================================
# Collect the relativistic disk flux correction factors from the CF pickle file

#----------------------------------------------------------------------------------------------------
# gGR(r,i) <-- accounts for photon propagation
def get_gGR(fpCF, sigma_r=0, sigma_i=0):
    
    # Load in the table of gGR(r,i) factors...
    with open(fpCF, 'rb') as fp:
        fpick    = pickle.load(fp)
        r_grid   = fpick['r_grid']  # [Rg]
        i_grid   = fpick['i_grid']  # [rad]
        gGR_grid = fpick['gGR']     # gGR(r,i)
    # ...smooth gGR(r,i) using a Gaussian kernel
    gGR_filt = ndimage.filters.gaussian_filter(gGR_grid, sigma=[sigma_r, sigma_i*c.deg2rad], mode='nearest')
    # ...create the gGR(r,i) interpolation function
    gGR = RectBivariateSpline(x=r_grid, y=i_grid, z=gGR_filt, kx=1, ky=1, s=0)
    #gGR = interp2d(x=r_grid, y=i_grid, z=gGR_filt, kind='linear', fill_value=(0.0,0.0), bounds_error=False)
    return gGR

#----------------------------------------------------------------------------------------------------
# d/dr[gGR(r,i)] <-- inner disk radius partial derivative of gGR
def get_dgGR_dr(fpCF, sigma_r=0, sigma_i=0):
    
    # Load in the table of gGR(r,i) factors...
    with open(fpCF, 'rb') as fp:
        fpick    = pickle.load(fp)
        r_grid   = fpick['r_grid']  # [Rg]
        i_grid   = fpick['i_grid']  # [rad]
        gGR_grid = fpick['gGR']     # gGR(r,i)
    # ...calculate the partial derivative d/dr[gGR(r,i)], where the r-grid spacing is uniform
    dr      = r_grid[1] - r_grid[0]
    di      = i_grid[1] - i_grid[0]
    dgGR_dr = np.gradient(gGR_grid, *[dr, di], edge_order=2)[0]
    # ...smooth d/dr[gGR(r,i)] using a Gaussian kernel
    dgGR_dr_filt = ndimage.filters.gaussian_filter(dgGR_dr, sigma=[sigma_r, sigma_i*c.deg2rad], mode='nearest')
    # ...create the d/dr[gGR(r,i)] interpolation function
    dgGR_dr = RectBivariateSpline(x=r_grid, y=i_grid, z=dgGR_dr_filt, kx=1, ky=1, s=0)
    #dgGR_dr = interp2d(x=r_grid, y=i_grid, z=dgGR_dr_filt, kind='linear', fill_value=(0.0,0.0), bounds_error=False)
    return dgGR_dr

#----------------------------------------------------------------------------------------------------
# d/di[gGR(r,i)] <-- inner disk inclination partial derivative of gGR
def get_dgGR_di(fpCF, sigma_r=0, sigma_i=0):
    
    # Load in the table of gGR(r,i) factors...
    with open(fpCF, 'rb') as fp:
        fpick    = pickle.load(fp)
        r_grid   = fpick['r_grid']  # [Rg]
        i_grid   = fpick['i_grid']  # [rad]
        gGR_grid = fpick['gGR']     # gGR(r,i)
    # ...calculate the partial derivative d/di[gGR(r,i)], where the i-grid spacing is uniform
    dr      = r_grid[1] - r_grid[0]
    di      = i_grid[1] - i_grid[0]
    dgGR_di = np.gradient(gGR_grid, *[dr, di], edge_order=2)[1]
    # ...smooth d/di[gGR(r,i)] using a Gaussian kernel
    dgGR_di_filt = ndimage.filters.gaussian_filter(dgGR_di, sigma=[sigma_r, sigma_i*c.deg2rad], mode='nearest')
    # ...create the d/di[gGR(r,i)] interpolation function
    dgGR_di = RectBivariateSpline(x=r_grid, y=i_grid, z=dgGR_di_filt, kx=1, ky=1, s=0)
    #dgGR_di = interp2d(x=r_grid, y=i_grid, z=dgGR_di_filt, kind='linear', fill_value=(0.0,0.0), bounds_error=False)
    return dgGR_di

#----------------------------------------------------------------------------------------------------
# gNT(r) <-- accounts for disk structure
def get_gNT(fpCF, sigma_r=0):
    
    # Load in the table of gNT(r) factors...
    with open(fpCF, 'rb') as fp:
        fpick    = pickle.load(fp)
        r_grid   = fpick['r_grid']  # [Rg]
        gNT_grid = fpick['gNT']     # gNT(r)
    # ...smooth gNT(r) using a Gaussian kernel
    gNT_filt = ndimage.filters.gaussian_filter(gNT_grid, sigma=sigma_r, mode='nearest')
    # ...create the gNT(r) interpolation function
    gNT = interp1d(x=r_grid, y=gNT_filt, kind='linear', fill_value='extrapolate', bounds_error=False)
    return gNT

#----------------------------------------------------------------------------------------------------
# d/dr[gNT(r)] <-- inner disk radius derivative of gNT
def get_dgNT_dr(fpCF, sigma_r=0):
    
    # Load in the table of gNT(r) factors...
    with open(fpCF, 'rb') as fp:
        fpick    = pickle.load(fp)
        r_grid   = fpick['r_grid']  # [Rg]
        gNT_grid = fpick['gNT']     # gNT(r)
    # ...calculate the derivative d/dr[gNT(r)], where the r-grid spacing is uniform
    dr      = r_grid[1] - r_grid[0]
    dgNT_dr = np.gradient(gNT_grid, *[dr], edge_order=2)
    # ...smooth d/dr[gNT(r)] using a Gaussian kernel
    dgNT_dr_filt = ndimage.filters.gaussian_filter(dgNT_dr, sigma=sigma_r, mode='nearest')
    # ...create the d/dr[gNT(r)] interpolation function
    dgNT_dr = interp1d(x=r_grid, y=dgNT_dr_filt, kind='linear', fill_value='extrapolate', bounds_error=False)
    return dgNT_dr

#====================================================================================================
# Collect the DataThief output

#----------------------------------------------------------------------------------------------------
# Deal with the issue of DataThief outputting duplicate x-values.
# When this happens, we average together the leftmost and rightmost y(x).
def fix_duplicates(x, y):

    # Sort x and y
    x, y  = np.array(x), np.array(y)
    isort = np.argsort(x)
    x, y  = x[isort], y[isort]

    # Loop through the sorted x-array and keep non-duplicate entries
    x_keep = []
    y_keep = []
    xL, yL = x[0], y[0]  # Initialize the leftmost (x,y)
    Nx     = np.size(x)
    for i in np.arange(1,Nx):
        # Taking one step rightward
        xR = x[i]
        yR = y[i]
        # If the leftmost-x and the current right-x are different...
        if (xL != xR):
            # ...then average them together and keep the result...
            x_keep.append(0.5*(xL+x[i-1]))
            y_keep.append(0.5*(yL+y[i-1]))
            # ...and update the leftmost (x,y)
            xL = xR
            yL = yR
        # Otherwise, keep stepping rightward until we encounter a different x-value
    
    # Deal with the last x-entry
    if (xL != x[-1]):
        x_keep.append(x[-1])
        y_keep.append(y[-1])
    if (xL == x[-1]):
        x_keep.append(0.5*(xL+x[-1]))
        y_keep.append(0.5*(yL+y[-1]))

    return x_keep, y_keep

#----------------------------------------------------------------------------------------------------
# Convert the marginal density contained in the two-column ASCII table output by DataThief...
# ...into a Python object with a .pdf attribute.
def get_DataThief_fx(file):

    # Read in the DataThief data
    data = asciitable.read(file, delimiter=',', comment='#')
    x    = data.col1
    f_x  = data.col2

    # Fix the duplicate entries in the DataThief output
    x, f_x = fix_duplicates(x=x, y=f_x)

    # Avoid slightly negative f_x values by adding min(f_x)
    f_x += np.min(f_x)

    # Normalize f_x
    f_x /= np.trapz(y=f_x, x=x)

    # Create a function that interpolates f_x
    f_x_interp = interp1d(x=x, y=f_x, kind='linear', fill_value=(0.0,0.0), bounds_error=False)

    # Using the interpolated f_x, generate the f_x object with a .pdf attribute
    f_x = generate_f_x(f_x_interp=f_x_interp)

    return x, f_x

#====================================================================================================
# Binary mass function joint density f_FK2(F,K2) and marginal density f_F(F)

#----------------------------------------------------------------------------------------------------
# Perform a change of variables {P,K2} --> {F,K2} to get the joint density f_FK2(F,K2)
class generate_fFK2():
    # Inputs: Marginal densities f_P(P), f_K2(K2)
    def __init__(self, f_P, f_K2):
        self.f_P  = f_P
        self.f_K2 = f_K2
    def pdf(self, F, K2):
        P     = 2.0 * np.pi * c.G * F / K2**3.0
        J     = 2.0 * np.pi * c.G / K2**3.0
        f_FK2 = self.f_P.pdf(P) * self.f_K2.pdf(K2) * np.abs(J)
        return f_FK2

#----------------------------------------------------------------------------------------------------
# Calculate the binary mass function marginal density f_F(F)...
# ...by marginalizing the joint density f_FK2(F,K2) over the semi-amplitude K2
def get_fF_massfunc(F, K2, f_P, f_K2):
    # Inputs: Marginal densities {f_P, f_K2} and 1D arrays {F, K2} for interpolation
    f_FK2      = generate_fFK2(f_P=f_P, f_K2=f_K2)
    f_F_trapz  = np.trapz( axis=1, x=K2, y=f_FK2.pdf(F=F[:,None], K2=K2[None,:]) )
    f_F_interp = interp1d(x=F, y=f_F_trapz, kind='linear', fill_value=(0.0,0.0), bounds_error=False)
    f_F        = generate_f_x(f_x_interp=f_F_interp)
    return f_F

#====================================================================================================
# Joint density f_MDi(M,D,i)

#----------------------------------------------------------------------------------------------------
# Calculate the joint density f_MDi(M,D,i) when...
# ...{M,D,i} are independent
class get_fMDi_indep():
    # Inputs: Marginal densities f_M(M), f_D(D), f_i(i)
    def __init__(self, f_M, f_D, f_i):
        self.f_M = f_M
        self.f_D = f_D
        self.f_i = f_i
    def pdf(self, M, D, i):
        f_MDi = self.f_M.pdf(M) * self.f_D.pdf(D) * self.f_i.pdf(i)
        return f_MDi

#----------------------------------------------------------------------------------------------------
# Class for generating a Python object f_MDi with a .pdf attribute...
# ...given the interpolation function f_MDi_interp
class generate_f_MDi():
    # Input: Interpolation function for the joint density f_MDi(M,D,i)
    def __init__(self, f_MDi_interp):
        self.f_MDi_interp = f_MDi_interp
    def pdf(self, M, D, i):
        MDi   = (M, D, i)
        f_MDi = self.f_MDi_interp(MDi)
        return f_MDi
        # Access this output joint density like this...
        # f_MDi.pdf(M=M[:,None,None], D=D[None,:,None], i=i[None,None,:])

#----------------------------------------------------------------------------------------------------
# Calculate the joint density f_MDi(M,D,i) when...
# ...{M,i} are related through the binary mass function and...
# ...{F,i,M2} are independent

#....................................................................................................
# Perform a change of variables {F,i,M2} --> {M,i,M2} to get the joint density f_MiM2(M,i,M2)
class generate_f_MiM2():
    # Inputs: Marginal densities {f_F, f_i, f_M2}
    def __init__(self, f_F, f_i, f_M2):
        self.f_F  = f_F
        self.f_i  = f_i
        self.f_M2 = f_M2
    def pdf(self, M, i, M2):
        F      = M**3.0 * np.sin(i)**3.0 / (M + M2)**2.0
        J      = M**2.0 * np.sin(i)**3.0 / (M + M2)**2.0 * (3.0 - 2.0 * M / (M + M2))
        f_MiM2 = self.f_F.pdf(F) * self.f_i.pdf(i) * self.f_M2.pdf(M2) * np.abs(J)
        return f_MiM2

#....................................................................................................
# Generate the f_MDi object with a .pdf attribute
def get_fMDi_massfunc(M, i, M2, D, f_F, f_i, f_M2, f_D):
    # Inputs: Marginal densities {f_F, f_i, f_M2, f_D} and 1D arrays {M, i, M2, D} for interpolation
        
    # Get the f_MiM2 object with a .pdf attribute
    f_MiM2 = generate_f_MiM2(f_F=f_F, f_i=f_i, f_M2=f_M2)

    # Calculate f_Mi by marginalizing f_MiM2 over M2
    f_Mi = np.trapz(y=f_MiM2.pdf(M=M[:,None,None], i=i[None,:,None], M2=M2[None,None,:]), x=M2, axis=2)

    # Calculate f_MDi assuming {M,i} and D are independently distributed
    # NOTE: Providing the input "i" array calculated from the input "D" array...
    #       ...makes the joint density f_Mi a conditional joint density f_Mi|D
    f_MDi = f_Mi[:,None,:] * f_D.pdf(D[None,:,None])
    
    # Create a function that interpolates f_MDi
    f_MDi_interp = RegularGridInterpolator(points=(M,D,i), values=f_MDi, method='linear', fill_value=0.0, bounds_error=False)
    
    # Using the interpolated f_MDi, generate the f_MDi object with a .pdf attribute
    f_MDi = generate_f_MDi(f_MDi_interp=f_MDi_interp)
    
    return f_MDi

#....................................................................................................
# Calculate the black hole mass marginal density f_M(M)...
# ...by marginalizing the joint density f_MiM2(M,i,M2) over the inclination i and companion mass M2
def get_fM_massfunc(M, i, M2, f_F, f_i, f_M2):
    # Inputs: Marginal densities {f_F, f_i, f_M2} and 1D arrays {M, i, M2} for interpolation
    f_MiM2     = generate_f_MiM2(f_F=f_F, f_i=f_i, f_M2=f_M2)
    f_M_trapz  = np.trapz( axis=1, x=i,  y=\
                 np.trapz( axis=2, x=M2, y=f_MiM2.pdf(M=M[:,None,None], i=i[None,:,None], M2=M2[None,None,:]) ) )
    f_M_interp = interp1d(x=M, y=f_M_trapz, kind='linear', fill_value=(0.0,0.0), bounds_error=False)
    f_M        = generate_f_x(f_x_interp=f_M_interp)
    return f_M

#====================================================================================================
# Kinematic jet model relating the distance and jet inclination

#----------------------------------------------------------------------------------------------------
# Calculate the jet inclination from the distance and proper motions
def calc_iJet(D, mu_app, mu_rec):
    iJet = np.arctan( 1.16e-2 * (mu_app * mu_rec) / (mu_app - mu_rec) * D * c.cm2kpc)
    return iJet

#----------------------------------------------------------------------------------------------------
# Perform a change of variables to get the marginal density f_i(i) from f_D(D)
class generate_fiJet():
    # Inputs: Marginal density f_D(D) and the proper motions of the approaching/receding jets (mas/d)
    def __init__(self, f_D, mu_app, mu_rec):
        self.f_D    = f_D     # [cm]^-1
        self.mu_app = mu_app  # [mas/day]
        self.mu_rec = mu_rec  # [mas/day]
    def pdf(self, i):
        D   = (1.0 / 1.16e-2) * (self.mu_app - self.mu_rec) / (self.mu_app * self.mu_rec) * np.tan(i)       * c.kpc2cm  # [cm]
        J   = (1.0 / 1.16e-2) * (self.mu_app - self.mu_rec) / (self.mu_app * self.mu_rec) / np.cos(i)**2.0  * c.kpc2cm  # [cm]
        f_i = self.f_D.pdf(D) * np.abs(J)
        return f_i

#====================================================================================================
# Ozel et al. (2010) black hole mass distribution
def get_fM_Ozel():
    M6p3  = 6.3  * c.sun2g  # [g]
    M1p57 = 1.57 * c.sun2g  # [g]
    f_M   = expon(loc=M6p3, scale=M1p57)
    return f_M  # [g^-1]
    
#====================================================================================================

#----------------------------------------------------------------------------------------------------
# Truncated normal distributions (symmetric or asymmetric)

# Symmetric truncated normal distribution with...
# (mu, sigma) = (x_val, x_err) that spans [x_min, x_max]
def get_truncnorm(x_val, x_err, x_min, x_max):
    f_x = truncnorm(a=(x_min-x_val)/x_err, b=(x_max-x_val)/x_err, loc=x_val, scale=x_err)
    return f_x

# Asymmetric truncated normal distribution with...
# (mu, sigmaL, sigmaR) = (x_val, x_errL, x_errR) that spans [x_min, x_max]
# NOTE: This is hacky...there is a discontinuity at the peak of the distribution
def get_asym_truncnorm(x_val, x_errL, x_errR, x_min, x_max, Nx=10000):
 
    # Get the left and right halves of the asymmetric normal distribution
    f_xL = truncnorm(a=(x_min-x_val)/x_errL, b=0, loc=x_val, scale=x_errL)
    f_xR = truncnorm(a=0, b=(x_max-x_val)/x_errR, loc=x_val, scale=x_errR)
 
    # Calculate the asymmetric normal distribution on a grid
    x   = np.linspace(x_min, x_max, Nx)
    i0  = np.argmin(np.abs(x - x_val))
    f_x = np.zeros(Nx)
    f_x[0:i0] = f_xL.pdf(x[0:i0]) * 0.5  # 1/2 needed for normalization
    f_x[i0::] = f_xR.pdf(x[i0::]) * 0.5  # 1/2 needed for normalization

    # Create a function that interpolates f_x
    f_x_interp = interp1d(x=x, y=f_x, kind='linear', fill_value=(0.0,0.0), bounds_error=False)

    # Using the interpolated f_x, generate the f_x object with a .pdf attribute
    f_x = generate_f_x(f_x_interp=f_x_interp)

    return f_x

#----------------------------------------------------------------------------------------------------
# Exponential distribution
def get_expon(x_val, x_err):
    f_x = expon(loc=x_val, scale=x_err)
    return f_x

#====================================================================================================

# Use the {r, fcol, M, D, i} peaks in their respective marginal distributions to...
# approximate the location of the K-peak in the unknown f_K(K) marginal distribution
# NOTE: All inputs are either non-dimensional or in CGS units
def calc_K_peak(f_r, fcol, M, D, i, gGR, gNT):
    r      = np.linspace(1, 9, 1000)
    r_peak = r[np.argmax(f_r.pdf(r))]
    K      = r_peak**2.0 / fcol**4.0 * ( (c.G * M / c.c**2.0) / D )**2.0 * np.cos(i) \
           * ( 0.5 + 0.75 * np.cos(i) ) * gGR.ev(xi=r_peak, yi=i, dx=0, dy=0) * gNT(r_peak) \
           * c.cgs2Kflux
    return K
