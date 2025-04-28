import numpy as np
import constants as c
import h5py
import pickle
from argparse import ArgumentParser,RawTextHelpFormatter
from multiprocessing import Pool

import time, datetime
from scipy.interpolate import interp1d
from scipy.signal import unit_impulse
from scipy.special import eval_chebyt, eval_chebyu, eval_legendre, lpmv, factorial
from numpy.polynomial.chebyshev import chebweight
from numpy.polynomial.legendre import legweight

from scipy import optimize,integrate

from collect_CF_results import get_minmax, get_truncnorm, get_gGR, get_dgGR_dr, get_gNT, get_dgNT_dr

# set numpy options
np.set_printoptions(precision=3, suppress=True, linewidth=200)

# globals
a = -1
b = 1

# input
parser = ArgumentParser(
    formatter_class=RawTextHelpFormatter,
    description="""Convert f_r(r) to f_K(K), marginalizing over {fcol, M, D, inc}.

To do this, we must invert the following integral transform equation
to solve for f_K(K):
    
  f_r(r) = \int_{x} f_K(h^-1(r,fcol,x)) f_f(fcol) f_x(x) |J(r,fcol,x)| dfcol dx
  K      = h^-1(r,fcol,x) <-- inverse transformation function
  x      = {M, D, inc}
  f_x(x) = f_MDi(M, D, i)

By approximating f_K(K) and f_r(r) as linear combinations of
orthonormal basis functions, which have the special property that the
inner product evaluates to a delta function, we can solve a linear
system and approximate f_K(K).""",
    epilog="""Output:
-------
HDF5 file containing the relevant results (scroll to the bottom of this code).

Examples:
---------
# GRO J1655-40
>> fout=test_J1655.h5
>> fin=CF_results_J1655.p
>> python convert_fr2fK.py ${fin} --fout ${fout} --NM 18 --ND 18 --Ni 18 --K_max 800 --r_min 1 --r_max 6 --Nproc 6

Notes:
------
fcolMarg should usually be set to 'False' because the continuum
fitting practitioners effectively did not marginalize over fcol. But,
we provide the option to marginalize over fcol to test if the tiny
uncertainties considered during continuum fitting make any
difference. If fcolMarg is set to 'True', then you must provide the
inputs: fcol_val, fcol_err, Nstd, Nf""")

parser.add_argument('fin',type=str,
                    help=('Input pickle file '
                          +'<-- output from running various '
                          +'routines in collect_CF_results.py'))
parser.add_argument('--fout',type=str,default='test.h5',
                    help='Output HDF5 file. Defaults to test.h5')
# TODO(JMM): look up nalp_m0
parser.add_argument('--basis',type=str,
                    choices=['cheb1'],
                    #choices=['nalp_m0', 'cheb1', 'cheb2', 'leg'],
                    default='cheb1',
                    help='Basis function to use. Default is cheb1.')
parser.add_argument('--rpdisc',type=str,
                    default = 'modes',
                    choices=['modes','nodes','CDF'],
                    help="Use modal or nodal discretization in r'")
parser.add_argument('--Nn',type=int,default=18,
                     help="Number of coefficients for f_K(K'). Default is 18.")
parser.add_argument('--Nm',type=int,default=None,
                     help="Number of coefficients for f_r(r'). Defaults to Nn")
parser.add_argument('--Nr',type=int,default=None,
                    help=("Number of grid points for the dependent variable r',"
                          +" for approximating f_r'(r')"))
parser.add_argument('--NT',type=int,default=None,
                    help=("Number of grid points for the "+
                          "dependent variable r', for approximating ~T_n(r')"))
parser.add_argument('--NK',type=int,default=None,
                    help=("Number of grid points for the independent variable K', "
                          +"for approximating f_K'(K')"))
parser.add_argument('--NM',type=int,required=True,
                    help="Number of grid points for independent variable M")
parser.add_argument('--ND',type=int,required=True,
                    help="Number of grid points for independent variable D")
parser.add_argument('--Ni',type=int,required=True,
                    help="Number of grid points for independent variable i")
parser.add_argument('--K_min',type=float,default=0,
                    help=("Minimum disk flux normalization [km^2 (kpc/10)^-2]."
                          +" Default is 0."))
parser.add_argument('--K_max',type=float,required=True,
                    help=("Maximum disk flux normalization [km^2 (kpc/10)^-2]."
                          +" Required."))
parser.add_argument('--r_min',type=float,default=None,
                    help="Minimum inner disk radius [Rg]. Required.")
parser.add_argument('--r_max',type=float,default=None,
                    help="Maximum inner disk radius [Rg]. Required.")
parser.add_argument('--nogGR',action='store_true',
                    help="Don't include the gGR(r,i) or gNT(r) disk flux correction factors")
parser.add_argument('--r_GR',type=float,default=0,
                    help="gGR smoothing kernel for the inner disk radius [Rg]. Default is 0.")
parser.add_argument('--i_GR',type=float,default=0,
                    help="gGR smoothing kernel for the inner disk inclination [deg]. Default is 0.")
parser.add_argument('--r_dGR',type=float,default=0,
                    help="d/dr[gGR] smoothing kernel for the inner disk radius [Rg]. Default is 0.")
parser.add_argument('--i_dGR',type=float,default=0,
                    help="d/dr[gGR] smoothing kernel for the inner disk inclination [deg]. Default is 0.")
parser.add_argument('--r_NT',type=float,default=0,
                    help="gNT smoothing kernel for the inner disk radius [Rg]. Default is 0.")
parser.add_argument('--r_dNT',type=float,default=0,
                    help="d/dr[gNT] smoothing kernel for the inner disk radius [Rg]. Default is 0.")
parser.add_argument('--lbounds',type=float,default=5,
                    help='Lagrange multiplier for boundaries')
parser.add_argument('--lunit',type=float,default=1,
                    help='Lagrange multiplier for unitarity')
parser.add_argument('--lpos',type=float,default=1,
                    help='Lagrange multiplier for positivity')
parser.add_argument('--lfilter',type=float,default=1,
                    help='Lagrange multiplier for exponential filter')
parser.add_argument('--constrainKBnds',action='store_true',
                    help='Enforce f_K(-1)=f_K(1)=0 in solver')
parser.add_argument('--constrainRBnds',action='store_true',
                    help='Enforce f_r(-1)=f_r(1)=0 in solver')
parser.add_argument('--constrainUnitarity',action='store_true',
                     help='Enforce f_K integrates to 1 in solver')
parser.add_argument('--constrainPos',action='store_true',
                    help='Enforce f_K > 0 in solver')
parser.add_argument('--solver',type=str,default=None,
                    choices=['Nelder-Mead','Powell','CG',
                             'BFGS','Newton-CG',
                             'L-BFGS-B','TNC','COBYLA',
                             'SLSQP','trust-constr',
                             'dogleg','trust-ncg',
                             'trust-exact',
                             'trust-krylov'],
                    help='Solver used')
parser.add_argument('--Nres',type=int,default=10,
                    help=("Grid resolution multiplicative factor"
                          +" used only for the {Nr, NT, NK}"
                          +" that are unset"))
parser.add_argument('--Nproc',type=int,default=1,
                    help='Number of processors to run on')
parser.add_argument('--fcolMarg',action='store_true',
                    help='Marginalize over fcol')
parser.add_argument('--fcol_val',type=float,default=None,
                    help="Mean for the color correction factor [-]")
parser.add_argument('--fcol_err',type=float,default=None,
                    help="Uncertainty for the color correction factor [-]")
parser.add_argument('--Nstd',type=float,default=None,
                    help='Number of standard deviations for truncating f_f(fcol)')
parser.add_argument('--Nf',type=int,default=None,
                    help='Number of grid points for the independent variable fcol')
parser.add_argument('--fixMDi',action='store_true',
                    help='Fix weird zeros in MDi')
parser.add_argument('--smoothFr',action='store_true',
                    help='Smooth out Fr with a kernel')
parser.add_argument('--FrSmthWidth',type=int,default=11,
                    help='Width (in points) of Fr smoothing kernel')

#===============================================================================
# Fixup class for f_MDi if needed
class f_MDi_fixed:
    def __init__(self,ds,NM,ND,Ni):
        M_min, M_max = ds['M_min'], ds['M_max']  # [g]
        D_min, D_max = ds['D_min'], ds['D_max']  # [cm]
        i_min, i_max = ds['i_min'], ds['i_max']  # [rad]
        M = np.linspace(M_min,M_max,NM+1)
        D = np.linspace(D_min,D_max,ND+1)
        i = np.linspace(i_min,i_max,Ni+1)
        f_M = ds['f_M'].pdf(M)
        f_D = ds['f_D'].pdf(D)
        f_i = ds['f_i'].pdf(i)
        self.f_M_interp = self.fix_pair(M,f_M)
        self.f_D_interp = self.fix_pair(D,f_D)
        self.f_i_interp = self.fix_pair(i,f_i)

    def fix_pair(self,x,y):
        x_fixed = x[y > 0]
        y_fixed = y[y > 0]
        if x_fixed[0] != x[0]:
            x_fixed = np.concatenate([np.array([x[0]]),x_fixed])
            y_fixed = np.concatenate([np.array([y[0]]),y_fixed])
        if x_fixed[-1] != x[-1]:
            x_fixed = np.concatenate([x_fixed,np.array([x[-1]])])
            y_fixed = np.concatenate([y_fixed,np.array([y[-1]])])
        return interp1d(x_fixed,y_fixed,kind='linear',copy=True,
                        bounds_error = False, fill_value=0)

    def pdf(self,M,D,i):
        return self.f_M_interp(M)*self.f_D_interp(D)*self.f_i_interp(i)

# smooth f_r if needed
def smooth(x,window_len=13,window='hanning'):

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2):-(window_len//2)]

class f_r_fixed:
    def __init__(self,ds,NR,rmin=None,rmax=None,width=None):
        f_r = ds['f_rCF']
        r_min = ds['rCF_min'] if rmin is None else rmin
        r_max = ds['rCF_max'] if rmax is None else rmax
        r = np.linspace(r_min,r_max,NR)
        f_rg = f_r.pdf(r)
        f_fixed = smooth(f_rg,width)
        f_fixed[0] = 0
        f_fixed[-1] = 0
        self.f_r_interp = interp1d(r,f_fixed,kind='linear',copy=True,
                                   bounds_error=False,fill_value=0)

    def pdf(self,r):
        return self.f_r_interp(r)

    def __call__(self,x):
        return self.pdf(x)

#===============================================================================
# BASIS FUNCTIONS

def set_basis_functions(basis):
    if basis == 'cheb1':
        eval_Tn = eval_chebyt
        weight = chebweight
        def coeff_norm(c_n):
            out = c_n.copy()
            out[0]  *= 1.0/np.pi
            out[1:] *= 2.0/np.pi
            return out
    elif basis == 'cheb2':
        eval_Tn = eval_chebyu
        weight = lambda x: 1./chebweight(x=x)
        coeff_norm = lambda c_n: 2.*c_n/np.pi
    elif basis == 'leg':
        eval_Tn = eval_legendre
        weight = legweight
        coeff_norm = lambda c_n: 0.5*(2.0*np.arange(len(c_n)) + 1.0)
    elif basis == 'nalp_m0':
        def eval_Tn(n, x):
            m,l = 0,n
            return (lpmv(m, l, x)
                    * (-1.0)**m * np.sqrt((l+0.5) * factorial(l-m)
                                          / factorial(l+m)))
        weight = lambda x: np.ones_like(x)
        coeff_norm = lambda c_n: c_n
    else:
        raise ValueError("Unknown basis")
    return eval_Tn, weight, coeff_norm

#===============================================================================
# GRIDS

# Re-bin x from bin edges (x_edges <-- size Nx) to bin centers
# (x_cents <-- size Nx-1)
def edges2cents(x_edges):
    x_cents = 0.5 * (x_edges[:-1] + x_edges[1:])
    return x_cents

# Re-scale x (with domain [x_min, x_max]) to x' (with domain [a, b])
# Re-bin x from bin edges to bin centers <-- clip=True (use this when
# input x is an edges array)
def rescale_x2xP(x, x_min, x_max, a=-1.0, b=1.0, clip=False):
    xP = (b - a) * (x - x_min) / (x_max - x_min) + a
    if clip:
        xP = 0.5 * (xP[:-1] + xP[1:])
    return xP

# Re-scale x' (with domain [a, b]) to x (with domain [x_min, x_max])
# Re-bin x from bin edges to bin centers <-- clip=True (use this when
# input xP is an edges array)
def rescale_xP2x(xP, x_min, x_max, a=-1.0, b=1.0, clip=False):
    x = (x_max - x_min) * (xP - a) / (b - a) + x_min
    if clip:
        x = 0.5 * (x[:-1] + x[1:])
    return x

# Convert f_x(x) to f_x'(x')
# ALTERNTATIVE METHOD: f_xP = f_x / np.trapz(y=f_x, x=xP)
def rescale_fx2fxP(f_x, x_min, x_max, a=-1., b=1.):
    dx_dxP = (x_max - x_min) / (b - a)
    f_xP   = f_x * dx_dxP
    return f_xP

# Convert f_x'(x') to f_x(x)
# ALTERNTATIVE METHOD: f_x = f_xP / np.trapz(y=f_xP, x=x)
def rescale_fxP2fx(f_xP, x_min, x_max, a=-1., b=1.):
    dxP_dx = (b - a) / (x_max - x_min)
    f_x    = f_xP * dxP_dx
    return f_x

# ================================================================================
# COEFFICIENT FORMULA

# Apply the "coefficient formula", c_l = <f_z'(z'), T_l(z')>, to
# calculate and return...
# ...the coefficients c_l, the basis functions T_l(z'),...
# ...and lcf_z'(z') = \sum c_l T_l(z'), the linear combination
#    approximation of the function f_z'(z')
#
# NOTE: This function is a generic application of the coefficient
#       formula.  In other words, we can use this function to
#       calculate C_m and A_nk for our specific problem.
def calc_cl_Tl_lcf_base(zP, f_zP, Nl,
                        eval_Tn, weight, coeff_norm):

    # Number of z'-bins
    NzP = np.size(zP)

    # Array of integers l = 0,1,...Nl-1
    lArr = np.arange(Nl)

    # Basis functions T_l(z')
    Tl_zP = np.zeros([Nl, NzP])
    for l in np.arange(Nl):
        for i in np.arange(NzP):
            Tl_zP[l,i] = eval_Tn(l, zP[i])

    # Weighting function W(z')
    W_zP = weight(zP)
    
    # Coefficients c_l
    c_l = np.zeros(Nl)
    for l in lArr:
        Igrand = W_zP * f_zP * Tl_zP[l,:]
        c_l[l] = np.trapz(y=Igrand, x=zP)

    # Normalize the coefficents c_l
    # ...Some supported basis functions are just orthogonal, not orthonormal.
    # ...The property of the inner product we exploit only works for orthonormality.
    # ...But we can normalize the coefficients to satisfy this requirement.
    c_l = coeff_norm(c_l)
    
    # Approximate f_z'(z') as a linear combination of
    # ortho-"normalized" basis functions
    lcf_zP = np.zeros(NzP)
    for i in np.arange(NzP):
        lcf_zP[i] = np.sum(c_l * Tl_zP[:,i])

    return c_l, Tl_zP, lcf_zP

# ================================================================================
# A_ni physics

# Calculate the limb-darkening law
def calc_Yld(inc):
    Yld = 0.5 + 0.75 * np.cos(inc)
    return Yld

# Calculate K(r, fcol, M, D, inc) in units of [-]
def calc_K(r, fcol, M, D, inc, GRflag, fcolMarg):
    
    # Limb-darkening Y(inc)
    Yld = calc_Yld(inc=inc)
    
    # General relativistic flux correction factors gGR(r, inc) and gNT(r)
    if GRflag:
        gGR = gGR_interp.ev(xi=r, yi=inc, dx=0, dy=0)
        gNT = gNT_interp(r)
    else:
        gGR = np.ones(Ni)
        gNT = 1.0

    # Calculate K [-]
    if fcolMarg: # <-- K = K(fcol,M,D,inc)
        K = (r**2.0 / fcol[:,None,None,None]**4.0 * ((c.G * M[None,:,None,None] / c.c**2.0) / D[None,None,:,None])**2.0 * np.cos(inc[None,None,None,:]) * Yld[None,None,None,:] * gGR[None,None,None,:] * gNT)
    else: # <-- K = K(M,D,inc)
        K = (r**2.0 / fcol_val**4.0 * ((c.G * M[:,None,None] / c.c**2.0) / D[None,:,None])**2.0 * np.cos(inc[None,None,:]) * Yld[None,None,:] * gGR[None,None,:] * gNT)
    return K

# Calculate the Jacobian in the K --> r transformation in units of [Rg]^-1
# NOTE: The input r is always a single value
def calc_J_K2r(r, K, inc, GRflag, fcolMarg):
    # Calculate J_K2r w/ gGR
    if GRflag:
        gGR        = gGR_interp.ev(xi=r, yi=inc, dx=0, dy=0)
        dgGR_dr    = dgGR_dr_interp.ev(xi=r, yi=inc, dx=0, dy=0)  # Yes, dx=0 is correct here - see get_dgGR_dr()
        dln_gGR_dr = dgGR_dr / gGR
        gNT        = gNT_interp(r)
        dgNT_dr    = dgNT_dr_interp(r)
        dln_gNT_dr = dgNT_dr / gNT
        if fcolMarg:
            J_K2r = K * ((2.0 / r) + dln_gGR_dr[None,None,None,:] + dln_gNT_dr)
        else:
            J_K2r = K * ((2.0 / r) + dln_gGR_dr[None,None,:] + dln_gNT_dr)
    else: # Calculate J_K2r w/o gGR
        J_K2r = K * (2.0 / r)
    return J_K2r

# Calculate K' in the K' --> r' transformation and the Jacobian J_K-->r
def calc_KP_JK2r(rP, fcol, M, D, inc,
                 r_min, r_max, K_min, K_max, a, b,
                 GRflag, fcolMarg):

    # Convert r' --> r
    r = rescale_xP2x(xP=rP, x_min=r_min, x_max=r_max, a=a, b=b, clip=False)

    # Calculate the inverse transformation function K(r, fcol, M, D, inc)
    K = calc_K(r, fcol, M, D, inc, GRflag, fcolMarg)

    # Convert K --> K'
    KP = rescale_x2xP(x=K, x_min=K_min, x_max=K_max, a=a, b=b, clip=False)

    # Calculate J_K2r
    J_K2r = calc_J_K2r(r, K, inc, GRflag, fcolMarg)

    return KP, J_K2r

# Calculate ~T_n(r'), A_nk, lcT_n(r')
# NOTE: This is the slow part of the code, so I tried to take
#       advantage of vectorization ("broadcasting"). I also confirmed
#       that broadcasting over r' is not worth the very minor
#       performance improvement.
def calc_TnrP(n,
              NT,
              Nf, NM, ND, Ni,
              TP,fcol,M,D,inc,
              r_min,r_max,K_min,K_max,
              a,b,
              eval_Tn, weight, coeff_norm,
              GRflag,
              fcolMarg):

    # Initialize the ~T_n(r') array
    Tn_rP = np.zeros(NT)
    for r in np.arange(NT):

        # Dimensions of the arrays below: [Nf, NM, ND, Ni]

        # Compute K', given r' and the "known" {fcol, M, D, inc}-grid
        # Compute the Jacobian of the K --> r transformation
        if fcolMarg:
            KP_grid, J_K2r = calc_KP_JK2r(TP[r], fcol,
                                          M, D, inc,
                                          r_min, r_max,
                                          K_min, K_max,
                                          a, b,
                                          GRflag, fcolMarg)
        else:
            KP_grid, J_K2r = calc_KP_JK2r(TP[r], fcol_val,
                                          M, D, inc,
                                          r_min, r_max,
                                          K_min, K_max,
                                          a, b,
                                          GRflag,fcolMarg)

        # On the multi-dimensional grid: [fcol, M, D, inc]...
        # ...compute the product of marginal densities:
        #    f_f(fcol) * f_M(M) * f_D(D) * f_i(inc)
        if fcolMarg:
            f_x = f_fcol.pdf(
                fcol[:,None,None,None])*f_MDi.pdf(M=M[None,:,None,None],
                                                  D=D[None,None,:,None],
                                                  i=inc[None,None,None,:])
        else:
            f_x = f_MDi.pdf(M=M[:,None,None], D=D[None,:,None], i=inc[None,None,:])

        # Compute the n^th basis function T_n(K') corresponding to
        # each value on the K'-grid
        # NOTE: K' cannot lie outside of its [a,b] range <-- we set
        # T_n(K') = 0 if/when this happens
        if fcolMarg:
            n_grid = np.full(shape=[Nf, NM, ND, Ni], fill_value=n, dtype=int)
        else:
            n_grid = np.full(shape=[NM, ND, Ni],     fill_value=n, dtype=int)
        Tn_KP = eval_Tn(n_grid, KP_grid)
        Tn_KP[np.logical_or(KP_grid < a, KP_grid > b)] = 0.0
                        
        # Calculate the integrand for the K' --> r' transformation
        Igrand_KP2rP = (r_max - r_min) / (K_max - K_min) * Tn_KP * f_x * np.abs(J_K2r)

        # Calculate the ~T_n(r') integral for each r' value
        if fcolMarg:
            Tn_rP[r] = np.trapz( axis=0, x=fcol, y=\
                            np.trapz( axis=1, x=M, y=\
                                np.trapz( axis=2, x=D, y=\
                                    np.trapz( axis=3, x=inc,
                                              y=Igrand_KP2rP ) ) ) )
        else:
            Tn_rP[r] = np.trapz( axis=0, x=M, y=\
                            np.trapz( axis=1, x=D, y=\
                                np.trapz( axis=2, x=inc,
                                          y=Igrand_KP2rP ) ) )

    # Calculate A_nk, T_k(r'), lcT_n(r')
    # the linear combination approximation of ~T_n(r')
    print("...calculating A_nk for n=%s" % str(n))
    A_nk, Tk_rP, lcTn_rP = calc_cl_Tl_lcf_base(TP, Tn_rP, Nk,
                                               eval_Tn,weight,coeff_norm)
    
    # Return a dictionary of the results
    output = {'n':n, 'Tn_rP':Tn_rP, 'A_nk':A_nk,
              'Tk_rP':Tk_rP, 'lcTn_rP':lcTn_rP}
    return output


# ================================================================================
# integration
def get_CDF(f,x):
    return integrate.cumtrapz(f,x,initial=0)


# ================================================================================
# Main loop

if __name__ == "__main__":
    args = parser.parse_args()

    # Check that [a,b] = [-1,1] for certain basis function choices
    basis = args.basis

    if ( (basis == 'nalp_m0') or (basis == 'cheb1')
         or (basis == 'cheb2') or (basis == 'leg') ):
        if ( (a != -1.0) or (b != 1.0) ):
            raise ValueError("[a,b] must be set to [-1,1] for chosen basis '%s'"
                             % basis)

    GRflag = not args.nogGR
    fcolMarg = args.fcolMarg
    Nn = args.Nn     # c_n for f_K'(K'), N+1


    # M_m for f_r'(r'), M+1
    Nm = Nn if args.Nm is None else args.Nm
                     
    Nk = Nm          # Number of coefficients for A_nk, K+1
                     # i.e., number of polynomials to use when
                     # approximating ~T_n(r')

    # Minimum grid "resolution" <-- Number of grid zones per smallest
    # polynomial "wavelength"
    NK = Nn*args.Nres if args.NK is None else args.NK
    Nr = Nm*args.Nres if args.Nr is None else args.Nr
    NT = Nk*args.Nres if args.NT is None else args.NT
    if NK < Nn*args.Nres:
        raise ValueError("NK < Nn*Nres --> Try increasing NK")
    if Nr < Nm*args.Nres:
        raise ValueError("Nr < Nm*Nres --> Try increasing Nr")
    if NT < Nk*args.Nres:
        raise ValueError("NT < Nk*Nres --> Try increasing NT")

    # Start time [seconds]
    tic = time.time()

    # Load the input pickle file
    with open(args.fin,'rb') as f:
        fpick = pickle.load(f)

    # Collect the GR correction factor grids gGR(r[Rg], inc[rad]) and gNT(r[Rg])
    if not args.nogGR:
        gGR_interp     = get_gGR(    fpCF=args.fin, sigma_r=args.r_GR,  sigma_i=args.i_GR)   # gGR(r[Rg], i[rad])       <-- interpolated
        dgGR_dr_interp = get_dgGR_dr(fpCF=args.fin, sigma_r=args.r_dGR, sigma_i=args.i_dGR)  # d/dr[gGR(r[Rg], i[rad])] <-- interpolated
        gNT_interp     = get_gNT(    fpCF=args.fin, sigma_r=args.r_NT)                       # gNT(r[Rg])               <-- interpolated
        dgNT_dr_interp = get_dgNT_dr(fpCF=args.fin, sigma_r=args.r_dNT)                      # d/dr[gNT(r[Rg])]         <-- interpolated
        
    K_min = args.K_min*c.Kflux2cgs
    K_max = args.K_max*c.Kflux2cgs

    NM = args.NM
    ND = args.ND
    Ni = args.Ni

    # Collect the marginal density f_r(r) and specify the Min/Max r-values
    f_r   = fpick['f_rCF']                                           # [Rg]^-1
    r_min = fpick['rCF_min'] if args.r_min is None else args.r_min # [Rg]
    r_max = fpick['rCF_max'] if args.r_max is None else args.r_max # [Rg]

    if args.smoothFr:
        f_r = f_r_fixed(fpick,Nr,r_min,r_max,args.FrSmthWidth)

    # Specify the marginal density f_f(fcol) and the Min/Max fcol-value
    fcolCF = fpick['fcolCF']
    if args.fcolMarg:
        fcol_val = fcolCF if args.fcol_val is None else args.fcol_val
        fcol_err = args.fcol_err
        Nstd = args.Nstd
        if fcol_err is None:
            raise ValueError("Must specify fcol_err when fcolMarg is True")
        fcol_min, fcol_max = get_minmax(x_val=fcol_val, x_err=fcol_err, Nstd=Nstd)
        if (fcol_min < 1.0): fcol_min = 1.0  # <-- IMPORTANT!
        f_fcol = get_truncnorm(x_val=fcol_val, x_err=fcol_err,
                               x_min=fcol_min, x_max=fcol_max)
    else:
        fcol_val = fcolCF if args.fcol_val is None else args.fcol_val
        fcol_err = 0.0    if args.fcol_err is None else args.fcol_err
        fcol_min, fcol_max = fcol_val, fcol_val
    Nf = np.min([NM, ND, Ni]) if args.Nf is None else args.Nf

    # Collect the joint density f_MDi(M,D,i) and specify the Min/Max
    # {M, D, i}-values
    f_MDi        = fpick['f_MDi']                  # [g cm rad]^-1
    M_min, M_max = fpick['M_min'], fpick['M_max']  # [g]
    D_min, D_max = fpick['D_min'], fpick['D_max']  # [cm]
    i_min, i_max = fpick['i_min'], fpick['i_max']  # [rad]

    if args.fixMDi:
        f_MDi = f_MDi_fixed(fpick,NM,ND,Ni)

    eval_Tn, weight, coeff_norm = set_basis_functions(args.basis)
    calc_cl_Tl_lcf = lambda zP, f_zP, Nl: calc_cl_Tl_lcf_base(zP,f_zP,Nl,
                                                              eval_Tn,
                                                              weight,
                                                              coeff_norm)
    # Grid edges in CGS units
    # NOTE: T-edges are are associated with ~T_n(r')
    K_edges    = np.linspace(K_min,    K_max,    NK+1)  # [-]
    r_edges    = np.linspace(r_min,    r_max,    Nr+1)  # [Rg]
    T_edges    = np.linspace(r_min,    r_max,    NT+1)  # [Rg]
    fcol_edges = np.linspace(fcol_min, fcol_max, Nf+1)  # [-]
    M_edges    = np.linspace(M_min,    M_max,    NM+1)  # [g]
    D_edges    = np.linspace(D_min,    D_max,    ND+1)  # [cm]
    i_edges    = np.linspace(i_min,    i_max,    Ni+1)  # [rad]

    # Grid centers in CGS units. We will use these in the workflow
    # below
    K,r,fcol,M,D,inc = [edges2cents(e) for e in\
                        [K_edges,r_edges,fcol_edges,M_edges,D_edges,i_edges]]

    # Re-scale and re-bin the r, T, K domains to span [a,b] NOTE:
    # Again, T' is just a higher resolution version of r' to better
    # approximate ~T_n(r')
    rP = rescale_x2xP(x=r_edges, # r --> r'
                      x_min=r_min, x_max=r_max,
                      a=a, b=b, clip=True)  
    TP = rescale_x2xP(x=T_edges, # r --> r' (higher grid resolution)
                      x_min=r_min, x_max=r_max,
                      a=a, b=b, clip=True)
    KP = rescale_x2xP(x=K_edges, # K --> K'
                      x_min=K_min, x_max=K_max,
                      a=a, b=b, clip=True)

    # ======================================================================
    print("\nCALCULATING C_m...")
    
    # The idea is to get C_m's by approximating f_r'(r') as a linear
    # combination of basis functions...
    # ...and then use the known f_r'(r') to check if this
    #    approximation lcf_r'(r') is acceptable...
    # ...and therefore lending confidence to the coefficents C_m
    #
    # "P" Notation ("P" for "prime"):
    #     f_r  <-- f_r(r),   which is the marginal density of r
    #     f_rP <-- f_r'(r'), which is the marginal density of r'

    # Re-scale f_r(r) accordingly: f_r(r) --> f_r'(r')
    f_rP = rescale_fx2fxP(f_x=f_r.pdf(r),
                          x_min=r_min, x_max=r_max, a=a, b=b)

    # Calculate: C_m, T_m(r'), lcf_r'(r')
    C_m, Tm_rP, lcf_rP = calc_cl_Tl_lcf(zP=rP, f_zP=f_rP, Nl=Nm)

    # Re-scale and re-bin lcf_r'(r') back to the original r domain:
    # lcf_r'(r') --> lcf_r(r)
    lcf_r = rescale_fxP2fx(f_xP=lcf_rP,
                           x_min=r_min, x_max=r_max, a=a, b=b)

    # Print out the C_m coefficients
    print("\nC_m = ", C_m)

    # ======================================================================
    # CALCULATE THE COEFFICIENT MATRIX: A_nk
    #
    # The idea is to get the A_nm's by approximating ~T_n(r') as a
    # linear expansion of basis functions...
    # ...and then use the known ~T_n(r') to check if this
    #    approximation ~lcT_n(r') is acceptable...
    # ...and therefore lending confidence to the coefficents A_nk
    #
    # "P" Notation ("P" for "prime"):
    #     Tn_rP <-- ~T_n(r'), which is integral expression with
    #                         T_n(h^-1(r',fcol,x)) in the integrand

    # Calculate (in parallel): ~T_n(r')
    print("\nCALCULATING ~T_n(r') AND ITS LINEAR "
          +"COMBINATION APPROXIMATION A_nk, lcT_n(r')...\n")
    nArr = np.arange(Nn)
    def task(n):
        return calc_TnrP(n,NT,Nf,NM,ND,Ni,
                         TP,fcol,M,D,inc,
                         r_min,r_max,K_min,K_max,
                         a,b,
                         eval_Tn,weight,coeff_norm,
                         GRflag,fcolMarg)

    with Pool(args.Nproc) as p:
        results = p.map(task,nArr)

    # extract results
    Tn_rP   = np.zeros([Nn, NT])  # NT is the number of r' bins
    A_nk    = np.zeros([Nn, Nk])  # Nk is the number of k coefficients
    Tk_rP   = np.zeros([Nn, Nk, NT])
    lcTn_rP = np.zeros([Nn, NT])

    for n in range(Nn):
        Tn_rP[n,:]   = results[n]['Tn_rP']
        A_nk[n,:]    = results[n]['A_nk']
        Tk_rP[n,:,:] = results[n]['Tk_rP']
        lcTn_rP[n,:] = results[n]['lcTn_rP']

    CDF_f_rP = get_CDF(f_rP,rP)
    CDF_Tn_rP = integrate.cumtrapz(Tn_rP,x=rP,
                                   initial=0,
                                   axis=1)

    print("\nCALCULATING c_n...")
    # Transpose the coefficient row vector C_m to get a column vector
    m8rx_C_m = np.matrix(C_m).T

    # Transpose the coefficient matrix A_nk to get A_kn
    m8rx_A_kn = np.matrix(A_nk).T

    # Collect the (M x N) part of the (K x N) coefficient matrix A_kn
    m8rx_A_mn = m8rx_A_kn[0:Nm,:]
    print("\nA_m0 = ", np.array(m8rx_A_mn[:,0].flatten())[0])

    # Condition number of the matrix

    # NOTE: As a rule of thumb, if the condition number is 10^x, then
    # you may lose up to x digits of accuracy
    cond = np.linalg.cond(m8rx_A_mn)
    print("\nlog10(Condition Number) = ", np.log10(cond))

    # Basis functions T_n(K')
    Tn_KP = np.zeros([Nn, NK])
    for n in np.arange(Nn):
        for i in np.arange(NK):
            Tn_KP[n,i] = eval_Tn(n, KP[i])
    TnL = np.zeros(Nn)
    TnR = np.zeros(Nn)
    for n in np.arange(Nn):
        TnL[n] = eval_Tn(n,a)
        TnR[n] = eval_Tn(n,b)

    # basis functions T_n(r')
    Tn_RP_L = Tn_rP[:,0]
    Tn_RP_R = Tn_rP[:,-1]

    # exponential filter
    log_fmin = -(14-np.log10(cond))
    log_fmax = 0
    filter_eps = 1.0
    log_filter = np.linspace(log_fmin,log_fmax,Nn)
    exp_filter = filter_eps*np.exp(log_filter)

    def eval_c_n(c_n,i):
        return np.sum(c_n*Tn_KP[:,i])
    def leftConstraint(c_n):
        #return np.sum(c_n*TnL)
        return eval_c_n(c_n,a)
    def rightConstraint(c_n):
        #return np.sum(c_n*TnR)
        return eval_c_n(c_n,b)
    def unityConstraint(c_n):
        cgrid = np.einsum('n,ni',c_n,Tn_KP)
        #w = weight(KP)
        #integrand = w*cgrid
        integrand = cgrid
        integral = np.trapz(x=KP,y=integrand)
        return integral
    def filterConstraint(c_n):
        return ((c_n*exp_filter)**2).sum()

    if args.rpdisc == 'modes':
        def residual(c_n,
                     lbounds,lunit,lpos,lfilter):
            out = 0
            # main condition
            diff = (C_m - np.dot(c_n,A_nk))
            for i in range(len(diff)):
                out += diff[i]**2
            # vanishing boundaries
            out += lbounds*leftConstraint(c_n)**2
            out += lbounds*rightConstraint(c_n)**2
            # vanishing boundaries in r'
            out += lbounds*Tn_RP_L.dot(c_n)**2
            out += lbounds*Tn_RP_R.dot(c_n)**2
            # unitarity
            out += lunit*(unityConstraint(c_n)-1)**2
            # positivity
            for i in np.arange(NK):
                val = eval_c_n(c_n,i)
                diff = val - np.abs(val)
                out += (lpos/NK)*diff**2
            # filter on modes
            out += lfilter*filterConstraint(c_n)
            return out
    elif args.rpdisc == 'nodes':
        def residual(c_n,lbounds,lunit,lpos,lfilter):
            out = 0
            # main condition
            out += ((f_rP - np.dot(c_n,Tn_rP))**2).sum()
            # vanishing boundaries
            out += lbounds*leftConstraint(c_n)**2
            out += lbounds*rightConstraint(c_n)**2
            # unitarity
            out += lunit*(unityConstraint(c_n)-1)**2
            # positivity
            for i in np.arange(NK):
                val = eval_c_n(c_n,i)
                diff = val - np.abs(val)
                out += (lpos/NK)*diff**2
            # filter on modes
            out += lfilter*filterConstraint(c_n)
            return out
    elif args.rpdisc == 'CDF':
        def residual(c_n,lbounds,lunit,lpos,lfilter):
            out = 0
            # main condition
            out += ((CDF_f_rP - np.dot(c_n,CDF_Tn_rP))**2).sum()
            # vanishing boundaries
            out += lbounds*leftConstraint(c_n)**2
            out += lbounds*rightConstraint(c_n)**2
            # unitarity
            out += lunit*(unityConstraint(c_n)-1)**2
            # positivity
            for i in np.arange(NK):
                val = eval_c_n(c_n,i)
                diff = val - np.abs(val)
                out += (lpos/NK)*diff**2
            # filter on modes
            out += lfilter*filterConstraint(c_n)
            return out

    # constraints
    constraints = []
    if args.constrainKBnds:
        constraints += [
            optimize.LinearConstraint(TnL,0,0),
            optimize.LinearConstraint(TnR,0,0),
        ]
    if args.constrainRBnds:
        constraints += [
            optimize.LinearConstraint(Tn_RP_L,0,0),
            optimize.LinearConstraint(Tn_RP_R,0,0),
        ]
    if args.constrainUnitarity:
        constraints += [
            optimize.NonlinearConstraint(unityConstraint,1,1),
        ]
    if args.constrainPos:
        constraints += [
            optimize.LinearConstraint(Tn_KP.transpose(),0,np.inf)
        ]

    # first attempt at a solution
    options = {'disp':True,'maxiter':1024}
    multipliers=(args.lbounds,args.lunit,args.lpos,args.lfilter)
    c_n_guess = np.zeros(Nn)
    #c_n_guess = np.random.uniform(-1,1,Nn)
    if constraints:
        sol = optimize.minimize(residual,c_n_guess,
                                args=multipliers,
                                method=args.solver,
                                constraints=constraints,
                                options=options)#,
                                #tol=1e-14)
    else:
        sol = optimize.minimize(residual,c_n_guess,
                                args=multipliers,
                                method=args.solver,
                                options=options)#,
                                #tol=1e-14)
    c_n = sol.x

    print("\nc_n  = ", c_n)

    print("\nAPPROXIMATING f_K(K)...")

    # Basis functions T_n(K')
    Tn_KP = np.zeros([Nn, NK])
    for n in np.arange(Nn):
        for i in np.arange(NK):
            Tn_KP[n,i] = eval_Tn(n, KP[i])

    # Approximate f_K'(K') as a linear combination of
    # ortho-"normalized" basis functions
    lcf_KP = np.zeros(NK)
    for i in np.arange(NK):
        lcf_KP[i] = np.sum(c_n * Tn_KP[:,i])

    # Re-scale and re-bin lcf_K'(K') back to the original K domain:
    # lcf_K'(K') --> lcf_K(K)
    lcf_K = rescale_fxP2fx(f_xP=lcf_KP, x_min=K_min, x_max=K_max, a=a, b=b)

    # CHECK IF THE APPROXIMATION lcf_K(K) RECOVERS f_r(r) ("lc" prefix
    # stands for linear combination)
    print("\n'RECOVERING' f_r(r) FROM THE APPROXIMATION OF f_K(K)...")

    # Create an interpolation function from the lcf_K marginal density
    f_K_interp = interp1d(x=K, y=lcf_K, kind='linear',
                          fill_value=(0.0,0.0),
                          bounds_error=False)  # [-]^-1

    # Initialize the "recovered" f_r(r) array
    recf_r = np.zeros(Nr)
    for i in np.arange(Nr):
        # Dimensions of the arrays below: [Nf, NM, ND, Ni]

        # Calculate the inverse transformation function K(r, fcol, M, D, inc)
        if fcolMarg:
            K_grid = calc_K(r[i], fcol, M, D, inc,
                            GRflag, fcolMarg)
        else:
            K_grid = calc_K(r[i], fcol_val, M, D, inc,
                            GRflag, fcolMarg)

        # Compute the Jacobian of the K --> r transformation
        J_K2r = calc_J_K2r(r[i], K_grid, inc, GRflag, fcolMarg)

        # On the multi-dimensional grid: [fcol, M, D, inc]...
        # ...compute the product of marginal densities:
        #    f_f(fcol) * f_M(M) * f_D(D) * f_i(inc)
        if fcolMarg:
            f_x = (f_fcol.pdf(fcol[:,None,None,None])
                   * f_MDi.pdf(M=M[None,:,None,None],
                               D=D[None,None,:,None],
                               i=inc[None,None,None,:]))
        else:
            f_x = f_MDi.pdf(M=M[:,None,None],
                            D=D[None,:,None],
                            i=inc[None,None,:])

        # Compute the marginal density f_K(K) for each value on the K-grid
        f_K_grid = f_K_interp(K_grid)
                        
        # Calculate the integrand for the K --> r transformation
        Igrand_K2r = f_K_grid * f_x * np.abs(J_K2r)

        # Calculate the "recovered" f_r(r) for each r value
        if fcolMarg:
            recf_r[i] = np.trapz( axis=0, x=fcol, y=\
                            np.trapz( axis=1, x=M, y=\
                                np.trapz( axis=2, x=D, y=\
                                    np.trapz( axis=3, x=inc, y=Igrand_K2r ) ) ) )
        else:
            recf_r[i] = np.trapz( axis=0, x=M, y=\
                            np.trapz( axis=1, x=D, y=\
                                np.trapz( axis=2, x=inc, y=Igrand_K2r ) ) )

    print("\nWRITING OUT RESULTS...")

    # Convert Min/Max and Val/Err quantities from CGS units to astronomy units
    K_min, K_max = K_min * c.cgs2Kflux, K_max * c.cgs2Kflux  # [km^2 (kpc/10)^-2]
    M_min, M_max = M_min * c.g2sun,     M_max * c.g2sun      # [Msun]
    D_min, D_max = D_min * c.cm2kpc,    D_max * c.cm2kpc     # [kpc]
    i_min, i_max = i_min * c.rad2deg,   i_max * c.rad2deg    # [deg]

    # Collect marginal densitites on a grid and convert from CGS to astronomy units
    # NOTE: Important to convert the grids after calculating the marginals
    lcf_K = lcf_K / c.cgs2Kflux  # [km^2 (kpc/10)^-2]^-1
    K     = K     * c.cgs2Kflux  # [km^2 (kpc/10)^-2]
    M     = M     * c.g2sun      # [Msun]
    D     = D     * c.cm2kpc     # [kpc]
    inc   = inc   * c.rad2deg    # [deg]
    if fcolMarg:
        f_fcol = f_fcol.pdf(fcol)
    else:
        f_fcol = unit_impulse(shape=Nf, idx='mid')

    # Write to the output HDF5 file
    with h5py.File(args.fout,'w') as f:
        # inputs
        f.create_dataset('a',        data=a)
        f.create_dataset('b',        data=b)
        f.create_dataset('Nf',       data=Nf)
        f.create_dataset('NM',       data=NM)
        f.create_dataset('ND',       data=ND)
        f.create_dataset('Ni',       data=Ni)
        f.create_dataset('fcol_val', data=fcol_val)    # [-]
        f.create_dataset('fcol_err', data=fcol_err)    # [-]
        f.create_dataset('fcol_min', data=fcol_min)    # [-]
        f.create_dataset('fcol_max', data=fcol_max)    # [-]
        f.create_dataset('M_min',    data=M_min)       # [Msun]
        f.create_dataset('M_max',    data=M_max)       # [Msun]
        f.create_dataset('D_min',    data=D_min)       # [kpc]
        f.create_dataset('D_max',    data=D_max)       # [kpc]
        f.create_dataset('i_min',    data=i_min)       # [deg]
        f.create_dataset('i_max',    data=i_max)       # [deg]
        # pdfs
        f.create_dataset('fcolCF',   data=fcolCF)      # [-]
        f.create_dataset('fcol',     data=fcol)        # [-]
        f.create_dataset('f_fcol',   data=f_fcol)      # [-]^-1
        f.create_dataset('M',        data=M)           # [Msun]
        f.create_dataset('D',        data=D)           # [kpc]
        f.create_dataset('inc',      data=inc)         # [deg]
        # f_r'(r') and f_r(r)
        f.create_dataset('Nm',       data=Nm)
        f.create_dataset('Nr',       data=Nr)
        f.create_dataset('r_min',    data=r_min)       # [Rg]
        f.create_dataset('r_max',    data=r_max)       # [Rg]
        f.create_dataset('r',        data=r)           # [Rg]
        f.create_dataset('rP',       data=rP)          # [-]
        f.create_dataset('f_r',      data=f_r.pdf(r))  # [Rg]^-1
        f.create_dataset('f_rP',     data=f_rP)        # [-]^-1
        f.create_dataset('C_m',      data=C_m)         # [-]
        f.create_dataset('Tm_rP',    data=Tm_rP)       # [-]
        f.create_dataset('lcf_rP',   data=lcf_rP)      # [-]^-1
        f.create_dataset('lcf_r',    data=lcf_r)       # [Rg]^-1
        # ~T_n(r')
        f.create_dataset('Nk',       data=Nk)
        f.create_dataset('NT',       data=NT)
        f.create_dataset('TP',       data=TP)          # [-]
        f.create_dataset('Tn_rP',    data=Tn_rP)       # [-]^-1
        f.create_dataset('A_nk',     data=A_nk)        # [-]
        f.create_dataset('Tk_rP',    data=Tk_rP)       # [-]^-1
        f.create_dataset('lcTn_rP',  data=lcTn_rP)     # [-]^-1
        # f_K'(K') and f_K(K)
        f.create_dataset('Nn',       data=Nn)
        f.create_dataset('NK',       data=NK)
        f.create_dataset('K_min',    data=K_min)       # [km^2 (kpc/10)^-2]
        f.create_dataset('K_max',    data=K_max)       # [km^2 (kpc/10)^-2]
        f.create_dataset('K',        data=K)           # [km^2 (kpc/10)^-2]
        f.create_dataset('KP',       data=KP)          # [-]
        f.create_dataset('c_n',      data=c_n)         # [-]
        f.create_dataset('Tn_KP',    data=Tn_KP)       # [-]^-1
        f.create_dataset('lcf_KP',   data=lcf_KP)      # [-]^-1
        f.create_dataset('lcf_K',    data=lcf_K)       # [km^2 (kpc/10)^-2]^-1
        # recf_r
        f.create_dataset('recf_r',   data=recf_r)      # [Rg]^-1
        # calculation time
        toc    = time.time()                           # End time [s]
        tictoc = toc - tic                             # Total run time [s]
        f.create_dataset('tictoc',   data=tictoc)      # [s]

    print("\nTOTAL RUN TIME (HH:MM:SS) = ",
          str(datetime.timedelta(seconds=tictoc)))
    print("")

    #==========================================================================
    # PLOTS

    # Plotting preferences
    import matplotlib.pyplot as plt
    dpi = 300
    xsize, ysize = 8.4, 5.6
    Lgap, Rgap, Bgap, Tgap = 0.15, 0.025, 0.175, 0.05
    left, right, bottom, top = Lgap, 1.0-Rgap, Bgap, 1.0-Tgap
    fs, fs_sm, lw, pad, tlmaj, tlmin = 24, 24, 2, 10, 10, 5
    
    # Disk flux normalization marginal density, f_K(K)
    fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"${\rm Disc\ Flux\ Normalization},\ K_{\rm flux}\ [{\rm km^{2}/(kpc/10)^{2}}]$", fontsize=fs, ha='center', va='top')
    ax.set_ylabel(r"${\rm Marginal\ Density},\ f_{K}(K_{\rm flux})$", fontsize=fs, ha='center', va='bottom')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.plot(K, lcf_K, linewidth=lw, linestyle='solid', color='C1', zorder=1)
    fig.savefig('test_K.png', bbox_inches=0, dpi=dpi)
    plt.close()
    
    # Inner disk radius marginal density, f_r(r)
    fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"${\rm Inner\ Disc\ Radius},\ r_{\rm in}\ [R_{\rm g}]$", fontsize=fs, ha='center', va='top')
    ax.set_ylabel(r"${\rm Marginal\ Density},\ f_{r}(r_{\rm in})$", fontsize=fs, ha='center', va='bottom')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.plot(r, f_r.pdf(r), linewidth=lw, linestyle='dashed', color='C0', zorder=2)
    ax.plot(r, lcf_r,      linewidth=lw, linestyle='solid',  color='C1', zorder=1)
    ax.plot(r, recf_r,     linewidth=lw, linestyle='dotted', color='C2', zorder=3)
    fig.savefig('test_r.png', bbox_inches=0, dpi=dpi)
    plt.close()
