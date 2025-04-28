import numpy as np
import constants as c
import h5py
import pickle
import misc, sys
import time, datetime
from scipy.interpolate import interp1d
from collect_CF_results import get_minmax, get_truncnorm, get_gGR, get_dgGR_dr, get_gNT, get_dgNT_dr
from multiprocessing import Pool, cpu_count
from gs_stats import interpolated_median
from pdf_trans import spin2risco, check_norm

'''
Purpose:
--------
Convert f_K(K) to f_r(r), marginalizing over {fcol, M, D, inc}.

To do this, we must calculate the following integral equation:
    f_r(r) = \int_{x} f_K(h^-1(r,fcol,x)) f_f(fcol) f_x(x) |J(r,fcol,x)| dfcol dx
    K      = h^-1(r,fcol,x) <-- inverse transformation function
    x      = {M, D, inc}
    f_x(x) = f_MDi(M, D, i)

Example:
--------
See fr_J1655.py
'''
#====================================================================================================
class fK2fr():

    #====================================================================================================
    # INPUTS AND SETUP
    
    def __init__(self, fpCF, fr2K, GRflag=True, fcolMarg=True,
                 r_GR=0, i_GR=0, r_dGR=0, i_dGR=0, r_NT=0, r_dNT=0):
        '''
        fpCF      - Pickle file containing the continuum fitting results (e.g., from running J1655.py)
        fr2K      - HDF5 file output by convert_fr2K.py, contains the disc flux normalization marginal density f_K
        GRflag    - Flag for whether or not to include the gGR(r,i) disk flux correction factor
        fcolMarg  - Flag for whether or not to marginalize over fcol
        r_GR,  i_GR  - Gaussian smoothing kernels (sigma_r[Rg], sigma_i[deg]) for gGR(r,i)
        r_dGR, i_dGR - Gaussian smoothing kernels (sigma_r[Rg], sigma_i[deg]) for d/dr[gGR(r,i)]
        r_NT         - Gaussian smoothing kernels sigma_r[Rg] for gNT(r)
        r_dNT        - Gaussian smoothing kernels sigma_r[Rg] for d/dr[gNT(r)]

        NOTES:
        - fcolMarg=True assumes a truncated normal distribution with fcol_min = 1.0 enforced
        - Probably only necessary to smooth d/dr[gGR(r,i)], and possibly gGR(r,i)
        '''

        # Collect the inputs...
        self.fpCF     = fpCF
        self.fr2K     = fr2K
        self.GRflag   = GRflag
        self.fcolMarg = fcolMarg

        # Collect the relativistic disk flux correction factor interpolation functions w/ the specified smoothing
        if (GRflag is True):
            self.gGR_interp     = get_gGR(    fpCF=fpCF, sigma_r=r_GR,  sigma_i=i_GR)   # gGR(r[Rg],i[rad])       <-- interpolated
            self.dgGR_dr_interp = get_dgGR_dr(fpCF=fpCF, sigma_r=r_dGR, sigma_i=i_dGR)  # d/dr[gGR(r[Rg],i[rad])] <-- interpolated
            self.gNT_interp     = get_gNT(    fpCF=fpCF, sigma_r=r_NT)                  # gNT(r[Rg])              <-- interpolated
            self.dgNT_dr_interp = get_dgNT_dr(fpCF=fpCF, sigma_r=r_dNT)                 # d/dr[gNT(r[Rg])]        <-- interpolated

        # Load the input pickle file and collect: gGR, f_MDi, fcolCF
        with open(fpCF, 'rb') as fp:
            fpick = pickle.load(fp)

            # Collect the joint density f_MDi(M,D,i)
            self.f_MDi = fpick['f_MDi']  # [g cm rad]^-1

            # Collect the continuum fitting color correction factor fcolCF
            self.fcolCF = fpick['fcolCF']  # [-]

        # Collect the marginal density f_K(K) <-- interpolated *and* normalized
        fh5   = h5py.File(self.fr2K, 'r')
        K     = fh5.get('K')[:]     * c.Kflux2cgs  # [-]
        lcf_K = fh5.get('lcf_K')[:] / c.Kflux2cgs  # [-]^-1
        fh5.close()
        lcf_K_norm = lcf_K / np.trapz(y=lcf_K, x=K)  # Normalized [-]^-1
        self.f_K   = interp1d(x=K, y=lcf_K_norm, kind='linear', fill_value=(0.0,0.0), bounds_error=False)  # Interpolated [-]^-1

    #====================================================================================================
    # GRIDS

    #----------------------------------------------------------------------------------------------------
    # Re-bin x from bin edges (x_edges <-- size Nx) to bin centers (x_cents <-- size Nx-1)
    def edges2cents(self, x_edges):
        x_cents = 0.5 * (x_edges[:-1] + x_edges[1:])
        return x_cents

    #----------------------------------------------------------------------------------------------------
    # Construct the parameter grids
    def calc_grids(self, r_min, r_max, Nr, NK, Nf, NM, ND, Ni, fcol_min, fcol_max):

        f = h5py.File(self.fr2K, 'r')
        # Specify the Min/Max r-values
        if (r_min is None): r_min = f.get('r_min')[()]  # [Rg]
        if (r_max is None): r_max = f.get('r_max')[()]  # [Rg]
        # Specify the Min/Max K-values
        K_min = f.get('K_min')[()] * c.Kflux2cgs  # [-]
        K_max = f.get('K_max')[()] * c.Kflux2cgs  # [-]
        # Grid sizes for {r, K, fcol, M, D, i}
        if (Nr is None): Nr = f.get('Nr')[()]
        if (NK is None): NK = f.get('NK')[()]
        if (NM is None): NM = f.get('NM')[()]
        if (ND is None): ND = f.get('ND')[()]
        if (Ni is None): Ni = f.get('Ni')[()]
        f.close()

        # Collect the Min/Max {M, D, i}-values
        with open(self.fpCF, 'rb') as fp:
            fpick = pickle.load(fp)
            M_min, M_max = fpick['M_min'], fpick['M_max']  # [g]
            D_min, D_max = fpick['D_min'], fpick['D_max']  # [cm]
            i_min, i_max = fpick['i_min'], fpick['i_max']  # [rad]

        # Grid edges in CGS units
        r_edges    = np.linspace(r_min, r_max, Nr+1)  # [Rg]
        K_edges    = np.linspace(K_min, K_max, NK+1)  # [-]
        M_edges    = np.linspace(M_min, M_max, NM+1)  # [g]
        D_edges    = np.linspace(D_min, D_max, ND+1)  # [cm]
        i_edges    = np.linspace(i_min, i_max, Ni+1)  # [rad]
    
        # Grid centers in CGS units
        r    = self.edges2cents(r_edges)  # [Rg]
        K    = self.edges2cents(K_edges)  # [-]
        M    = self.edges2cents(M_edges)  # [g]
        D    = self.edges2cents(D_edges)  # [cm]
        inc  = self.edges2cents(i_edges)  # [rad]

        # fcol grid
        if (self.fcolMarg is False):
            fcol = None
        if (self.fcolMarg is True):
            if (Nf is None): Nf = np.min([NM, ND, Ni])
            fcol_edges = np.linspace(fcol_min, fcol_max, Nf+1)  # [-]
            fcol       = self.edges2cents(fcol_edges)           # [-]

        # Return a dictionary containing the {r, K, fcol, M, D, inc}-grids
        grids = {'r':r, 'K':K, 'fcol':fcol, 'M':M, 'D':D, 'inc':inc}
        return grids

    #====================================================================================================
    # CALCULATE f_r(r)

    #----------------------------------------------------------------------------------------------------
    # Calculate the limb-darkening law
    def calc_Yld(self, inc):
        Yld = 0.5 + 0.75 * np.cos(inc)
        return Yld
    
    #----------------------------------------------------------------------------------------------------
    # Calculate K(r, fcol, M, D, inc) in units of [-]
    def calc_K(self, r, fcol, M, D, inc):
    
        # Limb-darkening Y(inc)
        Yld = self.calc_Yld(inc=inc)
    
        # Relativistic disk flux correction factors: gGR(r, inc) and gNT(r)
        if (self.GRflag is False):
            gGR = np.ones(np.size(inc))
            gNT = 1.0
        if (self.GRflag is True):
            gGR = self.gGR_interp.ev(xi=r, yi=inc, dx=0, dy=0)
            gNT = self.gNT_interp(r)

        # Calculate K [-]
        if (self.fcolMarg is False): # <-- K = K(M,D,inc)
            K = r**2.0 / fcol**4.0 * ((c.G * M[:,None,None] / c.c**2.0) / D[None,:,None])**2.0 * np.cos(inc[None,None,:]) * Yld[None,None,:] * gGR[None,None,:] * gNT
        if (self.fcolMarg is True):  # <-- K = K(fcol,M,D,inc)
            K = r**2.0 / fcol[:,None,None,None]**4.0 * ((c.G * M[None,:,None,None] / c.c**2.0) / D[None,None,:,None])**2.0 * np.cos(inc[None,None,None,:]) * Yld[None,None,None,:] * gGR[None,None,None,:] * gNT
        return K

    #----------------------------------------------------------------------------------------------------
    # Calculate the Jacobian in the K --> r transformation in units of [Rg]^-1
    # NOTE: The input 'r' is always a single value
    def calc_J_K2r(self, r, K, inc):

        # Calculate J_K2r w/ gGR and gNT
        if (self.GRflag is True):
            gGR        = self.gGR_interp.ev(xi=r, yi=inc, dx=0, dy=0)
            dgGR_dr    = self.dgGR_dr_interp.ev(xi=r, yi=inc, dx=0, dy=0)
            dln_gGR_dr = dgGR_dr / gGR
            gNT        = self.gNT_interp(r)
            dgNT_dr    = self.dgNT_dr_interp(r)
            dln_gNT_dr = dgNT_dr / gNT
            if (self.fcolMarg is False): J_K2r = K * ( (2.0 / r) + dln_gGR_dr[None,None,:]      + dln_gNT_dr)
            if (self.fcolMarg is True):  J_K2r = K * ( (2.0 / r) + dln_gGR_dr[None,None,None,:] + dln_gNT_dr)

        # Calculate J_K2r w/o gGR and gNT
        if (self.GRflag is False):
            J_K2r = K * (2.0 / r)

        return J_K2r

    #----------------------------------------------------------------------------------------------------
    # Calculate f_r(r) w/ the fcol uncertainties
    # NOTE: The input 'r' is always a single value
    def task_fr(self, r, fcol, M, D, inc, f_fcol):

        # Dimensions of the arrays below: [Nf, NM, ND, Ni]

        # Calculate the inverse transformation function K(r, fcol, M, D, inc)
        K_grid = self.calc_K(r=r, fcol=fcol, M=M, D=D, inc=inc)

        # Compute the Jacobian of the K --> r transformation
        J_K2r = self.calc_J_K2r(r=r, K=K_grid, inc=inc)

        # On the multi-dimensional grid: [fcol, M, D, inc]...
        # ...compute the product of marginal densities: f_f(fcol) * f_M(M) * f_D(D) * f_i(inc)
        if (self.fcolMarg is False):
            f_x = self.f_MDi.pdf(M=M[:,None,None], D=D[None,:,None], i=inc[None,None,:])
        if (self.fcolMarg is True):
            f_x = f_fcol.pdf(fcol[:,None,None,None]) \
                * self.f_MDi.pdf(M=M[None,:,None,None], D=D[None,None,:,None], i=inc[None,None,None,:])

        # Calculate the integrand for the K --> r transformation
        Igrand_K2r = self.f_K(K_grid) * f_x * np.abs(J_K2r)

        # Calculate f_r(r)
        if (self.fcolMarg is False):
            f_r = np.trapz( axis=0, x=M, y=\
                      np.trapz( axis=1, x=D, y=\
                          np.trapz( axis=2, x=inc, y=Igrand_K2r ) ) )
        if (self.fcolMarg is True):
            f_r = np.trapz( axis=0, x=fcol, y=\
                      np.trapz( axis=1, x=M, y=\
                          np.trapz( axis=2, x=D, y=\
                              np.trapz( axis=3, x=inc, y=Igrand_K2r ) ) ) )
        return f_r
    
    #....................................................................................................
    # Calculate f_r(r) <-- in parallel
    def calc_fr(self,
                fcol_val=None, fcol_err=None,
                r_min=None, r_max=None, Nr=None,
                NK=None, Nf=None, NM=None, ND=None, Ni=None,
                Nstd=None, Nproc=None):
        '''
        fcol_val, fcol_err - Mean, (+/-)Error --> Color correction factor [-]
        r_min, r_max       - Minimum, Maximum --> Inner disk radius [Rg]
        Nr                 - Number of grid points for the dependent variable r
        NK, Nf, NM, ND, Ni - Number of grid points for the independent variables {K, fcol, M, D, inc}
        Nstd               - Number of standard deviations for truncating the fcol marginal density f_f(fcol)
        Nproc              - Number of processors to run on
        '''
    
        # Start time [seconds]
        tic = time.time()

        # Specify the marginal density f_f(fcol) and the Min/Max fcol-value
        if (self.fcolMarg is False):
            if (fcol_val is     None): print("\nERROR: Must specify fcol_val when fcolMarg is False");   quit()
            if (fcol_err is not None): print("\nERROR: Cannot specify fcol_err when fcolMarg is False"); quit()
            fcol_min, fcol_max = None, None
            f_fcol = None
        if (self.fcolMarg is True):
            if (fcol_val is None): fcol_val = self.fcolCF
            if (fcol_err is None): print("\nERROR: Must specify fcol_err when fcolMarg is True"); quit()
            if (Nstd     is None): Nstd = 4.0
            fcol_min, fcol_max = get_minmax(x_val=fcol_val, x_err=fcol_err, Nstd=Nstd)
            if (fcol_min < 1.0): fcol_min = 1.0  # <-- IMPORTANT!
            f_fcol = get_truncnorm(x_val=fcol_val, x_err=fcol_err, x_min=fcol_min, x_max=fcol_max)

        # Calculate the {r, K, fcol, M, D, i}-grids
        grids = self.calc_grids(r_min, r_max, Nr, NK, Nf, NM, ND, Ni, fcol_min, fcol_max)
        r     = grids['r']
        K     = grids['K']
        fcol  = grids['fcol']
        M     = grids['M']
        D     = grids['D']
        inc   = grids['inc']
        if (fcol is None): fcol = fcol_val
        
        # Parallel workhorse
        global task  # <-- HACK! Some weird thing about Pool and pickling and the function task() being inside a class
        def task(r):
            return self.task_fr(r, fcol, M, D, inc, f_fcol)
        if ( (Nproc is None) or (Nproc > cpu_count()) ): Nproc = cpu_count()-1
        with Pool(Nproc) as p:
            results = p.map(task, r)

        # Extract the results
        Nr  = np.size(r)
        f_r = np.zeros(Nr)
        for i in range(Nr):
            f_r[i] = results[i]

        # End time [s]
        toc = time.time()

        # Total run time [s]
        tictoc = toc - tic
        print("\nTOTAL RUN TIME (HH:MM:SS) = ", str(datetime.timedelta(seconds=tictoc)))
        print("")
        
        return r, f_r
        
#====================================================================================================
# ANALYSIS SCRIPTS

#----------------------------------------------------------------------------------------------------
# Normalize a distribution if it is not already (within some absolute tolerance)
def normalize(x, f_x, atol):
    norm = check_norm(x, f_x, atol=atol)
    if (norm is False):
        print("\nWARNING: Forcing normalization\n")
        f_x_norm = f_x / np.trapz(y=f_x, x=x)
    if (norm is True):
        f_x_norm = f_x
    return f_x_norm

#----------------------------------------------------------------------------------------------------
# Calculate the effect of fcol value on f_r(r) w/...
# ...fcol_val = fcolCF ---or--- [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
# ...fcol_err = 0.0
def fr_fcolVal(fpCF, fr2K, GRflag, fcol_val, r_min=None, r_max=None, Nproc=None,
               r_GR=0, i_GR=0, r_dGR=0, i_dGR=0, r_NT=0, r_dNT=0):
    
    inst = fK2fr(fpCF, fr2K, GRflag=GRflag, fcolMarg=False,
                 r_GR=r_GR, i_GR=i_GR, r_dGR=r_dGR, i_dGR=i_dGR, r_NT=r_NT, r_dNT=r_dNT)

    if isinstance(fcol_val, (int,float)):
        r, f_r = inst.calc_fr(fcol_val=fcol_val, r_min=r_min, r_max=r_max, Nproc=Nproc)
    else:
        r, f_r = [], []
        for i in range(np.size(fcol_val)):
            x, f_x = inst.calc_fr(fcol_val=fcol_val[i], r_min=r_min, r_max=r_max, Nproc=Nproc)
            r.append(x), f_r.append(f_x)

    return r, f_r

#----------------------------------------------------------------------------------------------------
# Calculate the effect of fcol error on f_r(r) w/...
# ...fcol_val = fcolCF
# ...fcol_err = [0.1, 0.2, 0.3]
def fr_fcolErr(fpCF, fr2K, GRflag, fcol_err, r_min=None, r_max=None, Nproc=None,
               r_GR=0, i_GR=0, r_dGR=0, i_dGR=0, r_NT=0, r_dNT=0):
    
    inst = fK2fr(fpCF, fr2K, GRflag=GRflag, fcolMarg=True,
                 r_GR=r_GR, i_GR=i_GR, r_dGR=r_dGR, i_dGR=i_dGR, r_NT=r_NT, r_dNT=r_dNT)

    if isinstance(fcol_err, (int,float)):
        r, f_r = inst.calc_fr(fcol_val=fcol_err, r_min=r_min, r_max=r_max, Nproc=Nproc)
    else:
        r, f_r = [], []
        for i in range(np.size(fcol_err)):
            x, f_x = inst.calc_fr(fcol_err=fcol_err[i], r_min=r_min, r_max=r_max, Nproc=Nproc)
            r.append(x), f_r.append(f_x)

    return r, f_r

#----------------------------------------------------------------------------------------------------
# Calculate the fcol value needed to give a specified black hole spin measurement (i.e., where CDF = 0.5)
def fcol_bhspin(fpCF, fr2K, GRflag, r_min, r_max, spin, fcolL, fcolR, atol,
                r_GR=0, i_GR=0, r_dGR=0, i_dGR=0, r_NT=0, r_dNT=0):

    # Defaults (sloppy, I know)
    f = h5py.File(fr2K, 'r')
    if (r_min is None): r_min = f.get('r_min')[()]  # [Rg]
    if (r_max is None): r_max = f.get('r_max')[()]  # [Rg]
    f.close()
    if (spin  is None): spin  = 0.0
    if (fcolL is None): fcolL = 1.0
    if (fcolR is None): fcolR = 2.4
    if (atol  is None): atol  = 1e-3

    # Convert the BH spin to the ISCO
    risco = spin2risco(spin)

    # Instance of the fK2fr class
    inst = fK2fr(fpCF, fr2K, GRflag=GRflag, fcolMarg=False,
                 r_GR=r_GR, i_GR=i_GR, r_dGR=r_dGR, i_dGR=i_dGR, r_NT=r_NT, r_dNT=r_dNT)
    
    # Initial left (L), right (R) fcol guesses...
    rL, f_rL = inst.calc_fr(fcol_val=fcolL, r_min=r_min, r_max=r_max)
    rR, f_rR = inst.calc_fr(fcol_val=fcolR, r_min=r_min, r_max=r_max)
    normL    = check_norm(rL, f_rL, atol=atol)
    normR    = check_norm(rR, f_rR, atol=atol)
    if (normL is False): print("\nWARNING: Try increasing fcolL (or decreasing r_min) because f_rL(r) is not normalized.\n")
    if (normR is False): print("\nWARNING: Try decreasing fcolR (or increasing r_max) because f_rR(r) is not normalized.\n")
    # ...and the corresponding BH spins
    '''
    rL_val, yL_val = interpolated_median(x=rL, y=f_rL)
    rR_val, yR_val = interpolated_median(x=rR, y=f_rR)
    if (risco < rL_val): print("\nERROR: Try decreasing fcolL because risco < rL.\n"); quit()
    if (risco > rR_val): print("\nERROR: Try increasing fcolR because risco > rR.\n"); quit()
    '''
    
    fcolM = 0.5 * (fcolL + fcolR)
    dr    = atol + 1
    while (dr > atol):
        # Calculate the normalized f_r for the fcol guess
        rM, f_rM = inst.calc_fr(fcol_val=fcolM, r_min=r_min, r_max=r_max)
        f_rM     = normalize(x=rM, f_x=f_rM, atol=atol)
        # Find where the CDF = 0.5
        rM_val, yM_val = interpolated_median(x=rM, y=f_rM)
        # Update the fcol guess
        if (risco <= rM_val): fcolR = fcolM
        if (risco >  rM_val): fcolL = fcolM
        fcolM = 0.5 * (fcolL + fcolR)
        dr    = np.abs(rM_val - risco)
    
    return fcolM

#----------------------------------------------------------------------------------------------------
# Write out the results to an HDF5 file
def fr_write(fout,
             fcolCF,   rCF,  f_rCF,
                       r,    f_r,
             fcol_err, rErr, f_rErr,
             fcol_val, rVal, f_rVal,
             r_min,    r_max,
             GRflag, r_GR, i_GR, r_dGR, i_dGR, r_NT, r_dNT,
             fcol_a0, fcol_aMax, fcol_aFe=None):
    f = h5py.File(fout, 'w')
    f.create_dataset('fcolCF',    data=fcolCF)
    f.create_dataset('rCF',       data=rCF)
    f.create_dataset('f_rCF',     data=f_rCF)
    f.create_dataset('r',         data=r)
    f.create_dataset('f_r',       data=f_r)
    f.create_dataset('fcol_err',  data=fcol_err)
    f.create_dataset('rErr',      data=rErr)
    f.create_dataset('f_rErr',    data=f_rErr)
    f.create_dataset('fcol_val',  data=fcol_val)
    f.create_dataset('rVal',      data=rVal)
    f.create_dataset('f_rVal',    data=f_rVal)
    f.create_dataset('r_min',     data=r_min)
    f.create_dataset('r_max',     data=r_max)
    f.create_dataset('GRflag',    data=GRflag)
    f.create_dataset('r_GR',      data=r_GR)   # [Rg]
    f.create_dataset('i_GR',      data=i_GR)   # [deg]
    f.create_dataset('r_dGR',     data=r_dGR)  # [Rg]
    f.create_dataset('i_dGR',     data=i_dGR)  # [deg]
    f.create_dataset('r_NT',      data=r_NT)   # [Rg]
    f.create_dataset('r_dNT',     data=r_dNT)  # [Rg]
    f.create_dataset('fcol_a0',   data=fcol_a0)
    f.create_dataset('fcol_aMax', data=fcol_aMax)
    if (fcol_aFe is not None): f.create_dataset('fcol_aFe', data=fcol_aFe)
    f.close()

#----------------------------------------------------------------------------------------------------
# Run the f_r(r) analysis
def fr_analysis(fout, fpCF, fr2K, GRflag, fcol_val, fcol_err,
                r_min=None, r_max=None, fcolL=None, fcolR=None, atol=None, aFe=None, Nproc=None,
                r_GR=0, i_GR=0, r_dGR=0, i_dGR=0, r_NT=0, r_dNT=0):

    # Collect fcolCF
    with open(fpCF, 'rb') as fp:
        fpick  = pickle.load(fp)
        fcolCF = fpick['fcolCF']

    # Calcualte the "recovered" f_r(r) w/ fcol_val = fcolCF
    print("\nCalculating the 'recovered' f_r(r) w/ fcol_val = fcolCF")
    r, f_r = fr_fcolVal(fpCF=fpCF, fr2K=fr2K, GRflag=GRflag, fcol_val=fcolCF, r_min=r_min, r_max=r_max, Nproc=Nproc,
                        r_GR=r_GR, i_GR=i_GR, r_dGR=r_dGR, i_dGR=i_dGR, r_NT=r_NT, r_dNT=r_dNT)

    # Collect f_rCF(r)
    rCF = r
    with open(fpCF, 'rb') as fp:
        fpick = pickle.load(fp)
        f_rCF = fpick['f_rCF'].pdf(rCF)

    # Calculate the effect of fcol error on f_r(r) w/ fcol_err = [0.1, 0.2, 0.3] (fcol_val = fcolCF)
    print("Calculating the effect of fcol error on f_r(r) w/ fcol_err = [0.1, 0.2, 0.3] (fcol_val = fcolCF)")
    rErr, f_rErr = fr_fcolErr(fpCF=fpCF, fr2K=fr2K, GRflag=GRflag, fcol_err=fcol_err, r_min=r_min, r_max=r_max, Nproc=Nproc,
                              r_GR=r_GR, i_GR=i_GR, r_dGR=r_dGR, i_dGR=i_dGR, r_NT=r_NT, r_dNT=r_dNT)

    # Calculate the effect of fcol value on f_r(r) w/ fcol_val = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    print("Calculating the effect of fcol value on f_r(r) w/ fcol_val = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]")
    rVal, f_rVal = fr_fcolVal(fpCF=fpCF, fr2K=fr2K, GRflag=GRflag, fcol_val=fcol_val, r_min=r_min, r_max=r_max, Nproc=Nproc,
                              r_GR=r_GR, i_GR=i_GR, r_dGR=r_dGR, i_dGR=i_dGR, r_NT=r_NT, r_dNT=r_dNT)
    
    # Determine the fcol needed to measure a non-spinning black hole
    print("Determining the fcol needed to measure a non-spinning black hole")
    fcol_a0 = fcol_bhspin(fpCF=fpCF, fr2K=fr2K, GRflag=GRflag, r_min=r_min, r_max=r_max, spin=0.0, fcolL=fcolL, fcolR=fcolR, atol=atol,
                          r_GR=r_GR, i_GR=i_GR, r_dGR=r_dGR, i_dGR=i_dGR, r_NT=r_NT, r_dNT=r_dNT)
    print("\nNon-spinning black hole (a = 0) --> fcol = ", fcol_a0)
    print("\n")
    
    # Determine the fcol needed to measure a maximally-spinning black hole
    print("Determining the fcol needed to measure a maximally-spinning black hole")
    fcol_aMax = fcol_bhspin(fpCF=fpCF, fr2K=fr2K, GRflag=GRflag, r_min=r_min, r_max=r_max, spin=0.998, fcolL=fcolL, fcolR=fcolR, atol=atol,
                            r_GR=r_GR, i_GR=i_GR, r_dGR=r_dGR, i_dGR=i_dGR, r_NT=r_NT, r_dNT=r_dNT)
    print("\nMaximally-spinning black hole (a = 0.998) --> fcol = ", fcol_aMax)
    print("\n")
    
    # Determine the fcol needed to match the iron line spin measurement
    print("Determining the fcol needed to match the iron line spin measurement")
    if (aFe is None): fcol_aFe = None
    else:             fcol_aFe = fcol_bhspin(fpCF=fpCF, fr2K=fr2K, GRflag=GRflag, r_min=r_min, r_max=r_max, spin=aFe, fcolL=fcolL, fcolR=fcolR, atol=atol,
                                             r_GR=r_GR, i_GR=i_GR, r_dGR=r_dGR, i_dGR=i_dGR, r_NT=r_NT, r_dNT=r_dNT)
    print("\nIron line matching black hole spin --> fcol = ", fcol_aFe)
    print("\n")

    # Write out the results
    print("Writing out the results\n")
    fr_write(fout=fout, fcolCF=fcolCF, rCF=rCF, f_rCF=f_rCF, r=r, f_r=f_r,
             fcol_err=fcol_err, rErr=rErr, f_rErr=f_rErr,
             fcol_val=fcol_val, rVal=rVal, f_rVal=f_rVal,
             r_min=r_min, r_max=r_max, fcol_a0=fcol_a0, fcol_aMax=fcol_aMax, fcol_aFe=fcol_aFe,
             GRflag=GRflag, r_GR=r_GR, i_GR=i_GR, r_dGR=r_dGR, i_dGR=i_dGR, r_NT=r_NT, r_dNT=r_dNT)

#====================================================================================================
