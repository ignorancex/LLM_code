""" 
Manifest-Inclination (mi) Unidirection and Bidirectional Models
- Implemented in Numpy
- Malmquist-bias correction can be disabled by setting ml=None
- All models allow fixing parameters

Version: 20250915
"""
import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
import multiprocessing, emcee, os, corner

# prep for multithreading
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass  # Context already set, proceed
    
# disable automatic parallelization in NumPy
os.environ["OMP_NUM_THREADS"] = "1"

""" Probability Function for the Bidirectional Model """
def prob_mi_bidirect(v, m, d, theta, ml):
    """
    Conditional pdf of m: p(m|v,d) using NumPy, for vector inputs
    """
    # Define logV grid
    lgV = np.linspace(-1, 1, 1024)
    
    # Expand dimensions for broadcasting
    v = v[:, np.newaxis]  # Shape: (n, 1)
    m = m[:, np.newaxis]  # Shape: (n, 1)
    d = d[:, np.newaxis]  # Shape: (n, 1)
    lgV = lgV[np.newaxis, :]  # Shape: (1, 1024)

    # Deviate in logV: (n, 1024)
    dv = (lgV - v) / theta[3]
        
    # ln Schechter function (1, 1024); ln10 = 2.30258509
    lgX = theta[1] * lgV - theta[4]
    lnVF = 2.30258509 * (theta[5] + 1) * lgX - 10**lgX
    
    # Deviate in mass (n, 1024)
    dm = (m + d - (theta[1] * lgV + theta[0])) / theta[2]
    # Numerator integrand function (n, 1024)
    fexp = np.exp(-dv**2/2 - dm**2/2 + lnVF)

    # Denominator integrand functions (n, 1024)
    if ml is not None:
        # Deviate in mass limit (n, 1024)
        dml = (ml + d - (theta[1] * lgV + theta[0])) / theta[2]
        ferf = erfc(dml/1.4142135) * np.exp(-dv**2/2 + lnVF)
    else:
        ferf = 2 * np.exp(-dv**2/2 + lnVF)
    
    # Integration over dv axis (axis=1)
    num = np.sum(fexp, axis=1)  # shape: (n,)
    den = np.sum(ferf, axis=1)  # shape: (n,)
    
    # Calculate result
    const = 0.79788456 / theta[2]  # sqrt(2/PI) = 0.79788456
    valid = (num > 0) & (den > 0)  # Shape: (n,)
    result = np.zeros_like(den)    # Shape: (n,)
    result[valid] = const * num[valid] / den[valid]

    return result

""" Likelihood Function for the Bidirectional Model """
def lnL_mi_bidirect(theta, bounds, vs, ms, ds, ml, fixed_params=None, fixed_values=None):
    """
        vs, ms, ds: input data as numpy arrays

        theta: numpy array of free parameters
        Full theta: [0] intercept (gamma), [1] slope (beta), 
                    [2] m scatter (sig_m), [3] v scatter (sig_v), 
                    [4] knee of VF (v*),   [5] faint-end slope (alpha)

        ml: constant, detection limit of apparent mass
    
    Optional: the user can specify fixed parameters and their values:
        fixed_params: indices of parameters to fix (e.g., [4, 5] for v* and alpha)
        fixed_values: values of fixed parameters (e.g., [0.3, -1.27])
    """
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # Create full theta array, combining free and fixed parameters in the correct order
        if fixed_params is not None and fixed_values is not None:
            full_theta = np.zeros(6)
            free_mask = np.ones(6, dtype=bool)
            free_mask[fixed_params] = False
            full_theta[fixed_params] = fixed_values
            full_theta[free_mask] = theta
            theta = full_theta.copy() 
        # compute prob for every data point
        ps = prob_mi_bidirect(vs, ms, ds, theta, ml)
        # return the sum of the logarithmic
        return np.sum(np.log(ps[ps > 0]))
    return -np.inf

""" Likelihood Function for the Forward Model """
def lnL_mi_forward(theta, bounds, vs, ms, ds, ml, fixed_params=None, fixed_values=None):
    """ ln likelihood of the data set {vs, ms, ds} 
    given the model (theta), selection function (ml) """
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # Create full theta array, combining free and fixed parameters in the correct order
        if fixed_params is not None and fixed_values is not None:
            full_theta = np.zeros(3)
            free_mask = np.ones(3, dtype=bool)
            free_mask[fixed_params] = False
            full_theta[fixed_params] = fixed_values
            full_theta[free_mask] = theta
            theta = full_theta.copy() 
        # intercept, slope, m dispersion, velocity noise,
        # velocity function bv*, and faint-end slope
        c, b, sigm = theta
        # input data arrays
        dm  = (ms+ds-(b*vs+c))/sigm
        # compute conditional prob for each data point
        if ml is not None:
            dml = (ml+ds-(b*vs+c))/sigm
            ps = np.sqrt(2/np.pi)/sigm * np.exp(-dm**2/2)/erfc(dml/np.sqrt(2))
        else:
            ps = np.sqrt(2/np.pi)/sigm * np.exp(-dm**2/2)/2
        # return the sum of the logarithmic
        return np.sum(np.log(ps[ps > 0]))
    return -np.inf

""" Likelihood Function for the Inverse Model """
def lnL_mi_inverse(theta, bounds, vs, ms, ds, fixed_params=None, fixed_values=None):
    """ ln likelihood of the data set {vs, ms, ds} given the model (theta) """
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # Create full theta array, combining free and fixed parameters in the correct order
        if fixed_params is not None and fixed_values is not None:
            full_theta = np.zeros(3)
            free_mask = np.ones(3, dtype=bool)
            free_mask[fixed_params] = False
            full_theta[fixed_params] = fixed_values
            full_theta[free_mask] = theta
            theta = full_theta.copy() 
        # BTFR intercept, slope, dispersion in logV
        c, b, sigw = theta
        # deviates between TFR-predicted and observed logV 
        dv = (vs - (ms+ds-c)/b)/sigw
        # compute prob for each data point
        ps = np.exp(-dv**2/2) / (np.sqrt(2*np.pi)*sigw)
        # return the sum of the logarithmic
        return np.sum(np.log(ps[ps > 0]))
    return -np.inf

""" sampling posterior with emcee """
def emcee_mi_numpy(vs,ms,ds,               # input data arrays: logV-2.5, logmb, d=2logD
            outdir='mi_numpy',             # output folder
            method='forward',              # models: forward, inverse, or bidirect
            ml=None,                       # log apparent mass limit
            converge_check=False,          # check chain convergence before proceeding?
            ncpu=os.cpu_count(),           # number of CPUs to use
            nsteps=100, nrepeat=1,         # emcee iterations and number of repeats
            fixed_params=None,             # list of indices of parameters to fix
            fixed_values=None):            # list of values for fixed parameters
    
    """ Validate Inputs """
    methods = {'forward','inverse','bidirect'}
    if method not in methods:
        raise ValueError(f"Invalid Input Parameters: {method=}")

    """ Output Folder """
    if not os.path.exists(outdir): os.mkdir(outdir)
    ndata = len(ds)
    suffix = f'd{np.median(ds):.2f}_n{ndata}'

    """ Model Parameters: latex names for plots and limits for flat priors """
    # sigma = 0 causes issues, so use 1e-3 as the minimum
    eps = 1e-3
    if method == 'forward':
        params = ["$\\gamma$", "$\\beta$", "$\\sigma_m$"]
        bounds = np.array([[10.0, 2.5, eps],
                           [11.0, 4.5, 0.3]])
    elif method == 'inverse':
        params = ["$\\gamma$", "$\\beta$", "$\\sigma_v$"]
        bounds = np.array([[10.0, 2.5, eps],
                           [11.0, 4.5, 0.1]])
    elif method == 'bidirect':
        params = ["$\\gamma$", "$\\beta$", "$\\sigma_m$", "$\\sigma_v$", "$\\beta v_*$", "$\\alpha$"]
        bounds = np.array([[10.0, 2.5, eps, eps, -1.0, -2.0],
                           [11.0, 4.5, 0.3, 0.1,  1.0,  0.0]])
    
    """ remove fixed parameters from params and bounds """
    if fixed_params is not None:
        params = [p for i, p in enumerate(params) if i not in fixed_params]
        bounds = np.delete(bounds, fixed_params, axis=1)

    """ initial starting positions of the walkers """
    ndim = len(bounds[0,:])
    nwalkers = ncpu*int(np.ceil(2*ndim/ncpu)) 
    cnter = (bounds[1,:]+bounds[0,:])/2 # distribution means 
    scale = (bounds[1,:]-bounds[0,:])/2 # width of uniform distribution
    pos = cnter + scale * (np.random.uniform(size=(nwalkers, ndim))-0.5)
    # clip initial positions outside of bounds
    for i in range(ndim):
        pos[:,i] = np.clip(pos[:,i], bounds[0,i], bounds[1,i]) 

    """ MCMC-sampling """
    print(f'\n{method=}, {ml=}, {ndata=}, {nsteps=}, {nrepeat=}, {nwalkers=}, {ndim=}')
    for irepeat in range(1,nrepeat+1):
        print(f'\nThis is {irepeat=} out of {nrepeat=}')
        # set up backend file to save MCMC samples
        emcfile = outdir+f'/emcee_{suffix}.h5'
        backend = emcee.backends.HDFBackend(emcfile)
        if not os.path.exists(emcfile): 
            print(f'Reset backend because {emcfile} not present')
            backend.reset(nwalkers, ndim)
            print(f'Start MCMC from initial random positions')
            instate = pos
            chainexist = False
        else: 
            print('Start MCMC from where left off the last time')
            instate = None
            chainexist = True

        with multiprocessing.Pool() as pool:
            if method == 'forward':
                sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                    lnL_mi_forward, args=(bounds,vs,ms,ds,ml,fixed_params,fixed_values), 
                    pool=pool, backend=backend)
            elif method == 'inverse':
                sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                    lnL_mi_inverse, args=(bounds,vs,ms,ds,fixed_params,fixed_values), 
                    pool=pool, backend=backend)
            elif method == 'bidirect':
                sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                    lnL_mi_bidirect, args=(bounds,vs,ms,ds,ml,fixed_params,fixed_values), 
                    pool=pool, backend=backend)
            # check if chain is already long enough
            if chainexist and converge_check:
                chainshape = sampler.get_chain().shape
                tau = sampler.get_autocorr_time(quiet=True)
                burnin = int(2 * np.max(tau))
                thin = int(0.5 * np.min(tau))
                flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
                mids = np.array([np.percentile(flat_samples[:,i],50) for i in range(ndim)])
                if chainshape[0] > 50*np.max(tau): 
                    print("[Break]: Chain length already exceeds 50x Auto-correlation Time")
                    break                
            # start walking
            sampler.run_mcmc(instate, nsteps, progress=True)
        
        """ after nsteps, make a corner plot over full bounded range """
        # use trimmed, thinned, flattened sample for corner plots
        tau = sampler.get_autocorr_time(quiet=True)
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        mids = np.array([np.percentile(flat_samples[:,i],50) for i in range(ndim)])
        _ = corner.corner(flat_samples, labels=params, 
                title_quantiles=[0.16,0.50,0.84], quantiles=[0.50],
                show_titles=True, plot_contours=False, plot_density=False, bins=50,
                range=list(zip(bounds[0],bounds[1])))
        plt.savefig(outdir+f'/corner_{suffix}_{(sampler.get_chain().shape)[0]}.png', bbox_inches='tight')
        plt.clf()

    """ return best-fit parameters and flattened chains """
    return mids, flat_samples