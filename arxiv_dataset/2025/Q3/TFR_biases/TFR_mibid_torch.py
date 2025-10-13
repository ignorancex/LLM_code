""" 
Manifest-Inclination Bidirection Model (mibid)
- Implemented in PyTorch (torch)
- Malmquist-bias correction can be disabled by setting ml=None
- Allowed fixing parameters

Version: 20250915
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import emcee, os, corner

# MPS prefers float32
dtype = np.float32  
# set device to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def prob_mibid_torch(v, m, d, theta, ml):
    """
    Conditional pdf of m: p(m|v,d) using PyTorch with MPS device, for vector inputs
    """
    # Convert inputs to tensors and move to MPS device
    theta_tensor = torch.from_numpy(theta.astype(dtype)).to(device) 
    v = torch.from_numpy(v.astype(dtype)).to(device)
    m = torch.from_numpy(m.astype(dtype)).to(device)
    d = torch.from_numpy(d.astype(dtype)).to(device)

    # Define logV grid
    lgV = torch.linspace(-1, 1, 1024, device=device)
 
    # Expand dimensions for broadcasting
    v = v.unsqueeze(-1) # Shape: (n, 1)
    m = m.unsqueeze(-1) # Shape: (n, 1)
    d = d.unsqueeze(-1) # Shape: (n, 1)
    lgV = lgV.unsqueeze(0) # Shape: (1, 1024)

    # Deviate in logV: (n, 1024)
    dv = (lgV - v)/theta_tensor[3]
        
    # ln Schechter function (1, 1024)
    lgX = theta_tensor[1] * lgV - theta_tensor[4]
    lnVF = 2.3025851 * (theta_tensor[5] + 1) * lgX - 10**lgX
    
    # Deviate in mass (n, 1024)
    dm = (m + d - (theta_tensor[1] * lgV + theta_tensor[0])) / theta_tensor[2]
    # Integrand functions (n, 1024)
    fexp = torch.exp(-dv**2/2 - dm**2/2 + lnVF) 
    if ml is not None:
        # Deviate from mass limit (n, 1024)
        dml = (ml + d - (theta_tensor[1] * lgV + theta_tensor[0])) / theta_tensor[2]
        ferf = (1.0-torch.erf(dml/1.4142135)) * torch.exp(-dv**2/2 + lnVF)
    else:
        # Integrand functions (n, 1024)
        ferf = 2.0 * torch.exp(-dv**2/2 + lnVF)    
    
    # Integration over dv axis (axis=1)
    num = torch.sum(fexp, dim=1) # shape: (n,)
    den = torch.sum(ferf, dim=1) # shape: (n,)
    
    # Calculate result
    const = 0.79788456 / theta_tensor[2]  # sqrt(2/PI)
    valid = (num > 0) & (den > 0)  # Shape: (n,)
    result = torch.zeros_like(den)  # Shape: (n,)
    result[valid] = const * num[valid] / den[valid]

    return result
    
def lnL_mibid_torch(theta, bounds, vs, ms, ds, ml, fixed_params=None, fixed_values=None):
    """ 
    ln likelihood of the data set {vs, ms, ds} given the model (theta), 
    selection function (ml), and fixed parameters 
    
        v, m, d: input data as numpy arrays

        theta: numpy array of free parameters
        Full theta: [0] intercept (gamma), [1] slope (beta), 
                    [2] m scatter (sig_m), [3] v scatter (sig_v), 
                    [4] knee of VF (v*),   [5] faint-end slope (alpha)

        ml: constant, detection limit of apparent mass
    
        Optional: the user can specify fixed parameters and their values:
        fixed_params: list of indices of parameters to fix (e.g., [4, 5] for v* and alpha)
        fixed_values: list of values for fixed parameters (e.g., [0.3, -1.27])    
    """
    # Only calculate when free parameters are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # Create full theta array, combining free and fixed parameters
        if fixed_params is not None and fixed_values is not None:
            full_theta = np.zeros(6)
            free_mask = np.ones(6, dtype=bool)
            free_mask[fixed_params] = False
            full_theta[fixed_params] = fixed_values
            full_theta[free_mask] = theta
            theta = full_theta.copy() 
        # Compute prob for every data point
        ps = prob_mibid_torch(vs, ms, ds, theta, ml)
        # Return the sum of the logarithmic to CPU as NumPy scalar
        return torch.sum(torch.log(ps[ps > 0])).cpu().numpy()
    return -np.inf

def emcee_mibid_torch(vs, ms, ds,          # input data: logV-2.5, logmb, 2logD
            outdir='mibid_torch',          # output folder
            ml=5.736,                      # log apparent mass limit
            converge_check=False,          # check chain convergence before proceeding?
            ncpu=os.cpu_count(),           # number of CPUs (to be consistent w/ NumPy version)
            nsteps=100, nrepeat=1,         # emcee iterations and number of repeats
            fixed_params=None,             # list of indices of parameters to fix
            fixed_values=None):            # list of values for fixed parameters

    """ Output Folder """
    method = 'bidirmps'
    ndata = len(ds)
    if not os.path.exists(outdir): os.mkdir(outdir)
    suffix = f'd{np.median(ds):.2f}_n{ndata}'

    """ Model Parameters: latex names for plots and limits for flat priors """
    # sigma = 0 causes issues, so use 1e-3 as the minimum
    eps = 1e-3
    params = ["$\\gamma$", "$\\beta$", "$\\sigma_m$", "$\\sigma_w$", "$\\beta v_*$", "$\\alpha$"]
    bounds = np.array([[10.0, 2.5, eps, eps, -1.0, -2.0],
                       [11.0, 4.5, 0.3, 0.1,  1.0,  0.0]])

    """ remove fixed parameters from params and bounds """
    if fixed_params is not None:
        params = [p for i, p in enumerate(params) if i not in fixed_params]
        bounds = np.delete(bounds, fixed_params, axis=1)

    """ initial starting positions of the walkers """
    ndim = len(params)
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
        # set up backend to save MCMC samples
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
        # set up sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                lnL_mibid_torch, args=(bounds, vs, ms, ds, ml, fixed_params, fixed_values), 
                backend=backend)
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

    return mids, flat_samples