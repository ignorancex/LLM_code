""" 
Latent-Inclination Bidirectional Models (libid), Implemented in PyTorch
Version: 20250905
"""
import numpy as np
import matplotlib.pyplot as plt
import emcee, os, corner
import torch

# MPS prefers float32
dtype = np.float32  
# set device to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def prob_libid_torch(w, m, d, theta, ml, fixed_params=None, fixed_values=None):
    """
    Conditional pdf of m: p(m|v,d) using PyTorch with MPS device, for vector inputs
    
    Input:
        w, m, d: input data as numpy arrays

        theta: numpy array of free parameters
        Full theta: [0] intercept (gamma), [1] slope (beta), 
                    [2] m scatter (sig_m), [3] v scatter (sig_v), 
                    [4] knee of VF (v*),   [5] faint-end slope (alpha)

        ml: constant, detection limit of apparent mass
    
    Optional Input: 
        the user can specify fixed parameters and their values
        fixed_params: list of indices of parameters to fix (e.g., [4, 5] for v* and alpha)
        fixed_values: list of values for fixed parameters (e.g., [0.3, -1.27])
    """
    # Create full theta array, combining free and fixed parameters
    full_theta = np.zeros(6, dtype=dtype)
    if fixed_params is not None and fixed_values is not None:
        free_idx = 0
        for i in range(6):
            if i in fixed_params:
                full_theta[i] = fixed_values[fixed_params.index(i)]
            else:
                full_theta[i] = theta[free_idx]
                free_idx += 1
    else:
        full_theta = theta

    # convert to tensors
    theta = torch.tensor(full_theta, dtype=torch.float32, device=device)
    w = torch.from_numpy(w.astype(dtype)).to(device)
    m = torch.from_numpy(m.astype(dtype)).to(device)
    d = torch.from_numpy(d.astype(dtype)).to(device)

    # Broadcast m, d, w to handle vectorized computation
    md = m + d - theta[0]  # Shape: (n,)
    mld = ml + d - theta[0]  # Shape: (n,)
    bw = w * theta[1]  # Shape: (n,)

    # Define i grid (same for all inputs, shape: (1024,))
    i = torch.linspace(-2.5, 2.5, 1024, device=device)
    bi = i * theta[1]  # Shape: (1024,)
    valid_mask = i < 0  # Shape: (1024,)
    fi = torch.zeros_like(i, device=device)
    fi[valid_mask] = 10**(2 * i[valid_mask]) / torch.sqrt(1 - 10**(2 * i[valid_mask]))

    # Velocity distribution function (shape: (1024,)), ln10 = 2.30258509
    lnVF = 2.30258509 * (theta[5] + 1) * (bi - theta[4]) - 10**(bi - theta[4])

    # Expand md and mld to (n, 1024) for broadcasting with bi
    md_expanded = md.unsqueeze(-1)  # Shape: (n, 1)
    mld_expanded = mld.unsqueeze(-1)  # Shape: (n, 1)
    bi_expanded = bi.unsqueeze(0)  # Shape: (1, 1024)

    # Compute gexp and gerf with broadcasting (shape: (n, 1024)), 
    gexp = torch.exp(-(md_expanded - bi_expanded)**2 / (2 * theta[2]**2) + lnVF)
    # Workaround for torch.erfc on MPS: use 1 - erf; sqrt(2) = 1.41421356
    gerf = (1.0 - torch.erf((mld_expanded - bi_expanded) / (1.41421356 * theta[2]))) * torch.exp(lnVF)

    # FFT-based convolution (shape: (n, 1024))
    fifft = torch.fft.fft(fi)  # Shape: (1024,)
    Fexp = torch.fft.ifft(torch.fft.fft(gexp, dim=1) * fifft, dim=1).real
    Ferf = torch.fft.ifft(torch.fft.fft(gerf, dim=1) * fifft, dim=1).real
    # 511 = 1024 // 2 - 1 
    Fexp = torch.cat((Fexp[:, 511:], Fexp[:, :511]), dim=1)
    Ferf = torch.cat((Ferf[:, 511:], Ferf[:, :511]), dim=1)

    # Expand bw to (n, 1024) for Gfun computation
    bw_expanded = bw.unsqueeze(-1)  # Shape: (n, 1)
    Gfun = torch.exp(-(bw_expanded - bi_expanded)**2 / (2 * (theta[1] * theta[3])**2))  # Shape: (n, 1024)

    # Sum over the i dimension (dim=1) to get num and den (shape: (n,))
    num = torch.sum(Fexp * Gfun, dim=1)
    den = torch.sum(Ferf * Gfun, dim=1)

    # Compute the ratio with masking for valid values
    const = 0.79788456 / theta[2]  # sqrt(2/PI) = 0.79788456
    valid = (num > 0) & (den > 0)  # Shape: (n,)
    result = torch.zeros_like(num)  # Shape: (n,)
    result[valid] = const * num[valid] / den[valid]

    return result

def lnL_libid_torch(theta, bounds, ws, ms, ds, ml, fixed_params=None, fixed_values=None):
    """ ln likelihood of the data set {ws, ms, ds} 
    given the model (theta), selection function (ml), and velocity noise (sds) """
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # compute prob for every data point
        ps = prob_libid_torch(ws, ms, ds, theta, ml, fixed_params, fixed_values)
        # return the sum of the logarithmic to CPU as NumPy scaler
        return torch.sum(torch.log(ps[ps > 0])).cpu().numpy()
    return -np.inf

def emcee_libid_torch(ws, ms, ds,          # input data: logW, logmb, 2logD
            outdir='libid_torch',          # output folder
            ml=5.736,                      # log apparent mass limit
            converge_check=False,          # check chain convergence before proceeding?
            ncpu=os.cpu_count(),           # number of CPUs (to be consistent w/ NumPy version)
            nsteps=100, nrepeat=1,         # emcee iterations and number of repeats
            fixed_params=None,             # list of indices of parameters to fix
            fixed_values=None):            # list of values for fixed parameters)

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
    ndim = len(bounds[0,:])
    nwalkers = ncpu*int(np.ceil(2.5*ndim/ncpu)) 
    cnter = (bounds[1,:]+bounds[0,:])/2 # distribution means 
    scale = (bounds[1,:]-bounds[0,:])/2 # width of uniform distribution
    pos = cnter + scale * (np.random.uniform(size=(nwalkers, ndim))-0.5)
    # clip initial positions outside of bounds
    for i in range(ndim):
        pos[:,i] = np.clip(pos[:,i], bounds[0,i], bounds[1,i]) 

    """ MCMC-sampling """
    print(f'\n{method=}, {ndata=}, {nsteps=}, {nrepeat=}, {nwalkers=}, {ndim=}')
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
                lnL_libid_torch, args=(bounds, ws, ms, ds, ml, fixed_params, fixed_values), 
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

    """ return best-fit parameters and flattened chains """
    return mids, flat_samples