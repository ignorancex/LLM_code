""" 
Latent-Inclination (li) Unidirectional and Bidirectional Models
- Implemented in NumPy
? Allow disabling Malmquist-bias correction by setting ml=None ?
? Allow fixing parameters ?

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

""" Probability and Likelihood Functions for the Bidirectional Model """
def prob_li_bidirect(bi,fifft, md,mld,bw,sigm, bsigw,VF_a,VF_vs):
    """ 
    Conditional pdf of m: p(m|w,d), for scaler inputs
    Note: Vectorized version runs 1.8x slower than list comprehension
    """
    # FFT-integration over i to compute (f*g)(w)
    lnVF = 2.30258509*(VF_a+1)*(bi-VF_vs) - 10**(bi-VF_vs) # ln10 = 2.30258509
    gexp = np.exp(-(md-bi)**2/(2*sigm**2) + lnVF) 
    gerf = erfc((mld-bi)/(1.41421356*sigm)) * np.exp(lnVF) # sqrt(2) = 1.41421356
    # convolution theorem: (f*g)(w) = IFFT(FFT(f) * FFT(g))
    Fexp = np.fft.ifft(np.fft.fft(gexp)*fifft).real
    Ferf = np.fft.ifft(np.fft.fft(gerf)*fifft).real
    # use concatenate to replace fftshift, 511 = 1024 // 2 - 1
    Fexp = np.concatenate((Fexp[511:], Fexp[:511]))
    Ferf = np.concatenate((Ferf[511:], Ferf[:511]))
    # direct integration over w
    Gfun = np.exp(-(bw-bi)**2/(2*bsigw**2))
    num = np.sum(Fexp*Gfun)
    den = np.sum(Ferf*Gfun)
    # get the ratio
    if num > 0 and den > 0:
        const = 0.79788456/sigm # the term before the integrals, sqrt(2/PI) = 0.79788456
        return const*num/den
    return 0.0

def lnL_li_bidirect(theta, bounds, ws, ms, ds, ml):
    """ 
    ln likelihood of the data set {ws, ms, ds}, calculated w/ list comprehension

        theta: numpy array, 
            [0] intercept (gamma), [1] slope (beta), 
            [2] m scatter (sig_m), [3] v scatter (sig_v), 
            [4] knee of VF (v*),   [5] faint-end slope (alpha)
    """
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # get model parameters (c = gamma, b = beta, VF_vs = v_\star, VF_a = \alpha)
        c, b, sigm, sigw, VF_vs, VF_a = theta
        # hat_i = log sin i
        hi = np.linspace(-2.5,2.5,1024) 
        # pdf of hat_i for random orientation
        msk = hi < 0 
        fi = np.zeros_like(hi)
        fi[msk] = 10**(2*hi[msk]) / np.sqrt(1 - 10**(2*hi[msk]))
        # Fourier transform of p(i)
        fifft = np.fft.fft(fi)
        # bi = beta * log sin i
        bi = b*hi
        bsigw = b*sigw
        # input data arrays
        mds  = ms+ds-c
        mlds = ml+ds-c
        bws  = b*ws
        # compute prob for every data point
        ps = np.array([prob_li_bidirect(bi,fifft, md,mld,bw, sigm,bsigw,VF_a,VF_vs) for (md,mld,bw) in zip(mds,mlds,bws)])
        # return the sum of the logarithmic
        return np.sum(np.log(ps[ps > 0]))
    return -np.inf

""" Probability and Likelihood Functions for the Forward Model """
def prob_li_forward(bi,ft, x,xl,bw, sigm,alpha):
    """ conditional pdf of m: p(m|w,d ; sig_m, beta, alpha)"""
    # velocity function, (blogW - bv*) - blogsini = b log(W/sini) - bv*
    lnVF = 2.30258509*(alpha+1)*(bw-bi) - 10**(bw-bi) # ln10 = 2.30258509
    # x+blogsini = logM-(blogW+c) + blogsini = logM-(blog(W/sini)+c)
    fexp = np.exp(-(x+bi)**2/(2*sigm**2)+lnVF) * ft
    ferf = erfc((xl+bi)/(1.4142135*sigm))*np.exp(lnVF) * ft
    # simple integration
    num = np.sum(fexp)
    den = np.sum(ferf) 
    if num > 0 and den > 0:
        const = np.sqrt(2/np.pi)/sigm # the term before the integrals
        return const*num/den
    return 0.0

def lnL_li_forward(theta, bounds, ws, ms, ds, ml):
    """ ln likelihood of the data set {ws, ms, ds} 
    given the model (theta), selection function (ml) """
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # intercept, slope, m dispersion, velocity noise,
        # velocity function bv*, and faint-end slope
        c, b, sigm, VF_vs, VF_a = theta
        # t = sini^2 from 0 to 1
        t = np.linspace(1e-4, 0.9999, 1000)
        ft = 1/np.sqrt(1-t)
        bi = b * np.log10(t)/2
        # input data arrays
        xs  = ms+ds-(b*ws+c) # deviate between observed and TFR mass
        xls = ml+ds-(b*ws+c) # deviate between mass limit and TFR mass
        bws = b*ws-VF_vs     # bw - v* = logX for Schechter velocity function
        # compute conditional prob for each data point
        ps = np.array([prob_li_forward(bi,ft, x,xl,bw, sigm,VF_a) for (x,xl,bw) in zip(xs,xls,bws)])
        # return the sum of the logarithmic
        return np.sum(np.log(ps[ps > 0]))
    return -np.inf

""" Likelihood Function for the Inverse Model """
def lnL_li_inverse(theta, bounds, ws, ms, ds):
    """ ln likelihood of the data set {ws, ms, ds} given the model (theta) """
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # BTFR intercept, slope, dispersion in logW
        c, b, sigw = theta
        # t := sini^2 from 0 to 1
        t = np.linspace(1e-4, 0.9999, 1000)
        hi = np.log10(t)/2 # hat_i := logsini = (log sini^2)/2
        ft = 1/np.sqrt(1-t) # pdf(t)
        dt = t[1]-t[0]
        # deviates between TFR-predicted and observed logW 
        xs = (ms+ds-c)/b - ws
        # compute prob for each data point
        ps = np.array([np.sum(np.exp(-(x+hi)**2/(2*sigw**2)) * ft) for x in xs])
        # properly normalize
        ps *= dt/(2*np.sqrt(2*np.pi)*sigw)
        # return the sum of the logarithmic
        return np.sum(np.log(ps[ps > 0]))
    return -np.inf

def emcee_li_numpy(ws,ms,ds,            # input data arrays: logW-2.5, logmb, 2logD
            outdir='liuni_numpy',          # output folder
            method='forward',              # models: forward, inverse, or dual
            ml=5.736,                      # log apparent mass limit
            converge_check=False,          # check chain convergence before proceeding?
            ncpu=os.cpu_count(),           # number of CPUs to use
            nsteps=100, nrepeat=1):        # emcee iterations and number of repeats
    
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
        params = ["$\\gamma$", "$\\beta$", "$\\sigma_m$", "$\\beta v_*$", "$\\alpha$"]
        bounds = np.array([[10.0, 2.5, eps, -1.0, -2.0],
                           [11.0, 4.5, 0.3,  1.0,  0.0]])
    elif method == 'inverse':
        params = ["$\\gamma$", "$\\beta$", "$\\sigma_w$"]
        bounds = np.array([[10.0, 2.5, eps],
                           [11.0, 4.5, 0.1]])
    elif method == 'bidirect':
        params = ["$\\gamma$", "$\\beta$", "$\\sigma_m$", "$\\sigma_w$", "$\\beta v_*$", "$\\alpha$"]
        bounds = np.array([[10.0, 2.5, eps, eps, -1.0, -2.0],
                           [11.0, 4.5, 0.3, 0.1,  1.0,  0.0]])

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
                    lnL_li_forward, args=(bounds,ws,ms,ds,ml), 
                    pool=pool, backend=backend)
            elif method == 'inverse':
                sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                    lnL_li_inverse, args=(bounds,ws,ms,ds), 
                    pool=pool, backend=backend)
            elif method == 'bidirect':
                sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                    lnL_li_bidirect, args=(bounds,ws,ms,ds,ml), 
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