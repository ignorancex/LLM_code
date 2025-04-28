import os
import sys
import logging
import glob
import numpy as np
import pycbc
from pycbc.inference.io import loadfile
from pycbc.pool import BroadcastPool
from scipy.stats import gaussian_kde, uniform
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize']=(11,6)
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize']=20
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "STIX"
rcParams["mathtext.fontset"] = "stix"

eosdir = '/home/abandopa/projects/github-bns-eos-population/eos_population/eft_cache/'
eosdir += 'eos_data/nsat'

def extract_samples(postfile):
    samples=np.loadtxt(postfile,unpack=True)
    return samples

def plot_errorbar(data,frac_error,ax,ax2,eosid,tag,rad_value,det,years,color1,color2):
    yerr = [np.abs(data[:, i] - data[:, 1]) for i in [0, 2]]
    ax.fill_between(np.arange(len(data)) + 1, data[:, 2], data[:, 0],
                    color=color1, alpha=0.5)
    x=(years/len(data))*np.linspace(0,len(data),len(data))
    # plot EOS value
    label='EOS {} '.format(str(eosid))+tag
    ax.axhline(rad_value, ls='dashed',label=label,color=color2)
    ax.set(xlabel=r'Number of years of operation at'+det+'sensitivity', ylabel=r'$R_{1.4} \, [\mathrm{km}]$',
           xlim=(0, (years/len(data))*(len(data) + 1)))
    ax.set_ylim(10.0,13.5)
    ax2.semilogy(x,frac_error,linestyle=':',color=color2)
    ax2.set_ylabel(r'Fractional uncertainty $\Delta \rm R/\rm R$')
    ax2.set_ylim(0.0001,0.06)
    return ax,ax2

def radius14_prior(x):
    datfiles = glob.glob('{}/*.dat'.format(eosdir))
    rprior_samples = []
    for f in datfiles:
        rdat, mdat, _ = np.loadtxt(f, unpack=True)
        interp = interp1d(mdat, rdat)
        rprior_samples.append(interp(1.4))
    rprior = gaussian_kde(rprior_samples)
    return rprior(x)

def calc_kdes(data):
    postdata, x = data
    kdes = [gaussian_kde(d, bw_method=bw)(x) for d in postdata]
    return kdes

def combine_posteriors(posteriors, x):
    prior = radius14_prior(x)
    cum_cpost_arr = np.ones(len(prior))
    ci_bounds = []
    fractional_error=[]
    thatx = x.copy()
    priorinterp = interp1d(x, prior)
    for j, post in enumerate(posteriors):
        postinterp = interp1d(x, post)
        cpostinterp = interp1d(thatx, cum_cpost_arr)
        if j == 0:
            xmin, xmax = x.min(), x.max()
        else:
            xmin, xmax = thatx[np.searchsorted(cdf, [0.01, 0.99])]
        xnew = np.linspace(xmin, xmax, 100)

        xdiff = np.array([np.abs(thatx - xn).min() for xn in xnew])
        addmask = xdiff > 1e-4
        thisx = np.concatenate([thatx, xnew[addmask]])
        thisx.sort()

        cum_cpost_arr = cpostinterp(thisx) * postinterp(thisx) / priorinterp(thisx)
        cum_cpost_arr *= 1. / trapz(cum_cpost_arr, x=thisx)
        postcdf = [trapz(postinterp(thisx)[:i], x=thisx[:i]) for i in range(1, len(thisx))]
        cdf = [trapz(cum_cpost_arr[:i], x=thisx[:i]) for i in range(1, len(thisx))]
        xlo, xmed, xhi = thisx[np.searchsorted(cdf, [0.05, 0.5, 0.95])]
        ci_bounds.append((xlo, xmed, xhi))
        fractional_error.append(xhi-xlo)
        thatx = thisx
    cum_cpost = cum_cpost_arr
    
if __name__=="__main__":
    pycbc.init_logging(verbose=True)
    postdir1 = sys.argv[1]
    postdir2 = sys.argv[2]
    postdir3 = sys.argv[3]
    det = sys.argv[4]
    nyears = int(sys.argv[5])

    postfiles1 = glob.glob('{}/out*[0-9].txt'.format(postdir1))
    postfiles2 = glob.glob('{}/out*[0-9].txt'.format(postdir2))
    postfiles3 = glob.glob('{}/out*[0-9].txt'.format(postdir3))

    nevents = len(postfiles1)

    rseed = 8333929
    logging.info("Shuffling posterior files using seed %s", rseed)
    np.random.seed(rseed)
    np.random.shuffle(postfiles1)
    logging.info("Reading posterior data")
    data = [extract_samples(p) for
        p in postfiles1[:nevents]]
    # filter for outliers
    medians = [np.median(d) for d in data]
    popmean = np.mean(medians)
    popstd = np.std(medians)
    outlier_mask = (np.abs(np.array(medians) - popmean)) > 2 * popstd
    outliers = np.array(postfiles1[:nevents])[outlier_mask]
    outlier_meds = np.array(medians)[outlier_mask]
    for o, m in zip(outliers, outlier_meds):
        logging.info("%s is an outlier! Median value is %s", o, m)

    eosdatfiles = glob.glob('{}/*.dat'.format(eosdir))
    rvals = []
    for f in eosdatfiles:
        eosid = int(os.path.basename(f).split('.')[0])
        rdat, mdat, ldat = np.loadtxt(f, unpack=True)
        rvals.append(np.interp(1.4, mdat, rdat))

    bw = 0.4
    xx = np.linspace(min(rvals), max(rvals), 1000)
    # set up multiprocessing pool
    nprocs = 64
    pool = BroadcastPool(nprocs)
    batchlen = int(nevents / nprocs)
    batched_data = [(data[i*batchlen:(i+1)*batchlen], xx)
                    for i in range(nprocs)]
    nleft = nevents - batchlen * nprocs
    for i in range(nleft):
        batched_data[i][0].append(data[-(i+1)])

    logging.info("Generating KDEs on %s processors", nprocs)
    r = pool.map_async(calc_kdes, batched_data)
    kdedata = [k for l in r.get() for k in l]

    np.random.seed()
    testids = np.random.randint(0, nevents-1, 10)
    testprior = radius14_prior(xx)
    plotdir = '/home/abandopa/projects/github-bns-eos-population/eos_population/test_plots'
    logging.info("Combining %s posteriors", len(kdedata))
    combined_posterior_1280, ci95_bounds_1280, frac_eos1280 = combine_posteriors(kdedata, xx)


    nevents = len(postfiles2)
    logging.info("Shuffling posterior files using seed %s", rseed)
    np.random.seed(rseed)
    np.random.shuffle(postfiles2)
    logging.info("Reading posterior data")
    data = [extract_samples(p) for
            p in postfiles2[:nevents]]
    # filter for outliers
    medians = [np.median(d) for d in data1]
    popmean = np.mean(medians)
    popstd = np.std(medians)
    outlier_mask = (np.abs(np.array(medians) - popmean)) > 2 * popstd
    outliers = np.array(postfiles2[:nevents])[outlier_mask]
    outlier_meds = np.array(medians)[outlier_mask]
    for o, m in zip(outliers, outlier_meds):
        logging.info("%s is an outlier! Median value is %s", o, m)

    eosdatfiles = glob.glob('{}/*.dat'.format(eosdir))
    rvals1 = []
    for f in eosdatfiles:
        eosid = int(os.path.basename(f).split('.')[0])
        rdat1, mdat1, ldat1 = np.loadtxt(f, unpack=True)
        rvals1.append(np.interp(1.4, mdat1, rdat1))

    bw = 0.4
    xx = np.linspace(min(rvals1), max(rvals1), 1000)
    # set up multiprocessing pool
    nprocs = 64
    pool = BroadcastPool(nprocs)
    batchlen = int(nevents / nprocs)
    batched_data = [(data1[i*batchlen:(i+1)*batchlen], xx)
                        for i in range(nprocs)]
    nleft = nevents - batchlen * nprocs
    for i in range(nleft):
        batched_data[i][0].append(data1[-(i+1)])

    logging.info("Generating KDEs on %s processors", nprocs)
    r = pool.map_async(calc_kdes, batched_data)
    kdedata = [k for l in r.get() for k in l]

    np.random.seed()
    testids = np.random.randint(0, nevents-1, 10)
    testprior = radius14_prior(xx)
    logging.info("Combining %s posteriors", len(kdedata))
    combined_posterior_895, ci95_bounds_895, frac_eos895 = combine_posteriors(kdedata, xx)

    nevents = len(postfiles3)
    logging.info("Shuffling posterior files using seed %s", rseed)
    np.random.seed(rseed)
    np.random.shuffle(postfiles3)
    logging.info("Reading posterior data")
    data2 = [extract_samples(p) for
            p in postfiles3[:nevents]]
    # filter for outliers
    medians = [np.median(d) for d in data2]
    popmean = np.mean(medians)
    popstd = np.std(medians)
    outlier_mask = (np.abs(np.array(medians) - popmean)) > 2 * popstd
    outliers = np.array(postfiles3[:nevents])[outlier_mask]
    outlier_meds = np.array(medians)[outlier_mask]
    for o, m in zip(outliers, outlier_meds):
        logging.info("%s is an outlier! Median value is %s", o, m)
    eosdatfiles = glob.glob('{}/*.dat'.format(eosdir))
    rvals2 = []
    for f in eosdatfiles:
        eosid2 = int(os.path.basename(f).split('.')[0])
        rdat2, mdat2, ldat2 = np.loadtxt(f, unpack=True)
        rvals2.append(np.interp(1.4, mdat2, rdat2))

    bw = 0.4
    xx = np.linspace(min(rvals2), max(rvals2), 1000)
    # set up multiprocessing pool
    nprocs = 16
    pool = BroadcastPool(nprocs)
    batchlen = int(nevents / nprocs)
    batched_data = [(data2[i*batchlen:(i+1)*batchlen], xx)
                    for i in range(nprocs)]
    nleft = nevents - batchlen * nprocs
    for i in range(nleft):
        batched_data[i][0].append(data2[-(i+1)])

    logging.info("Generating KDEs on %s processors", nprocs)
    r = pool.map_async(calc_kdes, batched_data)
    kdedata = [k for l in r.get() for k in l]
    np.random.seed()
    testids = np.random.randint(0, nevents-1, 10)
    testprior = radius14_prior(xx)
    logging.info("Combining %s posteriors", len(kdedata))
    combined_posterior_477, ci95_bounds_477, frac_eos477 = combine_posteriors(kdedata, xx)

    logging.info("Plotting")
    fig, ax = plt.subplots()
    ax2=ax.twinx()
    for i in range(0,len(postfiles2)):
        frac_eos477[i]=frac_eos477[i]*(1.0/10.643797)
    for i in range(0,len(postfiles1)):
        frac_eos895[i]=frac_eos895[i]*(1.0/11.86984)
    for i in range(0,len(postfiles)):
        frac_eos1280[i]=frac_eos1280[i]*(1.0/12.9670722)
    print(frac_eos477[0],frac_eos477[len(frac_eos477)-1])
    print(frac_eos895[0],frac_eos895[len(frac_eos895)-1])
    print(frac_eos1280[0],frac_eos1280[len(frac_eos1280)-1])

    plot_errorbar(ci95_bounds_1280,frac_eos1280,ax,ax2,eosid=1280, tag="      (nsat)",rad_value=12.95,detector=det,years=nyears,color1='limegreen',color2='green')
    plot_errorbar(ci95_bounds_895,frac_eos895, ax,ax2,eosid=895, tag="(nsat)",rad_value=11.86984,detector=det,years=nyears,color1='sandybrown',color2='orangered')
    plot_errorbar(ci95_bounds_477,frac_eos477, ax,ax2,eosid=477, tag="(nsat)",rad_value=10.643797,detector=det,years=nyears,color1='lightblue',color2='blue')
    plt.show()
