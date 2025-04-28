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

cluster_process = os.environ['cluster_process_var']

def extract_samples(postfile):
    samples=np.loadtxt(postfile,unpack=True)
    return samples


def calc_kdes(data):
    postdata, x = data
    kdes = [gaussian_kde(d, bw_method=bw)(x) for d in postdata]
    return kdes

def plot_errorbar(data,frac_error,ax,ax2,eosid,tag,rad_value,color1,color2):
    yerr = [np.abs(data[:, i] - data[:, 1]) for i in [0, 2]]
    #ax.errorbar(np.arange(len(data)) + 1, data[:, 1], yerr=yerr, capsize=2,ls='none',color=color1)
    ax.fill_between(np.arange(len(data)) + 1, data[:, 2], data[:, 0],
                    color=color1, alpha=0.5)
    #print(len(data))
    x=np.linspace(1,len(data),len(data))
    # plot EOS value
    label='EOS {} '.format(str(eosid))+tag
    ax.axhline(rad_value, ls='dashed',label=label,color=color2)
    ax.set(xlabel='number of events', ylabel=r'$R_{1.4} \, (\mathrm{km})$',
           xlim=(0, len(data) + 1))
    ax.set_ylim(10.0,13.5)
    #ax2 = ax.twinx()
    ax2.semilogy(x,frac_error,linestyle=':',color=color2)
    ax2.set_ylabel(r'Fractional uncertainty $\Delta \rm R/\rm R$')
    ax2.set_ylim(0.0006,0.06)
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

    return cum_cpost, np.array(ci_bounds),fractional_error

if __name__ == "__main__":
    pycbc.init_logging(verbose=True)
    output_dir = sys.argv[1]
    filtered_injection_dir1=sys.argv[2]
    full_injection_dir1=sys.argv[3]
    number_of_events=int(sys.argv[4])
    number_of_years=float(sys.argv[5])
    inj_radius = float(sys.argv[6])
   
    rand_array=np.random.uniform(100000,1000000,500)
    
    #Calculate the KDEs for the analyzed PE runs, and store them in a 1d array
    #injids is another 1d array that holds values for the corresponding injection ids (labels to identify the output files)
    
    injids=[]
    distances=[]
    output_files=[]
    filtered_injection_files = glob.glob('{}/*.hdf'.format(filtered_injection_dir1))
    for i,f in enumerate(filtered_injection_files1):
        fp = h5py.File(f, 'r') 
        injid=int(f[f.find('injection_')+10:f.find('.hdf')])
        injids.append(injid)
        distances.append(float(fp['distance'][0]))
    

    for ii in range(0,len(injids)):
        injid=injids[ii]
        output_file=output_dir1+'/out{}.txt'.format(injid)
        output_files.append(output_file)
    
    nevents=len(output_files1)
    print(nevents)
    data = [extract_samples(p) for p in output_files]


    eosdatfiles = glob.glob('{}/*.dat'.format(eosdir))
    rvals = []
    for f in eosdatfiles:
        eosid = int(os.path.basename(f).split('.')[0])
        rdat, mdat, ldat = np.loadtxt(f, unpack=True)
        rvals.append(np.interp(1.4, mdat, rdat))
            
    bw = 0.2
    xx = np.linspace(min(rvals), max(rvals), 1000)
    logging.info("Calculating kdes for %s events", nevents)    
    kdedata = calc_kdes((data,xx))
    
    
    # Reweighting the injection files to draw a unique realization of the universe
    dmin=20.0
    dmax=2000.0
    x=np.linspace(dmin,dmax,1000)
    def src_dist(x):
        return (1.0/(dmax-dmin))*np.ones(len(x))

    tdist = UniformPowerLaw(dim=3, distance=(dmin,dmax+0.1))
    pop_probs = np.array([tdist.pdf(distance=p) for p in x])

    target_dist= interp1d(x,pop_probs,kind="linear")
    def ratio_target_to_src(x):
        return target_dist(x)/src_dist(x)
    
    
    for counter in range (0,500):
        raw_injection_files=[]
        distances_for_injections=[]
        injids_for_injections=[]
        logging.info("Reading injections")
        injfiles = glob.glob('{}/*.hdf'.format(full_injection_dir1))
        file1=open(output_dir+"/shuffle_events"+str(i+1)+".txt","a")
        counter+=1
        for i,f in enumerate(injfiles):
            fp = h5py.File(f, 'r') 
            d=float(fp['distance'][0])
            raw_injection_files.append(f)
            distances_for_injections.append(d)
            injid=int(f[f.find('injection_')+10:f.find('.hdf')])
            injids_for_injections.append(injid)
        
        weights=ratio_target_to_src(distances_for_injections)
        weights=weights/np.sum(weights)
        
    
        # Drawing a realization of the universe with weights calculated above
        events=np.random.choice(injids_for_injections,size=number_of_events,p=weights)
        detected_events=[]
        for event_id in events:
            if(event_id in injids):
                detected_events.append(event_id)
        number_of_detected_events=len(detected_events)
        #Shuffle the order of events
        rseed=int(rand_array[counter])
        logging.info("Shuffling posterior files using seed %s", rseed)
        np.random.seed(rseed)
        np.random.shuffle(detected_events)
        #Create an array of KDEs for the observed events, in the order that they are observed in the 
        #specific realization of the universe
        kdes_for_events=[]
        for ii in range(0,len(detected_events)):
            loc=injids.index(detected_events[ii])
            kdes_for_events.append(kdedata[loc])
            
        logging.info("Combining %s posteriors", len(kdes_for_events))
        combined_posterior, ci95_bounds, width_ci95_interval = combine_posteriors(kdes_for_events, xx)
        for c in range(0,len(detected_events)):
            frac_error=width_ci95_interval[c]/inj_radius
            file1.write(str(c+1)+" "+str((number_of_years/number_of_detected_events)*(c+1))+" "+str(frac_error)+"\n")
        logging.info("Number of events %s",len(detected_events))
        logging.info("Frac error %s",frac_error)
        file1.close()