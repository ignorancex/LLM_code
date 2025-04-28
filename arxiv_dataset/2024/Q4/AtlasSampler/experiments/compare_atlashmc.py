"""
Script to run nuts and atlas on the same problem with the stepsize tuned by nuts, and compare results. 
The script also generates reference samples by running nuts with target acceptance of 0.95 if needed.
"""
import numpy as np
import os, sys
import matplotlib.pyplot as plt
    
from atlassampler import Atlas_HMC
from atlassampler.wrappers import cmdstanpy_wrapper
import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)

import models
import plotting
import nuts
import default_args

# Setup MPI environment
from mpi4py import MPI
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()
print('My rank is ',wrank)

# Set some paths
SAVEFOLDER = '/mnt/ceph/users/cmodi/atlassampler/'
REFERENCE_FOLDER = "/mnt/ceph/users/cmodi/PosteriorDB/"
BRIDGESTAN = "/mnt/home/cmodi/Research/Projects/bridgestan/"
MODELDIR = '../'

#######
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser = default_args.add_default_args(parser)
parser.add_argument('--stepinit_factor', type=float, default=1.0, help='initial stepsize factor')


args = parser.parse_args()
args.n_chains = wsize
print("Model name : ", args.exp)

# Load model and reference samples in rank 0. Generate if necessary.
reference_path =  f'{REFERENCE_FOLDER}/{args.exp}/'
run_nuts = bool(not wrank)
model, D, lp, lp_g, ref_samples, files = models.stan_model(args.exp, args.n, 
                                                           bridgestan_path=BRIDGESTAN, 
                                                           model_directory=MODELDIR, 
                                                           reference_samples_path=reference_path,
                                                           run_nuts=run_nuts)
if args.n == 0 : savepath = f'{SAVEFOLDER}/{args.exp}'
else: savepath = f'{SAVEFOLDER}/{args.exp}-{D}/'
comm.Barrier()
###################################
# NUTS
# Load samples if present. Run NUTS otherwise
np.random.seed(args.seed)
if wrank == 0:
    savefolder_nuts = f"{savepath}/nuts/target{args.target_accept:0.2f}/"
    print(f"\nTrying to load NUTS results on rank 0 from folder :  {savefolder_nuts}")
    try:
        samples_nuts, step_size, n_leapfrogs_nuts = nuts.load_results(savefolder_nuts, wsize, args.n_samples)
    except Exception as e:
        print("Exception in loading NUTS results : ", e)
        args.nuts = 1
    
    if args.nuts == 1:
        print("\nNow run NUTS on rank 0")
        samples_nuts, sampler, step_size, n_leapfrogs_nuts = nuts.run_nuts(stanfile = files[0], 
                                                                    datafile = files[1], 
                                                                    args = args, 
                                                                    savefolder=savefolder_nuts, 
                                                                    return_all=True)        
else:
    step_size = 0.
    samples_nuts = np.zeros([args.n_chains, D])
comm.Barrier()

# Scatter step size and initial point to different ranks
comm.Barrier()
if wrank == 0 : print()
step_size = comm.scatter(step_size, root=0)
q0 = comm.scatter(samples_nuts[:, 0], root=0)
q0 = model.param_unconstrain(q0)
print(f"Step size in rank {wrank}: ", step_size)
comm.Barrier()


#####################
# Atlas
folder = 'atlas-hmc'
savefolder = f"{savepath}/{folder}/offset{args.offset:0.2f}/"
if args.stepinit_factor != 1.:
    savefolder = f"{savefolder}"[:-1] + f"-sinitfac{args.stepinit_factor:0.1f}/"
if args.stepsize_distribution == 'lognormal':
    savefolder = f"{savefolder}"[:-1] + f"-{args.stepsize_distribution}/"
    if args.stepsize_sigma != 2.:
        savefolder = f"{savefolder}"[:-1] + f"-stepsig{args.stepsize_sigma}/"
if args.suffix != "":
    savefolder = f"{savefolder}"[:-1] + f"-{args.suffix}/"
    
os.makedirs(f'{savefolder}', exist_ok=True)
print(f"Saving Atlas runs in folder : {savefolder}")

# Start run
np.random.seed(args.seed)
kernel = Atlas_HMC(D, lp, lp_g, 
                   mass_matrix = np.eye(D), 
                   offset = args.offset,
                   min_nleapfrog = args.min_leapfrog,
                   max_nleapfrog = args.max_leapfrog,
                   stepsize_distribution = args.stepsize_distribution,
                   stepsize_sigma = args.stepsize_sigma)
sampler = kernel.sample(q0,
                        seed = wrank,
                        n_leapfrog = args.n_leapfrog, 
                        step_size = step_size, 
                        n_samples = args.n_samples, 
                        n_burnin = 0,
                        n_stepsize_adapt = 0,
                        n_leapfrog_adapt = 0.,
                        target_accept = args.target_accept)


print(f"Acceptance for Atlas Sampler in chain {wrank} : ", np.unique(sampler.accepts, return_counts=True))

# Save samples and gather on rank 0 for diagnostics
sampler.save(path=f"{savefolder}", suffix=f"-{wrank}")
samples_constrained = []
for s in sampler.samples:
    samples_constrained.append(model.param_constrain(s))
np.save(f"{savefolder}/samples_constrained-{wrank}", samples_constrained)
comm.Barrier()

samples = comm.gather(samples_constrained, root=0)
accepts = comm.gather(sampler.accepts, root=0)
gradcounts = comm.gather(sampler.gradcounts, root=0)
stepsizes = comm.gather(kernel.step_size, root=0)

#####################
# Diagnostics
if wrank == 0:
    
    samples = np.stack(samples, axis=0)
    print("\nTotal accpetances for Atlas Sampler : ", np.unique(np.stack(accepts), return_counts=True))
    
    print("\nPlotting")
    suptitle = f"{args.exp}:D={D}; " + savefolder.split(f'{folder}/')[1][:-1]
    samples_list = [samples_nuts, samples]
    labels = ["NUTS", "Atlas"]
        
        # plot histograms
    try:
        plotting.plot_histograms(samples_list, 
                                 nplot = min(5, D), 
                                 labels = labels, 
                                 savefolder = savefolder, 
                                 suptitle = suptitle,
                                 reference_samples = ref_samples)
        plt.savefig('tmp.png')
        plt.close()
    except Exception as e:
        print(e)
        
    # plot cost
    try:
        plt.figure()
        normalize = n_leapfrogs_nuts.sum(axis=1).mean()
        toplot = [n_leapfrogs_nuts.sum(axis=1)/normalize, np.array(gradcounts).sum(axis=1)/normalize]
        plt.boxplot(toplot, patch_artist=True,
                    boxprops=dict(facecolor='C0', color='C0', alpha=0.5), labels=labels)
        plt.grid(which='both', lw=0.3)
        plt.ylabel('# Leapfrogs', fontsize=12)
        plt.axhline(1., color='k', ls=":")
        plt.suptitle(suptitle)
        plt.savefig(f"{savefolder}/gradcounts")
        plt.close()
    except Exception as e:
        print(e)

    # plot rmse
    try:
        if ref_samples is not None:
            counts_list = [n_leapfrogs_nuts, np.array(gradcounts)]    
            plotting.boxplot_rmse(ref_samples, samples_list, counts_list, labels, 
                    savefolder=savefolder, suptitle=suptitle)
            plt.close()

        if args.exp == 'funnel':
            samples_list0 = [i[..., 0:1] for i in samples_list]
            plotting.boxplot_rmse(ref_samples[..., 0:1], samples_list0, counts_list, labels, 
                    savefolder=savefolder, suptitle=suptitle, savename='rmse_logscale')
            plt.close()
            #
            samples_list1 = [i[..., 1:] for i in samples_list]
            plotting.boxplot_rmse(ref_samples[..., 1:], samples_list1, counts_list, labels, 
                    savefolder=savefolder, suptitle=suptitle, savename='rmse_latents')
            plt.close()
    except Exception as e:
        print(e)

comm.Barrier()    

sys.exit()
