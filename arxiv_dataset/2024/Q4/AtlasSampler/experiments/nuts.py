import numpy as np
import os, sys
import matplotlib.pyplot as plt

import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)

from atlassampler.wrappers import cmdstanpy_wrapper
import models


#######
import argparse
parser = argparse.ArgumentParser(description='Arguments for NUTS sampler')
parser.add_argument('--exp', type=str, help='which experiment')
parser.add_argument('-n', type=int, default=0, help='dimensionality or model number')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--n_chains', type=int, default=16, help='number of chains')
parser.add_argument('--n_samples', type=int, default=100, help='number of samples')
parser.add_argument('--n_burnin', type=int, default=100, help='number of burnin/warmup iterations')
parser.add_argument('--n_stepsize_adapt', type=int, default=100, help='number of iterations for step size adaptation')
parser.add_argument('--n_metric_adapt', type=int, default=0, help='number of iterations for metric adaptation')
parser.add_argument('--target_accept', type=float, default=0.80, help='target acceptance rate')
parser.add_argument('--step_size', type=float, default=0.1, help='initial step size')
parser.add_argument('--metric', type=str, default='unit_e', help='metric for NUTS')


# NUTS
def run_nuts(stanfile, datafile, args, seed=999, savefolder=None, verbose=True, return_all=False):

    if verbose: print("Run NUTS")
    
    np.random.seed(seed)
    if savefolder is not None:
        os.makedirs(savefolder, exist_ok=True)

    cmd_model = csp.CmdStanModel(stan_file = stanfile)
    sampler = cmd_model.sample(data = datafile, 
                                chains = args.n_chains,
                                iter_sampling = args.n_samples,
                                iter_warmup = max(args.n_burnin, 1000),
                                seed = args.seed,
                                metric = args.metric,
                                step_size = args.step_size,
                                adapt_delta = args.target_accept,
                                ## cmdstanpy bug? does not adapt if specified
                                #adapt_engaged = True,
                                #adapt_metric_window = args.n_metric_adapt, 
                                #adapt_init_phase = args.n_stepsize_adapt,
                                #adapt_step_size = args.n_stepsize_adapt,
                                show_console = False,
                                show_progress = verbose,
                                save_warmup = False)

    draws_pd = sampler.draws_pd()
    step_size = sampler.step_size
    metric = sampler.metric
    samples, n_leapfrogs = cmdstanpy_wrapper(draws_pd)
    difference = np.diff(samples[..., 0])
    print("accept/reject for NUTS: ", difference.size - (difference == 0 ).sum(),  (difference == 0 ).sum())
    
    if savefolder is not None:
        np.save(f'{savefolder}/samples', samples)
        np.save(f'{savefolder}/leapfrogs', n_leapfrogs)
        np.save(f'{savefolder}/stepsize', step_size)
        if metric is not None: np.save(f'{savefolder}/metric', metric)

    if return_all: 
         return samples, sampler, step_size, n_leapfrogs
    else:
         return samples


def load_results(folder, n_chains, n_samples):
    samples_nuts = np.load(f"{folder}/samples.npy")
    step_size = np.load(f"{folder}/stepsize.npy")
    n_leapfrogs_nuts = np.load(f"{folder}/leapfrogs.npy")
    assert samples_nuts.shape[0] >= n_chains
    assert samples_nuts.shape[1] >= n_samples
    samples_nuts = samples_nuts[:n_chains, :n_samples]
    n_leapfrogs_nuts = n_leapfrogs_nuts[:n_chains, :n_samples]
    step_size = step_size[:n_chains]
    return samples_nuts, step_size, n_leapfrogs_nuts
        

if __name__ == "__main__":

    args = parser.parse_args()
    experiment = args.exp
    n = args.n
    parentfolder = '/mnt/ceph/users/cmodi/atlassampler/'
    print("Model name : ", experiment)
    model, D, lp, lp_g, ref_samples, files = models.stan_model(experiment, n)
    if n == 0 : savefolder = f'{parentfolder}/{experiment}/nuts/target{args.target_accept:0.2f}/'
    else: savefolder = f'{parentfolder}/{experiment}-{D}/nuts/target{args.target_accept:0.2f}/'

    stanfile, datafile = files
    samples = run_nuts(stanfile, datafile, args, seed=args.seed, savefolder=savefolder, verbose=True)

    if ref_samples is not None:
        plt.figure()
        plt.hist(ref_samples[..., 0].flatten(), density=True, alpha=1,
                    bins='auto', lw=2, histtype='step', color='k', label='Reference')
        plt.hist(samples[..., 0].flatten(), density=True, alpha=0.5,
                    bins='auto', label='NUTS');

        print("\nPlotting")
        plt.legend()
        plt.title(savefolder.split(f'nuts/')[1][:-1])
        plt.savefig('tmp.png')
        plt.savefig(f"{savefolder}/hist")
        plt.close()
