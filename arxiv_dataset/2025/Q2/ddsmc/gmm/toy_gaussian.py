"""
This code was modified from https://github.com/gabrielvc/mcg_diff/blob/1a3cdd22c971ca13ce06a48eaaa00d2a6602b5c1/scripts/viz_gaussian.py

Many modifications were. For example, to including more algorithms than MCGDiff, enable using command line arguments to specify variables, sampling in 
batches to enable a larger number of samples, running on CUDA, computing Wasserstein distances, and modifications to how results wree visualized. These modifications were done October 2024 - May 2025

As the original work is under an GPL 3.0 license, so is this source file. You should have received a copy of the license, otherwise see (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

import torch
import matplotlib.pyplot as plt
from functools import partial
import argparse
import ot
import time
from tqdm import tqdm

from toy_utils.results_utils import split_number, plot_samples
from toy_utils.score_model import ScoreModelDDSMC
from toy_utils.operator import GaussianOperator, Toy_Hfunc

from ddsmc.ddsmc_sampler import DDSMCDDPM

from mcg_diff.particle_filter import mcg_diff
from mcg_diff.sgm import ScoreModel
from mcg_diff.utils import NetReparametrized, get_optimal_timesteps_from_singular_values

from tds.tds import TDS

from ddrm.ddrm import ddrm

from dcps.dcps_diffusion_utils import DCPSEpsilonNet
from dcps.dcps import dcps

from daps.daps import DAPSSampler


# Our mixture of Gaussians:
# q_data(x_0) = \sum_i w_i N(x_0|\mu_i, I)
# q_{t|0}(x_t|x_0) = N(x_t|\sqrt{\bar\alpha_t}x_0, (1- \bar\alpha_t) I)
# q_{t}(x_t) \propto \sum_i w_i N(x_t|, \sqrt{\bar\alpha_t}\mu_i, I)
def ou_mixt(alpha_t, means, dim, weights):
    cat = torch.distributions.Categorical(weights, validate_args=False)

    ou_norm = torch.distributions.MultivariateNormal(
        torch.vstack(tuple((alpha_t**.5) * m for m in means)),
        torch.eye(dim, device=means[0].device).repeat(len(means), 1, 1), validate_args=False)
    return torch.distributions.MixtureSameFamily(cat, ou_norm, validate_args=False)

def get_posterior(obs, prior, A, Sigma_y):
    device = A.device
    modified_means = []
    modified_covars = []
    weights = []
    precision = torch.linalg.inv(Sigma_y)
    for loc, cov, weight in zip(prior.component_distribution.loc,
                                prior.component_distribution.covariance_matrix,
                                prior.mixture_distribution.probs):
        new_dist = gaussian_posterior(obs,
                                      A,
                                      torch.zeros_like(obs),
                                      precision,
                                      loc,
                                      cov)
        modified_means.append(new_dist.loc)
        modified_covars.append(new_dist.covariance_matrix)
        prior_x = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
        residue = obs - A @ new_dist.loc
        log_constant = -(residue[None, :] @ precision @ residue[:, None]) / 2 + \
                       prior_x.log_prob(new_dist.loc) - \
                       new_dist.log_prob(new_dist.loc)
        weights.append(torch.log(weight).item() + log_constant)
    weights = torch.tensor(weights).to(device)
    weights = weights - torch.logsumexp(weights, dim=0)
    cat = torch.distributions.Categorical(logits=weights)
    ou_norm = torch.distributions.MultivariateNormal(loc=torch.stack(modified_means, dim=0).to(device),
                                                     covariance_matrix=torch.stack(modified_covars, dim=0).to(device))
    return torch.distributions.MixtureSameFamily(cat, ou_norm)


def gaussian_posterior(y,
                       likelihood_A,
                       likelihood_bias,
                       likelihood_precision,
                       prior_loc,
                       prior_covar):
    prior_precision_matrix = torch.linalg.inv(prior_covar)
    posterior_precision_matrix = prior_precision_matrix + likelihood_A.T @ likelihood_precision @ likelihood_A
    posterior_covariance_matrix = torch.linalg.inv(posterior_precision_matrix)
    posterior_mean = posterior_covariance_matrix @ (likelihood_A.T @ likelihood_precision @ (y - likelihood_bias) + prior_precision_matrix @ prior_loc)
    try:
        posterior_covariance_matrix = (posterior_covariance_matrix + posterior_covariance_matrix.T) / 2
        return torch.distributions.MultivariateNormal(loc=posterior_mean, covariance_matrix=posterior_covariance_matrix, validate_args=False)
    except ValueError:
        u, s, v = torch.linalg.svd(posterior_covariance_matrix, full_matrices=False)
        s = s.clip(1e-12, 1e6).real
        posterior_covariance_matrix = u.real @ torch.diag_embed(s) @ v.real
        posterior_covariance_matrix = (posterior_covariance_matrix + posterior_covariance_matrix.T) / 2
        return torch.distributions.MultivariateNormal(loc=posterior_mean, covariance_matrix=posterior_covariance_matrix, validate_args=False)


def build_extended_svd(A: torch.tensor):
    U, d, V = torch.linalg.svd(A, full_matrices=True)
    coordinate_mask = torch.ones_like(V[0])
    coordinate_mask[len(d):] = 0
    return U, d, coordinate_mask, V


def generate_measurement_equations(dim, dim_y, mixt, device):
    A = torch.randn((dim_y, dim), device=device)

    u, diag, coordinate_mask, v = build_extended_svd(A)
    diag = torch.sort(torch.rand_like(diag), descending=True).values

    A = u @ (torch.diag(diag) @ v[coordinate_mask == 1, :])
    init_sample = mixt.sample()
    std = (torch.rand((1,)))[0]* max(diag)
    var_observations = std**2

    init_obs = A @ init_sample
    init_obs += torch.randn_like(init_obs) * std
    return A.to(device), var_observations.to(device), init_obs.to(device)

def get_args():
    parser = argparse.ArgumentParser(
        description="Toy Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device", type=str.lower, default="cuda", help="Seed for random number generator"
    )
    
    parser.add_argument(
        "--seed", type=int, default=100, help="Seed for random number generator"
    )

    parser.add_argument(
        "--num_particles", type=int, default=256, help="Number of particles"
    )

    parser.add_argument(
        "--num_samples", type=int, default=300, help="Number of samples"
    )

    parser.add_argument(
        "--dim_x", type=int, default=8, help="Dimension of data"
    )

    parser.add_argument(
        "--dim_y", type=int, default=1, help="Dimension of observation"
    )

    parser.add_argument(
        "--num_steps", type=int, default=20, help="Number of diffusion steps"
    )

    parser.add_argument(
        "--eta", type=float, default=1., help="Value of eta in DDIM formulation (1=DDPM, 0=DDIM, or interpolation)"
    )

    parser.add_argument(
        "--eta_ddsmc", type=float, default=0., help="Value of eta in DDSMC (0=Decoupled, 1=regular DDPM)"
    )

    parser.add_argument(
        "--max_num_ode_steps", type=int, help="Max number of internal ODE steps in DDSMC"
    )

    parser.add_argument(
        "--kappa", type=float, default=1e-2, help="Value of kappa"
    )

    parser.add_argument(
        "--dcps_M", type=int, default=50, help="Value of M in DCPS"
    )

    parser.add_argument(
        "--algo", type=str, help="name of algo to run"
    )

    parser.add_argument(
        "--use_ode", action="store_true", help="Use ODE solution in DDSMC"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Run in debug mode"
    )

    args = parser.parse_args()
    return args

def main(args):
    ######
    ## DATA GENERATION
    #######
    random_state = args.seed
    n_samples = args.num_samples
    device = args.device
    torch.manual_seed(random_state)

    dim_y = args.dim_y
    dim_x = args.dim_x
    # setup of the inverse problem
    means = []
    for i in range(-2, 3):
        means += [torch.tensor([-8. * i, -8. * j] * (dim_x // 2), device=device) for j in range(-2, 3)]
    weights = torch.randn(len(means), device=device) ** 2
    weights = weights / weights.sum()
    ou_mixt_fun = partial(ou_mixt,
                        means=means,
                        dim=dim_x,
                        weights=weights)

    mixt = ou_mixt_fun(1)

    A, var_observations, init_obs = generate_measurement_equations(dim_x, dim_y, mixt, device)
    posterior = get_posterior(init_obs, mixt, A, torch.eye(dim_y, device=device)*var_observations)

    # limit the number of samples per batch, adjust as necessary
    if dim_x > 750:
        max_samples_per_batch = 50
    else:
        # Although it fits in memory, running 10k 8-dim samples in one batch is extremely slow, but running 10 batches of 1k samples is very fast
        max_samples_per_batch = 1000
    # Increase number of samples per batch for non-SMC methods. Can also be adjusted as necessary
    if args.algo.lower() in ["dcps", "ddrm", "daps"]:
        if dim_x < 750:
            max_samples_per_batch *= 256
        else:
            max_samples_per_batch = 250
    target_samples = torch.cat([posterior.sample((n,)) for n in split_number(n_samples, max_samples_per_batch)]).cpu()
    betas = torch.linspace(.02, 1e-4, steps=999, device=device)
    alphas_cumprod = torch.cumprod(torch.tensor([1, ] + [1 - beta for beta in betas], device=device), dim=0)

    observation = init_obs.to(device)
    forward_operator = A
    observation_noise = var_observations
    reference_samples = target_samples

    ######
    ## MODEL SETUP
    #######
    num_particles = args.num_particles
    score_network = lambda x, alpha_bar_t: torch.func.grad(lambda y: ou_mixt_fun(alpha_bar_t).log_prob(y).sum())(x)
    n_steps = args.num_steps

    u, diag, coordinate_mask, vt = build_extended_svd(forward_operator)

    adapted_timesteps = get_optimal_timesteps_from_singular_values(alphas_cumprod=alphas_cumprod,
                                                                singular_value=diag,
                                                                n_timesteps=n_steps,
                                                                var=observation_noise,
                                                                mode='else').to(device)

    eta = args.eta

    #######
    # SETUP AND RUNNING DDSMC
    #######
    if args.algo.lower() == "ddsmc":
        operator_prime = GaussianOperator(A, torch.sqrt(observation_noise), prime=True)
        model_prime = ScoreModelDDSMC(score_network, operator_prime.V.T).to(device)
        sampler = DDSMCDDPM(alphas_cumprod, num_particles, device, args.eta_ddsmc, ddim_taus=adapted_timesteps, use_ode=args.use_ode, max_num_ode_steps=args.max_num_ode_steps)
        print("Sampling DDSMC")
        torch.manual_seed(random_state)
        x_start_batch = torch.stack([torch.randn((num_particles*n, reference_samples.shape[1])) for n in split_number(n_samples, max_samples_per_batch)])
        x_start_batch = x_start_batch.to(device)
        observation_prime = (operator_prime.U.T @ observation)
        st = time.time()
        samples_ddsmc_list = []
        for x_start in tqdm(x_start_batch):
            samples_ddsmc, state_ddsmc = sampler.sample(model_prime, (operator_prime.V.T @ x_start.T).T, operator_prime, observation_prime, verbose=args.verbose)
            samples_ddsmc = (operator_prime.V @ samples_ddsmc.T).T
            samples_ddsmc_list.append(samples_ddsmc)
        samples = torch.cat(samples_ddsmc_list).cpu()
        et = time.time()
        runtime = et - st

    ######
    # SETUP AND RUNNING TDS
    ######
    elif args.algo.lower() == "tds":
        sampler = TDS(alphas_cumprod, num_particles, "multinomial", device)
        model = ScoreModelDDSMC(score_network).to(device)
        operator = GaussianOperator(A, torch.sqrt(observation_noise))
        print("Sampling TDS")
        torch.manual_seed(random_state)
        x_start_batch = torch.stack([torch.randn((num_particles*n, reference_samples.shape[1])) for n in split_number(n_samples, max_samples_per_batch)])
        x_start_batch = x_start_batch.to(device)
        st = time.time()
        samples_tds_list = []
        for x_start in tqdm(x_start_batch):
            samples_tds, state_tds = sampler.sample(model, x_start, operator, observation, adapted_timesteps, eta)
            samples_tds_list.append(samples_tds)
        samples = torch.cat(samples_tds_list).cpu()
        et = time.time()
        runtime = et - st

    ######
    # SETUP AND RUNNING MCGDiff
    ######
    elif args.algo.lower() == "mcgdiff":
        score_model = ScoreModel(NetReparametrized(
            base_score_module=lambda x, t: - score_network(x, alphas_cumprod[t]) * ((1 - alphas_cumprod[t]) ** .5),
            orthogonal_transformation=vt),
            alphas_cumprod=alphas_cumprod,
            device=device)

        def mcg_diff_fun(initial_samples):
            samples, log_weights = mcg_diff(
                initial_particles=initial_samples,
                observation=(u.T @ observation),
                score_model=score_model,
                likelihood_diagonal=diag,
                coordinates_mask=coordinate_mask.bool(),
                var_observation=observation_noise,
                timesteps=adapted_timesteps,
                eta=eta,
                gaussian_var=args.kappa,
            )
            return vt.T @ \
                samples[torch.distributions.Categorical(logits=log_weights, validate_args=False).sample(sample_shape=(1,))][0]
        sampler = mcg_diff_fun
        print("Sampling MCGDiff")
        samples_mcgdiff_list = []
        torch.manual_seed(random_state)
        x_start_batch = torch.stack([torch.randn((num_particles*n, reference_samples.shape[1])) for n in split_number(n_samples, max_samples_per_batch)])
        x_start_batch = x_start_batch.to(device)
        st = time.time()
        for x_start in tqdm(x_start_batch):
            initial_samples = x_start.reshape((-1, num_particles, dim_x))
            samples_mcgdiff = torch.func.vmap(sampler, in_dims=(0,), randomness='different')(initial_samples)
            samples_mcgdiff_list.append(samples_mcgdiff)
        et = time.time()
        samples = torch.cat(samples_mcgdiff_list).cpu()
        runtime = et - st
    elif args.algo.lower() == "ddrm":
        epsilon_model = lambda x, t: - score_network(x, alphas_cumprod[t]) * ((1 - alphas_cumprod[t]) ** .5)
        inverse_problem = (observation.reshape((1, -1)), Toy_Hfunc(forward_operator), torch.sqrt(observation_noise))
        torch.manual_seed(random_state)
        x_start_batch = torch.stack([torch.randn((n, reference_samples.shape[1])) for n in split_number(n_samples, max_samples_per_batch)])
        x_start_batch = x_start_batch.to(device)
        samples_ddrm_list = []
        st = time.time()
        for x_start in tqdm(x_start_batch):
            ddrm_samples = ddrm(
                x_start,
                epsilon_model,
                inverse_problem,
                betas,
                args.device,
                n_steps - 1,
            )
            samples_ddrm_list.append(ddrm_samples)
        samples = torch.cat(samples_ddrm_list).cpu()
        et = time.time()
        runtime = et - st
    elif args.algo.lower() == "dcps":
        timesteps = torch.linspace(0, 999, n_steps).long()
        L = 3
        K = 2
        M = args.dcps_M
        gamma = 1e-2
        n_steps = n_steps // (L - 1)
        obs_timesteps = torch.linspace(0, 999, L)
        H_funcs = Toy_Hfunc(forward_operator)
        epsilon_net = DCPSEpsilonNet(net=lambda x, t: - score_network(x, alphas_cumprod[t]) * ((1 - alphas_cumprod[t]) ** .5),
        alphas_cumprod=alphas_cumprod, 
        timesteps=timesteps)
        torch.manual_seed(random_state)
        x_start_batch = torch.stack([torch.randn((n, reference_samples.shape[1])) for n in split_number(n_samples, max_samples_per_batch)])
        x_start_batch = x_start_batch.to(device)
        samples_dcps_list = []
        st = time.time()
        for x_start in tqdm(x_start_batch):
            samples_dcps_list.append(dcps(
                initial_noise=x_start,
                epsilon_net=epsilon_net,
                ip_type="linear",
                obs=observation.reshape((1, -1)),
                A=H_funcs.H,
                obs_std=torch.sqrt(observation_noise),
                n_steps=n_steps,
                obs_timesteps=obs_timesteps,
                optimizer="SGD",
                gradient_steps=K,
                learning_rate=1.0,
                langevin_steps=M,
                gamma=gamma,
                poisson_rate=0.1,  # not used, so default
                noise_type="gaussian",
            ))
        samples = torch.cat(samples_dcps_list).cpu()
        et = time.time()
        runtime = et - st
    elif args.algo.lower() == "daps":
        operator = GaussianOperator(A, torch.sqrt(observation_noise), prime=False)
        timesteps = adapted_timesteps
        sampler = DAPSSampler(alphas_cumprod, timesteps, args.max_num_ode_steps)
        torch.manual_seed(random_state)
        x_start_batch = torch.stack([torch.randn((n, reference_samples.shape[1])) for n in split_number(n_samples, max_samples_per_batch)])
        x_start_batch = x_start_batch.to(device)
        samples_daps_list = []
        st = time.time()
        for x_start in tqdm(x_start_batch):
            samples_daps_list.append(sampler.sample(x_start, score_network, observation.reshape(1, observation.shape[-1]), operator))
        et = time.time()
        runtime = et - st
        samples = torch.cat(samples_daps_list).cpu()
    else:
        raise ValueError(f"{args.algo} not found")
    assert samples.shape[0] == n_samples
    print(f"took {runtime}")

    swd_projections = 1000
    swd_p = 1
    plot_samples(samples, reference_samples, args)
    swd = float(ot.sliced.sliced_wasserstein_distance(samples, reference_samples, seed=random_state, n_projections=swd_projections, p=swd_p))
    print(f"{args.algo} sliced Wasserstein distance:")
    print(swd)
    loglik = float(torch.sum(posterior.log_prob(samples.to(args.device))))
    print(f"{args.algo} log likelihood:")
    print(loglik)


if __name__ == "__main__":
    args = get_args()
    main(args)