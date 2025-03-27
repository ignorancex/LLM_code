from src.models.lorenz_96_model import Lorenz96SDE
from src.scoring_rules.scoring_rules import EnergyScore, KernelScore 
from src.transformers import BoundedVarTransformer
from src.sampler.gradient_estimation import build_gradient_estimation_fn, build_neural_l96_gradient_estimation_fn
from src.sampler.kernels import build_sgnht_lfi_kernel
from src.sampler.optimisers import build_adam_lfi_optimizer
from src.sampler.sgMCMC import SGMCMC
import jax
from jax import jit, random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from abcpy.continuousmodels import ProbabilisticModel, Continuous, InputConnector, Uniform, Normal
from abcpy.statistics import Statistics, Identity
import pickle
import functools
from einops import rearrange

torch.set_default_dtype(torch.float64)

def joint_log_prob(param_torch_unconstrained, obs, model, scoring_rule, transformer, n_samples_per_param=500):
    # 1. Transform to constrained space
    # 2. Calculate LAJ (unconstrained)
    # 3. Calculate log prior (constrained)
    # 4. Calculate log likelihood / score (constrained) (weights should be initialised in the scoring rule class)

    param_torch_constrained = transformer.inverse_transform(param_torch_unconstrained, use_torch=True)
    # LAJ
    laj = transformer.jac_log_det_inverse_transform(param_torch_unconstrained, use_torch=True)

    # Log prior
    log_prior = 0 # Uniform prior

    # Log likelihood    
    sims = model.torch_forward_simulate(params=param_torch_constrained, num_forward_simulations=n_samples_per_param, normalise=True) 
    log_ll = scoring_rule.loglikelihood(y_obs = obs, y_sim = sims, use_torch=True)
    model.scores.append(log_ll.detach()) #Detach otherwise mem explodes!
    return laj + log_prior + log_ll

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    n_samples = 20000
    dt = 1e-4
    for idx in range(5):
        obs_path = f'./experiments/Exp-5/obs/lorenz96_obs_{idx}.pty'
        init_obs_path = f'./experiments/Exp-5/obs/lorenz96_obs_initial{idx}.pty'


        # Transformer for parameters
        lower_limit_arr = np.array([1.4, 0, 0.])
        upper_limit_arr = np.array([2.2, 1, 1.5])
        bounded_trans = BoundedVarTransformer(lower_bound=lower_limit_arr, upper_bound=upper_limit_arr)

        # Observations
        obs = torch.load(obs_path)
        init_value = torch.load(init_obs_path)
        print(obs.shape)
        print(init_value.shape)

        obs = rearrange(obs, 't d -> t 1 d')
        y0 = init_value.double()
        obs = (obs - y0.mean()) / y0.std()
        obs = rearrange(obs, 't 1 d -> 1 (t d)').double()

        # Scoring Rule
        es = EnergyScore(weight=1)

        # Model
        l96 = Lorenz96SDE(y0=y0)

        # Setup joint log prob function
        joint_log_prob_func = functools.partial(joint_log_prob, model=l96, scoring_rule=es, transformer=bounded_trans)

        # Sampler
        sampler = SGMCMC(l96, observations=obs, joint_log_prob=joint_log_prob_func, transformer=bounded_trans, n_samples=n_samples)

        # Run SGMCMC
        op = sampler.sample(use_mamba=False, step_size=dt)

        with open(f'sgnht_l96_{idx}.pickle', 'wb') as handle:
            pickle.dump(op, handle, protocol=pickle.HIGHEST_PROTOCOL)

