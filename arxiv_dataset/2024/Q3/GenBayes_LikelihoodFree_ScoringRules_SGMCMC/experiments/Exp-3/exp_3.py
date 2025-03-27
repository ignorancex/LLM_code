from src.models.g_and_k_model import mv_g_and_k
from src.scoring_rules.scoring_rules import EnergyScore, KernelScore 
from src.transformers import BoundedVarTransformer
from src.utils import heuristics_estimate_w, estimate_bandwidth
from abcpy.continuousmodels import ProbabilisticModel, Continuous, InputConnector, Uniform, Normal
from abcpy.statistics import Statistics, Identity
from abcpy.backends import BackendDummy
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
    log_prior = 0 #Uniform prior

    # Log likelihood    
    sims = model.torch_forward_simulate(param_torch_constrained, n_samples_per_param)
    log_ll = scoring_rule.loglikelihood(y_obs = obs, y_sim = sims, use_torch=True)
    model.scores.append(log_ll.detach()) #Detach otherwise mem explodes!

    return laj + log_prior + log_ll

if __name__ == "__main__":
    n_samples = 110000

    # Transformer for parameters
    L = np.array([0,0,0,0,-np.sqrt(3)/3])
    U = np.array([4,4,4,4, np.sqrt(3)/3])
    b = BoundedVarTransformer(lower_bound=L, upper_bound=U)

    # Observations
    obs_well_specified = torch.load("./experiments/Exp-3/gk_obs_4.pt")
    obs_misspecified = torch.load("./experiments/Exp-3/gk_obs_5.pt")
    obs_count_list = (1,10,20,50,70,100,200,400)

    # Energy Scoring Rule
    es = EnergyScore(weight=1)

    # Model
    gk = mv_g_and_k()

    # Setup joint log prob function
    joint_log_prob_func = functools.partial(joint_log_prob, model=gk, scoring_rule=es, transformer=b)

    # Energy Score
    exp_num = 1
    score_name = 'es'
    for obs in (obs_well_specified, obs_misspecified):
        for i in obs_count_list:
            # Sampler
            sampler = SGMCMC(gk, observations=obs[:i], joint_log_prob=joint_log_prob_func, transformer=b, n_samples=n_samples)
            # Run sampler
            op = sampler.sample(use_mamba=True)

            with open('sgnht_mvgk_obs-' + str(i) + '_exp-' + str(exp_num) + '_' + score_name + '.pkl', 'wb') as handle:
                pickle.dump(op, handle, protocol=pickle.HIGHEST_PROTOCOL)
        exp_num += 1

    # Run Kernel Score Experiments
    exp_num = 1
    score_name = 'ks'
    for obs in (obs_well_specified, obs_misspecified):
        for i in obs_count_list:
            # Kernel Score
            # W and Sigma estimation
            gk = mv_g_and_k()
            es = EnergyScore()
            obs_est = obs
            backend = BackendDummy()
            statistics = Identity()

            bw = estimate_bandwidth(model_abc=gk, statistics=statistics, backend=backend, n_theta=1000, n_samples_per_param=500)

            ks_no_weight = KernelScore(sigma=bw)
            weight = heuristics_estimate_w(model_abc=gk, observation=obs_est[0].numpy(), target_SR=ks_no_weight, reference_SR=es, backend=backend, n_theta=1000, n_theta_prime=1000, n_samples_per_param=100)

            print(bw)
            print(weight)
            ks = KernelScore(sigma=bw, weight=weight)
            joint_log_prob_func = functools.partial(joint_log_prob, model=gk, scoring_rule=ks, transformer=b)

            # Sampler
            sampler = SGMCMC(gk, observations=obs[:i], joint_log_prob=joint_log_prob_func, transformer=b, n_samples=n_samples, w=weight)
            # Run sampler
            op = sampler.sample(use_mamba=True)

            with open('sgnht_mvgk_obs-' + str(i) + '_exp-' + str(exp_num) + '_' + score_name + '.pkl', 'wb') as handle:
                pickle.dump(op, handle, protocol=pickle.HIGHEST_PROTOCOL)
        exp_num += 1



