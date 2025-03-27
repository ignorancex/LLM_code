import torch
from abcpy.output import Journal
import numpy as np
import matplotlib.pyplot as plt
from sgmcmcjax.ksd import imq_KSD
import jax.numpy as jnp
import abcpy.statistics
import matplotlib.pyplot as plt
from abcpy.acceptedparametersmanager import AcceptedParametersManager
from abcpy.approx_lhd import SynLikelihood, SemiParametricSynLikelihood
from abcpy.inferences import InferenceMethod
from abcpy.output import Journal
from scipy.stats import gaussian_kde

class GradientNANException(Exception):
    pass

# the following is based on the RejectionABC; I use it to sample from prior.
class DrawFromPrior(InferenceMethod):
    model = None
    rng = None
    n_samples = None
    backend = None

    n_samples_per_param = None  # this needs to be there otherwise it does not instantiate correctly

    def __init__(self, root_models, backend, seed=None, discard_too_large_values=True):
        self.model = root_models
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.discard_too_large_values = discard_too_large_values
        # An object managing the bds objects
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

    def sample(self, n_samples, n_samples_per_param):
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param
        self.accepted_parameters_manager.broadcast(self.backend, 1)

        # now generate an array of seeds that need to be different one from the other. One way to do it is the
        # following.
        # Moreover, you cannot use int64 as seeds need to be < 2**32 - 1. How to fix this?
        # Note that this is not perfect; you still have small possibility of having some seeds that are equal. Is there
        # a better way? This would likely not change much the performance
        # An idea would be to use rng.choice but that is too
        seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
        # check how many equal seeds there are and remove them:
        sorted_seed_arr = np.sort(seed_arr)
        indices = sorted_seed_arr[:-1] == sorted_seed_arr[1:]
        # print("Number of equal seeds:", np.sum(indices))
        if np.sum(indices) > 0:
            # the following can be used to remove the equal seeds in case there are some
            sorted_seed_arr[:-1][indices] = sorted_seed_arr[:-1][indices] + 1
        # print("Number of equal seeds after update:", np.sum(sorted_seed_arr[:-1] == sorted_seed_arr[1:]))
        rng_arr = np.array([np.random.RandomState(seed) for seed in sorted_seed_arr])
        rng_pds = self.backend.parallelize(rng_arr)

        parameters_simulations_pds = self.backend.map(self._sample_parameter, rng_pds)
        parameters_simulations = self.backend.collect(parameters_simulations_pds)
        parameters, simulations = [list(t) for t in zip(*parameters_simulations)]

        parameters = np.array(parameters)
        simulations = np.array(simulations)

        parameters = parameters.reshape((parameters.shape[0], parameters.shape[1]))
        if len(simulations.shape) == 4:
            simulations = simulations.reshape((simulations.shape[0], simulations.shape[2], simulations.shape[3],))
        elif len(simulations.shape) == 5:
            simulations = simulations.reshape((simulations.shape[0], simulations.shape[2], simulations.shape[4],))

        return parameters, simulations

    def sample_in_chunks(self, n_samples, n_samples_per_param, max_chunk_size=10 ** 4):
        """This splits the data generation in chunks. It is useful when generating large datasets with MPI backend,
        which gives an overflow error due to pickling very large objects."""
        parameters_list = []
        simulations_list = []
        samples_to_sample = n_samples
        while samples_to_sample > 0:
            parameters_part, simulations_part = self.sample(min(samples_to_sample, max_chunk_size), n_samples_per_param)
            samples_to_sample -= max_chunk_size
            parameters_list.append(parameters_part)
            simulations_list.append(simulations_part)
        parameters = np.concatenate(parameters_list)
        simulations = np.concatenate(simulations_list)
        return parameters, simulations

    def _sample_parameter(self, rng, npc=None):
        ok_flag = False

        while not ok_flag:
            self.sample_from_prior(rng=rng)
            theta = self.get_parameters(self.model)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)

            # if there are no potential infinities there (or if we do not check for those).
            # For instance, Lorenz model may give too large values sometimes (quite rarely).
            if np.sum(np.isinf(np.array(y_sim).astype("float32"))) > 0 and self.discard_too_large_values:
                print("y_sim contained too large values for float32; simulating again.")
            else:
                ok_flag = True

        return theta, y_sim


def heuristics_estimate_w(model_abc, observation, target_SR, reference_SR, backend, n_theta=100,
                          n_theta_prime=100, n_samples_per_param=100, seed=42, return_values=["median"]):
    """Here observation is a list, and all of them are used at once in the SR. """
    # target_scoring_rule = dict_implemented_scoring_rules()[target_SR](statistics, **target_SR_kwargs)
    # reference_scoring_rule = dict_implemented_scoring_rules()[reference_SR](statistics)
    target_scoring_rule = target_SR
    reference_scoring_rule = reference_SR

    # generate the values of theta from prior
    theta_vect, simulations_theta_vect = DrawFromPrior([model_abc], backend, seed=seed).sample(n_theta,
                                                                                               n_samples_per_param)
    # generate the values of theta_prime from prior
    theta_prime_vect, simulations_theta_prime_vect = DrawFromPrior([model_abc], backend, seed=seed + 1).sample(
        n_theta_prime, n_samples_per_param)

    # now need to estimate w; here we assume the reference post
    # corresponding to the reference sr has no weight factor, so that log BF =
    w_estimates = np.zeros((n_theta, n_theta_prime))
    target_sr_1 = np.zeros((n_theta))
    reference_sr_1 = np.zeros((n_theta))
    target_sr_2 = np.zeros((n_theta_prime))
    reference_sr_2 = np.zeros((n_theta_prime))
    for i in range(n_theta):
        simulations_theta_i = simulations_theta_vect[i]
        simulations_theta_i = [data for data in simulations_theta_i]  # convert to list
        target_sr_1[i] = target_scoring_rule.loglikelihood(observation, simulations_theta_i)
        reference_sr_1[i] = reference_scoring_rule.loglikelihood(observation, simulations_theta_i)

    for j in range(n_theta_prime):
        simulations_theta_prime_j = simulations_theta_prime_vect[j]
        simulations_theta_prime_j = [data for data in simulations_theta_prime_j]  # convert to list
        target_sr_2[j] = target_scoring_rule.loglikelihood(observation, simulations_theta_prime_j)
        reference_sr_2[j] = reference_scoring_rule.loglikelihood(observation, simulations_theta_prime_j)

    # actually loglik is (- SR), but we have - factor both in numerator and denominator -> doesn't matter
    for i in range(n_theta):
        for j in range(n_theta_prime):
            w_estimates[i, j] = (reference_sr_1[i] - reference_sr_2[j]) / (
                    target_sr_1[i] - target_sr_2[j])

    w_estimates = w_estimates.flatten()
    print("There were ", np.sum(np.isnan(w_estimates)), " nan values out of ", n_theta * n_theta_prime)
    w_estimates = w_estimates[~np.isnan(w_estimates)]  # drop nan values

    return_list = []
    if "median" in return_values:
        return_list.append(np.median(w_estimates))
    if "mean" in return_values:
        return_list.append(np.mean(w_estimates))

    return return_list[0] if len(return_list) == 1 else return_list


def estimate_bandwidth(model_abc, statistics, backend, n_theta=100, n_samples_per_param=100, seed=42,
                       return_values=["median"]):
    """Estimate the bandwidth for the gaussian kernel in KernelSR. Specifically, it generates n_samples_per_param
    simulations for each theta, then computes the pairwise distances and takes the median of it. The returned value
    is the median (by default; you can also compute the mean if preferred) of the latter over all considered values
    of theta.  """

    # generate the values of theta from prior
    theta_vect, simulations_theta_vect = DrawFromPrior([model_abc], backend, seed=seed).sample(n_theta,
                                                                                               n_samples_per_param)
    if not isinstance(statistics, abcpy.statistics.Identity):
        simulations_theta_vect_list = [x for x in simulations_theta_vect.reshape(-1, simulations_theta_vect.shape[-1])]
        simulations_theta_vect = statistics.statistics(simulations_theta_vect_list)
        simulations_theta_vect = simulations_theta_vect.reshape(n_theta, n_samples_per_param,
                                                                simulations_theta_vect.shape[-1])

    print("Simulations shape for learning bandwidth", simulations_theta_vect.shape)

    distances = np.zeros((n_theta, n_samples_per_param * (n_samples_per_param - 1)))
    for theta_index in range(n_theta):
        simulations = simulations_theta_vect[theta_index]
        distances[theta_index] = np.linalg.norm(
            simulations.reshape(1, n_samples_per_param, -1) - simulations.reshape(n_samples_per_param, 1, -1), axis=-1)[
            ~np.eye(n_samples_per_param, dtype=bool)].reshape(-1)

    # distances = distances.reshape(n_theta, -1)  # reshape
    # take the median over the second index:
    distances_median = np.median(distances, axis=-1)

    return_list = []
    if "median" in return_values:
        return_list.append(np.median(distances_median.flatten()))
    if "mean" in return_values:
        return_list.append(np.mean(distances_median.flatten()))

    return return_list[0] if len(return_list) == 1 else return_list

def ts_sig_transform(paths, at=False, ll=False, scale=1.,use_cuda=False):
    #PATHS SHOULD BE tensor(B,T,D)

    paths = scale*paths
    def _lead_lag(x):
        return torch.cat([torch.repeat_interleave(x,2,dim=1)[:,:-1,:], torch.repeat_interleave(x,2,dim=1)[:,1::,:]], dim=2) 

    def _add_time(x, use_cuda):
        if use_cuda:
            return torch.cat((torch.linspace(0.,1.,x.shape[1]).cuda().reshape(1,x.shape[1],1).repeat(x.shape[0],1,1), x), -1)
        else:
            return torch.cat((torch.linspace(0.,1.,x.shape[1]).reshape(1,x.shape[1],1).repeat(x.shape[0],1,1), x), -1)

    if ll:
        paths = _lead_lag(paths)
    if at:
        paths = _add_time(paths, use_cuda)

    return paths

def get_grads(param, transformer, model, n_samples_per_param, data, scoring_rule, w=1, grad_log_prior=None):
    #param here is in unbounded space
    param_torch_unconstrained = torch.from_numpy(np.array(transformer.transform(np.asarray(param))))
    grad_LAJ = transformer.log_det_gradient(param_torch_unconstrained)
    if torch.isnan(grad_LAJ).any():
        print("WARNING: NAN GRADIENT IN LOG DET GRADIENT")


    param_torch_constrained = transformer.inverse_transform(param_torch_unconstrained, use_torch=True)#.detach()
    if grad_log_prior:
        grad_LP = grad_log_prior(param_torch_constrained)
        if torch.isnan(grad_LP).any():
            print("WARNING: NAN GRADIENT IN LOG PRIOR GRADIENT")
    else:
        grad_LP = 0

    if model:
        sims = model.torch_forward_simulate(param_torch_constrained, n_samples_per_param)
        score = scoring_rule.score(observations=data, simulations=sims, use_torch=True)
        score.backward()
        sr_grad = model.get_grads()
        if torch.isnan(sr_grad).any():
            print("WARNING: NAN GRADIENT IN SCORING RULE GRADIENT")
    else:
        sr_grad = 0

    param_grad = grad_LP + grad_LAJ - w * sr_grad 
    return param_grad

def transform_neural_lorenz_parameter(params):
    # Takes the unconstrained params
    # param 769-dim tensor
    # Want to transform 768 dim MLP params to 
    # (20,8) = 160 x 20 x (20,20) = 400 x 20 x (8,20) = 160 x 8 
    # param 349-dim tensor
    # Want to transform 348 dim MLP params to 
    # (20,8) = 160 x 20 x (8,20) = 160 x 8 
    # param 111-dim tensor
    # Want to transform 110 dim MLP params to 
    # (6,8) = 48 x 6 x (8,6) = 48 x 8 
    # Returns 2-tuple of (sigma, MLP params) 
    mlp_params = torch.split(params[1:], [48, 6, 48, 8])
    mlp_params = (mlp_params[0].view(6, 8), mlp_params[1], mlp_params[2].view(8, 6), mlp_params[3])
    mlp_params = tuple(torch.nn.Parameter(p) for p in mlp_params)

    sigma = params[0].detach()
    sigma.requires_grad = True

    return [sigma, mlp_params]