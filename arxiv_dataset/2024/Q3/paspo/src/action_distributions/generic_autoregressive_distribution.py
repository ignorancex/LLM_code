from typing import Any
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import List, TensorType
import torch
import numpy as np

from action_distributions.helper_distributions import ListDistribution
from action_distributions.raw_distribution_funcs import _create_distribution_beta_list, _create_distribution_gaussian_list, _create_distribution_truncated_normal_list
from utils.lp_solver import solve_min_max




class TorchAutoregressiveGenericDistribution(TorchDistributionWrapper):
    def __init__(self, inputs: List[Any], model: TorchModelV2):
        super().__init__(inputs, model)
        
        self.A = model.A
        self.b = model.b
        
        self._n_dim = model.n_dimensions
        
        if model.action_dist_name == "beta":
            self._dist_func = _create_distribution_beta_list
        elif model.action_dist_name == "truncnorm":
            self._dist_func = _create_distribution_truncated_normal_list
        elif model.action_dist_name == "squashed_gaussian":
            self._dist_func = _create_distribution_gaussian_list
        else:
            raise ValueError(f"Unknown action_dist_name: {model.action_dist_name}")
        
        self.add_only_action = model.add_only_action
        self._dists = None
        self._rsample = model.use_r_sample #use rsample if the model is set to use it
        
        self.feas_tol = model.model_config["custom_model_config"]["feas_tol"]
        self.opt_tol = model.model_config["custom_model_config"]["opt_tol"]
        
        self.parallel_lp = model.model_config["custom_model_config"]["use_lp_parallel"]
        
        self.lp_sub_proc = model.lp_sub_proc #get form model so that it is not reinitialized every time
        
        
    
    def evaluate_polytope_boundaries(self, A, b, unallocated):
        n = A.shape[-1]

        if self.parallel_lp and unallocated.shape[0] > 1:
            self.lp_sub_proc.send([(A[i], b[i], 0, n, unall.item()) for i, unall in enumerate(unallocated)])
            min_max = self.lp_sub_proc.recv(unallocated.shape[0])
        else:            
            min_max = [solve_min_max(A[i], b[i], 0, n, unall.item()) for i, unall in enumerate(unallocated)]

        
        
        min_max = np.array(min_max)

        return min_max[:,0], min_max[:,1]
    
    def _calc_mean(self, distribution):        
        x = torch.stack([self._transforms(dist, dist.base_dist.mode) for dist in distribution.distributions], dim=-1)
        
        return x
        

    def _transforms(self, distribution, a):
        for transform in distribution.transforms:
            a = transform(a)
        return a

    def _forward(self, existing_actions = None, deterministic=False, rsample = False):
        batch_size = self.inputs.shape[0]
        
        unallocated = np.ones((batch_size,1), dtype=np.float32)

        A = self.A[None, ...].repeat(batch_size, axis=0)
        b = self.b[None, ...].repeat(batch_size, axis=0)

        min, max = self.evaluate_polytope_boundaries(A, b, unallocated)
        
        parameters_0 = self.model.heads[0](self.inputs)

        distribution_0 = self._dist_func(parameters_0, min, max)
        
        if existing_actions is not None:
            action = existing_actions[..., 0][..., None]
        elif deterministic:
            action = self._calc_mean(distribution_0)[..., None]
        else:
            action = distribution_0.rsample()[..., None] if rsample else distribution_0.sample()[..., None]
            
        actions = [action]
        bounds = [torch.Tensor(np.concatenate((min[:, None], max[:, None]), axis=-1)).to(action.device)]
        
        action_np = action.detach().cpu().numpy()
        unallocated -= action_np

        b = b - (action_np * A[...,0])
        A = A[...,1:]
        distributions = [distribution_0]
        parameters = [parameters_0]

        for i, module in enumerate(self.model.heads[1:]):
            if self.add_only_action:
                parameters_current = module(torch.cat([self.inputs, *actions], dim=-1))
            else:
                parameters_current = module(torch.cat([self.inputs, *actions, *parameters, *bounds], dim=-1))

            parameters.append(parameters_current)

            min, max = self.evaluate_polytope_boundaries(A,b, unallocated)

            distribution = self._dist_func(parameters_current, min, max)

            distributions.append(distribution)
            
            bounds.append(torch.Tensor(np.concatenate((min[:, None], max[:, None]), axis=-1)).to(action.device))
            
            if existing_actions is not None:
                action = existing_actions[..., i+1][..., None]
            elif deterministic:
                action = self._calc_mean(distribution)[..., None]
            else:
                action = distribution.rsample()[..., None] if rsample else distribution.sample()[..., None]

            action_np = action.detach().cpu().numpy()

            b = b - (action_np * A[...,0])
            A = A[...,1:]

            unallocated -= action_np

            actions.append(action)

        actions = torch.cat(actions, dim=-1)
        last_action = 1.0 - actions.sum(dim=-1, keepdim=True)
        last_action = torch.relu(last_action) #due to numerical precision errors
        actions = torch.cat([actions, last_action], dim=-1)        

        
        # #check if any of the last actions is negative
        # if (actions < 0.0).any():
        #     print("Warning: action is negative")
        #     print(f"{actions[actions < 0.0]}")
        
        # if ((actions.sum(dim=-1) - 1.0) > 0.0).any():
        #     print("Warning: Sum of actions is not 1.0")
        #     print(f"{actions.sum(dim=-1)[actions.sum(dim=-1) != 1.0]}")        
        
        distributions = ListDistribution(distributions, sum=True)
        
        
        return actions, distributions

    def deterministic_sample(self) -> Any:
        if self._dists is not None:
            raise RuntimeError("Do not create new samples (sampling requires the autoregressive model to be run again)")

        actions, distributions = self._forward(deterministic=True)
        
        self._action_logp = distributions.log_prob(actions)

        self._dists = distributions
        
        return actions

    def sample(self) -> Any:
        if self._dists is not None:
            raise RuntimeError("Do not create new samples (sampling requires the autoregressive model to be run again)")
            
        actions, distributions = self._forward(rsample=self._rsample)
        self._action_logp = distributions.log_prob(actions)
        
        self._dists = distributions
        
        return actions

        
        
    def sampled_action_logp(self):
        return self._action_logp
    
    def logp(self, actions: Any) -> Any:
        if self._dists is None:
            if actions.sum() == 0.0:
                actions = None  #this is very annoying since rllib needs to do some inits with dummy values....
            actions, distributions = self._forward(existing_actions=actions, rsample=self._rsample)
            self._dists = distributions
            
            
        return self._dists.log_prob(actions)
    
    def entropy(self) -> Any:
        if self._dists is None:
            raise RuntimeError("Expected entropy requires already samples")
        
        return self._dists.entropy()
    
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config["custom_model_config"]["state_encoder_hidden_dim"]  # controls model output feature vector size