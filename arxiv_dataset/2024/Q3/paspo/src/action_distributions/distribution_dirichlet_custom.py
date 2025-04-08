""" This implementation fixes the RLlib implementation"""

import torch
import numpy as np
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

from ray.rllib.utils.typing import TensorType

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.annotations import override, DeveloperAPI
import math

torch, nn = try_import_torch()


@DeveloperAPI
class TorchDirichletCustom(TorchDistributionWrapper):
    """Dirichlet distribution for continuous actions that are between
    [0,1] and sum to 1.
    e.g. actions that represent resource allocation."""

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        """
        self.epsilon = torch.tensor(1e-6).to(inputs.device)
        concentration = torch.exp(inputs) + self.epsilon
                
        self.dist = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )
        super().__init__(concentration, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        """
        Returns the expected value for the dirichlet
        :return:
        """

        # removed depreciated warning by adding dim=1
        self.last_sample = nn.functional.softmax(self.dist.concentration, dim=1)
        return self.last_sample
            
    @override(ActionDistribution)
    def sample(self) -> TensorType:
        self.last_sample = self.dist.rsample()
        return self.last_sample
    
    @override(ActionDistribution)
    def logp(self, x):
        """
        Calculates the log prob to observe a certain action
        Support of Dirichlet are positive real numbers. x is already
        an array of positive numbers, but we clip to avoid zeros due to
        numerical errors.
        :param x:
        :return:
        """
        x = torch.max(x, self.epsilon)
        x = x / torch.sum(x, dim=-1, keepdim=True)
                
        # #annoying initial dummy loss by rllib
        # if x.sum().item() == 0.0:
        #     x = torch.max(x, torch.tensor(1e-7).to(x.device))
        #     x = x / torch.sum(x, dim=-1, keepdim=True)
        
        logp = self.dist.log_prob(x)
        
        return logp

    @override(ActionDistribution)
    def entropy(self):
        """
        Returns the distributions entropy
        :return:
        """
        return self.dist.entropy()


    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        """
        Returns the output shape of the model
        :param action_space:
        :param model_config:
        :return:
        """
        return np.prod(action_space.shape)



@DeveloperAPI
class TorchDirichletCustomStable(TorchDistributionWrapper):
    """Dirichlet distribution for continuous actions that are between
    [0,1] and sum to 1.
    e.g. actions that represent resource allocation."""

    def __init__(self, inputs, model):
        """Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.
        See issue #4440 for more details.
        """
        self.epsilon = torch.tensor(1e-6).to(inputs.device)
        self.log_epsilon = math.log(1e-6)
        
       
        concentration = torch.exp(inputs) + self.epsilon
        
        self.dist = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True,
        )
        super().__init__(concentration, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        """
        Returns the expected value for the dirichlet
        :return:
        """

        # removed depreciated warning by adding dim=1
        self.last_sample = nn.functional.softmax(self.dist.concentration, dim=1)
        return self.last_sample
            
    @override(ActionDistribution)
    def sample(self) -> TensorType:
        self.last_sample = self.dist.rsample()
        return self.last_sample
    
    @override(ActionDistribution)
    def logp(self, x):
        """
        Calculates the log prob to observe a certain action
        Support of Dirichlet are positive real numbers. x is already
        an array of positive numbers, but we clip to avoid zeros due to
        numerical errors.
        :param x:
        :return:
        """
        x = torch.max(x, self.epsilon)
        x = x / torch.sum(x, dim=-1, keepdim=True)
                
        # #annoying initial dummy loss by rllib
        # if x.sum().item() == 0.0:
        #     x = torch.max(x, torch.tensor(1e-7).to(x.device))
        #     x = x / torch.sum(x, dim=-1, keepdim=True)
        
        logp = self.dist.log_prob(x)

        logp = torch.clamp(logp, min=self.log_epsilon)
        
        return logp

    @override(ActionDistribution)
    def entropy(self):
        """
        Returns the distributions entropy
        :return:
        """
        return self.dist.entropy()

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        """
        Returns the output shape of the model
        :param action_space:
        :param model_config:
        :return:
        """
        return np.prod(action_space.shape)
