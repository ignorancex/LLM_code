import numpy as np

import torch


class Updater(object):

    def __init__(self,
                 configs: object,
                 policy_network: torch.nn.Module,
                 reward_network: torch.nn.Module) -> object:
        
        self.configs = configs
        self.device = configs.device

        self.policy_network = policy_network
        self.reward_network = reward_network
        self.policy_optimizer = None
        self.reward_optimizer = None

    def calculate_bc_loss(self,
                          action_dist: torch.Tensor,
                          output_action: torch.Tensor) -> float:
        
        loss_nll = -action_dist.log_prob(output_action).sum(axis=-1)
        batch_loss = loss_nll.mean()

        return batch_loss
    
    def multivariate_gaussian_nll_loss(self,
                                       action_true: torch.FloatTensor,
                                       action_pred_mu: torch.FloatTensor,
                                       action_log_std: torch.FloatTensor,
                                       std_weight: float = 1.0) -> torch.Tensor:
        
        # size of action space
        n_dims = action_true.shape[1]
        
        # mean squared error given mean and standard deviation
        mse = 0.5 * torch.sum(torch.square((action_true - action_pred_mu) / torch.exp(action_log_std)),
                              dim=1)
        # sum of predicted log standard deviations to penalize higher uncertainties in predictions
        sigma_trace = torch.sum(action_log_std,
                                dim=1)
        # constant term related to the natural logarithm of 2 pi
        log_2_pi = 0.5 * n_dims * np.log(2 * np.pi)
        
        # multivariate Gaussian negative log-likelihood loss function
        log_likelihood = mse + log_2_pi + std_weight * sigma_trace

        # batch loss summed to apply different weights to different samples in the future
        total_loss = torch.sum(log_likelihood, dim=0)

        return total_loss
    
    def calculate_max_entropy_loss(self,
                                   demonstration_rewards: torch.Tensor,
                                   robot_rewards: torch.Tensor,
                                   log_probability: torch.Tensor) -> torch.Tensor:
        
        # max-entropy loss function
        loss = - torch.mean(demonstration_rewards) + \
            torch.log(torch.mean(torch.exp(robot_rewards) / (torch.exp(log_probability) + 1e-7)))
        
        return loss
    
    def calculate_irl_loss(self,
                           demo_traj_reward: torch.Tensor,
                           robot_traj_reward: torch.Tensor,
                           log_probability: torch.Tensor,
                           nu_factor: torch.Tensor) -> float:
        
        # max-entropy inverse reinforcement learning loss function (similar to guided cost learning)
        loss = -torch.mean(demo_traj_reward) + \
                    (torch.exp(nu_factor) * \
                        (torch.logsumexp(robot_traj_reward - log_probability,
                                         dim=0,
                                         keepdim=True)) - \
                        torch.log(torch.Tensor([len(robot_traj_reward)]))
                    )

        return loss
    
    def calculate_sample_traj_loss(self,
                                   nu_factor: torch.Tensor,
                                   robot_traj_reward: torch.Tensor,
                                   log_probability: torch.Tensor,
                                   M_num: int) -> float:
        
        # loss value estimated from approximation of the background distribution (Z) in max-entropy IRL
        loss = torch.exp(nu_factor) * \
            (torch.logsumexp(torch.mean(robot_traj_reward) - log_probability,
                             dim=0,
                             keepdim=True)) - \
            torch.log(torch.Tensor([M_num]))
        
        return loss
    
    def calculate_partition_function(self,
                                     nu_factor: torch.Tensor,
                                     robot_traj_reward: torch.Tensor,
                                     log_probability: torch.Tensor) -> float:
        
        # loss value estimated from approximation of the background distribution (Z) in max-entropy IRL
        loss = torch.exp(nu_factor) * \
            (torch.exp(torch.mean(robot_traj_reward)) / torch.prod(torch.exp(log_probability)))
        
        return loss

    def calculate_max_margin_loss(self,
                                  demo_traj_reward: torch.Tensor,
                                  robot_traj_reward: torch.Tensor,
                                  nu_factor: torch.Tensor) -> float:
        
        # max-margin loss function
        loss = torch.max(torch.zeros_like(demo_traj_reward),
                         robot_traj_reward[:-1] - demo_traj_reward + 1)
        
        return torch.sum(loss)
    
    def calculate_policy_gradient_loss(self,
                                       cumulative_log_probs: torch.Tensor,
                                       advantages: torch.Tensor,
                                       entropy: torch.Tensor,
                                       entropy_weight: int = 1e-2) -> torch.Tensor:
        
        # negative log-likelihood multiplied by cummulative rewards (advantage values)
        weighted_log_probs = cumulative_log_probs * advantages
        policy_loss = - torch.mean(weighted_log_probs, dim=0)

        # entropy regularization
        entropy_loss = - entropy_weight * torch.mean(entropy, dim=0)

        loss = policy_loss + entropy_loss
        
        return loss

    def initialize_optimizers(self) -> None:
        
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(),
                                                 lr=self.configs.policy_lr,
                                                 weight_decay=self.configs.policy_weight_decay)
        self.reward_optimizer = torch.optim.Adam(self.reward_network.parameters(),
                                                 lr=self.configs.reward_lr,
                                                 weight_decay=self.configs.reward_weight_decay)
        
        self.policy_scheduler = torch.optim.lr_scheduler.StepLR(self.policy_optimizer,
                                                                step_size=self.configs.policy_scheduler_step_size,
                                                                gamma=self.configs.policy_scheduler_gamma)
        self.reward_scheduler = torch.optim.lr_scheduler.StepLR(self.reward_optimizer,
                                                                step_size=self.configs.reward_scheduler_step_size,
                                                                gamma=self.configs.reward_scheduler_gamma)
    
    def run_policy_optimizer(self,
                             policy_loss: torch.Tensor) -> None:
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_scheduler.step()
    
    def run_reward_optimizer(self,
                             reward_loss: torch.Tensor) -> None:
        
        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()
        self.reward_scheduler.step()
