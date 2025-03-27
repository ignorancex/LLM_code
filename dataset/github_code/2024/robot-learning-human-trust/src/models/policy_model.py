import torch

from typing import Tuple


class RobotPolicy(torch.nn.Module):

    def __init__(self,
                 state_size: int = 3,
                 hidden_size: int = 32,
                 out_size: int = 7,
                 log_std_min: float = -24,
                 log_std_max: float = 2.4,
                 device: str = "cpu") -> object:
        
        super(RobotPolicy,
              self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device

        # policy network backbone linear layers
        self.backbone = torch.nn.Sequential(torch.nn.Linear(state_size, hidden_size, bias=True),
                                            torch.nn.ReLU())
        
        # mean and log standard deviation of Gaussian policy
        self.policy_mu = torch.nn.Linear(hidden_size, out_size, bias=True)
        self.policy_logstd = torch.nn.Linear(hidden_size, out_size, bias=True)
    
    def forward(self,
                x: torch.Tensor) -> Tuple[torch.Tensor,
                                          torch.Tensor]:
        
        # propagate through policy network backbone
        x = self.backbone(x)

        # mean and log standard deviation of Gaussian policy
        action_mu = self.policy_mu(x)
        action_log_std = self.policy_logstd(x)
        
        # constrain logits to reasonable values to match with demonstration distribution variance
        act_log_std = torch.clamp(input=action_log_std,
                                  min=self.log_std_min,
                                  max=self.log_std_max)
        action_std = torch.exp(act_log_std)
        
        # deterministic action resembling mean of Gaussian distribution
        return action_mu, action_std
    
    def estimate_action(self,
                        state: torch.Tensor,
                        is_policy_inference: bool = False,
                        is_deterministic: bool = True) -> Tuple[torch.Tensor,
                                                                torch.Tensor,
                                                                torch.Tensor,
                                                                torch.Tensor]:
        
        # forward pass to get mean and std of Gaussian distribution
        if is_policy_inference:
            with torch.no_grad():
                action_mu, action_std = self.forward(x=state)
        
        else:
            action_mu, action_std = self.forward(x=state)
        
        # pre-squashed action distibution is assumed to be normal with mean and std
        action_distibution = torch.distributions.Normal(loc=action_mu,
                                                        scale=action_std)
        
        if is_deterministic:
            # deterministic action
            pi_action = action_mu
        else:
            # sampling using reparameterization trick
            pi_action = action_distibution.rsample()
        
        # log probability of the action
        action_log_prob = action_distibution.log_prob(value=pi_action).sum(axis=-1)

        # policy distribution entropy
        action_entropy = action_distibution.entropy().mean()
        
        # squashed (with Tanh) action [-1, 1]
        action = torch.tanh(pi_action)

        return action, action_std, action_log_prob, action_entropy
