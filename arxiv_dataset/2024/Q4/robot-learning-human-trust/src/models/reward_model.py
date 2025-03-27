import torch


class RewardFunction(torch.nn.Module):

    def __init__(self,
                 state_size: int = 3,
                 action_size: int = 7,
                 hidden_size: int = 64,
                 out_size: int = 1,
                 device: str = "cpu") -> object:
        
        super(RewardFunction,
              self).__init__()

        self.device = device
        
        # reward network backbone linear layers
        self.backbone = torch.nn.Sequential(torch.nn.Linear(state_size + action_size, hidden_size, bias=True),
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Linear(hidden_size, hidden_size, bias=True),
                                            torch.nn.LeakyReLU())
        
        # reward function output being a squashed (with Tanh) to result in a value between -1 and 1
        self.reward_value = torch.nn.Sequential(torch.nn.Linear(hidden_size, out_size, bias=True),
                                                torch.nn.Tanh())
    
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        # propagate through reward network backbone
        x = self.backbone(x)
        value = self.reward_value(x)

        return value
    
    def estimate_reward(self,
                        state: torch.Tensor,
                        action: torch.Tensor,
                        is_reward_inference: bool = False) -> torch.Tensor:
        
        if len(state.shape) == 1 and len(action.shape) == 1:
            x = torch.cat((state, action))
        elif len(state.shape) == 2 and len(action.shape) == 2:
            x = torch.cat((state, action), dim=1)
        else:
            raise ValueError("Invalid tensor shapes for concatenation.")
        
        if is_reward_inference:
            with torch.no_grad():
                value = self.forward(x=x)
        
        else:
            value = self.forward(x=x)
        
        return value
