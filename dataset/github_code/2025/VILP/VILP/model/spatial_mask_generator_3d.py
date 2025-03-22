import torch
from torch.nn import Module

class MultiDimMaskGenerator(Module):
    def __init__(self,
                 action_dims,  # Tuple of (action_dim1, action_dim2, action_dim3)
                 max_n_obs_steps=2,
                 fix_obs_steps=True,
                 action_visible=False,
                 device='cuda:1'):
        super().__init__()
        self.action_dims = action_dims
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible
        self.device = device
    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D1, D2, D3 = shape
        assert (D1, D2, D3) == (self.action_dims[0], 
                                self.action_dims[1], 
                                self.action_dims[2])

        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        
        dim_mask = torch.zeros(size=(D1, D2, D3), dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()

        # Assuming action_dims and obs_dims cover different parts of the D1, D2, D3 dimensions entirely
        # Update the action mask to True for action dimensions
        is_action_dim[:self.action_dims[0], :self.action_dims[1], :self.action_dims[2]] = True
        
        # The observation dimensions are the complement of the action dimensions
        is_obs_dim = ~is_action_dim

        # Extend is_action_dim and is_obs_dim to match the input shape (B, T, D1, D2, D3)
        is_action_dim = is_action_dim.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
        is_obs_dim = is_obs_dim.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
        
        # Determine the number of observation steps for each batch
        obs_steps = torch.full((B,), fill_value=self.max_n_obs_steps, device=device)

        
        steps = torch.arange(0, T, device=device).expand(B, T)
        obs_time_mask = (steps.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) < obs_steps.view(-1, 1, 1, 1, 1))
        obs_time_mask = obs_time_mask.expand(-1, -1, D1, D2, D3)

        # Apply observation dimension mask
        # Note: is_obs_dim should be defined similarly to is_action_dim in the adjusted 3D context
        obs_mask = obs_time_mask & is_obs_dim
        
        mask = obs_mask

        return mask