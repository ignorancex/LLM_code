import math
import torch
import torch.nn as nn
from models.cond_mlp import SimpleMLPAdaLN


class RectFlowHead(nn.Module):
    def __init__(self, token_embed_dim, decoder_embed_dim, 
                 num_sampling_steps='50',
                 head_width=1024, head_depth=6):
        super(RectFlowHead, self).__init__()
        self.token_embed_dim = token_embed_dim
        self.flow_net = SimpleMLPAdaLN(
            in_channels=self.token_embed_dim,
            model_channels=head_width,
            out_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            num_res_blocks=head_depth
        )
        self.num_sampling_steps = int(num_sampling_steps)
    
    def forward(self, target, z, mask=None):
        # Probability of the target
        x0 = torch.randn_like(target)
        x1 = target
        t = torch.rand(len(x0)).to(x0.device)
        xt = t[:, None] * x1 + (1-t[:, None]) * x0

        velocity = self.flow_net(xt, t, z)

        y = x1 - x0
        rec_loss = (velocity - y).pow(2).mean(dim=-1)
        if mask is not None:
            rec_loss = (rec_loss * mask).sum() / mask.sum()

        return rec_loss

    def sample(self, z, temperature=1.0, cfg=1.0):
        x_next = torch.randn(z.size(0), self.token_embed_dim, device=z.device)
        t_steps = torch.linspace(0, 1, self.num_sampling_steps+1, dtype=torch.float32)

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            time_input = torch.ones(x_cur.size(0), device=z.device, dtype=torch.float32) * t_cur
            d_cur = self.flow_net(x_cur, time_input, z)
            x_next = x_cur + (t_next - t_cur) * d_cur

        return x_next
    

def modulate(x, shift, scale):
    return x * (1 + scale) + shift
