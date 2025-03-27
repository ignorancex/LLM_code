import torch
from torch import nn

from eval.utils import next_multiple
from flash_dropout.layers import DropoutMM


class BasicNet(nn.Module):
    def __init__(
        self,
        sample: torch.Tensor,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        dropout: dict,
    ):
        super().__init__()
        assert num_layers >= 0
        self.output_dim = output_dim

        self.in_proj = DropoutMM(sample[0].numel(), hidden_dim, **dropout)
        self.out_proj = DropoutMM(
            hidden_dim, next_multiple(output_dim, base=128), **dropout
        )

        self.layers = nn.ModuleList(
            [DropoutMM(hidden_dim, hidden_dim, **dropout) for _ in range(num_layers)]
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        x = self.in_proj(x)
        x = self.act(x)

        for layer in self.layers:
            x = layer(x)
            x = self.act(x)

        x = self.out_proj(x)[..., : self.output_dim]
        return x
