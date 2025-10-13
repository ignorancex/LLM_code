from torch import nn
from torch.nn import functional as F


class Linear(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)

    @classmethod
    def code(cls):
        return 'linear'

class Adapter(nn.Module):
    def __init__(self, input_dim: int, dropout=0.5, reduction=2, amplification=2):
        super().__init__()
        hidden_dim, output_dim = input_dim // reduction, input_dim * amplification
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.layer_norm(x) # pre-norm
        return F.normalize(self.mlp(x), dim=-1)

    @classmethod
    def code(cls):
        return 'adapter'

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.proj(x)
        return F.normalize(x, dim=-1)

    @classmethod
    def code(cls):
        return 'projector'

class Phi(nn.Module):
    """
    Textual Inversion Phi network.
    Takes as input the visual features of an image and outputs the pseudo-work embedding.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)

    @classmethod
    def code(cls):
        return 'phi'