import torch
import torch.nn as nn


class Base:
    def __init__(
        self,
        module: nn.Module,
        activation: str = "relu",
        in_channels_0: int = 3,
        device: str = "cuda",
    ):
        # These could be the number of input/output channels of a convolutional layer or the number of input/output features of a linear layer
        # I opted for in_channels/out_channels for convenience
        self.in_channels = module.weight.shape[1] 
        self.out_channels = module.weight.shape[0]
        self.rows = min(self.out_channels, self.in_channels)
        self.columns = max(self.out_channels, self.in_channels)
        self.in_channels_0 = in_channels_0
        self.device = device
        self.activation = activation
        self.ramanujan_mask = None

    def _ortho_gen(self, rows, columns) -> torch.tensor:
        rand_matrix = torch.randn((max(rows, columns), min(rows, columns)))
        q, _ = torch.linalg.qr(rand_matrix)
        orthogonal_matrix = q[:, :columns]
        return orthogonal_matrix.T if columns > rows else orthogonal_matrix

    def _ortho_generator(self) -> torch.tensor:
        if self.activation == "relu" and self.in_channels != self.in_channels_0:
            rows = self.out_channels // 2
            columns = self.in_channels // 2
            orthogonal_matrix = self._concat(self._ortho_gen(rows, columns))

        else:
            rows = self.out_channels
            columns = self.in_channels
            orthogonal_matrix = self._ortho_gen(rows, columns)

        return orthogonal_matrix.to('cuda')
