import torch


def transform_darcy(x: torch.Tensor) -> torch.Tensor:
    a, u = x[:, 0, :, :], x[:, 1, :, :]
    a = torch.where(a > 7.5, torch.tensor(12.0, device=x.device), torch.tensor(3.0, device=x.device))
    return torch.stack([a, u], dim=1)


class DatasetNormalizer:

    def __init__(self, dataset_name, stats):
        """
        Initialize the normalizer with dataset-specific parameters.
        """
        self.dataset_name = dataset_name
        self.mean = torch.tensor(stats["mean"]).reshape(1, 2, 1, 1)
        self.std = torch.tensor(stats["std"]).reshape(1, 2, 1, 1)
        self._transform = lambda x: x
        if dataset_name == "darcy":
            self._transform = transform_darcy

    def _check_shape(self, x: torch.Tensor):
        # Assuming x has shape (batch_size, channels, height, width)
        assert len(x.shape) == 4, f"Expected 4D tensor, got {len(x.shape)}D"
        assert x.shape[1] == 2, f"Expected 2 channels, got {x.shape[1]}"
        return True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._check_shape(x)
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        x_normalized = (x - self.mean) * (0.5 / self.std)
        return x_normalized

    def denormalize(self, x_normalized: torch.Tensor) -> torch.Tensor:
        self._check_shape(x_normalized)
        self.mean = self.mean.to(x_normalized.device)
        self.std = self.std.to(x_normalized.device)
        x = x_normalized / (0.5 / self.std) + self.mean
        return x

    def transform(self, x: torch.Tensor, denormalize=False) -> torch.Tensor:
        self._check_shape(x)
        if denormalize:
            x = self.denormalize(x)
        return self._transform(x)
