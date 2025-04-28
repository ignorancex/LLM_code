from typing import Optional, Tuple

import torch

from torch import nn


class Detector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DAE(Detector):
    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        return x + self.epsilon * noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(self.add_noise(x))
        return torch.norm((x - self.decode(z)).view(x.shape[0], -1), dim=1)  # type: ignore

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        loss = torch.norm((x - self.decode(z)).view(x.shape[0], -1), dim=1)
        return 1 - torch.exp(-loss)  # type: ignore


class DeepSVDD(Detector):
    def __init__(self):
        super().__init__()
        self.c: Optional[torch.Tensor] = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return torch.norm(z - self.c, dim=1)  # type: ignore

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        if self.c is None:
            raise RuntimeError("The center is not initialized.")

        return 1 - torch.exp(-self.forward(x))  # type: ignore

    def set_center(self, c: torch.Tensor) -> None:
        self.c = c


class PreTrainableSVDD(DeepSVDD):
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if self.c is None:
            return torch.norm((x - self.decode(z)).view(x.shape[0], -1), dim=1)  # type: ignore
        else:
            return torch.norm(z - self.c, dim=1)  # type: ignore


class DenseDAE(DAE):
    def __init__(self, n_in: int, n_latent: int, n_h: int, epsilon: float):
        super().__init__(epsilon=epsilon)
        self.n_in = n_in
        self.n_latent = n_latent
        self.n_h = n_h
        self.encoder, self.decoder = generate_dense_layers(n_in=n_in, n_latent=n_latent, n_h=n_h, use_bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # type: ignore

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)  # type: ignore


class DenseSVDD(PreTrainableSVDD):
    def __init__(self, n_in: int, n_latent: int, n_h: int):
        super().__init__()
        self.n_in = n_in
        self.n_latent = n_latent
        self.n_h = n_h
        self.encoder, self.decoder = generate_dense_layers(n_in=n_in, n_latent=n_latent, n_h=n_h, use_bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # type: ignore

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)  # type: ignore


class MLP(Detector):
    def __init__(self, n_in: int, n_h: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.network(x))

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class ConvolutionalDAE(DAE):
    def __init__(self, epsilon: float):
        super().__init__(epsilon=epsilon)
        self.encoder, self.decoder = generate_convolutional_layers(use_affine=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # type: ignore

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)  # type: ignore


class ConvolutionalSVDD(PreTrainableSVDD):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = generate_convolutional_layers(use_affine=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # type: ignore

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)  # type: ignore


class CNN(Detector):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),  # 32 - 5 + 1 = 28
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2),  # 28 / 2 = 14
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),  # 14 - 5 + 1 = 10
            nn.BatchNorm2d(20),
            nn.Dropout2d(),
            nn.MaxPool2d(2),  # 10 / 2 = 5
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5 * 5 * 20, 50),  # 5 * 5 * 20 = 500
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.network(x))

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def generate_dense_layers(n_in: int, n_latent: int, n_h: int, use_bias: bool) -> Tuple[nn.Module, nn.Module]:
    encoder = nn.Sequential(
        nn.Linear(n_in, n_h, bias=use_bias),
        nn.ReLU(),
        nn.Linear(n_h, n_h, bias=use_bias),
        nn.ReLU(),
        nn.Linear(n_h, n_h, bias=use_bias),
        nn.ReLU(),
        nn.Linear(n_h, n_latent, bias=use_bias),
    )

    decoder = nn.Sequential(
        nn.Linear(n_latent, n_h, bias=use_bias),
        nn.ReLU(),
        nn.Linear(n_h, n_h, bias=use_bias),
        nn.ReLU(),
        nn.Linear(n_h, n_h, bias=use_bias),
        nn.ReLU(),
        nn.Linear(n_h, n_in, bias=use_bias),
    )

    return encoder, decoder


def generate_convolutional_layers(use_affine: bool) -> Tuple[nn.Module, nn.Module]:
    """
    The following CNN architectures are based on https://github.com/lukasruff/Deep-SVDD-PyTorch/
    """

    # Encoder
    encoder = nn.Sequential(
        nn.Conv2d(3, 32, 5, bias=False, padding=2),
        nn.BatchNorm2d(32, affine=use_affine),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 5, bias=False, padding=2),
        nn.BatchNorm2d(64, affine=use_affine),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 5, bias=False, padding=2),
        nn.BatchNorm2d(128, affine=use_affine),
        nn.LeakyReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 128, bias=False),
        nn.BatchNorm1d(128, affine=use_affine),
        nn.LeakyReLU(),
    )

    # Decoder
    decoder = nn.Sequential(
        nn.Unflatten(1, (8, 4, 4)),
        nn.ConvTranspose2d(8, 128, 5, bias=False, padding=2),
        nn.BatchNorm2d(128, affine=use_affine),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2),
        nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2),
        nn.BatchNorm2d(64, affine=use_affine),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2),
        nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2),
        nn.BatchNorm2d(32, affine=use_affine),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2),
        nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2),
        nn.Sigmoid(),
    )

    return encoder, decoder


def load_dense_model(algorithm: str, n_in: int, n_latent: int, n_h: int, epsilon: float) -> Detector:
    if algorithm in ["AE", "ABC", "PUAE"]:
        return DenseDAE(n_in=n_in, n_latent=n_latent, n_h=n_h, epsilon=epsilon)
    elif algorithm in ["DeepSVDD", "DeepSAD", "PUSVDD", "LOE", "SOEL"]:
        return DenseSVDD(n_in=n_in, n_latent=n_latent, n_h=n_h)
    else:
        return MLP(n_in=n_in, n_h=n_h)


def load_convolutional_model(algorithm: str, epsilon: float) -> Detector:
    if algorithm in ["AE", "ABC", "PUAE"]:
        return ConvolutionalDAE(epsilon=epsilon)
    elif algorithm in ["DeepSVDD", "DeepSAD", "PUSVDD", "LOE", "SOEL"]:
        return ConvolutionalSVDD()
    else:
        return CNN()
