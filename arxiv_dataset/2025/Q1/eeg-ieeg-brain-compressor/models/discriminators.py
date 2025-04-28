from typing import List, Tuple

import torch
from torch import nn
from torch.nn.utils import weight_norm


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions: Tuple[Tuple[int, int, int]] = (
            (1024, 256, 1024),
            (2048, 512, 2048),
            (512, 128, 512),
        ),
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(resolution=r) for r in resolutions]
        )

    def forward(self, y: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        fmap_rs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int, int],
        channels: int = 64,
        in_channels: int = 1,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope
        window = torch.hann_window(self.resolution[0])
        self.register_buffer("window", window)
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel_size=(7, 5),
                        stride=(2, 2),
                        padding=(3, 2),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(5, 3),
                        stride=(2, 1),
                        padding=(2, 1),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(5, 3),
                        stride=(2, 2),
                        padding=(2, 1),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=3, stride=(2, 1), padding=1
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=3, stride=(2, 2), padding=1
                    )
                ),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(
        self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        x = self.spectrogram(x.squeeze(1))
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        magnitude_spectrogram = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs()

        return magnitude_spectrogram
