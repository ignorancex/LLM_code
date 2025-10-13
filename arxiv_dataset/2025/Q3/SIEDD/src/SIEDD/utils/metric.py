import torch
import numpy as np
from pydantic import BaseModel
import torch.nn.functional as F
from flip_evaluator.flip_python_api import evaluate
from collections.abc import MutableMapping
from typing import Any


def flatten(
    dictionary: MutableMapping[str, MutableMapping | Any], parent_key="", separator="/"
) -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


class CompressionMetrics(BaseModel):
    bpp: float
    total_bits: int
    total_KB: float
    total_pixels: int
    parameters: int


class QualityMetrics(BaseModel):
    psnr: float
    ssim: float
    mse: float


class Metrics(BaseModel):
    metrics: QualityMetrics
    compression_metrics: CompressionMetrics
    quant_metrics: QualityMetrics | None = None
    quant_compression_metrics: CompressionMetrics | None = None
    time: float
    fps: float


def reduce_compression_metrics(metrics_per_frame: list[CompressionMetrics]):
    bpps: list[float] = []
    total_bits: list[int] = []
    total_KBs: list[float] = []
    total_pixels: list[int] = []
    parameters: list[int] = []
    for x in metrics_per_frame:
        bpps.append(x.bpp)
        total_bits.append(x.total_bits)
        total_KBs.append(x.total_KB)
        total_pixels.append(x.total_pixels)
        parameters.append(x.parameters)
    return CompressionMetrics(
        bpp=sum(total_bits) / sum(total_pixels),
        total_bits=sum(total_bits),
        total_KB=sum(total_KBs),
        total_pixels=sum(total_pixels),
        parameters=sum(parameters),
    )


def reduce_quality_metrics(metrics_per_frame: list[QualityMetrics]):
    mses: list[float] = []
    ssims: list[float] = []
    psnrs: list[float] = []
    for x in metrics_per_frame:
        mses.append(x.mse)
        psnrs.append(x.psnr)
        ssims.append(x.ssim)
    return QualityMetrics(
        mse=float(np.mean(mses)),
        psnr=float(np.mean(psnrs)),
        ssim=float(np.mean(ssims)),
    )


def reduce_metrics(metrics_per_frame: list[Metrics]):
    quality_metrics = [frame.metrics for frame in metrics_per_frame]
    compression_metrics = [frame.compression_metrics for frame in metrics_per_frame]
    quant_metrics = [
        frame.quant_metrics
        for frame in metrics_per_frame
        if frame.quant_metrics is not None
    ]
    if len(quant_metrics) == 0:
        quant_metrics = None
    quant_compression_metrics = [
        frame.quant_compression_metrics
        for frame in metrics_per_frame
        if frame.quant_compression_metrics is not None
    ]
    if len(quant_compression_metrics) == 0:
        quant_compression_metrics = None

    time: list[float] = []
    fps: list[float] = []
    for x in metrics_per_frame:
        time.append(x.time)
        fps.append(x.fps)
    return Metrics(
        metrics=reduce_quality_metrics(quality_metrics),
        compression_metrics=reduce_compression_metrics(compression_metrics),
        quant_metrics=reduce_quality_metrics(quant_metrics)
        if quant_metrics is not None
        else None,
        quant_compression_metrics=reduce_compression_metrics(quant_compression_metrics)
        if quant_compression_metrics is not None
        else None,
        time=sum(time),
        fps=float(np.mean(fps)),
    )


class MSEPSNR:
    def __init__(self, max_val=1.0, device=None):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_val_log = 20 * torch.log10(torch.tensor(max_val, device=self.device))

    @torch.no_grad()
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor, batch: bool = False):
        # Calculate MSE
        dims = tuple(range(1, img1.ndim))
        if batch:
            mse_per_img = [((i1 - i2) ** 2).mean() for i1, i2 in zip(img1, img2)]
            mse = torch.tensor(mse_per_img, dtype=img1.dtype, device=img1.device)
        else:
            mse = torch.mean((img1 - img2) ** 2, dim=dims)
        # Calculate PSNR
        psnr = self.max_val_log - 10 * torch.log10(mse)

        return psnr, mse


class SSIM:
    def __init__(self, window_size=11, channel=3, device=None):
        self.window_size = window_size
        self.channel = channel
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.window = self.create_window(window_size, channel)

    def create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [
                    np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                    for x in range(window_size)
                ]
            )
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        size_average: bool = False,
        batch: bool = False,
    ):
        if batch:
            assert img1.ndim == 4
            ssims = [
                self(i1[None], i2[None], True, False) for i1, i2 in zip(img1, img2)
            ]
            return torch.tensor(ssims, dtype=img1.dtype, device=img1.device)
        img2 = img2.to(img1.dtype)
        if img1.ndim == 3:
            (channel, height, width) = img1.size()
        else:
            (_, channel, height, width) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = (
                self.create_window(self.window_size, channel)
                .to(img1.device)
                .type(img1.dtype)
            )
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean((1, 2, 3))


@torch.no_grad()
def FLIP(img1: torch.Tensor, img2: torch.Tensor):
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    if img1.ndim == 4:
        imgs1 = list(img1_np)  # c x h x w
        imgs2 = list(img2_np)
    else:
        imgs1 = [img1]
        imgs2 = [img2]
    flip_maps = []
    flips = []
    for i1, i2 in zip(imgs1, imgs2):
        flip_map, flip, _ = evaluate(i1, i2, "LDR")
        flip_maps.append(flip_map)
        flips.append(flip)
    return flip_maps, flips
