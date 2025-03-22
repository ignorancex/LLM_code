from typing import Tuple
import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    """Create 1-D gauss kernel

    Parameters
    ----------
    size: int
        The size of gauss kernel.
    sigma: float
        Sigma of normal distribution.

    Returns
    -------
    torch.Tensor
        1-D kernel, of shape=(1, 1, size).
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(inputs: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """Blur input with 1-D kernel (valid padding)

    Parameters
    ----------
    inputs: torch.Tensor
        A batch of tensors to be blured, of shape=(N,C,H,W).
    window: torch.Tensor
        1-D gauss kernel, of shape=(1, 1, size).

    Returns
    -------
        torch.Tensor: blured tensors
    """
    c = inputs.shape[1]
    out = F.conv2d(input, win, groups=c)
    out = F.conv2d(out, win.transpose(2, 3), groups=c)
    return out


def stat_gaussian_filter(inputs: torch.Tensor, win: torch.Tensor):
    """Blur input with 1-D kernel, applying `same` padding.

    Parameters
    ----------
    inputs: torch.Tensor
        A batch of tensors to be blured, of shape=(N,C,H,W).
    win: torch.Tensor
        1-D gauss kernel, of shape=(1, 1, size).

    Returns
    -------
    torch.Tensor
        Blured tensors, of shape=(N,C,H,W).
    """
    c = inputs.shape[1]
    k = win.shape[-1]
    inputs = F.pad(inputs, (k // 2, k // 2), value=inputs.mean().item())
    out = F.conv2d(inputs, win, groups=c)
    out = F.pad(out, (0, 0, k // 2, k // 2), value=inputs.mean().item())
    out = F.conv2d(out, win.transpose(2, 3), groups=c)
    return out


def _ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float,
    win: torch.Tensor,
    reduction: bool = True,
    k: Tuple[float, float] = (0.01, 0.03),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate statistical ssim map between inputs.

    Parameters
    ----------
    x: torch.Tensor
        Images, of shape=(N,C,H,W).
    y: torch.Tensor
        Images, of shape=(N,C,H,W).
    data_range: float
        Value range of input images. Usually 1.0 or 255.
    win: torch.Tensor
        1-D gauss kernel, of shape=(1, 1, size).
    reduction: bool
        If reduction=True, ssim of all images will be averaged as a scalar.
    k: list[float]
        Cut-off values for fraction numerical stability.

    Returns
    -------
    torch.Tensor
        Ssim results, of shape=(N,C,H,W).
    torch.Tensor
        Contrast time structure fraction results, of shape=(N,C,H,W).
    """
    k1, k2 = k
    compensation = 1.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    win = win.to(x.device, dtype=x.dtype)

    mu1 = gaussian_filter(x, win)
    mu2 = gaussian_filter(y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(x * x, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(y * y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(x * y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def _stat_ssim(
    x, y, data_range, win, reduction=True, k=(0.01, 0.03)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate statistical Ssim map between inputs.

    Parameters
    ----------
    x: torch.Tensor
        Images, of shape=(N,C,H,W).
    y: torch.Tensor
        Images, of shape=(N,C,H,W).
    data_range: float
        Value range of input images. Usually 1.0 or 255.
    win: torch.Tensor
        1-D gauss kernel, of shape=(1, 1, size).
    reduction: bool
        If reduction=True, ssim of all images will be averaged as a scalar.
    k: list[float]
        Cut-off values for fraction numerical stability.

    Returns
    -------
    torch.Tensor
        Statistica Ssim results, of shape=(N,C,H,W).
    torch.Tensor
        Contrast time structure fraction results, of shape=(N,C,H,W).
    """
    k1, k2 = k
    compensation = 1.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    win = win.to(x.device, dtype=x.dtype)

    mu1 = stat_gaussian_filter(x, win)
    mu2 = stat_gaussian_filter(y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    x_ = x - mu1
    y_ = y - mu2

    sigma1_sq = compensation * (stat_gaussian_filter(x_ * x_, win))
    sigma2_sq = compensation * (stat_gaussian_filter(y_ * y_, win))
    sigma12 = compensation * (stat_gaussian_filter(x_ * y_, win))

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 255.0,
    reduction: bool = True,
    win_size: int = 11,
    win_sigma: int = 3,
    win: torch.Tensor = None,
    k: Tuple[int, int] = (1e-13, 1e-13),
    nonnegative_ssim: bool = False,
) -> torch.Tensor:
    """Interface for Structural Similarity function.

    Parameters
    ----------
    x: torch.Tensor
        Images, of shape=(N,C,H,W).
    y: torch.Tensor
        Images, of shape=(N,C,H,W).
    data_range: float
        Value range of input images. Usually 1.0 or 255.
    reduction: bool
        If reduction=True, ssim of all images will be averaged as a scalar.
    win_size: int
        The size of the gaussian kernel.
    win_sigma: float
        Standard deviation in pixels units of the gaussian kernel.
    win: torch.Tensor
        1-D gauss kernel, of shape=(1, 1, size).
    k: list[float]
        Cut-off values for fraction numerical stability.
    nonnegative_ssim: bool
        Wether to force the ssim response to be nonnegative with relu function.

    Returns
    -------
    torch.Tensor
        Ssim results, of shape=(N,C,H,W).
    """

    if len(x.shape) != 4:
        raise ValueError("Input images should be 4-d tensors.")

    if not x.type() == y.type():
        raise ValueError("Input images should have the same dtype.")

    if not x.shape == y.shape:
        raise ValueError("Input images should have the same shape.")

    if win is not None:
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(x.shape[1], 1, 1, 1)

    ssim_per_channel, _ = _ssim(
        x, y, data_range=data_range, win=win, reduction=False, k=k
    )
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if reduction == "mean":
        return ssim_per_channel.mean()
    elif reduction == "none":
        return ssim_per_channel.mean(1)


def ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 255.0,
    reduction: bool = True,
    win_size: int = 11,
    win_sigma: int = 3,
    win: torch.Tensor = None,
    weights: list[float] = None,
    k: Tuple[int, int] = (1e-13, 1e-13),
) -> torch.Tensor:
    """Interface for Multiscale Structural Similarity function.

    Parameters
    ----------
    x: torch.Tensor
        Images, of shape=(N,C,H,W).
    y: torch.Tensor
        Images, of shape=(N,C,H,W).
    data_range: float
        Value range of input images. Usually 1.0 or 255.
    reduction: bool
        If reduction=True, ssim of all images will be averaged as a scalar.
    win_size: int
        The size of the gaussian kernel.
    win_sigma: float
        Standard deviation in pixels units of the gaussian kernel.
    win: torch.Tensor
        1-D gauss kernel, of shape=(1, 1, size).
    weights: list[float]
        Weights for different levels in the multiscale computation.
    k: list[float]
        Cut-off values for fraction numerical stability.

    Returns
    -------
    torch.Tensor
        Ssim results, of shape=(N,C,H,W).
    """
    if len(x.shape) != 4:
        raise ValueError("Input images should be 4-d tensors.")

    if not x.type() == y.type():
        raise ValueError("Input images should have the same dtype.")

    if not x.shape == y.shape:
        raise ValueError("Input images should have the same dimensions.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(x.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2**4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % (
        (win_size - 1) * (2**4)
    )

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(x.device, dtype=x.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(x.shape[1], 1, 1, 1)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(
            x, y, win=win, data_range=data_range, reduction=False, k=k
        )

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = (x.shape[2] % 2, x.shape[3] % 2)
            x = F.avg_pool2d(x, kernel_size=2, padding=padding)
            y = F.avg_pool2d(y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if reduction == "mean":
        return ms_ssim_val.mean()
    elif reduction == "none":
        return ms_ssim_val.mean(1)


def stat_ssim(
    x,
    y,
    data_range=255,
    reduction=True,
    win_size=11,
    win_sigma=3,
    win=None,
    k=(1e-13, 1e-13),
    nonnegative_ssim=False,
) -> torch.Tensor:
    """Interface for Statistical Structural Similarity function.

    Parameters
    ----------
    x: torch.Tensor
        Images, of shape=(N,C,H,W).
    y: torch.Tensor
        Images, of shape=(N,C,H,W).
    data_range: float
        Value range of input images. Usually 1.0 or 255.
    reduction: bool
        If reduction=True, ssim of all images will be averaged as a scalar.
    win_size: int
        The size of the gaussian kernel.
    win_sigma: float
        Standard deviation in pixels units of the gaussian kernel.
    win: torch.Tensor
        1-D gauss kernel, of shape=(1, 1, size).
    k: list[float]
        Cut-off values for fraction numerical stability.
    nonnegative_ssim: bool
        Wether to force the ssim response to be nonnegative with relu function.

    Returns
    -------
    torch.Tensor
        Stat-Ssim results, of shape=(N,C,H,W).
    """

    if len(x.shape) != 4:
        raise ValueError("Input images should be 4-d tensors.")

    if not x.type() == y.type():
        raise ValueError("Input images should have the same dtype.")

    if not x.shape == y.shape:
        raise ValueError("Input images should have the same shape.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(x.shape[1], 1, 1, 1)

        if data_range == 1.0:
            # rescale inputs in unit range
            # no effect on previously rescaled data
            xmax = x.flatten(1, -1).max(-1).values.reshape([-1, 1, 1, 1])
            ymax = y.flatten(1, -1).max(-1).values.reshape([-1, 1, 1, 1])
            maxes = torch.max(xmax, ymax)
            xmin = x.flatten(1, -1).min(-1).values.reshape([-1, 1, 1, 1])
            ymin = y.flatten(1, -1).min(-1).values.reshape([-1, 1, 1, 1])
            mins = torch.min(xmin, ymin)
            x = (x - mins) / (maxes - mins)
            y = (y - mins) / (maxes - mins)

    ssim_per_channel, _ = _stat_ssim(
        x, y, data_range=data_range, win=win, reduction=False, k=k
    )
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if reduction == "mean":
        return ssim_per_channel.mean()
    elif reduction == "none":
        return ssim_per_channel.mean(1)


class SSIM(torch.nn.Module):
    """Strctural Similarity class."""

    def __init__(
        self,
        data_range: float = 255,
        reduction: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        k: list[float] = [0.01, 0.03],
        nonnegative_ssim: bool = False,
    ):
        """
        Parameters
        ----------
        data_range: float
            Value range of input images. Usually 1.0 or 255.
        reduction: bool
            If reduction=True, ssim of all images will be averaged as a scalar.
        win_size: int
            The size of the gaussian kernel.
        win_sigma: float
            Standard deviation in pixels units of the gaussian kernel.
        channel: int
            Number of input channels. Defaults to 3.
        k: list[float]
            Cut-off values for fraction numerical stability.
        nonnegative_ssim: bool
            Wether to force the ssim response to be nonnegative with relu function.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.reduction = reduction
        self.data_range = data_range
        self.k = k
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the ssim between two images.

        Parameters
        ----------
        x: torch.Tensor
            Images, of shape=(N,C,H,W).
        y: torch.Tensor
            Images, of shape=(N,C,H,W).

        Returns
        -------
        torch.Tensor
            Ssim results, of shape=(N,C,H,W).
        """
        return ssim(
            x,
            y,
            data_range=self.data_range,
            reduction=self.reduction,
            win=self.win,
            k=self.k,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    """Multiscale Strctural Similarity class."""

    def __init__(
        self,
        data_range: float = 255,
        reduction: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        weights: list[float] = None,
        k: list[float] = [0.01, 0.03],
    ):
        """
        Parameters
        ----------
        data_range: float
            Value range of input images. Usually 1.0 or 255.
        reduction: bool
            If reduction=True, ssim of all images will be averaged as a scalar.
        win_size: int
            The size of the gaussian kernel.
        win_sigma: float
            Standard deviation in pixels units of the gaussian kernel.
        channel: int
            Number of input channels. Defaults to 3.
        weights: list[float]
            Weights for different levels in the multiscale computation.
        k: list[float]
            Cut-off values for fraction numerical stability.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.reduction = reduction
        self.data_range = data_range
        self.weights = weights
        self.k = k

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the multi scale ssim between two images.

        Parameters
        ----------
        x: torch.Tensor
            Images, of shape=(N,C,H,W).
        y: torch.Tensor
            Images, of shape=(N,C,H,W).

        Returns
        -------
        torch.Tensor
            Multi scale Ssim results, of shape=(N,C,H,W).
        """
        return ms_ssim(
            x,
            y,
            data_range=self.data_range,
            reduction=self.reduction,
            win=self.win,
            weights=self.weights,
            k=self.k,
        )


class STAT_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        reduction: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        k: list[float] = [0.01, 0.03],
        nonnegative_ssim: bool = False,
    ):
        """
        Parameters
        ----------
        data_range: float
            Value range of input images. Usually 1.0 or 255.
        reduction: bool
            If reduction=True, ssim of all images will be averaged as a scalar.
        win_size: int
            The size of the gaussian kernel.
        win_sigma: float
            Standard deviation in pixels units of the gaussian kernel.
        channel: int
            Number of input channels. Defaults to 3.
        k: list[float]
            Cut-off values for fraction numerical stability.
        nonnegative_ssim: bool
            Wether to force the ssim response to be nonnegative with relu function.
        """

        super(STAT_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.reduction = reduction
        self.data_range = data_range
        self.k = k
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the statistical ssim between two images.

        Parameters
        ----------
        x: torch.Tensor
            Images, of shape=(N,C,H,W).
        y: torch.Tensor
            Images, of shape=(N,C,H,W).

        Returns
        -------
        torch.Tensor
            Ssim results, of shape=(N,C,H,W).
        """
        return stat_ssim(
            x,
            y,
            data_range=self.data_range,
            reduction=self.reduction,
            win=self.win,
            k=self.k,
            nonnegative_ssim=self.nonnegative_ssim,
        )
