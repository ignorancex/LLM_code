"""
For each image recovery task, we define an explicit forward model.

NOTE - for most forward models, we treat each number in the image
as a "pixel", so it has H x W x Ch total pixels.
"""
import math
import sys
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DEVICE = 'cuda:0'


class ForwardModel(ABC):
    viewable = False

    @abstractmethod
    def __call__(self, img):
        pass

    @abstractmethod
    def __str__(self):
        pass


class NoOp(ForwardModel):
    viewable = True

    def __init__(self, **fm_kwargs):
        pass

    def __call__(self, img):
        return img

    def __str__(self):
        return 'NoOp'


def get_random_mask(img_shape: Tuple[int, int, int], fraction_kept: float = None, n_kept: int = None, device=None):
    """
    For image of shape CHW, returns random boolean
    mask of the same shape.
    """
    if n_kept is None and fraction_kept is None:
        raise ValueError()

    n_pixels = np.prod(img_shape)
    if fraction_kept:
        n_kept = int(fraction_kept * n_pixels)

    mask = torch.zeros(img_shape)
    if device:
        mask = mask.to(device)

    random_coords = torch.randperm(int(n_pixels))
    for i in range(n_kept):
        random_coord = np.unravel_index(random_coords[i], img_shape)
        mask[random_coord] = 1
    return mask


class GaussianCompressiveSensing(ForwardModel):
    """
    Wide Gaussian measurement matrix
    img_shape - 3 x H x W
    n_measure - number of rows for the measurement matrix
    """
    def __init__(self, img_shape, n_measure, device=DEFAULT_DEVICE):
        self.n_measure = n_measure
        self.img_shape = img_shape
        self.device = device
        self.A = torch.randn(np.prod(img_shape), n_measure, device=device)
        self.A /= math.sqrt(n_measure)

    def __call__(self, img):
        return img.view(img.shape[0], -1) @ self.A

    def __str__(self):
        return f'GaussianCompressiveSensing.n_measure={self.n_measure}'


# from IAGAN repo:
def rand_mask(size, thresh):
    half_size = np.floor(size / 2).astype('int32')
    idxX = np.mod(np.floor(np.abs(np.random.randn(half_size, half_size)) * thresh), half_size)
    idxY = np.mod(np.floor(np.abs(np.random.randn(half_size, half_size)) * thresh), half_size)
    mask_t = torch.zeros(size, size)
    mask_t[idxY, idxX] = 1
    # Duplicate
    dupIdx = [i for i in range(half_size - 1, -1, -1)]
    mask_t[:half_size, half_size:] = mask_t[:half_size, dupIdx]  # flip x
    mask_t[half_size:, :half_size] = mask_t[dupIdx, :half_size]  # flip y
    x, y = np.meshgrid(dupIdx, dupIdx)
    mask_t[half_size:, half_size:] = mask_t[y, x]  # flip x and y
    mask = np.array(mask_t)

    ratio = np.sum(mask == 1) / mask.size
    mask_t = mask_t.unsqueeze(0).unsqueeze(0).unsqueeze(4)
    mask_t = torch.cat((mask_t, mask_t), 4)
    return mask_t, ratio


def compress_FFT(x, mask):
    batch_size = x.shape[0]
    r = x[:, 0:1, :, :] # 1 , c, h , w
    g = x[:, 1:2, :, :]
    b = x[:, 2:3, :, :]
    R = torch.rfft(r, signal_ndim=2, normalized=True, onesided=False).float()
    breakpoint()
    R = torch.rfft(r, signal_ndim=2, normalized=True, onesided=False).float().view(batch_size, -1)
    G = torch.rfft(g, signal_ndim=2, normalized=True, onesided=False).float().view(batch_size, -1)
    B = torch.rfft(b, signal_ndim=2, normalized=True, onesided=False).float().view(batch_size, -1)
    mask = mask.view(-1)
    R_masked = R[:, mask == 1]
    G_masked = G[:, mask == 1]
    B_masked = B[:, mask == 1]
    X_masked = torch.cat((R_masked.unsqueeze(1), G_masked.unsqueeze(1), B_masked.unsqueeze(1)), dim=1)
    breakpoint()
    return X_masked


def compress_FFT_t(X, mask):
    shape = mask.shape
    mask = mask.view(-1)
    R = torch.zeros_like(mask)
    R[mask == 1] = X[:, 0]
    R = R.reshape(shape)
    G = torch.zeros_like(mask)
    G[mask == 1] = X[:, 1]
    G = G.reshape(shape)
    B = torch.zeros_like(mask)
    B[mask == 1] = X[:, 2]
    B = B.reshape(shape)
    r = torch.irfft(R, signal_ndim=2, normalized=True, onesided=False)
    g = torch.irfft(G, signal_ndim=2, normalized=True, onesided=False)
    b = torch.irfft(B, signal_ndim=2, normalized=True, onesided=False)
    x = torch.cat((r, g, b), dim=1)
    return x


"""
"""


class FFTCompressiveSensing(ForwardModel):
    def __init__(self, img_shape, n_measure, device=DEFAULT_DEVICE):
        self.n_measure = n_measure
        self.mask = get_random_mask(img_shape, n_kept=self.n_measure).to(device)

    def __call__(self, img):
        raise NotImplementedError('TODO - add batch dimension')
        print(img.shape)
        img = torch.rfft(img, 2, onesided=False)
        print(img.shape)
        img = self.mask[..., None] * img
        print(img.shape)
        img = torch.irfft(img, 2, onesided=False)
        print(img.shape)
        return img

    def __str__(self):
        return f'FFTCompressiveSensing.n_measure={self.n_measure}'


class InpaintingScatter(ForwardModel):
    """
    Mask random pixels
    """
    viewable = True

    def __init__(self, img_shape, fraction_kept, device=DEFAULT_DEVICE):
        """
        img_shape - 3 x H x W
        fraction_kept - number in [0, 1], what portion of pixels to retain
        """
        assert fraction_kept <= 1 and fraction_kept >= 0
        self.fraction_kept = fraction_kept
        self.A = get_random_mask(img_shape, fraction_kept=self.fraction_kept).to(device)

    def __call__(self, img):
        return self.A[None, ...] * img

    def __str__(self):
        return f'InpaintingScatter.fraction_kept={self.fraction_kept}'


class SuperResolution(ForwardModel):
    viewable = True

    def __init__(self, scale_factor, mode='linear', align_corners=True, **kwargs):
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, img):
        res = F.interpolate(img, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return res

    def __str__(self):
        return (f'SuperResolution.scale_factor={self.scale_factor}' f'.mode={self.mode}')


def get_forward_model(fm_name, **fm_kwargs):
    return getattr(sys.modules[__name__], fm_name)(**fm_kwargs)


if __name__ == '__main__':
    from PIL import Image

    from utils import load_target_image
    img = load_target_image('./images/val_celeba128_cropped20/192571.pt', 128)
    img = img.unsqueeze(0)  # batch dim
    mask, ratio = rand_mask(128, 128 * 0.5 / 2)
    print('before', img.shape, mask.shape, ratio, 'min/max:', img.min(), img.max())
    masked_img = compress_FFT(img, mask)
    print('during', masked_img.shape, masked_img.min(), masked_img.max())
    rec_img = compress_FFT_t(masked_img, mask)
    print('after', rec_img.shape, rec_img.min(), rec_img.max())

    def to_img(x):
        return (x * 255).to(torch.uint8).squeeze().numpy().transpose([1, 2, 0])

    Image.fromarray(to_img(rec_img)).save('test_fft.img.png')
    Image.fromarray(to_img(mask.squeeze()[..., 0])).save('test_fft.mask0.png')
    Image.fromarray(to_img(mask.squeeze()[..., 1])).save('test_fft.mask1.png')
