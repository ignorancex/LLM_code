import h5py
import torch
from torch import Tensor
from numpy import random as np_random
from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import Dataset
import os


def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2


def _assert_image_tensor(img: Tensor) -> None:
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")


def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img


def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    k_size_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-k_size_half, k_size_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_random_kernel2d(
    kernel_size: List[int], sigma_val: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    # image_channels should be 3 for RGB

    min_val = sigma_val[0]
    max_val = sigma_val[1]
    sigma1 = np_random.uniform(min_val, max_val)
    sigma2 = np_random.uniform(min_val, sigma1)
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma1).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma2).to(device, dtype=dtype)
    kernel_2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    #
    return kernel_2d


class GenerateGaussianKernel(Dataset):
    @staticmethod
    def load_from_cache(mode, root_path, total_num_kernels, kernel_size, sigma_val):
        cache_path = os.path.join(root_path, 'mode_%s_kernel_%s_sigma_val_%s_total_num_kernels_%d.h5' % (
            mode, str(kernel_size), str(sigma_val), total_num_kernels
        ))

        if os.path.exists(cache_path):
            with h5py.File(cache_path, 'r') as f:
                kernels = f['kernels'][:]

        else:
            kernels = torch.zeros(size=[total_num_kernels, 1] + kernel_size, dtype=torch.float32, device=torch.device('cpu'))
            for i in tqdm(range(total_num_kernels)):
                kernels[i, 0, :, :] = _get_gaussian_random_kernel2d(
                    kernel_size=kernel_size, sigma_val=sigma_val, dtype=torch.float32, device=torch.device('cpu'))

            with h5py.File(cache_path, 'w') as f:
                f.create_dataset(name='kernels', data=kernels)

        return kernels

    def __init__(
        self,
        mode,
        root_path,
        kernel_size,
        sigma_val,
    ):

        super().__init__()

        if mode == 'tra':
            total_num_kernels = 10000
        elif mode in ['val', 'tst']:
            total_num_kernels = 100
        else:
            raise ValueError()

        kernel_size = [kernel_size, kernel_size]
        sigma_val = [1, sigma_val]

        self.kernels = self.load_from_cache(
            mode, root_path, total_num_kernels, kernel_size, sigma_val
        )

    def __len__(self):
        return self.kernels.shape[0]

    def __getitem__(self, item):
        return self.kernels[item],
