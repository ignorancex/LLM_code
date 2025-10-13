import h5py
import torch
from torch.utils.data import Dataset
from glob import glob
import os
import skimage.io as sio
import tqdm
import numpy as np
from einops import rearrange, repeat
import tifffile
from sota_module.dataset.kernel_generator import _get_gaussian_random_kernel2d


"""
The following code is copied from https://github.com/samuro95/GSPnP/blob/master/PnP_restoration/utils/utils_sr.py
Commit id: 20f7293b656da044d1c51efaabd1b440eb412696
"""

# def downsample(x, sf=3):
#     x = torch.complex(x, torch.zeros_like(x))
#
#     y = torch.fft.ifftshift(x)
#     y = torch.fft.fft2(y)
#     y = torch.fft.fftshift(y)
#
#     src_width, src_height = x.shape
#     tar_width, tar_height = src_width // sf, src_height // sf
#
#     y = y[
#         (src_width - tar_width) // 2: -1 * (src_width - tar_width) // 2,
#         (src_height - tar_height) // 2: -1 * (src_height - tar_height) // 2
#         ]
#
#     y = torch.fft.ifftshift(y)
#     y = torch.fft.ifft2(y)
#     y = torch.fft.fftshift(y)
#
#     y = torch.abs(y)
#
#     return y


# def upsample(x, sf=1):
#     y = torch.fft.ifftshift(x)
#     y = torch.fft.fft2(y)
#     y = torch.fft.fftshift(y)
#
#     src_width, src_height = x.shape
#     tar_width, tar_height = src_width * sf, src_height * sf
#
#     y = torch.nn.functional.pad(
#         y, ((tar_width - src_width) // 2, (tar_width - src_width) // 2, (tar_height - src_height) // 2,
#             (tar_height - src_height) // 2)
#     )
#
#     y = torch.fft.ifftshift(y)
#     y = torch.fft.ifft2(y)
#     y = torch.fft.fftshift(y)
#
#     y = torch.real(y)
#
#     return y


def dim_pad_circular(input_, padding, dimension):
    input_ = torch.cat([input_, input_[[slice(None)] * (dimension - 1) +
                                       [slice(0, padding)]]], dim=dimension - 1)
    input_ = torch.cat([input_[[slice(None)] * (dimension - 1) +
                               [slice(-2 * padding, -padding)]], input_], dim=dimension - 1)
    return input_


def pad_circular(input_, padding):
    """
    Arguments
    :param input_: tensor of shape :math:`(N, C_{\text{in}}, H, [W, D]))`
    :param padding: (tuple): m-elem tuple where m is the degree of convolution
    Returns
    :return: tensor of shape :math:`(N, C_{\text{in}}, [D + 2 * padding[0],
                                     H + 2 * padding[1]], W + 2 * padding[2]))`
    """
    offset = 3
    for dimension in range(input_.dim() - offset + 1):
        input_ = dim_pad_circular(input_, padding[dimension], dimension + offset)
    return input_


def upsample(x, sf=3):
    """s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, N x C x W x H
    """
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    """s-fold downsampler
    Keeping the upper-left pixel for each distinct sf x sf patch and discarding the others
    x: tensor image, N x C x W x H
    """
    st = 0
    return x[..., st::sf, st::sf]


def imfilter(x, k):
    """
    x: image, N x c x H x W
    k: kernel, c x 1 x h x w
    """
    x = pad_circular(x, padding=((k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2))
    x = torch.nn.functional.conv2d(x, k, groups=x.shape[1])
    return x


def G(x, k, sf=3):
    """
    x: image, N x c x H x W
    k: kernel, c x 1 x h x w
    sf: scale factor
    center: the first one or the middle one
    Matlab function:
    tmp = imfilter(x,h,'circular');
    y = downsample2(tmp,K);
    """
    if k is not None:
        x = imfilter(x, k)
    if sf >= 2:
        x = downsample(x, sf=sf)

    return x


def Gt(x, k, sf=3):
    """
    x: image, N x c x H x W
    k: kernel, c x 1 x h x w
    sf: scale factor
    center: the first one or the middle one
    Matlab function:
    tmp = upsample2(x,K);
    y = imfilter(tmp,h,'circular');
    """
    if sf >= 2:
        x = upsample(x, sf=sf)
    if k is not None:
        x = imfilter(x, torch.flip(k, [-1, -2]))

    return x


"""
Copy-paste ends here
"""


def addwgn(x: torch.Tensor, input_snr):
    noiseNorm = torch.norm(x.flatten()) * 10 ** (-input_snr / 20)

    noise = torch.randn(x.size())

    noise = noise / torch.norm(noise.flatten()) * noiseNorm

    y = x + noise

    return y, noise


def to_rgb(img):
    img = rearrange(img, '1 c w h -> w h c')
    img[img < 0] = 0
    img[img > 1] = 1
    img = (img * 255).to(torch.uint8).numpy()

    return img


def grad_theta(x, y, theta, sf):

    with torch.inference_mode(mode=False):

        theta = theta.clone().requires_grad_()

        predict = G(x, theta, sf)

        loss_theta = torch.nn.MSELoss(reduction='sum')(y.clone(), predict)

        theta_grad = torch.autograd.grad(
            loss_theta, theta
        )

    return theta_grad[0]


IDX2KERNEL_MAPPING = {
    0: ['Levin09.mat', 0],
    1: ['Levin09.mat', 1],
    2: ['Levin09.mat', 2],
    3: ['Levin09.mat', 3],
    4: ['Levin09.mat', 4],
    5: ['Levin09.mat', 5],
    6: ['Levin09.mat', 6],
    7: ['Levin09.mat', 7],
    9: ['kernels_12.mat', 0],
    10: ['kernels_12.mat', 1],
    11: ['kernels_12.mat', 2],
    12: ['kernels_12.mat', 3],
    13: ['kernels_12.mat', 4],
    14: ['kernels_12.mat', 5],
    15: ['kernels_12.mat', 6],
    16: ['kernels_12.mat', 7],
    17: ['kernels_12.mat', 8],
    18: ['kernels_12.mat', 9],
    19: ['kernels_12.mat', 10],
    20: ['kernels_12.mat', 11],
    21: [_get_gaussian_random_kernel2d([25, 25], [1, 10], torch.float32, torch.device('cpu')), None]
}


def load_kernel_via_idx(
        idx,
        root_path='/opt/dataset/natural_image/'
):

    mat_file, kernel_idx = IDX2KERNEL_MAPPING[idx]

    if idx <= 20:
        if mat_file in ['Levin09.mat']:
            from mat73 import loadmat

            kernel_file = loadmat(os.path.join(root_path, mat_file))
            kernel = kernel_file['kernels'][kernel_idx]

        else:
            from scipy.io import loadmat

            kernel_file = loadmat(os.path.join(root_path, mat_file))
            kernel = kernel_file['kernels'][0][kernel_idx]

        kernel = torch.from_numpy(kernel).to(torch.float32)

    else:

        kernel = mat_file

    return kernel


class NaturalImageDatasetBase(Dataset):

    def get_folder_name(self) -> str:
        pass

    def getitem_helper(self, item):

        cache_folder = os.path.join(self.root_path, self.get_folder_name(),
                                    "cache_id_%s_noise_snr_%s_kernel_idx_%s_down_sampling_factor_%s" % (
                                        self.cache_id, self.noise_snr, self.kernel_idx, self.down_sampling_factor)
                                    )

        target_h5 = os.path.join(cache_folder, "item_%d.h5" % item)

        if (self.cache_id is None) or ((self.cache_id is not None) and (not os.path.exists(target_h5))):

            x = sio.imread(self.file_paths[item])
            x = torch.from_numpy(x)
            x = x / torch.max(x)
            x = x.to(torch.float32)

            if self.kernel_idx is not None:
                kernel = load_kernel_via_idx(self.kernel_idx, self.root_path)
            else:
                kernel = None

            if x.dim() == 3:  # RGB
                x = rearrange(x, 'w h c -> 1 c h w' if x.shape[0] > x.shape[1] else 'w h c -> 1 c w h')

                if self.kernel_idx is not None:
                    kernel = repeat(kernel, 'x y -> c x y', c=3)
                    kernel = rearrange(kernel, 'c x y -> c 1 x y')

            else:  # Gray
                x = rearrange(x, 'w h -> 1 1 h w' if x.shape[0] > x.shape[1] else 'w h -> 1 1 w h')

                if self.kernel_idx is not None:
                    kernel = rearrange(kernel, 'x y -> 1 1 x y')

            y = G(x, kernel, sf=self.down_sampling_factor)
            if self.noise_snr > 0:
                y, _ = addwgn(y, self.noise_snr)

            if self.cache_id is not None:

                if not os.path.exists(cache_folder):
                    os.mkdir(cache_folder)

                with h5py.File(target_h5, 'w') as f:
                    f.create_dataset('x', data=x)
                    f.create_dataset('kernel', data=kernel)
                    f.create_dataset('y', data=y)

                qc_folder = os.path.join(cache_folder, "item_%d" % item)

                if not os.path.exists(qc_folder):
                    os.mkdir(qc_folder)

                tifffile.imwrite(os.path.join(qc_folder, 'x.tiff'), to_rgb(x), imagej=True)
                tifffile.imwrite(os.path.join(qc_folder, 'kernel.tiff'), kernel.numpy(), imagej=True)
                tifffile.imwrite(os.path.join(qc_folder, 'y.tiff'), to_rgb(y), imagej=True)

        else:

            with h5py.File(target_h5, 'r') as f:
                x = f['x'][:]
                kernel = f['kernel'][:]
                y = f['y'][:]

                x, kernel, y = [torch.from_numpy(i) for i in [x, kernel, y]]

        return x, kernel, y

    def __init__(
            self,
            root_path,
            noise_snr=0,
            kernel_idx=None,
            down_sampling_factor=1,
            cache_id=None,
            is_preload=True
    ):

        self.root_path = root_path

        file_paths = glob(os.path.join(root_path, self.get_folder_name(), "*.png"))
        file_paths.sort()

        self.file_paths = file_paths

        self.noise_snr = noise_snr
        self.kernel_idx = kernel_idx
        self.down_sampling_factor = down_sampling_factor

        self.cache_id = cache_id
        self.is_preload = is_preload

        # Start preloading the dataset (if needed).
        self.getitem_cache = []
        if self.is_preload:

            for item in tqdm.tqdm(range(len(self)), desc="Preloading data from %s" % root_path):
                self.getitem_cache.append(self.getitem_helper(item))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        if self.is_preload:
            return self.getitem_cache[item]

        else:
            return self.getitem_helper(item)


class Set12(NaturalImageDatasetBase):

    def get_folder_name(self) -> str:
        return "set12"


class CBSD68(NaturalImageDatasetBase):

    def get_folder_name(self) -> str:
        return "CBSD68"


class NaturalImageDataset(Dataset):

    SUBSET_MAP = {
        'set12_tst': [Set12]
    }

    def __init__(
            self,
            subset,
            root_path,
            noise_snr=0,
            kernel_idx=None,
            down_sampling_factor=1,
            cache_id=None,
            is_preload=True
    ):

        assert subset in self.SUBSET_MAP

        self.dataset_set = []
        self.indexes_map = []
        for dataset_idx, dataset_class in enumerate(self.SUBSET_MAP[subset]):

            dataset = dataset_class(
                root_path=root_path,
                noise_snr=noise_snr,
                kernel_idx=kernel_idx,
                down_sampling_factor=down_sampling_factor,
                cache_id=cache_id,
                is_preload=is_preload
            )

            self.dataset_set.append(dataset)

            for slice_idx in range(len(dataset)):
                self.indexes_map.append([dataset_idx, slice_idx])

    def __len__(self):
        return len(self.indexes_map)

    def __getitem__(self, item):
        dataset_idx, slice_idx = self.indexes_map[item]

        return self.dataset_set[dataset_idx][slice_idx]


if __name__ == '__main__':

    NaturalImageDataset(
        subset='set12_test',
        root_path='/opt/dataset/natural_image',
        is_preload=True,
        noise_snr=50,
        kernel_idx=10,
        down_sampling_factor=2,
        cache_id="nips2022_beta"
    )

    # x_, kernel_, y_ = Set12(
    #     root_path='/opt/dataset/natural_image',
    #     is_preload=True,
    #     noise_snr=50,
    #     kernel_idx=10,
    #     down_sampling_factor=2,
    #     cache_id="nips2022_beta"
    # )[0]
    #
    # # x_, kernel_, y_ = CBSD68(
    # #     root_path='/opt/dataset/natural_image',
    # #     is_preload=False,
    # #     noise_snr=50,
    # #     kernel_idx=10,
    # #     down_sampling_factor=3,
    # #     cache_id="nips2022_beta"
    # # )[0]
    #
    # iter_ = tqdm.tqdm(range(1000))
    #
    # tifffile.imwrite(os.path.join('/opt/experiment/x_gt.tiff'), to_rgb(x_), imagej=True)
    # tifffile.imwrite(os.path.join('/opt/experiment/x_init.tiff'), to_rgb(y_), imagej=True)
    #
    # # x_ = Gt(y_, kernel_, sf=2)
    # x_ = upsample(y_, sf=2)
    # for ii in iter_:
    #     dc = torch.norm(G(x_, kernel_, sf=2) - y_)
    #
    #     grad = Gt(G(x_, kernel_, sf=2) - y_, kernel_, sf=2)
    #     x_ = x_ - 0.1 * grad
    #
    #     iter_.set_description("dc=%f" % dc)
    #
    # tifffile.imwrite(os.path.join('/opt/experiment/x_hat.tiff'), to_rgb(x_), imagej=True)
