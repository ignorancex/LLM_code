import torch
import numpy as np
from torch.utils.data import Dataset
import tqdm
from .utility import addwgn
from sigpy.mri.app import EspiritCalib
from sigpy import Device
#import cupy
import sigpy
from sota_module.utility import check_and_mkdir
import os
import pickle


def ftran(y, smps, mask):
    """
    compute adjoint of fast MRI, x = smps^H F^H mask^H x

    :param y: under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: zero-filled image
    """
    #print(f"[pmri - ftran] y.shape: {y.shape} / y.dtype: {y.dtype}")
    #print(f"[pmri - ftran] smps.shape: {smps.shape} / smps.dtype: {smps.dtype}")
    #print(f"[pmri - ftran] mask.shape: {mask.shape} / mask.dtype: {mask.dtype}")

    # mask^H
    y = y * mask.unsqueeze(1)
    if (smps.shape[1] == 12):
        # F^H
        x = torch.fft.ifft2(y)
    else:
        # F^H
        y = torch.fft.ifftshift(y, [-2, -1])
        x = torch.fft.ifft2(y, norm='ortho')
        x = torch.fft.fftshift(x, [-2, -1])
    '''
    # F^H
    y = torch.fft.ifftshift(y, [-2, -1])
    x = torch.fft.ifft2(y, norm='ortho')
    x = torch.fft.fftshift(x, [-2, -1])
    '''
    # smps^H
    x = x * torch.conj(smps)
    x = x.sum(1)

    #print(f"[pmri - ftran] x.shape: {x.shape} / x.dtype: {x.dtype}")

    return x


def fmult(x, smps, mask):
    """
    compute forward of fast MRI, y = mask F smps x

    :param x: groundtruth or estimated image, shape: batch, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: undersampled measurement
    """
    #print(f"[pmri - fmult] x.shape: {x.shape} / x.dtype: {x.dtype}")
    #print(f"[pmri - fmult] smps.shape: {smps.shape} / smps.dtype: {smps.dtype}")
    #print(f"[pmri - fmult] mask.shape: {mask.shape} / mask.dtype: {mask.dtype}")
    if len(x.shape) != 3:
        x = x.permute([0, 2, 3, 1]).contiguous()
        x = torch.view_as_complex(x)

    # smps
    x = x.unsqueeze(1)
    y = x * smps

    if (smps.shape[1] == 12):
        # F
        y = torch.fft.fft2(y)
    else:
        # F
        y = torch.fft.ifftshift(y, [-2, -1])
        y = torch.fft.fft2(y, norm='ortho')
        y = torch.fft.fftshift(y, [-2, -1])

    '''
    # F
    y = torch.fft.ifftshift(y, [-2, -1])
    y = torch.fft.fft2(y, norm='ortho')
    y = torch.fft.fftshift(y, [-2, -1])
    '''

    # mask
    mask = mask.unsqueeze(1)
    y = y * mask

    #print(f"[pmri] y.shape: {y.shape} / y.dtype: {y.dtype}")
    return y


def uniformly_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, randomly_return: bool = True):

    ny = img_size[-1]

    ACS_START_INDEX = (ny // 2) - (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)
    ACS_END_INDEX = (ny // 2) + (int(ny * acs_percentage * (2 / acceleration_rate)) // 2)

    if ny % 2 == 0:
        ACS_END_INDEX -= 1

    mask = np.zeros(shape=(acceleration_rate,) + img_size, dtype=np.float32)
    mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1

    for i in range(ny):
        for j in range(acceleration_rate):
            if i % acceleration_rate == j:
                mask[j, ..., i] = 1

    if randomly_return:
        mask = mask[np.random.randint(0, acceleration_rate)]
    else:
        mask = mask[0]

    mask = torch.from_numpy(mask)
    # mask = torch.fft.fftshift(mask)

    return mask


def compute_y_center_low_k_hamming(y, size):

    assert size % 2 == 0

    n_batch, n_coil, n_x, n_y = y.shape

    center_x = n_x // 2
    center_y = n_y // 2

    mask = torch.zeros(size=(n_x, n_y), dtype=torch.float32, device=y.device)

    window_x = torch.hamming_window(size, device=y.device)
    window_y = torch.hamming_window(size, device=y.device)

    window_2d = torch.sqrt(torch.outer(window_x, window_y))

    mask[center_x - (size // 2): center_x + (size // 2), center_y - (size // 2): center_y + (size // 2)] = window_2d

    mask = torch.fft.fftshift(mask)

    return torch.fft.ifft2(y * mask)


def divided_by_rss(smps):
    return smps / (torch.sum(torch.abs(smps) ** 2, 1, keepdim=True).sqrt() + 1e-10)


_mask_fn = {
    'uniformly_cartesian': uniformly_cartesian_mask
}


class ParallelMRI(Dataset):

    def __init__(
            self,
            groundtruth,
            acceleration_rate,
            noise_snr,
            num_of_coil,
            compute_smps: str = 'groundtruth',
            compute_smps_low_k_size: int = 20,
            mask_pattern: str = 'uniformly_cartesian',
            cache_id=None
    ):
        """

        :param acceleration_rate:
        :param noise_snr:
        :param num_of_coil: -1 denotes not simulate coils, 0/1 denotes single-coil (all 1).
        :param mask_pattern:
        """

        self.acceleration_rate = acceleration_rate
        self.noise_snr = noise_snr
        self.mask_pattern = mask_pattern
        self.num_of_coil = num_of_coil
        self.compute_smps = compute_smps
        self.compute_smps_low_k_size = compute_smps_low_k_size

        assert self.mask_pattern in ['uniformly_cartesian']
        assert self.compute_smps in ['groundtruth', 'low_k', 'esp']

        if self.num_of_coil == -1:
            self.x, self.smps = groundtruth['x'].to(torch.complex64), groundtruth['smps'].to(torch.complex64)

        elif self.num_of_coil > 1:
            self.x = groundtruth['x'].to(torch.complex64)

            n_batch, n_x, n_y = self.x.shape
            self.smps = sigpy.mri.birdcage_maps((self.num_of_coil, n_x, n_y), dtype=np.complex64)
            self.smps = torch.from_numpy(self.smps)
            self.smps = torch.unsqueeze(self.smps, 0).expand([n_batch, -1, -1, -1])

        else:
            raise NotImplementedError()

        self.num_data, self.width, self.height = self.x.shape

        self.mask = torch.zeros((self.num_data, self.width, self.height), dtype=torch.float32)
        for i in tqdm.tqdm(range(self.num_data), desc='generating sampling mask'):
            self.mask[i, :, :] = _mask_fn[self.mask_pattern]((self.width, self.height), self.acceleration_rate)

        if cache_id is not None:
            root_path = '/opt/dataset/cache_deq_cal/'
            check_and_mkdir(root_path)

            file_name = root_path + '%s_MRI_acceleration_rate_%d_noise_snr%d_num_of_coil%d_compute_smps%s_low_k_size%d_mask%s.pl' % (
                cache_id, acceleration_rate, noise_snr, num_of_coil, compute_smps, compute_smps_low_k_size, mask_pattern)

            if not os.path.exists(file_name):
                print("Cannot find cached data in disk, starting generating and saving.")
                self.cache_data = self.caching_data()

                with open(file_name, 'wb') as f:
                    pickle.dump(self.cache_data, f)

            else:
                print("Found cached data in disk, loading it.")
                with open(file_name, 'rb') as f:
                    self.cache_data = pickle.load(f)

        else:
            print("Not to use cached data, noted that it will cause different results for different running.")
            self.cache_data = self.caching_data()

    def caching_data(self):
        l = []

        for item in tqdm.tqdm(range(len(self)), desc='caching data'):
            l.append(self.__getitem__helper(item=item))

        return l

    def __len__(self):
        return self.num_data

    def __getitem__(self, item):
        return self.cache_data[item]

    def __getitem__helper(self, item):
        x, smps_gt, mask = [torch.unsqueeze(i[item], 0) for i in [self.x, self.smps, self.mask]]

        x_angle = torch.angle(x)
        x_abs = torch.abs(x)

        x_abs -= torch.min(x_abs)
        x_abs /= torch.max(x_abs)

        x = x_abs * np.exp(1j * x_angle)

        y = fmult(x, smps_gt, mask)

        if self.noise_snr > 0:
            y, _ = addwgn(y, self.noise_snr)

        if self.compute_smps == 'low_k':
            smps = compute_y_center_low_k_hamming(y, size=self.compute_smps_low_k_size)
            smps = divided_by_rss(smps)

        elif self.compute_smps == 'esp':
            smps = EspiritCalib(torch.squeeze(y, 0).numpy(), device=Device(0), show_pbar=False).run()
            smps = cupy.asnumpy(smps)
            smps = torch.from_numpy(smps)
            smps = torch.unsqueeze(smps, 0)

        else:
            smps = smps_gt

        x0 = ftran(y, smps, mask)

        x0, smps, mask, y, x, smps_gt = [torch.squeeze(i, 0) for i in [x0, smps, mask, y, x, smps_gt]]

        return x0, smps, y, {'mask': mask}, x, smps_gt
        # x0, theta0, y, (extra variable for forward model), x_groundtruth, theta_groundtruth
