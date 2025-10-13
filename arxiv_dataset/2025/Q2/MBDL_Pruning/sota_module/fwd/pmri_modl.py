import h5py
import os
import numpy as np
import tifffile
import torch
import tqdm


def gradient_smps(smps, x, y, mask):
    x_adjoint = torch.conj(x)

    ret = fmult(x, smps, mask) - y
    ret = ret * mask.unsqueeze(1)

    ret = torch.fft.ifftshift(ret, [-2, -1])
    ret = torch.fft.ifft2(ret, norm='ortho')
    ret = torch.fft.fftshift(ret, [-2, -1])

    ret = ret * x_adjoint.unsqueeze(1)

    return ret


def ftran(y, smps, mask):
    """
    compute adjoint of fast MRI, x = smps^H F^H mask^H x

    :param y: under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: zero-filled image
    """

    # mask^H
    y = y * mask.unsqueeze(1)

    # F^H
    y = torch.fft.ifftshift(y, [-2, -1])
    x = torch.fft.ifft2(y, norm='ortho')
    x = torch.fft.fftshift(x, [-2, -1])

    # smps^H
    x = x * torch.conj(smps)
    x = x.sum(1)

    return x


def fmult(x, smps, mask):
    """
    compute forward of fast MRI, y = mask F smps x

    :param x: groundtruth or estimated image, shape: batch, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: undersampled measurement
    """

    # smps
    x = x.unsqueeze(1)
    y = x * smps

    # F
    y = torch.fft.ifftshift(y, [-2, -1])
    y = torch.fft.fft2(y, norm='ortho')
    y = torch.fft.fftshift(y, [-2, -1])

    # mask
    mask = mask.unsqueeze(1)
    y = y * mask

    return y


def addwgn(x: torch.Tensor, input_snr):
    noiseNorm = torch.norm(x.flatten()) * 10 ** (-input_snr / 20)

    noise = torch.randn(x.size()).to(x.device)

    noise = noise / torch.norm(noise.flatten()) * noiseNorm

    y = x + noise
    return y


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def np_complex_normalize(x):
    x_angle = np.angle(x)
    x_abs = np.abs(x)

    x_abs -= np.amin(x_abs)
    x_abs /= np.amax(x_abs)

    x = x_abs * np.exp(1j * x_angle)

    return x


def np_normalize_to_uint8(x):
    x -= np.amin(x)
    x /= np.amax(x)

    x = x * 255
    x = x.astype(np.uint8)

    return x


def uniformly_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, randomly_return: bool = False):

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

    return mask


def compute_y_center_low_k_hamming(y, size):

    assert size % 2 == 0

    n_coil, n_x, n_y = y.shape

    center_x = n_x // 2
    center_y = n_y // 2

    mask = torch.zeros(size=(n_x, n_y), dtype=torch.float32, device=y.device)

    window_x = torch.hamming_window(size, device=y.device)
    window_y = torch.hamming_window(size, device=y.device)

    window_2d = torch.sqrt(torch.outer(window_x, window_y))

    mask[center_x - (size // 2): center_x + (size // 2), center_y - (size // 2): center_y + (size // 2)] = window_2d
    mask = mask.unsqueeze(0)

    # mask^H
    y = y * mask

    # F^H
    y = torch.fft.ifftshift(y, [-2, -1])
    x = torch.fft.ifft2(y, norm='ortho')
    x = torch.fft.fftshift(x, [-2, -1])

    return x


_mask_fn = {
    'uniformly_cartesian': uniformly_cartesian_mask
}


def divided_by_rss(smps):
    return smps / (torch.sum(torch.abs(smps) ** 2, 1, keepdim=True).sqrt() + 1e-10)


def load_generated_dataset_handle(
        mode,
        root_path,
        acceleration_rate,
        noise_snr,
        num_of_coil,
        mask_pattern: str = 'uniformly_cartesian',
        birdcage_maps_dim: int = 2,
        smps_hat_method=None,
        acs_percentage=0.175,
        randomly_return=True,
        low_k_size=3
):

    assert mode in ['tra', 'val', 'tst']

    check_and_mkdir(root_path)
    ret = {}

    """
    generate groundtruth images (x)
    """
    x_h5 = os.path.join(root_path, '%s_x.h5' % mode)
    if not os.path.exists(x_h5):
        with h5py.File(os.path.join('/opt/dataset', 'dataset.hdf5'), 'r') as f:

            if mode == 'tst':
                x = f['tstOrg'][30:-30]

            else:
                num_data = f['trnOrg'].shape[0]

                if mode == 'tra':
                    x = f['trnOrg'][:int(num_data * 0.8)]

                else:
                    x = f['trnOrg'][int(num_data * 0.8):]

        tmp = np.ones_like(x)
        for i in range(x.shape[0]):
            tmp[i] = np_complex_normalize(x[i])
        x = tmp

        with h5py.File(x_h5, 'w') as f:
            f.create_dataset(name='x', data=x)

        tmp = np.ones(shape=x.shape, dtype=np.uint8)
        for i in range(x.shape[0]):
            tmp[i] = np_normalize_to_uint8(abs(x[i]))
        tifffile.imwrite(x_h5.replace('.h5', '_qc.tiff'), data=tmp, compression ='zlib', imagej=True)

    ret.update({
        'x': x_h5
    })

    """
    generate groundtruth sensitivity maps (smps)
    """
    if num_of_coil == -1:
        smps_path = os.path.join(root_path, '%s_smps_num_of_coil_%d' % (mode, num_of_coil))
        smps_h5 = os.path.join(smps_path, '%s_smps.h5' % mode)
    else:
        smps_path = os.path.join(root_path, '%s_smps_num_of_coil_%d_birdcage_maps_dim_%d' % (
            mode, num_of_coil, birdcage_maps_dim))
        smps_h5 = os.path.join(smps_path, '%s_smps.h5' % mode)

    check_and_mkdir(smps_path)

    if not os.path.exists(smps_h5):
        if num_of_coil == -1:  # use sensitivity map in the original dataset file.
            with h5py.File(os.path.join('/opt/dataset', 'dataset.hdf5'), 'r') as f:

                if mode == 'tst':
                    smps = f['tstCsm'][30:-30]

                else:
                    num_data = f['trnOrg'].shape[0]

                    if mode == 'tra':
                        smps = f['trnCsm'][:int(num_data * 0.8)]

                    else:
                        smps = f['trnCsm'][int(num_data * 0.8):]

        else:

            with h5py.File(x_h5, 'r', swmr=True) as f:
                x = f['x'][:]

            from sigpy.mri import birdcage_maps

            n_batch, n_x, n_y = x.shape
            if birdcage_maps_dim == 2:
                smps = birdcage_maps((num_of_coil, n_x, n_y), dtype=np.complex64)
                smps = np.expand_dims(smps, 0)
            else:
                smps = birdcage_maps((num_of_coil, n_batch, n_x, n_y), dtype=np.complex64)
                smps = np.transpose(smps, [1, 0, 2, 3])

        with h5py.File(smps_h5, 'w') as f:
            f.create_dataset(name='smps', data=smps)

        tmp = np.ones(shape=smps.shape, dtype=np.uint8)
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                tmp[i, j] = np_normalize_to_uint8(abs(smps[i, j]))
        tifffile.imwrite(smps_h5.replace('.h5', '_qc.tiff'), data=tmp, compression ='zlib', imagej=True)

    ret.update({
        'smps': smps_h5
    })

    """
    generate undersampling mask (mask)
    """
    meas_path = os.path.join(smps_path, '%s_meas_mask_pattern_%s_acceleration_rate_%d' % (
        mode, mask_pattern, acceleration_rate))
    if mask_pattern == 'uniformly_cartesian':
        meas_path += '_acs_percentage_%.3f_randomly_return_%s' % (acs_percentage, str(randomly_return))

    check_and_mkdir(meas_path)

    mask_h5 = os.path.join(meas_path, '%s_mask.h5' % mode)
    if not os.path.exists(mask_h5):
        with h5py.File(x_h5, 'r') as f:
            n_batch, n_x, n_y = f['x'].shape

        mask = np.stack(
            [_mask_fn[mask_pattern]((n_x, n_y), acceleration_rate, acs_percentage, randomly_return)
             for _ in range(n_batch)], 0)

        with h5py.File(mask_h5, 'w') as f:
            f.create_dataset(name='mask', data=mask)

        tifffile.imwrite(mask_h5.replace('.h5', '_qc.tiff'), data=mask, compression='zlib', imagej=True)

    ret.update({
        'mask': mask_h5
    })

    """
    generate noisy measurements (y)
    """
    y_path = os.path.join(meas_path, '%s_noise_snr_%d' % (mode, noise_snr))
    check_and_mkdir(y_path)

    y_h5 = os.path.join(y_path, '%s_y.h5' % mode)
    if not os.path.exists(y_h5):
        with h5py.File(x_h5, 'r') as f:
            x = f['x'][:]

        with h5py.File(smps_h5, 'r') as f:
            smps = f['smps'][:]

        with h5py.File(mask_h5, 'r') as f:
            mask = f['mask'][:]

        x, smps, mask = [torch.from_numpy(i) for i in [x, smps, mask]]
        y = fmult(x, smps, mask)
        y = addwgn(y, noise_snr)

        with h5py.File(y_h5, 'w') as f:
            f.create_dataset(name='y', data=y)

        x_hat = ftran(y, torch.ones_like(y), mask)

        tmp = np.ones(shape=x_hat.shape, dtype=np.uint8)
        for i in range(tmp.shape[0]):
            tmp[i] = np_normalize_to_uint8(abs(x_hat[i]).numpy())
        tifffile.imwrite(y_h5.replace('.h5', '_zero_filled_qc.tiff'), data=tmp, compression='zlib', imagej=True)

    ret.update({
        'y': y_h5
    })

    """
    estimate coil sensitivity maps (smps_hat)
    """
    assert smps_hat_method in ['esp', None, 'low_k']

    if smps_hat_method is not None:

        if smps_hat_method == 'esp':
            smps_hat_h5 = os.path.join(y_path, '%s_smps_hat_method_%s.h5' % (mode, smps_hat_method))
        elif smps_hat_method == 'low_k':
            smps_hat_h5 = os.path.join(y_path, '%s_smps_hat_method_%s_low_k_size_%d.h5' % (
                mode, smps_hat_method, low_k_size))
        else:
            raise NotImplementedError()

        if not os.path.exists(smps_hat_h5):
            os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy'
            from sigpy.mri.app import EspiritCalib
            from sigpy import Device
            import cupy

            with h5py.File(y_h5, 'r') as f:
                y = f['y'][:]

            smps_hat = np.ones_like(y)
            for i in tqdm.tqdm(range(y.shape[0]), desc='Estimating coil sensitivity maps (smps_hat)'):
                if smps_hat_method == 'esp':
                    tmp = EspiritCalib(y[i], device=Device(0), show_pbar=False).run()
                    tmp = cupy.asnumpy(tmp)
                    smps_hat[i] = tmp
                elif smps_hat_method == 'low_k':
                    tmp = compute_y_center_low_k_hamming(torch.from_numpy(y[i]), size=low_k_size).numpy()
                    smps_hat[i] = tmp
                else:
                    raise NotImplementedError()

                if smps_hat_method == 'low_k':
                    smps_hat = divided_by_rss(torch.from_numpy(smps_hat)).numpy()

            with h5py.File(smps_hat_h5, 'w') as f:
                f.create_dataset(name='smps_hat', data=smps_hat)

            with h5py.File(mask_h5, 'r') as f:
                mask = f['mask'][:]

            y, smps_hat, mask = [torch.from_numpy(i) for i in [y, smps_hat, mask]]
            x_hat = ftran(y, smps_hat, mask)

            tmp = np.ones(shape=smps_hat.shape, dtype=np.uint8)
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    tmp[i, j] = np_normalize_to_uint8(abs(smps_hat[i, j]).numpy())
            tifffile.imwrite(smps_hat_h5.replace('.h5', '_qc.tiff'), data=tmp, compression='zlib', imagej=True)

            tmp = np.ones(shape=x_hat.shape, dtype=np.uint8)
            for i in range(tmp.shape[0]):
                tmp[i] = np_normalize_to_uint8(abs(x_hat[i]).numpy())
            tifffile.imwrite(y_h5.replace('.h5', '_smps_combined_qc.tiff'), data=tmp, compression='zlib', imagej=True)

        ret.update({
            'smps_hat': smps_hat_h5
        })

    return ret


class ParallelMRIMoDL:
    def __init__(
            self,
            mode: str,
            root_path: str,
            acceleration_rate: int,
            noise_snr: int,
            num_of_coil: int,
            is_pre_load: bool,
            mask_pattern: str = 'uniformly_cartesian',
            birdcage_maps_dim: int = 2,
            smps_hat_method=None,
            acs_percentage=0.175,
            randomly_return=True,
            low_k_size=3,
    ):

        self.is_pre_load = is_pre_load
        self.smps_hat_method = smps_hat_method

        self.raw_paths = load_generated_dataset_handle(
            mode=mode,
            root_path=root_path,
            acceleration_rate=acceleration_rate,
            noise_snr=noise_snr,
            num_of_coil=num_of_coil,
            mask_pattern=mask_pattern,
            birdcage_maps_dim=birdcage_maps_dim,
            smps_hat_method=smps_hat_method,
            acs_percentage=acs_percentage,
            randomly_return=randomly_return,
            low_k_size=low_k_size
        )

        with h5py.File(self.raw_paths['mask'], 'r') as f:
            self.mask = torch.from_numpy(f['mask'][:])

        with h5py.File(self.raw_paths['smps'], 'r') as f:
            if f['smps'].shape[0] == 1:
                self.smps = torch.from_numpy(f['smps'][:])

        with h5py.File(self.raw_paths['x'], 'r') as f:
            self.n_batch = f['x'].shape[0]

        if self.is_pre_load:
            self.pre_load_data = []

            for item in tqdm.tqdm(range(self.n_batch), desc='Pre-loading data'):
                self.pre_load_data.append(self.__getitem__helper(item))

    def __getitem__helper(self, item):

        with h5py.File(self.raw_paths['x'], 'r', swmr=True) as f:
            x = torch.from_numpy(f['x'][item])

        with h5py.File(self.raw_paths['smps'], 'r', swmr=True) as f:
            if f['smps'].shape[0] > 1:
                smps = torch.from_numpy(f['smps'][item])
            else:
                smps = self.smps[0]

        with h5py.File(self.raw_paths['y'], 'r', swmr=True) as f:
            y = torch.from_numpy(f['y'][item])

        if self.smps_hat_method is not None:
            with h5py.File(self.raw_paths['smps_hat'], 'r', swmr=True) as f:
                smps_hat = torch.from_numpy(f['smps_hat'][item])

        else:
            smps_hat = smps

        mask = self.mask[item]

        x_hat = ftran(
            y.unsqueeze(0),
            smps_hat.unsqueeze(0),
            mask.unsqueeze(0)
        ).squeeze(0)

        return x_hat, smps_hat, y, mask, x, smps

    def __len__(self):
        return self.n_batch

    def __getitem__(self, item):
        if self.is_pre_load:
            return self.pre_load_data[item]

        else:
            return self.__getitem__helper(item)
