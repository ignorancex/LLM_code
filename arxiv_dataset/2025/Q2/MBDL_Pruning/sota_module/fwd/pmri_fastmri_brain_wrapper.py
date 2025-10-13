import h5py

from sota_module.fwd.pmri_fastmri_brain import RealMeasurement, uniformly_cartesian_mask, ftran, load_real_dataset_handle, fmult, addwgn, np_normalize_to_uint8
import os

os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba'

import numpy as np
import tqdm
from sota_module.fwd.pmri_modl import divided_by_rss, compute_y_center_low_k_hamming
import torch
import tifffile


def generate_cache_data(
        root_path,
        file_idx,
        slice_idx,
        smps_method,
        acceleration_rate,
        acs_percentage,
        noise_snr,
        smps_hat_method,
        low_k_size=4,
        is_save_qc_tiff=False,
        num_of_coils=20,
):
    """
    """

    """
    x
    """

    gt_folder = os.path.join(root_path, 'groundtruth')

    if not os.path.exists(gt_folder):
        os.mkdir(gt_folder)

    x_h5 = os.path.join(gt_folder, "x_file_idx_%d_slice_idx_%d.h5" % (file_idx, slice_idx))

    if not os.path.exists(x_h5):
        ret = load_real_dataset_handle(
            file_idx,
            acceleration_rate=1,
            is_return_y_smps_hat=True,
        )

        with h5py.File(ret['x_hat'], 'r', swmr=True) as f:
            x = f['x_hat'][slice_idx]

            half_dim = x.shape[-2] // 4
            x = x[half_dim:-half_dim, :]

            x = torch.from_numpy(x)

            x_abs = torch.abs(x)
            x_angle = torch.angle(x)

            x_abs = x_abs - x_abs.min()
            x_abs = x_abs / x_abs.max()

            x = x_abs * torch.exp(1j * x_angle)

        with h5py.File(x_h5, 'w') as f:
            f.create_dataset(name='x', data=x)

        if is_save_qc_tiff:
            tifffile.imwrite(x_h5.replace('.h5', '_qc.tiff'), data=abs(x).numpy(), compression='zlib', imagej=True)

    else:
        with h5py.File(x_h5, 'r') as f:
            x = torch.from_numpy(f['x'][:])

    """
    smps
    """

    smps_folder = os.path.join(gt_folder, "smps_method_%s" % smps_method)

    if smps_method == 'synthetic':
        smps_folder += "_num_of_coil_%d" % num_of_coils
    if not os.path.exists(smps_folder):
        os.mkdir(smps_folder)

    smps_h5 = os.path.join(smps_folder, "smps_file_idx_%d_slice_idx_%d.h5" % (file_idx, slice_idx))

    if not os.path.exists(smps_h5):

        if smps_method == "original":

            ret = load_real_dataset_handle(
                file_idx,
                acceleration_rate=1,
                is_return_y_smps_hat=True,
            )

            with h5py.File(ret['smps_hat'], 'r', swmr=True) as f:
                smps = f['smps_hat'][slice_idx]

                half_dim = smps.shape[-2] // 4
                smps = smps[:, half_dim:-half_dim, :]

                smps = torch.from_numpy(smps)
                smps = divided_by_rss(smps.unsqueeze(0)).squeeze(0)

        elif smps_method == "synthetic":

            from sigpy.mri import birdcage_maps

            n_x, n_y = x.shape
            smps = birdcage_maps((num_of_coils, n_x, n_y), dtype=np.complex64)
            smps = torch.from_numpy(smps)

        else:
            raise NotImplementedError()

        with h5py.File(smps_h5, 'w') as f:
            f.create_dataset(name='smps', data=smps)

        if is_save_qc_tiff:
            tifffile.imwrite(smps_h5.replace('.h5', '_qc.tiff'), data=abs(smps).numpy(), compression='zlib', imagej=True)

    else:
        with h5py.File(smps_h5, 'r') as f:
            smps = torch.from_numpy(f['smps'][:])

    """
    Measurement
    """

    meas_folder = os.path.join(smps_folder, 'mask_y')

    if not os.path.exists(meas_folder):
        os.mkdir(meas_folder)

    meas_folder = os.path.join(meas_folder, "acceleration_rate_%d_acs_percentage_%f_noise_snr_%d" % (
        acceleration_rate, acs_percentage, noise_snr))

    if not os.path.exists(meas_folder):
        os.mkdir(meas_folder)

    meas_h5 = os.path.join(meas_folder, "file_idx_%d_slice_idx_%d.h5" % (file_idx, slice_idx))

    if not os.path.exists(meas_h5):

        if acceleration_rate > 1:
            mask = uniformly_cartesian_mask(
                img_size=x.shape,
                acceleration_rate=acceleration_rate,
                acs_percentage=acs_percentage
            )

            mask = torch.from_numpy(mask)
        else:
            mask = torch.ones(size=x.shape, dtype=torch.float32)

        y = fmult(
            x.unsqueeze(0),
            smps.unsqueeze(0),
            mask.unsqueeze(0)
        ).squeeze(0)

        y = addwgn(y, noise_snr)

        with h5py.File(meas_h5, 'w') as f:
            f.create_dataset(name='y', data=y)
            f.create_dataset(name='mask', data=mask)

        if is_save_qc_tiff:
            tifffile.imwrite(meas_h5.replace('.h5', '_mask_qc.tiff'), data=abs(mask).numpy(), compression='zlib', imagej=True)

    else:
        with h5py.File(meas_h5, 'r') as f:
            y = torch.from_numpy(f['y'][:])
            mask = torch.from_numpy(f['mask'][:])

    """
    SMPs
    """

    smps_hat_folder = os.path.join(meas_folder, "smps_hat_method_%s" % smps_hat_method)

    if smps_hat_method in ['low_k']:
        smps_hat_folder += "_low_k_size_%d" % low_k_size

    if not os.path.exists(smps_hat_folder):
        os.mkdir(smps_hat_folder)

    smps_hat_h5 = os.path.join(smps_hat_folder, "file_idx_%d_slice_idx_%d.h5" % (file_idx, slice_idx))

    if not os.path.exists(smps_hat_h5):

        from sigpy.mri.app import EspiritCalib
        from sigpy import Device
        import cupy

        if smps_hat_method == 'esp':
            tmp = EspiritCalib(y.numpy(), device=Device(0), show_pbar=False).run()
            tmp = cupy.asnumpy(tmp)
            smps_hat = torch.from_numpy(tmp)

        elif smps_hat_method == 'low_k':
            smps_hat = compute_y_center_low_k_hamming(y, size=low_k_size)

        else:
            raise NotImplementedError()

        smps_hat = divided_by_rss(smps_hat.unsqueeze(0)).squeeze(0)

        x_hat = ftran(
            y.unsqueeze(0),
            smps_hat.unsqueeze(0),
            mask.unsqueeze(0)
        ).squeeze(0)

        if is_save_qc_tiff:
            tifffile.imwrite(smps_hat_h5.replace('.h5', '_x_hat_qc.tiff'), data=abs(x_hat).numpy(), compression='zlib', imagej=True)
            tifffile.imwrite(smps_hat_h5.replace('.h5', '_smps_hat_qc.tiff'), data=abs(smps_hat).numpy(), compression='zlib', imagej=True)

        with h5py.File(smps_hat_h5, 'w') as f:
            f.create_dataset(name='smps_hat', data=smps_hat)
            f.create_dataset(name='x_hat', data=x_hat)

    else:

        with h5py.File(smps_hat_h5, 'r') as f:
            x_hat = f['x_hat'][:]
            smps_hat = f['smps_hat'][:]

    return x_hat, smps_hat, y, mask, x, smps


class ParallelMRIFastMRI:
    def __init__(
            self,
            mode: str,
            root_path: str,
            is_pre_load: bool,
            smps_method,
            acceleration_rate,
            acs_percentage,
            noise_snr,
            smps_hat_method,
            low_k_size=4,
            num_of_coils=20,
    ):

        if mode == 'tra':
            idx_list = range(1310)  # 1585
        elif mode == 'val':
            idx_list = range(1310, 1350)  # 110
        else:
            idx_list = range(1350, 1375)  # 97

        self.dataset = RealMeasurement(
                        idx_list=idx_list,
                        acceleration_rate=1,
                        is_return_y_smps_hat=True
                    )

        self.root_path = root_path
        self.is_preload = is_pre_load

        self.smps_method = smps_method
        self.acceleration_rate = acceleration_rate
        self.acs_percentage = acs_percentage
        self.noise_snr = noise_snr
        self.smps_hat_method = smps_hat_method
        self.low_k_size = low_k_size
        self.num_of_coils = num_of_coils

        # Start preloading the dataset (if needed).
        self.getitem_cache = []
        if self.is_preload:

            for item in tqdm.tqdm(range(len(self)), desc="Preloading data from %s" % root_path):
                self.getitem_cache.append(self.getitem_helper(item))

    def __len__(self):

        return len(self.dataset)

    def getitem_helper(self, item):
        file_idx, _, slice_idx = self.dataset.index_maps[item]

        return generate_cache_data(
            root_path=self.root_path,
            file_idx=file_idx,
            slice_idx=slice_idx,
            smps_method=self.smps_method,
            acceleration_rate=self.acceleration_rate,
            acs_percentage=self.acs_percentage,
            noise_snr=self.noise_snr,
            smps_hat_method=self.smps_hat_method,
            low_k_size=self.low_k_size,
            is_save_qc_tiff=False,
            num_of_coils=self.num_of_coils,
        )

    def __getitem__(self, item):
        if self.is_preload:
            return self.getitem_cache[item]

        else:
            return self.getitem_helper(item)

# class ParallelMRIFastMRI:
#
#     def get_esp_smps_hat_helper(self, file_idx, slice_idx, y):
#
#         if self.smps_hat_method in ['esp']:
#
#             cache_folder = os.path.join(
#                 self.root_path,
#                 "cached_esp_smps_hat_acceleration_rate_%d_acs_percentage_%f" % (
#                     self.acceleration_rate, self.acs_percentage),
#                 )
#
#         elif self.smps_hat_method in ['low_k']:
#
#             cache_folder = os.path.join(
#                 self.root_path,
#                 "cached_smps_hat_acceleration_rate_%d_acs_percentage_%f_low_k_%d" % (
#                     self.acceleration_rate, self.acs_percentage, self.low_k_size),
#             )
#
#         else:
#
#             raise NotImplementedError()
#
#         if not os.path.exists(cache_folder):
#             os.mkdir(cache_folder)
#
#         target_h5 = os.path.join(cache_folder, "file_idx_%d_slice_idx_%d.h5" % (file_idx, slice_idx))
#
#         if not os.path.exists(target_h5):
#
#             y = y.numpy()
#
#             from sigpy.mri.app import EspiritCalib
#             from sigpy import Device
#             import cupy
#
#             if self.smps_hat_method == 'esp':
#                 tmp = EspiritCalib(y, device=Device(0), show_pbar=False).run()
#                 tmp = cupy.asnumpy(tmp)
#                 smps_hat = tmp
#             elif self.smps_hat_method == 'low_k':
#                 tmp = compute_y_center_low_k_hamming(torch.from_numpy(y), size=self.low_k_size).numpy()
#                 smps_hat = tmp
#             else:
#                 raise NotImplementedError()
#
#             if self.smps_hat_method == 'low_k':
#                 smps_hat = divided_by_rss(torch.from_numpy(smps_hat)).numpy()
#
#             with h5py.File(target_h5, 'w') as f:
#                 f.create_dataset(name='smps_hat', data=smps_hat)
#
#         else:
#             with h5py.File(target_h5, 'r') as f:
#                 smps_hat = f['smps_hat'][:]
#
#         return torch.from_numpy(smps_hat)
#
#     def __init__(
#             self,
#             mode: str,
#             root_path: str,
#             is_pre_load: bool,
#             acceleration_rate: int,
#             acs_percentage: float = 0.175,
#             smps_hat_method='esp',
#             low_k_size=4,
#     ):
#
#         if mode == 'tra':
#             idx_list = range(1310)  # 1585
#         elif mode == 'val':
#             idx_list = range(1310, 1350)  # 110
#         else:
#             idx_list = range(1350, 1375)  # 97
#
#         self.dataset = RealMeasurement(
#             idx_list=idx_list,
#             acceleration_rate=1,
#             is_return_y_smps_hat=True
#         )
#
#         self.root_path = root_path
#         self.acceleration_rate = acceleration_rate
#         self.acs_percentage = acs_percentage
#         self.smps_hat_method = smps_hat_method
#         self.low_k_size = low_k_size
#
#         self.is_preload = is_pre_load
#
#         # Start preloading the dataset (if needed).
#         self.getitem_cache = []
#         if self.is_preload:
#
#             for item in tqdm.tqdm(range(len(self)), desc="Preloading data from %s" % root_path):
#                 self.getitem_cache.append(self.getitem_helper(item))
#
#     def __len__(self):
#
#         return len(self.dataset)
#
#     def getitem_helper(self, item):
#
#         x, smps, y_fully_sampled, _, file_idx, slice_idx = self.dataset[item]
#
#         mask = uniformly_cartesian_mask(
#             img_size=x.shape,
#             acceleration_rate=self.acceleration_rate,
#             acs_percentage=self.acs_percentage
#         )
#
#         x, mask, y_fully_sampled, smps = [torch.from_numpy(i) for i in [x, mask, y_fully_sampled, smps]]
#
#         y = y_fully_sampled * mask.unsqueeze(0)
#
#         if self.acceleration_rate > 1:
#             smps_hat = self.get_esp_smps_hat_helper(file_idx, slice_idx, y)
#
#             x_hat = ftran(
#                 y.unsqueeze(0),
#                 smps_hat.unsqueeze(0),
#                 mask.unsqueeze(0)
#             ).squeeze(0)
#
#         else:
#             smps_hat = smps
#             x_hat = x
#
#         return x_hat, smps_hat, y, mask, x, smps
#
#     def __getitem__(self, item):
#         if self.is_preload:
#             return self.getitem_cache[item]
#
#         else:
#             return self.getitem_helper(item)
#
#
# if __name__ == '__main__':
#     ParallelMRIFastMRI(root_path='/opt/dataset/cache_deq_cal/pmri_fastmri', mode='tst', is_pre_load=True, acceleration_rate=4)[0]
