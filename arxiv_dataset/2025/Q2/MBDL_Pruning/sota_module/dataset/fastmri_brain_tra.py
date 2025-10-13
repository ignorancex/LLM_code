import h5py
import glob
import os
import torch
from sigpy.mri.app import EspiritCalib
from sigpy import Device
import cupy
import tqdm
from torch.utils.data import Dataset
from utility import check_and_mkdir
from skimage.color import gray2rgb
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tifffile import imwrite


DROP_ID = [

]


def get_num_slice_from_raw(raw_path):
    with h5py.File(raw_path, 'r') as f:
        return f['kspace'].shape[0]


def get_id_from_raw(raw_path):
    return raw_path.split('/')[-1].replace('.h5', '')


def image_normalize_and_add_label(x, label):
    height, width = x.shape

    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    x = (x * 255).to(torch.uint8).numpy()

    x = gray2rgb(x)
    x = Image.fromarray(x)

    ImageDraw.Draw(
        x  # Image
    ).text(
        (10, (height - width) // 2 + 10),  # Coordinates
        label,  # Text
        (255, 255, 255)  # Color
    )

    x = ImageOps.grayscale(x)

    x = torch.from_numpy(np.array(x))
    x = x[(height - width) // 2:(-1 * (height - width) // 2), :]

    return x


def load_from_raw_and_cache_smps(slice_index, raw_path):

    smps_path = os.path.join("/".join(raw_path.split('/')[:-1]), 'smps')
    check_and_mkdir(smps_path)

    raw_id = get_id_from_raw(raw_path)
    smps_path = os.path.join(smps_path, raw_id + '_smps.h5')

    imgs_path = os.path.join("/".join(raw_path.split('/')[:-1]), 'imgs')
    check_and_mkdir(imgs_path)

    raw_id = get_id_from_raw(raw_path)
    imgs_path = os.path.join(imgs_path, raw_id + '_imgs.h5')

    if not os.path.exists(smps_path):
        with h5py.File(raw_path, 'r') as f:
            kspace = torch.from_numpy(f['kspace'][:])

        num_slice = kspace.shape[0]
        iter_ = tqdm.tqdm(range(num_slice), desc='Generate coil sensitivity map')

        smps = torch.zeros_like(kspace)
        for i in iter_:
            tmp = EspiritCalib(kspace[i].numpy(), device=Device(0), show_pbar=False).run()
            tmp = cupy.asnumpy(tmp)
            tmp = torch.from_numpy(tmp)

            smps[i] = tmp

        with h5py.File(smps_path, 'w') as f:
            f.create_dataset(name='smps', data=smps)

        x_uncombined = torch.fft.ifft2(kspace)
        x_uncombined = torch.fft.ifftshift(x_uncombined, [-1, -2])

        x_combined_smps = x_uncombined * torch.conj(smps)
        x_combined_smps = x_combined_smps.sum(0)

        with h5py.File(imgs_path, 'w') as f:
            f.create_dataset(name='x_combined_smps', data=x_combined_smps)

        smps = smps[slice_index]
        x_combined_smps = x_combined_smps[slice_index]

    else:

        with h5py.File(smps_path, 'r', swmr=True) as f:
            smps = torch.from_numpy(f['smps'][slice_index])

        with h5py.File(imgs_path, 'r', swmr=True) as f:
            x_combined_smps = torch.from_numpy(f['x_combined_smps'][slice_index])

    with h5py.File(raw_path, 'r', swmr=True) as f:
        kspace = torch.from_numpy(f['kspace'][slice_index])

    x_uncombined = torch.fft.ifft2(kspace)
    x_uncombined = torch.fft.ifftshift(x_uncombined, [-1, -2])

    return x_uncombined, smps, x_combined_smps, raw_id


class FastMRIBrainTra(Dataset):
    name = "fastmri_brain_tra"

    def __init__(self, root_path, type_='T2'):

        assert type_ in ['T2', 'T1', 'T1PRE', 'T1POST', 'FLAIR']

        self.root_path = root_path
        self.type_ = type_

        self.file_paths = glob.glob(os.path.join(root_path, "*_AX%s_*.h5" % type_))
        self.file_paths.sort()

        self.indexes_map = []
        for file_index in range(len(self.file_paths)):

            raw_id = get_id_from_raw(self.file_paths[file_index])
            if raw_id in DROP_ID:
                print('Drop index=', file_index, 'id=', raw_id)
                continue

            num_slice = get_num_slice_from_raw(self.file_paths[file_index])
            for slice_index in range(num_slice):
                # Drop the last five slice.
                self.indexes_map.append([file_index, slice_index])

    def __len__(self):
        return len(self.indexes_map)

    def __getitem__(self, item):
        file_index, slice_index = self.indexes_map[item]

        x_uncombined, smps, x_combined_smps, raw_id = load_from_raw_and_cache_smps(
            slice_index=slice_index,
            raw_path=self.file_paths[file_index]
        )

        return {
            'x_uncombined': x_uncombined,
            'smps': smps,
            'x': x_combined_smps,
            'id': raw_id
        }

    def all_cache_smps_and_generate_combined_images(self):

        # rss = torch.zeros(size=(len(self), 400, 400), dtype=torch.uint8)
        # x = torch.zeros(size=(len(self), 400, 400), dtype=torch.uint8)
        for item in tqdm.tqdm(range(len(self))):
            tmp = self[item]

        #     tmp_rss = image_normalize_and_add_label(
        #         torch.sqrt(torch.sum(torch.abs(tmp['x_uncombined']) ** 2, 0)),
        #         label=tmp['id']
        #         )
        #     height, width = tmp_rss.shape
        #     rss[item, :height, :width] = tmp_rss
        #
        #     tmp_x = x[item, :width, :height] = image_normalize_and_add_label(
        #         torch.abs(tmp['x']),
        #         label=tmp['id']
        #         )
        #     height, width = tmp_x.shape
        #     x[item, :height, :width] = tmp_x
        #
        # imwrite(os.path.join(self.root_path, '%s_rss_qc.tiff' % self.type_), rss.numpy(), imagej=True)
        # imwrite(os.path.join(self.root_path, '%s_x_qc.tiff' % self.type_), x.numpy(), imagej=True)
