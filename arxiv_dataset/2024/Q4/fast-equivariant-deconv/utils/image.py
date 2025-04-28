import os
import numpy as np
import nibabel as nib
from scipy.linalg import inv, polar

class Image():
    def __init__(self, data_path, data_name, mask_name=None, verbose=False, cache=True):
        # Print option
        self.verbose = verbose
        # Folder path with the image to load and image name
        self.data_path = data_path
        self.data_name = data_name
        self.mask_name = mask_name
        self.cache = cache

        # Load data
        self.image, self.header, self.affine, self.Rinv, self.S = self.load_image()
        self.mask = self.load_mask()
        
    def load_image(self):
        if self.verbose:
            print('Loading Image')
        data_path = f'{self.data_path}/{self.data_name}.nii'
        is_gz = os.path.exists(f'{data_path}.gz')
        data_path = data_path + ('.gz' if is_gz else '')
        assert os.path.exists(data_path), f'Image not found at {data_path}'
        data = nib.load(data_path, keep_file_open=True)
        header = data.header
        affine = data.affine
        R, S = polar(affine[:3, :3])
        Rinv = inv(R)
        image = data
        if self.cache:
            image = image.get_fdata()
        assert len(image.shape) == 4, f'Invalid image shape {image.shape}'
        if self.verbose:
            print(f'Loaded {image.shape} data at {data_path}')
            print(f'with affine \n {affine}')
        return image, header, affine, Rinv, S

    def load_mask(self):
        mask_path = f'{self.data_path}/{self.mask_name}.nii'
        is_gz = os.path.exists(f'{mask_path}.gz')
        mask_path = mask_path + ('.gz' if is_gz else '')
        if os.path.exists(mask_path):
            if self.verbose:
                print(f'Loading mask at {mask_path}')
            mask = nib.load(mask_path).get_fdata()
        else:
            if self.verbose:
                print(f'Mask not found at {mask_path}')
            mask = np.ones(self.image.shape[:3])
        assert mask.shape == self.image.shape[:3], f'Invalid mask shape {mask.shape} and image shape {self.image.shape}'
        return mask
    
    def plot_image(self, z_slice=None, d_slice=None):
        import matplotlib.pyplot as plt
        if z_slice is None:
            z_slice = self.image.shape[2]//2
        assert z_slice < self.image.shape[2], f'Invalid z_slice {z_slice} and image shape {self.image.shape}'
        if d_slice is None:
            if self.cache:
                to_plot = np.mean((self.image[:, :, z_slice] * self.mask[:, :, z_slice, None]), axis=2)
            else:
                to_plot = np.mean((self.image.slicer[:, :, z_slice:z_slice+1].get_fdata(caching='unchanged')[:, :, 0] * self.mask[:, :, z_slice, None]), axis=2)
        else:
            if self.cache:
                to_plot = (self.image[:, :, z_slice, d_slice] * self.mask[:, :, z_slice])
            else:
                to_plot = (self.image.slicer[:, :, z_slice:z_slice+1, d_slice:d_slice+1].get_fdata(caching='unchanged')[:, :, 0, 0] * self.mask[:, :, z_slice])
        plt.imshow(to_plot, cmap='gray')
        plt.show()