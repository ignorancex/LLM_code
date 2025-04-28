import os
import nibabel as nib
import numpy as np


class Fodf():
    def __init__(self, data_path, fodf_anisotropic_names=['fodf'], fodf_isotropic_names=['fodf_gm', 'fodf_csf'], verbose=False, loading_method='memmap'):
        # Print option
        self.verbose = verbose
        # Folder path with the image to load and image name
        self.data_path = data_path
        self.fodf_anisotropic_names = fodf_anisotropic_names
        self.fodf_isotropic_names = fodf_isotropic_names
        self.loading_method = loading_method
        self.cache = self.check_cache_vs_disk()

        # Load data
        self.fodf_anisotropic, self.fodf_isotropic = self.load_fodfs()
        self.shape_anisotropic, self.shape_isotropic = self.get_shape()
        if self.loading_method=='memmap':
            self.add_to_memmap()

    def load_fodfs(self):
        fodf_anisotropic = None
        fodf_isotropic = None
        if self.cache:
            if len(self.fodf_anisotropic_names)>0:
                fodf_anisotropic = []
                for fodf_name in self.fodf_anisotropic_names:
                    fodf_anisotropic.append(self.load_fodf(fodf_name))
                fodf_anisotropic = np.stack(fodf_anisotropic, axis=0)
                self.shape_anisotropic = (len(self.fodf_anisotropic_names), *self.load_fodf(self.fodf_anisotropic_names[0]).shape)
            if len(self.fodf_isotropic_names)>0:
                fodf_isotropic = []
                for fodf_name in self.fodf_isotropic_names:
                    fodf_isotropic.append(self.load_fodf(fodf_name))
                fodf_isotropic = np.stack(fodf_isotropic, axis=0)
        return fodf_anisotropic, fodf_isotropic
    
    def get_shape(self):
        if self.cache:
            if len(self.fodf_anisotropic_names)>0:
                shape_anisotropic = self.fodf_anisotropic.shape
            if len(self.fodf_isotropic_names)>0:
                shape_isotropic = self.fodf_isotropic.shape
        else:
            if len(self.fodf_anisotropic_names)>0:
                shape_anisotropic = (len(self.fodf_anisotropic_names), *self.load_fodf(self.fodf_anisotropic_names[0]).shape)
            if len(self.fodf_isotropic_names)>0:
                shape_isotropic = (len(self.fodf_isotropic_names), *self.load_fodf(self.fodf_isotropic_names[0]).shape)
        if self.verbose:
            print(f'fODF anisotropic shape: {shape_anisotropic}')
            print(f'fODF isotropic shape: {shape_isotropic}')
        return shape_anisotropic, shape_isotropic
        
    def load_fodf(self, fodf_name):
        data_path = f'{self.data_path}/{fodf_name}.nii'
        is_gz = os.path.exists(f'{data_path}.gz')
        data_path = data_path + ('.gz' if is_gz else '')
        if self.verbose:
            print(f'Loading fodf at {data_path}')
        assert os.path.exists(data_path), f'fodf not found at {data_path}'
        data = nib.load(data_path, keep_file_open=True)
        image = data.get_fdata()
        assert len(image.shape) == 4, f'Invalid image shape {image.shape}'
        if self.verbose:
            print(f'Loaded {image.shape} fODF at {data_path}')
        return image
    
    def add_to_memmap(self):
        if self.verbose:
            print('Loading memmap')
        if len(self.fodf_anisotropic_names)>0:
            filename = f'{self.data_path}/fodf_anisotropic.memmap'
            if self.verbose:
                print(f'Loading fODF anisotropic memmap {filename}')
            if not os.path.exists(filename):
                if self.verbose:
                    print(f'Creating fODF memmap {filename}')
                self.fodf_anisotropic_memmap = np.memmap(filename, dtype='float32', mode='w+', shape=self.fodf_anisotropic.shape)
                self.fodf_anisotropic_memmap[:] = self.fodf_anisotropic
                self.fodf_anisotropic_memmap.flush()
                del self.fodf_anisotropic_memmap
            self.fodf_anisotropic = np.memmap(filename, dtype='float32', mode='r', shape=self.shape_anisotropic)
        if len(self.fodf_isotropic_names)>0:
            filename = f'{self.data_path}/fodf_isotropic.memmap'
            if self.verbose:
                print(f'Loading fODF isotropic memmap {filename}')
            if not os.path.exists(filename):
                if self.verbose:
                    print(f'Creating fODF memmap {filename}')
                self.fodf_isotropic_memmap = np.memmap(filename, dtype='float32', mode='w+', shape=self.fodf_isotropic.shape)
                self.fodf_isotropic_memmap[:] = self.fodf_isotropic
                self.fodf_isotropic_memmap.flush()
                del self.fodf_isotropic_memmap
            self.fodf_isotropic = np.memmap(filename, dtype='float32', mode='r', shape=self.shape_isotropic)

    def check_cache_vs_disk(self):
        # Add image in cache if loading method is Numpy or if normalized image need to be added to disk 
        if self.loading_method == 'numpy':
            if self.verbose:
                print('Loading fODF in memory and use numpy array data')
            return True
        elif self.loading_method == 'memmap':
            file_exist = True
            if len(self.fodf_anisotropic_names)>0:
                filename = f'{self.data_path}/fodf_anisotropic.memmap'
                file_exist = file_exist and os.path.exists(filename)
            if len(self.fodf_isotropic_names)>0:
                filename = f'{self.data_path}/fodf_isotropic.memmap'
                file_exist = file_exist and os.path.exists(filename)
            if self.verbose:
                if file_exist:
                    print(f'Memmap fODF already exist at {filename}')
                else:
                    print(f'Loading fODF in memory to create memmap data')
            return not file_exist
        else:
            raise NotImplementedError(f'Invalid fODF loading method {self.loading_method}')