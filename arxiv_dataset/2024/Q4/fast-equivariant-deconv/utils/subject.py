import os
import numpy as np
import nibabel as nib

from .gradient import Gradient
from .response_function import ResponseFunctions
from .sampling import Sampling
from .image import Image
from .fodf import Fodf


class SubjectdMRI():
    def __init__(self, data_path, response_function_name='dhollander', verbose=False,
                 features_name='features', mask_name='mask', bvecs_name='bvecs.bvecs', bvals_name='bvals.bvals', gradient_mask_input_name=None,
                 rf_isotropic_names=['gm_response', 'csf_response'], fodf_path=None, fodf_isotropic_names=['fodf_gm', 'fodf_csf'],
                 normalize_per_shell=False, normalize_in_mask=False, sh_degree=18, loading_method='numpy', h5_file=None):
        # Print additionnal information
        self.verbose = verbose
        # Folder path with all the subject data
        self.data_path = data_path
        self.features_name = features_name
        self.mask_name = mask_name
        self.bvecs_name = bvecs_name
        self.bvals_name = bvals_name
        self.gradient_mask_input_name = gradient_mask_input_name
        self.loading_method = loading_method
        self.h5_file = h5_file
        assert (self.h5_file is None) or (self.loading_method == 'h5' and (not self.h5_file is None)), f'Invalid loading method {self.loading_method} for h5 file'
        # Response function options
        self.response_function_name = response_function_name
        self.rf_isotropic_names = rf_isotropic_names
        self.normalize_per_shell = normalize_per_shell
        self.normalize_in_mask = normalize_in_mask
        self.normed_by = 'response' * (not self.normalize_in_mask) + 'mask' * self.normalize_in_mask
        # fODF options
        self.fodf_path = fodf_path
        self.fodf_isotropic_names = fodf_isotropic_names
        # SHC matrix options
        self.sh_degree = sh_degree

        # Load data
        self.use_cache = self.check_cache_vs_disk()
        self.image = self.load_image()
        self.get_masked_voxel_coordinate()
        self.gradient = self.load_gradient()
        self.response_functions = self.load_response_function()
        self.fodfs, self.has_fodf = self.load_fodfs()
        if self.use_cache:
            self.normalize_data()
        self.loading()
        self.feature_s2sh, self.feature_sh2s = self.create_interpolator()

    def loading(self):
        if self.verbose:
            print(f'Loading method: {self.loading_method}')
        if self.loading_method=='h5':
            self.add_to_h5()
        elif self.loading_method=='memmap':
            self.add_to_memmap()
        elif self.loading_method=='nibabel':
            self.add_to_nibabel()
        elif self.loading_method in 'numpy':
            pass
        else:
            raise NotImplementedError(f'Invalid loading method {self.loading_method}')
        
    def check_cache_vs_disk(self):
        # Add image in cache if loading method is Numpy or if normalized image need to be added to disk 
        if self.loading_method == 'numpy':
            return True
        elif self.loading_method == 'nibabel':
            filename = f'{self.data_path}/{self.features_name}_NormedBy_{self.normed_by}_PerShellNormed_{self.normalize_per_shell}.nii'
            return not os.path.exists(filename)
        elif self.loading_method == 'h5':
            return not (self.data_path in self.h5_file)
        elif self.loading_method == 'memmap':
            filename = f'{self.data_path}/{self.features_name}_NormedBy_{self.normed_by}_PerShellNormed_{self.normalize_per_shell}.memmap'
            return not os.path.exists(filename)
        else:
            raise NotImplementedError(f'Invalid loading method {self.loading_method}')
        
    def load_image(self):
        if self.verbose:
            print('-'*30)
            print('-'*6, f' Loading Subject: {self.data_path}', '-'*6)
        image = Image(self.data_path, self.features_name, mask_name=self.mask_name, verbose=self.verbose, cache=self.use_cache)
        return image

    def load_fodfs(self):
        if not self.fodf_path is None:
            full_fodf_path = f'{self.data_path}/{self.fodf_path}'
            if self.verbose:
                print(f'Loading fODFs: {full_fodf_path}')
            fodfs = Fodf(full_fodf_path, fodf_anisotropic_names=['fodf'], fodf_isotropic_names=self.fodf_isotropic_names, verbose=self.verbose, loading_method=self.loading_method)
            assert fodfs.fodf_anisotropic.shape[1:4] == self.image.image.shape[:3], f'Invalid Anisotropic fODF shape {fodfs.fodf_anisotropic.shape[1:4]} and Features shape {self.image.image.shape[:3]}'
            assert fodfs.fodf_anisotropic.shape[0] == 1, f'Invalid Anisotropic fODF shape {fodfs.fodf_anisotropic.shape[0]} and Provided tissue {1}'
            if len(self.fodf_isotropic_names)>0:
                assert fodfs.fodf_isotropic.shape[1:4] == self.image.image.shape[:3], f'Invalid Isotropic fODF shape {fodfs.fodf_isotropic.shape[1:4]} and Features shape {self.image.image.shape[:3]}'
                assert fodfs.fodf_isotropic.shape[0] == len(self.fodf_isotropic_names), f'Invalid Isotropic fODF shape {fodfs.fodf_isotropic.shape[0]} and Provided tissue {len(self.fodf_isotropic_names)}'
            has_fodf = True
        else:
            print(f'No fODF path provided')
            has_fodf = False
            fodfs = None
        return fodfs, has_fodf

    def get_masked_voxel_coordinate(self):
        ind = np.arange(self.image.mask.size)[self.image.mask.flatten() != 0]
        self.x, self.y, self.z = np.unravel_index(ind, self.image.mask.shape)
        self.coord_voxel = np.stack((self.x, self.y, self.z), axis=1)
        self.N_voxel = len(self.x)
        if self.verbose:
            print(f'Number of voxel in mask: {self.N_voxel}')

    def get_patched_voxel_coordinate(self, patch_size):
        #x_min = np.arange(self.mask.shape[0])[np.sum(self.mask, aixs=(1, 2))>0][0]
        #x_max = np.arange(self.mask.shape[0])[np.sum(self.mask, aixs=(1, 2))>0][-1]
        #print(f"x_min: {x_min}, x_max: {x_max}")
        step = 1
        patched_x = np.arange(patch_size//2, self.image.mask.shape[0] - patch_size//2, step)
        patched_y = np.arange(patch_size//2, self.image.mask.shape[1] - patch_size//2, step)
        patched_z = np.arange(patch_size//2, self.image.mask.shape[2] - patch_size//2, step)
        patched_coord_voxel = np.array(np.meshgrid(patched_x, patched_y, patched_z)).T.reshape(-1, 3)
        patched_coord_voxel_keep = []
        for coord in patched_coord_voxel:
            if np.sum(self.image.mask[coord[0]-patch_size//2:coord[0]+(patch_size // 2) + (patch_size%2),
                                      coord[1]-patch_size//2:coord[1]+(patch_size // 2) + (patch_size%2),
                                      coord[2]-patch_size//2:coord[2]+(patch_size // 2) + (patch_size%2)]) > 0:
                patched_coord_voxel_keep.append(coord)
        N_patched_voxel = len(patched_coord_voxel_keep)
        return patched_coord_voxel_keep, N_patched_voxel

    def load_gradient(self):
        if self.verbose:
            print(f'Loading gradient table')
        gradient = Gradient(self.data_path, bvecs_name=self.bvecs_name, bvals_name=self.bvals_name, mask_name=self.gradient_mask_input_name, Rinv=self.image.Rinv, verbose=self.verbose)
        assert gradient.bvecs.shape[0] == self.image.image.shape[3], f'Invalid B-Gradients shape {gradient.bvecs.shape} and Features shape {self.image.image.shape}'
        return gradient
    
    def load_response_function(self):
        if self.verbose:
            print(f'Loading response function: {self.response_function_name}')
        response_function_path = f'{self.data_path}/response_functions/{self.response_function_name}'
        if not os.path.exists(f'{self.data_path}/response_functions/{self.response_function_name}'):
            if self.verbose:
                print(f'Response function not found at {response_function_path}')
                print(f'Should run response function estimation first')
                print(f'mkdir -p {response_function_path}')
                print('conda deactivate')
                print('conda activate mrtrix3')
                print(f'dwi2response dhollander {self.data_path}/{self.features_name}.nii.gz {response_function_path}/wm_response.txt {response_function_path}/gm_response.txt {response_function_path}/csf_response.txt -mask {self.data_path}/{self.mask_name}.nii.gz -voxels {response_function_path}/voxels.nii.gz -fslgrad {self.data_path}/{self.bvecs_name} {self.data_path}/{self.bvals_name}')
                print('conda deactivate')
            response_function_path = None
        else:
            response_functions = ResponseFunctions(response_function_path, rf_isotropic_names=self.rf_isotropic_names, verbose=self.verbose, normalize_per_shell=self.normalize_per_shell)
            assert response_functions.n_shell == self.gradient.unique_bvals.shape[0], f'Number of shells in response functions ({response_functions.n_shell}) does not match number of shells in bvals ({self.gradient.unique_bvals.shape[0]})'
            if response_functions.provided_bvals:
                assert np.all(response_functions.bvals == self.gradient.unique_bvals), f'Unique bvals in response functions ({response_functions.bvals}) does not match unique bvals in bvals ({self.gradient.unique_bvals})'
        return response_functions
        
    def normalize_data(self):
        if not self.response_functions is None:
            if self.verbose:
                print(f"Normalizing data: {'shell wise'*self.response_functions.normalize_per_shell}{'B0 wise'*(not self.response_functions.normalize_per_shell)}")
                print(f"Normalization with: {'in mask'* self.normalize_in_mask}{'with response function'*(not self.normalize_in_mask)}")
            if self.response_functions.normalize_per_shell:
                for i, b in enumerate(self.gradient.unique_bvals):
                    self.image.image[:, :, :, self.gradient.bvals==b] = self.image.image[:, :, :, self.gradient.bvals==b] / self.response_functions.norm[i]
                    if self.normalize_in_mask:
                        b_mean =  self.image.image[self.image.mask>0][:, self.gradient.bvals==b].mean()
                        self.image.image[:, :, :, self.gradient.bvals==b] = self.image.image[:, :, :, self.gradient.bvals==b] / b_mean
                        self.response_functions.norm[i] = self.response_functions.norm[i] * b_mean
                        if self.response_functions.n_anisotropic>0:
                            self.response_functions.rf_anisotropics[:, i] = self.response_functions.rf_anisotropics[:, i] / b_mean
                        if self.response_functions.n_isotropic>0:
                            self.response_functions.rf_isotropics[:, i] = self.response_functions.rf_isotropics[:, i] / b_mean
            else:
                self.image.image = self.image.image / self.response_functions.norm
                if self.normalize_in_mask:
                    b0_mean =  self.image.image[self.image.mask>0][:, self.gradient.bvals==0].mean()
                    self.image.image = self.image.image / b0_mean
                    self.response_functions.norm = self.response_functions.norm * b0_mean
                    if self.response_functions.n_anisotropic>0:
                        self.response_functions.rf_anisotropics = self.response_functions.rf_anisotropics / b0_mean
                    if self.response_functions.n_isotropic>0:
                        self.response_functions.rf_isotropics = self.response_functions.rf_isotropics / b0_mean
        elif self.normalize_in_mask:
            if self.verbose:
                print(f'No response function provided')
                print(f"Normalizing data: {'shell wise'*self.response_functions.normalize_per_shell}{'B0 wise'*(not self.response_functions.normalize_per_shell)}")
                print(f"Normalization with: {'in mask'* self.normalize_in_mask}{'with response function'*(not self.normalize_in_mask)}")
            if self.normalize_per_shell:
                for i, b in enumerate(self.gradient.unique_bvals):
                    b_mean =  self.image.image[self.image.mask>0][:, self.gradient.bvals==b].mean()
                    self.image.image[:, :, :, self.gradient.bvals==b] = self.image.image[:, :, :, self.gradient.bvals==b] / b_mean
            else:
                b0_mean =  self.image.image[self.image.mask>0][:, self.gradient.bvals==0].mean()
                self.image.image = self.image.image / b0_mean

    def add_to_h5(self):
        if self.verbose:
            print(f'Adding data to h5 file')
        if not self.data_path in self.h5_file:
            if self.verbose:
                print(f'Creating group {self.data_path}')
            self.h5_file.create_dataset(self.data_path, data=self.image.image)
        del self.image.image

    def add_to_memmap(self):
        filename = f'{self.data_path}/{self.features_name}_NormedBy_{self.normed_by}_PerShellNormed_{self.normalize_per_shell}.memmap'
        if self.verbose:
            print(f'Adding data to memmap at {filename}')
        if not os.path.exists(filename):
            if self.verbose:
                print(f'Creating memmap {filename}')
            self.memmap = np.memmap(filename, dtype='float32', mode='w+', shape=self.image.image.shape)
            self.memmap[:] = self.image.image
            self.memmap.flush()
        self.memmap = np.memmap(filename, dtype='float32', mode='r', shape=self.image.image.shape)
        del self.image.image

    def add_to_nibabel(self):
        filename = f'{self.data_path}/{self.features_name}_NormedBy_{self.normed_by}_PerShellNormed_{self.normalize_per_shell}.nii'
        if self.verbose:
            print(f'Adding data to nibabel at {filename}')
        if not os.path.exists(filename):
            if self.verbose:
                print(f'Creating nibabel {filename}')
            self.image.image = nib.Nifti1Image(self.image.image, self.image.affine, self.image.header)
            nib.save(self.image.image, filename)
        self.image.image = nib.load(filename)

    def create_interpolator(self):
        max_sh_degree = 8
        if self.verbose:
            print(f'Creating interpolator')
        # Get input interpolator
        feature_s2sh = np.zeros((len(self.gradient.bvals_masked), len(self.gradient.unique_bvals_masked), int((max_sh_degree + 1)*(max_sh_degree/2 + 1))))
        for i, s in enumerate(self.gradient.unique_bvals_masked):
            vertice = self.gradient.bvecs_masked[self.gradient.bvals_masked == s]
            sampling = Sampling(vertice, max_sh_degree, max_sh_degree=max_sh_degree, constant=(s==0))
            feature_s2sh[:, i][self.gradient.bvals_masked == s, :sampling.S2SH.shape[1]] = sampling.S2SH
        # Get output interpolator
        feature_sh2s = np.zeros((len(self.gradient.unique_bvals), int((self.sh_degree + 1)*(self.sh_degree/2 + 1)), len(self.gradient.bvals)))
        for i, s in enumerate(self.gradient.unique_bvals):
            vertice = self.gradient.bvecs[self.gradient.bvals == s]
            sampling = Sampling(vertice, self.sh_degree, max_sh_degree=max_sh_degree, constant=(s==0))
            feature_sh2s[i][:sampling.SH2S.shape[0], self.gradient.bvals == s] = sampling.SH2S
        if self.verbose:
            print(f'feature_s2sh: {feature_s2sh.shape}')
            print(f'feature_sh2s: {feature_sh2s.shape}')
        return feature_s2sh, feature_sh2s
            

    def plot(self, slice=0):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5*self.gradient.unique_bvals.shape[0] + 1, 5))
        if self.cache:
            data = self.image.image[:, :, slice]
        else:
            img = self.image.image.slicer[:, :, slice:slice+1]
            data = img.get_fdata()
        for i, b in enumerate(self.gradient.unique_bvals):
            plt.subplot(1, self.gradient.unique_bvals.shape[0] + 1, i + 1)
            plt.imshow(np.mean(data[:, :, 0, self.gradient.bvals==b], axis=-1))
            plt.title(f'B-value: {b}')
            plt.colorbar()
        img.uncache()
        plt.subplot(1, self.gradient.unique_bvals.shape[0] + 1, self.gradient.unique_bvals.shape[0] + 1)
        plt.imshow(self.image.mask[:, :, slice])
        plt.title(f'Mask')
        plt.show()