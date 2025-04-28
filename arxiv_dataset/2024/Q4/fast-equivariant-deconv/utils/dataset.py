from torch.utils.data import Dataset
import torch
import numpy as np
from .response_function import GroupResponseFunctions


class SingleSubjectdMRI(Dataset):
    def __init__(self, subject, trained_bvals_input, trained_bvals_output, patch_size=1, concatenate=False, verbose=True):
        self.verbose = verbose
        self.subject = subject
        # Patch options
        self.concatenate = concatenate
        assert (not concatenate) or (patch_size%2==1), 'Patch size must be odd when concatenating'
        self.patch_size_output = int((patch_size * (not concatenate)) + (1 * concatenate))
        self.patch_size_input = patch_size
        # Get mask of bvals used to train the model
        self.trained_bvals_input = np.unique(trained_bvals_input)
        self.trained_bvals_output = np.unique(trained_bvals_output)
        self.trained_bvals_input_mask, mask_input_unique_bvals, index_bvals_sel_input, _, _ = subject.gradient.select_bvals(self.trained_bvals_input)
        self.trained_bvals_input_mask = self.trained_bvals_input_mask * subject.gradient.mask
        self.trained_bvals_output_mask, _, _, mask_output_unique_bvals, index_bvals_sel_output = subject.gradient.select_bvals(self.trained_bvals_output)
        # Get constant
        self.n_bval_input = int(np.sum(self.trained_bvals_input_mask))
        self.n_shell_input = len(self.trained_bvals_input)
        self.n_shc_input = subject.feature_s2sh.shape[2]
        self.s2sh_input = np.zeros((self.n_bval_input, self.n_shell_input, self.n_shc_input)) # V x S x C
        print(self.s2sh_input.shape, index_bvals_sel_input, subject.feature_s2sh.shape, self.trained_bvals_input_mask.shape, mask_input_unique_bvals)
        self.s2sh_input[:, index_bvals_sel_input] = subject.feature_s2sh[:, mask_input_unique_bvals>0] # V x S x C
        if self.verbose:
            print(f'Input V: {self.n_bval_input}')
            print(f'Input S: {self.n_shell_input}')
            print(f'Input C: {self.n_shc_input}')
            print(f'Input mask: \n {self.trained_bvals_input_mask}')
            print(f'Input index bvals input: \n {index_bvals_sel_input}')
            print(f'mask_input_unique_bvals: \n {mask_input_unique_bvals}')
            print(f'input unique bvals: \n {self.trained_bvals_input}')

        self.n_bval_output = int(np.sum(self.trained_bvals_output_mask))
        self.n_shell_output = len(self.trained_bvals_output)
        self.n_shc_output = subject.feature_sh2s.shape[1]
        self.sh2s_output = np.zeros((self.n_shell_output, self.n_shc_output, self.n_bval_output)) # S x C x V
        self.sh2s_output[index_bvals_sel_output] = subject.feature_sh2s[:, :, self.trained_bvals_output_mask>0][mask_output_unique_bvals>0] # S x C x V
        if self.verbose:
            print(f'Output V: {self.n_bval_output}')
            print(f'Output S: {self.n_shell_output}')
            print(f'Output C: {self.n_shc_output}')
            print(f'Output mask: \n {self.trained_bvals_output_mask}')
            print(f'Output index bvals output: \n {index_bvals_sel_output}')
            print(f'mask_output_unique_bvals: \n {mask_output_unique_bvals}')
            print(f'output unique bvals: \n {self.trained_bvals_output}')
        
        self.patched_coord_voxel, self.N_patched_voxel = self.subject.get_patched_voxel_coordinate(self.patch_size_input)
        self.loading_method = self.subject.loading_method


    def get_lower_upper_bound(self, coords, shape, patch_size):
        lower = np.maximum(coords - (patch_size // 2), 0)
        lower_patch = lower - (coords - (patch_size // 2))
        upper = np.minimum(coords + (patch_size // 2) + (patch_size%2), shape)
        upper_patch = patch_size - ((coords + (patch_size // 2) + (patch_size%2)) - upper)
        coord_idx = patch_size // 2 - lower_patch
        return lower, upper, lower_patch, upper_patch, coord_idx


    def __len__(self):
        return len(self.patched_coord_voxel)

    def __getitem__(self, index):
        # Get voxel coordinates
        x, y, z = self.patched_coord_voxel[index]
                # Get patch coordinates and # of bvals
        lower, upper, lower_patch, upper_patch, coord_idx = self.get_lower_upper_bound(np.array([x, y, z]), self.subject.image.mask.shape[:3], self.patch_size_input)

        # Load data, depending on loading method
        if self.loading_method == 'h5':
            max_patch_data = torch.from_numpy(self.subject.h5_file[self.subject.data_path][lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])
        elif self.loading_method == 'memmap':
            max_patch_data = torch.from_numpy(self.subject.memmap[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])
        elif self.loading_method == 'nibabel':
            normalization = self.subject.response_function.norm
            max_patch_data = torch.from_numpy(self.subject.image.image.dataobj[lower[0]:upper[0]][:, lower[1]:upper[1]][:, :, lower[2]:upper[2]]) / normalization
        elif self.loading_method == 'numpy':
            max_patch_data = torch.from_numpy(self.subject.image.image[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])

        # Get input features
        input_features = torch.zeros((self.patch_size_input, self.patch_size_input, self.patch_size_input, self.n_bval_input))
        input_features[lower_patch[0]:upper_patch[0], lower_patch[1]:upper_patch[1], lower_patch[2]:upper_patch[2]] = max_patch_data[..., self.trained_bvals_input_mask>0] # P x P x P x V
        if self.concatenate:
            input_features = torch.flatten(input_features, start_dim=0, end_dim=2) # P*P*P x V
            input_features = input_features[:, :, None, None, None] # P*P*P x V x 1 x 1 x 1
        else:
            input_features = input_features.permute(3, 0, 1, 2)[None] # 1 x V x P x P x P

        # Get output features and mask
        output_features = torch.zeros((self.patch_size_output, self.patch_size_output, self.patch_size_output, self.n_bval_output))
        output_mask = torch.zeros((self.patch_size_output, self.patch_size_output, self.patch_size_output))
        if not self.concatenate:
            output_features[lower_patch[0]:upper_patch[0], lower_patch[1]:upper_patch[1], lower_patch[2]:upper_patch[2]] = max_patch_data[..., self.trained_bvals_output_mask>0]
            output_mask[lower_patch[0]:upper_patch[0], lower_patch[1]:upper_patch[1], lower_patch[2]:upper_patch[2]] = torch.from_numpy(self.subject.image.mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])  # P x P x P
        else:
            output_features[:] = max_patch_data[coord_idx[0]:coord_idx[0]+1, coord_idx[1]:coord_idx[1]+1, coord_idx[2]:coord_idx[2]+1][..., self.trained_bvals_output_mask>0]
            output_mask = torch.from_numpy(self.subject.image.mask[x:x+1, y:y+1, z:z+1])  # P x P x P
        output_features = output_features.permute(3, 0, 1, 2) # V x P x P x P
        output_mask = output_mask[None] # 1 x P x P x P

        # Get output B0
        output_b0 = torch.mean(output_features[self.subject.gradient.bvals[self.trained_bvals_output_mask>0]==0], axis=0)
        output_b0 = output_b0[None] # 1 x P x P x P

        # Get input signal to shc matrix
        input_signal_to_shc = torch.Tensor(self.s2sh_input) # V x S x C

        # Get output shc to signal matrix
        output_shc_to_signal = torch.Tensor(self.sh2s_output)  # S x C x V

        batch = {'input_features': input_features, 'output_features': output_features, 'output_b0': output_b0, 'output_mask': output_mask, 'input_signal_to_shc': input_signal_to_shc, 'output_shc_to_signal': output_shc_to_signal, 'coords': torch.Tensor(self.patched_coord_voxel[index])}

        return batch

class MultiSubjectdMRI(Dataset):
    def __init__(self, subject_list, patch_size=1, concatenate=False, verbose=False):
        self.verbose = verbose
        self.subject_list = subject_list
        # Patch options
        self.concatenate = concatenate
        assert (not concatenate) or (patch_size%2==1), 'Patch size must be odd when concatenating'
        self.patch_size_output = int((patch_size * (not concatenate)) + (1 * concatenate))
        self.patch_size_input = patch_size
        # Response Functions
        self.group_response_functions = GroupResponseFunctions([subject.response_functions for subject in subject_list], verbose=self.verbose)
        # Get input/output bvals information
        self.bvals_input = np.unique([b for subject in subject_list for b in subject.gradient.bvals_masked]).tolist()
        self.bvals_output = np.unique([b for subject in subject_list for b in subject.gradient.bvals]).tolist()
        self.max_n_shell_input = len(self.bvals_input)
        self.max_n_shell_output = len(self.bvals_output)
        # Get input/output bvals mapping per subject
        bvals_index_per_subject_input = []
        bvals_index_per_subject_output = []
        for i, subject in enumerate(subject_list):
            bvals_index_per_subject_input.append([])
            bvals_index_per_subject_output.append([])
            for b in subject.gradient.unique_bvals_masked:
                bvals_index_per_subject_input[i].append(self.bvals_input.index(b))
            for b in subject.gradient.unique_bvals:
                bvals_index_per_subject_output[i].append(self.bvals_output.index(b))
        self.bvals_index_per_subject_input = bvals_index_per_subject_input
        self.bvals_index_per_subject_output = bvals_index_per_subject_output
        # Get input max V and C
        self.max_n_bvecs_input = max([self.subject_list[i].gradient.bvecs_masked.shape[0] for i in range(len(self.subject_list))])
        self.max_n_shc_input = max([self.subject_list[i].feature_s2sh.shape[2] for i in range(len(self.subject_list))])
        # Get output max V and C
        self.max_n_bvecs_output = max([self.subject_list[i].gradient.bvecs.shape[0] for i in range(len(self.subject_list))])
        self.max_n_shc_output = max([self.subject_list[i].feature_sh2s.shape[1] for i in range(len(self.subject_list))])
        # Get number of voxels in all masks
        self.n_voxels = sum([subject.N_voxel for subject in self.subject_list])
        # Create hash table to get subject index from voxel index
        self.voxel_to_subject = np.zeros(self.n_voxels, dtype=int)
        self.n_voxel_up_to_subject = np.zeros(len(self.subject_list), dtype=int)
        self.voxel_index = 0
        for i in range(len(self.subject_list)):
            self.voxel_to_subject[self.voxel_index:self.voxel_index + self.subject_list[i].N_voxel] = i
            self.n_voxel_up_to_subject[i] = self.voxel_index
            self.voxel_index += self.subject_list[i].N_voxel
        # Check loading method
        self.loading_method = self.subject_list[0].loading_method
        for subject in self.subject_list:
            assert subject.loading_method == self.loading_method, 'All subjects must have the same loading method'
        # Check if fodf loading
        self.has_fodf = self.subject_list[0].has_fodf
        self.max_n_shc_fodf_anisotropic = 0
        self.n_fodf_anisotropic = 0
        if self.has_fodf:
            self.n_fodf_anisotropic = len(self.subject_list[0].fodfs.fodf_anisotropic_names)
            self.n_fodf_isotropic = len(self.subject_list[0].fodfs.fodf_isotropic_names)
            assert self.n_fodf_anisotropic == self.group_response_functions.n_anisotropic, 'fODF and RF should have the same number of anisotropic components'
            assert self.n_fodf_isotropic == self.group_response_functions.n_isotropic, 'fODF and RF should have the same number of isotropic components'
            for subject in self.subject_list:
                assert subject.has_fodf == self.has_fodf, 'All subjects must have the same fodf'
                assert len(subject.fodfs.fodf_anisotropic_names)== self.n_fodf_anisotropic, 'All subjects must have the same fodf anisotropic'
                assert len(subject.fodfs.fodf_isotropic_names)== self.n_fodf_isotropic, 'All subjects must have the same fodf isotropic'
                if self.has_fodf and self.n_fodf_anisotropic>0:
                    self.max_n_shc_fodf_anisotropic = max(self.max_n_shc_fodf_anisotropic, subject.fodfs.fodf_anisotropic.shape[-1])

        if self.verbose:
            print('-'*30)
            print('-'*6, ' Dataset information: ', '-'*6)
            print(f'Number of subjects: {len(self.subject_list)}')
            print(f'Concatenate: {self.concatenate}')
            print(f'Patch size input: {self.patch_size_input} - Patch size output: {self.patch_size_output}')
            print(f'Max Shell input: {self.max_n_shell_input} - Input unique bvals: \n {self.bvals_input}')
            print(f'Max Shell output: {self.max_n_shell_output} - Output unique bvals: \n {self.bvals_output}')
            print(f'Input max N Vector: {self.max_n_bvecs_input} - Input max N Coefficient: {self.max_n_shc_input}')
            print(f'Output max N Vector: {self.max_n_bvecs_output} - Output max N Coefficient: {self.max_n_shc_output}')
            print(f'Number of voxels (dataset size): {self.n_voxels}')
            print(f'Loading method: {self.loading_method}')
            print('-'*30)

    def get_lower_upper_bound(self, coords, shape, patch_size):
        lower = np.maximum(coords - (patch_size // 2), 0)
        lower_patch = lower - (coords - (patch_size // 2))
        upper = np.minimum(coords + (patch_size // 2) + (patch_size%2), shape)
        upper_patch = patch_size - ((coords + (patch_size // 2) + (patch_size%2)) - upper)
        coord_idx = patch_size // 2 - lower_patch
        return lower, upper, lower_patch, upper_patch, coord_idx
        
    def __len__(self):
        return self.n_voxels
    
    def __getitem__(self, index):
        # Get subject index
        subject_index = self.voxel_to_subject[index]
        # Get voxel index
        voxel_index = index - self.n_voxel_up_to_subject[subject_index]
        # Get subject
        subject = self.subject_list[subject_index]
        # Get voxel coordinates
        x, y, z = subject.coord_voxel[voxel_index]
        # Get patch coordinates and # of bvals
        lower, upper, lower_patch, upper_patch, coord_idx = self.get_lower_upper_bound(np.array([x, y, z]), subject.image.mask.shape[:3], self.patch_size_input)
        input_n_bvals_patch = len(subject.gradient.bvals_masked)
        output_n_bvals_patch = len(subject.gradient.bvals)
        if self.n_fodf_anisotropic>0:
            n_shc_fodf_anisotropic = subject.fodfs.fodf_anisotropic.shape[-1]

        # Load data, depending on loading method
        if self.loading_method == 'h5':
            max_patch_data = torch.from_numpy(subject.h5_file[subject.data_path][lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])
        elif self.loading_method == 'memmap':
            max_patch_data = torch.from_numpy(subject.memmap[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])
        elif self.loading_method == 'nibabel':
            normalization = self.group_response_functions.response_function_list[subject_index].norm
            max_patch_data = torch.from_numpy(subject.image.image.dataobj[lower[0]:upper[0]][:, lower[1]:upper[1]][:, :, lower[2]:upper[2]]) / normalization
        elif self.loading_method == 'numpy':
            max_patch_data = torch.from_numpy(subject.image.image[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])

        # Get input features
        input_features = torch.zeros((self.patch_size_input, self.patch_size_input, self.patch_size_input, self.max_n_bvecs_input))
        input_features[lower_patch[0]:upper_patch[0], lower_patch[1]:upper_patch[1], lower_patch[2]:upper_patch[2], :input_n_bvals_patch] = max_patch_data[..., subject.gradient.mask>0] # P x P x P x V
        if self.concatenate:
            input_features = torch.flatten(input_features, start_dim=0, end_dim=2) # P*P*P x V
            input_features = input_features[:, :, None, None, None] # P*P*P x V x 1 x 1 x 1
        else:
            input_features = input_features.permute(3, 0, 1, 2)[None] # 1 x V x P x P x P

        # Get output features and mask
        output_features = torch.zeros((self.patch_size_output, self.patch_size_output, self.patch_size_output, self.max_n_bvecs_output))
        output_mask = torch.zeros((self.patch_size_output, self.patch_size_output, self.patch_size_output))
        if not self.concatenate:
            output_features[lower_patch[0]:upper_patch[0], lower_patch[1]:upper_patch[1], lower_patch[2]:upper_patch[2], :output_n_bvals_patch] = max_patch_data
            output_mask[lower_patch[0]:upper_patch[0], lower_patch[1]:upper_patch[1], lower_patch[2]:upper_patch[2]] = torch.from_numpy(subject.image.mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])  # P x P x P
        else:
            output_features[..., :output_n_bvals_patch] = max_patch_data[coord_idx[0]:coord_idx[0]+1, coord_idx[1]:coord_idx[1]+1, coord_idx[2]:coord_idx[2]+1]
            output_mask = torch.from_numpy(subject.image.mask[x:x+1, y:y+1, z:z+1])  # P x P x P
        output_features = output_features.permute(3, 0, 1, 2) # V x P x P x P
        output_mask = output_mask[None] # 1 x P x P x P

        # Get output B0
        output_b0 = torch.mean(output_features[subject.gradient.bvals==0], axis=0)
        output_b0 = output_b0[None] # 1 x P x P x P

        # Get input signal to shc matrix
        input_signal_to_shc = torch.zeros((self.max_n_bvecs_input, self.max_n_shell_input, self.max_n_shc_input))
        input_signal_to_shc[:input_n_bvals_patch, :, :subject.feature_s2sh.shape[2]][:, self.bvals_index_per_subject_input[subject_index]] = torch.Tensor(subject.feature_s2sh) # V x S x C

        # Get output shc to signal matrix
        output_shc_to_signal = torch.zeros((self.max_n_shell_output, self.max_n_shc_output, self.max_n_bvecs_output))
        output_shc_to_signal[:, :subject.feature_sh2s.shape[1], :output_n_bvals_patch][self.bvals_index_per_subject_output[subject_index]] = torch.Tensor(subject.feature_sh2s) # S x C x V

        batch = {'input_features': input_features, 'output_features': output_features, 'output_b0': output_b0, 'output_mask': output_mask, 'input_signal_to_shc': input_signal_to_shc, 'output_shc_to_signal': output_shc_to_signal}

        if self.has_fodf:
            if self.n_fodf_anisotropic>0:
                max_patch_data_fodf_anisotropic = torch.from_numpy(subject.fodfs.fodf_anisotropic[:, lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])
                output_anisotropic_fodf = torch.zeros((self.n_fodf_anisotropic, self.patch_size_output, self.patch_size_output, self.patch_size_output, self.max_n_shc_fodf_anisotropic))
                if not self.concatenate:
                    output_anisotropic_fodf[:, lower_patch[0]:upper_patch[0], lower_patch[1]:upper_patch[1], lower_patch[2]:upper_patch[2], :n_shc_fodf_anisotropic] = max_patch_data_fodf_anisotropic
                else:
                    output_anisotropic_fodf[..., :n_shc_fodf_anisotropic] = max_patch_data_fodf_anisotropic[:, coord_idx[0]:coord_idx[0]+1, coord_idx[1]:coord_idx[1]+1, coord_idx[2]:coord_idx[2]+1]
                output_anisotropic_fodf = output_anisotropic_fodf.permute(0, 4, 1, 2, 3) # T x C x P x P x P
                batch['output_anisotropic_fodf'] = output_anisotropic_fodf
            if self.n_fodf_isotropic>0:
                max_patch_data_fodf_isotropic = torch.from_numpy(subject.fodfs.fodf_isotropic[:, lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])
                output_isotropic_fodf = torch.zeros((self.n_fodf_isotropic, self.patch_size_output, self.patch_size_output, self.patch_size_output, 1))
                if not self.concatenate:
                    output_isotropic_fodf[:, lower_patch[0]:upper_patch[0], lower_patch[1]:upper_patch[1], lower_patch[2]:upper_patch[2], :] = max_patch_data_fodf_isotropic
                else:
                    output_isotropic_fodf[..., :] = max_patch_data_fodf_isotropic[:, coord_idx[0]:coord_idx[0]+1, coord_idx[1]:coord_idx[1]+1, coord_idx[2]:coord_idx[2]+1]
                output_isotropic_fodf = output_isotropic_fodf.permute(0, 4, 1, 2, 3) # T x C x P x P x P
                batch['output_isotropic_fodf'] = output_isotropic_fodf
        return batch
