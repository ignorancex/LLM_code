from .voxceleb import Voxceleb2H5Dataset
import torchvision
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random
import h5py

from src.blender.utils import (
    encode_face_segmentation,
    DrawMethod,
    dilate,
    make_composed_random_irregular_mask,
    make_random_irregular_mask,
    make_affine_augmentation,
    make_portrait_mask
)


class BlenderDataset(Voxceleb2H5Dataset):
    def __init__(
        self,
        root_path: str,
        source_transform=None,
        flip_target: bool = True,
        affine_source: bool = True,
        make_noise: bool = True,
        flip_p: float = 0.5,
        affine_p: float = 0.5,
        noise_p: float = 0.5,
        max_rot: float = 15,
        max_shear: float = 15,
        **kwargs
    ) -> None:
        super(BlenderDataset, self).__init__(
            root_path,
            source_len=1,
            **kwargs
        )
        
        self.source_transform = source_transform
        if source_transform is None:
            self.source_transform = self.to_tensor
            
        self.flip_transform = lambda x: x
        if flip_target:
            self.flip_transform = torchvision.transforms.RandomHorizontalFlip(p=flip_p)
        self.flip_p = flip_p
        self.max_rot = max_rot
        self.max_shear = max_shear
        self.affine_source = affine_source
        self.affine_p = affine_p
        self.make_noise = make_noise
        self.noise_p = noise_p
        
            
    
    def __getitem__(self, idx):
        h5_path = self.h5_paths[self.valid_idxs[idx]]
        
        with h5py.File(h5_path) as f:
            seg_len = len(f['face_wide'])
            start_frame = np.random.choice(seg_len)
            if seg_len == 1:
                side_frame = start_frame
            else:
                side_frame = (start_frame + np.random.choice(seg_len - 1)) % seg_len
            
            face_wide = f['face_wide'][start_frame]
            mask_source = torch.tensor(f['face_wide_parsing_segformer_B5_ce'][start_frame])
            face_orig = self.to_tensor(face_wide)
            face_target = face_orig
            mask_target = mask_source
            face_source = self.source_transform(face_wide)
            gray_source = TF.rgb_to_grayscale(face_source[[2, 1, 0], ...])
            
            if random.random() < self.flip_p:
                face_target = TF.hflip(face_target)
                mask_target = TF.hflip(mask_target)
                
                # face_orig = TF.hflip(face_orig)
                face_source = TF.hflip(face_source)
                mask_source = TF.hflip(mask_source)
                gray_source = TF.hflip(gray_source)
                
            
            if self.affine_source and random.random() < self.affine_p:
                h, w = face_wide.shape[:2]
                params = torchvision.transforms.RandomAffine.get_params([-self.max_rot, self.max_rot], None, None, [-self.max_shear, self.max_shear], [h, w])
                face_source = make_affine_augmentation(face_source[None, ...], params)[0]
                gray_source = make_affine_augmentation(gray_source[None, ...], params)[0]
                mask_source = make_affine_augmentation(mask_source[None, None, :, :], params)[0, 0, :, :]
            
            
            
            face_side = self.source_transform(f['face_wide'][side_frame])
            mask_side = torch.tensor(f['face_wide_parsing_segformer_B5_ce'][side_frame])
            
            if random.random() < self.flip_p:
                face_side = TF.hflip(face_side)
                mask_side = TF.hflip(mask_side)
            
            if self.affine_source and random.random() < self.affine_p:
                h, w = face_wide.shape[:2]
                params = torchvision.transforms.RandomAffine.get_params([-self.max_rot, self.max_rot], None, None, [-self.max_shear, self.max_shear], [h, w])
                face_side = make_affine_augmentation(face_side[None, ...], params)[0]
                mask_side = make_affine_augmentation(mask_side[None, None, :, :], params)[0, 0, :, :]
                
            mask_source_noise = make_composed_random_irregular_mask(mask_source)
            mask_target_noise = make_composed_random_irregular_mask(mask_target)
            mask_side_noise = make_composed_random_irregular_mask(mask_side)
            
            
            if (not self.make_noise) or (self.make_noise and (random.random() >= self.noise_p)):
                mask_source_noise.fill(0.)
                mask_target_noise.fill(0.)
                mask_side_noise.fill(0.)
            
        
        mask_source = mask_source[None].to(torch.float32)
        mask_target = mask_target[None].to(torch.float32)
        mask_side = mask_side[None].to(torch.float32)
        
        mask_source_noise = mask_source_noise[None].astype(np.float32)
        mask_target_noise = mask_target_noise[None].astype(np.float32)
        mask_side_noise = mask_side_noise[None].astype(np.float32)

        face_source = face_source * make_portrait_mask(mask_source)
        
        return {
            'face_orig': face_orig,
            'face_source': face_source,
            'gray_source': gray_source,
            'face_target': face_target,
            'face_side': face_side,
            
            'mask_source': mask_source,
            'mask_target': mask_target,
            'mask_side': mask_side,
            
            'mask_source_noise': mask_source_noise,
            'mask_target_noise': mask_target_noise,
            'mask_side_noise': mask_side_noise,
        }

    def __len__(self):
        dataset_len = len(self.valid_idxs)
        if self.samples_cnt is not None:
            dataset_len = min(dataset_len, self.samples_cnt)
        return dataset_len
