import os
os.environ['HDF5_PLUGIN_PATH'] = '/home/jovyan/paramonov/HeadSwapNoRuns/h5_jpeg'
import torch
import h5py
import numpy as np
import random
import torchvision
import torchvision.transforms.functional as TF
import torchvision
from typing import Optional
from glob import glob
from collections import defaultdict

from repos.emoca.gdl.datasets.ImageDatasetHelpers import bbox2point
from src.utils.crops import emoca_crop


def dict_factory():
    return defaultdict(dict)

class Voxceleb2H5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path: str,
        source_len: int,
        samples_cnt: Optional[int] = None,
        shuffle=False,
        transform=None,
        subset_size=None,
        flip_transform=False,
        return_masks=True, 
        cross=False
    ) -> None:
        self.root_path = root_path
        self.source_len = source_len
        self.cross = cross
        self.return_masks = return_masks
        self.flip_transform = flip_transform
        self.h5_dict_tree = defaultdict(dict_factory)
        self.valid_idxs = []
        self.samples_cnt = samples_cnt
        
        self.h5_paths = sorted(glob(f'{root_path}/*/*/*.h5'))
            
        if shuffle:
            rng = random.Random(46)
            rng.shuffle(self.h5_paths)

        self.to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if transform is None:
            transform = self.to_tensor
        self.transform = transform
        
        
        for idx, h5_path in enumerate(self.h5_paths):
            if subset_size is not None:
                if idx >= subset_size:
                    break
            head, h5_file = os.path.split(h5_path)
            head, ref = os.path.split(head)
            head, id = os.path.split(head)
            
            try:
                with h5py.File(h5_path) as f:
                    seg_len = len(f['face_wide_mask'])
                    if self.return_masks:
                        assert 'face_wide_mask' in f
                    assert 'keypoints_68' in f
                    assert 'idx_68' in f and len(f['idx_68']) > self.source_len + 1
                    
            except Exception as e:
                print(h5_path, e)
                continue
            
            if seg_len < (self.source_len + 1):
                continue
            
            self.h5_dict_tree[id][ref][h5_file] = {'idx': len(self.valid_idxs), 'seg_len': seg_len}
            self.valid_idxs.append(idx)
    
    def get_sequence(self, video_tensor, is_flip=False):
        result = torch.stack([
            self.transform(img) if i == (video_tensor.shape[0] - 1) else self.to_tensor(img)
            for i, img in enumerate(video_tensor[:])
        ])
        return result
        

    def flip(self, x, is_image=True):  #flip target
        if is_image:
            x[-1] = torchvision.transforms.functional.hflip(x[-1])
        else:
            x[-1][..., 0] = 512 - x[-1][..., 0] #flip keypoints
        return x
            
    
    def __getitem__(self, idx):
        
        h5_path = self.h5_paths[self.valid_idxs[idx]]
        
        with h5py.File(h5_path) as f:
            seg_len = len(f['face_wide_mask'])
            idx_mask = f['idx_68'] #indices of frames with detected keypoints

            nums = np.random.choice(len(idx_mask), size=self.source_len if self.cross else (self.source_len + 1), replace=False)
            idxs = idx_mask[np.sort(nums).tolist()].tolist()

            if self.cross:
                permutation = [0]
            else:
                target_idx = np.random.choice(len(idxs))
                permutation = list(range(len(idxs)))
                permutation.pop(target_idx)
                permutation.append(target_idx)
            
            face_arc = f['face_arc'][idxs][permutation]
            face_wide = f['face_wide'][idxs][permutation]
        
            if self.return_masks:
                face_wide_mask = torch.stack([TF.to_tensor(x) for x in f['face_wide_mask'][idxs][permutation]]) 
                
            segmentation = torch.stack([TF.to_tensor(x) for x in f['face_wide_parsing_segformer_B5_ce'][idxs][permutation]])
            
            try:
                face_keypoints = f['keypoints_68'][idxs][permutation]
            except Exception as e:
                print(e)
                print(h5_path)
                face_keypoints = torch.zeros((face_wide.shape[0], 68, 2), dtype=torch.int16)

        #get target additionally for cross reenactment
        if self.cross:
            idx_target = np.random.randint(0, self.__len__())
            h5_path_target = self.h5_paths[self.valid_idxs[idx_target]]
            
            with h5py.File(h5_path_target) as f_target:
                seg_len_target = len(f_target['face_wide_mask'])
                idx_mask_target = f_target['idx_68'] #indices of frames with detected keypoints

                nums_target = np.random.choice(len(idx_mask_target), size=1, replace=False).tolist()
                idxs_target = idx_mask_target[nums_target].tolist()
                
                face_arc = np.vstack([face_arc, f_target['face_arc'][idxs_target]])
                face_wide = np.vstack([face_wide, f_target['face_wide'][idxs_target]])
            
                if self.return_masks:
                    
                    face_wide_mask_target = TF.to_tensor(f_target['face_wide_mask'][idxs_target][0]).unsqueeze(0)
                    face_wide_mask = torch.cat([face_wide_mask, face_wide_mask_target], 0)
                    
                segmentation_target =  TF.to_tensor(f_target['face_wide_parsing_segformer_B5_ce'][idxs_target][0]).unsqueeze(0)
                segmentation = torch.cat([segmentation, segmentation_target], 0)
                
                try:
                    face_keypoints_target = f_target['keypoints_68'][idxs_target]
                except Exception as e:
                    print(e)
                    print(h5_path_target)
                    face_keypoints_target = torch.zeros((face_wide_mask_target.shape[0], 68, 2), dtype=torch.int16)
                face_keypoints = np.stack([face_keypoints, face_keypoints_target])
        
        
        face_arc = self.get_sequence(face_arc)
        face_wide = self.get_sequence(face_wide)

        crop = emoca_crop(face_wide * face_wide_mask, face_keypoints)
                    
        if self.flip_transform:
            p = torch.rand(1)
            if p < 0.5:
                face_arc = self.flip(face_arc)
                face_wide = self.flip(face_wide)
                face_wide_mask = self.flip(face_wide_mask)
                segmentation = self.flip(segmentation)
                crop = self.flip(crop)
                face_keypoints = self.flip(face_keypoints, is_image=False)
        
        result = {
            'face_arc': face_arc,
            'face_wide': face_wide
        }
        if self.return_masks:
            result['face_wide_mask'] = face_wide_mask

        result['face_emoca'] = crop.to(torch.float16)
        result['keypoints'] = torch.tensor(face_keypoints).to(torch.int16)
        result['segmentation'] = segmentation
        return result
        

    def __len__(self):
        dataset_len = len(self.valid_idxs)
        if self.samples_cnt is not None:
            dataset_len = min(dataset_len, self.samples_cnt)
        return dataset_len