from typing import Sequence, Dict, Union
import math
import time

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
import torchvision

from utils.file import load_file_list
from utils.image import center_crop_arr, augment, random_crop_arr
from utils.degradation import (
    random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
)


class CodeformerDataset(data.Dataset):
    
    def __init__(
        self,
        file_list: str, # 原本应该是一个txt文件，这里改成dir
        out_size: int,
        crop_type: str,
        use_hflip: bool,



        file_list_dlg: str,
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.paths_dlg = load_file_list(file_list_dlg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip

        self.transform = torchvision.transforms.ToTensor()


    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        ref_path = self.paths_dlg[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                pil_ref = Image.open(ref_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        

        if self.transform is not None:
            img_gt = self.transform(pil_img) # [-1,1]
            img_gt = img_gt * 2 -1
            ref_gt = self.transform(pil_ref) # [0,1]
            # ref_gt = ref_gt * 2 -1

        
        return dict(img_gt=img_gt,ref_gt=ref_gt, txt="")


    def __len__(self) -> int:
        return len(self.paths)
