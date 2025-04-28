from torch.utils.data import Dataset
from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils import get_image_paths_from_dir
import os

@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, dataset_config.max_pixel_cond, flip=self.flip, to_normal=False) 
        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, dataset_config.max_pixel_ori, flip=self.flip, to_normal=True)  

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]
