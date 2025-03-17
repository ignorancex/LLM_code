from torch.utils.data import Dataset
# from data_list import Imagelist
from pathlib import Path
from torchvision.datasets.folder import default_loader

class SyntheticDataset(Dataset):
    def __init__(self, real_dset, real_root, syn_root, transform=None, target_transform=None) -> None:
        # real_dset should be of type Imagelist
        super().__init__()
        self.real_dset = real_dset
        self.transform = transform if transform is not None else real_dset.transform
        self.target_transform = target_transform if target_transform is not None else real_dset.target_transform
        self.mode_self = real_dset.mode_self
        self.real_root = real_root
        self.syn_root = syn_root
        self.loader = default_loader
        self.labels = real_dset.labels
        self.imgs = [Path(self.syn_root) / Path(p).relative_to(self.real_root) for p in self.real_dset.imgs]
        
    def __len__(self) -> int:
        return len(self.real_dset)
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.real_dset.labels[index]
        
        img = self.loader(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.mode_self:
            return img, target, index
        else:
            return img, target
    
        
        