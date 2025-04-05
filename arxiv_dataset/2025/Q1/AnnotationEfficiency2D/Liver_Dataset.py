import os, torch
import numpy as np
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, functional

class Liver_Dataset(Dataset):
    """class labels: background-0, liver&tumor on the liver-1 """

    def __init__(self,data_root, split='trainval', msk_styl ='perfe', data_transform=None, 
                 target_transform=None, crop=False, crop_size=320, flip=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # assert msk_styl in ['perfe','poly+','poly-','bbox_msk','rough','Annotations','sam_box','point_msk']
        assert split in ['trainval', 'test']
        
        self.data_root = data_root 
        namefiles = open(data_root+f'{split}.txt','r')
        self.filenames = namefiles.read().split('\n')[:-1]
        
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.styl = msk_styl if (msk_styl!= 'perfe' and split !='test') else 'msk'
        self.crop = crop
        self.crop_size = crop_size
        self.flip = flip

    def __len__(self):
        return len(self.filenames)

    
    def __getitem__(self, idx):
        
        idx = torch.tensor(idx).flatten()
        img_name = f'{self.data_root}/img/{self.filenames[idx]}' 
        msk_name = f'{self.data_root}/{self.styl}/{self.filenames[idx]}'
        
        img = Image.open(img_name).convert('L')
        msk = Image.open(msk_name).convert('L')
        
        if self.crop:
            im_w, im_h = img.size
            diff_w = max(0,self.crop_size-im_w)
            diff_h = max(0,self.crop_size-im_h)
            
            padding = (diff_w//2, diff_h//2, diff_w-diff_w//2, diff_h-diff_h//2)
            img = functional.pad(img, padding, 0, 'symmetric')
            msk = functional.pad(msk, padding, 255, 'constant')
                    
            t,l,h,w=RandomCrop.get_params(img,(self.crop_size,self.crop_size))

            img = functional.crop(img, t, l, h,w) 
            msk = functional.crop(msk, t, l, h,w)
        
        if self.flip:
            if np.random.rand()>0.5:
                img = ImageOps.mirror(img)
                msk = ImageOps.mirror(msk)
            if np.random.rand()>0.5:
                img = ImageOps.flip(img)
                msk = ImageOps.flip(msk)
        
        if self.data_transform:
            img = self.data_transform(img)
        if self.target_transform:
            msk = torch.tensor(np.array(self.target_transform(msk)),dtype=torch.long)
            ignore = msk==255
            msk[msk>0]=1
            msk[ignore]=255
            
        return img, msk
