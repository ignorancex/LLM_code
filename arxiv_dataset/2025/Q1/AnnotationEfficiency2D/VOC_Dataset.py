import os, torch
import numpy as np
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, functional
class VOCSeg_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,data_root, data_split='train', msk_styl ='', data_transform=None, 
                 target_transform=None,crop=True,crop_size=320,flip=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #sanity check
        assert data_split in ['train','trainaug','val']
        assert msk_styl in ['perfe','poly+','poly-','rough','scrib','bbox_msk']
        
        self.data_root = data_root 
        self.styl = 'perfe' if (data_split == 'val') else msk_styl

        #determine the train-test split to retrive data from
        self.data_split = data_split
        
        #get filenames in the split
        namefiles = open(f'{data_root}/ImageSets/Segmentation/{data_split}.txt','r')
        self.filenames = namefiles.read().split()
        
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.crop=crop
        self.crop_size=crop_size
        self.flip=flip
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        idx = torch.tensor(idx).flatten()
        # retrive image
        img_name = f'{self.data_root}/JPEGImages/{self.filenames[idx]}.jpg'
        
        #determine mask directory for the data split
        foo = 'SegmentationClassAug' if self.data_split=='trainaug' else 'SegmentationClass' 
        msk_name = f'{self.data_root}/{self.styl}/{foo}/{self.filenames[idx]}.png'
        
        img = Image.open(img_name)
        msk = Image.open(msk_name)
        
        
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
        
        if self.flip and (np.random.rand()>0.5):
            img = ImageOps.mirror(img)
            msk = ImageOps.mirror(msk)
            
            
        if self.data_transform:
            img = self.data_transform(img)
        if self.target_transform:
            msk = torch.tensor(np.array(self.target_transform(msk)),dtype=torch.long)
        
        return img, msk