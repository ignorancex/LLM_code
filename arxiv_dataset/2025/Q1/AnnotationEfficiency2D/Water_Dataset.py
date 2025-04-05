import os, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class Water_Dataset(Dataset):
    #A binary classificaition example 
    def __init__(self,data_root, split='trainval', msk_styl ='Annotations', data_transform=None, target_transform=None):
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
        self.styl = msk_styl if (msk_styl!= 'perfe' and split !='test') else 'Annotations'
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        idx = torch.tensor(idx).flatten()
        img_name = f'{self.data_root}/JPEGImages/{self.filenames[idx]}'
        if not os.path.exists(img_name):
            img_name = img_name[:-4]+'.jpg'
        
        msk_name = f'{self.data_root}/{self.styl}/{self.filenames[idx]}'
        
        try:
            img = Image.open(img_name).convert('RGB')
            msk = Image.open(msk_name).convert('L')
        
            if self.data_transform:
                img = self.data_transform(img)
            if self.target_transform:
                temp = np.array(self.target_transform(msk))
                msk = torch.tensor(temp>np.max(temp)//2,dtype=torch.long)
        except:
            print(img_name, msk_name)
            assert False
            
        return img, msk