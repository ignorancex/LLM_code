import h5py
import glob
import numpy as np
import torch.utils.data as data

class pix2pix(data.Dataset):
  def __init__(self, root):
    self.root = root

  def __getitem__(self, index):
    file_name=self.root+'/'+str(index)+'.h5'
    f=h5py.File(file_name,'r')
    haze_image=f['haze'][:]
    trans_map=f['trans'][:]
    ato_map=f['ato'][:]
    GT=f['gt'][:]

    haze_image=np.swapaxes(haze_image,0,2)
    trans_map=np.swapaxes(trans_map,0,2)
    ato_map=np.swapaxes(ato_map,0,2)
    GT=np.swapaxes(GT,0,2)

    haze_image=np.swapaxes(haze_image,1,2)
    trans_map=np.swapaxes(trans_map,1,2)
    ato_map=np.swapaxes(ato_map,1,2)
    GT=np.swapaxes(GT,1,2)

    return haze_image, GT,  trans_map, ato_map

  def __len__(self):
    train_list=glob.glob(self.root+'/*h5')
    return len(train_list)

