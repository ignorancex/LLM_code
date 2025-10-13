import skimage
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import io
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset


class WRIZZDataset(Dataset):
    def __init__(self,
                 folder_path,
                 csv_path,
                 resolution,
                 augmentation=None,
                 in_memory=False,
                 normalize=True,
                 skip_intra=False,
                 skip_cross=False):
        
        """ W-RIZZ (https://github.com/andreschreiber/W-RIZZ) style dataset for relative traversability estimation
        
        :param folder_path: path to folder containing the images
        :param csv_path: path to label csv
        :param resolution: resolution to use (H, W)
        :param augmentation: augmentations (or None if no augmentations)
        :param in_memory: whether to keep in memory
        :param normalize: whether to normalize images
        :param skip_intra: whether to skip intra-image labels
        :param skip_cross: whether to skip cross-image labels
        """

        super().__init__()
        
        self._samples = None
        self._folder_path = Path(folder_path)
        self._csv = pd.read_csv(csv_path)
        self._resolution = resolution # (H,W)
        self._augmentation = augmentation
        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225]) if normalize else None
        self._skip_intra = skip_intra
        self._skip_cross = skip_cross
        
        if in_memory:
            self._samples = [self._read_item(i) for i in range(self._csv.shape[0])]
    
    def _read_item(self, idx):
        """ Reads an item with at the specified index """

        entry = self._csv.iloc[idx]
        imageA_file = self._folder_path / entry['imageA_name']
        imageB_file = self._folder_path / entry['imageB_name']
        width, height = entry['width'], entry['height']
        annotation_str = entry['labels']
        
        # Read image
        imageA = io.imread(imageA_file)[:,:,:3] # :3 to remove potential alpha channel
        imageA = Image.fromarray(skimage.img_as_ubyte(self._rescale_color(imageA, resolution=self._resolution)))
        imageB = io.imread(imageB_file)[:,:,:3] # :3 to remove potential alpha channel
        imageB = Image.fromarray(skimage.img_as_ubyte(self._rescale_color(imageB, resolution=self._resolution)))
        
        # Read annotations
        annotations = []
        for a in annotation_str.split(';'):
            l = [int(s) for s in a[1:-1].split(',')]
            scale_x = self._resolution[1] / width
            scale_y = self._resolution[0] / height
            
            if l[0] == l[3] and self._skip_intra == True:
                continue
            if l[0] != l[3] and self._skip_cross == True:
                continue
            
            # l[-1] == 0 => eq; l[-1] == 1 => latter is more; l[-1] == -1 => former is more
            annotations.append((
                l[0], # 0 if pt1 is in imgA, 1 if it's in imgB
                max(0, min(round(scale_x * l[1]), self._resolution[1]-1)),
                max(0, min(round(scale_y * l[2]), self._resolution[0]-1)),
                l[3], # 0 if pt2 is in imgA, 1 if it's in imgB
                max(0, min(round(scale_x * l[4]), self._resolution[1]-1)),
                max(0, min(round(scale_y * l[5]), self._resolution[0]-1)),
                l[6]
            ))
        
        return {
            'data': (transforms.ToTensor()(imageA), transforms.ToTensor()(imageB), torch.tensor(annotations, dtype=torch.long)),
            'imageA': imageA_file,
            'imageB': imageB_file
        }
        
    def _rescale_color(self, image, resolution):
        """ Rescale a color image """
        if resolution is None:
            return image
        else:
            rescaled = skimage.transform.resize(image, resolution)
            return rescaled
    
    def __len__(self):
        """ Get size of dataset """
        return self._csv.shape[0]
    
    def __getitem__(self, idx, augment=True, normalize=True):
        """ Get an item from the dataset
        
        :param idx: index to fetch
        :param augment: if True, augmentations will be applied (if there are any)
        :param normalize: if True, image will be normalized
        :returns: a dict with keys 'data', 'imageA', 'imageB'
        """
        if self._samples is None:
            item = self._read_item(idx)
        else:
            item = self._samples[idx]

        data = item['data']
        if self._augmentation is not None and augment:
            data = self._augmentation(data)
        
        if self._normalize is not None and normalize:
            data = (*[self._normalize(d) for d in data[0:-1]], data[-1])
        
        return {
            'data': data,
            'imageA': str(item['imageA']),
            'imageB': str(item['imageB'])
        }
