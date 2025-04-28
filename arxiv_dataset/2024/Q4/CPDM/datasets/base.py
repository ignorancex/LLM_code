from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), max_pixel=255.0, flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.max_pixel = float(max_pixel)
        self.flip = flip
        self.to_normal = to_normal

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        
        img_path = self.image_paths[index]
        image = None
        
        try:
            np_image = np.load(img_path, allow_pickle=True)
            np_image = np_image / float(self.max_pixel)
            
            image = Image.fromarray(np_image) 
            image = transform(image) 

        except BaseException as e:
            print(img_path)    
        
        if self.to_normal:
            image = (image - 0.5) * 2.

        image_name = Path(img_path).stem
        
        return image, image_name
      