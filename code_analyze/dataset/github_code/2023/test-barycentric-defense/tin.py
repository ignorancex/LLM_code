import os
from PIL import Image
import torch.utils.data as data


def parseClasses(file):
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(0,len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames,classes

class TImgNetDataset(data.Dataset):
    """Dataset wrapping images and ground truths."""
    
    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parseClasses(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])

    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, y) where y is the label of the image.
            """
            img = None
            with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
            y = self.classidx[index]
            return img, y

    def __len__(self):
        return len(self.imgs)