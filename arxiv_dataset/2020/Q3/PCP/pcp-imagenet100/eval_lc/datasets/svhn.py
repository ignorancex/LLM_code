#  copy from AND
from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import numpy as np

class SVHNInstance(datasets.SVHN):
    """
    SVHNInstance Dataset
    """

    def __init__(self, root, train=True, transform=None, target_trainsform=None, download=False):
        self.train = train
        super(SVHNInstance, self).__init__(root, split=('train' if train else 'test'),
                                           transform=transform, target_transform=target_trainsform, download=download)

    def __getitem__(self, index):
        """
                Args:
                    index (int): Index
                Returns:
                    tuple: (image, target) where target is index of the target class.
            """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img1 = self.transform(img)
            if self.train:
                img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img1, img2, target, index

        return img1, target, index