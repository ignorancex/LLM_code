from typing import Callable, Optional, Tuple, List, Any
import torch, sys
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import numpy as np
import os
from PIL import Image

from .utils import getLogger
from .config import ROOT


class Compose(T.Compose):

    def __init__(self, transforms: List):
        super().__init__(transforms)
        assert isinstance(transforms, list), f"List of transforms required, but {type(transforms)} received ..."

    def append(self, transform: Callable):
        self.transforms.append(transform)

class IdentityTransform:

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __call__(self, x: Any) -> Any:
        return x

class OrderTransform:

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def append(self, transform: Callable, index: int = 0):
        self.transforms[index].append(transform)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '['
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n]'
        return format_string

    def __call__(self, data: Tuple) -> List:
        return [transform(item) for item, transform in zip(data, self.transforms)]


# https://github.com/VITA-Group/Adversarial-Contrastive-Learning/blob/937019219497b449f4cb61cc6118fdf32cc3de12/data/cifar10_c.py#L9
class CIFAR10C(Dataset):
    filename = "CIFAR-10-C"
    def __init__(
        self, root: str = ROOT, 
        transform: Optional[Callable] = None, 
        corruption_type: str = 'snow'
    ):
        root = os.path.join(root, self.filename)
        dataPath = os.path.join(root, '{}.npy'.format(corruption_type))
        labelPath = os.path.join(root, 'labels.npy')

        self.data = np.load(dataPath)
        self.label = np.load(labelPath).astype(np.long)
        self.transform = transform

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.label[idx]

        return img, label

    def __len__(self):
        return self.data.shape[0]
        

class CIFAR100C(CIFAR10C):
    filename = "CIFAR-100-C"


# class TinyImageNet(ImageFolder):
#     filename = "tiny-imagenet-200"
#     def __init__(
#         self, root, split='train', transform=None, target_transform=None, **kwargs,
#     ):
#         root = os.path.join(root, self.filename, split)
#         super().__init__(root, transform=transform, target_transform=target_transform, **kwargs)


class TinyImageNet(Dataset):
    filename = "tiny-imagenet-200"

    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = os.path.join(root, self.filename)
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

class ImageNette(ImageFolder):
    filename = "imagenette2-160"
    def __init__(
        self, root, split='train', transform=None, target_transform=None, **kwargs,
    ):
        root = os.path.join(root, self.filename, split)
        super().__init__(root, transform=transform, target_transform=target_transform, **kwargs)

class WrapperSet(Dataset):

    def __init__(
        self, dataset: Dataset,
        transforms: Optional[str] = None
    ) -> None:
        """
        Args:
            dataset: dataset;
            transforms: string spilt by ',', such as "tensor,none'
        """
        super().__init__()

        self.data = dataset

        try:
            counts = len(self.data[0])
        except IndexError:
            getLogger().info("[Dataset] zero-size dataset, skip ...")
            return

        if transforms is None:
            transforms = ['none'] * counts
        else:
            transforms = transforms.split(',')
        self.transforms = [AUGMENTATIONS[transform] for transform in transforms]
        if counts == 1:
            self.transforms = self.transforms[0]
        else:
            self.transforms = OrderTransform(self.transforms)
        getLogger().info(self.transforms)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int):
        data = self.data[index]
        return self.transforms(data)


AUGMENTATIONS = {
    'none' : Compose([IdentityTransform()]),
    'tensor': Compose([T.ToTensor()]),
    '128resize': Compose([
            T.Resize(144),
            T.CenterCrop(128),
            T.ToTensor(),
    ]),
    'cifar': Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
    ]),
    'tinyimagenet': Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(64),
            T.RandomHorizontalFlip(),
            T.ToTensor()
    ]),
    'imagenette': Compose([
            T.RandomResizedCrop(128),
            T.RandomHorizontalFlip(),
            T.ToTensor()
    ]),
}


