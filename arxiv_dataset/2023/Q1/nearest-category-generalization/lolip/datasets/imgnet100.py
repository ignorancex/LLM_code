from typing import Callable, Optional, Any, Tuple, List, Dict
import os

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import (
    make_dataset, default_loader, IMG_EXTENSIONS
)


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            excluded_classes=None,
            only_classes=None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        if only_classes and excluded_classes:
            raise ValueError("Cannot have only_calsses and excluded_classes both not None.")
        self.excluded_classes = excluded_classes
        self.only_classes = only_classes

        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        if self.excluded_classes:
            classes = [c for c in classes if c not in self.excluded_classes]
        elif self.only_classes:
            classes = [c for c in classes if c in self.only_classes]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            excluded_classes=None,
            only_classes=None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader,
                                          IMG_EXTENSIONS if is_valid_file is None else None,
                                          excluded_classes=excluded_classes,
                                          only_classes=only_classes,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

def get_imgnet100(trn_transform, tst_transform, excluded_classes=None,):
    imgnet_trn_dir = "/tmp2/ImageNet100/ILSVRC2012_img_train/"
    imgnet_val_dir = "/tmp2/ImageNet100/ILSVRC2012_img_val/"
    if excluded_classes is not None:
        classes = [d.name for d in os.scandir(imgnet_trn_dir) if d.is_dir()]
        classes.sort()
        excluded_classes = [classes[c] for c in excluded_classes]

        return (
            ImageFolder(imgnet_trn_dir, transform=trn_transform, excluded_classes=excluded_classes),
            ImageFolder(imgnet_val_dir, transform=tst_transform, excluded_classes=excluded_classes),
            ImageFolder(imgnet_trn_dir, transform=tst_transform, only_classes=excluded_classes),
            ImageFolder(imgnet_val_dir, transform=tst_transform, only_classes=excluded_classes),
        )
    else:
        return (
            ImageFolder(imgnet_trn_dir, transform=trn_transform, excluded_classes=excluded_classes),
            ImageFolder(imgnet_val_dir, transform=tst_transform, excluded_classes=excluded_classes),
        )
