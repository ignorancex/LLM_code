import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler
from torchvision import transforms
import torchvision

from randaugment import RandAugmentMC

from lightly.transforms.simclr_transform import SimCLRTransform

import pdb
import matplotlib.pyplot as plt 

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

from typing import Dict, List, Optional, Tuple, Union

import torchvision.transforms as T
from torch import Tensor

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.utils import IMAGENET_NORMALIZE

class SimCLRTransform_Custom(MultiViewTransform):
    """Implements the transformations for SimCLR [0, 1].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - ImageNet normalization

    Note that SimCLR v1 and v2 use the same data augmentations.

    - [0]: SimCLR v1, 2020, https://arxiv.org/abs/2002.05709
    - [1]: SimCLR v2, 2020, https://arxiv.org/abs/2006.10029

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of [tensor, tensor].

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value. For datasets with small images,
            such as CIFAR, it is recommended to set `cj_strenght` to 0.5.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        view_transform = SimCLRViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )
        super().__init__(transforms=[view_transform, view_transform])


class SimCLRViewTransform:
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return TRANSFORM["train_orig"](image), transformed
    


TRANSFORM = {
    "train_orig": transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    ),
    "train": SimCLRTransform_Custom(),
    # "test": SimCLRTransform(gaussian_blur=0.0),
    "test": transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    ),
    "strong": transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    ),
}


def pil_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class ImageList(Dataset):
    def __init__(self, root, list_file, transform, strong_transform=None):
        with (root / list_file).open("r") as f:
            paths = [p[:-1].split() for p in f.readlines()]
        self.imgs = [(root / p, int(l)) for p, l in paths]
        self.transform = transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        img = pil_loader(path)
        if self.strong_transform:
            img1 = self.strong_transform(img)
            img2 = self.strong_transform(img)
            return self.transform(img), img1, img2, label
        return self.transform(img), label


class _InfiniteSampler(Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        worker_init_fn=None,
        generator=None,
        drop_last=True,
        num_workers=4,
    ):

        sampler = RandomSampler(dataset, replacement=False, generator=generator)
        batch_sampler = BatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last
        )

        self._infinite_iterator = iter(
            DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler),
                worker_init_fn=worker_init_fn,
            )
        )

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0


class DALoader:
    def __init__(self, args, domain_type, mode, labeled=None, strong_transform=False):
        self.seed = args.seed
        self.bsize = args.bsize
        self.domain_name = args.dataset["domains"][
            args.source if domain_type == "source" else args.target
        ]
        self.root = Path(args.dataset["path"])
        self.text_root = Path(args.dataset["text_path"])

        self.domain_type = domain_type
        self.mode = mode
        self.labeled = labeled
        self.num_shot = 3 if args.shot == "3shot" else 1
        self.strong_transform = strong_transform

    def get_loader_name(self):
        if self.labeled is None:
            return f"{self.domain_type}_{self.mode}"
        return f"{self.domain_type}_{self.labeled}_{self.mode}"

    def get_list_file_name(self):
        if self.domain_type == "source":
            return "all.txt"
        if self.mode == "validation":
            return "val.txt"
        if self.labeled == "labeled":
            return f"train_{self.num_shot}.txt"
        return f"test_{self.num_shot}.txt"

    def get_loader(self):
        list_file = self.text_root / self.domain_name / self.get_list_file_name()
        transform_mode = "train" if self.mode == "train" else "test"
        strong_transform = TRANSFORM["strong"] if self.strong_transform else None
        dset = ImageList(
            self.root,
            list_file,
            transform=TRANSFORM[transform_mode],
            strong_transform=strong_transform,
        )

        g = get_generator(self.seed)

        if self.mode == "train":
            dloader = InfiniteDataLoader(
                dset,
                batch_size=self.bsize,
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=True,
                num_workers=4,
            )
        else:
            dloader = DataLoader(
                dset,
                batch_size=self.bsize,
                worker_init_fn=seed_worker,
                generator=g,
                shuffle=False,
                drop_last=False,
                num_workers=4,
                pin_memory=True,
            )
        return dloader


class DataIterativeLoader:
    def __init__(self, args, strong_transform=False):
        required_loader = [
            DALoader(args, *types)
            for types in [
                ["source", "train"],
                ["source", "test"],
                ["target", "train", "labeled"],
                ["target", "test", "labeled"],
                ["target", "train", "unlabeled", strong_transform],
                ["target", "test", "unlabeled"],
                ["target", "validation"],
            ]
        ]

        self.loaders = {
            loader_type.get_loader_name(): loader_type.get_loader()
            for loader_type in required_loader
        }

        self.s_iter = iter(self.loaders["source_train"])
        self.l_iter = iter(self.loaders["target_labeled_train"])
        self.u_iter = iter(self.loaders["target_unlabeled_train"])

        self.strong_transform = strong_transform

    def __iter__(self):
        while True:
            # sx, sy = next(self.s_iter)
            # sx, sy = sx.float().cuda(), sy.long().cuda()
            # grid_img = torchvision.utils.make_grid(sx, nrow=4)
            # plt.imshow(grid_img.detach().cpu().permute(1, 2, 0))
            # plt.savefig("./testing-images/source-transform.png")
            # plt.clf()
            # pdb.set_trace()
            ((sorig, sx1), (sorig, sx2)), sy = next(self.s_iter)
            sorig, sx1, sx2, sy = sorig.float().cuda(), sx1.float().cuda(), sx2.float().cuda(), sy.long().cuda() 

            # grid_img = torchvision.utils.make_grid(sx, nrow=4)
            # plt.imshow(grid_img.detach().cpu().permute(1, 2, 0))
            # plt.savefig("./testing-images/source-transform.png")
            # plt.clf()
            # grid_img = torchvision.utils.make_grid(ssx, nrow=4)
            # plt.imshow(grid_img.detach().cpu().permute(1, 2, 0))
            # plt.savefig("./testing-images/simclr-source-transform.png")
            
            # pdb.set_trace()

            # tx, ty = next(self.l_iter)
            # tx, ty = tx.float().cuda(), ty.long().cuda()
            # grid_img = torchvision.utils.make_grid(tx, nrow=4)
            # plt.imshow(grid_img.detach().cpu().permute(1, 2, 0))
            # plt.savefig("./testing-images/target-transform.png")

            
            ((torig, tx1), (torig, tx2)), ty = next(self.l_iter)
            torig, tx1, tx2, ty = torig.float().cuda(), tx1.float().cuda(), tx2.float().cuda(), ty.long().cuda() 

            # grid_img = torchvision.utils.make_grid(tx, nrow=4)
            # plt.imshow(grid_img.detach().cpu().permute(1, 2, 0))
            # plt.savefig("./testing-images/target-transform.png")
            # plt.clf()

            # grid_img = torchvision.utils.make_grid(ttx, nrow=4)
            # plt.imshow(grid_img.detach().cpu().permute(1, 2, 0))
            # plt.savefig("./testing-images/simclr-target-transform.png")

            # exit(0)

            if self.strong_transform:
                ux, ux1, ux2, _ = next(self.u_iter)
                ux = [ux.float().cuda(), ux1.float().cuda(), ux2.float().cuda()]
            else:
                # ux, _ = next(self.u_iter)
                ((uorig, ux1), (uorig, ux2)), uy = next(self.u_iter)
                # ux = [ux.float().cuda()]
                ux = [uorig.float().cuda()]

            

            # yield (sx, sy), (tx, ty), ux
            yield (sorig, sx1, sx2, sy), (torig, tx1, tx2, ty), ux

    def __len__(self):
        return 0


