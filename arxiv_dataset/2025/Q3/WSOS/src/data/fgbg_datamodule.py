from typing import Tuple, List
from types import SimpleNamespace

import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, ImageFolder, FashionMNIST
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
import numpy as np

from src.data.datamodule import DataModule
from src.data.texture_datamodule import TextureDataModule
from src.utils import transfer_color, otsu_threshold, fuse
from src.shared import Color
from src.utils.images_utils import unify_images_intensities, add_smooth_intensity_to_masks
from src.utils.utils import call_background_classifier, normalize_images
import h5py


class ForegroundTextureDataModule(DataModule):
    def __init__(
            self,
            background_dir: str,
            background_type: str = 'texture',
            data_dir: str = "data/",
            dataset_type: str = 'MNIST',
            im_size: List[int] = [64, 64],
            color: Color = None,
            random_resizing_shifting: bool = False,
            train_val_test_split: Tuple[float, float, float] = (.8, .1, .1),
            batch_size: int = 64,
            order_background_labels: bool = False,
            unify_fg_objects_intensity=False,
            background_classifier: str = None,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        super().__init__(background_classifier, order_background_labels)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.bg_dir = background_dir
        self.data_dir = data_dir
        self.background_type = background_type
        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.batch_size_per_device = batch_size

        self.im_size = im_size
        self.random_resizing_shifting = random_resizing_shifting
        self.color = color
        supported_datasets = ['MNIST', 'FashionMNIST', 'dsprites']
        if dataset_type not in supported_datasets:
            raise ValueError(f'{dataset_type} is not supported. The supported datasets are {supported_datasets}')
        self.dataset_type = dataset_type
        self.unify_fg_objects_intensity = unify_fg_objects_intensity
        self.bg_images_for_fg_images = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def assign_bg_labels(self):
        if self.background_classifier:
            bg_images = normalize_images(self.bg_images)
            self.bg_bg_labels = call_background_classifier(self.background_classifier, bg_images,
                                                           None, False, data_type=np.float32)
            if self.bg_images_for_fg_images is None:
                raise RuntimeError(f'`bg_images_for_fg_images` should not be None')
            bg_images = normalize_images(self.bg_images_for_fg_images)
            self.bg_fg_labels = call_background_classifier(self.background_classifier,
                                                           bg_images, None,
                                                           False, data_type=np.float32)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        if self.dataset_type == 'MNIST':
            trainset = MNIST(self.hparams.data_dir, train=True, download=True, transform=self.transforms)
            testset = MNIST(self.hparams.data_dir, train=False, download=True, transform=self.transforms)
        elif self.dataset_type == 'FashionMNIST':
            trainset = FashionMNIST(self.hparams.data_dir, train=True, download=True, transform=self.transforms)
            testset = FashionMNIST(self.hparams.data_dir, train=False, download=True, transform=self.transforms)
        elif self.dataset_type == 'dsprites':
            trainset, testset = self.get_dsprites_datasets()
        else:
            raise ValueError(f'{self.dataset_type} is not supported')
        self.fg_images = torch.concat((trainset.data, testset.data)).unsqueeze(1)
        self.fg_labels = torch.concat((trainset.targets, testset.targets))

        if self.background_type == 'texture':
            texture_dataset = TextureDataModule(2 * len(self.fg_images), self.bg_dir, self.hparams.data_dir,
                                                self.im_size,
                                                self.hparams.train_val_test_split, self.hparams.batch_size)
            texture_dataset.prepare_data()
            self.bg_images = texture_dataset.x
            self.bg_bg_labels = texture_dataset.y
        elif self.background_type == 'sas':
            self.bg_images, self.bg_bg_labels = self.process_sas(self.bg_dir, self.im_size)
        else:
            raise RuntimeError(f'{self.background_type} this background type is not supported.')
        self.process_images()

    def dominant_number_exceeds_80(self, tensor):
        flattened = tensor.flatten()
        unique_elements, counts = torch.unique(flattened, return_counts=True)
        max_count = counts.max().item()
        total_elements = flattened.numel()
        return max_count > 0.8 * total_elements

    def process_sas(self, hdf5_path, patch_size):
        file = h5py.File(hdf5_path, 'r')
        data = file['data']
        seg = file['segments']
        bg_images = []
        bg_labels = []
        for idx, image in enumerate(data):
            image = torch.tensor(abs(image))
            for patch_num in range(100):
                condition = True
                counter = 0
                patch = patch_label = None
                seg_ = torch.tensor(seg[idx])
                while condition and counter < 200:
                    x = torch.randint(0, high=image.shape[0] - patch_size[0], size=(1,)).item()
                    y = torch.randint(0, high=image.shape[1] - patch_size[1], size=(1,)).item()
                    patch_tmp = image[x:x + patch_size[0], y:y + patch_size[1]]
                    labels = seg_[x:x + patch_size[0], y:y + patch_size[1]]
                    if self.dominant_number_exceeds_80(labels):
                        condition = False
                        patch = patch_tmp
                    patch_label = torch.mode(labels.flatten()).values.item()
                    counter += 1
                if patch is not None:
                    patch = patch / patch.mean()
                    patch = torch.clip(patch, 0, 16)
                    bg_images.append(patch)
                    bg_labels.append(patch_label)
        bg_images = torch.stack(bg_images).unsqueeze(1)
        bg_labels = torch.tensor(bg_labels)
        return bg_images, bg_labels

    def _resize_fg(self):
        def get_uniform():
            return (.4 - .6) * torch.rand((1,)).item() + 1

        images_ = torch.zeros_like(self.fg_images)
        for i in range(self.fg_images.shape[0]):
            fg_size = get_uniform()
            fg = self.fg_images[i]
            x_resize = int(fg_size * fg.shape[1])
            y_resize = int(fg_size * fg.shape[2])
            tmp = torch.ones(fg.shape, dtype=torch.uint8)
            resize = torchvision.transforms.Resize((x_resize, y_resize))
            fg = resize(fg)
            random_x = torch.randint(0, self.fg_images[i].shape[1] - fg.shape[1], (1,)).item()
            random_y = torch.randint(0, self.fg_images[i].shape[2] - fg.shape[2], (1,)).item()
            tmp[0, random_x:random_x + fg.shape[1],
            random_y:random_y + fg.shape[2]] = fg
            images_[i] = tmp
        self.fg_images = images_

    def process_fg_images(self):
        self.fg_images = F.resize(self.fg_images, size=self.im_size,
                                  interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                  antialias=False)
        if self.random_resizing_shifting:
            self._resize_fg()
        if self.random_resizing_shifting and self.color:
            raise RuntimeError(f'using random_resizing_shifting with color images is not supported yet')
        self.masks, thresholds = otsu_threshold(self.fg_images)
        if self.unify_fg_objects_intensity:
            if self.dataset_type == 'FashionMNIST' or self.dataset_type == 'MNIST':
                self.fg_images = unify_images_intensities(self.fg_images, thresholds, target_min=200, target_max=250)
            elif self.dataset_type == 'dsprites':
                self.fg_images = add_smooth_intensity_to_masks(self.fg_images,
                                                               torch.randint(2, 100,
                                                                             size=(self.fg_images.shape[0], 1, 1, 1)))
                self.fg_images = unify_images_intensities(self.fg_images, thresholds, target_min=150, target_max=250)
            else:
                raise RuntimeError(
                    f'the parameters of `unify_images_intensities` is not setup for dataset:{self.dataset_type}')

    def smooth_transition(self, tensor, mask, min_value=0.0, max_value=1.0):
        """
        Adds a smooth transition from min_value to max_value within the non-zero region of the mask.
        Args:
            tensor (torch.Tensor): Input tensor (C, H, W), typically an image.
            mask (torch.Tensor): Binary mask (H, W) indicating the region of interest.
            min_value (float): Start value of the transition.
            max_value (float): End value of the transition.
        Returns:
            torch.Tensor: Tensor with the smooth transition applied to the masked region.
        """
        _, height, width = tensor.shape

        # Get the bounding box of the non-zero region in the mask
        nonzero_coords = torch.nonzero(mask)
        ymin, xmin = nonzero_coords.min(dim=0).values
        ymax, xmax = nonzero_coords.max(dim=0).values

        # Create coordinate grids within the bounding box
        y = torch.linspace(0, 1, ymax - ymin + 1).view(-1, 1).repeat(1, xmax - xmin + 1)
        x = torch.linspace(0, 1, xmax - xmin + 1).view(1, -1).repeat(ymax - ymin + 1, 1)

        # Generate a random angle and compute the transition within the mask's bounding box
        angle = torch.tensor(np.pi / 4)
        transition = (x * torch.cos(angle) + y * torch.sin(angle)).clamp(0, 1)

        # Scale transition to the [min_value, max_value] range
        transition = transition * (max_value - min_value) + min_value

        # Create a zero tensor matching the original image shape and fill the transition region
        transition_tensor = torch.zeros_like(tensor)
        transition_tensor[:, ymin:ymax + 1, xmin:xmax + 1] = transition.unsqueeze(0)

        # Apply the transition only to the masked region
        transformed_tensor = tensor + transition_tensor * mask.unsqueeze(0)

        return transformed_tensor

    def _blend_foreground_with_background(self, fg_images: torch.Tensor, bg_images: torch.Tensor,
                                          fg_masks: torch.Tensor):
        fg_images_blended = torch.zeros_like(fg_images, dtype=bg_images.dtype)
        for i in range(len(fg_images)):
            if self.background_type == 'sas':
                min_value = torch.quantile(bg_images[i], torch.empty(1).uniform_(.5, .7).item()).item()
                max_value = torch.quantile(bg_images[i], torch.empty(1).uniform_(.8, 1).item()).item()
                fg_image = self.smooth_transition(fg_images[i], fg_masks[i].squeeze(), min_value, max_value)
            else:
                fg_image = fg_images[i]
            fg_images_blended[i] = fuse(bg_images[i], fg_image, fg_masks[i].to(torch.uint8))
        return fg_images_blended

    def color_images(self, images: torch.Tensor, reference_path: str) -> torch.Tensor:
        transformer = self.transforms.Compose([transforms.Resize(self.im_size),
                                               transforms.ToTensor()])
        dataset = ImageFolder(reference_path, transform=transformer)
        tensor_list = [img for img, _ in dataset]
        reference_images = torch.stack(tensor_list)
        images = torch.repeat_interleave(images[:, None, ...], repeats=3, dim=1).to(reference_images.dtype)
        for idx, image in enumerate(images):
            images[idx] = transfer_color(image, reference_images[torch.randint(0, len(reference_images), (1,))[0]])
        return images

    def process_images(self):
        # process the fg images and produce masks.
        self.process_fg_images()
        if self.color:
            self.bg_images = self.color_images(self.bg_images, self.color.bg_color_reference)
            self.fg_images = self.color_images(self.fg_images, self.color.fg_color_reference)
            # make sure both have the same scale.
            max_value = torch.maximum(self.textures.max(), self.fg_images.max())
            self.bg_images = self.bg_images * max_value / self.bg_images.max()
            self.fg_images = self.fg_images * max_value / self.fg_images.max()
        # blend fg with bg.
        self.fg_images = self._blend_foreground_with_background(self.fg_images[:self.bg_images.shape[0] // 2],
                                                                self.bg_images[:self.bg_images.shape[0] // 2],
                                                                self.masks)
        self.masks = self.masks[:self.fg_images.shape[0]]
        self.fg_labels = self.fg_labels[:self.fg_images.shape[0]]
        self.bg_fg_labels = self.bg_bg_labels[:self.bg_images.shape[0] // 2]
        self.bg_images_for_fg_images = self.bg_images[:self.bg_images.shape[0] // 2]
        self.bg_bg_labels = self.bg_bg_labels[self.bg_images.shape[0] // 2:]
        self.bg_images = self.bg_images[self.bg_images.shape[0] // 2:]

    def get_dsprites_datasets(self) -> Tuple[SimpleNamespace, SimpleNamespace]:
        dataset_zip = np.load(self.data_dir + 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        imgs = torch.tensor(dataset_zip['imgs'])
        # that would be the class of each fg object
        latents_values = torch.tensor(dataset_zip['latents_values'][:, 1])
        shuffled_indices = torch.randint(0, len(imgs), size=(len(imgs),))
        imgs = imgs[shuffled_indices]
        latents_values = latents_values[shuffled_indices]
        # take only 70K images to be consistent with other datasets
        imgs = imgs[:70000]
        latents_values = latents_values[:70000]
        # latents_classes = dataset_zip['latents_classes']
        # a dummy data division into training and testing just to keep the API consistent.
        trainset, testset = SimpleNamespace(), SimpleNamespace()
        trainset.data = imgs[:imgs.shape[0] // 2]
        testset.data = imgs[imgs.shape[0] // 2:]
        trainset.targets = latents_values[:latents_values.shape[0] // 2]
        testset.targets = latents_values[latents_values.shape[0] // 2:]
        return trainset, testset


class AugmentForegroundTextureDataModule(ForegroundTextureDataModule):

    def __init__(
            self,
            background_dir: str,
            n_augments: int,
            background_type: str = 'texture',
            data_dir: str = "data/",
            dataset_type: str = 'MNIST',
            im_size: List[int] = [64, 64],
            color: Color = None,
            random_resizing_shifting: bool = False,
            train_val_test_split: Tuple[float, float, float] = (.8, .1, .1),
            batch_size: int = 64,
            order_background_labels: bool = False,
            unify_fg_objects_intensity=False,
            background_classifier: str = None,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__(background_dir, background_type, data_dir, dataset_type, im_size, color,
                         random_resizing_shifting, train_val_test_split, batch_size, order_background_labels,
                         unify_fg_objects_intensity, background_classifier, num_workers, pin_memory)
        self.already_done = False
        self.n_augments = n_augments

        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(size=self.im_size, scale=(0.8, 1.0)),  # Uses self.im_size
        ])

    def prepare_data(self) -> None:
        super().prepare_data()
        # Additional preparation specific to PatchBirdsDataModule if needed
        pass

    def setup(self, stage=None):
        super().setup(stage)
        if not self.already_done:
            self.data_train = self.create_augmented_dataset(torch.stack([batch[1] for batch in self.data_train]))
            self.data_val = self.create_augmented_dataset(torch.stack([batch[1] for batch in self.data_val]))
            self.data_test = self.create_augmented_dataset(torch.stack([batch[1] for batch in self.data_test]))
            self.already_done = True

    def create_augmented_dataset(self, images: torch.Tensor):
        all_patches = []
        for i in range(len(images)):
            patches = []
            for j in range(self.n_augments):
                patch = self.augmentation(images[i])
                patches.append(patch)
            all_patches.append(torch.stack(patches))
        all_patches = torch.stack(all_patches)  # Shape: (num_images, n_augmentation, channels, height, width)
        return TensorDataset(all_patches)


if __name__ == "__main__":
    _ = ForegroundTextureDataModule()
