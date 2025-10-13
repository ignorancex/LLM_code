import random
from enum import Enum
from typing import Union, Callable, Any, Sequence, Iterable, Optional

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.spatial import Voronoi, voronoi_plot_2d
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from .voronoi_shuffling import voronoi_finite_polygons_2d, cropping
from ..data_structures.datasets import SegmentationDataset, ImageLoader
from ..data_structures.folder_classification_datasets import SyntheticBackgroundsDataset
from ..data_structures.imagenet import ImageNetS50SegmentationDataset


class BGType(Enum):
    full_bg = 'full_bg'
    voronoi = 'voronoi'
    original = 'original'


class PasteOnBackground:
    """Transformation that takes an image"""

    def __init__(self,
                 background_loader: SegmentationDataset,
                 bg_type: BGType = BGType.full_bg,
                 bg_class: Union[Any, Sequence[Any]] = None, bg_seed: int = None,
                 num_imgs: int = 1,
                 forbidden_classes_bg: Union[Union[str, int], Sequence[Union[str, int]]] = None, filter_top_x: int = 5,
                 n_voronoi_points: int = 8, voronoi_offset: int = 100
                 ):
        # set up loaders
        self.bg_loader: SegmentationDataset = background_loader
        self.bg_type: BGType = bg_type
        self.bg_class: Union[Any, Sequence[Any]] = bg_class
        self.bg_seed: int = bg_seed
        self.num_imgs: int = num_imgs
        self.forbidden_classes_bg: Union[Union[str, int], Sequence[Union[str, int]]] = forbidden_classes_bg
        self.filter_top_x: int = filter_top_x
        self.n_voronoi_points: int = n_voronoi_points
        self.voronoi_offset: int = voronoi_offset

    @staticmethod
    def randomize_voronoi_diagram(bg_loader: SegmentationDataset, rng_seed=None, num_points=32, bg_class=None,
                                  forbidden_classes_bg=None,
                                  size=(256, 256), post_transform=None, voronoi_offset=100,
                                  ) -> Optional[PIL.Image.Image]:
        """
        Generate voronoi centers and fill each cell with a texture image of a certain class

        Parameters
        ----------
        num_points: Number of voronoi centers

        Returns
        ---------
        Voronoi Diagram
        """
        rng = np.random.default_rng(rng_seed)

        fig, ax1 = plt.subplots(dpi=300)
        points = rng.random((num_points, 2)) * size

        # add 4 distant dummy points
        points = np.append(points,
                           [[size[0] + voronoi_offset, size[1] + voronoi_offset], [-0, size[1] + voronoi_offset],
                            [size[0] + voronoi_offset, -0], [-0, -0]], axis=0)
        vor = Voronoi(points)

        voronoi_plot_2d(vor, ax1, show_vertices=False)
        plt.xlim([0, size[0]]), plt.ylim([0, size[1]])

        # generate Voronoi with finite cells
        regions, vertices = voronoi_finite_polygons_2d(vor)

        voronoi_img: PIL.Image.Image = Image.new('RGB', size, 0)

        regions = [r for r in regions if r]  # valid region (not empty polygonpoint list) and not -1 in region:
        imgs_to_paste_from: Iterable[PIL.Image.Image] = bg_loader.get_random_images(
            only_cats=bg_class, forbidden_imagenet_cat_ids=forbidden_classes_bg,
            num_imgs=len(regions))

        _bg_count = 0
        for region, img_to_paste_from in zip(regions, imgs_to_paste_from):
            _bg_count += 1
            polygon = vertices[region]

            voronoi_img = cropping(voronoi_img,
                                   img_to_paste_from,
                                   polygon,
                                   size)
        # Where no suitable backgrounds found?
        if _bg_count == 0:
            plt.close()
            return None

        if post_transform:
            voronoi_img = post_transform(voronoi_img)

        plt.close()

        return voronoi_img

    def __call__(self,
                 fg_image: PIL.Image.Image, fg_mask: torch.Tensor,
                 *,
                 bg_type: BGType = None,
                 bg_class: Union[Any, Sequence[Any]] = None, bg_seed: int = None,
                 num_imgs: int = None,
                 forbidden_classes_bg: Union[Union[str, int], Sequence[Union[str, int]]] = None,
                 filter_top_x: int = None,
                 n_voronoi_points: int = None, voronoi_offset: int = None,
                 strict: bool = True,
                 ) -> tuple[tuple[PIL.Image.Image, ...], torch.Tensor]:
        """Return num_imgs versions of the foreground in fg_image pasted into backgrounds from self.bg_loader.
        The foreground is specified by fg_mask, and valid backgrounds by bg_class and forbidden_classes_bg.

        Args:
            fg_image:       the foreground image that should be cropped and pasted onto a random background
            fg_mask:        the foreground mask to crop the foreground out of the foreground image
            bg_type:        choose between original, full_bg, voronoi
            bg_class:       choose between random (`None`) or the category ID(s) of your desired background segment
            bg_seed:        seed background creation for reproducibility
            forbidden_classes_bg: filter images predicted to contain the specified ImageNet category ID(s)
            filter_top_x:   the x top predictions of an imagenet pretrained image transformer will be considered for filtering.
            n_voronoi_points:     number of voronoi cells
            strict:         whether to raise a ValueError if the number of available backgrounds does not match num_imgs.

        Returns: tuple of ((pasted1, pasted2, ...), fg_mask_tensor);
            mind that the image tuple will be empty if no suitable background is found and strict == False.
        """
        bg_type = bg_type or self.bg_type
        bg_class = bg_class or self.bg_class
        bg_seed = bg_seed or self.bg_seed
        num_imgs = num_imgs or self.num_imgs
        forbidden_classes_bg = forbidden_classes_bg or self.forbidden_classes_bg
        filter_top_x = filter_top_x or self.filter_top_x
        n_voronoi_points = n_voronoi_points or self.n_voronoi_points
        voronoi_offset = voronoi_offset or self.voronoi_offset

        return self.paste_on_random_backgrounds(
            fg_image=fg_image, fg_mask=fg_mask,
            bg_loader=self.bg_loader, bg_type=bg_type, bg_class=bg_class, bg_seed=bg_seed,
            num_imgs=num_imgs,
            forbidden_classes_bg=forbidden_classes_bg, filter_top_x=filter_top_x,
            n_voronoi_points=n_voronoi_points, voronoi_offset=voronoi_offset,
            strict=strict,
        )

    @classmethod
    def paste_on_random_backgrounds(cls,
                                    fg_image: PIL.Image.Image, fg_mask: torch.Tensor,
                                    bg_loader: SegmentationDataset,
                                    *,
                                    bg_type: BGType = BGType.full_bg,
                                    bg_class: Union[Any, Sequence[Any]] = None, bg_seed: int = None,
                                    num_imgs: int = 1,
                                    forbidden_classes_bg: Union[Union[str, int], Sequence[Union[str, int]]] = None,
                                    filter_top_x: int = 5,
                                    n_voronoi_points: int = 8, voronoi_offset: int = 100,
                                    strict: bool = True,
                                    ) -> tuple[tuple[PIL.Image.Image, ...], torch.Tensor]:
        """
        Args:
            fg_image:       the foreground image that should be cropped and pasted onto a random background
            fg_mask:        the foreground mask to crop the foreground out of the foreground image
            bg_loader:      the background dataset to randomly load images from
            bg_type:        choose between original, full_bg, voronoi
            bg_class:       choose between random (`None`) or the category ID(s) of your desired background segment
            bg_seed:        seed background creation for reproducibility
            num_imgs:       if >1, create and return a tuple of that amount of images
            forbidden_classes_bg: filter images predicted to contain the specified ImageNet category ID(s)
            filter_top_x:   the x top predictions of an imagenet pretrained image transformer will be considered for filtering.
            n_voronoi_points:     number of voronoi cells
            voronoi_offset:  see self.voronoi_offset
            strict:         whether to raise a ValueError if the number of available backgrounds does not match num_imgs.
        """
        if bg_type == BGType.original:
            return tuple([fg_image] * num_imgs), fg_mask

        device: torch.device = fg_mask.device

        # validate the given class constraints
        if bg_class is not None:
            bg_class = [bg_class] if isinstance(bg_class, str) else bg_class
            for cat_id in bg_class:
                assert cat_id in bg_loader.cat_ids, \
                    f"Requested background category ID {cat_id} not in available categories. " \
                    f"Available are: {list(bg_loader.cat_ids)}"

        output_size = fg_image.size  # (width, height)
        adjust_size: Callable[[Union[np.ndarray, PIL.Image.Image]], torch.Tensor] = transforms.Compose([
            transforms.Lambda(lambda x: to_tensor(x).to(device) if not isinstance(x, torch.Tensor) else x.to(device)),
            transforms.Resize((output_size[1], output_size[0])),  # Resize to 256 while maintaining aspect ratio
            transforms.CenterCrop((output_size[1], output_size[0]))  # Center crop to 256x256
        ])

        # cut foreground segment and create mask
        fg_mask = torch.stack([fg_mask, fg_mask, fg_mask], dim=0)  # size 3 x width x height

        fg_image, fg_mask = adjust_size(fg_image), adjust_size(fg_mask)

        # build custom background
        random.seed(bg_seed)
        bg_images: Iterable[PIL.Image.Image]
        match bg_type:
            case BGType.full_bg:
                bg_images = bg_loader.get_random_images(
                    only_cats=bg_class, forbidden_imagenet_cat_ids=forbidden_classes_bg,
                    top_x=filter_top_x, seed=bg_seed,
                    num_imgs=num_imgs)
            case BGType.voronoi:
                bg_images = (img for _ in range(num_imgs) if (
                    img := cls.randomize_voronoi_diagram(
                        bg_loader=bg_loader,
                        rng_seed=bg_seed,
                        num_points=n_voronoi_points,
                        bg_class=bg_class,
                        forbidden_classes_bg=forbidden_classes_bg,
                        size=output_size,
                        voronoi_offset=voronoi_offset,
                    )) is not None)
            case _:
                raise ValueError(
                    f"Invalid background randomization type {bg_type=}. Choose from 'full_bg', 'voronoi', 'original'.")

        pasted_images: list[PIL.Image.Image] = []
        for bg_image in bg_images:
            bg_image = adjust_size(bg_image)

            # Add foreground to background
            bg_image[fg_mask] = fg_image[fg_mask]
            bg_image_pil: PIL.Image.Image = transforms.ToPILImage()(bg_image)  # TODO: remove?
            pasted_images.append(bg_image_pil)

        if strict and len(pasted_images) < num_imgs:
            raise ValueError(
                f"Strictly {num_imgs} backgrounds were requested, but only found {len(bg_images)} valid ones. "
                f"({bg_class=}, {forbidden_classes_bg=}, {bg_loader.cat_ids=}, {len(bg_loader)=})"")")

        return tuple(pasted_images), fg_mask[0, :, :].squeeze(0)


# TESTING SETUP
if __name__ == "__main__":
    import os, itertools, tqdm

    save_dir = './data/tmp/examples/'
    os.makedirs(save_dir, exist_ok=True)

    ex_fg_classes = (1, 20, 50)
    ex_bg_types: tuple[BGType, ...] = (BGType.full_bg, BGType.voronoi, BGType.original)
    ex_bg_classes = ('volcanic', 'jungle')  # ('television_studio', 'playground', None)

    ex_bg_seed = 111
    exp_fg_seed = 11
    ex_filterBG_classes = ('pug', 'n02104029', 'fg_class')

    # set common image loading size (if wished)
    img_loader = ImageLoader(img_shape=None)

    # init the background loader
    ex_bg_dataset_key = 'Places'
    # ex_bg_loader = PlacesDataset('./data/places_205_kaggle', image_loader=img_loader)
    ex_bg_loader = SyntheticBackgroundsDataset('./data/synthetic_backgrounds',
                                               image_loader=img_loader)
    # init the foreground loader
    ex_fg_loader = ImageNetS50SegmentationDataset('./data/ImageNetS/ImageNetS50/train',
                                                  './data/ImageNetS/ImageNetS50/train-semi-segmentation',
                                                  image_loader=img_loader)
    # init the paster
    fg_on_bg_generator = PasteOnBackground(background_loader=ex_bg_loader)

    settings_iterator = tqdm.tqdm(list(itertools.product(ex_fg_classes, ex_bg_classes, ex_bg_types)))
    for ex_fg_class, ex_bg_class, ex_bg_type in settings_iterator:
        settings_iterator.set_description(f"fg {ex_fg_class}, bg {ex_bg_class}, {ex_bg_type}")

        # get example foreground
        ex_fg_img_id = next(ex_fg_loader.get_random_img_ids(only_cats=ex_fg_class, num_imgs=1, seed=exp_fg_seed))
        ex_fg_image, ex_fg_mask = ex_fg_loader[ex_fg_img_id]
        ex_fg_img_fn = ex_fg_loader.get_img_filename(ex_fg_img_id)
        settings_iterator.set_description(f"fg {ex_fg_class}, bg {ex_bg_class}, {ex_bg_type} ({ex_fg_img_id})")

        # paste onto randomized background
        images, mask = fg_on_bg_generator(ex_fg_image, ex_fg_mask,
                                          bg_type=ex_bg_type, bg_class=ex_bg_class, bg_seed=ex_bg_seed,
                                          forbidden_classes_bg=ex_filterBG_classes, num_imgs=1)
        mask_np = Image.fromarray((mask.to(torch.uint8) * 255).cpu().numpy(), mode='L')

        settings_iterator.set_description(f"fg {ex_fg_class}, bg {ex_bg_class}, {ex_bg_type} saving ...")
        filename = f"{ex_fg_class}_seed{exp_fg_seed}_on_{ex_bg_class}_seed{ex_bg_seed}_{ex_bg_type}_{ex_fg_img_fn.replace(os.sep, '+')}.png"
        images[0].save(os.path.join(save_dir, f'{filename}.png'))
        mask_np.save(os.path.join(save_dir, f'{filename}_mask.png'))
