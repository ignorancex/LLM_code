import collections
import itertools
import os
import random
from abc import abstractmethod
from typing import Iterable, Tuple, Union, Optional, Callable, Any, Generator, Literal, Sequence, TYPE_CHECKING

import PIL
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .imagenet_labels import IMAGENET_CLASS_NAMES_BY_ID

    _CatID = Union[int, str]
    _ImgID = Union[int, str]
    _TransformType = Callable[[PIL.Image.Image, Union[torch.Tensor, dict['_CatID', torch.Tensor]]], Any]
    _ImageNetLabel = Literal[*IMAGENET_CLASS_NAMES_BY_ID.values()]


class AbstractDataset(Dataset):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __getitem__(self,
                    index: int
                    ) -> Tensor:
        pass

    @abstractmethod
    def __len__(self
                ) -> int:
        pass

    def get_item_no_reshape(self,
                            idx: int
                            ) -> Tensor:
        """
        Implement only in case if returning of single non-reshaped image is required.

        For instance: 
            ./ice/ice_detection.py -> _get_prots()
            ./concept_analysis/ice_concept_similarity.py -> _plot_sample_concepts()
        
        Otherwise use self[i]
        """
        raise NotImplementedError("Not implemented")

    def get_item_as_pil_image(self,
                              idx: int
                              ) -> Image.Image:
        raise NotImplementedError("Not implemented")


class ImageLoader:

    def __init__(self,
                 img_shape: Optional[Tuple[int, int]] = (640, 480)
                 ) -> None:
        self.img_shape: Optional[Tuple[int, int]] = img_shape

    def load_pil_img(self,
                     img_folder: str,
                     img_name: str = None
                     ) -> Image.Image:
        """Arguments: (img_folder, img_name) or only img_path."""
        img_path: str = os.path.join(img_folder, img_name) if img_name is not None else img_folder
        img_pil = Image.open(img_path).convert('RGB')
        if self.img_shape is not None:
            img_pil = img_pil.resize(self.img_shape)
        return img_pil



class SegmentationDataset(AbstractDataset):
    """A dataset that yields pairs of (image, segmentation) for given segmentation categories.
    If combine_masks is set to False, the segmentations of the different categories
    are not merged, and instead pairs of (image, {category: segmentation}) are returned.

    Uses an ImageLoader instance for loading images from file.
    """

    ALL_CAT_NAMES_BY_ID: dict['_CatID', str]
    ALL_CAT_IDS: list['_CatID']

    img_ids: list['_ImgID']

    DEFAULT_IMAGE_LOADER: ImageLoader = ImageLoader()

    def __init__(self,
                 imgs_path: str,
                 *,
                 category_ids: list['_CatID'] = None,
                 category_names_by_id: dict['_CatID', str] = None,
                 combine_masks: bool = True,
                 image_loader: ImageLoader = None,
                 transform: Optional['_TransformType'] = None,
                 tag: str = None,
                 device: Union[str, torch.device] = None):
        """
        Args:
            category_ids: optionally: restrict to given category IDs
            category_names_by_id: optionally: restrict to given pairs of category ID and name
            combine_masks: whether to combine segmentation masks from the category_ids or return as dict of masks
            transform: transformation applied after mask merging and before return from __getitem__
            tag: can be used for pretty printing
            device: torch device to use for image transformation operations;
                defaults to cuda if available
        """
        self.imgs_path = imgs_path
        """Root path to the images. Used in get_img_path()."""

        self.image_loader: ImageLoader = image_loader or self.DEFAULT_IMAGE_LOADER
        """The image loader applied to the numpy images"""

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        """Device to load masks to and do mask data_model_combinations on."""

        self.cat_name_by_id: dict['_CatID', str] = category_names_by_id or self.ALL_CAT_NAMES_BY_ID
        if category_ids is not None:
            self.cat_name_by_id = {i: n for i, n in self.cat_name_by_id.items() if i in category_ids}
        self._img_ids_by_cat_id: Optional[dict['_CatID', list['_ImgID']]] = None
        self._cat_ids_by_img_id: Optional[dict['_ImgID', list['_CatID']]] = None

        self.combine_masks: bool = combine_masks
        """Whether to merge all masks from the different category IDs into one."""

        self.transform: Optional['_TransformType'] = transform
        """Any transformation to apply before returning from __getitem__.
        Applied AFTER any mask combination (see `combine_masks`)."""

        self._tag = tag

        self._imagenet_class_specifiers_by_img_id: dict[str, dict] = {}
        self._timm_preprocess: Optional[Callable] = None
        self._timm_model: Optional[torch.nn.Module] = None
        """Model used for filtering out images containing certain classes in get_random_image."""

    @property
    def cat_ids(self) -> list['_CatID']:
        return list(self.cat_name_by_id.keys())

    @abstractmethod
    def get_img_filename(self, img_id: '_ImgID') -> str:
        pass

    @abstractmethod
    def get_cats(self, img_id: '_ImgID') -> Sequence['_CatID']:
        pass

    @property
    def img_ids_by_cat_id(self) -> dict['_CatID', list['_ImgID']]:
        """Mapping from category ID to image IDs including that category.
        Lazily populated only on first call (possibly expensive!)."""

        # on first call populate the mapping:
        if self._img_ids_by_cat_id is None:
            _img_ids_by_cat = collections.defaultdict(list)
            for img_id in self.img_ids:
                for cat_id in self.get_cats(img_id):
                    _img_ids_by_cat[cat_id].append(img_id)
            self._img_ids_by_cat_id = dict(_img_ids_by_cat)

        return self._img_ids_by_cat_id

    @property
    def cat_ids_by_img_id(self) -> dict['_ImgID', list['_CatID']]:
        """Mapping from image ID to corresponding category IDs.
        Lazily populated only on first call (possibly expensive!)."""

        # on first call populate the mapping:
        if self._cat_ids_by_img_id is None:
            _cat_ids_by_img_id: dict['_ImgID', list['_CatID']] = collections.defaultdict(list)
            for cat_id, img_ids in self.img_ids_by_cat_id.items():
                for img_id in img_ids:
                    _cat_ids_by_img_id[img_id].append(cat_id)
            self._cat_ids_by_img_id = dict(_cat_ids_by_img_id)

        return self._cat_ids_by_img_id

    def get_img_ids_with_cat(self, cat_id: '_CatID') -> list['_ImgID']:
        return self.img_ids_by_cat_id[cat_id]

    def get_img_path(self, img_id: '_ImgID') -> str:
        return os.path.join(self.imgs_path, self.get_img_filename(img_id=img_id))

    def load_img(self, img_id: '_ImgID') -> PIL.Image.Image:
        return self.image_loader.load_pil_img(self.get_img_path(img_id))

    @abstractmethod
    def load_segs(self, img_id: '_ImgID') -> dict['_CatID', torch.Tensor]:
        """Load the list of segmentation masks for image with given ID, ordered by self.category_ids."""
        pass

    def _combine_binary_masks_from_dict(self, masks_dict: dict['_CatID', Union[torch.Tensor, np.ndarray]]) -> torch.Tensor:
        masks_t: list[torch.Tensor] = [torch.as_tensor(m, device=self.device) for m in masks_dict.values()]
        return torch.stack(masks_t, dim=0).sum(dim=0) > 0

    def __getitem__(self, img_id: '_ImgID') -> tuple[PIL.Image.Image, Union[torch.Tensor, dict['_CatID', torch.Tensor]]]:
        """Return a tuple of (image, binary_segmentation) or (image, {category: binary_segmentation_of_category})."""
        img: PIL.Image.Image = self.load_img(img_id)
        segs: dict['_CatID', torch.Tensor] = self.load_segs(img_id)

        # Merging:
        if self.combine_masks:
            segs: torch.Tensor = self._combine_binary_masks_from_dict(segs)

        # Transformation:
        transformed = self.transform(img, segs) if self.transform is not None else (img, segs)
        return transformed

    @staticmethod
    def resize_seg(seg_mask: np.ndarray, target_seg_size: Tuple[int, int]) -> np.ndarray:
        """Utility method for resizing the numpy segmentation masks."""
        return resize(seg_mask.astype(float), target_seg_size)

    def __len__(self) -> int:
        return len(self.img_ids)

    def _check_img_id(self, img_id: str) -> bool:
        """Combination of several sanity checks to filter out invalid image-segmentation-pairs."""
        return self._check_exists(img_id) and self._check_seg_mask_sizes(img_id)

    def _check_exists(self, img_id: str) -> bool:
        """Check whether the image file exists."""
        return os.path.exists(self.get_img_path(img_id))

    def _check_seg_mask_sizes(self, img_id: str) -> bool:
        """Check whether any of the segmentation mask areas fulfills size constraints."""
        for seg_mask in self.load_segs(img_id).values():
            seg_mask_area: int = seg_mask.nelement() if isinstance(seg_mask, torch.Tensor) \
                else seg_mask.size
            seg_area = seg_mask.sum() / seg_mask_area
            if self.MIN_SEG_AREA <= seg_area <= self.MAX_SEG_AREA:
                return True
        return False

    MIN_SEG_AREA: float = 0.0
    MAX_SEG_AREA: float = 1.0

    @property
    def tag(self) -> str:
        return self._tag or self.__class__.__name__.lower()

    def shuffle(self) -> 'SegmentationDataset':
        """Shuffle the img_ids in place."""
        random.shuffle(self.img_ids)
        return self

    def subselect(self, img_ids: list['_ImgID']) -> 'SegmentationDataset':
        """Subselect the img_ids in place."""
        self.img_ids = [i for i in img_ids if i in self.img_ids]
        return self

    def get_random_img_ids(self,
                           only_cats: Union['_CatID', Iterable['_CatID']] = None,
                           diversify_cats: bool = True,
                           num_imgs: int = 1,
                           forbidden_imagenet_cat_ids: Union['_ImageNetLabel', list['_ImageNetLabel']] = None,
                           top_x: int = 5,
                           seed: int = None,
                           ) -> Generator['_ImgID', None, None]:
        """Fetch a random image of a specified category.

        Args:
            only_cats (str): IDs of the categories from which to sample
            diversify_cats: if True, images are taken from maximally different set of categories
            num_imgs: the number of image ids to return
            forbidden_imagenet_cat_ids: see check_for_classes
            top_x: see check_for_classes
            seed: set this randomness seed for reproducibility if not None

        Returns:
            Tuple[Tensor, str]: Transformed image tensor and its corresponding class name.
        """
        if seed is not None:
            random.seed(seed)

        # Subset categories:
        if only_cats is not None:
            if isinstance(only_cats, str) or not isinstance(only_cats, collections.abc.Iterable):
                only_cats: list['_CatID'] = [only_cats]
            for cat_id in only_cats:
                assert cat_id in self.cat_ids, f"Requested category ID {cat_id} not in available categories. Available are: {list(self.cat_ids)}"
        cat_ids: list['_CatID'] = list(only_cats or self.cat_ids)

        # Create an iterator over categories (simple, with category restrictions, and/or diversified)
        if only_cats is None and not diversify_cats:  # just return no category restriction
            cat_ids_iterable = (None for _ in itertools.count())
        elif diversify_cats:  # cycle through the list of categories:
            cat_ids_iterable = itertools.cycle(random.sample(cat_ids, len(cat_ids)))
        else:  # randomly choose a category for each image
            cat_ids_iterable = (random.choice(cat_ids) for _ in itertools.count())

        # Randomly select images
        num_imgs_selected = 0
        for cat_id in cat_ids_iterable:
            # if the category is restricted, only choose from images with that category:
            img_ids_to_select_from = self.img_ids if cat_id is None else self.img_ids_by_cat_id[cat_id]
            if len(img_ids_to_select_from) == 0: continue

            # choose image
            img_id = random.choice(img_ids_to_select_from)

            # discard image if it contains undesired ImageNet class
            # (requires to load and evaluate the image using a DNN!)
            if forbidden_imagenet_cat_ids is not None:
                if self.check_for_classes(img_id, forbidden_imagenet_cat_ids=forbidden_imagenet_cat_ids, top_x=top_x):
                    continue

            # add image to selection
            num_imgs_selected += 1
            yield img_id
            if num_imgs_selected >= num_imgs: break

    def get_random_images(self, only_cats: Union['_CatID', Iterable['_CatID']] = None,
                          diversify_cats: bool = True,
                          num_imgs: int = 1,
                          forbidden_imagenet_cat_ids: Union['_ImageNetLabel', Sequence['_ImageNetLabel']] = None,
                          top_x: int = 5,
                          seed: int = None,
                          ) -> Generator[PIL.Image.Image, None, None]:
        img_ids = self.get_random_img_ids(only_cats=only_cats, diversify_cats=diversify_cats,
                                          num_imgs=num_imgs,
                                          forbidden_imagenet_cat_ids=forbidden_imagenet_cat_ids, top_x=top_x,
                                          seed=seed)
        return (self.load_img(img_id) for img_id in img_ids)

    def get_random_image(self, only_cats: Union['_CatID', Iterable['_CatID']] = None,
                         diversify_cats: bool = True,
                         forbidden_imagenet_cat_ids: Union['_ImageNetLabel', Sequence['_ImageNetLabel']] = None,
                         top_x: int = 5,
                         seed: int = None, ) -> PIL.Image.Image:
        return next(self.get_random_images(only_cats=only_cats, diversify_cats=diversify_cats,
                                           num_imgs=1,
                                           forbidden_imagenet_cat_ids=forbidden_imagenet_cat_ids, top_x=top_x,
                                           seed=seed))

    @property
    def timm_model(self) -> torch.nn.Module:
        import timm
        if self._timm_model is None:
            self._timm_model = timm.create_model('vit_base_patch16_224', pretrained=True)
            self._timm_model.eval()

        return self._timm_model

    @property
    def timm_preprocess(self) -> Callable:
        from torchvision import transforms
        if self._timm_preprocess is None:
            self._timm_preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return self._timm_preprocess

    def check_for_classes(self, img_id: '_ImgID',
                          forbidden_imagenet_cat_ids: Union['_ImageNetLabel', Sequence['_ImageNetLabel']] = None,
                          top_x: int = 5,
                          filtering_device: Union[str, torch.device] = None):
        """For a given image, check whether one of the given ImageNet class names is present in the image.
        Class names may be specified by their ID (position in the ImageNet class list), WordNet ID, or
        the human readable name (see IMAGENET_LABELS and IMAGENET_CLASS_NAME_BY_WORDNET_ID)
        Uses a vision transformer model to determine which ImageNet classes are in the image.

        Args:
            img_id (_ImgID): the ID of the image to check
            forbidden_imagenet_cat_ids: List of ImageNet classes to check for (see predict_top_k_imagenet_classes)
            top_x: see predict_top_k_imagenet_classes
            filtering_device: see predict_top_k_imagenet_classes
        """
        from .imagenet_labels import IMAGENET_CLASS_NAMES_BY_ID, WORDNET_ID_BY_IMAGENET_CLASS_NAME

        # build set of filtered classes:
        if forbidden_imagenet_cat_ids is None or len(forbidden_imagenet_cat_ids) == 0:
            return False
        forbidden_imagenet_cat_ids = set(forbidden_imagenet_cat_ids)

        # Try to load class specifiers in the image from cache
        class_specifiers_in_img: dict[str, list[Union[int, str]]] = \
            self._imagenet_class_specifiers_by_img_id.get(img_id, None)
        # If that fails, load image and eval it
        if class_specifiers_in_img is None:
            image: PIL.Image.Image = self.load_img(img_id)
            probabilities = self.predict_imagenet_classes(image, filtering_device)

            # sort predictions by their probability
            _, top_out_ids = torch.topk(probabilities, probabilities.size()[-1])
            classes_in_img: list[int] = [top_out_id.item() for top_out_id in top_out_ids]

            # add the WordNet IDs and the common names of the classes as well to allow specification via either of them
            class_names_in_img: list[str] = [IMAGENET_CLASS_NAMES_BY_ID[class_id] for class_id in classes_in_img]
            wordnet_ids_in_img: list[str] = [WORDNET_ID_BY_IMAGENET_CLASS_NAME[class_name] for class_name in class_names_in_img]
            class_specifiers_in_img = {'id': classes_in_img,
                                       'wordnet_id': wordnet_ids_in_img,
                                       'name': class_names_in_img,
                                       }  # TODO: use named tuple
            # add to cache
            self._imagenet_class_specifiers_by_img_id[img_id] = class_specifiers_in_img

        # Check for presence of labels that shall be filtered
        top_k_class_specifiers_in_img = [s for specifiers in class_specifiers_in_img.values() for s in specifiers[:top_x]]
        filter_class_in_top_x = any((c in top_k_class_specifiers_in_img for c in forbidden_imagenet_cat_ids))
        return filter_class_in_top_x

    def predict_imagenet_classes(self,
                                 image: PIL.Image.Image,
                                 filtering_device: Union[str, torch.device] = None
                                 ) -> torch.Tensor:
        """Return the list of ImageNet class probabilities."""
        filtering_device = filtering_device or self.device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # prepare input
        input_tensor = self.timm_preprocess(image).unsqueeze(0).to(filtering_device)

        # Perform inference
        with torch.no_grad():
            output = self.timm_model.to(filtering_device)(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        return probabilities
