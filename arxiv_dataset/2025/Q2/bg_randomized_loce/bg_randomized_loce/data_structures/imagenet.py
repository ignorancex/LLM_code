import os
from collections import defaultdict
from typing import Union, TYPE_CHECKING

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .datasets import SegmentationDataset, ImageLoader

if TYPE_CHECKING:
    from .datasets import _TransformType, _ImgID, _CatID

class ImageNetS50SegmentationDataset(SegmentationDataset):
    ALL_CAT_NAMES_BY_ID = {
        1: 'n01443537',
        2: 'n01491361',
        3: 'n01531178',
        4: 'n01644373',
        5: 'n02104029',
        6: 'n02119022',
        7: 'n02123597',
        8: 'n02133161',
        9: 'n02165456',
        10: 'n02281406',
        11: 'n02325366',
        12: 'n02342885',
        13: 'n02396427',
        14: 'n02483362',
        15: 'n02504458',
        16: 'n02510455',
        17: 'n02690373',
        18: 'n02747177',
        19: 'n02783161',
        20: 'n02814533',
        21: 'n02859443',
        22: 'n02917067',
        23: 'n02992529',
        24: 'n03014705',
        25: 'n03047690',
        26: 'n03095699',
        27: 'n03197337',
        28: 'n03201208',
        29: 'n03445777',
        30: 'n03452741',
        31: 'n03584829',
        32: 'n03630383',
        33: 'n03775546',
        34: 'n03791053',
        35: 'n03874599',
        36: 'n03891251',
        37: 'n04026417',
        38: 'n04335435',
        39: 'n04380533',
        40: 'n04404412',
        41: 'n04447861',
        42: 'n04507155',
        43: 'n04522168',
        44: 'n04557648',
        45: 'n04562935',
        46: 'n04612504',
        47: 'n06794110',
        48: 'n07749582',
        49: 'n07831146',
        50: 'n12998815',
        1000: 'ignore'}

    #DEFAULT_IMAGE_LOADER = ImageLoader(img_shape=None)

    def __init__(self,
                 imgs_path='./data/ImageNetS/ImageNetS50/train',
                 masks_path='data/ImageNetS/ImageNetS50/train-semi-segmentation',
                 *,
                 category_ids: list[int] = None,
                 category_names_by_id: dict[int, str] = None,
                 image_loader: ImageLoader = None,
                 transform: '_TransformType' = None,
                 device: Union[torch.device, str] = None,
                 ):
        """
        Args:
            imgs_path (str): Path to the directory containing the images.
            masks_path (str): Path to the directory containing the segmentation masks.
            transform (callable, optional): Optional transform to be applied
                on an image and its corresponding mask.
            device: torch device to use for image transformation operations;
                defaults to cuda if available
        """
        super().__init__(imgs_path=imgs_path,
                         category_ids=category_ids,
                         category_names_by_id=category_names_by_id,
                         image_loader=image_loader,
                         transform=transform,
                         combine_masks=True, tag="ImageNetS50",
                         device=device)
        self.masks_path = masks_path
        self.ignore_index = 1000

        self.img_ids: list[str] = []
        self._populate_img_ids()

    def _populate_img_ids(self):
        """Set values for img_ids and _img_ids_by_cat_id."""
        self._img_ids_by_cat_id: dict[str, list[str]] = defaultdict(list)
        for sub_dir in (s for s in os.listdir(self.masks_path)
                        if os.path.isdir(os.path.join(self.masks_path, s))):
            cat_ids = [cid for cid, cname in self.cat_name_by_id.items() if cname == sub_dir]

            for filename in os.listdir(os.path.join(self.masks_path, sub_dir)):

                img_id = os.path.join(sub_dir, filename.rsplit('.png', 1)[0])
                mask: torch.Tensor = self._load_seg_with_cat_ids(os.path.join(self.masks_path, f"{img_id}.png"))

                has_mask = False
                for cat_id in (c for c in cat_ids if c != self.ignore_index):
                    if (mask == cat_id).sum() > 0:
                        self._img_ids_by_cat_id[cat_id].append(img_id)
                        has_mask = True
                if has_mask: self.img_ids.append(img_id)

    def get_img_filename(self, img_id: Union[str, int]) -> str:
        return f"{img_id}.JPEG"

    def _load_seg_with_cat_ids(self, mask_path: str) -> torch.Tensor:
        """Returns a numpy segmentation mask where each entry is the applicable category id."""
        maskRGB: torch.Tensor = torch.as_tensor(TF.pil_to_tensor(Image.open(mask_path)),
                                                dtype=torch.int16, device=self.device)
        mask: torch.Tensor = maskRGB[1, :, :] * 256 + maskRGB[0, :, :]  # R+G*256
        mask[mask == 0] = 1000
        return mask

    def load_segs(self, img_id: Union[str, int]) -> dict[int, torch.Tensor]:
        mask_path = os.path.join(self.masks_path, f"{img_id}.png")
        mask: torch.Tensor = self._load_seg_with_cat_ids(mask_path)
        masks: dict[int, torch.Tensor] = {}
        for c in (c for c in self.cat_ids if c != self.ignore_index):
            c_mask: torch.Tensor = (mask == c)
            if c_mask.sum() > 0:
                masks[c] = c_mask
        return masks

    def get_cats(self, img_id: '_ImgID') -> list['_CatID']:
        return self.cat_ids_by_img_id[img_id]



# TODO: add other ImageNetS variants