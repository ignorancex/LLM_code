import os
from collections import defaultdict
from typing import Optional, TYPE_CHECKING, Any, Union

from .datasets import SegmentationDataset, ImageLoader

if TYPE_CHECKING:
    import torch

if TYPE_CHECKING:
    _CatID = str
    _ImgID = str


class FolderClassificationDataset(SegmentationDataset):
    ALL_CAT_NAMES_BY_ID = {}
    DEFAULT_IMAGE_LOADER = ImageLoader(img_shape=None)
    DEFAULT_IMGS_PATH = None
    EXCLUDE_IMGS = []

    def __init__(self,
                 imgs_path: str = DEFAULT_IMGS_PATH,
                 *,
                 transform: Optional['_TransformType'] = None,
                 image_loader: ImageLoader = None,
                 device: Union['torch.device', str] = None,
                 category_ids: list[str] = None,
                 exclude_files: list[str] = None,
                 ):
        """
        Custom class to load a random image from a specific class in the Places Dataset.

        Args:
            imgs_path (str): Path to the root directory containing class folders.
                            The structure should be root/class_name/image.jpg
            exclude_files: paths of images to exclude, relative to imgs_path
        """
        # Create a mapping from class names to image file paths
        self._img_infos_by_cat_id: dict['_CatID', list[dict[str, Any]]] = \
            self._build_class_to_images(imgs_path, allowed_categories=category_ids,
                                        exclude=[os.path.join(imgs_path, frelpath) for frelpath in (exclude_files or self.EXCLUDE_IMGS)])
        self._img_infos_by_img_id: dict['_ImgID', dict[str, Any]] = {
            info['name']: info for infos in self._img_infos_by_cat_id.values() for info in infos}
        self.img_ids: list['_ImgID'] = list(self._img_infos_by_img_id.keys())
        self._cat_id_by_img_id: dict['_ImgID', '_CatID'] = {
            info['name']: cat_id for cat_id, infos in self._img_infos_by_cat_id.items() for info in infos}
        super().__init__(
            imgs_path=imgs_path,
            category_names_by_id={c: c for c in self._cat_id_by_img_id.values()},
            transform=transform, image_loader=image_loader,
            combine_masks=False,
            device=device,
        )

    @classmethod
    def _build_class_to_images(cls, imgs_path: str, allowed_categories: list['_CatID'] = None, exclude: list[str] = None) -> dict[
        '_CatID', list[dict[str, Any]]]:
        """
        Builds a dictionary mapping class names to lists of image paths.

        Returns:
            dict: {class_name: [list of image paths]}
        """
        class_to_images: dict['_CatID', list[dict[str, Any]]] = defaultdict(list)
        sub_dirs = [d for d in os.listdir(imgs_path) if os.path.isdir(os.path.join(imgs_path, d))]
        for sub_dir in sub_dirs:
            class_dirs = os.listdir(os.path.join(imgs_path, sub_dir))
            for class_dir in class_dirs:

                # filtering
                if allowed_categories is not None and class_dir not in allowed_categories:
                    continue

                # Images within parent dir
                class_to_images |= cls._get_img_info_in_dir(
                    os.path.join(imgs_path, sub_dir, class_dir),
                    class_id=class_dir,
                    exclude=exclude)

                # Images within subdirs
                subclass_dirs = [item for item in os.listdir(os.path.join(imgs_path, sub_dir, class_dir))
                                 if not os.path.isfile(os.path.join(imgs_path, sub_dir, class_dir, item))]
                for subclass_dir in subclass_dirs:

                    # filtering
                    if allowed_categories is not None and subclass_dir not in allowed_categories:
                        continue

                    class_to_images |= cls._get_img_info_in_dir(
                        os.path.join(imgs_path, sub_dir, class_dir),
                        class_id=class_dir + '+' + subclass_dir,
                        exclude=exclude,
                        )

        return class_to_images

    @staticmethod
    def _get_img_info_in_dir(class_dir_path: str, class_id: str, exclude: list[str] = None,
                             ) -> dict['_CatID', list[dict[str, Any]]]:
        class_to_images: dict['_CatID', list[dict[str, Any]]] = defaultdict(list)
        image_names = [item for item in os.listdir(class_dir_path)
                       if os.path.isfile(os.path.join(class_dir_path, item))
                       and not os.path.join(class_dir_path, item) in (exclude or [])
                       and os.stat(os.path.join(class_dir_path, item)).st_size != 0]
        for image_name in image_names:
            class_to_images[class_id].append({
                'name': image_name,
                'path': os.path.join(class_dir_path, image_name),
            })
        return class_to_images

    def get_cats(self, img_id) -> list['_CatID']:
        return [self._cat_id_by_img_id[img_id]]

    def get_img_filename(self, img_id: '_ImgID') -> str:
        return os.path.basename(self.get_img_path(img_id))

    def get_img_path(self, img_id: '_ImgID') -> str:
        return self._img_infos_by_img_id[img_id]['path']

    def load_segs(self, img_id: '_ImgID') -> dict[str, bool]:
        cat_ids_in_img = self.get_cats(img_id)
        return {c: (c in cat_ids_in_img) for c in self.cat_ids}

    def find_broken_relpaths(self) -> list[str]:
        """Check for each image in the dataset whether it can be loaded, and return list failures.
        For each image, the relative filepath in the folder is returned.

        The resulting list can be appended to `EXCLUDE_IMGS` blacklist.
        """
        # imports only needed for this specific validation step
        from tqdm import tqdm
        from pathlib import Path

        broken_ids = []
        for i in tqdm(self.img_ids):
            try:
                self[i]
            except:
                broken_ids.append(i)

        broken_relpaths = [str(Path(self.get_img_path(p)).relative_to(self.imgs_path)) for p in broken_ids]
        return broken_relpaths

class PlacesDataset(FolderClassificationDataset):
    DEFAULT_IMAGE_LOADER = ImageLoader(img_shape=(256, 256))  # Places dataset only has images of size (256, 256)
    DEFAULT_IMGS_PATH = './data/places_205_kaggle'
    EXCLUDE_IMGS = ('a/aqueduct/gsun_3eb12e26b495b4b8d2483e9f25609d1c.jpg',
                    'c/cottage_garden/gsun_6c03ae81ab181303144ed2bda2757c89.jpg',
                    't/tower/gsun_7853779f1b1cfe2dc31cc075733306d5.jpg',
                    's/slum/gsun_bc8fb5c54ce0cb411f6cf52914690d53.jpg',
                    'a/abbey/gsun_686bc828cd43897e2e6f03d4fd7807f5.jpg',
                    'a/abbey/gsun_68c7061f8f755ff7af217bf4d3efcfee.jpg',
                    'a/abbey/gsun_690b2dd3100da65fe3beb703c46465cb.jpg',
                    'a/abbey/gsun_6953f3e6ad8ec7a8ea007ba38591ed43.jpg',
                    'c/courthouse/gsun_130d717ef1d76ed0973e4885ccc96bd2.jpg',
                    )
    """List of images that are broken in the Places 205 Kaggle dataset.
    
    Can be found using find_broken_relpaths().
    """


class SyntheticBackgroundsDataset(FolderClassificationDataset):
    DEFAULT_IMGS_PATH = './data/synthetic_backgrounds'



if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description="""Find any not yet excluded broken images in the PlacesDataset at --imgs-path.""")
    parser.add_argument('--imgs-path', default="./data/places_205_kaggle")
    args = parser.parse_args()
    imgs_path = args.imgs_path
    data = PlacesDataset(imgs_path=imgs_path)
    print("Dataset ready.")

    broken_relpaths = data.find_broken_relpaths()
    print("Broken relative image paths not yet excluded:")
    pprint(broken_relpaths)
