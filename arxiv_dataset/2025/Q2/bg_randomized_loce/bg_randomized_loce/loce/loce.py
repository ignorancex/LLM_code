import os
import random
from collections.abc import Generator
from typing import Dict, List, Iterable, Tuple, Union

import numpy as np

from .loce_utils import get_projection
from ..utils.files import write_pickle, read_pickle
from ..utils.logging import log_warn


class LoCE:
    """
    Guided Concept Projection Vector (LoCE)

    Storage for single optimized instance
    """
    def __init__(self,
                 loce: np.ndarray,
                 loss: float,
                 projection: np.ndarray
                 ) -> None:
        """
        Args:
            loce (np.ndarray): projection vector with unit importance weights
            loss (float): final loss after LoCE optimization
            projection (np.ndarray): projection of LoCE with original activations used for optimization
        """
        
        self.loce = loce
        self.loss = loss
        self.projection = projection

    def project(self, activations: np.ndarray) -> np.ndarray:
        """
        Args:
            activations (np.ndarray): activations to project with stored LoCE vector
        """
        return get_projection(self.loce, activations)


class LoCEMultilayerStorage:
    """
    Storage for LoCEs obtained in multiple layers.

    Contains LoCE instances and meta information (layer, original image, segmentation, etc.)
    """

    def __init__(self,
                 image_path: str,
                 image_predictions: np.ndarray,
                 segmentation: np.ndarray,
                 segmentation_category_id: int
                 ) -> None:
        """        
        Args:
            image_path (str = None): path to image used for optimization
            image_predictions (np.ndarray = None): predictions made for image used for optimization
            segmentation (np.ndarray = None): original segmentation used for optimization
            segmentation_category_id (int): segmentation class (object class, etc.)
        """
        self.image_path = image_path
        self.image_predictions = image_predictions
        self.segmentation = segmentation
        self.segmentation_category_id = segmentation_category_id

        # init storage, make it instance specific
        self.loce_storage: Dict[str, LoCE] = {}  # {layer: loce}

    def set_loce(self,
                 layer: str,
                 loce: LoCE
                 ) -> None:
        """
        Add LoCE instance to storage

        Args:
            layer (str): layer of LoCE
            loce (LoCE): instance of LoCE
        """
        self.loce_storage[layer] = loce

    def get_loce(self, layer: str) -> LoCE:
        """
        Return LoCE from given layer

        Args:
            layer (str): retrieval layer

        Returns:
            loce (LoCE): retrieved LoCE instance
        """
        return self.loce_storage[layer]

    def get_storage_layers(self) -> List[str]:
        """
        Return all layer names contained in storage

        Returns:
            storage_layers (List[str]): list of storage layer names
        """
        return list(self.loce_storage.keys())
    
    def get_multilayer_loce(self, layers_to_concatenate: List[str]) -> np.ndarray:
        """
        Return multi-layer LoCE (MLLoCE)

        Args:
            layers_to_concatenate (List[str]): list of layers for LoCE

        Returns:
            multilayer_loce (np.ndarray): stacked LoCEs for multiple layers
        """
        loces = [self.get_loce(l).loce for l in layers_to_concatenate]

        multilayer_loce = np.concatenate(loces, axis=0)

        return multilayer_loce


class LoCEMultilayerStorageSaver:

    def __init__(self, working_directory: str) -> None:
        """
        Args:
            working_directory (str): working directory for input-output of LoCEMultilayerStorages
        """
        self.working_directory = working_directory

    def get_loce_storage_path_for_img_name(self,
                                           image_name: str,
                                           category_id: int) -> tuple[str, str]:
        """
        Generate image for LoCE storage from image and category_id

        Args:
            image_name (str): image name
            category_id (int): category of object used for optimization
        Returns: tuple of (file_path.pkl, error_file_path.err)
        """
        image_name_no_ext = image_name.split(".")[0]

        image_name_no_ext_with_prefix = '_'.join([str(category_id), image_name_no_ext])

        image_path_no_ext_with_prefix = os.path.join(self.working_directory, image_name_no_ext_with_prefix)

        out_path_pkl = image_path_no_ext_with_prefix + ".pkl"  # correct file name
        out_path_err = image_path_no_ext_with_prefix + ".err"  # error file name

        return out_path_pkl, out_path_err

    def save(self, loce_storage: LoCEMultilayerStorage, save_path: str = None) -> None:
        """
        Saves loce_storage to {save_path}

        Args:
            loce_storage (LoCEMultilayerStorage): storage to save

        Kwargs:
            save_path (str): saving path, if not given - evaluate path with self.get_loce_storage_path_for_img_name()
        """
        if save_path is None:
            save_path = self.get_loce_storage_path_for_img_name(os.path.basename(loce_storage.image_path), loce_storage.segmentation_category_id)

        write_pickle(loce_storage, save_path)

    def get_idxs_missing(self, img_filenames: Iterable[str], category_id: Union[str, int]) -> Generator[int]:
        """Provide a generator that yields indices from img_filenames for items of category_id not yet in this storage."""
        existing_storages: list[str] = os.listdir(self.working_directory)
        future_storage_names: Generator[str] = (
            os.path.basename(self.get_loce_storage_path_for_img_name(img_filename, category_id)[0])
            for img_filename in img_filenames)
        for idx, future_storage_name in enumerate(future_storage_names):
            if future_storage_name in existing_storages:
                continue
            yield idx


class LoCEMultilayerStorageDirectoryLoader:

    def __init__(self,
                 working_directory: str,
                 seed: int = None,
                 min_seg_area: float = 0.0,
                 max_seg_area: float = 1.0
                 ) -> None:
        """
        Args:
            working_directory (str): working directory for input-output of LoCEMultilayerStorages

        Kwargs:
            seed (int = None) seed for data sampling
            min_seg_area (float): minimal allowed segmentation area, LoCEs with smaller segmentation will be ignored
            max_seg_area (float): maximal allowed segmentation area, LoCEs with larger segmentation will be ignored;
                clipped to max. 1.0; applies filtering if max_seg_area - min_seg_area < min_seg_area
        """
        self.working_directory = working_directory
        self.seed = seed
        random.seed(seed)

        # check and apply segmentation-size based filtering constraints
        if not (max_seg_area > min_seg_area >= 0.0 and max_seg_area > 0.0):
            log_warn(f"(at LoCEMultilayerStorageDirectoryLoader): received invalid min_seg_area ({min_seg_area}) and max_seg_area ({max_seg_area}) combination; ignoring constraint.")
            min_seg_area, max_seg_area = 0.0, 1.0
        self.min_seg_area: float = max(0., min_seg_area)
        self.max_seg_area: float = min(1., max_seg_area)

        self.pkl_file_paths = self._select_pkl_files()

        # only filter if there is a true constraint
        if not np.isclose(self.max_seg_area - self.min_seg_area, 1.):
            self._filter_files_by_segmentation_size()

    def _select_pkl_files(self) -> List[str]:
        """
        Find file names of pickles with LoCEs

        Returns:
            selected_files (List[str]): filtered files by extension (only .pkl)
        """
        all_files_in_dir = os.listdir(self.working_directory)

        selected_files = []

        for fp in all_files_in_dir:

            if fp.split('.')[-1] == 'pkl':
                selected_files.append(fp)
            else:
                continue

        return selected_files
    
    def _filter_files_by_segmentation_size(self) -> None:
        """
        Additional filtering of LoCEs by min and max segmentation areas
        """
        filtered_files = []

        for fp in self.pkl_file_paths:
            loce: LoCEMultilayerStorage = read_pickle(os.path.join(self.working_directory, fp))

            seg_mask = loce.segmentation
            seg_area = seg_mask.sum() / seg_mask.size

            if (seg_area > self.min_seg_area) and (seg_area < self.max_seg_area):
                filtered_files.append(fp)
            
        self.pkl_file_paths = filtered_files

    def _categorize_files(self,
                          ) -> Dict[int, List[str]]:

        """
        Find file names of pickles with LoCEs

        Returns:
            (Dict[int, List[str]]) dictionary with LoCE file names - {tag_id: [file_name]}
        """
        file_paths = self.pkl_file_paths
        
        if not self.seed is None:
            random.shuffle(file_paths)

        selected_files = dict()

        for fp in file_paths:
            base_fn = os.path.basename(fp)
            category, name = base_fn.split('_', 1)
            
            # init key for current category
            if int(category) not in list(selected_files.keys()):
                selected_files[int(category)] = []
            
            selected_files[int(category)].append(fp)

        return selected_files
    
    def load(self,
             allowed_categories: Iterable[int] = None
             ) -> Dict[int, List[LoCEMultilayerStorage]]:
        """
        Get dictionary of LoCEMultilayerStorages lists per category

        Kwargs:
            allowed_categories (Iterable[int]): load only LoCE storage of allowed categories ids, None to load all

        Returns:
            loce_storages_dict (Dict[int, List[LoCEMultilayerStorage]]) LoCE storages per category
        """

        loce_file_names = self._categorize_files()

        if allowed_categories is not None:
            loce_storages_dict = {category: [] for category in sorted(allowed_categories)}
        else:
            loce_storages_dict = {category: [] for category in sorted(loce_file_names.keys())}

        for category in loce_storages_dict.keys():

            if category not in loce_file_names:
                continue

            for file_name in loce_file_names[category]:
                loce_storages_dict[category].append(read_pickle(os.path.join(self.working_directory, file_name)))

        #if allowed_categories is not None:
        #    loce_storages_dict = {t: [read_pickle([os.path.join(self.working_directory, f)]) for f in loce_file_names[t]] for t in allowed_categories}
        #else:
        #    loce_storages_dict = {t: [read_pickle([os.path.join(self.working_directory, f)]) for f in loce_file_names[t]] for t in sorted(loce_file_names.keys())}

        return loce_storages_dict
    
    def load_train_test_splits(self,
                               allowed_categories: Iterable[int] = None,
                               train_size: float = 0.8
                               )  -> Tuple[Dict[int, List[LoCEMultilayerStorage]], Dict[int, List[LoCEMultilayerStorage]]]:
        """
        Get 2 dictionaries of LoCEMultilayerStorages lists per category, where one dict is 'train' part, second one is 'test' part

        Kwargs:
            allowed_categories (Iterable[int] = None): load only LoCE storage of allowed categories ids, None to load all
            test_size (float = 0.2): size of 'test' split

        Returns:
            loce_storages_dict_train (Dict[int, List[LoCEMultilayerStorage]]) 'train' lists of LoCE storages per category
            loce_storages_dict_test (Dict[int, List[LoCEMultilayerStorage]]) 'test' lists of LoCE storages per category
        """
        loce_storages_dict = self.load(allowed_categories)

        # init output dicts
        loce_storages_dict_train = dict()
        loce_storages_dict_test = dict()

        # split each category separately
        for k in sorted(loce_storages_dict.keys()):
            loce_storages_k = loce_storages_dict[k]
            split_idx = int(len(loce_storages_k) * train_size)
            loce_storages_dict_train[k] = loce_storages_k[:split_idx]
            loce_storages_dict_test[k] = loce_storages_k[split_idx:]

        return loce_storages_dict_train, loce_storages_dict_test

