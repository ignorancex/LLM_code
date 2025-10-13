from __future__ import annotations

import itertools
import sys

# Handle backwards compatibility to python 3.11:

if sys.version_info >= (3, 12):
    from itertools import batched as itertools_batched
else:

    T = TypeVar("T")

    def itertools_batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

import random
from collections import defaultdict

from torch import OutOfMemoryError

from ..data_structures.datasets import SegmentationDataset, ImageLoader
from ..data_structures.mscoco import MSCOCOSegmentationDataset
from ..utils.logging import init_logger, log_info

init_logger()

import torch
from torch import Tensor
from torch.optim import AdamW
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F

from .loce import LoCE, LoCEMultilayerStorage, LoCEMultilayerStorageSaver
from .loce_utils import get_projection, LoCEActivationsTensorExtractor
from ..utils.files import mkdir
import PIL.Image

import os
from typing import Dict, Iterable, Tuple, List, Literal, Sequence, Callable, Optional, Union, TypeVar
from collections.abc import Generator

EPSILON = 0.0001
LoCEOptimizationCriterionType = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


class TorchCustomLoCEBatchOptimizer:

    def __init__(self,
                 loce_init: Literal["zeros", "ones", "random_uniform", "random_normal"] = "zeros",
                 seed: int = None,
                 objective_type: Literal["bce", "mse", "mae", "proper_bce"] = "bce",
                 denoise_activations: bool = False,
                 lr: float = 0.1,
                 epochs: int = 50,
                 num_acts_per_loce: Optional[int] = 1,
                 num_loces_per_batch: int = None,
                 device: Optional[Union[str, torch.device]] = None,
                 ) -> None:
        """
        Args:
            loce_init (str: Literal["zeros", "ones", "random_uniform", "random_normal"]): initialize loce vector with torch.zeros(), torch.ones(), torch.rand() or torch.randn()
            seed (int = None): seed for "random_uniform", "random_normal" loce_init
            objective_type (str): optimization objective
            denoise_activations (bool): denoising flag
            lr (float): default learning rate
            epochs (int): default number of epochs for training a LoCE
            num_acts_per_loce (int): number of activations in a sequence of activations to reserve for one LoCE;
                the activations and segmentations respectively must then be arranged as
                [act1_loce1, act2_loce1, ..., act1_loce2, act2_loce2, ...].
                Set to None to always consume the complete batch for one LoCE (e.g., for Net2Vec training).
            num_loces_per_batch (int): number of LoCEs to train in parallel
            device: the device to run the optimization on (defaults to 'cuda' if available)
        """
        self.seed: int = seed
        if not self.seed is None:
            torch.manual_seed(self.seed)
        self.loce_init = loce_init
        self.objective_type = objective_type
        self.denoise_activations = denoise_activations

        self.lr: float = lr
        self.epochs: int = epochs
        self.num_acts_per_loce: Optional[int] = num_acts_per_loce
        self.num_loces_per_batch: Optional[int] = num_loces_per_batch

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # modified BCE from Net2Vec paper
    @staticmethod
    def _objective_proper_bce(projection_vector: Tensor,  # 1     x channels x 1 x 1
                              target: Tensor,             # batch x 1        x w x h
                              acts_tensor: Tensor,        # batch x channels x w x h
                              alphas_batch: Tensor,       # batch x 1        x 1 x 1
                              ) -> Tensor:
        """Implementation of standard binary cross-entropy with weighting alphas_batch of the positive class.
        Assuming the target has values in [0,1], with <0.5 meaning (rather) positive, and vice versa."""
        # get the concept vector predictions of dim: batch x 1 x w x h
        pred_logits: Tensor = (acts_tensor * projection_vector).sum(dim=1, keepdim=True)

        # defining weights as required by torch BCE function:
        batch_size = target.size(0)
        alphas = alphas_batch.view(batch_size, 1, 1, 1)
        betas = 1 - alphas
        # include both the alphas as well as fuzziness of the target into the weighting
        target_bin = (target > 0.5).float()
        weights = (alphas * target_bin * target) + \
                  (betas * (1 - target_bin) * (1 - target))

        # actual weighted BCE calculation using torch function:
        bces = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits, target_bin, weight=weights, reduction='none')
        # reduce to one value per image in batch
        bce = torch.sum(bces, dim=[1, 2, 3])

        return bce

    # modified BCE from Net2Vec paper
    @staticmethod
    def _objective_bce(projection_vector: Tensor,
                       target: Tensor,
                       acts_tensor: Tensor,
                       alphas_batch: Tensor):

        weighted_activations = (acts_tensor * projection_vector).sum(dim=1, keepdim=True)

        preds = torch.sigmoid(weighted_activations)

        batch_size = preds.size(0)

        alpha = alphas_batch.view(batch_size, 1, 1, 1)
        beta = 1 - alpha

        # loss per sample
        pseudo_bce = -1. / batch_size * torch.sum(
            alpha * torch.mul(target, preds) + beta * torch.mul(1 - target, 1 - preds), dim=[1, 2, 3])

        return pseudo_bce

    @staticmethod
    def _objective_mae_reg(projection_vector: Tensor,
                           mask_tensor_binary: Tensor,
                           acts_tensor: Tensor,
                           alpha: float = None):

        weighted_activations = (acts_tensor * projection_vector).sum(dim=1, keepdim=True)

        preds = torch.sigmoid(weighted_activations)

        mae = (torch.abs(mask_tensor_binary - preds)).mean()

        l1 = torch.norm(projection_vector, 1) / len(projection_vector)
        l2 = torch.norm(projection_vector, 2) / len(projection_vector)

        regularization = (l1 + l2)

        return mae + regularization

    @staticmethod
    def _objective_mse_reg(projection_vector: Tensor,
                           mask_tensor_binary: Tensor,
                           acts_tensor: Tensor,
                           alpha: float = None):

        weighted_activations = (acts_tensor * projection_vector).sum(dim=1, keepdim=True)

        preds = torch.sigmoid(weighted_activations)

        mae = ((mask_tensor_binary - preds) ** 2).mean()

        l1 = torch.norm(projection_vector, 1) / len(projection_vector)
        l2 = torch.norm(projection_vector, 2) / len(projection_vector)

        regularization = (l1 + l2)

        return mae + regularization

    @classmethod
    def _get_loces(cls,
                   baseline_masks: Sequence[Tensor],
                   acts: Sequence[Tensor],
                   num_loces_per_batch: Optional[int] = None,
                   lr: float = 0.1,
                   epochs: int = 50,
                   num_acts_per_loce: int = 1,
                   objective_type: Literal['mae', 'mse', 'proper_bce', 'bce'] = 'bce',
                   loce_init: Union[np.ndarray, torch.Tensor, Literal['ones', 'random_uniform', 'random_normal', 'zeros']] = 'zeros',
                   device: Optional[Union[str, torch.device]] = None,
                   ) -> np.ndarray:
        """
        Optimize LoCEs in parallel.
        Each LoCE is optimized on num_acts_per_loce activation maps, and
        num_loces_per_batch are optimized in parallel.
        The acts must be arranged as [act1_loce1, act2_loce1, ..., act1_loce2, act2_loce2, ...], and masks analogously.
        Each mask must have dimensionality [1 x] width x height, each act map channels x width x height.

        Args:
            baseline_masks (Tensor): baseline segmentation
            acts (Tensor): activations of sample
        
        Kwargs:
            batch_size (int): batch size (i.e., number of LoCEs simultaneously trained); defaults to all
            lr (float): learning rate of optimizer
            epochs (int): optimization epochs
            num_acts_per_loce (int): number of activation maps reserved for training per one LoCE;
                len(acts) must be a multiple of this; set to None to set it to len(acts).
            loce_init (str): how to init the loce vector(s); if a tensor, use this to initialize the loce vector;
                else use the specified init strategy.
        
        Returns:
            (Dict[str, Any]) of optimization results
        """

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get the dimensionalities and batch sizes
        b: int = len(acts)
        c, w, h = acts[0].shape
        num_acts_per_loce: int = num_acts_per_loce or b
        # Either give all loces equal num of activation maps, or let it be only one:
        assert b % num_acts_per_loce == 0 or b < num_acts_per_loce, \
            f"activation map batch size must be a multiple of num_acts_per_concept_vector ({num_acts_per_loce}) but was {b}."
        num_loces: int = max(1, b // num_acts_per_loce)  # how many loces can be trained from the given acts (at least one)
        num_loces_per_batch: int = min(num_loces_per_batch, num_loces) if num_loces_per_batch is not None else num_loces
        num_acts_per_batch: int = num_loces_per_batch * num_acts_per_loce

        acts_device: Sequence[Tensor] = acts
        seg_masks_device: Sequence[Tensor] = baseline_masks

        criterion: LoCEOptimizationCriterionType = cls._lookup_loce_objective(objective_type)

        if isinstance(loce_init, str):
            loce_init_fn: Callable[[int, ...], Tensor] = cls._lookup_loce_init_fn(loce_init)
        else:
            loce_init: torch.Tensor = torch.as_tensor(loce_init, device=device, dtype=torch.float)
            assert loce_init.numel() % c == 0, \
                f"Received init vector with wrong number of elements: expected multiple of {c}, but got {loce_init.numel()}"
            loce_init = loce_init.reshape(-1, c, 1, 1)
            loce_init_fn = lambda *_, **__: loce_init

        loces = []

        start_idx = 0
        while start_idx < len(acts_device):
            end_idx: int = min(start_idx + num_acts_per_batch, len(acts_device))
            selected_slice = slice(start_idx, end_idx)
            start_idx = end_idx

            acts_batch: Tensor = torch.as_tensor(acts_device[selected_slice]).float().to(device)
            mask_batch: Tensor = torch.as_tensor(seg_masks_device[selected_slice]).float().to(device)
            if len(mask_batch.size()) < 4:
                mask_batch = mask_batch.unsqueeze(1)

            # for balancing background and foreground pixels (per sample)
            # alpha = 1 - (num_fg_pixels / num_pixels)
            alphas_batch: Tensor = (1. - (mask_batch > 0).float().mean(dim=(-1, -2))).to(device)

            loces_batch: Tensor = loce_init_fn((max(1, len(acts_batch)//num_acts_per_loce), c, 1, 1), device=device)
            loces_batch.requires_grad_()

            # repeat each loce by num_acts_per_loce, i.e., [loce1, loce2, ...] -> [loce1, loce1, ..., loce2, loce2, ...]
            # this gives loce_batch_spreaded.size() == [len(selected_idxs), c, 1, 1]
            loces_batch_spreaded: Tensor = \
                loces_batch.unsqueeze(1).expand(-1, min(num_acts_per_loce, len(acts_batch)), -1, -1, -1).reshape(-1, c, 1, 1) \
                if num_acts_per_loce > 1 else loces_batch

            opt = AdamW([loces_batch], lr=lr)

            #batch_losses = []

            for i in range(epochs):

                per_sample_loss = criterion(loces_batch_spreaded, mask_batch, acts_batch, alphas_batch)

                opt.zero_grad()
                per_sample_loss.mean().backward()
                opt.step()

            loces.append(loces_batch.detach().squeeze().cpu())
            #with torch.no_grad():
                    #loce.clamp_(None, None)

        #print(f"\tLoss ({init_fn}):", result['fun'])
        return torch.vstack(loces).cpu().numpy()

    @classmethod
    def _lookup_loce_objective(cls, objective_type: Literal['mae', 'mse', 'proper_bce', 'bce']) -> LoCEOptimizationCriterionType:
        # select loce optimization objective
        if objective_type == "mae":
            criterion = cls._objective_mae_reg
        elif objective_type == "mse":
            criterion = cls._objective_mse_reg
        elif objective_type == "proper_bce":
            criterion = cls._objective_proper_bce
        else:
            criterion = cls._objective_bce
        return criterion

    @staticmethod
    def _lookup_loce_init_fn(loce_init: Literal['ones', 'random_uniform', 'random_normal', 'zeros']
                             ) -> Callable[[int, ...], Tensor]:
        """Return the respective init function for the init specifier."""
        # loce init function
        if loce_init == "ones":
            loce_init_fn = torch.ones
        elif loce_init == "random_uniform":
            loce_init_fn = torch.rand
        elif loce_init == "random_normal":
            loce_init_fn = torch.randn
        else:
            loce_init_fn = torch.zeros
        return loce_init_fn

    def optimize_loces(self,
                       segmentations: Sequence[Tensor],
                       activations: Dict[str, Sequence[Tensor]],
                       init_with: dict[str, Optional[Union[torch.Tensor, np.ndarray]]] = None,
                       epochs: int = None,
                       ) -> Dict[str, np.ndarray]:
        """
        Get prototypes of LoCEs for a single sample in all given layers.
        Each LoCE is optimized on num_acts_per_loce activation maps, and
        num_loces_per_batch are optimized in parallel.
        The acts must be arranged as [act1_loce1, act2_loce1, ..., act1_loce2, act2_loce2, ...], and masks analoguesly.
        Each mask must have dimensionality width x height, each act map channels x width x height.


        Args:
            segmentations (Tensor[B,W,H]): segmentation masks (reshaped)
            activations (Dict[str, Tensor[B,C,W,H]]): per-layer dictionary of activations (reshaped)

        Returns:
            (Dict[str, np.ndarray]) per-layer lists of LoCEs
        """
        loces: dict[str, np.ndarray] = {l: None for l in activations.keys()}

        for layer, acts in activations.items():

            #if self.denoise_activations:
            #    cutoffs = compute_quantile_cutoffs(acts_current)
            #    acts_current = threshold_activations(acts_current, cutoffs)
            loce_init = init_with.get(layer, None) if init_with is not None else None
            loce_init = loce_init if loce_init is not None else self.loce_init

            concept_vectors: np.ndarray = self._get_loces(
                segmentations, acts,
                num_loces_per_batch=self.num_loces_per_batch, num_acts_per_loce=self.num_acts_per_loce,
                lr=self.lr, epochs=epochs or self.epochs,
                objective_type=self.objective_type, loce_init=loce_init,
                device=self.device,
            )

            loces[layer] = concept_vectors

        return loces


class LoCEOptimizationEngine:

    def __init__(self,
                 batch_optimizer: TorchCustomLoCEBatchOptimizer,
                 activations_extractor: LoCEActivationsTensorExtractor,
                 datasets: list[SegmentationDataset],
                 out_base_dir: str = "data/mscoco2017val/processed/loces",
                 n_imgs_per_category: int = 500,
                 target_shape: Tuple[int, int] = (100, 100),
                 batch_size: Optional[int] = 64,
                 ) -> None:
        """
        Args:
            batch_optimizer (TorchCustomLoCEBatchOptimizer): batch optimizer
            activations_extractor (LoCEActivationsTensorExtractor): activations extractor
            datasets: the datasets to (separately) run optimization on
            target_shape: Common target shape for activation maps and segmentation masks.
            batch_size: number of images simultaneously loaded and fed to optimization; all available if set to None

        Kwargs:
            out_base_dir (str = "./data/mscoco2017val/processed/loces"): base directory for outputs, subdirs will be created
            processor (BaseImageProcessor = None): Processor from Hugging Face
        """
        self.batch_optimizer: TorchCustomLoCEBatchOptimizer = batch_optimizer
        self.activations_extractor: LoCEActivationsTensorExtractor = activations_extractor
        self.propagator_tag: str = activations_extractor.propagator_tag.lower()

        self.datasets: list[SegmentationDataset] = datasets
        """Default list of datasets to (separately) run optimization on."""

        self.out_base_dir = out_base_dir
        """Default base directory for saving optimization results."""

        self.n_imgs_per_category: Optional[int] = n_imgs_per_category
        """If set, only the first n_imgs_per_category images are retrieved and used for training per category."""

        self.target_shape = target_shape
        """Common target shape for activation maps and segmentation masks."""

        self.batch_size: Optional[int] = batch_size

        # Ensure any required base directories exist:
        mkdir(out_base_dir)

    @property
    def device(self) -> Union[str, torch.device]:
        return self.batch_optimizer.device

    def run_optimization_all_segmenters(self,
                                        batch_size: int = None,
                                        verbose: bool = False
                                        ) -> None:
        """
        LEGACY METHOD.
        Perform optimization on all datasets.
        """
        self.run_optimization(batch_size=batch_size, verbose=verbose)


    def run_optimization(self,
                         *,
                         batch_size: int = None,
                         out_dir: str = None,
                         verbose: bool = False,
                         datasets: list[SegmentationDataset] = None,
                         overwrite: bool = False,
                         ) -> None:
        """
        Perform optimization. Results are saved to out_dir

        Args:
            batch_size (int): batch size (how many LoCEs are trained in parallel)
            out_dir (str, optional): fix the output directory; defaults to self.out_base_dir/loce_{dataset.tag}_{self.propagator_tag}
            verbose (bool, optional): whether to show progress bar
            datasets (list[SegmentationDataset]): (sub)set of segmentation datasets the optimization shall be applied to
            overwrite (bool): whether to find and skip already existing loce storages of image IDs
        """
        datasets = datasets or self.datasets

        # analyze each category
        datasets = tqdm(datasets, desc=f"{self.propagator_tag}", leave=False) if len(datasets) > 1 else datasets  # some progress logging
        for dataset in datasets:
            assert len(dataset.cat_ids) == 1, \
                "Found a dataset covering more than one category ID. This is currently not supported."
            category_id = dataset.cat_ids[0]

            # get saver
            out_dir = out_dir or os.path.join(self.out_base_dir, f'loce_{dataset.tag}_{self.propagator_tag}')
            mkdir(out_dir)
            loce_saver = LoCEMultilayerStorageSaver(out_dir)

            # get n images to optimize (avoids repetition if already was optimized)
            img_ids_to_optimize: list[str] = dataset.img_ids
            if len(img_ids_to_optimize) == 0:
                if verbose: log_info(f"Dataset {dataset.tag}, {category_id} empty.")
                continue

            if not overwrite:
                img_ids_to_optimize = self._get_img_ids_not_in_storage(
                    dataset=dataset, img_ids=img_ids_to_optimize, category_id=category_id, saver=loce_saver,
                    n_imgs=self.n_imgs_per_category)
                if len(img_ids_to_optimize) == 0:
                    if verbose: log_info(f"Skipping {self.propagator_tag}, {dataset.tag}, {category_id}. "
                                         f"Results exist from previous runs in {out_dir}.")
                    continue
            else:
                img_ids_to_optimize =  list(itertools.islice(img_ids_to_optimize, self.n_imgs_per_category))

            # actual optimization
            self._optimize_one_category_batchwise(dataset, img_ids_to_optimize, loce_saver, batch_size)

            # if verbose:
            #     # output stats for this category
            #     LoCEMultilayerStorageStats(out_dir, min_seg_area=0.0, max_seg_area=1.0).stats_one_category(category_id)

    def _optimize_one_category_batchwise(self,
                                         dataset: SegmentationDataset,
                                         img_ids_to_optimize: List[str],
                                         loce_saver: LoCEMultilayerStorageSaver,
                                         batch_size: int = None,
                                         ) -> None: #List[LoCEMultilayerStorage]:
        """
        Perform optimization. Results are saved to: self.out_base_dir/loce_{segmenter_tag}_{self.propagator_tag}
        """
        category_id = dataset.cat_ids[0]
        batch_size = batch_size or self.batch_size or len(img_ids_to_optimize)

        # running in batch
        #all_loce_storages: list[LoCEMultilayerStorage] = []
        start_idx = 0
        while start_idx < len(img_ids_to_optimize):

            # get batch of images
            batch_imgs = img_ids_to_optimize[start_idx:min(len(img_ids_to_optimize), start_idx + batch_size)]
            start_idx = start_idx + batch_size

            imgs_pil, seg_masks = zip(*(dataset[img_id] for img_id in batch_imgs))

            # flatten imgs and masks, obtain activations, and resize both activations and seg masks to the same target_shape
            activations, seg_masks_reshaped = self._get_reshaped_activations_segmentations(
                imgs_pil, seg_masks, target_shape=self.target_shape, activations_extractor=self.activations_extractor,
                device=self.device)  # len = #img_ids * #imgs_per_id

            # optimize LoCEs
            loces = self.batch_optimizer.optimize_loces(seg_masks_reshaped, activations)  # lengths = #img_ids * #imgs_per_id / #acts_per_loce

            # store LoCEs
            img_paths_used = [dataset.get_img_path(img_id) for img_id in batch_imgs] # len = #img_ids
            loce_storages = self._wrap_loce_storages(
                loces, img_paths_used, [s.cpu().numpy() for s in seg_masks],
                category_id, activations)

            self._save_loce_storages(loce_saver, loce_storages)

            del loce_storages
            del activations
            del imgs_pil
            del seg_masks
            del seg_masks_reshaped
            torch.cuda.empty_cache()

            #all_loce_storages.extend(loce_storages)

        #return all_loce_storages

    @staticmethod
    def _get_img_ids_not_in_storage(dataset: SegmentationDataset,
                                    category_id: int,
                                    saver: LoCEMultilayerStorageSaver,
                                    img_ids: Iterable[str] = None,
                                    n_imgs: int = None,
                                    ) -> list[str]:
        """Return the n_imgs (dataset.)img_ids that do not yet exist in the save storage."""
        img_ids: Iterable[str] = img_ids or dataset.img_ids
        img_filenames: Generator[str] = (dataset.get_img_filename(img_id) for img_id in img_ids)
        idxs_non_existing: Generator[int] = saver.get_idxs_missing(img_filenames, category_id)
        img_ids_non_existing: Generator[str] = (dataset.img_ids[idx] for idx in idxs_non_existing)
        if n_imgs is not None:
            img_ids_non_existing: Iterable[str] = itertools.islice(img_ids_non_existing, n_imgs)
        return list(img_ids_non_existing)

    @staticmethod
    def _save_loce_storages(storage_saver: LoCEMultilayerStorageSaver,
                            storages: Iterable[LoCEMultilayerStorage]
                            ):

        for storage in storages:

            image_name = os.path.basename(storage.image_path)
            category_id = storage.segmentation_category_id

            out_path_pkl, out_path_err = storage_saver.get_loce_storage_path_for_img_name(image_name, category_id)

            try:
                if not os.path.exists(out_path_pkl):
                    storage_saver.save(storage, out_path_pkl)
            except:
                open(out_path_err, 'a').close()

    @staticmethod
    def _wrap_loce_storages(loces: dict[str, np.ndarray],  # img_ids * #imgs_per_id / #acts_per_loce
                            img_paths_used: Sequence[str],  # img_ids
                            orig_segmentation_masks: Sequence[np.ndarray],  # img_ids * #imgs_per_id
                            segmentation_category_id: int,
                            # segmentations_reshaped: Tensor,
                            activations_reshaped: dict[str, Tensor],  # img_ids * #imgs_per_id
                            store_performance: bool = False,
                            store_masks: bool = False,
                            ) -> List[LoCEMultilayerStorage]:

        storages = []

        # If more than one img_id was used per LoCE: spread the LoCEs like [g1, g2, ...] -> [g1, g1, ..., g2, g2, ...]
        # Then the same LoCE is stored in each image, together with its respective output result.
        # This can happen if self.batch_optimizer.num_acts_per_loce > #images per img_id.
        len_loces, len_segs, len_img_paths_used = len(list(loces.values())[0]), len(orig_segmentation_masks), len(
            img_paths_used)
        if len_segs != len_loces:
            assert len_segs % len_loces == 0 and len_img_paths_used % len_loces == 0, \
                (
                    f"Got incompatible counts of LoCES ({len_loces}) and images used ({len_segs}), for {len_img_paths_used} image paths considered."
                    f"Count of image paths must be a multiple of LoCEs count, and images used must be a multiple of image paths.");
            # [g1, g2, ...] -> [g1, g1, ..., g2, g2, ...]
            loces = {l: [g for g in gs for _ in range(len_segs // len_loces)]  # = #acts_per_loce
                     for l, gs in loces.items()}
            # [p1, p2, ...] -> [p1, p1, ..., p2, p2, ...]
            img_paths_used = (img_path for img_path in img_paths_used
                              for _ in range(len_segs // len_img_paths_used))  # = #imgs_per_id

        for idx, (img_path, seg_msk) in enumerate(zip(img_paths_used, orig_segmentation_masks)):
            storage = LoCEMultilayerStorage(img_path, None,
                                            None if not store_masks else seg_msk,
                                            segmentation_category_id)

            for layer in loces.keys():
                loce_proj, loce_loss_iou = None, None

                loce_current = loces[layer][idx]

                if store_performance:
                    acts_current = activations_reshaped[layer][idx].detach().cpu().numpy()
                    seg_current = orig_segmentation_masks[idx]
                    #seg_current = segmentations_reshaped[idx].detach().cpu().numpy()
                    loce_proj = get_projection(loce_current, acts_current)
                    bin_proj = resize(loce_proj, seg_current.shape) > 0.5
                    #loce_loss = np.abs(seg_current - loce_proj / 255.).sum() / seg_current.size
                    bin_seg = seg_current.astype(bool)
                    loce_loss_iou = metrics.jaccard_score(bin_seg.flatten(), bin_proj.flatten())

                storage.set_loce(layer, LoCE(loce_current, loce_loss_iou, loce_proj))

            storages.append(storage)

        return storages

    _Img = TypeVar('_Img')
    _Mask = TypeVar('_Mask')

    @staticmethod
    def _to_flattened_pairs(imgs: Iterable[Union[_Img, tuple[_Img, ...]]],
                            seg_masks: Iterable[Union[_Mask, tuple[_Mask, ...]]]
                            ) -> Generator[tuple[_Img, _Mask], None, None]:
        """Assuming either images or segmentation masks may be provided as tuple in several variants,
        flatten into same-sized lists."""
        is_flat = lambda x: not isinstance(x, (tuple, list))
        for img_tuple, seg_mask_tuple in zip(imgs, seg_masks):
            if is_flat(img_tuple) and is_flat(seg_mask_tuple):
                yield img_tuple, seg_mask_tuple
            elif is_flat(seg_mask_tuple):
                for img in img_tuple:
                    yield img, seg_mask_tuple
            elif is_flat(img_tuple):
                for seg_mask in seg_mask_tuple:
                    yield img_tuple, seg_mask
            else:
                for img, seg_mask in zip(img_tuple, seg_mask_tuple):
                    yield img, seg_mask


    @classmethod
    def _get_reshaped_activations_segmentations(cls,
                                                imgs_pil: Iterable[Union[PIL.Image.Image, tuple[PIL.Image.Image, ...]]],
                                                seg_masks: Iterable[Union[torch.Tensor, tuple[torch.Tensor, ...]]],
                                                target_shape: tuple[int, int],
                                                activations_extractor: LoCEActivationsTensorExtractor,
                                                device: Union[torch.device, str] = None,
                                                ) -> tuple[dict[str, Tensor], Tensor]:
        # populate the segmentations and layer-wise activations
        activations: dict[str, list[Tensor]] = defaultdict(list)
        seg_masks_t: list[Tensor] = []
        for img_pil, seg_mask in cls._to_flattened_pairs(imgs_pil, seg_masks):
            # resize and tensorize activations
            seg_masks_t.append(torch.as_tensor(seg_mask, device=device, dtype=torch.float64
                                               ).unsqueeze(0).unsqueeze(0))  # dim 1 x 1 x H x W

            # obtain activations (resize later in batch)
            acts_dict, _ = activations_extractor.get_bchw_acts_preds_dict(img_pil, get_predictions=False)
            for l, act in acts_dict.items():
                activations[l].append(act.to(device))

        activations_reshaped: dict[str, Tensor] = {
            l: cls._interpolate(torch.vstack(a), size=target_shape, mode='bilinear')  # first stack (all same sized)
            for l, a in activations.items()}
        seg_masks_reshaped: Tensor = torch.vstack([
            cls._interpolate(seg_mask_t, size=target_shape, mode='bilinear')  # first interpolate, then stack
            for seg_mask_t in seg_masks_t])

        return activations_reshaped, seg_masks_reshaped

    @staticmethod
    def _interpolate(tens: torch.Tensor, size, mode: Literal['bilinear', 'nearest', 'linear', 'bicubic', 'trilinear', 'area', 'nearest-exact']= 'bilinear') -> torch.Tensor:
        """Call to torch.nn.functional.interpolate catching spurious error due to tensor size.
        See https://github.com/pytorch/pytorch/issues/81665
        """
        try:
            return F.interpolate(tens, size=size, mode=mode)
        except (RuntimeError, OutOfMemoryError):
            # Ensure that cache size is not the issue
            torch.cuda.empty_cache()
            # Interpolate each channel individually
            return torch.vstack([F.interpolate(tens[i].unsqueeze(0), size=size, mode=mode)
                                 for i in range(tens.size()[0])])


class LoCEOptimizationEngineMSCOCO(LoCEOptimizationEngine):
    """Implementation of LoCEOptimizationEngine for MS COCO dataset formats.
    Features convenience functions that automatically populate the datasets during initialization
    given the paths to MS COCO images, annotations, segmenter, and desired concepts (via mscoco_tags IDs)."""

    def __init__(self,
                 batch_optimizer: TorchCustomLoCEBatchOptimizer,
                 activations_extractor: LoCEActivationsTensorExtractor,
                 mscoco_imgs_path: str = "./data/mscoco2017val/val2017/",
                 mscoco_default_annots: str = "./data/mscoco2017val/annotations/instances_val2017.json",
                 mscoco_processed_annots: str = None,
                 out_base_dir: str = "./data/mscoco2017val/processed/loces",
                 n_imgs_per_category: int = 500,
                 target_shape: Tuple[int, int] = (100, 100),
                 mscoco_tags: Dict[int, str] = None,
                 image_loader: ImageLoader = ImageLoader(),
                 segmenter_tags: list[Literal['original', 'rectangle', 'ellipse']] = ('original',),
                 _annots_by_cat_id: dict[int, dict] = None,
                 ) -> None:
        """
        Args:
            batch_optimizer (TorchCustomLoCEBatchOptimizer): batch optimizer
            activations_extractor (LoCEActivationsTensorExtractor): activations extractor

        Kwargs:
            out_base_dir (str = "./data/mscoco2017val/processed/loces"): base directory for outputs, subdirs will be created
            processor (BaseImageProcessor = None): Processor from Hugging Face
        """
        # Create the COCO subsets for each segmenter tag and category ID
        mscoco_tags: dict[int, str] = _annots_by_cat_id.keys() if _annots_by_cat_id is not None else mscoco_tags
        self.mscoco_tags = mscoco_tags or MSCOCOSegmentationDataset.ALL_CAT_NAMES_BY_ID
        self.segmenter_tags = segmenter_tags or MSCOCOSegmentationDataset.coco_segmenters.keys()
        datasets: list[MSCOCOSegmentationDataset] = [MSCOCOSegmentationDataset(
            imgs_path=mscoco_imgs_path,
            all_annots_path=mscoco_default_annots,
            segmenter_tag=segmenter_tag,
            category_names_by_id=self.mscoco_tags,
            category_ids=[category_id],
            image_loader=image_loader,
            processed_annots_dir=mscoco_processed_annots,
            _annots_by_cat_id=_annots_by_cat_id,
            device=batch_optimizer.device,
        ) for segmenter_tag, category_id in itertools.product(self.segmenter_tags, self.mscoco_tags.keys())
        ]

        super().__init__(
            batch_optimizer=batch_optimizer,
            activations_extractor=activations_extractor,
            datasets=datasets,
            out_base_dir=out_base_dir,
            n_imgs_per_category=n_imgs_per_category,
            target_shape=target_shape,
        )

        # MS COCO format specific lookup variables (also occur in the datasets)
        self.mscoco_imgs_path = mscoco_imgs_path
        self.mscoco_default_annots = mscoco_default_annots
        self.mscoco_processed_annots = mscoco_processed_annots

        self.img_loader = image_loader

    def run_optimization(self,
                         segmenter_tag: Literal["original", "rectangle", "ellipse"] = None,
                         *,
                         batch_size: int = 64,
                         verbose: bool = False,
                         datasets: list[SegmentationDataset] = None,
                         ) -> None:
        """
        Perform optimization. Results are saved to: self.out_base_dir/loce_{segmenter_tag}_{self.propagator_tag}
        """
        datasets = datasets or self.datasets
        if segmenter_tag is not None:
            datasets = [d for d in datasets if d.segmenter_tag == segmenter_tag]

        super().run_optimization(datasets=datasets, batch_size=batch_size, verbose=verbose)


class Net2VecOptimizationEngine(LoCEOptimizationEngine):

    def __init__(self,
                 batch_optimizer: TorchCustomLoCEBatchOptimizer,
                 activations_extractor: LoCEActivationsTensorExtractor,
                 datasets: list[SegmentationDataset],
                 out_base_dir: str = "./data/mscoco2017val/processed/loces",
                 n_imgs_per_category: int = 500,
                 target_shape: Tuple[int, int] = (100, 100),
                 batch_size: Optional[int] = 64,
                 epochs: int = 50,
                 loading_batch_size: Optional[int] = 4,
                 ) -> None:
        """
        Args:
            batch_optimizer (TorchCustomLoCEBatchOptimizer): batch optimizer
            activations_extractor (LoCEActivationsTensorExtractor): activations extractor
            datasets: the datasets to (separately) run optimization on
            target_shape: Common target shape for activation maps and segmentation masks.
            batch_size: number of images simultaneously loaded and fed to optimization; all available if set to None

        Kwargs:
            out_base_dir (str = "./data/mscoco2017val/processed/loces"): base directory for outputs, subdirs will be created
            processor (BaseImageProcessor = None): Processor from Hugging Face
        """
        super().__init__(batch_optimizer=batch_optimizer, activations_extractor=activations_extractor,
                         datasets=datasets, out_base_dir=out_base_dir,
                         n_imgs_per_category=n_imgs_per_category,
                         target_shape=target_shape, batch_size=batch_size,)
        self.epochs: int = epochs
        self.loading_batch_size: int = loading_batch_size

    def _optimize_one_category_batchwise(self,
                                         dataset: SegmentationDataset,
                                         img_ids_to_optimize: list[str],
                                         loce_saver: LoCEMultilayerStorageSaver,
                                         batch_size: int = None,
                                         loading_batch_size: int = None,  # TODO: make available
                                         ) -> None: #List[LoCEMultilayerStorage]:
        """
        Perform optimization. Results are saved to: self.out_base_dir/loce_{segmenter_tag}_{self.propagator_tag}
        """
        category_id = dataset.cat_ids[0]
        batch_size = batch_size or self.batch_size or len(img_ids_to_optimize)
        epochs = self.epochs
        device = self.device
        loading_batch_size = loading_batch_size or self.loading_batch_size or batch_size

        # Fill the caches: collect activations and seg masks.
        cpu_act_cache: dict[str, Optional[torch.Tensor]] = None  # {layer: vstacked_flattened_acts}
        cpu_seg_cache: Optional[torch.Tensor] = None  # vstacked_flattened_seg_masks
        cpu_cache_idxs: dict[int, str] = {}  # {idx: slice}

        ex_seg_mask, ex_img_id = None, None  # needed for the storage mechanism below
        for batch_ids in tqdm(itertools_batched(img_ids_to_optimize, n=loading_batch_size),
                              desc=f"Batches of {loading_batch_size} loaded into cache", leave=False,
                              total=max(1, len(img_ids_to_optimize)//loading_batch_size)):
            imgs_pil, seg_masks = zip(*(dataset[img_id] for img_id in batch_ids))
            ex_seg_mask, ex_img_id = seg_masks[0], batch_ids[0]

            # flatten imgs and masks, obtain activations, and resize both activations and seg masks to the same target_shape
            activations, seg_masks_reshaped = self._get_reshaped_activations_segmentations(
                imgs_pil, seg_masks,
                target_shape=self.target_shape,
                activations_extractor=self.activations_extractor,
                device=self.device)  # len = #img_ids * #imgs_per_id

            # cache the activations and masks for the next epoch
            cpu_cache_idxs |= {(len(cpu_act_cache) if cpu_act_cache is not None else 0) + batch_i + idx: img_id
                               for batch_i, img_id in enumerate(batch_ids)
                               for idx in range(len(seg_masks_reshaped) // len(batch_ids))}
            cpu_act_cache = {l: torch.vstack([cpu_act_cache[l], activations[l].cpu()]) if cpu_act_cache is not None else activations[l].cpu()
                             for l in activations.keys()}
            cpu_seg_cache = torch.vstack([cpu_seg_cache, seg_masks_reshaped.cpu()]) if cpu_seg_cache is not None else seg_masks_reshaped.cpu()

        # Now we can shuffle:
        segs_count = len(cpu_seg_cache)
        new_idxs = random.sample(range(segs_count), segs_count)
        cpu_cache_idxs = {new_idxs[i]: cpu_cache_idxs[i] for i in cpu_cache_idxs.keys()}
        cpu_act_cache = {l: acts[new_idxs] for l, acts in cpu_act_cache.items()}
        cpu_seg_cache = cpu_seg_cache[new_idxs]


        # Actual optimization
        loces: Optional[dict[str, np.ndarray]] = None
        for batch_idxs in tqdm(itertools_batched(list(range(len(cpu_seg_cache))) * epochs, n=batch_size),
                               desc="Batches * Epochs", leave=False,
                               total=max(1, (len(cpu_seg_cache) * epochs) // batch_size)):
            batch_idxs: list[int] = list(batch_idxs)

            # load acts and segs from cache ...
            activations = {l: cpu_act_cache[l][batch_idxs].to(device) for l in cpu_act_cache.keys()}
            seg_masks_reshaped = cpu_seg_cache[batch_idxs].to(device)

            # optimize LoCEs
            # Note: epochs here is how often a batch is seen before proceeding; we want this to be 1
            per_batch_loces = self.batch_optimizer.optimize_loces(seg_masks_reshaped, activations,
                                                                  init_with=loces, epochs=1)  # lengths = #img_ids * #imgs_per_id / #acts_per_loce
            # The optimization stores one loce per batch; we only want that of the last one.
            loces = {layer: gs[-1] for layer, gs in per_batch_loces.items()}

        # store LoCEs (use only one image to spare storage space)
        store_img_ids = [ex_img_id]
        store_seg_masks = [ex_seg_mask]
        img_paths_used = [dataset.get_img_path(ex_img_id)] # len = #img_ids
        idxs = [i for img_id in store_img_ids for i, iid in cpu_cache_idxs.items() if iid == img_id]
        act_maps = {l: act[idxs] for l, act in cpu_act_cache.items()}
        loce_storages = self._wrap_loce_storages(
            {l: np.expand_dims(g, 0) for l, g in loces.items()},
            img_paths_used, [s.cpu().numpy() for s in store_seg_masks],
            category_id, act_maps)

        self._save_loce_storages(loce_saver, loce_storages)

        del loce_storages
        del activations
        del imgs_pil
        del seg_masks
        del seg_masks_reshaped
        torch.cuda.empty_cache()
