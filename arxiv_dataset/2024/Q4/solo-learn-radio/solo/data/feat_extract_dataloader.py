import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Callable, Optional, Tuple
from solo.data.custom_aug import AddNormChannels, Pad2Square
from solo.data.data_classes import *

def build_cfg_pipeline(cfg):
    """Creates a pipeline of transformations given a dataset and an augmentation Cfg node.
    The node needs to be in the following format:
        crop_size: int
        [OPTIONAL] mean: float
        [OPTIONAL] std: float
        rrc:
            enabled: bool
            crop_min_scale: float
            crop_max_scale: float
        color_jitter:
            prob: float
            brightness: float
            contrast: float
            saturation: float
            hue: float
        grayscale:
            prob: float
        gaussian_blur:
            prob: float
        solarization:
            prob: float
        equalization:
            prob: float
        horizontal_flip:
            prob: float
    """

    augmentations = []

    cfg = cfg[0]

    if cfg.norm_channels.enabled:
        augmentations.append(
            AddNormChannels(cfg.norm_channels),
        )
    
    if cfg.pad2square.enabled:
        augmentations.append(
            Pad2Square(),
        )

    if cfg.rrc.enabled:
        augmentations.append(
            transforms.RandomResizedCrop(
                cfg.crop_size,
                scale=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
                ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )
    else:
        augmentations.append(
            transforms.Resize(
                cfg.crop_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )

    if cfg.color_jitter.prob:
        augmentations.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        cfg.color_jitter.brightness,
                        cfg.color_jitter.contrast,
                        cfg.color_jitter.saturation,
                        cfg.color_jitter.hue,
                    )
                ],
                p=cfg.color_jitter.prob,
            ),
        )

    if cfg.grayscale.prob:
        augmentations.append(transforms.RandomGrayscale(p=cfg.grayscale.prob))

    if cfg.horizontal_flip.prob:
        augmentations.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip.prob))
    
    if cfg.vertical_flip.prob:
        augmentations.append(transforms.RandomVerticalFlip(p=cfg.vertical_flip.prob))

    augmentations.append(transforms.ToTensor())
    if cfg.meanstd.enabled:
        augmentations.append(transforms.Normalize(mean=cfg.meanstd.mean, std=cfg.meanstd.std))

    T = transforms.Compose(augmentations)

    return T

def prepare_dataset(
    T: Callable,
    data_dict: Optional[dict],
    cfg: dict = {}
) -> Dataset:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T (Callable): pipeline of transformations for the dataset.
        data_path (Optional[Union[str, Path]], optional): path where the
            data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """
    dataset = cfg.data.dataset
    assert dataset in ["rgz", "robin","frg","vlass", "mirabest", "hulk", "superhulk", "banner"]
    if dataset == "robin":
        print("-- ROBIN dataset init -- ")
        dataset   = ROBIN(data_dict=data_dict,  transform=T,  balancing_strategy=cfg.data.balancing_strategy, sample_size=cfg.data.sample_size)
        
    if dataset == "rgz":
        print("-- RGZ dataset init -- ")
        dataset   = RGZ(data_dict=data_dict,  transform=T,  balancing_strategy=cfg.data.balancing_strategy, sample_size=cfg.data.sample_size)
    
    if dataset == "frg":
        print("-- FRG dataset init -- ")
        dataset   = FRG(data_dict=data_dict,  transform=T,  balancing_strategy=cfg.data.balancing_strategy, sample_size=cfg.data.sample_size)
    
    if dataset == "vlass":
        print("-- VLASS dataset init -- ")
        dataset   = VLASS(data_dict=data_dict,  transform=T,  balancing_strategy=cfg.data.balancing_strategy, sample_size=cfg.data.sample_size)

    if dataset == "mirabest":
        print("-- Mirabest dataset init -- ")
        dataset   = MiraBest(data_dict=data_dict,  transform=T,  balancing_strategy=cfg.data.balancing_strategy, sample_size=cfg.data.sample_size)

    if dataset == "banner":
        print("-- Banner dataset init -- ")
        dataset   = CustomUnlabeledDataset(data_path=cfg.data.path,  transform=T,  loader_type=cfg.data.format, datalist=cfg.data.datalist)

    print(f"Dataset lenght: {len(dataset)}")
    return dataset

def prepare_data(
    cfg,
    idxs,
    subsampler
) -> DataLoader:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        subsampler: true/false, all dataset or a subset
    Returns:
        DataLoader: prepared dataloader.
    """

    T = build_cfg_pipeline(cfg.augmentations) # Build augmentations
    
    data_folder = cfg.data.path
    info_file = os.path.join(data_folder, cfg.data.datalist)
    data_dict = pd.read_json(info_file, orient="index")
    
    dataset = prepare_dataset(
        T=T,
        data_dict=data_dict,
        cfg=cfg
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=torch.utils.data.SubsetRandomSampler(idxs) if subsampler else None,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    return loader
