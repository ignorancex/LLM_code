"""Return training and evaluation/test datasets from config files."""
import functools
import glob
import logging
import os
import re

from monai.data import CacheDataset
from monai.transforms import *
from torch.utils.data import DataLoader, RandomSampler

from sade.datasets.transforms import (
    get_lesion_transform,
    get_train_transform,
    get_tumor_transform,
    get_val_transform,
)


def get_image_files_list(dataset_name: str, dataset_dir: str, splits_dir: str):
    if re.match(r"(lesion)|(brats)|(mslub)|(msseg)", dataset_name):
        image_files_list = [
            {"image": p, "label": p.replace(".nii.gz", "_label.nii.gz")}
            for p in glob.glob(f"{dataset_dir}/**/*.nii.gz", recursive=True)
            if "label" not in p  # very lazy, i know :)
        ]
    else:
        file_path = os.path.join(splits_dir, f"{ dataset_name.lower()}_keys.txt")
        assert os.path.exists(file_path), f"{file_path} does not exist"

        strip = lambda x: x.strip()
        if re.match(r"(abcd)|(multisource)", dataset_name):
            strip = lambda x: x.strip().replace("_", "")

        with open(file_path) as f:
            image_filenames = [strip(x) for x in f.readlines()]

        image_files_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")} for x in image_filenames
        ]

    image_files_list = sorted(image_files_list, key=lambda x: x["image"])

    return image_files_list


def get_datasets(config, training=False):
    """Return training and evaluation/test datasets from config files."""

    dataset_name = config.data.dataset
    # Directory that holds the data samples
    dataset_dir = config.data.dir_path
    # Directory that holds files with train/test splits and other filenames
    splits_dir = config.data.splits_dir

    if training:
        dataset_name = dataset_name.lower()
        train_file_list = get_image_files_list(
            f"{dataset_name}-train", dataset_dir, splits_dir
        )
        val_file_list = get_image_files_list(f"{dataset_name}-val", dataset_dir, splits_dir)
        test_file_list = get_image_files_list(
            f"{dataset_name}-test", dataset_dir, splits_dir
        )
    # An evaluation experiment
    else:
        experiment_dict = config.eval.experiment

        train_dataset = experiment_dict["train"]
        inlier_dataset = experiment_dict["inlier"]
        ood_dataset = experiment_dict["ood"]

        train_file_list = get_image_files_list(train_dataset, dataset_dir, splits_dir)
        val_file_list = get_image_files_list(inlier_dataset, dataset_dir, splits_dir)

        if re.match(r"(lesion)|(brats)|(mslub)|(msseg)", ood_dataset):
            if "lesion" in ood_dataset:
                dirname = f"slicer_lesions/{ood_dataset}/{dataset_name.upper()}"
            else:
                dirname = ood_dataset
            ood_dir_path = os.path.abspath(f"{dataset_dir}/..")
            dataset_dir = f"{ood_dir_path}/{dirname}"
            assert os.path.exists(dataset_dir), f"{dataset_dir} does not exist"
            logging.info(f"Loading ood samples from {dataset_dir}...")

        # Tumor will be added as a transformation to the inliers
        if re.match(r"tumor", ood_dataset):
            ood_dataset = inlier_dataset

        test_file_list = get_image_files_list(ood_dataset, dataset_dir, splits_dir)

    return train_file_list, val_file_list, test_file_list


def get_dataloaders(
    config,
    evaluation=False,
    ood_eval=False,
    num_workers=6,
    infinite_sampler=False,
):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      evaluation: If `True`, only val_transform will be used

    Returns:
      dataloaders, datasets.
    """
    if infinite_sampler:
        inf_sampler = functools.partial(
            RandomSampler, replacement=True, num_samples=int(1e100)
        )

    # Sanity checks
    # assert re.match(r"(abcd|ibis)", config.data.dataset.lower())
    # assert re.match(r"(tumor|lesion|ds-sa)", config.data.ood_ds.lower())

    cache_rate = config.data.cache_rate

    train_file_list, val_file_list, test_file_list = get_datasets(
        config, training=not evaluation
    )

    assert train_file_list is not None and len(train_file_list) > 0
    assert val_file_list is not None and len(val_file_list) > 0
    assert test_file_list is not None and len(test_file_list) > 0

    if evaluation:
        train_transform = get_val_transform(config)
        val_transform = get_val_transform(config)

        ood_ds_name = config.eval.experiment.ood.lower()
        if re.match(r"(lesion)|(brats)|(mslub)|(msseg)", ood_ds_name):
            test_transform = get_lesion_transform(config)
        elif re.match(r"tumor", ood_ds_name):
            test_transform = get_tumor_transform(config)
        else:
            test_transform = get_val_transform(config)
    else:
        train_transform = get_train_transform(config)
        val_transform = get_val_transform(config)
        test_transform = get_val_transform(config)

    train_ds = None
    if train_file_list is not None:
        train_ds = CacheDataset(
            train_file_list,
            transform=train_transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=True,
        )

    eval_ds = CacheDataset(
        val_file_list,
        transform=val_transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
        progress=True,
    )

    # Note: Will be an ood dataset if ood_eval==True
    test_ds = CacheDataset(
        test_file_list,
        transform=test_transform,
        cache_rate=0.0,
        num_workers=num_workers,
        progress=True,
    )

    # Get data loaders
    train_loader = None
    if train_ds:
        train_loader = DataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            num_workers=num_workers,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=num_workers > 0,
            sampler=inf_sampler(train_ds) if infinite_sampler else None,
        )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=num_workers > 0,
        sampler=inf_sampler(eval_ds) if infinite_sampler else None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=num_workers > 0,
        sampler=inf_sampler(test_ds) if infinite_sampler else None,
    )

    dataloaders = (train_loader, eval_loader, test_loader)
    datasets = (train_ds, eval_ds, test_ds)

    return dataloaders, datasets
