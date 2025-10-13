# data.py
import os
import torch
from torch.utils.data import DataLoader

from folder_dataset import (
    get_transforms,
    DatasetFromFolder,
)

def get_data_loader(
    real_train_data_dir="",
    real_test_data_dir="",
    metadata_dir=None,   # Not used if we are purely loading from folder
    dataset="imagenet",
    bs=32,
    eval_bs=32,
    n_img_per_cls=None,
    is_synth_train=False,
    n_shot=0,
    real_train_fewshot_data_dir="",
    is_pooled_fewshot=False,
    model_type=None,
):
    """
    Loads real train/test datasets (folder-based).
    If is_synth_train=True, we can skip returning real train loader, or set it to None.
    """
    train_transform, test_transform = get_transforms(model_type)

    # Real training loader

    train_dataset = DatasetFromFolder(
        data_root=real_train_data_dir,
        transform=train_transform,
        dataset=dataset,
        n_img_per_cls=16,
        is_train=True   # typically do not limit test set
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,   # or False, your choice
        num_workers=4,
        pin_memory=True,
    )

    # Real test loader (folder-based)
    test_dataset = DatasetFromFolder(
        data_root=real_test_data_dir,
        transform=test_transform,
        dataset=dataset,
        n_img_per_cls=None,   # typically do not limit test set
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_synth_train_data_loader(
    synth_train_data_dir="",
    bs=32,
    n_img_per_cls=None,
    dataset="imagenet",
    n_shot=0,
    real_train_fewshot_data_dir="",
    is_pooled_fewshot=False,
    model_type=None,
):
    """
    Loads synthetic images from 'synth_train_data_dir' (folder-based).
    Possibly also merges in real few-shot images if is_pooled_fewshot=True.
    """
    train_transform, test_transform = get_transforms(model_type)
    synth_dataset = DatasetFromFolder(
        data_root=synth_train_data_dir,
        transform=train_transform,  # or test_transform if you want less augmentation
        dataset=dataset,
        n_img_per_cls=n_img_per_cls,
        n_shot=n_shot,
        real_train_fewshot_data_dir=real_train_fewshot_data_dir,
        is_pooled_fewshot=is_pooled_fewshot,
        is_train = True,
        start_idx=0
    )
    synth_loader = DataLoader(
        synth_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return synth_loader
