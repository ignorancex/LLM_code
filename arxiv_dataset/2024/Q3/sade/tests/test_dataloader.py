import os

import ml_collections
import pytest
import torch

from sade.datasets.loaders import get_dataloaders, get_datasets


@pytest.fixture
def test_config():
    config = ml_collections.ConfigDict()
    data = config.data = ml_collections.ConfigDict()
    data.cache_rate = 0
    data.dataset = "testing"
    data.image_size = (176, 208, 160)
    data.num_channels = 2
    data.spacing_pix_dim = 1.0
    data.dir_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "dummy_data")
    data.splits_dir = data.dir_path

    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 1

    config.eval = ml_collections.ConfigDict()
    config.eval.batch_size = 1

    experiment = config.eval.experiment = ml_collections.ConfigDict()
    experiment.train = "testing-train"  # The dataset used for training MSMA
    experiment.inlier = "testing-val"
    experiment.ood = "tumor"

    return config


def test_dataset_image_lists(test_config):
    assert os.path.exists(test_config.data.dir_path)

    train, val, test = get_datasets(test_config)

    assert train is not None
    assert val is not None
    assert test is not None


def test_dataset_shapes(test_config):
    data = test_config.data
    data.image_size = (32, 32, 32)
    data.spacing_pix_dim = 8.0

    C, H, W, D = data.num_channels, *data.image_size
    _, datasets = get_dataloaders(
        test_config,
        evaluation=False,
        num_workers=1,
    )

    assert len(datasets) == 3
    for ds in datasets:
        assert ds is not None
        x = ds[0]
        assert x is not None
        assert x["image"].dtype is torch.float32
        assert x["image"].shape == (C, H, W, D)


def test_data_loader_iter(test_config):
    dataloaders, _ = get_dataloaders(
        test_config,
        evaluation=False,
        num_workers=2,
    )

    assert len(dataloaders) == 3

    for dl in dataloaders:
        assert dl is not None
        x = next(iter(dl))
        assert x is not None


def test_inf_data_loader(test_config):
    (_, eval_dl, _), _ = get_dataloaders(
        test_config,
        evaluation=True,
        num_workers=1,
        infinite_sampler=True,
    )

    assert eval_dl is not None
    dl_iter = iter(eval_dl)
    # Test data was only one batch
    # Infinite sampler should loop
    for _ in range(3):
        x = next(dl_iter)
        assert x is not None


def test_ood_data_loader(test_config):
    dataloaders, _ = get_dataloaders(
        test_config,
        evaluation=True,
        num_workers=1,
    )

    train_dl, eval_dl, test_dl = dataloaders

    for dl in (train_dl, eval_dl, test_dl):
        assert dl is not None
        x = next(iter(dl))
        assert x is not None


# def test_abcd_data_loader(test_config):
#     data = test_config.data
#     data.dir_path = "/DATA/Users/amahmood/braintyp/processed_v2/"
#     data.splits_dir = "/codespace/sade/sade/datasets/brains/"

#     if not os.path.exists(data.dir_path):
#         pytest.skip("Data not available")

#     data.image_size = (48, 64, 48)
#     data.spacing_pix_dim = 4.0
#     data.num_channels = 2
#     data.ood_ds = "lesion_load_20"

#     C, H, W, D = data.num_channels, *data.image_size

#     dataloaders, datasets = get_dataloaders(
#         test_config,
#         evaluation=False,
#         ood_eval=False,
#         num_workers=1,
#     )

#     assert len(datasets) == 3

#     for ds in datasets:
#         assert ds is not None
#         x = ds[0]
#         assert x is not None
#         assert x["image"].dtype is torch.float32
#         assert x["image"].shape == (C, H, W, D)


#     dataloaders, datasets = get_dataloaders(
#         test_config,
#         evaluation=True,
#         ood_eval=True,
#         num_workers=1,
#     )

#     train_dl, eval_dl, test_dl = dataloaders
#     assert train_dl is None

#     _, eval_ds, test_ds = datasets
#     assert test_ds is not None
#     x = test_ds[0]
#     assert x is not None
#     assert x["image"].dtype is torch.float32
#     assert x["image"].shape == (C, H, W, D)


#     for dl in (eval_dl, test_dl):
#         assert dl is not None
#         x = next(iter(dl))
#         assert x is not None
