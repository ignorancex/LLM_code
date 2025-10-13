import pytest
import hydra
import os

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.data.birds_datamodule import BirdsDataModule
from src.data.fgbg_datamodule import ForegroundTextureDataModule
from src.data.input_source_generator import InputSourceGenerator
from src.data.texture_datamodule import TextureDataModule


@pytest.mark.parametrize("params", [
    {"batch_size": 32, "images_number": 1000, "im_size": [90, 90]},
    {"batch_size": 32, "images_number": 900, "im_size": [40, 40]},
    {"batch_size": 128, "images_number": 1000, "im_size": [90, 90]},
    {"batch_size": 128, "images_number": 900, "im_size": [40, 40]},
])
def test_texture_datamodule(params, cfg_train: DictConfig) -> None:
    batch_size = params["batch_size"]
    images_number = params["images_number"]
    im_size = params["im_size"]
    HydraConfig().set_config(cfg_train)
    project_root = hydra.utils.get_original_cwd().split("test")[0]
    # Append it to the data_dir
    data_dir = os.path.join(project_root, "data/")
    dm = TextureDataModule(images_number=images_number, im_size=im_size, texture_dir=project_root + "data/textures",
                           data_dir=data_dir,
                           batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == images_number

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert tuple(x.shape) == (batch_size, 1, im_size[0], im_size[1])
    assert tuple(y.shape) == (batch_size,)


@pytest.mark.parametrize("params", [
    {"batch_size": 32, "random_resizing_shifting": False, "im_size": [90, 90], "dataset_type": 'MNIST'},
    {"batch_size": 32, "random_resizing_shifting": False, "im_size": [40, 40], "dataset_type": 'MNIST'},
    {"batch_size": 128, "random_resizing_shifting": False, "im_size": [90, 90], "dataset_type": 'FashionMNIST'},
    {"batch_size": 128, "random_resizing_shifting": True, "im_size": [40, 40], "dataset_type": 'FashionMNIST'},
    {"batch_size": 128, "random_resizing_shifting": True, "im_size": [40, 40], "dataset_type": 'MNIST'},
])
def test_foregroundtexture_datamodule(params, cfg_train: DictConfig) -> None:
    batch_size = params["batch_size"]
    im_size = params["im_size"]
    dataset_type = params['dataset_type']
    random_resizing_shifting = params['random_resizing_shifting']
    HydraConfig().set_config(cfg_train)
    project_root = hydra.utils.get_original_cwd().split("test")[0]
    # Append it to the data_dir
    data_dir = os.path.join(project_root, "data/")
    dm = ForegroundTextureDataModule(texture_dir=project_root + "data/textures",
                                     im_size=im_size,
                                     random_resizing_shifting=random_resizing_shifting,
                                     dataset_type=dataset_type,
                                     data_dir=data_dir,
                                     batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70000

    batch = next(iter(dm.train_dataloader()))
    fg_images, bg_images, masks, fg_labels, bg_labels, _ = batch
    assert len(fg_images) == batch_size
    assert len(bg_images) == batch_size
    assert len(masks) == batch_size
    assert len(fg_labels) == batch_size
    assert len(bg_labels) == batch_size

    assert tuple(fg_images.shape) == (batch_size, 1, im_size[0], im_size[1])
    assert tuple(bg_images.shape) == (batch_size, 1, im_size[0], im_size[1])
    assert tuple(masks.shape) == (batch_size, im_size[0], im_size[1])

    assert tuple(fg_labels.shape) == (batch_size,)
    assert tuple(bg_labels.shape) == (batch_size,)


@pytest.mark.parametrize("params", [
    {"batch_size": 32, "random_resizing_shifting": False, "im_size": [90, 90], "dataset_type": 'MNIST'},
    {"batch_size": 32, "random_resizing_shifting": False, "im_size": [40, 40], "dataset_type": 'MNIST'},
    {"batch_size": 128, "random_resizing_shifting": False, "im_size": [90, 90], "dataset_type": 'FashionMNIST'},
    {"batch_size": 128, "random_resizing_shifting": True, "im_size": [40, 40], "dataset_type": 'FashionMNIST'},
    {"batch_size": 128, "random_resizing_shifting": True, "im_size": [40, 40], "dataset_type": 'MNIST'},
])
def test_input_source_generator_datamodule(params, cfg_train: DictConfig) -> None:
    batch_size = params["batch_size"]
    im_size = params["im_size"]
    dataset_type = params['dataset_type']
    random_resizing_shifting = params['random_resizing_shifting']
    HydraConfig().set_config(cfg_train)
    project_root = hydra.utils.get_original_cwd().split("test")[0]
    data_dir = os.path.join(project_root, "data/")
    dm = ForegroundTextureDataModule(texture_dir=project_root + "data/textures",
                                     im_size=im_size,
                                     random_resizing_shifting=random_resizing_shifting,
                                     dataset_type=dataset_type,
                                     data_dir=data_dir,
                                     batch_size=batch_size)
    dm.setup()
    dm.prepare_data()
    dm = InputSourceGenerator(batch_size, data_dir, dm)
    dm.setup()
    dm.prepare_data()
    assert  dm.train_dataloader() and  dm.val_dataloader() and  dm.test_dataloader()

@pytest.mark.parametrize("params", [
    {
     "data_dir": '/lustre/cniel/data/CUB_200_2011/CUB_200_2011/images',
     "mask_dir": '/lustre/cniel/data/segmentations/',
     "im_size": [64, 64],
     "batch_size": 64
    },
])
def test_birds_datamodule(params, cfg_train: DictConfig) -> None:
    data_dir = params["data_dir"]
    mask_dir = params["mask_dir"]
    im_size = params['im_size']
    batch_size = params['batch_size']
    HydraConfig().set_config(cfg_train)
    dm = BirdsDataModule(data_dir, mask_dir,im_size, batch_size)
    dm.prepare_data()
    dm.setup()
    assert  dm.train_dataloader() and  dm.val_dataloader() and  dm.test_dataloader()
    batch = next(iter(dm.train_dataloader()))
    fgs, bgs, masks = batch
    assert len(fgs) == batch_size
    assert len(bgs) == batch_size
    assert len(masks) == batch_size