import os

import hydra
import pytest
import torch

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.losses.dswd import ProjNet, DSW
from src.data.fgbg_datamodule import ForegroundTextureDataModule


@pytest.mark.parametrize("params", [
    {"embedding_dim": 40 * 40, "num_projections": 1000, "p": 2, "max_iter": 100, "lam": 1.0, 'im_size': [40, 40]},
    # Add more test cases as needed
])
def test_dswd_loss(params, cfg_train: DictConfig):
    # Create synthetic data
    num_samples = 100
    embedding_dim = params["embedding_dim"]

    batch_size = 128
    im_size = params["im_size"]
    dataset_type = 'MNIST'
    random_resizing_shifting = True
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
    dm.setup()

    # Define the components for DSWD loss
    encoder = None  # You can set your encoder here if needed
    embedding_norm = 1.0  # Adjust as needed
    num_projections = params["num_projections"]
    projnet = ProjNet(embedding_dim)
    op_projnet = torch.optim.Adam(projnet.parameters(), lr=.0005)
    p = params["p"]
    max_iter = params["max_iter"]
    lam = params["lam"]

    # Initialize the DSWD loss
    dswd_loss = DSW(encoder, embedding_norm, num_projections, projnet, op_projnet, p, max_iter, lam)

    batch = next(iter(dm.train_dataloader()))
    fg_images, bg_images, masks, fg_labels, bg_labels = batch

    loss = dswd_loss(bg_images, fg_images)
    assert loss > 0

    loss = dswd_loss(fg_images, fg_images)
    expected_loss = 0.0  # Set the expected loss
    tolerance = 1e-4  # Adjust the tolerance as needed
    assert loss.cpu().item() == pytest.approx(expected_loss, abs=tolerance)

    loss = dswd_loss(bg_images, bg_images)
    expected_loss = 0.0  # Set the expected loss
    tolerance = 1e-4  # Adjust the tolerance as needed
    assert loss.cpu().item() == pytest.approx(expected_loss, abs=tolerance)


# Run the test
if __name__ == "__main__":
    pytest.main([__file__])
