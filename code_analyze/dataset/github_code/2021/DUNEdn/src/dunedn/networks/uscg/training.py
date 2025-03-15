""" This module provides functions for USCG network training and loading."""
from .uscg_dataloading import UscgDataset
import logging
from pathlib import Path
import torch
from .uscg_dataloading import UscgDataset
from .uscg_net import UscgNet
from .utils import make_dict_compatible
from dunedn import PACKAGE
from dunedn.training.metrics import DN_METRICS
from dunedn.training.losses import get_loss

logger = logging.getLogger(PACKAGE + ".uscg")


def load_and_compile_uscg_network(
    channel: str, msetup: dict, checkpoint_filepath: Path = None
) -> UscgNet:
    """Loads a USCG network.

    Parameters
    ---------
    channel: str
        Available options induction | collection.
    msetup: dict
        The model setup dictionary.
    checkpoint_filepath: Path
        The `.pth` checkpoint containing network weights to be loaded.

    Returns
    -------
    network: UscgNet
        The loaded neural network.
    """
    network = UscgNet(channel, **msetup["net_dict"])

    if checkpoint_filepath:
        logger.info(f"Loading weights at {checkpoint_filepath}")
        state_dict = torch.load(checkpoint_filepath, map_location=torch.device("cpu"))
        new_state_dict = make_dict_compatible(state_dict)
        network.load_state_dict(new_state_dict)

    # loss
    loss = get_loss(msetup["loss_fn"])()

    # optimizer
    optimizer = torch.optim.Adam(
        list(network.parameters()), float(msetup["lr"]), amsgrad=msetup["amsgrad"]
    )

    network.compile(loss, optimizer, DN_METRICS)

    return network


def uscg_training(setup: dict):
    """GCNN network training.

    Parameters
    ----------
    setup: dict
        Settings dictionary.
    """
    msetup = setup["model"]["uscg"]
    channel = "collection"
    # model loading
    network = load_and_compile_uscg_network(
        channel, msetup, setup["dev"], msetup["ckpt"]
    )

    # TODO: remove channel (collection | induction) hard coding
    # data loading
    gen_kwargs = {
        "task": setup["task"],
        "channel": channel,
        "dsetup": setup["dataset"],
    }
    train_generator = UscgDataset(
        "train", batch_size=msetup["batch_size"], **gen_kwargs
    )
    val_generator = UscgDataset(
        "val", batch_size=msetup["test_batch_size"], **gen_kwargs
    )
    test_generator = UscgDataset(
        "test", batch_size=msetup["test_batch_size"], **gen_kwargs
    )

    # training
    network.fit(
        train_generator,
        epochs=setup["model"]["epochs"],
        val_generator=val_generator,
        dev=setup["dev"],
    )

    # testing
    _, logs = network.predict(test_generator)
    network.metrics_list.print_metrics(logger, logs)
