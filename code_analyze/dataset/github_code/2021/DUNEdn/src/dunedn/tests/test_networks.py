"""
    Ensures DUNEdn networks objects run the forwrd pass without errors.
"""
import logging
from pathlib import Path
import torch
from dunedn.configdn import PACKAGE
from dunedn.networks.gcnn.training import load_and_compile_gcnn_network
from dunedn.networks.uscg.training import load_and_compile_uscg_network
from dunedn.networks.utils import get_supported_models
from dunedn.utils.utils import load_runcard

# instantiate logger
logger = logging.getLogger(PACKAGE + ".test")


def run_test_uscg(setup: dict):
    """Run USCG network test.

    Parameters
    ----------
    setup: dict
        Runcard settings.

    Raises
    ------
    AssertionError
        If model input and output shapes do not match.
    """
    # tuple containing induction and collection inference arguments
    msetup = setup["model"]["uscg"]
    batch_size = 1
    msetup["net_dict"]["h_collection"] = 64
    msetup["net_dict"]["w"] = 128

    # load dummy dataset
    dummy_dataset = torch.rand(
        batch_size, 1, msetup["net_dict"]["h_collection"], msetup["net_dict"]["w"]
    )

    # load cnn model
    model = load_and_compile_uscg_network("collection", msetup)
    model.eval()

    # forward pass
    output = model(dummy_dataset)

    # check that input and output have the same shape
    try:
        assert dummy_dataset.shape == output.shape
    except AssertionError as err:
        logger.critical(
            "Assertion fail: uscg model input and output shapes do not match"
        )
        raise err


def run_test_cnn(setup: dict):
    """Run CNN network test.

    Parameters
    ----------
    setup: dict
        Runcard settings.

    Raises
    ------
    AssertionError
        If model input and output shapes do not match.
    """
    # tuple containing induction and collection inference arguments
    msetup = setup["model"]["cnn"]

    # load dummy dataset
    dummy_dataset = torch.rand(msetup["batch_size"], 1, *setup["dataset"]["crop_size"])

    # load cnn model
    model = load_and_compile_gcnn_network("collection", msetup)
    model.eval()

    # forward pass
    output = model(dummy_dataset)

    # check that input and output have the same shape
    try:
        assert dummy_dataset.shape == output.shape
    except AssertionError as err:
        logger.critical(
            "Assertion fail: CNN model input and output shapes do not match"
        )
        raise err


def run_test_gcnn(setup: dict):
    """Run GCNN network test.

    Parameters
    ----------
    setup: dict
        Runcard settings.

    Raises
    ------
    AssertionError
        If model input and output shapes do not match.
    """
    # tuple containing induction and collection inference arguments
    msetup = setup["model"]["gcnn"]

    # load dummy dataset
    dummy_dataset = torch.rand(msetup["batch_size"], 1, *setup["dataset"]["crop_size"])

    # load cnn model
    model = load_and_compile_gcnn_network("collection", msetup)
    model.eval()

    # forward pass
    output = model(dummy_dataset)

    # check that input and output have the same shape
    try:
        assert dummy_dataset.shape == output.shape
    except AssertionError as err:
        logger.critical(
            "Assertion fail: GCNN model input and output shapes do not match"
        )
        raise err


def run_test(modeltype: str):
    """
    Run the appropriate test for the supported model.

    Parameters
    ----------
    modeltype: str
        Available options uscg | cnn | gcnn.
    """
    setup = load_runcard(Path("runcards/default.yaml"))
    logger.info("Running forward-pass test on %s model", modeltype)
    if modeltype == "cnn":
        run_test_cnn(setup)
    elif modeltype == "gcnn":
        run_test_gcnn(setup)
    elif modeltype == "uscg":
        run_test_uscg(setup)
    else:
        raise NotImplementedError(f"Modeltype not implemented, got {modeltype}")


def test_networks():
    """Test wrapper function."""
    for modeltype in get_supported_models():
        run_test(modeltype)


if __name__ == "__main__":
    test_networks()
