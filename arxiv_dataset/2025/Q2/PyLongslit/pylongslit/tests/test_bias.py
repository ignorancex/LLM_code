"""
PyLongslit test module for checking the bias module.
"""

import os
import matplotlib
import sys
import pytest


@pytest.mark.order(5)
def test_bias_GQ():
    """
    Test the bias function on the GQ1218+0832 dataset.
    """

    # this resets the memory - needs to be called explicitly by every test
    # therefore no method created for this repeated code

    for key in list(sys.modules.keys()):
        if key.startswith("pylongslit"):
            del sys.modules[key]

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    CONFIG_FILE = "GQ1218+0832.json"
    from pylongslit import set_config_file_path

    set_config_file_path(CONFIG_FILE)
    from pylongslit.mkspecbias import run_bias
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the bias function
    run_bias()

    assert os.path.exists(os.path.join(output_dir, "master_bias.fits"))


@pytest.mark.order(6)
def test_bias_SDSS():
    """
    Test the bias function on the SDSS_J213510+2728 dataset.
    """

    # this resets the memory - needs to be called explicitly by every test
    # therefore no method created for this repeated code

    for key in list(sys.modules.keys()):
        if key.startswith("pylongslit"):
            del sys.modules[key]

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    CONFIG_FILE = "SDSS_J213510+2728.json"
    from pylongslit import set_config_file_path

    set_config_file_path(CONFIG_FILE)
    from pylongslit.mkspecbias import run_bias
    from pylongslit.parser import output_dir

    matplotlib.use("Agg")  # Use non-interactive backend

    # Run the bias function
    run_bias()

    assert os.path.exists(os.path.join(output_dir, "master_bias.fits"))
