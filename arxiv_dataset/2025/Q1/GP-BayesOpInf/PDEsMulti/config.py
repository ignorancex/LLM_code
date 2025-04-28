# config.py
"""Configuration file containing project directives for file and folder names,
plot customizations, and so forth.
"""
import os
import sys
import time
import logging
import numpy as np

from config_heat import (
    spatial_domain,
    time_domain,
    initial_conditions,
    input_parameters,
    test_parameters,
    FullOrderModel,
    Basis,
    ReducedOrderModel,
    input_func_factory,
    CONSTANT_VALUE_BOUNDS,
    LENGTH_SCALE_BOUNDS,
    NOISE_LEVEL_BOUNDS,
    N_RESTARTS_OPTIMIZER,
)


# Paths -----------------------------------------------------------------------
FIGURES_FOLDER = os.path.join(
    "figures",
    time.strftime("%b%d").lower(),
    time.strftime("%H-%M-%S"),
)
LOG_FILE = "log.log"


def TRJFMT(index: int) -> str:
    """String format for the trajectory index."""
    return f"traj{int(index)+1:0>2d}"


def DIMFMT(stateindex: int) -> str:
    """String format for state variable index."""
    return f"r{int(stateindex)+1:0>2d}"


def _makefolder(*args) -> str:
    """Join arguments into a path to a folder. If the folder doesn't exist,
    make the folder as well. Return the resulting path.
    """
    folder = os.path.join(*args)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def figures_path() -> str:
    """Return the path to the folder containing all results figures."""
    # return _makefolder(BASE_FOLDER, FIGURES_FOLDER)   # Figures live by data.
    return _makefolder(os.getcwd(), FIGURES_FOLDER)  # Figures live by code.


# Logger ----------------------------------------------------------------------
_handler = logging.FileHandler(LOG_FILE, "a")
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
_handler.setLevel(logging.INFO)
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)

# Log the session header.
if hasattr(sys.modules["__main__"], "__file__"):
    _front = f"({os.path.basename(sys.modules['__main__'].__file__)})"
    _end = time.strftime("%Y-%m-%d %H:%M:%S")
    _mid = "-" * (79 - len(_front) - len(_end) - 20)
    _header = f"NEW SESSION {_front} {_mid} {_end}"
else:
    _header = f"NEW SESSION {time.strftime(' %Y-%m-%d %H:%M:%S'):->61}"
logging.info(_header)
print(f"Logging to {LOG_FILE}")


# Random seed -----------------------------------------------------------------
np.random.seed(29012024)
