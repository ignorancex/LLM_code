import os
from pathlib import Path
import logging

# set package name
PACKAGE = __name__.split(".")[0]  # "dunedn"

# Log levels
LOG_DICT = {
    "0": logging.ERROR,
    "1": logging.WARNING,
    "2": logging.INFO,
    "3": logging.DEBUG,
}

# Read the PDFFLOW environment variables
_log_level_idx = os.environ.get("DUNEDN_LOG_LEVEL")

# Logging
_bad_log_warning = None
if _log_level_idx not in LOG_DICT:
    _bad_log_warning = _log_level_idx
    _log_level_idx = None

if _log_level_idx is None:
    # If no log level is provided, set some defaults
    _log_level = LOG_DICT["2"]
else:
    _log_level = LOG_DICT[_log_level_idx]

# Configure pdfflow logging
logger = logging.getLogger(PACKAGE)
logger.setLevel(_log_level)

# Create and format the log handler
_console_handler = logging.StreamHandler()
_console_handler.setLevel(_log_level)
_console_format = logging.Formatter("[%(levelname)s] (%(name)s) %(message)s")
_console_handler.setFormatter(_console_format)
logger.addHandler(_console_handler)


def get_output_path():
    """
    Gets the DUNEdn output directory path from DUNEDN_OUTPUT_PATH environment
    variable. If the variable is not set, it returns `output` by default.

    Returns
    -------
        - Path, the path to output directory
    """
    root = Path(os.environ.get("DUNEDN_OUTPUT_PATH"))
    if root is not None:
        return root
    return Path("output")


def get_dunedn_search_path():
    """
    Retrieves the list of directories to look for the configuration card.
    Loads DUNEDN_SEARCH_PATH from environment variable (a colon separated
    list of folders).
    The first item is automatically set to the current directory.
    The last item is fixed to the configcards folder in the current directory.

    Set this variable with:
        `export DUNEDN_SEARCH_PATH=<new path>:$DUNEDN_SEARCH_PATH`

    Returns
    -------
        - list, of Path objects from DUNEDN_SEARCH_PATH
    """
    # get directories from colon separated list
    env_var = os.environ.get("DUNEDN_SEARCH_PATH")
    search_path = [] if env_var is None else env_var.split(":")

    # prepend current directory
    search_path.insert(0, ".")

    # append the configcards directory
    search_path.append("./configcards")

    # remove duplicates
    search_path = list(dict.fromkeys(search_path))

    # turn elements into Path objects
    return list(map(Path, search_path))
