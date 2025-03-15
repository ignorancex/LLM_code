""" This module contains utility functions of general interest. """
from typing import Any
import subprocess as sp
import multiprocessing
import shutil
from pathlib import Path, PosixPath
import logging
import yaml
import numpy as np
from dunedn.configdn import PACKAGE, get_dunedn_search_path


def check(check_instance: Any, check_list: list[Any]):
    """
    Checks that check_list contains check_instance object. If not, raises
    NotImplementedError.

    Parameters
    ----------
    check_instance: Any
        Object to check.
    check_list: list[Any]
        Available options.

    Raises
    ------
    NotImplementedError
        If ``check_instance`` is not in ``check_list``.
    """
    if not check_instance in check_list:
        raise NotImplementedError("Operation not implemented")


def smooth(smoothed: list[float], scalars: list[float], weight: float) -> list[float]:
    """Computes the next element of the moving average.

    In-place appending of the next element of the moving average to ``smoothed``.

    Parameters
    ----------
    smoothed: list[float]
        The list of smoothed scalar quantities.
    scalars: list[float]
        The list of scalar quantities to be smoothed.
    weight: float
        The weighting factor in the (0,1) range.

    Returns
    -------
    smoothed: list[float]
        The extended list of computed smoothed scalar quantities.

    Raises
    ------
    AssertionError
        If ``scalars`` does not have one element more that ``smoothed``.

    """
    assert len(scalars) - len(smoothed) == 1

    if len(scalars) == 1:
        smoothed.append(scalars[0])
    else:
        smoothed.append(weight * smoothed[-1] + (1 - weight) * scalars[-1])
    return smoothed


def moving_average(scalars: list[float], weight: float) -> list[float]:
    """Computes the moving avarage from a list of scalar quantities.

    Parameters
    ----------
    scalars: list[float]
        List of scalar quantities to be smoothed.
    weight: float
        The weighting factor in the (0,1) range. Higher values provide more
        smoothing power.

    Returns
    -------
    smoothed: list[float]
        The list of smoothed scalar quantities.
    """
    smoothed = []
    for i in range(len(scalars)):
        smooth(smoothed, scalars[: i + 1], weight)
    return smoothed


def median_subtraction(planes: np.ndarray) -> np.ndarray:
    """Computes median subtraction to input planes.

    Parameters
    ----------
    planes: np.ndarray
        The data to be normalized, of shape=(N,C,H,W).

    Returns
    -------
    output: np.ndarray
        The median subtracted data, of shape=(N,C,H,W).
    """
    medians = np.median(planes, axis=[1, 2, 3], keepdims=True)
    output = planes - medians
    return output


def confusion_matrix(hit, no_hit, t=0.5):
    """
    Return confusion matrix elements from arrays of scores and threshold value.

    Parameters:
        hit: np.array, scores of real hits
        no_hit: np.array, scores of real no-hits
        t: float, threshold
    Returns:
        tp, fp, fn, tn
    """
    tp = np.count_nonzero(hit > t)
    fn = np.size(hit) - tp

    tn = np.count_nonzero(no_hit < t)
    fp = np.size(no_hit) - tn

    return tp, fp, fn, tn


def add_info_columns(evt: np.ndarray) -> np.ndarray:
    """Adds event identifier and channel number columns to event.

    Events come with additional information placed in the two first comlumns of
    the 2D array. These must be removed to make the computation as they are not
    informative.
    When saving back the event, the information must be added again.

    Parameters
    ----------
    evt: np.ndarray
        The event w/o additional information, of shape=(nb channels, nb tdc ticks).

    Returns
    -------
        The event w additional information, of shape=(nb channels, 2 + nb tdc ticks).
    """
    nb_channels, _ = evt.shape
    channels_col = np.arange(nb_channels).reshape([-1, 1])
    event_col = np.zeros_like(channels_col)
    evt_with_info = np.concatenate([event_col, channels_col, evt], axis=1)
    return evt_with_info


# instantiate logger
logger = logging.getLogger(PACKAGE + ".train")


def path_constructor(loader, node):
    """PyYaml utility function."""
    value = loader.construct_scalar(node)
    return Path(value)


def load_runcard(runcard_file: Path) -> dict:
    """Load runcard from yaml file.

    Parameters
    ----------
    runcard_file: Path
        The yaml to dump the dictionary.

    Returns
    -------
    runcard: dict
        The loaded settings dictionary.

    Note
    ----
    The pathlib.Path objects are automatically loaded if they are encoded
    with the following syntax:
    ```
    path: !Path 'path/to/file'
    ```
    """
    if not isinstance(runcard_file, Path):
        runcard_file = Path(runcard_file)

    yaml.add_constructor("!Path", path_constructor)
    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    return runcard


def path_representer(dumper, data):
    """PyYaml utility function."""
    return dumper.represent_scalar("!Path", "%s" % data)


def save_runcard(fname: Path, setup: dict):
    """Save runcard to yaml file.

    Parameters
    ----------
    fname: Path
        The yaml output file.
    setup: Path
        The settings dictionary to be dumped.

    Note
    ----
    pathlib.PosixPath objects are automatically loaded.
    """
    yaml.add_representer(PosixPath, path_representer)
    with open(fname, "w") as f:
        yaml.dump(setup, f, indent=4)


def check_in_folder(folder: Path, should_force: bool):
    """Creates the query folder.

    The ``should_force`` parameters controls the function behavior in case
    ``folder`` exists. If true, it overwrites the existent directory, otherwise
    exits.

    Parameters
    ----------
    folder: Path
        The directory to be checked.
    should_force: bool
        Wether to replace the already existing directory.

    Raises
    ------
    FileExistsError
        If output folder exists and ``should_force`` is False.
    """
    try:
        folder.mkdir()
    except FileExistsError as error:
        if should_force:
            logger.warning(f"Overwriting output directory at {folder}")
            shutil.rmtree(folder)
            folder.mkdir()
        else:
            logger.error('Delete or run with "--force" to overwrite.')
            raise error
    else:
        logger.info(f"Creating output directory at {folder}")


def initialize_output_folder(output: Path, should_force: bool):
    """Creates the output directory structure.

    Parameters
    ----------
    output: Path
        The output directory.
    should_force: bool
        Wether to replace the already existing output directory.
    """
    check_in_folder(output, should_force)
    output.joinpath("cards").mkdir()
    output.joinpath("models").mkdir()


def get_configcard_path(fname):
    """Retrieves the configcard path.

    .. deprecated:: 2.0.0
        this function is not used anymore.

    If the supplied path is not a valid file, looks recursively into directories
    from DUNEDN_SEARCH_PATH environment variable to find the first match.

    Parameters
    ----------
    fname: Path
        Path to configcard yaml file.

    Returns
    -------
    Path, the retrieved configcard path

    Raises
    ------
        FileNotFoundError, if fname is not found.
    """
    if fname.is_file():
        return fname

    # get list of directories from DUNEDN_SEARCH_PATH env variable
    search_path = get_dunedn_search_path()

    # recursively look in search directories
    for base in search_path:
        candidate = base / fname.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Configcard {fname} not found. Please, update DUNEDN_SEARCH_PATH variable."
    )


def get_cpu_info() -> dict:
    """Parses ``lscpu`` command to dictionary.

    Returns
    -------
    cpu_info: dict
        The parsed command output.
    """
    output = sp.check_output("lscpu", shell=True).decode("utf-8")
    cpu_info = {}
    for line in output.split("\n"):
        line = line.strip()
        if line:
            splits = line.split(":")
            key = splits[0]
            value = ":".join(splits[1:])
            cpu_info[key.strip().lower()] = value.strip()
    return cpu_info


def get_nb_cpu_cores() -> int:
    """Returns the number of available cpus for the current process.

    Returns
    -------
    nb_cpus: int
        The number of available cpus for the current process.
    """
    return multiprocessing.cpu_count()
