# This file is part of DUNEdn by M. Rossi
"""
    This module contains the utility functions for the preprocessing step.
"""
import logging
from typing import Tuple
from pathlib import Path
from glob import glob
import numpy as np
from dunedn.configdn import PACKAGE
from dunedn.geometry.helpers import evt2planes
from dunedn.utils.utils import median_subtraction

# instantiate logger
logger = logging.getLogger(PACKAGE + ".preprocess")


def save_normalization_info(dir_name: Path, channel: str):
    """Stores on disk useful information to apply dataset normalization.

    Available normalizations are MinMax | Zscore | Mednorm

    Parameters
    ----------
    dir_name: Path
        Directory path to datasets.
    channel: str
        Induction | collection.
    """
    logger.info("Saving normalization info to %s", dir_name)
    fname = dir_name / f"train/planes/{channel}_noisy.npy"
    n = np.load(fname).flatten()

    # MinMax
    fname = dir_name / f"{channel}_minmax"
    np.save(fname, [n.min(), n.max()])

    # Zscore
    fname = dir_name / f"{channel}_zscore"
    np.save(fname, [n.mean(), n.std()])

    # Mednorm
    medians = np.median(n.reshape([n.shape[0], -1]), axis=1)
    med_min = (n - medians).min()
    med_max = (n - medians).max()
    fname = dir_name / f"{channel}_mednorm"
    np.save(fname, [med_min, med_max])


def get_crop(
    clear_plane: np.ndarray,
    nb_crops: int = 1000,
    crop_size: list[int] = [32, 32],
    pct=0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds crops centers indeces and return crops around them.

    Parameters
    ----------
    clear_plane: np.ndarray
        Clear plane of shape=(H,W).
    nb_crops: int
        Number of crops.
    crop_size: list
        Crop [height, width].
    pct: float
        Signal / background crops balancing.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Crop indices:

        - row indices, of shape=(nb_crops, crop_edge, 1).
        - column indices,  of shape=(nb_crops, 1, crop_edge).
    """
    x, y = clear_plane.shape
    c_x, c_y = crop_size[0] // 2, crop_size[1] // 2

    im = clear_plane != 0

    sgn = np.transpose(np.where(im == True))
    bkg = np.transpose(np.where(im == False))

    samples = []
    sample = np.random.choice(len(sgn), size=int(nb_crops * pct))
    samples.append(sgn[sample])

    sample = np.random.choice(len(bkg), size=int(nb_crops * (1 - pct)))
    samples.append(bkg[sample])

    samples = np.concatenate(samples)

    w = (
        np.minimum(np.maximum(samples[:, 0], c_x), x - c_x),
        np.minimum(np.maximum(samples[:, 1], c_y), y - c_y),
    )  # crops centers

    idx_h = (w[0][:, None] + np.arange(-c_x, c_x)[None])[:, :, None]
    idx_w = (w[1][:, None] + np.arange(-c_y, c_y)[None])[:, None, :]
    return (idx_h, idx_w)


def get_planes_and_dump(dname: Path, save_sample: bool):
    """
    Populates the ``<dname>/planes`` directory with APA planes arrays.

    Planes  come from events in the ``<dname>/events`` directory.
    Planes arrays have shape=(N,C,H,W).

    Parameters
    ----------
    dname: Path
        Path to train|val|test dataset subfolder.
    save_sample: bool
        Wether to save a smaller dataset from the original one.
    """
    # TODO: this function could probably be shortened
    iclear = []
    inoisy = []
    isimch = []
    cclear = []
    cnoisy = []
    csimch = []

    paths_clear = glob((dname / "evts/*noiseoff*").as_posix())
    assert len(paths_clear) != 0

    logger.info("Fetching files from %s", dname)
    for path_clear in paths_clear:
        path_noisy = Path(path_clear.replace("rawdigit_noiseoff", "rawdigit"))
        path_simch = Path(path_clear.replace("rawdigit_noiseoff", "simch_labels"))
        path_clear = Path(path_clear)

        logger.debug("  %s", path_clear.name)
        logger.debug("  %s", path_noisy.name)
        logger.debug("  %s", path_simch.name)

        c = np.load(path_clear)[:, 2:]
        n = np.load(path_noisy)[:, 2:]
        s = np.load(path_simch)[:, 2:]

        induction_c, collection_c = evt2planes(c)
        iclear.append(induction_c)
        cclear.append(collection_c)

        induction_n, collection_n = evt2planes(n)
        inoisy.append(induction_n)
        cnoisy.append(collection_n)

        induction_s, collection_s = evt2planes(s)
        isimch.append(induction_s)
        csimch.append(collection_s)

    reshape = lambda x: x.reshape((-1,) + x.shape[2:])
    iclear = reshape(np.stack(iclear))
    cclear = reshape(np.stack(cclear))

    inoisy = reshape(np.stack(inoisy))
    cnoisy = reshape(np.stack(cnoisy))

    isimch = reshape(np.stack(isimch))
    csimch = reshape(np.stack(csimch))

    # at this point planes have shape=(nb_events,N,1,H,W)
    # with N being the number of induction|collection planes in each event

    logger.info("Saving planes to %s/planes", dname)

    logger.debug("  collection clear planes: %s", cclear.shape)
    logger.debug("  collection noisy planes: %s", cnoisy.shape)
    logger.debug("  collection sim::SimChannel planes: %s", csimch.shape)
    logger.debug("  induction clear planes: %s", iclear.shape)
    logger.debug("  induction noisy planes: %s", inoisy.shape)
    logger.debug("  induction sim::SimChannel planes: %s", isimch.shape)

    # stack all the planes from different events together
    save = lambda x, y: np.save(dname / f"planes/{x}", y)

    save("induction_clear", iclear)
    save("collection_clear", cclear)

    save("induction_noisy", inoisy)
    save("collection_noisy", cnoisy)

    save("induction_simch", isimch)
    save("collection_simch", csimch)

    if save_sample:
        # extract a small collection sample from dataset
        logger.info("Saving sample dataset to %s/planes", dname)
        save("sample_collection_clear", cclear[:10])
        save("sample_collection_noisy", cnoisy[:10])
        save("sample_collection_simch", csimch[:10])


def crop_planes_and_dump(
    dir_name: Path, nb_crops: int, crop_size: list[int], pct: float
):
    """Populates the ``<dir_name>/crop`` folder.

    For each plane stored in ``<dir_name>/planes`` generate ``nb_crops`` of size
    ``crop_size`` according to fixed signal to background percentage.

    Parameters
    ----------
    dir_name: Path
        Directory path to datasets.
    nb_crops: int
        Number of crops from a single plane.
    crop_size: list[int]
        Crop size, (height, width).
    pct: float
        Signal to background crops balancing.
    """
    for s in ["induction", "collection"]:

        fname = dir_name / f"planes/{s}_clear.npy"
        cplanes = np.load(fname)[:, 0]

        fname = dir_name / f"planes/{s}_noisy.npy"
        nplanes = np.load(fname)

        logger.info("Cropping %s planes at %s", s, fname)

        nplanes = median_subtraction(nplanes)[:, 0]

        ccrops = []
        ncrops = []
        for cplane, nplane in zip(cplanes, nplanes):
            idx = get_crop(cplane, nb_crops=nb_crops, crop_size=crop_size, pct=pct)
            ccrops.append(cplane[idx][:, None])
            ncrops.append(nplane[idx][:, None])

        ccrops = np.concatenate(ccrops, 0)
        ncrops = np.concatenate(ncrops, 0)

        fname = dir_name / f"crops/{s}_noisy_{crop_size[0]}_{pct}"
        logger.info("Saving crops to %s", dir_name)

        logger.debug("%s{s} clear crops: %s", s, ccrops.shape)
        logger.debug("%s{s} noisy crops: %s", s, ncrops.shape)
        np.save(fname, ncrops)

        fname = dir_name / f"crops/{s}_clear_{crop_size[0]}_{pct}"
        np.save(fname, ccrops)
