"""Some random utils functions. Most are not actually useed in the paper"""

import pickle
import numpy as np
from pathlib import Path
import apogee.tools.path as apath
import apogee.spec.window as apwindow
import apoNN.src.evaluators as evaluators
import matplotlib
import pathlib


text_width = 513.11743
column_width = 242.26653
cmap = matplotlib.cm.get_cmap("viridis")
color1 = cmap(0.15)
color2 = cmap(0.75)


# def get_interstellar_bands(interstellar_locs = [[750,900],[2400,2450],[2600,2750],[4300,4600]]):
def get_interstellar_bands(
    interstellar_locs=[[720, 910], [2370, 2470], [2600, 2750], [4300, 4600]]
):
    """returns a mask with one entry for each wavelegnth bin and where locations containing interstellar bands return False
    Parameters
    ----------
    wavelegth_bins: np.array [[n_start,n_end]]
        contains start and end values of regions to be masked"""
    wavelength_grid = np.ones(8575) == 1
    for loc in interstellar_locs:
        wavelength_grid[loc[0] : loc[1]] = False

    return wavelength_grid, interstellar_locs


# def get_v_bands(interstellar_locs = [[5309, 5753], [3505, 3733], [7112, 7718], [655, 683], [1090, 1128]]):
def get_v_bands(
    interstellar_locs=[
        [5309, 6194],
        [3505, 3733],
        [7112, 7718],
        [655, 683],
        [1090, 1128],
        [8084, 8169],
        [7877, 7912],
        [1889, 3276],
    ]
):
    """returns a mask with one entry for each wavelegnth bin and where locations containing noisy pixels return False
    Parameters
    ----------
    wavelegth_bins: np.array [[n_start,n_end]]
        contains start and end values of regions to be masked"""
    wavelength_grid = np.ones(8575) == 1
    for loc in interstellar_locs:
        wavelength_grid[loc[0] : loc[1]] = False

    return wavelength_grid, interstellar_locs


def get_valid_intercluster_idxs(allStar=None, allStar_occam=None):
    """returns a boolean np.array of shape N_CLUST*N_POP whose positive entries describe pairs of stars to use for intercluster distances"""
    root_path = pathlib.Path(__file__).resolve().parents[2] / "outputs" / "data"
    if allStar is None:
        with open(root_path / "allStar.p", "rb") as f:
            allStar = pickle.load(f)
    if allStar_occam is None:
        with open(root_path / "allStar_occam.p", "rb") as f:
            allStar_occam = pickle.load(f)

    field_idxs = np.array(range(len(allStar)))
    occam_idxs = np.array(range(len(allStar_occam)))
    occam_idxs = np.repeat(occam_idxs[:, np.newaxis], len(field_idxs), axis=1)
    field_idxs = np.repeat(field_idxs[np.newaxis, :], len(occam_idxs), axis=0)
    occam_v = allStar_occam["VHELIO_AVG"][list(occam_idxs)]
    field_v = allStar["VHELIO_AVG"][list(field_idxs)]
    occam_e = allStar_occam["AK_TARG"][list(occam_idxs)]
    field_e = allStar["AK_TARG"][list(field_idxs)]
    occam_snr = allStar_occam["SNR"][list(occam_idxs)]
    field_snr = allStar["SNR"][list(field_idxs)]
    valid_idxs = (np.abs(occam_v - field_v) < 5) & (
        np.abs(occam_e - field_e) < 0.05
    )  # & (field_snr >200)
    return valid_idxs


def generate_loss_with_masking(loss):
    def loss_with_masking(x_pred, x_true, mask):
        return loss(
            x_pred[mask], x_true[mask]
        )  # mask contains the inputs we want to keep.

    return loss_with_masking


def dump(item, filename):
    filepath = (
        Path(__file__).parents[2].joinpath("outputs", "pickled_misc", f"{filename}.p")
    )
    with open(filepath, "wb") as f:
        pickle.dump(item, f)


def load(filename):
    filepath = (
        Path(__file__).parents[2].joinpath("outputs", "pickled_misc", f"{filename}.p")
    )
    with open(filepath, "rb") as f:
        item = pickle.load(f)
    return item


def get_window(elem):
    current_apogee_redux = apath._APOGEE_REDUX
    apath.change_dr(12)
    start, end = apwindow.waveregions(elem, asIndex=True)
    apath._APOGEE_REDUX = current_apogee_redux
    return (start, end)


def get_lines(elem, asIndex=False):
    current_apogee_redux = apath._APOGEE_REDUX
    apath.change_dr(12)
    lines = apwindow.lines(elem, asIndex=asIndex)
    apath._APOGEE_REDUX = current_apogee_redux
    return lines


def get_mask_elem(elem, trimmed=0):
    """trimmed: number of indexes to trim off of each windows edges"""
    spec_mask = np.zeros(8575)
    start, end = get_window(elem)
    for i in range(len(start)):
        line_idx = np.arange(start[i], end[i])
        spec_mask[line_idx] = 1
    return spec_mask


def allStar_to_calendar(allStar):
    """Converts an allstar fits file into an array containing the mjd (observation dates of stars)"""
    mjds = [
        [visit.split("-")[2] for visit in star.split(",")] for star in allStar["VISITS"]
    ]
    return mjds


def get_overlap(mjds, idx1, idx2):
    """given two stars defined by their idx in mjds, returns what percentage of their visits were observed together."""
    mjd1 = mjds[idx1]
    mjd2 = mjds[idx2]
    num_overlap = len(set(mjd1).intersection(mjd2))
    return 0.5 * (num_overlap / len(mjd1) + num_overlap / len(mjd2))


def make_intracluster_similarity_trends(V_occam, get_y):
    """Measures both the similarities for all stellar sibling pairs in a dataset and a y-parameter.
    Parameters
    ----------
    V_occam: Vectors.OccamVector
        OccamVector containing the final transformed representation on which metric learning is applied.
    get_y: function
        Function which takes idx1,idx2 - the indexes of a pair of stars -  and returns the y quantity of interest
    Outputs
    -------
    all_similarities: np.array
        Contains the similarities for all stars in the dataset
    all_y: np.array
        Contains the associated y values for every pair in all_similarities
    """
    all_similarities = []
    all_ys = []
    for cluster in V_occam.registry:
        clust_size = len(V_occam.registry[cluster])
        if clust_size > 1:
            combinations = evaluators.BaseEvaluator.get_combinations(clust_size)
            pairings = np.array(V_occam.registry[cluster][np.array(combinations)])
            v1 = V_occam.val[pairings[:, 0]]
            v2 = V_occam.val[pairings[:, 1]]
            similarities = np.linalg.norm(v1 - v2, axis=1)
            ys = np.array([get_y(pair[0], pair[1]) for pair in pairings])
            all_similarities.append(similarities)
            all_ys.append(ys)

    return np.concatenate(all_similarities), np.concatenate(all_ys)


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
