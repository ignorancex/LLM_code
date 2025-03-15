"""This module implements utility functions for the ``onnx`` example."""
from typing import Tuple
from time import time as tm
import shutil
import subprocess as sp
from pathlib import Path
from tqdm.auto import tqdm
from itertools import zip_longest
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dunedn.geometry.helpers import evt2planes
from dunedn.inference.hitreco import DnModel
from dunedn.inference.inference import thresholding_dn
from dunedn.networks.gcnn.utils import gcnn_inference_pass
from dunedn.networks.onnx.utils import gcnn_onnx_inference_pass
from dunedn.networks.utils import BatchProfiler
from dunedn.utils.utils import add_info_columns, get_cpu_info, get_nb_cpu_cores


def prepare_folders_and_paths(
    modeltype: str, version: str, base_folder: Path, ckpt_folder: Path
) -> Tuple[dict, dict]:
    """Utility function for onnx example notebook.

    Loads the folder and path names.

    Parameters
    ----------
    modeltype: str
        Available options: gcnn | cnn | uscg.
    version: str
        The training dataset version. Available options: v08 | v09
    base_folder: Path
        The output root folder.
    ckpt_folder: Path
        The checkpoint folder.

    Returns
    -------
    folders: dict
    paths: dict
    """
    # relative folders
    folders = {
        "base": base_folder,
        "out": base_folder / "models/onnx",
        "ckpt": ckpt_folder,
        "cards": base_folder / f"cards",
        "onnx_save": base_folder / f"models/onnx/saved_models/{modeltype}_{version}",
        "plot": base_folder / "models/onnx/plots",
        "id_plot": base_folder / "models/onnx/plots/identity",
        "pytorch_plot": base_folder / "models/onnx/plots/torch",
        "onnx_plot": base_folder / "models/onnx/plots/onnx",
    }

    # path to files
    paths = {
        "input": folders["out"] / "p2GeV_cosmics_inspired_rawdigit_evt8.npy",
        "target": folders["out"] / "p2GeV_cosmics_inspired_rawdigit_noiseoff_evt8.npy",
        "pytorch": folders["out"]
        / f"p2GeV_cosmics_inspired_rawdigit_torch_{modeltype}_evt8.npy",
        "onnx": folders["out"]
        / f"p2GeV_cosmics_inspired_rawdigit_onnx_{modeltype}_evt8.npy",
        "performance_csv": folders["out"] / f"{modeltype}_performance_comparison.csv",
    }
    return folders, paths


def check_in_output_folder(folders: dict):
    """Creates output folder directory tree.

    Utility function to create the main output folder and its sub-directory
    structure. The user should hold permissions to write in it.

    This function also copies necessary files into the output directory.

    Parameters
    ----------
    folders: dict
        Dictionary containing folders to be created.
    """
    # create directories
    folders["cards"].mkdir(parents=True, exist_ok=True)
    folders["onnx_save"].mkdir(parents=True, exist_ok=True)
    folders["id_plot"].mkdir(parents=True, exist_ok=True)
    folders["pytorch_plot"].mkdir(exist_ok=True)
    folders["onnx_plot"].mkdir(exist_ok=True)

    # copy runcard
    runcard_path = Path("../runcards/default.yaml")
    shutil.copyfile(runcard_path, folders["cards"] / "runcard.yaml")
    shutil.copyfile(runcard_path, folders["cards"] / "runcard_default.yaml")

    # extract input tarball
    tarzip = "dunetpc_inspired_v08_p2GeV_rawdigits.tar.gz"
    sp.run(["tar", "-xzf", tarzip, "-C", folders["out"]])


def inference(model: DnModel, evt: np.ndarray, fname: Path, dev: str = None):
    """Makes inference on event and computes time.

    Saves the output file to `fname`.

    Parameters
    ----------
    model: AbstractNet
        The pytorch or onnx based model.
    evt: np.ndarray
        The input raw data.
    fname: Path
        The output file name.
    dev: str
        Device hosting computation.

    Returns
    -------
    inference_time: float
        The elapsed time for inference.
    """
    start = tm()
    evt_dn = model.predict(evt, dev)
    inference_time = tm() - start

    # thresholding
    evt_dn = thresholding_dn(evt_dn)

    # add info columns
    evt_dn = add_info_columns(evt_dn)

    # save inference outputs
    np.save(fname, evt_dn)
    return inference_time


def add_platform_info(info: dict, nb_events: int, dev: str):
    """Adds platform dependent information to dictionary.

    Parameters
    ----------
    info: int
        The dictionary to be filled.
    nb_events: int
        The list length for each dictionary key.
    dev: str
        The device hosting PyTorch computation.
    """
    cpu_info = get_cpu_info()
    cpu_name = cpu_info["model name"]
    cuda = True if "cuda" in dev else False
    nb_cores = get_nb_cpu_cores()
    nb_all_cores = int(cpu_info["cpu(s)"])

    info["cpu_name"] = [cpu_name] * nb_events
    info["cuda"] = [cuda] * nb_events
    info["nb_cores"] = [nb_cores] * nb_events
    info["all_cores"] = [nb_all_cores] * nb_events


def compare_performance_onnx(
    pytorch_model: DnModel,
    onnx_model: DnModel,
    dev: str,
    batch_size_list: list,
    nb_batches: int = 100,
) -> pd.DataFrame:
    """Compares PyTorch and Onnx inference time for different batch sizes.

    Parameters
    ----------
    pytorch_model: DnModel
        The pytorch model to make inference
    onnx_model: DnModel
        The Onnx model to make inference
    dev: str
        Device hosting PyTorch computation.
    batch_size_list: list
        The list of batch sizes.
    nb_batches: int
        The number of batches to be passed to the network.

    Returns
    -------
    df: pd.DataFrame
        The dataframe containing profiled inference times.
    """
    input_shape = pytorch_model.cnetwork.input_shape

    performance = {
        "batch_size": [],
        "framework": [],
        "network": [],
        "time": [],
        "error": [],
    }

    add_platform_info(performance, len(batch_size_list) * 2, dev)

    for batch_size in tqdm(batch_size_list, desc="batch_size"):
        batched_input_shape = (nb_batches, batch_size) + input_shape
        inputs = torch.randn(batched_input_shape)

        torch_pr = BatchProfiler()
        gcnn_inference_pass(
            zip_longest(inputs, []), pytorch_model.cnetwork, dev, profiler=torch_pr
        )
        torch_mean, torch_err = torch_pr.get_stats()

        onnx_pr = BatchProfiler()
        gcnn_onnx_inference_pass(
            zip_longest(inputs, []), onnx_model.cnetwork, profiler=onnx_pr
        )
        onnx_mean, onnx_err = onnx_pr.get_stats()

        performance["batch_size"].extend([batch_size] * 2)
        performance["framework"].extend(["torch", "onnx"])
        performance["time"].extend([torch_mean, onnx_mean])
        performance["error"].extend([torch_err, onnx_err])
        performance["network"].extend([pytorch_model.modeltype, onnx_model.modeltype])

    df = pd.DataFrame(performance)
    return df


def enhance_plane(plane: np.ndarray):
    """Transforms plane to enhance details.

    Parameters
    ----------
    plane: np.ndarray
        The input plane, of shape=(nb wires, nb tdc ticks).

    Returns
    -------
    transformed: np.ndarray
        The transformed plane, of shape=(nb wires, nb tdc ticks).
    """
    pmin = plane.min()
    pmax = plane.max()
    transformed = pmax * ((plane - pmin) / pmax) ** 0.5 + pmin
    return transformed


def plot_image_sample(
    plane: np.ndarray, wire: int, outdir: Path, with_graphics: bool = False
):
    """Plots an APA plane.

    Parameters
    ----------
    plane: np.ndarray
        The reconstructed plane.
    wire: int
        The wire number in the APA plane.
    outdir: Path
        The directory where to save the plot.
    with_graphics: bool
        Wether to show matplotlib plots or not.
    """
    fname = outdir / "visual_collection_plane.png"
    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 300,
        }
    )
    plt.figure(figsize=[6.4 * 1.5, 4.8 * 1.5])
    plt.title(
        r"""Inspired to ProtoDUNE SP simulation
    Collection plane, ADC heatmap
    """,
        y=1.0,
        pad=-10,
    )
    # TODO: use a divergent cmap to show what's above/below 0 ADC (spikes/negative
    # tails)
    # use a sequential cmap giving more importance to higher values to highlight
    # spikes
    cmap = plt.get_cmap("PuBu")
    z = plt.imshow(
        enhance_plane(plane), aspect="auto", cmap=cmap
    )  # , vmin=vmin, vmax=vmax)
    plt.axhline(y=wire, color="orange", lw=1, alpha=0.6, linestyle="dashed")
    plt.colorbar()
    plt.savefig(fname, bbox_inches="tight")  # , dpi=300)
    print(f"Saved image at {fname}")
    if with_graphics:
        plt.show()
    plt.close()


def plot_wire_sample(
    wire: np.ndarray,
    wire_target: np.ndarray,
    outdir: Path,
    with_graphics: bool = False,
):
    """Plots a reconstructed wavefunction and its target one.

    Parameters
    ----------
    wire: np.ndarray
        The reconstructed wavefunction.
    wire_target: np.ndarray
        The target wavefunction.
    outdir: Path
        The directory where to save the plots.
    with_graphics: bool
        Wether to show matplotlib plots or not.
    """
    # plt.rcParams.update({"text.usetex": True})
    fname = outdir / "visual_denoised_wire.png"
    plt.figure(figsize=[6.4 * 1.5, 4.8 * 1.5])
    plt.title(
        r"""Inspired to ProtoDUNE SP simulation
    Collection wire, Raw waveform
    """,
        y=1.0,
        pad=-10,
    )
    plt.plot(wire, lw=0.3)
    plt.savefig(fname, bbox_inches="tight")
    print(f"Saved image at {fname}")
    if with_graphics:
        plt.show()
    plt.close()

    fname = outdir / "visual_clear_wire.png"
    plt.figure(figsize=[6.4 * 1.5, 4.8 * 1.5])
    plt.title(
        r"""Inspired to ProtoDUNE SP simulation
    Collection wire, Clear waveform
    """,
        y=1.0,
        pad=-10,
    )
    plt.plot(wire_target, lw=0.3, color="red")
    plt.savefig(fname, bbox_inches="tight")
    print(f"Saved image at {fname}")
    if with_graphics:
        plt.show()
    plt.close()


def plot_example(
    input_path: Path,
    target_path: Path,
    outdir: Path = None,
    with_graphics: bool = False,
):
    """Plots the first collection plane and a wire waveform from an event sample.

    Three plots are produced:

    - collection plane ADC heatmap
    - reconstructed waveform of a wire sample
    - target waveform

    The plots are saved in the ``outdir`` directory.
    If ``with_graphics`` is ``True``, plots are shown in the matplotlib GUI.

    Parameters
    ----------
    input_path: Path
        The path to the noisy inputs file.
    target_path: Path
        The path to the clear target file.
    outdir: Path
        Directory to save plots into. Defaults to the same directory of
        `input_path`.
    with_graphics: bool
        Wether to show matplotlib plots or not.
    """
    if outdir is None:
        outdir = input_path.parent
    evt = np.load(input_path)[:, 2:]
    # file_clear.unlink()
    evt_target = np.load(target_path)[:, 2:]
    # file_noisy.unlink()

    _, cplanes = evt2planes(evt)
    _, cplanes_target = evt2planes(evt_target)

    plane = cplanes[0, 0, 480:]
    plane_target = cplanes_target[0, 0, 480:]
    wire = 330

    plot_image_sample(plane, wire, outdir, with_graphics)

    plot_wire_sample(plane[wire], plane_target[wire], outdir, with_graphics)


def plot_comparison_catplot(
    data: pd.DataFrame,
    out_folder: Path,
    with_graphics: bool = False,
):
    """Categorical plot for PyTorch vs Onnx performance.

    Parameters
    ----------
    data: pd.DataFrame
        The dataframe with the collected inference timings.
    out_folder: Path
        The folder to save the categorical plot to.
    with_graphics: bool
        Wether to show matplotlib plots or not.
    """
    fname = out_folder / "pytorch_onnx_performance_comparison.png"

    pv_time = data.pivot_table(values="time", index="batch_size", columns="framework")
    pv_err = data.pivot_table(values="error", index="batch_size", columns="framework")

    rel = pd.DataFrame(
        (pv_time["torch"] - pv_time["onnx"]) / pv_time["torch"], columns=["speed-up"]
    )
    rel["maxes"] = np.maximum(pv_time["torch"], pv_time["onnx"])

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    title = (
        f"Platform: {data['cpu_name'][0]}, "
        f"cores: {data['nb_cores'][0]}\n"
        f"PyTorch is using CUDA: {data['cuda'][0]}"
    )
    ax.set_title(title, fontsize=18)
    pv_time.plot.bar(yerr=pv_err, ax=ax, alpha=0.5, rot=0, fontsize=18, xlabel="")

    for j, (_, row) in enumerate(rel.iterrows()):
        sgn = "+" if row["speed-up"] > 0 else ""
        ax.text(
            j,
            row["maxes"] + 0.05,
            f"{sgn}{row['speed-up']:.2f}x",
            ha="center",
            color="black",
            fontsize=18,
        )

    ax.set_xlabel("Batch size", fontsize=18)
    ax.set_ylabel("Time [s]", fontsize=18)
    ax.legend(fontsize=18)
    ax.tick_params(
        axis="x",
        direction="in",
        bottom=True,
        labelbottom=False,
        top=True,
        labeltop=False,
    )
    ax.tick_params(
        axis="x",
        direction="in",
        bottom=True,
        labelbottom=True,
        top=True,
        labeltop=False,
    )
    ax.tick_params(
        axis="y",
        direction="in",
        left=True,
        labelleft=True,
        right=True,
        labelright=False,
    )
    plt.savefig(fname, bbox_inches="tight", dpi=600)
    print(f"Saved image at {fname}")
    if with_graphics:
        plt.show()
    plt.close()
