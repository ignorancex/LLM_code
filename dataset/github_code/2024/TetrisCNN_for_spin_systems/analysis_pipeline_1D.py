import ast
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory

import humanize
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib import colormaps as cm
from matplotlib.ticker import AutoMinorLocator
from tqdm.auto import tqdm
from latex2sympy2 import latex2sympy

# Suppress warnings
logging.captureWarnings(True)


import argparse

import torch
import torch.nn as nn
from pysr import PySRRegressor
from sklearn.metrics import r2_score

from src.combinatorics import shifts_from_bounds
from src.loaders import Importer
from src.architectures import ShapeAdaptiveConvNet, SmallModel
from src.metrics import calculate_r2_and_mse, reshape_by_time
from src.auxiliary_functions import (
    apply_model_to_test_data,
    determine_corr_label,
    get_corr,
    round_expr,
)

os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# torch.set_num_threads(1)


def get_folder_gui(parent_dit, title):
    # Make the root window for the GUI
    root = Tk()
    root.withdraw()
    # Ask the user for the directory
    filename = askdirectory(initialdir=parent_dit, title=title)
    return filename


def sr_routine(
    model_path: Path,
    phys_data_path: Path,
    model_name="TetrisCNN",
    phys_model=None,
    phys_phase="TFIM_ferro",
    snaps=["y"],
    filt_num=1,
    normalize_reg=True,
    approx=False,
    skip_corrs=False,
    sr_fit=True,
    just_new=False,
    out_labels=None,
    corrs_sr_params_override=None,
    predictor_sr_params_override=None,
    skip_sr=False,
    BS=32,
    fs=30,
    lw=3,
    ms=10,
    figsize_base=(20, 10),
):
    """
    Performs the symbolic regression routine.

    Args:
        model_path (Path): Path to the model folder
        phys_data_path (Path): Path to the physical data folder
        model_name (str, optional): Model name. Defaults to "TetrisCNN".
        phys_model (str, optional): Physical model. Defaults to "Ising".
        phys_phase (str, optional): Physical phase. Defaults to "ferro".
        phys_shape (tuple, optional): Physical atom array shape. Defaults to (8, 8).
        defects (int, optional): Number of defects. Defaults to 0.
        snaps (list, optional): Measurement bases to use. Defaults to ["z"].
        filt_num (int, optional): Number of filters. Defaults to 1.
        normalize_reg (bool, optional): If True, normalizes the regression data. Defaults to True.
        approx (bool, optional): If True, fits the output to the consecutive highest activations of the alive kernels. Defaults to False.
        skip_corrs (bool, optional): If True, skips the correlations to activations mapping. Defaults to False.
        sr_fit (bool, optional): NOT WORKING! Throws weird errors. If True, performs the symbolic regression fit. Defaults to True.
        just_new (bool, optional): If True, only the new shifts are considered. Defaults to False.
        out_labels (dict, optional): Override of the dictionary with the labels for the output.If not given, it is pre-determined for Ising and XY. Defaults to None.
        corrs_sr_params_override (dict, optional): Dictionary with the override of symbolic regression parameters for the correlations. Defaults to None.
        predictor_sr_params_override (dict, optional): Dictionary with the override of symbolic regression parameters for the predictor. Defaults to None.
        skip_sr (bool, optional): If True, skips the symbolic regression. Defaults to False.
        BS (int, optional): Batch size. Defaults to 32.
        fs (int, optional): Font size for the plots. Defaults to 30.
        lw (int, optional): Line width for the plots. Defaults to 3.
        ms (int, optional): Marker size for the plots. Defaults to 10.
        figsize_base (tuple, optional): Base figure size. Defaults to (20, 10).
    Returns:
        None
    """
    beg_time = datetime.now()

    # Check if the model path is empty and exit if true
    if model_path == Path(""):
        print("No folder selected. Exiting...")
        sys.exit(0)

    # Define the root path for physical data
    if phys_data_path is not None:
        root = Path(phys_data_path)
    else:
        root = Path(f"./TFIM_datasets/")

    # Set default output labels if not provided
    if out_labels is None:
        if phys_model == "TFIM":
            out_labels = {
                "params": [r"$g$"],
            }
        else:
            raise ValueError("This model is not implemented yet!")

    # Create necessary directories for saving results
    model_path.joinpath("PNG").mkdir(exist_ok=True, parents=True)

    # ======================================
    #    Symbolic regression parameters
    # ======================================

    # Create directories for symbolic regression results
    model_sr_folder = model_path.joinpath("SR")
    model_sr_folder.mkdir(exist_ok=True, parents=True)

    sr_tmp_path = model_sr_folder.joinpath("Halls_of_fame")
    sr_tmp_path.mkdir(exist_ok=True, parents=True)

    # Set default symbolic regression parameters if not overridden
    if not corrs_sr_params_override:
        symbolic_reg_params = {
            "niterations": 400,  # < Increase me for better results
            "binary_operators": ["+", "*"],
            "unary_operators": [
                "neg",
            ],
            "constraints": {"*": (1, 1)},
            "model_selection": "best",
            "maxdepth": 10,
            "tempdir": sr_tmp_path,
        }
    else:
        symbolic_reg_params = corrs_sr_params_override

    # Set default predictor symbolic regression parameters if not overridden
    if not predictor_sr_params_override:
        predictor_symbolic_reg_params = {
            "niterations": 400,  # < Increase me for better results
            "binary_operators": ["+", "*"],
            "unary_operators": [
                "neg",
                "square",
                "abs",
                "inv(x) = 1/x",
            ],
            "constraints": {
                "*": (1, 1),
                "square": 3,
            },
            "nested_constraints": {
                "square": {"square": 1},
            },
            "complexity_of_operators": {
                "neg": 1,
                "abs": 1,
                "square": 2,
                "inv": 3,
            },
            "extra_sympy_mappings": {"inv": lambda x: 1 / x},
            "model_selection": "best",
            "maxdepth": 12,
            "tempdir": sr_tmp_path,
        }

    else:
        predictor_symbolic_reg_params = predictor_sr_params_override

    # A development flag to check plotting of kernels and the threshold but not to perform symbolic regression
    # skip_sr = True

    # ======================================
    # Dependent parameters, read from files
    # ======================================

    # Extract regression type from model path name
    reg = re.search("Regression=[a-zA-Z]*", model_path.name).group(0).split("=")[1]

    # Read parameters from params.txt file
    params_file = model_path.joinpath("params.txt")
    with open(params_file, "r") as f:
        for line in f:
            if "Used kernels" in line:
                tested_kernels_list = ast.literal_eval(line.split(":")[1].strip())
            if "Used bases" in line:
                snaps = list(line.split(":")[1].strip())

    # Determine various parameters based on the kernels used
    n_tested_kernels = len(tested_kernels_list)
    max_area = max([k[0][0] * k[0][1] for k in tested_kernels_list])
    max_dil = max([k[2] for k in tested_kernels_list])
    max_filt_num = max([k[1] for k in tested_kernels_list])
    out_feat_num = np.array([f_num for _, f_num, _ in tested_kernels_list]).sum()
    try:
        backportion_depth = int(
            re.search("backportion_depth=[0-9]", model_path.name).group(0).split("=")[1]
        )
    except AttributeError:
        backportion_depth = 0

    # Determine output dimensions
    out_dims = 1
    if reg is not None:
        out_dims = len(out_labels[reg])
    n_channels = len(snaps)

    # Set device to CPU
    device = torch.device("cpu")

    # Set activation function based on output dimensions
    if out_dims == 1:
        activation = torch.sigmoid
    else:
        activation = torch.softmax

    # Set loss criterion
    criterion = nn.MSELoss()

    # ======================================
    #       Data loading and model
    # ======================================

    # Data loading
    importer_class = Importer
    exp = importer_class(
        root,
        BS,
        reg=bool(reg is not None),
        snaps=snaps,
        device=device,
        phys_model=phys_model,
        phase=phys_phase,
        rescale_y=normalize_reg,
    )

    # Get data loaders
    trn_loader = exp.get_train_loader(batch_size=BS, shuffle=True)
    val_loader = exp.get_val_loader()
    test_loader = exp.get_test_loader()

    # Model definition
    backportion_dims_dict = {
        0: [out_feat_num, 32, out_dims],
        1: [out_feat_num, 32, 16, out_dims],
        2: [out_feat_num, 32, 64, 32, 8, 4, out_dims],
        3: [out_feat_num, 32, 64, 32, 16, 8, 4, out_dims],
    }
    backportion_dims = backportion_dims_dict[backportion_depth]

    # Initialize models
    net = ShapeAdaptiveConvNet(
        input_dim=n_channels, shape_list=tested_kernels_list, device=device
    )
    classifier = SmallModel(backportion_dims, device=device)

    # Load model weights
    net.load_state_dict(torch.load(f"{str(model_path)}/{model_name}_net.dict"))
    classifier.load_state_dict(
        torch.load(f"{str(model_path)}/{model_name}_classifier.dict")
    )

    # Move models to device
    net.to(device)
    classifier.to(device)

    # Set models to evaluation mode
    net.eval()
    classifier.eval()

    # =========================
    #     MODEL TESTING
    # =========================

    # For now it is hardcoded for TFIM, will be changed later
    # to a more universal solution
    loader = test_loader
    time_tab = exp.test_g_tab
    time_unique = np.unique(time_tab)
    exp.time = time_unique

    activation_criterion = activation if reg is None else criterion

    # Apply the model to the test data and get auxiliary information
    aux_dict = apply_model_to_test_data(
        loader,
        [net, classifier],
        device,
        activation_criterion,
        reg=reg,
        out_dims=out_dims,
        return_auxiliary=True,
        exp=exp,
        averaging_tab=time_tab,
    )

    logits_list = aux_dict["logits_list"]

    # Convert logits to numpy array
    logits_numpy = logits_list.detach().cpu().numpy()

    # Reshape logits to be averaged per measurement point
    logits_avg, logits_std = reshape_by_time(logits_numpy, time_tab)

    # Pruning threshold for the smaller correlations
    # None - no pruning. Not advised for 2-dimensional kernels of area greater than 4
    # float - pruning of smaller correlations enabled, to the precision given by the threshold. Optimal value is 1e-6
    alive_thresh = 0.1 * np.abs(logits_avg).max()
    # alive_thresh = 1e-10

    # Get all kernels and their activations
    all_kernels = [k for k in tested_kernels_list]
    all_activations = logits_avg.T
    all_activations_std = logits_std.T

    # Sorting the kernels by the highest absolute value of the activation
    # This line specifically sorts them lowest to highest. Therefore we must reverse the order of arrays sorted with it later.
    all_new_kernel_order = np.argsort(np.abs(all_activations).max(axis=1))[::-1]
    all_activations = all_activations[all_new_kernel_order]
    all_activations_std = all_activations_std[all_new_kernel_order]
    all_new_kernel_activation_amps = np.abs(all_activations).max(axis=1)

    # Retrieving the indices of the kernels that are above the threshold
    alive_mask = all_new_kernel_activation_amps > alive_thresh
    alive_kernel_inds = all_new_kernel_order[alive_mask]

    # Extracting the alive kernels and activations
    alive_kernels = [all_kernels[i] for i in alive_kernel_inds]
    alive_activations = all_activations[alive_mask]
    alive_activations_std = all_activations_std[alive_mask]

    # Reordering all the kernel labels by the new order
    all_kernels = [all_kernels[i] for i in all_new_kernel_order]

    # Plotting all kernels and alive kernels
    fig, axs = plt.subplots(1, 2, figsize=(figsize_base[0], figsize_base[1]))
    ax = axs[0]
    color_tab = [cm["tab10"](num) for num in range(n_tested_kernels)]
    for num, kernel in enumerate(all_kernels):
        color = color_tab[num]
        ax.plot(
            time_unique,
            all_activations[num],
            ".-",
            label=f"{kernel} | "
            + r"$a_{{{}}} = $".format(num)
            + f"{max(abs(all_activations[num].max()), abs(all_activations[num].min())):.3f}",
            color=color,
            linewidth=lw,
            markersize=ms,
        )
        ax.fill_between(
            time_unique,
            all_activations[num] - all_activations_std[num],
            all_activations[num] + all_activations_std[num],
            alpha=0.3,
            color=color,
        )
    ax.hlines(
        alive_thresh,
        time_unique.min(),
        time_unique.max(),
        color="crimson",
        linestyle="--",
        label=f"Threshold = {alive_thresh:.3f}",
        linewidth=lw,
    )
    ax.hlines(
        -alive_thresh,
        time_unique.min(),
        time_unique.max(),
        color="crimson",
        linestyle="--",
        linewidth=lw,
    )
    ax.hlines(
        0,
        time_unique.min(),
        time_unique.max(),
        color="k",
        linestyle="--",
        linewidth=lw,
    )
    ax.set_title(f"All kernels | Threshold {alive_thresh:.3f}", fontsize=fs)
    ax.legend(fontsize=0.5 * fs)

    ax = axs[1]
    for num, kernel in enumerate(alive_kernels):
        color = color_tab[num]
        ax.plot(
            time_unique,
            alive_activations[num],
            ".-",
            label=f"No. {num}: {kernel}",
            color=color,
            linewidth=lw,
            markersize=ms,
        )
        ax.fill_between(
            time_unique,
            alive_activations[num] - alive_activations_std[num],
            alive_activations[num] + alive_activations_std[num],
            alpha=0.3,
            color=color,
        )
    ax.hlines(
        alive_thresh,
        time_unique.min(),
        time_unique.max(),
        color="crimson",
        linestyle="--",
        label="Threshold",
        linewidth=lw,
    )
    ax.hlines(
        -alive_thresh,
        time_unique.min(),
        time_unique.max(),
        color="crimson",
        linestyle="--",
        linewidth=lw,
    )
    ax.hlines(
        0,
        time_unique.min(),
        time_unique.max(),
        color="k",
        linestyle="--",
        linewidth=lw,
    )
    ax.set_title(f"Alive kernels | Threshold {alive_thresh:.3f}", fontsize=fs)
    ax.legend(fontsize=0.5 * fs)

    for ax in axs:
        ax.set_xlabel("Time", fontsize=fs)
        ax.set_ylabel(r"$a_k$", fontsize=fs)
        ax.tick_params(axis="both", which="major", labelsize=fs)
        # add minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
        ax.grid(which="major", color="gray", linestyle="dashdot", linewidth=0.7)
    fig.savefig(model_path.joinpath("Alive_kernels.pdf"))
    plt.close()

    # Check if approximation is enabled
    if approx:
        num_approx = alive_activations.shape[0]
    else:
        num_approx = 1

    approx_fname = "test_stats_approx.txt"

    # If regression type is specified, write headers to the approximation file
    if reg is not None:
        with open(model_path.joinpath(approx_fname), "w") as f:
            s_names = [
                "Approximation number",
                "MSE",
                "Full MSE",
                "MSE % error",
                "R^2",
                "Full R^2",
                "R^2 % error",
            ]
            line_centered = f"{s_names[0]:^20} | "
            line_centered += " | ".join([f"{s:^11}" for s in s_names[1:]])
            f.write(line_centered + "\n")

        # Assess MSE and R^2 for the given approximation up to 3rd order
        for approx_num in range(3):
            X_assesment = logits_list.detach().cpu().numpy().copy()

            # Handle case where the highest activation kernel might be almost constant
            swapped_first_kernel = False
            if approx_num == 0:
                var_first_kernel = X_assesment[:, all_new_kernel_order[0]].var()
                var_second_kernel = X_assesment[:, all_new_kernel_order[1]].var()
                if var_first_kernel < var_second_kernel:
                    not_masked_columns = all_new_kernel_order[1]
                    swapped_first_kernel = True
                else:
                    not_masked_columns = all_new_kernel_order[0]
            else:
                not_masked_columns = all_new_kernel_order[: approx_num + 1]

            # Mask non-approximated kernels
            mask_assesment = np.ones(X_assesment.shape[1])
            mask_assesment[not_masked_columns] = False
            mask_assesment = mask_assesment.astype(bool)

            # Set non-approximated kernels to machine epsilon
            X_assesment[:, mask_assesment] = np.finfo(np.float64).eps
            X_assesment = torch.tensor(X_assesment).float().to(device)
            probs_assesment = classifier(X_assesment).detach().cpu().numpy()

            # Reshape probabilities to average per measurement point
            probs_assesment_avg, _ = reshape_by_time(probs_assesment, time_tab)

            # Calculate R^2 and MSE
            r2, mse = calculate_r2_and_mse(
                aux_dict["truth"],
                probs_assesment,
                time_tab=time_tab,
                exp=exp,
                old_way=False,
            )

            full_mse = aux_dict["MSE"]
            full_r2 = aux_dict["R^2"]

            # Print and log the results
            if approx_num == 0 and swapped_first_kernel and len(alive_kernels) > 1:
                print(f"Left kernels: {alive_kernels[1]}")
            else:
                print(f"Left kernels: {alive_kernels[:approx_num+1]}")
            print(
                f"Approximation no. {approx_num + 1} | MSE: {mse:^8.6f} | Full MSE: {full_mse:^8.6f} | MSE percent error: {abs(mse-full_mse)/full_mse*100:^6.1f}% | R^2: {r2:^5.3f} | Full R^2: {full_r2:^5.3f}  | R^2 percent error: {abs(r2-full_r2)/full_r2*100:^6.1f}%"
            )

            with open(model_path.joinpath(approx_fname), "a") as f:
                f.write(
                    f"{approx_num + 1:^20} | {mse:^11.8f} | {full_mse:^11.8f} | {abs(mse-full_mse)/full_mse*100:^11.1f} | {r2:^11.8f} | {full_r2:^11.8f} | {abs(r2-full_r2)/full_r2*100:^11.1f}\n"
                )

        # Log the full MSE and R^2
        with open(model_path.joinpath(approx_fname), "a") as f:
            f.write(
                f"{'Full':^20} | {full_mse:^11.8f} | {full_mse:^11.8f} | {abs(full_mse-full_mse)/full_mse*100:^11.1f} | {full_r2:^11.8f} | {full_r2:^11.8f} | {abs(full_r2-full_r2)/full_r2*100:^11.1f}\n"
            )
            f.write("\n")

    # If symbolic regression is skipped, exit the program
    if skip_sr:
        print("Exiting...")
        sys.exit()

    figsize_base = (2 * figsize_base[0], 2 * figsize_base[1])

    final_fig, final_axs = plt.subplots(
        1,
        3,
        figsize=(figsize_base[0], figsize_base[1] * 0.5),
        gridspec_kw={"wspace": 0.3},
    )

    # ===============================================
    #   Mapping between correlators and activations
    # ===============================================

    if not skip_corrs:
        # Initialize dictionary to store correlations and labels for each kernel
        kernels_corrs_labels_dict = {}

        # Clone the test dataset and move it to the device
        tt = test_loader.dataset.X.clone().float().to(device)
        try:
            print(
                f"Correlation pruning: {just_new} | Kernel alive threshold: {alive_thresh:.3f}"
            )
        except TypeError:
            print(
                f"Correlation pruning threshold: None | Kernel alive threshold: {alive_thresh:.3f}"
            )

        # Iterate over each alive kernel
        for num, kernel in enumerate(alive_kernels):
            ker_shape, k_copies, k_dil = kernel
            kernel_lab = str(kernel)
            print(
                f"Kernel no. {num} | activation = {max(abs(alive_activations[num].max()), abs(alive_activations[num].min())):.3f} | shape {ker_shape} | {k_copies} copies | dilation {k_dil}"
            )

            # Determine if pruning is needed based on kernel volume
            volume = ker_shape[0] * ker_shape[1] * tt.shape[1]
            prune = volume >= 9 or just_new

            # Generate correlation computation instructions for a given kernel
            shifts_dict_list = shifts_from_bounds(
                *ker_shape,
                tt.shape[1],
                just_new=prune,
                dilation=k_dil,
                include_origin=False,
            )

            # Initialize dictionary for the current kernel
            kernels_corrs_labels_dict[kernel_lab] = {
                "activations": [],
                "activations_std": [],
                "corrs": [],
                "corrs_std": [],
                "labels": [],
            }
            max_tmp = 0
            kernels_corrs_labels_dict[kernel_lab]["activations"] = alive_activations[
                num
            ]
            kernels_corrs_labels_dict[kernel_lab][
                "activations_std"
            ] = alive_activations_std[num]

            # If there are many possible correlators, use tqdm for tracking generation progress
            if len(shifts_dict_list) > 1e3:
                iterator = tqdm(
                    range(len(shifts_dict_list)),
                    desc="Correlators",
                    position=0,
                    leave=False,
                )
                info_bar = tqdm(total=1, position=1, leave=False, bar_format="{desc}")
                info_bar.set_description("Picked 0 correlators at threshold 0")
                n_corr = 0
            else:
                iterator = range(len(shifts_dict_list))

            # Iterate over each shift
            for c_ind in iterator:
                shifts_tmp = shifts_dict_list[c_ind]
                i_tab = shifts_tmp["i"]
                j_tab = shifts_tmp["j"]
                basis_tab = shifts_tmp["channel"]

                # Ensure the origin is included in the shifts
                if not (
                    np.array_equal(i_tab, np.array([0]))
                    and np.array_equal(j_tab, np.array([0]))
                ):
                    i_tab = np.concatenate([np.array([0]), i_tab])
                    j_tab = np.concatenate([np.array([0]), j_tab])
                    basis_tab = np.concatenate([np.array([0]), basis_tab])

                # Convert basis indices to snapshot names
                basis_tab = np.array([snaps[b] for b in basis_tab])

                # Determine the label for the correlation
                lab = determine_corr_label(
                    i_tab, j_tab, basis_tab=basis_tab, dollars=True
                )

                # Compute the correlation and its standard deviation
                corrs, std = get_corr(
                    tt,
                    1,
                    nreals=exp.trn_val_test_sizes["test"] + [1000],
                    corr_inds=shifts_tmp,
                    std_dev=True,
                    rmse=False,  # IMPORTANT! If set to true, the correlations cannot be fit for the correlator.
                    dim=1,
                )

                # Update the maximum correlation value if needed
                if np.abs(corrs).max() > max_tmp:
                    max_tmp = np.abs(corrs).max()
                    if len(shifts_dict_list) > 1e3:
                        info_bar.set_description(
                            f"Picked {n_corr} correlators at threshold 0.5 * correlations_max_value = {max_tmp:.3f}"
                        )

                # ================================================================================

                # We only choose to prune the correlations if their number is greater than 1e+3
                # or the maximum correlation is greater than a fraction of the maximum correlation.abs().max()
                # This is to avoid the situation where we have thousands of almost irrelevant correlations,
                # which would make the symbolic regression impossible to perform.
                # A definite bound on the number of correlations is capacity of UInt16 = 65536.
                # This is a constraint of the PySR library.

                condition = (np.abs(corrs).max() > 0.5 * max_tmp) or len(
                    shifts_dict_list
                ) < 1e3

                # ================================================================================

                if condition:
                    # Append the correlations and their standard deviations to the dictionary
                    kernels_corrs_labels_dict[kernel_lab]["corrs"].append(corrs)
                    kernels_corrs_labels_dict[kernel_lab]["corrs_std"].append(std)
                    # Remove dollar signs from the label and append it to the dictionary
                    lab = lab.replace("$", "")
                    kernels_corrs_labels_dict[kernel_lab]["labels"].append(lab)
                    # Update the progress bar if there are many correlators
                    if len(shifts_dict_list) > 1e3:
                        n_corr += 1
                        info_bar.set_description(
                            f"Picked {n_corr} correlators at threshold 0.5 * correlations_max_value = {max_tmp:.3f}"
                        )
                else:
                    pass
            # Convert the list of correlations to a numpy array
            kernels_corrs_labels_dict[kernel_lab]["corrs"] = np.array(
                kernels_corrs_labels_dict[kernel_lab]["corrs"]
            )

        # Iterate over each kernel and perform symbolic regression
        for ker_num, kernel_lab in enumerate(kernels_corrs_labels_dict.keys()):
            kernel = ast.literal_eval(kernel_lab)
            ker_shape, k_copies, k_dil = kernel

            # Prepare the data for symbolic regression
            X = kernels_corrs_labels_dict[kernel_lab]["corrs"].T
            y = kernels_corrs_labels_dict[kernel_lab]["activations"]

            corr_tab = kernels_corrs_labels_dict[kernel_lab]["corrs"]
            corr_std_tab = kernels_corrs_labels_dict[kernel_lab]["corrs_std"]
            labels_tab = kernels_corrs_labels_dict[kernel_lab]["labels"]

            # Create a dictionary for the labels
            labels_dict = {
                "x_{{{}}}".format(num): lab for num, lab in enumerate(labels_tab)
            }
            labels_dict = dict((re.escape(k), v) for k, v in labels_dict.items())
            pattern = re.compile("|".join(labels_dict.keys()))

            # Adjust symbolic regression parameters based on the number of features
            tmp_symbolic_reg_params = symbolic_reg_params.copy()
            if X.shape[1] > 10:
                tmp_symbolic_reg_params["niterations"] *= 3
                tmp_symbolic_reg_params["maxdepth"] *= 3

            # Set the file name for the symbolic regression equations
            ker_eq_file_name = f"sr_fit_ker_{ker_num}"
            tmp_symbolic_reg_params["equation_file"] = sr_tmp_path.joinpath(
                f"{ker_eq_file_name}.csv"
            )

            # Set constraints for 1x1 kernels
            if ker_shape == (1, 1):
                tmp_symbolic_reg_params["constraints"] = {"mult": (1, 1)}
            else:
                pass

            # Initialize the symbolic regression model
            model = PySRRegressor(**tmp_symbolic_reg_params)
            if sr_fit:
                # Fit the model to the data
                model.fit(X, y)
            else:
                # Load the model from a file
                model = model.from_file(sr_tmp_path.joinpath(f"{ker_eq_file_name}.pkl"))

            # Get the hall of fame equations from the model
            eqs = model.get_hof()

            # Plotting the full set of equations
            plot_condition = True

            if plot_condition:
                fig = plt.figure(figsize=(figsize_base[1], figsize_base[0]))
                axs = fig.subplot_mosaic([["corrs"], ["eqs_all"], ["act_vs_best"]])
            else:
                fig = plt.figure(figsize=(figsize_base[0], figsize_base[1]))
                fig.subplots_adjust(top=0.8)
                axs = fig.subplot_mosaic(
                    [["corrs", "eqs_all"], ["act_vs_best", "eqs_best"]]
                )
            r2_tab = np.array([r2_score(y, eq(X)) for eq in eqs.lambda_format])

            eqs["r2"] = np.round(r2_tab, 2)

            eqs_sorted = eqs.sort_values(
                by=["r2", "complexity"], ascending=[False, True]
            )

            # Plot all fitted equations vs true values
            for ax in [axs["eqs_all"], final_axs[0]]:
                used_corr_inds = set()
                ax.plot(y, y, "k--", label="Baseline", linewidth=lw)
                already_plotted_sp = []
                for num, eq in enumerate(eqs.lambda_format):
                    color = plt.cm.viridis(num / len(eqs.lambda_format))
                    eq_lab = model.latex(num)
                    eq_sp = latex2sympy(eq_lab)
                    if eq_sp in already_plotted_sp:
                        continue
                    already_plotted_sp.append(eq_sp)
                    for found_x in re.findall(r"x_{\d+}", eq_lab):
                        matched = re.search(r"\d+", found_x)
                        if matched is not None:
                            used_corr_inds.update([int(matched.group())])
                    ax.plot(
                        y,
                        eq(X),
                        ".",
                        label=r"${}$".format(eq_lab)
                        + " | "
                        + r"$R^2=$"
                        + f"{r2_tab[num]:.3f}",
                        color=color,
                        linewidth=lw,
                        markersize=ms,
                    )
                ax.set_xlabel("Truth", fontsize=fs)
                ax.set_ylabel("Prediction", fontsize=fs)
                ax.set_title("All fitted equations vs true values", fontsize=fs)

            # Plot best symbolic regression equations vs target activation
            for ax in [axs["act_vs_best"], final_axs[1]]:
                ax.plot(exp.time, y, ".--", label="Target", linewidth=lw)
                best_eq_num = 3
                i = 0
                already_plotted_sp = []
                if len(eqs_sorted) < best_eq_num:
                    best_eq_num = len(eqs_sorted)
                for num in range(best_eq_num):
                    eq = eqs_sorted.iloc[i].lambda_format
                    eq_ind = eqs_sorted.index[i]
                    eq_sp = eqs_sorted.iloc[i].sympy_format
                    eq_sp = round_expr(eq_sp, 3)
                    eq_sp = sp.simplify(eq_sp)
                    if eq_sp in already_plotted_sp:
                        if i < len(eqs_sorted) - 1:
                            i += 1
                        continue
                    already_plotted_sp.append(eq_sp)
                    eq_lab = sp.latex(eq_sp)
                    eq_lab = pattern.sub(
                        lambda m: labels_dict[re.escape(m.group(0))], eq_lab
                    )
                    color = plt.cm.viridis(i / 3)
                    ax.plot(
                        exp.time,
                        eq(X),
                        ".-",
                        label=r"${}$".format(eq_lab)
                        + " | "
                        + r"$R^2=$"
                        + f"{r2_tab[eq_ind]:.3f}",
                        color=color,
                        linewidth=lw,
                        markersize=ms,
                    )
                    i += 1

                # Perform least squares fit if there are few labels
                if len(labels_tab) < 1e2:
                    X_lsq = np.hstack([X, np.ones(len(X)).reshape(-1, 1)])

                    coeffs = np.linalg.lstsq(X_lsq, y, rcond=None)[0]
                    r2_score(y, X_lsq @ coeffs.T)
                    lt = labels_tab.copy()
                    lt += [""]
                    eq_raw = np.vstack([coeffs, lt]).T[::-1]
                    eq = r"$"
                    for coeff, lab in eq_raw:
                        if round(float(coeff), 3) != 0:
                            eq += r"{:+.3f}{}".format(float(coeff), lab)
                    eq += r"$"
                    for found_x in re.findall(r"x_{\d+}", eq):
                        matched = re.search(r"\d+", found_x)
                        if matched is not None:
                            used_corr_inds.update([int(matched.group())])
                    ax.plot(
                        exp.time,
                        X_lsq @ coeffs.T,
                        ".--",
                        color="crimson",
                        label=f"Least squares fit: R^2 = {r2_score(y, X_lsq@coeffs.T):.2f}\n"
                        + r"$y=$"
                        + eq,
                    )

                ax.set_xlabel("Time", fontsize=fs)
                ax.set_ylabel("Value", fontsize=fs)
                ax.set_title(
                    "Best symbolic regression equations vs target activation",
                    fontsize=fs,
                )

            # Plot all correlations fitted to the kernel
            ax = axs["corrs"]
            for corr_num, (corr, std) in enumerate(zip(corr_tab, corr_std_tab)):
                if corr_num in used_corr_inds:
                    ax.plot(
                        exp.time,
                        corr,
                        ".-",
                        label="$" + labels_tab[corr_num] + "$",
                        linewidth=lw,
                        markersize=ms,
                    )
                    ax.fill_between(exp.time, corr - std, corr + std, alpha=0.3)
            ax.set_xlabel("Time", fontsize=fs)
            ax.set_ylabel("Value", fontsize=fs)
            ax.set_title("All correlations fitted to the kernel", fontsize=fs)

            # Plot best fitted equations vs true values if not in plot condition
            if not plot_condition:
                ax = axs["eqs_best"]

                ax.plot(y, y, "k--", label="Baseline", linewidth=lw)

                best_eq_num = 3
                i = 0
                already_plotted_sp = []
                if len(eqs_sorted) < best_eq_num:
                    best_eq_num = len(eqs_sorted)
                for num in range(best_eq_num):
                    eq = eqs_sorted.iloc[i].lambda_format
                    eq_ind = eqs_sorted.index[i]
                    eq_sp = eqs_sorted.iloc[i].sympy_format
                    eq_sp = round_expr(eq_sp, 3)
                    eq_sp = sp.simplify(eq_sp)
                    if eq_sp in already_plotted_sp:
                        if i < len(eqs_sorted) - 1:
                            i += 1
                        continue
                    already_plotted_sp.append(eq_sp)
                    eq_lab = sp.latex(eq_sp)
                    eq_lab = pattern.sub(
                        lambda m: labels_dict[re.escape(m.group(0))], eq_lab
                    )
                    color = plt.cm.viridis(i / 3)
                    ax.plot(
                        y,
                        eq(X),
                        ".",
                        label=r"${}$".format(eq_lab)
                        + " | "
                        + r"$R^2=$"
                        + f"{r2_tab[eq_ind]:.3f}",
                        color=color,
                        linewidth=lw,
                        markersize=ms,
                    )
                    i += 1
                ax.set_xlabel("Truth", fontsize=fs)
                ax.set_ylabel("Prediction", fontsize=fs)
                ax.set_title("Best fitted equations vs true values", fontsize=fs)

            # Set legend and grid for each subplot
            for num, ax in enumerate(axs.values()):
                if num == 0 and not plot_condition:
                    ax.legend(
                        fontsize=0.8 * fs,
                        loc="center right",
                        bbox_to_anchor=(-0.15, 0.5),
                        ncol=(len(ax.lines) // 8) + 1,
                    )
                elif num == 1 or (plot_condition and num == 0):
                    ax.legend(
                        fontsize=0.8 * fs,
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                        ncol=(len(ax.lines) // 8) + 1,
                    )
                else:
                    ax.legend(
                        fontsize=0.8 * fs,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.15),
                    )
                ax.tick_params(axis="both", which="major", labelsize=fs)
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
                ax.grid(
                    which="major",
                    color="gray",
                    linestyle="dashdot",
                    linewidth=0.7,
                )

            # Determine if pruning is needed based on kernel volume
            volume = ker_shape[0] * ker_shape[1] * tt.shape[1]
            if volume >= 9:
                prune = True
            else:
                prune = just_new

            # Set the title for the figure
            fig.suptitle(
                f"Symbolic Regression fit\nCorrelations in data to kernel activation\nKernel {ker_shape} | {k_copies} copies | dilation {k_dil} | Pruning {prune} | Kernel alive threshold: {alive_thresh:.3f}",
                size=1.5 * fs,
            )

            # Save the figure as PDF and PNG
            fig.savefig(
                model_sr_folder.joinpath(f"SR_kernel_{ker_num}_corrs.pdf"),
                bbox_inches="tight",
            )
            fig.savefig(
                model_path.joinpath("PNG").joinpath(f"SR_kernel_{ker_num}_corrs.png"),
                bbox_inches="tight",
            )

            plt.close(fig)

    # ===============================================
    #  Mapping between activations and final outputs
    # ===============================================
    # Copy all activations and their standard deviations
    activations = all_activations.copy().T
    activations_std = all_activations_std.copy().T

    # Reshape probabilities and truth values to average per measurement point
    probs, probs_std = reshape_by_time(aux_dict["probs"], time_tab)
    truth, truth_std = reshape_by_time(aux_dict["truth"], time_tab)

    # Set X and y for symbolic regression
    X = activations.copy()
    y = probs.copy()

    # Define the equation file name for all activations
    eq_file_name = "sr_act_fit_all"
    predictor_symbolic_reg_params_all = predictor_symbolic_reg_params.copy()
    predictor_symbolic_reg_params_all["equation_file"] = sr_tmp_path.joinpath(
        f"{eq_file_name}.csv"
    )
    model_all = PySRRegressor(**predictor_symbolic_reg_params_all)

    # Fit the model or load from file
    if sr_fit:
        model_all.fit(X, y)
    else:
        model_all = model_all.from_file(sr_tmp_path.joinpath(f"{eq_file_name}.pkl"))

    # Loop over the number of approximations
    for approx_num in range(num_approx):
        # =====================================================
        # Assessment of MSE and R^2 for the given approximation
        # Set not-approximated kernels to (numeric) 0 - Machine epsilon
        # Numeric 0 --> np.finfo(np.float64).eps

        X_assesment = logits_list.detach().cpu().numpy().copy()
        """
        We might get a 1x1 kernel as the one with the highest activation,
        but to be almost a constant at the same time. To counter it, for the
        single-kernel approximations, we will take the second highest activation.
        This criterion is based on the variance of the kernel activations.
        For now this version is turned off in individual analysis.
        """
        swapped_first_kernel = False
        not_masked_columns = all_new_kernel_order[: approx_num + 1]
        mask_assesment = np.ones(X_assesment.shape[1])
        mask_assesment[not_masked_columns] = False
        mask_assesment = mask_assesment.astype(bool)

        # Set not-approximated kernels to (numeric) 0 - Machine epsilon
        # Numeric 0 --> np.finfo(np.float64).eps
        X_assesment[:, mask_assesment] = np.finfo(np.float64).eps
        X_assesment = torch.tensor(X_assesment).float().to(device)
        probs_assesment = classifier(X_assesment).detach().cpu().numpy()

        # Reshape probabilities by time
        probs_assesment_avg, _ = reshape_by_time(probs_assesment, time_tab)

        # Calculate R^2 and MSE
        r2, mse = calculate_r2_and_mse(
            aux_dict["truth"],
            probs_assesment,
            time_tab=time_tab,
            exp=exp,
            old_way=False,
        )

        full_mse = aux_dict["MSE"]
        full_r2 = aux_dict["R^2"]

        # =====================================================

        predictor_symbolic_reg_params_alive = predictor_symbolic_reg_params.copy()

        # Set equation file name and prepare data for alive activations
        if approx:
            eq_file_name_alive = f"sr_act_fit_alive_approx_{approx_num+1}"
            if approx_num == 0 and swapped_first_kernel:
                X_alive = alive_activations[1].copy().reshape(-1, 1)
                X_alive_std = alive_activations_std[1].copy().reshape(-1, 1)
            else:
                X_alive = alive_activations[: approx_num + 1].copy().T
                X_alive_std = alive_activations_std[: approx_num + 1].copy().T
            predictor_symbolic_reg_params_alive["equation_file"] = sr_tmp_path.joinpath(
                f"{eq_file_name_alive}.csv"
            )
        else:
            eq_file_name_alive = "sr_act_fit_alive"
            X_alive = alive_activations.copy().T
            X_alive_std = alive_activations_std.copy().T
            predictor_symbolic_reg_params_alive["equation_file"] = sr_tmp_path.joinpath(
                f"{eq_file_name_alive}.csv"
            )

        # Fit the model or load from file for alive activations
        model_alive = PySRRegressor(**predictor_symbolic_reg_params_alive)
        if sr_fit:
            model_alive.fit(X_alive, y)
        else:
            model_alive = model_alive.from_file(
                sr_tmp_path.joinpath(f"{eq_file_name_alive}.pkl")
            )

        # Get the hall of fame equations
        eq_tab = model_all.get_hof()
        eq_tab_alive = model_alive.get_hof()

        # Ensure eq_tab and eq_tab_alive are lists
        if type(eq_tab) != list:
            eq_tab = [eq_tab]
        if type(eq_tab_alive) != list:
            eq_tab_alive = [eq_tab_alive]

        fig = plt.figure(
            figsize=(
                figsize_base[0],
                0.5 * figsize_base[1] * (1.25 + len(eq_tab)),
            )
        )
        if len(eq_tab) > 1:
            fig.subplots_adjust(top=0.85)
        else:
            fig.subplots_adjust(top=0.75)

        mosaic = [["acts_and_target(s)", "acts_alive_and_target(s)"]]
        for i in range(0, len(eq_tab)):
            mosaic.append([f"best_act_vs_target_{i}", f"alive_best_act_vs_target_{i}"])
        axs = fig.subplot_mosaic(mosaic)

        labels_for_probs = out_labels[reg]

        acts_color_tab = [cm["tab10"](num) for num in range(activations.shape[1])]
        targets_color_tab = plt.cm.Dark2(np.arange(0, probs.shape[1]))
        truth_color_tab = plt.cm.Dark2(np.arange(0, truth.shape[1]))

        ax = axs["acts_and_target(s)"]

        for num, (act, std) in enumerate(zip(activations.T, activations_std.T)):
            act_color = acts_color_tab[num]
            ax.plot(
                exp.time,
                act,
                ".-",
                label=r"$a_{{{}}}$".format(num),
                color=act_color,
                linewidth=lw,
                markersize=ms,
            )
            ax.fill_between(exp.time, act - std, act + std, alpha=0.3, color=act_color)

        for num, (prob, std, lab) in enumerate(
            zip(probs.T, probs_std.T, labels_for_probs)
        ):
            target_color = targets_color_tab[num]
            ax.plot(
                exp.time,
                prob,
                ".--",
                label="Predicted " + lab,
                linewidth=lw,
                markersize=ms,
                color=target_color,
            )
            ax.fill_between(exp.time, prob - std, prob + std, alpha=0.3)

        for num, (tr, std, lab) in enumerate(
            zip(truth.T, truth_std.T, labels_for_probs)
        ):
            truth_color = truth_color_tab[num]
            ax.plot(
                exp.time,
                tr,
                ".:",
                label="True " + lab,
                linewidth=lw,
                markersize=ms,
                color=truth_color,
            )
            ax.fill_between(exp.time, tr - std, tr + std, alpha=0.3)

        ax.set_title("All activations and target(s)", fontsize=fs)

        ax = axs["acts_alive_and_target(s)"]

        for num, (act, std) in enumerate(zip(X_alive.T, X_alive_std.T)):
            if approx_num == 0 and swapped_first_kernel:
                num += 1
            act_color = acts_color_tab[num]
            ax.plot(
                exp.time,
                act,
                ".-",
                label=r"$a_{{{}}} | $".format(num) + str(alive_kernels[num]),
                color=act_color,
                linewidth=lw,
                markersize=ms,
            )
            ax.fill_between(exp.time, act - std, act + std, alpha=0.3, color=act_color)

        for num, (prob, std, lab) in enumerate(
            zip(probs.T, probs_std.T, labels_for_probs)
        ):
            target_color = targets_color_tab[num]
            ax.plot(
                exp.time,
                prob,
                ".--",
                label="Predicted " + lab,
                linewidth=lw,
                markersize=ms,
                color=target_color,
            )
            ax.fill_between(exp.time, prob - std, prob + std, alpha=0.3)

        for num, (tr, std, lab) in enumerate(
            zip(truth.T, truth_std.T, labels_for_probs)
        ):
            truth_color = truth_color_tab[num]
            ax.plot(
                exp.time,
                tr,
                ".:",
                label="True " + lab,
                linewidth=lw,
                markersize=ms,
                color=truth_color,
            )
            ax.fill_between(exp.time, tr - std, tr + std, alpha=0.3)

        if approx and approx_num + 1 != num_approx:
            if approx_num < 3:
                order = {0: "st", 1: "nd", 2: "rd"}
            else:
                order = {i: "th" for i in range(3, 10)}
            ax.set_title(
                "${{{}}}^\mathrm{{{}}} $ order approximation and targets".format(
                    approx_num + 1, order[approx_num]
                ),
                fontsize=fs,
            )
        else:
            ax.set_title("Alive activations and target(s)", fontsize=fs)

        for eq_tab_num, (eqs, eqs_alive) in enumerate(zip(eq_tab, eq_tab_alive)):
            # Idk why but this throws a LinAlgError for some equations when model is loaded from file
            r2_tab = np.array(
                [
                    r2_score(y[:, eq_tab_num].reshape(-1, 1), eq(X))
                    for eq in eqs.lambda_format
                ]
            )
            r2_tab_alive = np.array(
                [
                    r2_score(y[:, eq_tab_num], eq(X_alive))
                    for eq in eqs_alive.lambda_format
                ]
            )

            eqs["r2"] = np.round(r2_tab, 2)
            eqs_alive["r2"] = np.round(r2_tab_alive, 2)

            eqs_sorted = eqs.sort_values(
                by=["r2", "complexity"], ascending=[False, True]
            )
            eqs_alive_sorted = eqs_alive.sort_values(
                by=["r2", "complexity"], ascending=[False, True]
            )

            # # =====================================================
            ax = axs[f"best_act_vs_target_{eq_tab_num}"]

            target_color = targets_color_tab[eq_tab_num]
            truth_color = truth_color_tab[eq_tab_num]

            if len(labels_tab) < 1e2:
                X_lsq = np.hstack([X, np.ones(len(X)).reshape(-1, 1)])
                y_lsq = y[:, eq_tab_num]

                lt = [r"a_{{{}}}".format(n) for n in range(X.shape[1])] + [""]

                print("Simplifying the least squares equation")
                coeffs_full = np.linalg.lstsq(X_lsq, y_lsq, rcond=None)[0]
                if r2_score(y_lsq, X_lsq @ coeffs_full.T) > 0.95:
                    coeffs = coeffs_full.copy()
                    for ind in range(-1, -coeffs_full.shape[0] - 1, -1):
                        coeffs_tmp = coeffs.copy()
                        coeffs_tmp[ind] = 0
                        term = lt[ind]
                        if term == "":
                            term = "Constant"
                        status_string = f"Without {term} | R^2 = {r2_score(y_lsq, X_lsq @ coeffs_tmp.T):.2f}"
                        if r2_score(y_lsq, X_lsq @ coeffs_tmp.T) > 0.95:
                            coeffs[ind] = 0
                            status_string += " -- Term removed"
                        else:
                            status_string += " -- Term stays"
                        print(status_string)
                    if r2_score(y_lsq, X_lsq @ coeffs.T) < 0.95:
                        coeffs[ind + 1] = coeffs_full[ind + 1]
                else:
                    coeffs = coeffs_full.copy()

                eq_raw = np.vstack([coeffs, lt]).T
                eq = r"$"
                for coeff, lab in eq_raw:
                    if round(float(coeff), 3) != 0:
                        eq += r"{:+.3f}{}".format(float(coeff), lab)
                eq += r"$"

                print("Least squares equation simplified: ", eq[1:-2])

                ax.plot(
                    exp.time,
                    X_lsq @ coeffs.T,
                    ".--",
                    color="crimson",
                    label=f"Least squares fit: R^2 = {r2_score(y_lsq, X_lsq@coeffs.T):.2f}\n"
                    + labels_for_probs[eq_tab_num]
                    + r"$=$"
                    + eq,
                )

            ax.plot(
                exp.time,
                y[:, eq_tab_num],
                ".--",
                label="Predicted " + labels_for_probs[eq_tab_num],
                linewidth=lw,
                markersize=ms,
                color=target_color,
            )
            ax.plot(
                exp.time,
                truth[:, eq_tab_num],
                ".:",
                label="True " + labels_for_probs[eq_tab_num],
                linewidth=lw,
                markersize=ms,
                color=truth_color,
            )

            best_eq_num = 3
            i = 0
            already_plotted_sp = []
            if len(eqs_sorted) < best_eq_num:
                best_eq_num = len(eqs_sorted)
            for num in range(best_eq_num):
                eq = eqs_sorted.iloc[i].lambda_format
                eq_ind = eqs_sorted.index[i]
                eq_sp = eqs_sorted.iloc[i].sympy_format
                eq_sp = round_expr(eq_sp, 3)
                eq_sp = sp.simplify(eq_sp)
                if eq_sp in already_plotted_sp:
                    if i < len(eqs_sorted) - 1:
                        i += 1
                    continue
                already_plotted_sp.append(eq_sp)
                eq_lab = sp.latex(eq_sp)
                eq_lab = eq_lab.replace("x", "a")
                color = plt.cm.viridis(i / 3)
                ax.plot(
                    exp.time,
                    eq(X),
                    ".-",
                    label=labels_for_probs[eq_tab_num]
                    + r"$ = {}$".format(eq_lab)
                    + " | "
                    + r"$R^2=$"
                    + f"{r2_tab[eq_ind]:.3f}",
                    color=color,
                    alpha=0.5,
                    linewidth=lw,
                    markersize=ms,
                )
                i += 1
            ax.set_title(
                f"Best symbolic regression for " + labels_for_probs[eq_tab_num],
                fontsize=fs,
            )

            # # =====================================================
            for final_num, ax in enumerate(
                [axs[f"alive_best_act_vs_target_{eq_tab_num}"], final_axs[2]]
            ):
                target_color = targets_color_tab[eq_tab_num]
                truth_color = truth_color_tab[eq_tab_num]

                ax.plot(
                    exp.time,
                    y[:, eq_tab_num],
                    ".--",
                    label="Predicted " + labels_for_probs[eq_tab_num],
                    linewidth=lw,
                    markersize=ms,
                    color=target_color,
                )

                ax.plot(
                    exp.time,
                    truth[:, eq_tab_num],
                    ".:",
                    label="True " + labels_for_probs[eq_tab_num],
                    linewidth=lw,
                    markersize=ms,
                    color=truth_color,
                )

                best_eq_num = 3
                if len(eqs_alive_sorted) < best_eq_num:
                    best_eq_num = len(eqs_alive_sorted)
                for i in range(best_eq_num):
                    eq = eqs_alive_sorted.iloc[i].lambda_format
                    eq_ind = eqs_alive_sorted.index[i]
                    eq_sp = eqs_alive_sorted.iloc[i].sympy_format
                    eq_sp = round_expr(eq_sp, 3)
                    eq_sp = sp.simplify(eq_sp)
                    eq_lab = sp.latex(eq_sp)
                    eq_lab = eq_lab.replace("x", "a")
                    if approx_num == 0 and swapped_first_kernel:
                        eq_lab = eq_lab.replace("_{0}", "_{1}")
                    color = plt.cm.viridis(i / 3)
                    ax.plot(
                        exp.time,
                        eq(X_alive),
                        ".-",
                        label=labels_for_probs[eq_tab_num]
                        + r"$ = {}$".format(eq_lab)
                        + " | "
                        + r"$R^2=$"
                        + f"{r2_tab_alive[eq_ind]:.3f}",
                        color=color,
                        alpha=0.5,
                        linewidth=lw,
                        markersize=ms,
                    )
                ax.set_title(
                    f"Best symbolic regression for "
                    + labels_for_probs[eq_tab_num]
                    + " (alive)",
                    fontsize=fs,
                )

                if final_num == 1:
                    ax.set_ylabel(r"Predicted $g$", fontsize=fs)

        for num, ax in enumerate(axs.values()):
            if num < len(axs.values()) - 1:
                if num % 2 == 1:
                    ax.legend(
                        fontsize=0.8 * fs,
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                    )
                else:
                    ax.legend(
                        fontsize=0.8 * fs,
                        loc="center right",
                        bbox_to_anchor=(-0.15, 0.5),
                    )
            else:
                ax.legend(
                    fontsize=0.8 * fs,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.2),
                )
            ax.set_xlabel("Time", fontsize=fs)
            ax.set_ylabel("Value", fontsize=fs)
            ax.tick_params(axis="both", which="major", labelsize=fs)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
            ax.grid(which="major", color="gray", linestyle="dashdot", linewidth=0.7)

        if reg is not None:
            fig.suptitle(
                f"Symbolic Regression fit\nKernel activations to network output\nPruning {just_new} | Kernel alive threshold: {alive_thresh:.3f}\nMSE: {mse:^6.4f} | Full MSE: {full_mse:^6.4f} | MSE % error: {abs(mse-full_mse)/full_mse*100:^6.1f}%\n"
                + "$R^2$"
                + f": {r2:^6.4f} | Full "
                + "$R^2$"
                + f": {full_r2:^6.4f}  | "
                + "$R^2$"
                + f" % error: {abs(r2-full_r2)/full_r2*100:^6.1f}%",
                size=1.5 * fs,
            )
        else:
            fig.suptitle(
                f"Symbolic Regression fit\nKernel activations to network output\nPruning {just_new} | Kernel alive threshold: {alive_thresh:.3f}",
                size=1.5 * fs,
            )

        if approx and approx_num + 1 != num_approx:
            fig.savefig(
                model_sr_folder.joinpath(f"SR_acts_vs_probs_approx_{approx_num+1}.pdf"),
                bbox_inches="tight",
            )
            fig.savefig(
                model_path.joinpath("PNG").joinpath(
                    f"SR_acts_vs_probs_approx_{approx_num+1}.png"
                ),
                bbox_inches="tight",
            )
        else:
            fig.savefig(
                model_sr_folder.joinpath(f"SR_acts_vs_probs.pdf"),
                bbox_inches="tight",
            )
            fig.savefig(
                model_path.joinpath("PNG").joinpath(f"SR_acts_vs_probs.png"),
                bbox_inches="tight",
            )

        plt.close(fig)

    let_tab = ["(a)", "(b)", "(c)"]
    for num, ax in enumerate(final_axs):
        ax.legend(
            fontsize=0.8 * fs,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
        )
        if num > 0:
            ax.set_xlabel(r"True $g$", fontsize=fs)
        else:
            ax.set_xlabel("True activation value", fontsize=fs)

        if num < 2:
            ax.set_ylabel("Predicted activation value", fontsize=fs)
        else:
            ax.set_ylabel(r"Predicted $g$", fontsize=fs)

        ax.set_title(f"{let_tab[num]}", fontsize=1.2 * fs, pad=20)

        ax.tick_params(axis="both", which="major", labelsize=fs)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
        ax.grid(which="major", color="gray", linestyle="dashdot", linewidth=0.7)

    neurips_figs_savefolder = Path("./NeurIPS_plots")
    neurips_figs_savefolder.mkdir(exist_ok=True, parents=True)
    neurips_figs_savefolder.joinpath("PNG").mkdir(exist_ok=True, parents=True)
    final_fig.savefig(
        neurips_figs_savefolder.joinpath(f"./TFIM_{''.join(snaps)}_results_fig_6.pdf"),
        bbox_inches="tight",
    )
    final_fig.savefig(
        neurips_figs_savefolder.joinpath("PNG").joinpath(f"./TFIM_{''.join(snaps)}_results_fig_6.png"),
        bbox_inches="tight",
    )

    print(
        f"Finished symbolic regression routine at {datetime.strftime(datetime.now(), format='%H:%M:%S, %d/%m/%Y')}."
    )
    end_time = datetime.now()
    print(f"Elapsed time: {humanize.naturaldelta(end_time - beg_time)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform symbolic regression on the model"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the model folder", default=""
    )
    parser.add_argument(
        "--approx",
        help="Fit the output to the consecutive highest activations of the alive kernels",
        default=None,
        action="count",
    )
    parser.add_argument(
        "--skip_corrs",
        help="Skip the correlations to activations mapping",
        default=None,
        action="count",
    )
    parser.add_argument(
        "--skip_sr_fit",
        help="Perform the symbolic regression fit or just plot the results",
        default=None,
        action="store_true",
    )
    parser.add_argument(
        "--phys_data_path",
        type=str,
        help="Path to the folder with experimental data",
        default=None,
    )

    args = parser.parse_args()
    print(args)

    phys_data_path = args.phys_data_path
    root_path = Path("./Models/")
    root_path.mkdir(exist_ok=True, parents=True)
    phys_model = "TFIM"

    filename = get_folder_gui(str(root_path), "Select the model folder")
    model_path = Path(filename)

    if model_path.name == root_path.name or model_path == Path(""):
        print("No folder selected. Exiting...")
        sys.exit(0)

    sr_routine(
        model_path,
        phys_data_path,
        phys_model=phys_model,
    )
