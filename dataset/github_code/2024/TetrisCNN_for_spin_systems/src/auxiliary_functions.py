import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import UnivariateSpline

from src.metrics import calculate_r2_and_mse

import contextlib
import itertools
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch
from prettytable.colortable import ColorTable, Theme
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
)
from termcolor import cprint

from src.metrics import calculate_r2_and_mse


def prediction_based_method(
    exp,
    truth,
    prediction,
    reg,
    phys_model,
    shape,
    logits_numpy,
    time_tab,
    pred_labels=None,
    defects=0,
    kernel_names=None,
    spl_k=4,
    spl_s=0,
    n_interp=100,
    save=None,
    merge_params=False,
):
    logits_df = pd.DataFrame(logits_numpy)
    logits_df["time"] = time_tab
    logits_avg = logits_df.groupby("time", sort=False).mean().to_numpy()
    logits_std = logits_df.groupby("time", sort=False).std().to_numpy()

    r2, mse = calculate_r2_and_mse(
        truth, prediction, time_tab=time_tab, exp=exp, old_way=False
    )

    if truth.ndim == 1:
        truth = truth.reshape(-1, 1)
        prediction = prediction.reshape(-1, 1)
    for num, (column_true, column_pred) in enumerate(zip(truth.T, prediction.T)):
        column_true, column_pred = column_true.T, column_pred.T
        if exp.y_scaler is not None:
            if type(exp.y_scaler) == list:
                column_true = exp.y_scaler[num].inverse_transform(
                    column_true.reshape(-1, 1)
                )
                column_pred = exp.y_scaler[num].inverse_transform(
                    column_pred.reshape(-1, 1)
                )
            else:
                column_true = exp.y_scaler.inverse_transform(column_true.reshape(-1, 1))
                column_pred = exp.y_scaler.inverse_transform(column_pred.reshape(-1, 1))
        res = pd.DataFrame(
            np.hstack([column_true.reshape(-1, 1), column_pred.reshape(-1, 1)]),
            columns=["Truth", "Prediction"],
        )
        preds_avg_col = res.groupby("Truth", sort=False).mean().to_numpy().flatten()
        preds_std_col = res.groupby("Truth", sort=False).std().to_numpy().flatten()
        truth_rescaled = res.groupby("Truth", sort=False).mean().index.to_numpy()
        if num == 0:
            preds_grouped = preds_avg_col
            preds_grouped_std = preds_std_col
            truth_grouped = truth_rescaled
        else:
            preds_grouped = np.vstack([preds_grouped, preds_avg_col])
            preds_grouped_std = np.vstack([preds_grouped_std, preds_std_col])
            truth_grouped = np.vstack([truth_grouped, truth_rescaled])
    preds_grouped = preds_grouped.T
    preds_grouped_std = preds_grouped_std.T
    truth_grouped = truth_grouped.T

    if merge_params:
        if preds_grouped.ndim != 1:
            preds_grouped = preds_grouped.sum(axis=1).reshape(-1, 1)
            preds_grouped_std = preds_grouped_std.sum(axis=1).reshape(-1, 1)
            truth_grouped = truth_grouped.sum(axis=1).reshape(-1, 1)
        else:
            preds_grouped = preds_grouped.reshape(-1, 1)
            preds_grouped_std = preds_grouped_std.reshape(-1, 1)
            truth_grouped = truth_grouped.reshape(-1, 1)

    if reg == "time":
        preds_grouped = preds_grouped.reshape(-1, 1)
        preds_grouped_std = preds_grouped_std.reshape(-1, 1)
        truth_grouped = truth_grouped.reshape(-1, 1)

    # Numerical derivative
    mean_val = (preds_grouped[:-1] + preds_grouped[1:]) / 2
    deriv = np.diff(preds_grouped, axis=0) / (mean_val[1] - mean_val[0])

    t_interp = np.linspace(exp.time[0], exp.time[-1], n_interp)
    if reg == "time" or merge_params:
        # Spline interpolation
        spl = UnivariateSpline(exp.time, preds_grouped, k=spl_k, s=spl_s)
        preds_spl = spl(t_interp)
        der_spl = spl.derivative(n=1)(t_interp)

        t_spl = UnivariateSpline(exp.time, truth_grouped, k=spl_k, s=spl_s)
        truth_spl = t_spl(t_interp)
    else:
        delta_interp = np.linspace(
            truth_grouped[:, 0].min(), truth_grouped[:, 0].max(), n_interp
        )
        omega_interp = np.linspace(
            truth_grouped[:, 1].max(), truth_grouped[:, 1].min(), n_interp
        )
        params_interp = [delta_interp, omega_interp]
        for i in range(preds_grouped.shape[1]):
            spl = UnivariateSpline(exp.time, preds_grouped[:, i], k=spl_k, s=spl_s)
            preds_spl = spl(t_interp)
            der_spl = spl.derivative(n=1)(t_interp)
            der_spl_true = spl.derivative(n=1)(exp.time)
            if i == 0:
                preds_spl_interp = preds_spl
                der_spl_interp = der_spl
                der_spl_true_interp = der_spl_true
            else:
                preds_spl_interp = np.vstack([preds_spl_interp, preds_spl])
                der_spl_interp = np.vstack([der_spl_interp, der_spl])
                der_spl_true_interp = np.vstack([der_spl_true_interp, der_spl_true])
        preds_spl_interp = preds_spl_interp.T
        der_spl_interp = der_spl_interp.T
        der_spl_true_interp = der_spl_true_interp.T
        for i in range(truth_grouped.shape[1]):
            spl = UnivariateSpline(exp.time, truth_grouped[:, i], k=spl_k, s=spl_s)
            truth_spl = spl(t_interp)
            if i == 0:
                truth_spl_interp = truth_spl
            else:
                truth_spl_interp = np.vstack([truth_spl_interp, truth_spl])
        truth_spl_interp = truth_spl_interp.T

    fig = plt.figure(layout="constrained", figsize=(30, 8))

    subfigs = fig.subfigures(1, 2, hspace=0.07)
    if reg == "time" or merge_params:
        axs = subfigs[0].subplots(2, 2)
    else:
        axs = subfigs[0].subplot_mosaic(
            [
                ["delta", "delta_num_der"],
                ["delta_spline", "delta_spline_der"],
                ["omega", "omega_num_der"],
                ["omega_spline", "omega_spline_der"],
            ]
        )

    if reg == "time" or merge_params:
        ax = axs[0, 0]
        ax.plot(
            truth_grouped,
            preds_grouped,
            ".:",
            color="steelblue",
            label="Measurement points",
        )
        ax.plot(
            truth_grouped,
            truth_grouped,
            ".--",
            color="k",
            label="Ground truth",
        )
        ax.set_xlabel("Ground truth value")
        ax.set_ylabel("Predicted value")
        ax.set_title("Exact prediction")

        ax = axs[0, 1]
        ax.plot(mean_val, deriv, ".:", color="steelblue", label="Derivative")
        ax.set_xlabel("Ground truth difference value")
        ax.set_ylabel("Predicted Parameter Derivative")
        ax.set_title("Numerical derivative")

        ax = axs[1, 0]
        ax.plot(truth_spl, preds_spl, color="lightsalmon", label="Interpolation")
        ax.plot(
            truth_grouped,
            preds_grouped,
            ".:",
            color="steelblue",
            label="Measurement points",
        )
        ax.plot(truth_spl, truth_spl, color="k", label="Ground truth")
        ax.set_xlabel("Ground truth value")
        ax.set_ylabel("Predicted value")
        ax.set_title(f"Spline interpolation | k={spl_k}, s={spl_s}")

        ax = axs[1, 1]
        ax.plot(t_interp, der_spl, color="lightsalmon", label="Interpolation")
        ax.plot(
            exp.time,
            spl.derivative(n=1)(exp.time),
            "o",
            color="steelblue",
            label="Measurement points",
        )
        if phys_model == "Ising":
            max_val_ind = np.argmax(np.abs(der_spl))
            ax.vlines(
                t_interp[max_val_ind],
                der_spl.min(),
                der_spl.max(),
                linestyle="dashdot",
                color="teal",
                label="Predicted phase transition point",
            )
            U = 1.95
            pred_tr_val = (1.25 * U + 4.66 * U / 2) / (2 * np.pi)
            exp_trans_ind = np.where(np.abs(truth_spl - pred_tr_val) < 1e-1)
            ax.vlines(
                t_interp[exp_trans_ind],
                der_spl.min(),
                der_spl.max(),
                linestyle="dotted",
                color="k",
                label="Experimentally expected phase transition point",
            )
        elif phys_model == "TFIM":
            max_val_ind = np.argmax(np.abs(der_spl))
            ax.vlines(
                t_interp[max_val_ind],
                der_spl.min(),
                der_spl.max(),
                linestyle="dashdot",
                color="teal",
                label="Predicted phase transition point",
            )
            pred_tr_val = 0.5
            exp_trans_ind = np.where(np.abs(truth_spl - pred_tr_val) < 1e-2)
            ax.vlines(
                t_interp[exp_trans_ind],
                der_spl.min(),
                der_spl.max(),
                linestyle="dotted",
                color="k",
                label="Experimentally expected phase transition point",
            )
        ax.set_xlabel(r"$t_{off}$ (ns)")
        ax.set_ylabel("Predicted Parameter Derivative")
        ax.set_title(f"Spline derivative | k={spl_k}, s={spl_s}")

    else:
        ax = axs["delta"]
        ax.plot(
            truth_grouped[:, 0],
            preds_grouped[:, 0],
            ".:",
            color="steelblue",
            label="Measurement points",
        )
        ax.plot(
            truth_grouped[:, 0],
            truth_grouped[:, 0],
            ".--",
            color="k",
            label="Ground truth",
        )
        ax.set_xlabel("Ground truth value")
        ax.set_ylabel("Predicted value")
        ax.set_title("Exact prediction | Delta")

        ax = axs["delta_num_der"]
        ax.plot(
            mean_val[:, 0],
            deriv[:, 0],
            ".:",
            color="steelblue",
            label="Measurement points",
        )
        ax.set_xlabel("Ground truth difference value")
        ax.set_ylabel("Predicted Parameter Derivative")
        ax.set_title("Numerical derivative | Delta")

        ax = axs["delta_spline"]
        ax.plot(
            truth_spl_interp[:, 0],
            preds_spl_interp[:, 0],
            color="lightsalmon",
            label="Interpolation",
        )
        ax.plot(
            truth_grouped[:, 0],
            preds_grouped[:, 0],
            ".:",
            color="steelblue",
            label="Measurement points",
        )
        ax.plot(
            truth_spl_interp[:, 0],
            truth_spl_interp[:, 0],
            "--",
            color="k",
            label="Ground truth",
        )
        ax.set_xlabel("Ground truth value")
        ax.set_ylabel("Predicted value")
        ax.set_title(f"Spline interpolation | k={spl_k}, s={spl_s} | Delta")

        ax = axs["delta_spline_der"]
        ax.plot(
            truth_spl_interp[:, 0],
            der_spl_interp[:, 0],
            color="lightsalmon",
            label="Interpolation",
        )
        ax.plot(
            truth_grouped[:, 0],
            der_spl_true_interp[:, 0],
            "o",
            color="steelblue",
            label="Measurement points",
        )
        ax.set_xlabel("Ground truth difference value")
        ax.set_ylabel("Predicted Parameter Derivative")
        ax.set_title(f"Spline derivative | k={spl_k}, s={spl_s} | Delta")

        ax = axs["omega"]
        ax.plot(
            truth_grouped[:, 1],
            preds_grouped[:, 1],
            ".:",
            color="steelblue",
            label="Measurement points",
        )
        ax.plot(
            truth_grouped[:, 1],
            truth_grouped[:, 1],
            ".--",
            color="k",
            label="Ground truth",
        )
        ax.set_xlabel("Ground truth value")
        ax.set_ylabel("Predicted value")
        ax.set_title("Exact prediction | Omega")

        ax = axs["omega_num_der"]
        ax.plot(
            mean_val[:, 1],
            deriv[:, 1],
            ".:",
            color="steelblue",
            label="Measurement points",
        )
        ax.set_xlabel("Ground truth difference value")
        ax.set_ylabel("Predicted Parameter Derivative")
        ax.set_title("Numerical derivative | Omega")

        ax = axs["omega_spline"]
        ax.plot(
            truth_spl_interp[:, 1],
            preds_spl_interp[:, 1],
            color="lightsalmon",
            label="Interpolation",
        )
        ax.plot(
            truth_spl_interp[:, 1],
            truth_spl_interp[:, 1],
            "--",
            color="k",
            label="Ground truth",
        )
        ax.plot(
            truth_grouped[:, 1],
            preds_grouped[:, 1],
            ".:",
            color="steelblue",
            label="Measurement points",
        )
        ax.set_xlabel("Ground truth value")
        ax.set_ylabel("Predicted value")
        ax.set_title(f"Spline interpolation | k={spl_k}, s={spl_s} | Omega")

        ax = axs["omega_spline_der"]
        ax.plot(
            truth_spl_interp[:, 1],
            der_spl_interp[:, 1],
            color="lightsalmon",
            label="Interpolation",
        )
        ax.plot(
            truth_grouped[:, 1],
            der_spl_true_interp[:, 1],
            "o",
            color="steelblue",
            label="Measurement points",
        )
        ax.set_xlabel("Ground truth difference value")
        ax.set_ylabel("Predicted Parameter Derivative")
        ax.set_title(f"Spline derivative | k={spl_k}, s={spl_s} | Omega")

    if reg == "time" or merge_params:
        for ax in axs.flatten():
            ax.minorticks_on()
            ax.legend()
    else:
        for ax in axs.values():
            ax.minorticks_on()
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.legend()

    axs = subfigs[1].subplots(2, 1, sharex=True)

    ax = axs[0]

    if pred_labels is None:
        if reg == "params":
            if not merge_params:
                pred_labels = [r"$\delta$", r"$\Omega$"]
            else:
                pred_labels = [r"$\delta + \Omega$"]
        else:
            pred_labels = [r"$t$"]

    if merge_params:
        pred_labels = [" + ".join(pred_labels)]

    if len(pred_labels) == 1:
        colors = [["steelblue", "cadetblue"]]
    elif len(pred_labels) == 2:
        colors = [["steelblue", "cadetblue"], ["crimson", "lightcoral"]]
    for i in range(preds_grouped.shape[1]):
        ax.plot(
            exp.time,
            truth_grouped[:, i],
            ".--",
            label="True " + pred_labels[i],
            color=colors[i][0],
        )
        ax.plot(
            exp.time,
            preds_grouped[:, i],
            ".:",
            label="Predicted " + pred_labels[i],
            color=colors[i][1],
        )
        ax.fill_between(
            exp.time,
            preds_grouped[:, i] - preds_grouped_std[:, i],
            preds_grouped[:, i] + preds_grouped_std[:, i],
            alpha=0.3,
        )
    ax.set_ylabel("Predicted value")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.minorticks_on()
    ax.legend()

    ax = axs[1]

    colormap = plt.cm.get_cmap("tab10")
    if kernel_names is None:
        for i in range(logits_avg.shape[1]):
            ax.plot(exp.time, logits_avg[:, i], ".:", label=f"Kernel {i}")
            ax.fill_between(
                exp.time,
                logits_avg[:, i] - logits_std[:, i],
                logits_avg[:, i] + logits_std[:, i],
                alpha=0.1,
            )
    else:
        for k_num, name in enumerate(kernel_names):
            ax.plot(
                exp.time,
                logits_avg[:, k_num],
                ".:",
                label=f"Kernel {k_num}\nSize {kernel_names[k_num][0]}, dilation {kernel_names[k_num][2]}",
                color=colormap(k_num),
            )
            ax.fill_between(
                exp.time,
                logits_avg[:, k_num] - logits_std[:, k_num],
                logits_avg[:, k_num] + logits_std[:, k_num],
                alpha=0.1,
                color=colormap(k_num),
            )
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Kernel activation $a_k$")
    ax.legend()
    ax.minorticks_on()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    subfigs[1].suptitle(
        "Average activation and prediction as a function of experiment time",
        size=15,
    )

    fig.suptitle(
        f"Prediction based method | {phys_model} | Size: {shape} | Defects: {defects} | "
        + r"$R^2 = $"
        + f"{r2:.4f} | "
        + r"$MSE = $"
        + f"{mse:.4f}",
        size=25,
    )

    if save is not None:
        save = Path(save)
        save.joinpath("PNG").mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save.joinpath(
                f"prediction_based_method_size={shape}_defects={defects}.pdf"
            ),
            bbox_inches="tight",
        )
        fig.savefig(
            save.joinpath("PNG").joinpath(
                f"prediction_based_method_size={shape}_defects={defects}.png"
            ),
            bbox_inches="tight",
        )


def apply_model_to_test_data(
    loader,
    models,
    device,
    activation_criterion,
    return_auxiliary=False,
    save_stats=None,
    reg=None,
    out_dims=1,
    subset_classess=None,
    exp=None,
    averaging_tab=None,
    use_batching=False,
):
    criterion = mean_squared_error

    if save_stats is not None:
        load_path = Path(save_stats)

    net, classifier = models
    with torch.no_grad():
        test_tensor = loader.dataset.X.clone().float().to(device)
        if use_batching:
            logits_list = []
            for X, _ in loader:
                X = X.float().to(device)
                logits = net(X)
                # This is to avoid nan values in the output in case of testing dataset with defects
                logits = torch.nan_to_num(logits)
                logits_list.append(logits)
            logits_list = torch.cat(logits_list)
        else:
            logits_list = net(test_tensor)
            # This is to avoid nan values in the output in case of testing dataset with defects
            logits_list = torch.nan_to_num(logits_list)

        if reg is None:
            if out_dims == 1:
                logits = classifier(logits_list)[
                    ..., 0
                ]  # <---- this worked without onehot encoding
                probs = activation_criterion(logits)
                preds = (probs > 0.5).detach().cpu().numpy().astype(np.float32)
            else:
                logits = classifier(logits_list)
                probs = activation_criterion(logits, dim=1)
                preds = probs.argmax(1).type(torch.float).detach().cpu().numpy()
            probs = probs.detach().cpu().numpy()
            truth = loader.dataset.Y.numpy()
            r2, test_MSE = calculate_r2_and_mse(
                truth, probs, time_tab=averaging_tab, exp=exp, old_way=False
            )
            stats = {
                "classification_report": classification_report(truth, preds),
                "confusion_matrix": confusion_matrix(truth, preds),
            }
            aux_dict = {
                "classification_report": classification_report(truth, preds),
                "confusion_matrix": confusion_matrix(truth, preds),
                "truth": truth,
                "probs": probs,
                "preds": preds,
                "logits_list": logits_list,
                "MSE": test_MSE,
                "R^2": r2,
            }
            print(classification_report(truth, preds))
            print(confusion_matrix(truth, preds))
            if save_stats is not None:
                if subset_classess is None:
                    param_file = load_path.joinpath("test_accs.txt")
                else:
                    param_file = load_path.joinpath(
                        f"test_accs_cl={subset_classess}.txt"
                    )
                with open(param_file, "w", encoding="utf-8") as f:
                    with contextlib.redirect_stdout(f):
                        print("Accuracies and confusion matrix on test set\n\n")
                        print(classification_report(truth, preds))
                        print(confusion_matrix(truth, preds))
        else:
            if use_batching:
                answer_list = []
                truth_list = []
                for X, y in loader:
                    X = X.float().to(device)
                    y = y.float().to(device)
                    logits = classifier(net(X))
                    answer_list.append(logits)
                    truth_list.append(y)
                answer = torch.cat(answer_list)
                truth = torch.cat(truth_list)
            else:
                answer = classifier(logits_list)
                truth = loader.dataset.Y.float().to(device)

            if reg == "time":
                answer = answer.squeeze(-1)
                truth = truth.squeeze(-1)
            answer = np.nan_to_num(answer.detach().cpu().numpy())
            truth = truth.detach().cpu().numpy()
            r2, test_MSE = calculate_r2_and_mse(
                truth, answer, time_tab=averaging_tab, exp=exp, old_way=False
            )
            probs = answer
            stats = {
                "MSE": test_MSE,
                "R^2": r2,
            }
            aux_dict = {
                "MSE": test_MSE,
                "R^2": r2,
                "truth": truth,
                "probs": probs,
                "logits_list": logits_list,
            }
            if save_stats is not None:
                param_file = load_path.joinpath("test_stats.txt")
                with open(param_file, "w", encoding="utf-8") as f:
                    with contextlib.redirect_stdout(f):
                        print("Statistics on test set\n\n")
                        print(f"Test MSE : {test_MSE:.4f} | Test R^2 : {r2:.3f}")
            print(f"Test MSE : {test_MSE:.4f} | Test R^2 : {r2:.3f}")
    if return_auxiliary:
        return aux_dict
    return stats


class CheckpointBest(object):
    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.checkpointed_epoch = 0
        self.checkpointed_model_parts = None

    def __call__(
        self,
        current_valid_loss,
        current_epoch,
        model,
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.checkpointed_model_parts = model
            self.checkpointed_epoch = current_epoch
            return True
        return False

    def get_checkpointed_model(self):
        return self.checkpointed_model_parts

    def get_best_valid_loss(self):
        return self.best_valid_loss

    def get_checkpointed_epoch(self):
        return self.checkpointed_epoch


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, warmup=100):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.warmup = warmup

    def early_stop(self, validation_loss, epoch):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
        else:
            self.counter += 1

        if self.counter >= self.patience and epoch >= self.warmup:
            return True
        return False


def count_parameters(model, device):
    th = Theme(
        default_color="36",
        vertical_color="32",
        horizontal_color="32",
        junction_color="36",
    )
    th_2 = Theme(
        default_color="92",
        vertical_color="32",
        horizontal_color="32",
        junction_color="36",
    )
    table = ColorTable(["Modules", "Parameters"], theme=th)
    table_2 = ColorTable(["Additional stats", ""], theme=th_2)
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, f"{params:,}"])
        total_params += params
    print(table)

    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    table_2.add_row(["Total parameters", f"{total_params:,}"])
    table_2.add_row(["Trainable parameters", f"{total_trainable_params:,}"])
    table_2.add_row(["Computation device", device])
    print(table_2)
    return total_params


# Helper function to split an array into subarrays given a list of desired shapes
def split_array(array, shapes):
    split_indices = np.cumsum(shapes)[:-1]
    subarrays = np.split(array, split_indices)
    for num, arr in enumerate(subarrays):
        assert arr.shape[0] == shapes[num]
    return subarrays


def determine_mask(shifts_dict, dim):
    mask = None

    if dim == 1:
        shifts_tab = np.array(shifts_dict["i"])
        max_shift = shifts_tab.max()
        if max_shift > 0:
            mask = (
                slice(None),
                slice(None),
                slice(None, -max_shift),
                slice(None),
            )
    elif dim == 2:
        x_shifts_tab = np.array(shifts_dict["i"])
        y_shifts_tab = np.array(shifts_dict["j"])
        max_x_shift = x_shifts_tab.max()
        max_y_shift = y_shifts_tab.max()
        if max_x_shift > 0 and max_y_shift > 0:
            mask = (
                slice(None),
                slice(None),
                slice(None, -max_x_shift),
                slice(None, -max_y_shift),
            )
        elif max_x_shift > 0 and max_y_shift == 0:
            mask = (
                slice(None),
                slice(None),
                slice(None, -max_x_shift),
                slice(None),
            )
        elif max_x_shift == 0 and max_y_shift > 0:
            mask = (
                slice(None),
                slice(None),
                slice(None),
                slice(None, -max_y_shift),
            )

    return mask


def get_corr(
    tensor,
    mult,
    nreals,
    corr_inds,
    ch_num=0,
    std_dev=False,
    rmse=False,
    raw=False,
    dim=1,
):
    """
    A universal function for generating a list of correlations from snapshots dataset. It allows for single- or
    multichannel correlations in free combinations.

    :param tensor: Input tensor, expected to be of shape [B, C, N_1, N_2], where B - batch size, C - number of channels,
    N_1, N_2 are spatial dimensions of snapshot. B = N_pts * nreals is a flattened dimension of all realizations from
    N_pts measurement points in the phase diagram.

    :param mult: Tunable eigenvalue of spin operator. For uniform normalization of all correlations it should
    be set to 1, for recreation of physically meaningful results a choice of 1/2 is more suited.

    :param nreals: Number of realizations per point in phase diagram.

    :param ch_num: Selection of output channel for the correlator. It is directly connected to the input channels,
    which are always sorted lexicographically. The canonical order is `x y z`, and default value for it is channel 0.

    :param corr_inds: Parameter controlling the calculated correlation , closely intertwined with ch_num.
    When corr_inds == 0:
        An expected value of single-site spin operator is calculated, for channel `ch_num`.
    When corr_inds is Int:
        A two-body correlator is calculated for channel `ch_num` and spins $S^d_i S^d_{i+corr_inds}$
    [Deprecated] When corr_inds is a list, it can be a correlator of arbitrarily many spins in various channels. In particular:
        1. When it contains just integers, eg. [1, 2, 4], then it is a correlator of `i` with `i+1`, 'i+2`, `i+4`.

        2. When the list elements are tuples, they allow for inter-channel correlators. The tuple's structure should be:
        `(correlation length, channel shift)`. The first parameter is the distance between `i`-th spin and the one
        whose correlator is being calculated, spin `i+correlation length`. `channel shift` is responsible for making
        inter-channel correlators. If set to 0, the added correlator will be in the same channels as input.
        When set to 1, it rolls the channels by one index with PBC, i.e. `x y z` -> `y z x`. When set to 2, it rolls the
         channels by two indices, etc.
         As an example, when given an input of channels `x y z` a spin operator is added with tuple (2, 1),
         it produces correlator $S^x_i S^y_{i+2}$ in channel 0, $S^y_i S^z_{i+2}$ in channel 1, and $S^z_i S^x_{i+2}$
         in channel 2.
    When corr_inds is a dictionary, it is a dictionary of lists, where the keys are the physical indices of the correlator
    and the values are the correlator lengths and channel shifts. This is useful for more complex correlators, where
    the same correlator is calculated for different distances and channel shifts. The dictionary should be structured as
    follows:
    {
        'i': [correlation lengths], # for 1D snapshots
        'j': [correlation lengths], # for 2D snapshots
        'channel': [channel shifts] # for multichannel correlators (optional)
    }
    The `channel` key is optional, and if not present, the correlator is calculated in the same channel as the input.
    If the `channel` key is present, then the correlator is calculated in the channel shifted by the amount specified
    in the list. The `channel` key is only used for multichannel correlators, and is ignored for single-channel
    correlators.

    :param std_dev: If this is set to `True`, then the function additionally outputs standard deviation taken along the
    axis of snapshot realizations.

    :param rmse: If set to `True`, the function returns root-mean-square magnetization instead of standard mean
    magnetization. Default value is `False`.

    :param raw: If set to `True`, the function returns raw data, without any averaging over realizations.

    :param dim: Number of physical dimensions of the snapshot. Default value is 1, which corresponds to 1D snapshots.


    :return:
    A np.array of correlations computed according to `corr_inds`, and an array of standard deviations if
    `std_dev == True`
    """
    tt = tensor.detach().clone().cpu()
    tt *= mult

    # Null shift flag
    if type(corr_inds) is dict:
        for key in corr_inds.keys():
            if max(corr_inds[key]) == 0:
                null_shift = True
            else:
                null_shift = False
                break

    # Extraction of single correlator
    if type(corr_inds) is int or null_shift:
        # Replaced standard mean magnetization with root-mean-square magnetization \sqrt{<m_z^2>}
        if corr_inds == 0 or null_shift:
            if rmse:
                corrs = (
                    tt[:, ch_num, :, :].mean(axis=[1, 2])
                    * tt[:, ch_num, :, :].mean(axis=[1, 2])
                ).numpy()
                if std_dev:
                    std = (
                        tt[:, ch_num, :, :].mean(axis=[1, 2])
                        * tt[:, ch_num, :, :].mean(axis=[1, 2])
                    ).numpy()
            else:
                corrs = tt[:, ch_num, :, :].mean(axis=[1, 2]).numpy()
                if std_dev:
                    std = tt[:, ch_num, :, :].mean(axis=[1, 2]).numpy()
        else:
            corrs = (
                (
                    tt[:, ch_num, :-corr_inds, :]
                    * tt.roll(-corr_inds, 2)[:, ch_num, :-corr_inds, :]
                ).mean(axis=1)
            ).numpy()
            if std_dev:
                std = (
                    (
                        tt[:, ch_num, :-corr_inds, :]
                        * tt.roll(-corr_inds, 2)[:, ch_num, :-corr_inds, :]
                    ).mean(axis=1)
                ).numpy()
    # Extraction of multiple single-channel correlators. Deprecated, but left for compatibility
    elif type(corr_inds) is list:
        if type(corr_inds[0]) is int:
            corr_inds = np.array(corr_inds)
            max_corrlen = corr_inds.max()
            corrs = tt[:, :, :-max_corrlen, :]
            for shift in corr_inds:
                corrs *= tt.roll(-shift, 2)[:, :, :-max_corrlen, :]
            data = corrs[:, ch_num, :, :].mean(axis=1)
            if std_dev:
                std = data.numpy()
            corrs = data.numpy()
        else:
            corr_inds = np.array(corr_inds)
            max_corrlen = corr_inds.max()
            corrs = tt[:, :, :-max_corrlen, :]
            for pos_shift, ch_shift in corr_inds:
                corrs *= tt.roll(shifts=(-pos_shift, -ch_shift), dims=(2, 1))[
                    :, :, :-max_corrlen, :
                ]
            data = corrs[:, ch_num, :, :].mean(axis=1)
            if std_dev:
                std = data.numpy()
            corrs = data.numpy()

    # New implementation, where the correlator is extracted from a dictionary.
    elif type(corr_inds) is dict:
        is_multichannel = "channel" in corr_inds.keys()
        if is_multichannel:
            is_multichannel = max(corr_inds["channel"]) > 0
        mask = determine_mask(corr_inds, dim)
        if mask is not None:
            corrs = tt[mask]
        else:
            corrs = tt

        # This might fail if there were different number of realizations for each channel, but for now we do not have such data
        if not is_multichannel:
            if dim == 1:
                shifts_tab = np.array(corr_inds["j"])
                for shift in shifts_tab:
                    if mask is not None:
                        corrs *= tt.roll(-shift, 2)[mask]
                    else:
                        corrs *= tt.roll(-shift, 2)
                data = corrs[:, ch_num, :, :].mean(axis=1)
                if std_dev:
                    std = data.numpy()
                corrs = data.numpy()

            elif dim == 2:
                x_shifts_tab = np.array(corr_inds["i"])
                y_shifts_tab = np.array(corr_inds["j"])

                for x_shift, y_shift in zip(x_shifts_tab, y_shifts_tab):
                    if mask is not None:
                        corrs *= tt.roll(shifts=(-x_shift, -y_shift), dims=(2, 3))[mask]
                    else:
                        corrs *= tt.roll(shifts=(-x_shift, -y_shift), dims=(2, 3))
                data = corrs[:, ch_num, :, :].mean(axis=[1, 2])
                if std_dev:
                    std = data.numpy()
                corrs = data.numpy()
            else:
                raise ValueError("Incorrect dimensionality of the snapshot")
        else:
            if dim == 1:
                shifts_tab = np.array(corr_inds["i"])
                channel_shifts_tab = np.array(corr_inds["channel"])

                for shift, ch_shift in zip(shifts_tab, channel_shifts_tab):
                    if mask is not None:
                        corrs *= tt.roll(shifts=(-shift, -ch_shift), dims=(2, 1))[mask]
                    else:
                        corrs *= tt.roll(shifts=(-shift, -ch_shift), dims=(2, 1))
                data = corrs[:, ch_num, :, :].mean(axis=1)
                if std_dev:
                    std = data.numpy()
                corrs = data.numpy()
            elif dim == 2:
                x_shifts_tab = np.array(corr_inds["i"])
                y_shifts_tab = np.array(corr_inds["j"])
                channel_shifts_tab = np.array(corr_inds["channel"])

                for x_shift, y_shift, ch_shift in zip(
                    x_shifts_tab, y_shifts_tab, channel_shifts_tab
                ):
                    if mask is not None:
                        corrs *= tt.roll(
                            shifts=(-x_shift, -y_shift, -ch_shift),
                            dims=(2, 3, 1),
                        )[mask]
                    else:
                        corrs *= tt.roll(
                            shifts=(-x_shift, -y_shift, -ch_shift),
                            dims=(2, 3, 1),
                        )
                data = corrs[:, ch_num, :, :].mean(axis=[1, 2])

                if std_dev:
                    std = data.numpy()
                corrs = data.numpy()
            else:
                raise ValueError("Incorrect dimensionality of the snapshot")
    else:
        raise ValueError("Incorrect type of corr_inds")

    # Averaging over realizations in every data point
    if not raw:
        if type(nreals) is int:
            corrs = corrs.reshape([-1, nreals]).mean(axis=1)
            if std_dev:
                std = std.reshape([-1, nreals]).std(axis=1)
        else:
            corrs_reshaped = split_array(corrs, nreals)
            mean_corrs = np.zeros(len(corrs_reshaped))
            std_corrs = np.zeros(len(corrs_reshaped))
            for num, arr in enumerate(corrs_reshaped):
                mean_corrs[num] = arr.mean()
                std_corrs[num] = arr.std()
            corrs = mean_corrs
            std = std_corrs

    if std_dev:
        return corrs, std
    else:
        return corrs


def determine_corr_label(
    i_tab,
    j_tab=None,
    dil_tab=None,
    basis_tab=None,  # This corresponds to the channel number
    power_tab=None,
    dollars=True,
):
    if j_tab is None:
        j_tab = [0] * len(i_tab)
    if len(i_tab) != len(j_tab):
        raise ValueError("i_tab and j_tab must have the same length")

    if dil_tab is None:
        dil_tab = [None] * len(i_tab)
    else:
        if len(i_tab) != len(dil_tab):
            raise ValueError("i_tab and dil_tab must have the same length")

    if basis_tab is None:
        basis_tab = ["z"] * len(i_tab)
    else:
        if len(i_tab) != len(basis_tab):
            raise ValueError("i_tab and basis_tab must have the same length")

    if power_tab is None:
        power_tab = [1] * len(i_tab)
    else:
        if len(i_tab) != len(power_tab):
            raise ValueError("i_tab and power_tab must have the same length")

    lab = r"\overline{\langle "
    for i, j, dil, basis, exponent in zip(i_tab, j_tab, dil_tab, basis_tab, power_tab):
        if dil is not None:
            dil_x, dil_y = dil
        if exponent > 0:
            lab_i = "i" if i == 0 else f"i{i:+}"
            lab_j = "j" if j == 0 else f"j{j:+}"
            lab_ij = r"({}, {})".format(lab_i, lab_j)
            lab_S = r"S_{{{}}}^{{{}}}".format(lab_ij, basis)
            if exponent > 1:
                lab += r"({})^{{{}}}".format(lab_S, exponent)
            else:
                lab += lab_S
    lab += r"\rangle}"
    if lab == r"\overline{\langle \rangle}":
        lab = "C"
    elif dollars:
        lab = r"$" + lab + r"$"

    return lab


def round_expr(expr, num_digits):
    rounded = {}
    for atom in expr.atoms(sp.Number):
        if abs(atom) > 1 / 10**num_digits:
            rounded[atom] = round(atom, num_digits)
        else:
            rounded[atom] = float(f"{atom:+.{num_digits}e}")
    return expr.xreplace(rounded)
