import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt

from ._plots import plot_model_rejection, plot_model_vs_mascon_rejection, plot_model_vs_mascon_contours


def save_results(loss_log, weighted_average_log, validation_results, model, folder):
    """Stores the results of a run

    Args:
        loss_log (list): list of losses recorded
        weighted_average_log (list): list of weighted average losses recorded
        validation_results (pandas.df): results of the validation as dataframe
        model (torch model): Torch model that was trained
        folder (str): results folder of the run
    """
    print(f"Saving run results to {folder} ...", end="")
    np.save(os.path.join(folder, "loss_log.npy"), loss_log)
    np.save(os.path.join(folder, "weighted_average_log.npy"), loss_log)
    torch.save(model.state_dict(), os.path.join(folder, "last_model.mdl"))
    validation_results.to_csv(os.path.join(folder, "validation_results.csv"), index=False)
    print("Done.")


def save_plots(model, encoding, mascon_points, lr_log, loss_log, weighted_average_log, vision_loss_log, n_inferences,
               folder, c, N):
    """Creates plots using the model and stores them

    Args:
        model (torch nn): trained model
        encoding (func): encoding function
        mascon_points (torch tensor): Points of the mascon model
        lr_log (list): list of learning rates
        loss_log (list): list of losses recorded
        weighted_average_log (list): list of weighted average losses recorded
        vision_loss_log (list): list of vision loss values
        n_inferences (list): list of number of model evaluations
        folder (str): results folder of the run
    """
    print("Creating rejection plot...", end="")
    plot_model_rejection(model, encoding, views_2d=True,
                         bw=True, N=N, alpha=0.1, s=50, save_path=os.path.join(folder, "rejection_plot_iter999999.png"), c=c)
    print("Done.")
    print("Creating model_vs_mascon_rejection plot...", end="")
    plot_model_vs_mascon_rejection(
        model, encoding, mascon_points, N=N, save_path=os.path.join(folder, "model_vs_mascon_rejection.png"), c=c)
    print("Done.")

    print("Creating model_vs_mascon_contours plot...", end="")
    plot_model_vs_mascon_contours(
        model, encoding, mascon_points, N=N, save_path=os.path.join(folder, "contour_plot_iter999999.png"), c=c)
    print("Done.")

    print("Creating loss plots...", end="")
    plt.figure()
    abscissa = np.cumsum(n_inferences)
    plt.semilogy(abscissa, loss_log)
    plt.semilogy(abscissa, weighted_average_log)
    plt.semilogy(abscissa, vision_loss_log)
    plt.xlabel("Thousands of model evaluations")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Weighted Average Loss",
                "Vision Loss"])
    plt.savefig(os.path.join(folder, "loss_plot.png"), dpi=150)
    print("Done.")

    print("Creating LR plot...")
    plt.figure()
    abscissa = np.cumsum(n_inferences)
    plt.semilogy(abscissa, lr_log)
    plt.xlabel("Thousands of model evaluations")
    plt.ylabel("LR")
    plt.savefig(os.path.join(folder, "lr_plot.png"), dpi=150)


def save_plots_v2(model, encoding, sample, lr_log, loss_log, weighted_average_log, n_inferences,
                  folder, c, N):
    """Creates plots using the model and stores them

    Args:
        model (torch nn): trained model
        encoding (func): encoding function
        sample (str): the body's name
        lr_log (list): list of learning rates
        loss_log (list): list of losses recorded
        weighted_average_log (list): list of weighted average losses recorded
        n_inferences (list): list of number of model evaluations
        folder (str): results folder of the run
    """
    print("Creating rejection plot...", end="")
    plot_model_rejection(
        model, encoding, views_2d=True,
        bw=True, N=N, alpha=0.1, s=50, save_path=os.path.join(folder, "rejection_plot_iter999999.png"), c=c
    )
    print("Done.")
    print("Creating acceleration plot...", end="")
    # plot_compare_acceleration(
    #     sample=sample,
    #     compare_mode=('model', 'polyhedral'),
    #     model_1=(model, encoding, c),
    #     save_path=folder + "model_vs_polyhedral_acc.png"
    # )
    print("Done.")

    print("Creating loss plots...", end="")
    plt.figure()
    abscissa = np.cumsum(n_inferences)
    plt.semilogy(abscissa, loss_log)
    plt.semilogy(abscissa, weighted_average_log)
    plt.xlabel("Thousands of model evaluations")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Weighted Average Loss",
                "Vision Loss"])
    plt.savefig(os.path.join(folder, "loss_plot.png"), dpi=150)
    print("Done.")

    print("Creating LR plot...")
    plt.figure()
    abscissa = np.cumsum(n_inferences)
    plt.semilogy(abscissa, lr_log)
    plt.xlabel("Thousands of model evaluations")
    plt.ylabel("LR")
    plt.savefig(os.path.join(folder, "lr_plot.png"), dpi=150)
