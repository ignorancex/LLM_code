# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:11:29 2023

@author: David J. Kedziora
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

list_num_events = [None,
                   "1000",
                   "10000",
                   "100000",
                   "1000000",
                   None]

experimental_contexts = [None,
                         "1p2uW_3000cps",
                         "2p5uW_4000cps",
                         "4uW_4100cps",
                         "8uW_5100cps",
                         "10uW_6000cps",
                         "10uW_12000cps",
                         "20uW_7000cps",
                         "30uW_7000cps",
                         None]

def plot_all(in_df, in_metric, in_axis_label, key_target, model_type, is_fitting = False):
    if is_fitting:
        model_type = "fitting"
        in_df = in_df.groupby(["experimental_context", "num_events_test"]).mean().reset_index()

    # Create a grid of plots
    grid = sns.FacetGrid(in_df, col="experimental_context", hue="num_events",
                         col_wrap=4, col_order=order, height=4)
    grid.map(sns.lineplot, "num_events_test", in_metric, marker="o")

    # Set x-axis to be logarithmic
    grid.set(xscale="log")

    # Customize plot labels
    grid.set_axis_labels("Num Events Test", in_axis_label)
    grid.set_titles("Experimental Context: {col_name}")
    if not is_fitting:
        grid.add_legend(title="Num Events Train", title_fontsize='12')
    plt.suptitle("Model: '%s', Target: '%s'" % (model_type, key_target), fontsize=16)

    grid.map(plt.grid, linestyle=':', alpha=0.5)

    plt.tight_layout()

    plt.subplots_adjust(right=0.9125)  # Adjust the right margin

    save_filename = "%s_for_%s_%s.png" % (in_metric, key_target, model_type)  # Specify the desired file name and format (e.g., "plot.png")
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')

def plot_average(in_df, in_metric, in_axis_label, key_target, model_type, is_fitting = False):
    if is_fitting:
        model_type = "fitting"
        in_df = in_df.groupby(["num_events_test"]).mean().reset_index()

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot all lines for different num_events
    ax = sns.lineplot(data=in_df, x="num_events_test", y=in_metric, hue="num_events", marker="o", palette="deep")

    # Set x-axis to be logarithmic
    plt.xscale("log")

    # Customize plot labels
    plt.xlabel("Num Events Test")
    plt.ylabel(in_axis_label)
    plt.title("Model: '%s', Target: '%s', Average Over Context-Transfers" % (model_type, key_target))
    if is_fitting:
        ax.get_legend().remove()
    else:
        plt.legend(title="Num Events Train", title_fontsize='12')

    ax.grid(linestyle=':', alpha=0.5)

    plt.tight_layout()

    save_filename = "%s_avg_for_%s_%s.png" % (in_metric, key_target, model_type)  # Specify the desired file name and format (e.g., "plot.png")
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')


for model_type in ["linear_regression"]:

    for key_target in ["estimate", "best"]:

        # Load the CSV data into a DataFrame (replace with your actual CSV filename)
        csv_filename = "loss_for_%s_%s.csv" % (key_target, model_type)
        df = pd.read_csv(csv_filename)

        if key_target == "best":
            df["rmse_ml_vs_fitting"] = df["rmse_ml"] - df["rmse_fitting"]

        df_average = df.groupby(["num_events", "num_events_test"]).mean().reset_index()

        # Set the order of experimental contexts based on the specified order
        order = [context for context in experimental_contexts if context is not None]

        plot_all(df, "rmse_ml", "RMSE (ML)", key_target, model_type)

        plot_average(df_average, "rmse_ml", "RMSE (ML)", key_target, model_type)

        if key_target == "best":
            plot_all(df, "rmse_fitting", "RMSE (Fitting)", key_target, model_type, is_fitting = True)

            plot_average(df_average, "rmse_fitting", "RMSE (Fitting)", key_target, model_type, is_fitting = True)

            plot_all(df, "rmse_ml_vs_fitting", "RMSE (ML vs. Fitting)", key_target, model_type)

            plot_average(df_average, "rmse_ml_vs_fitting", "RMSE (ML vs. Fitting)", key_target, model_type)

        plt.show()