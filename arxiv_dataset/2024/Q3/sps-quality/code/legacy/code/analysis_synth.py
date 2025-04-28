# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:11:09 2024

@author: David J. Kedziora
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import root_mean_squared_error

dir_data = "../data/ml"
dir_output = "../results/ml"
dir_results = dir_output

models = {"alpha": None,
          "ols": "OLS",
          "plsr": "PLSR",
          "sgd": "SGD",
          "rf": "RF",
          "xgb": "XGB",
          "omega": None}

substring_file = "test_sps_quality"

list_num_events = [None,
                   "1000",
                   "10000",
                   "100000",
                   "1000000",
                   None]

experimental_contexts = {"alpha": None,
                         "1p2uW_3000cps": "1p2uW",
                         "2p5uW_4000cps": "2p5uW",
                         "4uW_4100cps": "4uW",
                         "8uW_5100cps": "8uW",
                         "10uW_6000cps": "10uW--",
                         "10uW_12000cps": "10uW+",
                         "20uW_7000cps": "20uW",
                         "30uW_7000cps": "30uW",
                         "omega": None}

string_context = "Context"
string_model = "Model"
string_loss_early = "Loss (Early)"
string_loss_mid = "Loss (Mid)"
string_loss_late = "Loss (Late)"
string_loss_full = "Loss (Full)"
string_loss_valid = "Loss (Valid.)"
string_scaler = "Scaler"
string_learning_rate = "Learning Rate"
string_alpha = "Regularisation Term"
string_estimators = "Estimators"
string_max_depth = "Max Depth"
string_min_leaf = "Min Samples Per Leaf"
string_min_split = "Min Samples To Split"

def string_colour(val):
    val_adjust = val*2
    color = plt.cm.YlOrRd(val_adjust)
    latex_color = "{:.2f},{:.2f},{:.2f}".format(*color[:3])
    return f"\\cellcolor[rgb]{{{latex_color}}} {val:.4f}"

df_paper = pd.DataFrame(columns=[string_context, string_model, 
                                 string_loss_early, string_loss_mid, string_loss_late,
                                 string_loss_full, string_loss_valid])
df_hpo = pd.DataFrame(columns=[string_context, string_model, 
                               string_scaler, string_learning_rate, string_alpha,
                               string_estimators, string_max_depth, string_min_leaf, string_min_split])

for id_model, model in models.items():
    if not model is None:

        fig, axes = plt.subplots(8, 4, figsize=(10, 18))
        max_kde_value = 0
        idx_context = 0
        for experimental_context, short_context in experimental_contexts.items():
            if not short_context is None:

                idx_events = 0
                for num_events in list_num_events:
                    if not num_events is None:

                        # Get the fit.
                        df_data = pd.read_csv("%s/%s_%s_events_%s.csv" 
                                            % (dir_data, substring_file, num_events, experimental_context))
                        g_best = df_data["best"]
                        g_fit = df_data["estimate"]

                        # Calculate loss value of the fit.
                        loss_fit = root_mean_squared_error(g_best, g_fit)

                        print(experimental_context)
                        print(num_events)
                        print(f"{'Fit':<8} - Loss: {format(loss_fit, '.2e')}")
                        
                        # Get validation loss and hyperparameters.
                        # Newer pipelines in the info file will replace older ones, e.g imports.
                        # So just take the values for the latest.
                        hp_scaler = None
                        hp_learning_rate = None
                        hp_alpha = None
                        hp_estimators = None
                        hp_max_depth = None
                        hp_min_leaf = None
                        hp_min_split = None
                        with open("%s/results/info_pipelines.txt" % (dir_results + "/synth_" + id_model), "r") as file_info:
                            lines = file_info.readlines()
                            for line in reversed(lines):
                                if "Pipe_HPO_%s:" % (idx_events*8 + idx_context) in line:
                                    loss_validation = float(line.split("Initial Loss: ")[-1])
                                    hp_scaler = "SKLearn_Online_Scaler_Standard" in line
                                    hp_scaler = "Y" if hp_scaler else "N"
                                    for substring in ["eta_zero", "learning_rate"]:
                                        if substring in line:
                                            hp_learning_rate = float(line.split(substring + ": ")[-1].split(",")[0])
                                            hp_learning_rate = f"{hp_learning_rate: .2e}"
                                    if "alpha" in line:
                                        hp_alpha = float(line.split("alpha: ")[-1].split(")")[0])
                                        hp_alpha = f"{hp_alpha: .2e}"
                                    if "n_estimators" in line:
                                        hp_estimators = int(line.split("n_estimators: ")[-1].split(",")[0])
                                    if "max_depth: " in line:
                                        hp_max_depth = int(line.split("max_depth: ")[-1].split(",")[0])
                                    if "min_samples_leaf: " in line:
                                        hp_min_leaf = int(line.split("min_samples_leaf: ")[-1].split(",")[0])
                                    if "min_samples_split: " in line:
                                        hp_min_split = int(line.split("min_samples_split: ")[-1].split(")")[0])
                                    break

                        df_results = pd.read_csv("%s/results/responses_(size_category==%s)&(context==%s).csv"
                                                % (dir_results + "/synth_" + id_model, num_events, experimental_context))

                        g_tl = df_results["L0:(size_category==" + num_events + 
                                        ")&(context!=" + experimental_context + ")_best"]


                        # Grab loss data from result files.
                        loss_range_other = [np.inf, 0]
                        for temp_context in experimental_contexts:
                            if not temp_context is None:
                                if temp_context == experimental_context:
                                    loss = df_results["L0:(size_category==" + num_events + 
                                        ")&(context!=" + temp_context + "):loss"].iloc[-1]
                                # else:
                                #     loss_other = df_results["L0:(size_category==" + num_events + 
                                #         ")&(context!=" + temp_context + "):loss"].iloc[-1]
                                #     if loss_other < loss_range_other[0]:
                                #         loss_range_other[0] = loss_other
                                #     if loss_other > loss_range_other[1]:
                                #         loss_range_other[1] = loss_other
                                        # g_worst = df_results["L0:(context!=" + temp_context + ")_g_best"]

                        part_one = f"{f'{id_model.upper()}':<8} - Loss (TL): {format(loss, '.2e')}"
                        print(f"{f'{part_one}':<32} - Loss (Valid.): {format(loss_validation, '.2e')}")

                        # print(experimental_context)
                        # print("Loss Transfer Learning: %.2e" % loss)
                        # print("Loss Fitting: %.2e" % loss_fit)
                        # print("Loss Range (Inclusive Training): %.2e - %.2e" % (loss_range_other[0], loss_range_other[1]))

        #                 # Plot.
        #                 x = range(10, (1 + len(g_best))*10, 10)

                        ax = axes[idx_context, idx_events]

                        cmap = sns.light_palette('brown', as_cmap=True)
                        sns.kdeplot(x=g_fit, y=g_tl, fill=True, cmap=cmap, ax=ax)

                        ax.plot([0, 1], [0, 1], linestyle="--", color="black")
                        ax.plot([g_best.iloc[-1]], [g_best.iloc[-1]], marker="o", color="black")
                        ax.plot([g_fit.mean()], [g_tl.mean()], marker="x", color="black")

                        # if idx_context == 0 and idx_events == 0:
                        #     # sns.kdeplot(g_fit, label="Fit", ax=ax, color="orange", common_norm=True)
                        #     # sns.kdeplot(g_tl, label="TL", ax=ax, color="green", common_norm=True)

                        #     # max_kde_value = max(ax.get_ylim()[1], max_kde_value)

                        #     sns.kdeplot(x=g_fit, y=g_tl, fill=True, cmap='Reds', ax=ax)

                        #     ax.plot([g_best.iloc[-1]], [g_best.iloc[-1]], marker="x", color="blue")
                        # else:
                        #     # sns.kdeplot(g_fit, ax=ax, color="orange")
                        #     # sns.kdeplot(g_tl, ax=ax, color="green")

                        #     # max_kde_value = max(ax.get_ylim()[1], max_kde_value)

                        #     sns.kdeplot(x=g_fit, y=g_tl, fill=True, cmap='Reds', ax=ax)

                        #     ax.plot([g_best.iloc[-1]], [g_best.iloc[-1]], marker="x", color="blue")

                        ax.set_aspect('equal')

                        # ax.set_xlim(0, 1)
                        # ax.set_ylim(0, 1)

                        ax.set_xlim(max(0, min(min(g_fit),min(g_tl))), min(1, max(max(g_fit),max(g_tl))))
                        ax.set_ylim(max(0, min(min(g_fit),min(g_tl))), min(1, max(max(g_fit),max(g_tl))))

                        ax.set_xlabel(None)
                        ax.set_ylabel(None)
                        if idx_context == 0:
                            ax.set_title("Events: ~%s" % num_events)
                        if idx_events == 3:
                            ax.set_ylabel(experimental_context, rotation=270, labelpad=20)
                            ax.yaxis.set_label_position("right")

        #                 events_per_sec = df_data["events"].iloc[-1]/((len(g_best))*10)

        #                 print(events_per_sec)
        #                 print(1000/events_per_sec)
        #                 print(10000/events_per_sec)

        #                 ax.plot([1000/events_per_sec, 1000/events_per_sec], [0, 1], color="black", linestyle=":")
        #                 ax.plot([10000/events_per_sec, 10000/events_per_sec], [0, 1], color="black", linestyle=":")

        #                 if idx == 0:
        #                     ax.plot(x, g_best, label = "Best", linestyle="--")
        #                     ax.plot(x, g_fit, label = "Fit")
        #                     ax.plot(x, g_tl, label = "TL")
        #                     # ax.plot(x, g_worst, label = "Cheat ML")
        #                 else:
        #                     ax.plot(x, g_best, linestyle="--")
        #                     ax.plot(x, g_fit)
        #                     ax.plot(x, g_tl)
        #                     # ax.plot(x, g_worst)
        #                 ax.set_title(experimental_context)
        #                 ax.grid(alpha=0.2)

        #                 # # Add labels and a legend.
        #                 # plt.xlabel("Measurement (s)")
        #                 # plt.ylabel("$g^{(2)}(0)$")
        #                 # plt.title(experimental_context)
        #                 # plt.legend()

        #                 # # Show the plot.
        #                 # plt.show()

                        idx_events += 1

                idx_context += 1

        # ax.set_xlim(10, (1 + len_max)*10)
        # ax.set_ylim(0, 1)
        # ax.set_xscale("log")

        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, max_kde_value)
        # ax.set_ylim(0, 1)

        # fig.supxlabel("Measurement (s)")
        # fig.supylabel("$g^{(2)}(0)$")
                
        # fig.supxlabel("$g^{(2)}(0)$")
        # fig.supylabel("Density (500 Samples)")

        fig.supxlabel("$g^{(2)}(0)$ - Fit")
        fig.supylabel("$g^{(2)}(0)$ - Transfer Learning")

        # # fig.text(0.5, 0.04, "Measurement (s)", ha = "center")
        # # fig.text(0.04, 0.5, "$g^{(2)}(0)$", va = "center", rotation = "vertical")

        # fig.legend(bbox_to_anchor=(0.975, 0.96))

        plt.subplots_adjust(hspace=0, wspace=0)
        plt.tight_layout()

        # Show the plot.
        plt.show()

        # fig.savefig(dir_output + "/experiment.png", bbox_inches="tight")