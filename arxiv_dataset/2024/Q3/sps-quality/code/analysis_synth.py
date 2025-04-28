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
          "sgd": "SGD",
          "plsr": "PLSR",
          "rf": "RF",
          "gb": "GB",
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

dict_fit = dict()
dict_valid = dict()
dict_loss = dict()

idx_model = 0
for id_model, model in models.items():
    if not model is None:

        dict_fit[id_model] = dict()
        dict_valid[id_model] = dict()
        dict_loss[id_model] = dict()

        fig_loss, axes_loss = plt.subplots(4, 2, figsize=(10, 12), sharex = True)
        # fig_hist, axes_hist = plt.subplots(8, 4, figsize=(10, 18))
        fig_hist, axes_hist = plt.subplots(5, 4, figsize=(10, 12))
        max_kde_value = 0
        idx_context = 0
        for experimental_context, short_context in experimental_contexts.items():
            if not short_context is None:

                dict_fit[id_model][experimental_context] = dict()
                dict_valid[id_model][experimental_context] = dict()
                dict_loss[id_model][experimental_context] = dict()

                idx_events = 0
                for num_events_test in list_num_events:
                    if not num_events_test is None:

                        dict_loss[id_model][experimental_context][num_events_test] = dict()

                        # Get the fit.
                        df_data = pd.read_csv("%s/%s_%s_events_%s.csv" 
                                            % (dir_data, substring_file, num_events_test, experimental_context))
                        g_best = df_data["best"]
                        g_fit = df_data["estimate"]

                        # Calculate loss value of the fit.
                        loss_fit = root_mean_squared_error(g_best, g_fit)
                        dict_fit[id_model][experimental_context][num_events_test] = loss_fit

                        print(experimental_context)
                        print(num_events_test)
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
                        dict_valid[id_model][experimental_context][num_events_test] = loss_validation

                        df_results = pd.read_csv("%s/results/responses_(size_category==%s)&(context==%s).csv"
                                                % (dir_results + "/synth_" + id_model, num_events_test, experimental_context))

                        g_tl = None
                        loss_tl = None
                        for num_events_train in list_num_events:
                            if not num_events_train is None:
                                g_focus = df_results["L0:(size_category==" + num_events_train + 
                                                     ")&(context!=" + experimental_context + ")_best"]
                                loss_focus = root_mean_squared_error(g_best, g_focus)
                                dict_loss[id_model][experimental_context][num_events_test][num_events_train] = loss_focus
                                if num_events_test == num_events_train:
                                    g_tl = g_focus
                                    loss_tl = loss_focus

                        part_one = f"{f'{id_model.upper()}':<8} - Loss (TL): {format(loss_tl, '.2e')}"
                        print(f"{f'{part_one}':<32} - Loss (Valid.): {format(loss_validation, '.2e')}")

                        # Plot histogram.
                        if idx_context not in [1,4,5]:
                            idx_hist = idx_context
                            if idx_context > 1:
                                idx_hist -= 1
                            if idx_context > 5:
                                idx_hist -= 2
                            ax_hist = axes_hist[idx_hist, idx_events]

                            cmap = sns.light_palette("brown", as_cmap=True)
                            sns.kdeplot(x=g_fit, y=g_tl, fill=True, cmap=cmap, ax=ax_hist)

                            ax_hist.plot([0, 1], [0, 1], linestyle="--", color="black")
                            ax_hist.plot([g_best.iloc[-1]], [g_best.iloc[-1]], marker="o", color="black")
                            ax_hist.plot([g_fit.mean()], [g_tl.mean()], marker="x", color="black")

                            ax_hist.set_aspect("equal")

                            ax_hist.set_xlim(max(0, min(min(g_fit),min(g_tl))), min(1, max(max(g_fit),max(g_tl))))
                            ax_hist.set_ylim(max(0, min(min(g_fit),min(g_tl))), min(1, max(max(g_fit),max(g_tl))))

                            ax_hist.set_xlabel(None)
                            ax_hist.set_ylabel(None)
                            if idx_context == 0:
                                ax_hist.set_title("Events: ~%s" % num_events_test)
                            if idx_events == 3:
                                ax_hist.set_ylabel(experimental_context, rotation=270, labelpad=20)
                                ax_hist.yaxis.set_label_position("right")

                            ax_hist.grid(alpha=0.2)
                            # plt.setp(ax_hist.get_xticklabels(), rotation=-45, horizontalalignment="left")

                        idx_events += 1

                # Plot comparative loss.
                ax_loss = axes_loss[int(idx_context/2), idx_context%2]
                ax_loss.plot([1000, 1000000], [0, 0], color="k", linestyle="--")
                for num_events_train in list_num_events[1:-1]:
                    loss_values = [(dict_loss[id_model][experimental_context][num_events_test][num_events_train]
                                    - dict_fit[id_model][experimental_context][num_events_test]) for num_events_test in list_num_events[1:-1]]
                    if idx_context == 0:
                        ax_loss.plot([int(x) for x in list_num_events[1:-1]], loss_values, label = f"{int(num_events_train):.0e}")
                    else:
                        ax_loss.plot([int(x) for x in list_num_events[1:-1]], loss_values)

                ax_loss.set_title(experimental_context)
                ax_loss.grid(alpha=0.2)

                idx_context += 1

        # Finalise the context-specific plots.
        fig_hist.subplots_adjust(hspace=0, wspace=0)
        fig_hist.tight_layout(rect=[0.05, 0.03, 1, 1])

        fig_hist.supxlabel("$g^{(2)}(0)$ - Fit")
        fig_hist.supylabel("$g^{(2)}(0)$ - Transfer Learning")

        fig_hist.savefig(dir_output + "/synth_hist_" + model + ".png", bbox_inches="tight")

        ax_loss.set_xlim(1000, 1000000)
        ax_loss.set_xscale("log")

        fig_loss.subplots_adjust(hspace=0, wspace=0)
        fig_loss.tight_layout(rect=[0.03, 0.03, 1, 1])

        fig_loss.supxlabel("Test Data - Number of Events")
        fig_loss.supylabel("Comparative Loss (Model vs. Fit)")
        if id_model == "ols":
            fig_loss.legend(bbox_to_anchor=(0.95, 0.95), title = "Training - Events")
        else:
            fig_loss.legend(bbox_to_anchor=(0.95, 0.65), title = "Training - Events")

        fig_loss.savefig(dir_output + "/synth_loss_" + model + ".png", bbox_inches="tight")



fig_loss_avg, axes_loss_avg = plt.subplots(3, 2, figsize=(10, 10), sharex = True)
idx_model = 0
for id_model, model in models.items():
    if not model is None:
        # Plot context-averaged loss.
        ax_loss_avg = axes_loss_avg[idx_model%3, int(idx_model/3)]
        ax_loss_avg.plot([1000, 1000000], [0, 0], color="k", linestyle="--")
        for num_events_train in list_num_events[1:-1]:
            loss_values = [np.mean([(dict_loss[id_model][experimental_context][num_events_test][num_events_train]
                                     - dict_fit[id_model][experimental_context][num_events_test])
                                    for experimental_context, short_context in experimental_contexts.items()
                                    if short_context is not None])
                           for num_events_test in list_num_events[1:-1]]
            
            if idx_model == 0:
                ax_loss_avg.plot([int(x) for x in list_num_events[1:-1]], loss_values, label = f"{int(num_events_train):.0e}")
            else:
                ax_loss_avg.plot([int(x) for x in list_num_events[1:-1]], loss_values)

        ax_loss_avg.set_title(model)
        ax_loss_avg.grid(alpha=0.2)
        
        idx_model += 1

ax_loss_avg.set_xlim(1000, 1000000)
ax_loss_avg.set_xscale("log")

fig_loss_avg.subplots_adjust(hspace=0, wspace=0)
fig_loss_avg.tight_layout(rect=[0.03, 0.03, 1, 1])

fig_loss_avg.supxlabel("Test Data - Number of Events")
fig_loss_avg.supylabel("Comparative Loss (Model vs. Fit)")
fig_loss_avg.legend(bbox_to_anchor=(0.85, 0.25), title = "Training - Events")

fig_loss_avg.savefig(dir_output + "/synth_loss_avg.png", bbox_inches="tight")

# Show the plots.
plt.show()