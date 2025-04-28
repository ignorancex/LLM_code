# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 23:28:49 2024

@author: David J. Kedziora
"""

import pandas as pd
import matplotlib.pyplot as plt
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

substring_file = "sps_cumsum_norm"

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

fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex = True, sharey = True)
len_max = 0
idx_context = 0
for experimental_context, short_context in experimental_contexts.items():
    if not short_context is None:

        # Get the fit.
        df_data = pd.read_csv("%s/%s_%s.csv" % (dir_data, substring_file, experimental_context))
        g_best = df_data["g_best"]
        g_fit = df_data["g_fit"]
        len_max = max(len_max, len(g_best))

        # Plot fit.
        x = range(10, (1 + len(g_best))*10, 10)

        ax = axes[int(idx_context/2), idx_context%2]

        events_per_sec = df_data["events"].iloc[-1]/((len(g_best))*10)

        sec_1000 = 1000/events_per_sec
        sec_10000 = 10000/events_per_sec

        ax.plot([sec_1000, sec_1000], [0, 1], color="black", linestyle=":")
        ax.plot([sec_10000, sec_10000], [0, 1], color="black", linestyle=":")

        if idx_context == 0:
            ax.plot(x, g_best, label = "Best", linestyle="--")
            ax.plot(x, g_fit, label = "Fit")
        else:
            ax.plot(x, g_best, linestyle="--")
            ax.plot(x, g_fit)

        # Calculate loss values of the fit.
        cutoff_1000 = int(sec_1000/10)
        cutoff_10000 = int(sec_10000/10)

        loss_full = root_mean_squared_error(g_best, g_fit)
        loss_early = root_mean_squared_error(g_best[0:cutoff_1000], g_fit[0:cutoff_1000])
        loss_mid = root_mean_squared_error(g_best[cutoff_1000:cutoff_10000], g_fit[cutoff_1000:cutoff_10000])
        loss_late = root_mean_squared_error(g_best[cutoff_10000:], g_fit[cutoff_10000:])

        print(experimental_context)
        print(f"{'Fit':<8} - Loss: {format(loss_full, '.2e')}")

        df_paper = pd.concat([df_paper, 
                              pd.DataFrame({string_context: [short_context], string_model: ["Fit"],
                                            string_loss_early: [string_colour(loss_early)],
                                            string_loss_mid: [string_colour(loss_mid)],
                                            string_loss_late: [string_colour(loss_late)],
                                            string_loss_full: [string_colour(loss_full)]})], 
                              ignore_index=True, axis = 0)

        # Get the model results.
        for id_model, model in models.items():
            if not model is None:

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
                with open("%s/results/info_pipelines.txt" % (dir_results + "/basic_" + id_model), "r") as file_info:
                    lines = file_info.readlines()
                    for line in reversed(lines):
                        if "Pipe_HPO_%s:" % idx_context in line:
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

                df_results = pd.read_csv("%s/results/responses_(context==%s).csv" 
                                        % (dir_results + "/basic_" + id_model, experimental_context))
                g_tl = df_results["L0:(context!=" + experimental_context + ")_g_best"]

                # Plot model results.
                if idx_context == 0:
                    ax.plot(x, g_tl, label = model)
                else:
                    ax.plot(x, g_tl)

                # Calculate loss values.
                loss_full = root_mean_squared_error(g_best, g_tl)
                loss_early = root_mean_squared_error(g_best[0:cutoff_1000], g_tl[0:cutoff_1000])
                loss_mid = root_mean_squared_error(g_best[cutoff_1000:cutoff_10000], g_tl[cutoff_1000:cutoff_10000])
                loss_late = root_mean_squared_error(g_best[cutoff_10000:], g_tl[cutoff_10000:])

                part_one = f"{f'{id_model.upper()}':<8} - Loss (TL): {format(loss_full, '.2e')}"
                print(f"{f'{part_one}':<32} - Loss (Valid.): {format(loss_validation, '.2e')}")

                # Store loss values.
                df_paper = pd.concat([df_paper,
                                      pd.DataFrame({string_context: [short_context], string_model: [model],
                                                    string_loss_early: [string_colour(loss_early)],
                                                    string_loss_mid: [string_colour(loss_mid)],
                                                    string_loss_late: [string_colour(loss_late)],
                                                    string_loss_full: [string_colour(loss_full)],
                                                    string_loss_valid: [string_colour(loss_validation)]})], 
                                      ignore_index=True, axis = 0)
                
                # Store hyperparameters.
                df_hpo = pd.concat([df_hpo,
                                    pd.DataFrame({string_context: [short_context], string_model: [model],
                                                  string_scaler: [hp_scaler],
                                                  string_learning_rate: [hp_learning_rate],
                                                  string_alpha: [hp_alpha],
                                                  string_estimators: [hp_estimators],
                                                  string_max_depth: [hp_max_depth],
                                                  string_min_leaf: [hp_min_leaf],
                                                  string_min_split: [hp_min_split]})],
                                    ignore_index=True, axis = 0)

        ax.set_title(experimental_context)
        ax.grid(alpha=0.2)

        print()

        idx_context += 1

ax.set_xlim(10, (1 + len_max)*10)
ax.set_ylim(0, 1)
ax.set_xscale("log")

fig.supxlabel("Measurement (s)")
fig.supylabel("$g^{(2)}(0)$")

fig.legend(bbox_to_anchor=(0.975, 0.725))

plt.subplots_adjust(hspace=0, wspace=0)
plt.tight_layout()

# Show the plot.
plt.show()

fig.savefig(dir_output + "/experiment.png", bbox_inches="tight")

df_paper.set_index([string_context, string_model], inplace=True)
string_table = df_paper.to_latex(na_rep="")
string_table = string_table.replace("lllllll", "llccccc")
string_table = string_table.replace("multirow[t]", "multirow")
string_table = string_table.replace("\n\\multirow", "\\\\[-3.5mm]\n\\multirow")
print(string_table)

models_values = [value for value in models.values() if value is not None]
experimental_contexts_values = [value for value in experimental_contexts.values() if value is not None]

df_hpo[string_model] = pd.Categorical(df_hpo[string_model], categories=models_values, ordered=True)
df_hpo[string_context] = pd.Categorical(df_hpo[string_context], categories=experimental_contexts_values, ordered=True)
df_hpo.sort_values(by=[string_model, string_context], inplace=True)
df_hpo.set_index([string_model, string_context], inplace=True)
string_hpo = df_hpo.to_latex(na_rep="")
string_hpo = string_hpo.replace("lllllllll", "llccccccc")
string_hpo = string_hpo.replace("multirow[t]", "multirow")
print(string_hpo)