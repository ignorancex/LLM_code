import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from config import (
    INFERENCE_FULL_PATH,
    INDIE_VAR_NAME_DICT,
    DATASET_STYLE,
    PLOTS_FULL_PATH,
    UNDERSP_METRIC,
    INSTRUCTION_PROMPTS,
    MODELS_PARAMS,
    FOR_PAPER,
    MODELS_TO_PLOT,
    MODEL_TO_PLOT_NAME,
    PROMPT_2_NAME,
    NAME_2_PROMPT,
    convert_to_file_save_name,
)

matplotlib.rc("text", usetex=False)

# # ######################################### PLOTTING PARAMS ############################################
LEGEND_LABELS = ["Female", "Male", "Neutral"]

DECIMAL_PLACES = 1

# Common paper layout
Y_LABEL = "Averaged softmax probability"
LABEL_ROTATION = 90
COLORS = ["m", "c", "orange"]


# # ######################################### PLOTTING FUNCS ############################################


def get_results(filename, indie_var_name):
    results = pd.read_csv(filename, header=0)
    results.set_index(indie_var_name, inplace=True)
    results = results.drop("Unnamed: 0", axis=1)
    return results


def get_results_all_models(
    filename, indie_var_name, baseline_included=False, dir=INFERENCE_FULL_PATH
):
    results = pd.read_csv(f"{dir}/{filename}", header=0)
    results.set_index(indie_var_name, inplace=True)
    # dates should be strings like other indie vars
    results.index = results.index.astype("str")
    results = results.drop("Unnamed: 0", axis=1)
    if baseline_included:
        results = results.drop("baseline", axis=0)
    return results


def plot_spurious_correlations(
    axe, i, xs, ys, n_fit, colors=COLORS, legend_labels=LEGEND_LABELS
):
    # largely pulled from:
    # https://stackoverflow.com/questions/28505008/numpy-polyfit-how-to-get-1-sigma-uncertainty-around-the-estimated-curve
    p, C_p = np.polyfit(xs, ys, n_fit, cov=True)
    t = np.linspace(min(xs) - 1, max(xs) + 1, 10 * len(xs))  # This is x-axis
    TT = np.vstack([t ** (n_fit - i) for i in range(n_fit + 1)]).T

    # matrix multiplication calculates the polynomial values
    yi = np.dot(TT, p)
    C_yi = np.dot(TT, np.dot(C_p, TT.T))
    sig_yi = 2 * np.sqrt(np.diag(C_yi))

    axe.fill_between(t, yi + sig_yi, yi - sig_yi, color=colors[i], alpha=0.25)
    axe.plot(t, yi, "-", color=colors[i])
    axe.plot(ys, "o", color=colors[i], label=legend_labels[i], alpha=0.75)

    axe.set_title(title)
    return axe


def process_and_plot_results_from_csvs(
    filenames,
    _y_label,
    legend_labels,
    title,
    axe,
    indie_var_name,
    max_xlabels=None,
    n_fit=1,
    no_plot=False,
):
    all_stats = {}
    all_ys = {}
    for i, filename in enumerate(filenames):
        gender = filename[:1]
        result = get_results_all_models(filename, indie_var_name)
        ys = result.mean(axis=1)
        all_ys[gender] = ys

        # In absence of better gender-equity spectrum
        xs = list(range(len(ys)))

        # get linear regression stats
        slope, _intercept, r_value, _p_value, _std_err = stats.linregress(xs, ys)
        all_stats[gender] = {
            "slope": slope,
            "r_value": r_value,
        }

        # plot spurious correlations
        axe = plot_spurious_correlations(axe, i, xs, ys, n_fit)
    max_xlabels = 8 if indie_var_name == "place" else max_xlabels
    axe.xaxis.set_major_locator(MaxNLocator(max_xlabels))
    axe.yaxis.set_major_locator(MaxNLocator(integer=True))

    # # Calc deltas (currently limited to f, m)
    delta_ys = all_ys["f"] - all_ys["m"]
    (
        delta_slope,
        delta_intercept,
        delta_r_value,
        delta_p_value,
        delta_std_err,
    ) = stats.linregress(xs, delta_ys)

    xs = np.array(xs)
    n = len(xs)  # number of observations
    xs_bar = np.mean(xs)  # mean of x values

    # Calculation of standard deviation of x values
    s_xx = np.sum((xs - xs_bar) ** 2)
    # Calculation of residuals
    delta_ys_predicted = delta_intercept + delta_slope * xs
    delta_residuals = delta_ys - delta_ys_predicted
    # Calculation of standard deviation of residuals
    delta_s_e = np.sqrt(np.sum(delta_residuals**2) / (n - 2))

    # Calculate the standard error of the slope (se_slope)
    delta_se_slope = delta_s_e / np.sqrt(s_xx)
    linear_coeffs = {
        # As anticipated/noted in paper, increase of gender-neutral pronoun predictions by GPT-4
        # warranted inclusion of gender-neutral pronoun probabilities in method 2
        "delta_slope": (all_stats["f"]["slope"] + all_stats["n"]["slope"])
        - all_stats["m"]["slope"],
        "avg_r2": np.mean(
            [
                all_stats["f"]["r_value"] ** 2 + all_stats["n"]["r_value"] ** 2,
                all_stats["m"]["r_value"] ** 2,
            ]
        ),
        "delta_se_slope": delta_se_slope,
    }

    return linear_coeffs


def plot_linear_coeffs(
    all_linear_coeffs,
    models_params,
    indie_var_name,
    dataset_name,
    prompt_idx,
    no_legend=False,
    dir=PLOTS_FULL_PATH,
    figsize=(5, 3),
):
    num_prompts = len(PROMPT_2_NAME)
    avg_linear_coeffs = {model: {} for model in MODELS_PARAMS}
    for l in all_linear_coeffs:

        for m in l.keys():
            for k, v in l[m].items():
                if k not in avg_linear_coeffs[m]:
                    avg_linear_coeffs[m][k] = v
                else:
                    avg_linear_coeffs[m][k] = avg_linear_coeffs[m][k] + v
                avg_linear_coeffs[m][k] = avg_linear_coeffs[m][k] / num_prompts

    linear_coeffs_df = pd.DataFrame(avg_linear_coeffs).T
    linear_coeffs_df = linear_coeffs_df.rename(
        index=lambda x: x.split("Prompt")[0].strip()
    )

    title = f"Delta Coefficients for {INDIE_VAR_NAME_DICT[indie_var_name]} vs Gender Pronouns"
    y_label = "Delta Slope and $\it{r}$ Coefficients"
    plt.figure(figsize=figsize)
    linear_coeffs_df[["delta_slope", "avg_r2"]].plot(
        kind="bar", color=["gray", "black"]
    )
    plt.title(title)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    if no_legend:
        plt.legend("", frameon=False)
    else:
        plt.legend(["Slope", "r^2"], loc="best")
    file_path = os.path.join(
        dir, f"delta_avg_linear_slope_r_{indie_var_name}_P{prompt_idx}"
    )
    plt.savefig(file_path, dpi=100)
    plt.close()
    print(f"Saved plots to {file_path}")

    # plot vs size
    title = f"Difference in Female vs Male Slope Coefficients for {INDIE_VAR_NAME_DICT[indie_var_name]} vs Gender Pronouns"
    approx_model_logsizes = [
        MODELS_PARAMS[m]["approx_num_weights"] for m in linear_coeffs_df.index
    ]
    avg_r2 = linear_coeffs_df["avg_r2"].values
    delta_slope = linear_coeffs_df["delta_slope"].values
    y_label = "Delta Slope"
    plt.figure(figsize=figsize)
    colors = [
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "k",
        "orange",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "k",
    ][
        : len(approx_model_logsizes)
    ]  # TODO validate
    plt.scatter(
        approx_model_logsizes, delta_slope, s=(avg_r2**2) * 5000, alpha=0.5, c=colors
    )
    plt.xscale("log")
    plt.xlabel("Number of Parameters")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if not FOR_PAPER:
        xoffsets = [0] * len(linear_coeffs_df)  # [0,0,0,0,0,0,0,0,0,0,0,1e11,1e11,1e11]
        yoffsets = [0] * len(linear_coeffs_df)  # [0,0,0,0,0.2,0,0,0,0,0,0,0,0,0]
        for i, model_name in enumerate(linear_coeffs_df.index):
            plt.text(
                approx_model_logsizes[i] - xoffsets[i],
                delta_slope[i] - yoffsets[i],
                f"{i}",  #:{model_name}
                size=7,
                alpha=0.75,
            )

    file_path = os.path.join(
        dir, f"scatter_delta_linear_se_slope_{indie_var_name}_P{prompt_idx}"
    )
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"Saved plots to {file_path}")


def get_files_to_plot(model_name, indie_var_name, prompt_idx=None):
    file_version_of_model_name = convert_to_file_save_name(
        MODELS_PARAMS[model_name]["model_call_name"]
    )

    if MODELS_PARAMS[model_name]["is_instruction"]:
        file_run_version = f"_testFalse_P{prompt_idx}"
        display_prompt_idx = PROMPT_2_NAME[prompt_idx] if FOR_PAPER else prompt_idx
        #TODO add back prompt_idx, now missing from appendix plots
        title = f"{model_name}"  
    else:
        file_run_version = "_testFalse"
        title = model_name
    filenames = [
        f"fp_{DATASET_STYLE}_{file_version_of_model_name}_{indie_var_name}_{file_run_version}.csv",
        f"mp_{DATASET_STYLE}_{file_version_of_model_name}_{indie_var_name}_{file_run_version}.csv",
        f"np_{DATASET_STYLE}_{file_version_of_model_name}_{indie_var_name}_{file_run_version}.csv",
    ]
    return filenames, title


###################################### PLOTS ############################################
################################## main plots #######################################
if __name__ == "__main__":
    # %%
    assert (
        UNDERSP_METRIC == False
    ), "** Set `METHOD_IDX` to `correlation_measurement` config.py for plotting spuriousness. **"
    legend_labels = LEGEND_LABELS
    run_version = "_testFalse"
    y_label = Y_LABEL
    num_plots = len(MODELS_PARAMS)
    height = 4
    width = 32

    for indie_var_name in INDIE_VAR_NAME_DICT.keys():
        all_prompts_linear_coeffs = []
        for prompt_idx, prompts in INSTRUCTION_PROMPTS.items():
            fig, ax = plt.subplots(1, num_plots, sharex=True)
            fig.set_figheight(height)
            fig.set_figwidth(width)
            fig.suptitle(
                f"Unnormalized Softmax Probabilities for Predicted Pronoun vs {INDIE_VAR_NAME_DICT[indie_var_name]}",
                fontsize=16,
            )

            linear_coeffs = {}
            for model_idx, model_name in enumerate(MODELS_PARAMS.keys()):
                to_save_file_run_version = f"_testFalse_P{prompt_idx}"
                try:
                    filenames, title = get_files_to_plot(
                        model_name, indie_var_name, prompt_idx=prompt_idx
                    )
                    axe = ax[model_idx]
                    axe.tick_params(axis="x", labelrotation=LABEL_ROTATION)
                    linear_coeffs[model_name] = process_and_plot_results_from_csvs(
                        filenames, y_label, legend_labels, title, axe, indie_var_name
                    )

                except FileNotFoundError as e:
                    print(e)
                    continue

            #  TODO pull out into function
            handles, labels = axe.get_legend_handles_labels()
            fig.tight_layout()
            file_name = (
                f"fig_{DATASET_STYLE}_{indie_var_name}_{to_save_file_run_version}"
            )
            file_path = os.path.join(PLOTS_FULL_PATH, file_name)
            plt.savefig(file_path, dpi=100)
            plt.close()
            print(f"Saved plots to {file_path}")

            all_prompts_linear_coeffs.append(linear_coeffs)

        plot_linear_coeffs(
            all_prompts_linear_coeffs,
            MODELS_PARAMS,
            indie_var_name,
            DATASET_STYLE,
            prompt_idx,
        )

    if FOR_PAPER:
        num_plots = len(MODELS_PARAMS)
        height = 22
        width = 16
        for indie_var_name in INDIE_VAR_NAME_DICT.keys():
            plotting_params = MODELS_TO_PLOT[MODEL_TO_PLOT_NAME]
            fig, ax = plt.subplots(
                plotting_params["n_rows"], plotting_params["n_cols"], sharex=True
            )

            fig.set_figheight(plotting_params["h"])
            fig.set_figwidth(plotting_params["w"])
            models = plotting_params["models"]
            all_linear_coeffs = {}
            for ax_idx in models:
                model_name = models[ax_idx]["model"]
                if MODELS_PARAMS[model_name]["is_instruction"]:
                    prompt_idx = NAME_2_PROMPT[models[ax_idx]["prompt"]]
                else:
                    prompt_idx = None
                to_save_file_run_version = f"{MODEL_TO_PLOT_NAME}_{indie_var_name}"
                filenames, title = get_files_to_plot(
                    model_name, indie_var_name, prompt_idx=prompt_idx
                )
                axe = ax[ax_idx] if type(ax_idx) == int else ax[ax_idx[0]][ax_idx[1]]
                axe.tick_params(axis="x", labelrotation=LABEL_ROTATION)
                all_linear_coeffs[title] = process_and_plot_results_from_csvs(
                    filenames,
                    y_label,
                    legend_labels,
                    title,
                    axe,
                    indie_var_name,
                    max_xlabels=plotting_params["max_xlabels"],
                )

            handles, labels = axe.get_legend_handles_labels()
            fig.tight_layout()
            file_name = (
                f"final_fig_{DATASET_STYLE}_{indie_var_name}_{to_save_file_run_version}"
            )
            file_path = os.path.join(PLOTS_FULL_PATH, file_name)
            plt.savefig(file_path, dpi=plotting_params["dpi"])
            plt.close()
            print(f"Saved plots to {file_path}")
