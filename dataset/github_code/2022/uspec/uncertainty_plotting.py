# %%
import matplotlib.patches as mpatches
import csv
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from collections import defaultdict
import seaborn as sns
import warnings
import math

import os
from pathlib import Path
from config import (
    MODELS_PARAMS,
    WINOGENDER_SCHEMA_PATH,
    INFERENCE_FULL_PATH,
    OCC_STATS_FILE,
    PLOTS_FULL_PATH,
    STATS_FULL_PATH,
    NUM_AVE_UNCERTAIN,
    ORIGINAL_DATES,
    UNDERSP_METRIC,
    TESTING,
    GENDERED_IDS,
    WINO_G_ID,
    INTENDED_INST_VERSION,
    convert_to_file_save_name,
)

VERBOSE = True
DECIMAL_PLACES = 3

######################## PLOTTING PARAMS #####################
sns.set_style("white")
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

STATIC_WINO_PARTICIPS = ["man", "woman", "someone"]
ALL_WINO_PARTICIPS = STATIC_WINO_PARTICIPS + ["other"]
WINO_G_ID_ID_PARTS = GENDERED_IDS + ["unspecified"]
ALL_PARTS = ALL_WINO_PARTICIPS if not WINO_G_ID else WINO_G_ID_ID_PARTS

COLORS = ["m", "g", "b", "r", "k"]
if WINO_G_ID:
    PROFESSIONAL_COLORS = ["blue", "darkblue", "lime"]
else:
    PROFESSIONAL_COLORS = ["green", "limegreen", "lime", "darkgreen"]
PARTICIP_COLORS = ["blue", "darkblue", "palegreen", "forestgreen"]

LEGEND_LABELS = ["Female", "Male"]
PRONOUN_DICT = {"f": "Female", "m": "Male", "n": "Neutral"}

file_save_model_name_2_model = {
    convert_to_file_save_name(v["model_call_name"]): k
    for (k, v) in MODELS_PARAMS.items()
}
# %%


def get_uncertainty_metric(softmax_data, num_uncertain=NUM_AVE_UNCERTAIN):
    if softmax_data.empty:
        return math.nan
    return abs(
        softmax_data[-num_uncertain:].values.mean()
        - softmax_data[:num_uncertain].values.mean()
    )


def get_df_per_model_from_all_files_in_dir(
    inference_path, intended_instruction_version=None, indie_vars=None
):
    files = Path(inference_path).glob("*.csv")

    # Will init model_results as below for plotting:
    # {'bert-base-uncased': {'man': [], 'woman': [], 'someone': [], 'other': []},
    # ...
    # 'roberta-large': {'man': [], 'woman': [], 'someone': [], 'other': []}}
    model_results = {
        model: {participants: [] for participants in ALL_PARTS}
        for model in MODELS_PARAMS
    }
    # Will init model_results similarly but with defaultdict as below for plotting
    stats_model_results = {model: defaultdict() for model in MODELS_PARAMS}

    for f in files:
        prog = re.compile(r".*(.)p_wikibio_([\.\w-]+)_(date)_+(\w+)_(\w+)_([01])_([\w]+)")
        m = prog.match(f.stem)
        try:
            (
                gender,
                model_name,
                indie_var_extracted,
                occupation,
                participant,
                template_version,
                inference_version,
            ) = (
                m.group(1),
                m.group(2),
                m.group(3),
                m.group(4),
                m.group(5),
                m.group(6),
                m.group(7),
            )
        except AttributeError as e:
            if VERBOSE:
                print(f"{e} in {f.stem}.")
            continue

        model_name = file_save_model_name_2_model[model_name]
        is_instruction = inference_version.split("_")[-1].startswith("P")
        col_name = f"{gender}_{indie_var_extracted}_{occupation}_{template_version}"
        try:
            if is_instruction and inference_version != intended_instruction_version:
                continue

            data = pd.read_csv(f)
            if indie_vars is None:
                indie_vars = data[indie_var_extracted]
            data.drop("Unnamed: 0", axis=1, inplace=True)

            # Dropping first row, that contains the input text, eg dropping:
            # "In DATE: The counselor disclosed to the patient that he was..."
            data.drop(indie_var_extracted, axis=1, inplace=True)
            data.columns = [col_name]

            if not WINO_G_ID and participant not in STATIC_WINO_PARTICIPS:
                participant = "other"

            try:
                model_results[model_name][participant].append(data)
            except KeyError:
                continue  # we dont have inference run for all models yet.

            particip_version = f"{participant}_{template_version}"
            if not stats_model_results[model_name].keys().__contains__(occupation):
                stats_model_results[model_name][occupation] = defaultdict()
            if (
                not stats_model_results[model_name][occupation]
                .keys()
                .__contains__(particip_version)
            ):
                stats_model_results[model_name][occupation][
                    particip_version
                ] = defaultdict()
            stats_model_results[model_name][occupation][particip_version] = data

        except pd.errors.EmptyDataError as _e:
            print(f"No data for {col_name} in {f.stem}")

    return model_results, stats_model_results, indie_var_extracted


# %%


def load_examples(path=WINOGENDER_SCHEMA_PATH):
    bergsma_pct_female = {}
    bls_pct_female = {}
    with open(os.path.join(path, OCC_STATS_FILE)) as f:
        next(f, None)  # skip the headers
        for row in csv.reader(f, delimiter="\t"):
            occupation = row[0]
            bergsma_pct_female[occupation] = float(row[1])
            bls_pct_female[occupation] = float(row[2])
    return bergsma_pct_female, bls_pct_female


# %%
def get_wino_results_uncertainty(path, model_results, bls_pct_female_sorted_dict):
    check_col = "man" if not WINO_G_ID else "male"

    # only populated values for models already eval-ed
    wino_results = {
        model: dict()
        for model in model_results.keys()
        if model_results[model][check_col]
    }
    for model in wino_results.keys():
        try:
            for particip in ALL_PARTS:
                wino_results[model][particip] = pd.concat(
                    model_results[model][particip], axis=1
                )
        except:
            if VERBOSE:
                print(f"Nothing to concat: {model}, {particip}.")
            continue

    detail_filters = "f_0", "f_1", "n_0", "n_1", "m_0", "m_1"

    wino_results_uncertainty = {model: dict() for model in wino_results.keys()}

    for occ in bls_pct_female_sorted_dict.keys():
        for model in wino_results_uncertainty.keys():
            wino_results_uncertainty[model][occ] = dict()
            occ_filter = f"date_{occ}"

            for particip in ALL_PARTS:
                wino_results_uncertainty[model][occ][particip] = dict()

                for filter_version in detail_filters:
                    f = filter_version.split("_")
                    filter = f"{f[0]}_{occ_filter}_{f[1]}"

                    filtered_results = wino_results[model][particip].filter(
                        like=filter, axis=1
                    )

                    wino_results_uncertainty[model][occ][particip][
                        filter_version
                    ] = get_uncertainty_metric(filtered_results)

    return wino_results, wino_results_uncertainty


# %%


def threshold_df(df, col, threshold, greater_than=False):
    if greater_than:
        return df[df[col] > threshold]
    else:
        return df[df[col] <= threshold]


# %%
def get_uc_model_stats(stats_model_results, model_name):
    uc_model_stats = {
        occ: defaultdict() for occ in stats_model_results[model_name].keys()
    }
    for occ in uc_model_stats.keys():
        for p_name, p_data in stats_model_results[model_name][occ].items():
            uc_model_stats[occ][p_name] = get_uncertainty_metric(p_data)

    uc_model_stats_df = pd.DataFrame.from_dict(uc_model_stats).transpose()

    uc_model_stats_df.to_csv(
        os.path.join(STATS_FULL_PATH, f"{model_name}_delta_stats.csv")
    )
    return uc_model_stats_df


def get_true_positives(uc_model_stats_df, underspec_particip_list, threshold=1.0):
    tp = 0
    for p_v in underspec_particip_list:
        tp += len(threshold_df(uc_model_stats_df, p_v, threshold, greater_than=True))
    all_p = len(underspec_particip_list) * len(uc_model_stats_df)
    return round(tp / all_p, DECIMAL_PLACES)


def get_true_negatives(uc_model_stats_df, wellspec_particip_list, threshold=1.0):
    tn = 0
    for p_v in wellspec_particip_list:
        tn += len(threshold_df(uc_model_stats_df, p_v, threshold))
    all_n = len(wellspec_particip_list) * len(uc_model_stats_df)
    return round(tn / all_n, DECIMAL_PLACES)


# %%


def plot_occ_detail(wino_results, model="roberta-large", occ="Doctor"):
    for g in ["f"]:
        sent_n = 0
        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(6)
        for i, particip in enumerate(ALL_PARTS):
            sent_n += 1
            # Arbitrarily selecting female pronouns for this plot
            ax.plot(
                ORIGINAL_DATES,
                wino_results[model][particip][f"{g}_date_{occ.lower()}_0"],
                color=COLORS[i],
                marker="o",
                alpha=0.5,
                label=f"{sent_n}) Partcipant: '{particip}'; Coreferent: {occ}",
            )
        for i, particip in enumerate(ALL_PARTS):
            sent_n += 1
            ax.plot(
                ORIGINAL_DATES,
                wino_results[model][particip][f"{g}_date_{occ.lower()}_1"],
                color=COLORS[i],
                marker="v",
                alpha=0.5,
                label=f"{sent_n}) Partcipant: '{particip}'; Coreferent: '{particip}'",
            )

        handles, labels = ax.get_legend_handles_labels()

        legend_loc = "center"
        legend_anchor = (0.0, 0.0, 1.0, 1.33)

        fig.legend(handles, labels, loc=legend_loc, bbox_to_anchor=legend_anchor)

        plt.ylabel("Softmax Probability over Gendered Predictions")

        plt.title(
            f"Softmax Probability for {PRONOUN_DICT[g]} Pronouns vs Date in {occ} Sentence"
        )
        file_path = os.path.join(PLOTS_FULL_PATH, f"{g}_{occ}_{model}")
        plt.savefig(file_path, dpi=200)
        print(f"Plot saved to {file_path}")


# %%
def plot_all_winogender_occs(
    bls_pct_female_sorted_dict,
    wino_results_uncertainty,
    intended_version,
    occ_detail="doctor",
):
    if not WINO_G_ID:
        VERSIONS = {("f_0", "n_0"): "Professional", ("f_1", "n_1"): "Participant"}
        legend_prefix = "Participant is "
        style = "Coreference"
    else:
        VERSIONS = {("f_0", "n_0"): "Professional"}
        legend_prefix = ""
        style = "Pronoun"

    MARKERS = (
        ["1", "_", "2", "|", "3", "4", ".", "x"] if not WINO_G_ID else ["|", "_", "3"]
    )
    for model in wino_results_uncertainty.keys():
        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(20)
        added_legend_items = False
        for occ in bls_pct_female_sorted_dict.keys():
            j = 0
            for i, particip in enumerate(ALL_PARTS):
                for versions, template_type in VERSIONS.items():
                    if template_type == "Professional":
                        colors = PROFESSIONAL_COLORS
                    else:
                        colors = PARTICIP_COLORS

                    spec_metric = 0
                    for version in versions:  # sum both female and neutral texts
                        spec_metric += wino_results_uncertainty[model][occ][particip][
                            version
                        ]
                    if spec_metric == math.nan or math.isnan(spec_metric):
                        spec_metric = -100.0  # plot off page

                    if not WINO_G_ID:
                        legend_postfix = f". Coref with {template_type}"
                    else:
                        legend_postfix = ""
                    if not added_legend_items:
                        particip_label = (
                            f"'{particip}'" if particip != "other" else "<other>"
                        )
                        ax.scatter(
                            occ,
                            spec_metric,
                            color=colors[i],
                            marker=MARKERS[j],
                            alpha=0.95,
                            label=legend_prefix + particip_label + legend_postfix,
                        )
                    else:
                        ax.scatter(
                            occ,
                            spec_metric,
                            color=colors[i],
                            marker=MARKERS[j],
                            alpha=0.95,
                        )
                    j += 1
                    if occ == occ_detail:
                        print(f"uncertainty: {model} {occ} {particip} {version}")
                        print(spec_metric)
            added_legend_items = True

        ax.tick_params(axis="x", labelrotation=90)
        ax.set_ylim([-0.1, 100]) 
        ax.set_yscale("symlog")
        ax.margins(x=0.01)
        fig.tight_layout()
        plt.legend(ncol=2, fontsize=16, loc="upper left")
        title = f" {model} Task Specification for {style} Resolution"
        plt.ylabel("Specification Metric")
        plt.title(title)
        model_file_save_name = convert_to_file_save_name(
            MODELS_PARAMS[model]["model_call_name"]
        )
        filename = f"bls_{model_file_save_name}_{template_type}_winomod{WINO_G_ID}_{intended_instruction_version}"
        file_path = os.path.join(PLOTS_FULL_PATH, filename.replace('.', '_'))
        print(f"Plot saved to {file_path}")
        try:
            plt.savefig(file_path, dpi=150)
        except matplotlib.units.ConversionError as e:
            print(e)
            continue


if __name__ == "__main__":
    # %%
    assert (
        UNDERSP_METRIC == True
    ), "** Set `METHOD_IDX` to `specification_metric` config.py for this script **"

    intended_instruction_version = INTENDED_INST_VERSION

    # Per the winogender_schema
    if not WINO_G_ID:
        wellspec_particip_versions = ["man_1", "woman_1"]
        underspec_particip_versions = [
            "man_0",
            "woman_0",
            "someone_0",
            "other_0",
            "someone_1",
            "other_1",
        ]
    else:
        wellspec_particip_versions = [f"{g_id}_0" for g_id in GENDERED_IDS]
        underspec_particip_versions = [
            "unspecified_0",
        ]

    _, bls_pct_female = load_examples()
    bls_pct_female_sorted = sorted(bls_pct_female.items(), key=lambda item: item[1])
    bls_pct_female_sorted_dict = {
        occ: {"pct": pct} for occ, pct in bls_pct_female_sorted
    }

    (
        model_results,
        stats_model_results,
        indie_var_extracted,
    ) = get_df_per_model_from_all_files_in_dir(
        INFERENCE_FULL_PATH, intended_instruction_version=intended_instruction_version
    )

    # %%

    wino_results, wino_results_uncertainty = get_wino_results_uncertainty(
        INFERENCE_FULL_PATH, model_results, bls_pct_female_sorted_dict
    )

    sns.set_context("poster")
    plot_all_winogender_occs(
        bls_pct_female_sorted_dict,
        wino_results_uncertainty,
        intended_instruction_version,
    )

    # %%
    if not TESTING:
        tptn = {}
        threshold = 0.5
        for model in wino_results_uncertainty.keys():
            uc_model_stats = get_uc_model_stats(stats_model_results, model)

            # Correct detection of underspecified tasks
            tpr_all = get_true_positives(
                uc_model_stats,
                underspec_particip_versions,
                threshold=threshold,
            )

            # Correct detection of well-specified tasks
            tnr_all = get_true_negatives(
                uc_model_stats,
                wellspec_particip_versions,
                threshold=threshold,
            )
            tptn[model] = {"tp": tpr_all, "tn": tnr_all, "b_acc" : (tpr_all + tnr_all)/2}

            if not WINO_G_ID:
                tpr_gender = get_true_positives(
                    uc_model_stats,
                    ["man_0", "woman_0"],
                    threshold=threshold,
                )
                tptn[model] = {"tp": tpr_all, "tn": tnr_all,  "b_acc" : (tpr_all + tnr_all)/2}
                tp_gender_string = f"and true postive gender co-occuring: {tpr_gender}"
            else:
                tp_gender_string = ""

            print(
                f"""For {model}: 
                true postive rate: {tpr_all} 
                true negative rate: {tnr_all} 
                balanced accuracy: {(tpr_all + tnr_all)/2}
                {tp_gender_string}
                """
            )
            tptn_df = pd.DataFrame(tptn).T
            tptn_df.to_csv(
                os.path.join(
                    STATS_FULL_PATH,
                    f"tptn_winomod{WINO_G_ID}_{INTENDED_INST_VERSION}.csv",
                )
            )


# %%
