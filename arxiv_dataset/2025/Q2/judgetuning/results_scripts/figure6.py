import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

parent_dir = Path(__file__).parent.parent

# Add it to the sys.path
if parent_dir not in sys.path:
    sys.path.append(str(parent_dir))

from judgetuning.utils import load_judge_evaluations


def rename(s: str):
    if "/" not in s:
        return s
    s = s.split("/")[1].lower()
    for token in ["-instruct", "-gptq-int8", "-it", "-fp8", "meta-"]:
        s = s.replace(token, "")
    return s


def model_size(s: str):
    return int(s.split("-")[-1].replace("b", ""))


if __name__ == "__main__":

    df_configs_per_eval, mat_cost_per_eval, mat_human_agreement_per_eval = (
        load_judge_evaluations()
    )
    df_configs = df_configs_per_eval[400]

    cols = [
        "model",
        "temperature",
        "n_sample_with_shift",
        "score_type",
        "provide_explanation",
        "provide_example",
        "provide_answer",
        "json_output",
    ]

    for col in cols:
        print(col, sorted(df_configs.loc[:, col].unique()))

    col_order = {
        "model": [
            "qwen2.5-7b",
            "llama-3.1-8b",
            "gemma-2-9b",
            "gemma-2-27b",
            "qwen2.5-32b",
            "qwen2.5-72b",
            "llama-3.1-70b",
        ],
        "temperature": [0.0, 0.01, 0.1, 1.0],
        "n_sample_with_shift": [0, 1],
        "score_type": ["best-model", "preference", "multi", "likert", "pair"],
        "provide_explanation": [False, True],
        "provide_example": [False, True],
        "provide_answer": [False, True],
        "json_output": [False, True],
    }

    fig, axes = plt.subplots(1, len(cols), figsize=(12, 3), sharey=True, sharex="col")
    N = 100

    models = df_configs.model.apply(rename).unique()

    df_plot = df_configs.copy()
    # VLLM overides temp to 1e-2 min
    df_plot.temperature = df_plot.temperature.replace(0.0001, 0.01)
    df_leaderboard = df_plot.sort_values("human_agreement", ascending=False).loc[
        :, ["human_agreement"] + cols
    ]
    df_leaderboard.model = df_leaderboard.model.apply(rename)
    df_leaderboard["model_size"] = df_leaderboard.model.apply(rename).apply(model_size)
    df_top_large = df_leaderboard.head(N).copy()
    df_top_large["model_size_flag"] = ">10B"
    df_top_small = (
        df_leaderboard.loc[df_leaderboard.model_size < 10, :]
        .sort_values("human_agreement", ascending=False)
        .head(N)
    )
    df_top_small["model_size_flag"] = "<10B"
    df_top = pd.concat([df_top_large, df_top_small]).reset_index(drop=True)
    df_top = df_top.replace("best-model-identifier", "best-model")

    for i, col in enumerate(cols):
        ax = axes[i]
        xx = (
            df_top[[col, "model_size_flag"]].groupby([col, "model_size_flag"]).size()
            / N
        )
        # plot with a hue of model_size_flag no legend needed and rename the titles
        # if col == "model": i want to order the x axis according to the model size
        xx = xx.unstack().reindex(col_order[col]).stack()
        xx.unstack().plot(kind="bar", ax=ax, legend=False, xlabel=col.replace("_", "-"))
        ax.set_ylabel("")
        ax.xaxis.set_label_position("top")
        ax.grid()

        if i == 0:
            ax.legend(loc="upper left", prop={"size": 10})

    # fig.text(0.04, 0.5, 'Proportion in top 100', va='center', rotation='vertical')
    fig.supylabel("Proportion in top 100")
    # plt.suptitle("Hyperparameter analysis")
    plt.tight_layout()
    plt.show()
    # plt.suptitle("Survival analysis")
    plt.savefig(parent_dir / "figures/hyperparameter-importance.pdf")
