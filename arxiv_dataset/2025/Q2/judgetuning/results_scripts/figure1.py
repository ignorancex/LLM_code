import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from judgetuning.judge_paths import judge_tuning_datadir
from judgetuning.utils import figure_path, download_hf
from judgetuning.annotation_dataset import AlpacaEvalDataset, ArenaHardDataset


def load_dataframe(name: str):
    path = judge_tuning_datadir / "scaling-eval-data" / name
    if not path.exists():
        print(f"Local folder {path} not found, downloading.")
        download_hf(name=f"scaling-eval-data/{name}", local_path=path.parent.parent)
    else:
        print(f"Local folder {path} found, loading eval data from there.")
    return pd.read_csv(path)


def load_rows_spearman(n_instrs: list[int]):
    models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "length",
    ]

    df_ae = load_dataframe("alpaca-eval.csv.zip")
    df_ah = load_dataframe("arena-hard.csv.zip")
    df_ae_pivot = df_ae.pivot_table(
        index=["instruction_index"], columns=["model", "model2"], values="preference"
    )
    df_ah_pivot = df_ah.pivot_table(
        index=["instruction_index"], columns=["model", "model2"], values="preference"
    )

    ae_models = df_ae_pivot[models[0]].columns.tolist()
    ah_models = df_ah_pivot[models[0]].columns.tolist()
    test_models = list(set(ae_models).intersection(ah_models))
    val_ae_models = [x for x in ae_models if x not in test_models]
    val_ah_models = [x for x in ah_models if x not in test_models]
    print(
        f"Number of models in test/validation AE/validation AH: {len(test_models), len(val_ae_models), len(val_ah_models)}"
    )

    df_both = pd.concat(
        [df_ae[df_ae.model2.isin(test_models)], df_ah[df_ah.model2.isin(test_models)]],
        ignore_index=False,
    )
    df_both_pivot = df_both.pivot_table(
        index=["instruction_index"], columns=["model", "model2"], values="preference"
    )

    ae_dataset = AlpacaEvalDataset()
    chatbot_ae = ae_dataset.chatbot_arena_elo_ratings()

    ah_dataset = ArenaHardDataset()
    chatbot_ah = ah_dataset.chatbot_arena_elo_ratings()

    chatbot_test = chatbot_ah.loc[test_models]
    # chatbot_val_ae = chatbot_ae.loc[val_ae_models]
    # chatbot_val_ah = chatbot_ah.loc[val_ah_models]

    n_seeds = 100
    rows_spearman = []
    for n_seed in tqdm(list(range(n_seeds))):
        for n_instr in n_instrs:
            random_instr = np.random.choice(df_both_pivot.index.tolist(), n_instr)
            for model in models:
                # select data
                corr = spearmanr(
                    df_both_pivot.loc[random_instr, model]
                    .mean(axis=0)
                    .loc[test_models],
                    chatbot_test,
                )[0]
                # a bit slow
                rows_spearman.append(
                    {
                        "model": model,
                        "#instructions": n_instr,
                        "corr": corr,
                        "split": "both",
                    }
                )

                random_instr = np.random.choice(df_ae_pivot.index.tolist(), n_instr)
                corr = spearmanr(
                    df_ae_pivot.loc[random_instr, model].mean(axis=0).loc[test_models],
                    chatbot_test,
                )[0]
                rows_spearman.append(
                    {
                        "model": model,
                        "#instructions": n_instr,
                        "corr": corr,
                        "split": "AE",
                    }
                )

                random_instr = np.random.choice(df_ah_pivot.index.tolist(), n_instr)
                corr = spearmanr(
                    df_ah_pivot.loc[random_instr, model].mean(axis=0).loc[test_models],
                    chatbot_test,
                )[0]
                rows_spearman.append(
                    {
                        "model": model,
                        "#instructions": n_instr,
                        "corr": corr,
                        "split": "AH",
                    }
                )
    return rows_spearman


def main():

    rows_spearman = load_rows_spearman([10, 25, 50, 100, 200, 400, 800, 1600])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True, sharex=False)
    df_results = pd.DataFrame(rows_spearman)
    df_results = df_results[(df_results.model != "length")]
    df_results.model = df_results.model.apply(lambda s: s.split("/")[1].split("-")[1])

    df_results_length = pd.DataFrame(rows_spearman)
    length_corr_dict = (
        df_results_length.loc[df_results_length.model == "length", ["split", "corr"]]
        .groupby("split")
        .mean()
        .to_dict()["corr"]
    )

    color_index = {
        "0.5B": 0,
        "1.5B": 1,
        "3B": 2,
        "7B": 3,
        "32B": 4,
        "72B": 5,
    }
    df_results["color_index"] = df_results.model.apply(lambda s: color_index[s])

    lineplot_common_kwargs = dict(
        x="#instructions",
        y="corr",
        hue="model",
        marker=".",
        legend=None,
        hue_order=list(color_index.keys()),
        palette="viridis",
        errorbar="sd",
    )
    ax = axes[0]
    ax = sns.lineplot(
        df_results[(df_results.split == "AE") & (df_results["#instructions"] <= 805)],
        ax=ax,
        **lineplot_common_kwargs,
    )
    ax.axhline(length_corr_dict["AE"], label="length", ls="dotted", color="black")
    ax.set_title("Instruction Alpaca-Eval")

    ax = axes[1]
    ax = sns.lineplot(
        df_results[(df_results.split == "AH") & (df_results["#instructions"] <= 500)],
        ax=ax,
        **lineplot_common_kwargs,
    )
    ax.axhline(length_corr_dict["AH"], label="length", ls="dotted", color="black")
    ax.set_title("Instruction Arena-Hard")

    ax = axes[2]
    lineplot_common_kwargs.pop("legend")
    ax = sns.lineplot(
        df_results[df_results.split == "both"], ax=ax, **lineplot_common_kwargs
    )
    ax.axhline(length_corr_dict["both"], label="length", ls="dotted", color="black")
    ax.legend(loc="right")
    # TODO show length in plot
    for ax in axes:
        ax.grid()
        ax.set_xscale("log")
        ax.set_xlabel("# Instructions")
        ax.set_ylabel("Spearman correlation")
        ax.set_ylim([None, 1.0])
    ax.set_title("Instruction both")
    plt.tight_layout()
    plt.savefig(figure_path(None, "scaling-spearman-corr.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
