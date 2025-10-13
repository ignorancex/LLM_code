import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from judgetuning.utils import figure_path
from results_scripts.figure1 import load_dataframe

fig, ax = plt.subplots(1, 1, figsize=(5, 3))


def proj(x, threshold: float = 0.0):
    if x < 0.5 - threshold:
        return 0
    elif x > 0.5 + threshold:
        return 1
    else:
        return 0.5


def compute_human_agreement(df):
    return (df.preference.apply(proj) == df.human_preference).mean()


def bootstrap_human_agreement(df, n_instr: int | None = None, n_seeds: int = 10):
    if n_instr is None:
        vals = [
            compute_human_agreement(df.sample(frac=1, replace=True))
            for _ in range(n_seeds)
        ]
    else:
        vals = [
            compute_human_agreement(df.sample(n=n_instr, replace=True))
            for _ in range(n_seeds)
        ]
    return vals


def load_df_human_agreement_scaling():
    models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
    ]
    annotation_dict_human = {}
    for model in models:
        name = model.split("/")[1] + ".csv.zip"
        annotation_dict_human[model] = load_dataframe(name)
    n_seeds = 1000
    rows_human = []
    for seed in tqdm(range(n_seeds)):
        for model in annotation_dict_human.keys():
            df_to_eval = annotation_dict_human[model]
            for n_instr in [10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6500]:
                rows_human.append(
                    {
                        "#instructions": n_instr,
                        "Human agreement": compute_human_agreement(
                            df_to_eval.sample(n=n_instr, replace=True)
                        ),
                        "model": model,
                    }
                )

    return pd.DataFrame(rows_human)


if __name__ == "__main__":
    df_results_human = load_df_human_agreement_scaling()

    df_results_human.model = df_results_human.model.apply(
        lambda s: s.split("/")[1].split("-")[1]
    )

    hue_order = ["0.5B", "1.5B", "3B", "7B", "32B", "72B"]
    sns.lineplot(
        df_results_human,
        x="#instructions",
        y="Human agreement",
        hue="model",
        hue_order=hue_order,
        palette="viridis",
        errorbar="sd",
        marker=".",
        ax=ax,
    )
    ax.set_xlim([10, 6400])
    ax.axhline(1.0 / 3, color="red", label="Random", ls="dotted")
    # run evaluate_length.sh to get this number
    ax.axhline(0.4255343337880, color="black", label="Length", ls="dotted")

    ax.legend(loc="lower center", ncols=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    plt.tight_layout()
    plt.savefig(figure_path(None, "scaling-human-agreement.pdf"))
    plt.show()
