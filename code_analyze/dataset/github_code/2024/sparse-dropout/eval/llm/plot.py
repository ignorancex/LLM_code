import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from absl import app

from eval.utils import load_runs


def main(_):
    import miscellaneous.plot_style

    wandb.login()
    run_ids = list(range(6, 20))
    runs = load_runs("andylolu2", "flash-dropout-llm", run_ids)

    metrics = {
        "train/loss": ("Train loss", "min"),
        "val/loss": ("Loss", "min"),
    }

    data = []
    for run in runs:
        logs = list(run.scan_history(page_size=10000))
        logs_df = pd.DataFrame(logs)

        for metric, (_, mode) in metrics.items():
            agg = np.nanmax if mode == "max" else np.nanmin
            data.append(
                {
                    "p": run.config["model"]["dropout"]["p"],
                    "Variant": run.config["model"]["dropout"]["variant"],
                    "metric": metric,
                    "value": agg(logs_df[metric]),
                }
            )

    df = pd.DataFrame(data)
    print(df.sort_values(["metric", "Variant", "p"]))

    def sem(x):
        return x.sem(ddof=0)

    stats = df.groupby(["Variant", "p", "metric"])["value"].agg(["mean", sem])
    stats["interval"] = 1.96 * stats["sem"]
    for name, group in stats.groupby("metric"):
        print("\n", f"--- {name} ---")
        print(group.sort_values("mean"))

    for ms in [["train/loss", "val/loss"]]:
        fig, ax = plt.subplots(figsize=(5, 4))
        sub_df = df[df["metric"].isin(ms)]
        sub_df = sub_df.rename(columns={"metric": "Metric"})
        sub_df["Metric"] = sub_df["Metric"].replace(
            {
                "train/loss": "Train loss",
                "val/loss": "Val loss",
            }
        )
        sub_df["Variant"] = sub_df["Variant"].replace(
            {
                "blockwise[cuda]": "SparseDrop",
                "vanilla": "Dropout",
            }
        )
        sns.lineplot(
            data=sub_df,
            x="p",
            y="value",
            hue="Variant",
            style="Metric",
            marker="o",
            markersize=4,
            ax=ax,
        )
        ax.set(
            xlim=(-0.02, None),
            xlabel="$p$",
            ylabel="",
        )

        fig.tight_layout()

        name = "|".join([m.replace("/", "-") for m in ms])
        fig.savefig(f"./logs/llm-shakespeare-{name}.png", dpi=300)


if __name__ == "__main__":
    app.run(main)
