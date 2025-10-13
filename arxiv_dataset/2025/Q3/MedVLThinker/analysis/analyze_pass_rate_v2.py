"""
find outputs/estimate_pass_rate -maxdepth 1 -name 'qwen*' -exec python analysis/analyze_pass_rate_v2.py -d {} \;


rm -rf outputs/estimate_pass_rate_flat
mkdir -p outputs/estimate_pass_rate_flat

for f in outputs/estimate_pass_rate/*/pass_rate_distribution.pdf; do
  # Extract model and dataset names from the path
  model=$(basename "$(dirname "$f")")
  # Create new filename
  newname="${model}_pass_rate_distribution.pdf"
  # Copy to flat directory
  cp "$f" "outputs/estimate_pass_rate_flat/$newname"
done

# Create a tar archive of the flat directory
tar -cvf outputs/estimate_pass_rate_flat/pass_rates.tar -C outputs/estimate_pass_rate_flat .
"""

import dotenv

dotenv.load_dotenv(override=True)
import collections
import json
import os
from pathlib import Path
from types import SimpleNamespace

import click
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import tqdm


@click.command()
@click.option("--result_dir", "-d", type=click.Path(exists=True), default="outputs")
def main(**kargs):
    args = SimpleNamespace(**kargs)
    result_dir = args.result_dir
    result_dir = Path(result_dir)
    # Load the gathered results

    results_file = result_dir / "eval_results.jsonl"
    if not os.path.exists(results_file):
        click.echo(f"Results file {results_file} does not exist.")
        return

    results = []
    with open(results_file, "r") as f:
        for line in tqdm.tqdm(f, desc="Loading results", unit="line"):
            sample = json.loads(line)
            results.append(
                {
                    "num_rollouts": sample["num_rollouts"],
                    "num_correct": sample["num_correct"],
                }
            )
    results_df = pd.DataFrame(results)

    num_correct_counter = collections.Counter(results_df["num_correct"])
    num_correct_counter = {
        k: num_correct_counter[k] for k in sorted(num_correct_counter.keys())
    }

    num_correct_percentages = {
        k: v / len(results_df) * 100 for k, v in num_correct_counter.items()
    }
    print(f"Number of correct answers: {num_correct_counter}")

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.pie(
    #     num_correct_counter.values(),
    #     labels=num_correct_counter.keys(),
    #     autopct="%1.1f%%",
    #     startangle=90,
    #     colors=plt.cm.tab20.colors[: len(num_correct_counter)],
    # )

    # ax.set_title(f"Pass Rate Distribution ({len(results_df)} samples)\n{result_dir}")

    # save_fig_path = result_dir / "pass_rate_distribution.png"
    # fig.savefig(save_fig_path, bbox_inches="tight", dpi=300)
    # print(f"Pass rate distribution saved to {save_fig_path}")
    plot(num_correct_counter, result_dir)


def plot(num_correct_counter, result_dir):
    # ------------------------------------------------------------------
    # 1 .  Prepare data -------------------------------------------------
    # ------------------------------------------------------------------
    # Convert your Counter/dict to a tidy Series and sort by key
    data = pd.Series(num_correct_counter, name="count").sort_index()
    # Compute percentages once for the bar labels
    pct = (data / data.sum()).round(3)  # keeps three-decimals accuracy
    labels = pct.apply(lambda x: f"{x:.1%}")  # e.g. 34.6 %

    # ------------------------------------------------------------------
    # 2 .  Figure aesthetics -------------------------------------------
    # ------------------------------------------------------------------
    sns.set_theme(
        context="paper",  # font sizes tuned for print
        style="whitegrid",  # light grid behind bars
        palette="colorblind",  # colour-blind safe palette
        font="DejaVu Sans",  # replace with journal requirement if any
    )

    fig, ax = plt.subplots(figsize=(6, 4))  # 1-column width, ~4 : 3 ratio

    # ------------------------------------------------------------------
    # 3 .  Plot ---------------------------------------------------------
    # ------------------------------------------------------------------
    # palette = sns.cubehelix_palette(
    #     n_colors=len(data),
    #     start=0.5,     # hue
    #     rot=-0.75,     # direction/amount of rotation
    #     light=0.85,    # lightest shade
    #     dark=0.25,     # darkest shade
    # )
    # generate a dark-to-light gradient, one colour per bar
    palette = sns.color_palette("crest", n_colors=len(data))  # “rocket” reversed
    # └─ alternatives: "mako", "crest", "viridis", "flare", …
    sns.barplot(
        x=data.values,
        y=data.index.astype(str),  # y-axis labels: class names as strings (0,1,…)
        orient="h",
        ax=ax,
        palette=palette,
    )

    # Add text labels at the end of each bar (percentage)
    for bar, txt in zip(ax.patches, labels):
        ax.text(
            bar.get_width() + 0.005 * data.max(),  # small offset from bar end
            bar.get_y() + bar.get_height() / 2,
            txt,
            va="center",
            ha="left",
            fontsize="small",
        )

    # ------------------------------------------------------------------
    # 4 .  Final touches -----------------------------------------------
    # ------------------------------------------------------------------
    fontsize = 22
    ax.set_xlabel("# Samples", fontsize=fontsize)
    ax.set_ylabel("# Correct Answers", fontsize=fontsize)

    # ──-  Choose which classes to label  ───────────────────────────────
    step = 2  # keep every 2nd level → 0, 2, 4, …
    keep = data.index[::step]  # data.index is an Int64Index: 0,1,2,…
    ax.set_yticks(keep)  # where the ticks sit
    ax.set_yticklabels(keep.astype(str))  # what the ticks show

    # If you also want the grid lines only at those kept positions:
    ax.yaxis.set_major_locator(mticker.FixedLocator(keep))
    ticker_fontsize = 20
    ax.tick_params(
        axis="both", labelsize=ticker_fontsize
    )  # 'both' applies to x and y axis

    sns.despine(trim=True, left=False, bottom=True)
    plt.tight_layout()

    save_fig_path = result_dir / "pass_rate_distribution.png"
    fig.savefig(save_fig_path, bbox_inches="tight", dpi=300)
    save_fig_path = result_dir / "pass_rate_distribution.pdf"
    fig.savefig(save_fig_path, bbox_inches="tight")
    print(f"Pass rate distribution saved to {save_fig_path}")


if __name__ == "__main__":
    main()
