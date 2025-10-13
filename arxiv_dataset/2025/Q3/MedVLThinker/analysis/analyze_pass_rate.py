import dotenv

dotenv.load_dotenv(override=True)
import collections
import json
import os
from pathlib import Path
from types import SimpleNamespace

import click
import matplotlib.pyplot as plt
import pandas as pd
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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(
        num_correct_counter.values(),
        labels=num_correct_counter.keys(),
        autopct="%1.1f%%",
        startangle=90,
        colors=plt.cm.tab20.colors[: len(num_correct_counter)],
    )

    ax.set_title(f"Pass Rate Distribution ({len(results_df)} samples)\n{result_dir}")

    save_fig_path = result_dir / "pass_rate_distribution.png"
    fig.savefig(save_fig_path, bbox_inches="tight", dpi=300)
    save_fig_path = result_dir / "pass_rate_distribution.pdf"
    fig.savefig(save_fig_path, bbox_inches="tight")
    print(f"Pass rate distribution saved to {save_fig_path}")


if __name__ == "__main__":
    main()
