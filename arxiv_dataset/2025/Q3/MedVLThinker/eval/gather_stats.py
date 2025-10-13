import collections
import json
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
import pandas as pd


@click.command()
@click.option(
    "--input_dir",
    "-d",
    type=click.Path(exists=True),
    default="outputs",
    help="Directory containing the result files.",
)
@click.option(
    "--output_file", "-o", type=click.Path(), default="outputs/gathered_results.tsv"
)
def main(**kargs):
    args = SimpleNamespace(**kargs)

    input_dir = Path(args.input_dir)

    # recursively find all `acc-eval_results.json`
    result_files = list(input_dir.rglob("acc-eval_results.json"))
    if not result_files:
        click.echo("No acc-eval_results.json files found.")
        return

    click.echo(f"Found {len(result_files)} acc-eval_results.json files.")
    all_results = collections.defaultdict(list)
    for file_path in result_files:
        click.echo(f"Parsing {file_path}")
        results = parse_acc_eval_results(file_path)

        for acc_type, acc_data in results.items():
            all_results[acc_type].append(acc_data)

    for acc_type in all_results:
        # Convert lists of dicts to DataFrame
        df = pd.DataFrame(all_results[acc_type])
        # Set the index to file_path
        df = df.set_index("file_path")
        # Sort by index
        df = df.sort_index()
        # Rename columns to include the accuracy type
        all_results[acc_type] = df

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    is_first = True
    for acc_type, output_df in all_results.items():
        mode = "w" if is_first else "a"
        is_first = False

        output_df.to_csv(output_file, sep="\t", mode=mode)
        click.echo(f"Results saved to {output_file}")


def parse_acc_eval_results(file_path):
    """Parse a single acc-eval_results.json file.
    {
    "accuracy_total": {
        "GBaker/MedQA-USMLE-4-options": 0.10385916359163591,
        "openlifescienceai/headqa": 0.08821770334928229,
        "openlifescienceai/medmcqa": 0.11620646593589566
    },
    "accuracy_pass@total": {
        "GBaker/MedQA-USMLE-4-options": 0.6783517835178352,
        "openlifescienceai/headqa": 0.5980861244019139,
        "openlifescienceai/medmcqa": 0.7236020535590398
    },
    "total_num_rollouts": 375296,
    "total_num_correct": 43197,
    "num_samples": 23456
    }
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    parsed_data_dict = {}
    for acc_type, acc_dict in data.items():
        parsed_data = {}
        if not acc_type.startswith("accuracy_"):
            continue
        for dataset_name, acc_value in acc_dict.items():
            key = dataset_name
            parsed_data[key] = acc_value
        parsed_data["file_path"] = str(file_path)
        parsed_data_dict[acc_type] = parsed_data

    return parsed_data_dict


if __name__ == "__main__":
    main()
