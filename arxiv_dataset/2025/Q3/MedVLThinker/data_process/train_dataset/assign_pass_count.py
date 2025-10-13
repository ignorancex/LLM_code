"""
Add `pass_count` field to the dataset based on the results from a JSONL file.

`pass_count=-1` indicates that the sample was not evaluated.

python data_process/train_dataset/assign_pass_count.py \
    --dataset_name UCSC-VLAA/MedVLThinker-m23k-tokenized \
    --split train \
    --results_jsonl_path outputs/estimate_pass_rate/Qwen2.5-VL-7B-Instruct/med-vlm-m23k/eval_results.jsonl
"""

import dotenv

dotenv.load_dotenv(override=True)


import json
from pathlib import Path
from types import SimpleNamespace

import click
import datasets


@click.command()
@click.option(
    "--dataset_name", type=str, required=True, help="Path to the dataset directory."
)
@click.option(
    "--subset", type=str, default=None, help="Subset of the dataset to process."
)
@click.option(
    "--split", type=str, default="train", help="Split of the dataset to process."
)
@click.option(
    "--results_jsonl_path",
    type=str,
    required=True,
    help="Path to the results JSONL file.",
)
@click.option(
    "--num_proc",
    type=int,
    default=16,
    help="Number of processes to use for parallel processing.",
)
def main(**kargs):
    args = SimpleNamespace(**kargs)

    dataset = args.dataset_name
    subset = args.subset
    split = args.split
    dataset = datasets.load_dataset(dataset, subset, split=split)
    print(f"Loaded dataset: {dataset}: {len(dataset)} samples")

    results_jsonl_path = Path(args.results_jsonl_path)
    num_proc = args.num_proc
    dataset = assign_pass_count(dataset, results_jsonl_path, num_proc)
    breakpoint()


def assign_pass_count(dataset, results_jsonl_path, num_proc, keep_in_memory=False):
    results_jsonl_path = Path(results_jsonl_path)
    if not results_jsonl_path.exists():
        raise FileNotFoundError(f"Results JSONL file not found: {results_jsonl_path}")

    dataset_index_to_pass_count = {}
    with results_jsonl_path.open("r") as f:
        for line in f:
            sample = json.loads(line)
            dataset_index = sample["dataset_index"]
            pass_count = sample["num_correct"]
            dataset_index_to_pass_count[dataset_index] = pass_count
    print(f"Loaded pass counts for {len(dataset_index_to_pass_count)} samples")

    def assign_pass_count(sample):
        dataset_index = sample["dataset_index"]
        if dataset_index in dataset_index_to_pass_count:
            sample["pass_count"] = dataset_index_to_pass_count[dataset_index]
        else:
            sample["pass_count"] = -1
        return sample

    dataset = dataset.map(
        assign_pass_count,
        num_proc=num_proc,
        desc="Assigning pass counts",
        keep_in_memory=keep_in_memory,
    )
    expected_num_missed_samples = len(dataset) - len(dataset_index_to_pass_count)
    num_missed_samples = len(
        dataset.filter(
            lambda x: x["pass_count"] == -1, num_proc=num_proc, keep_in_memory=True
        )
    )
    print(
        f"Expected missed samples: {expected_num_missed_samples}, Actual missed samples: {num_missed_samples}"
    )
    return dataset


if __name__ == "__main__":
    main()
