"""
First add `pass_count` field, then remove samples with certain pass count.

Add `pass_count` field to the dataset based on the results from a JSONL file.

`pass_count=-1` indicates that the sample was not evaluated.

python data_process/train_dataset/order_easy_to_hard.py \
    --dataset_name UCSC-VLAA/MedVLThinker-m23k-tokenized \
    --split train \
    --results_jsonl_path outputs/estimate_pass_rate/Qwen2.5-VL-7B-Instruct/med-vlm-m23k/eval_results.jsonl
    --save_to_disk_path data/local/med-vlm-m23k-easy_to_hard
"""

import dotenv

dotenv.load_dotenv(override=True)


import json
from pathlib import Path
from types import SimpleNamespace

import click
import datasets
from assign_pass_count import assign_pass_count


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
@click.option(
    "--remove_pass_count_below",
    type=int,
    default=1,
    help="Remove samples with pass count below this value.",
)
@click.option(
    "--remove_pass_count_above",
    type=int,
    default=7,
    help="Remove samples with pass count above this value.",
)
@click.option(
    "--order_type",
    type=str,
    default="easy_to_hard",
    help="Order type: 'easy_to_hard' or 'hard_to_easy'.",
)
# @click.option("--shuffle_before_ordering", is_flag=True, default=False, help="Shuffle the dataset before ordering.")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
@click.option(
    "--save_to_disk_path",
    type=str,
    required=True,
    help="Path to save the ordered dataset to disk.",
)
@click.option(
    "--keep_in_memory",
    type=bool,
    default=True,
    help="Keep the dataset in memory after processing.",
)
def main(**kargs):
    args = SimpleNamespace(**kargs)
    print(f"Arguments: {args}")

    dataset = args.dataset_name
    subset = args.subset
    split = args.split
    dataset = datasets.load_dataset(dataset, subset, split=split)
    print(f"Loaded dataset: {dataset}: {len(dataset)} samples")

    results_jsonl_path = Path(args.results_jsonl_path)
    num_proc = args.num_proc
    keep_in_memory = args.keep_in_memory
    dataset = assign_pass_count(dataset, results_jsonl_path, num_proc, keep_in_memory)

    # remove
    remove_pass_count_below = args.remove_pass_count_below
    remove_pass_count_above = args.remove_pass_count_above
    print(
        f"Removing samples with pass count below {remove_pass_count_below} and above {remove_pass_count_above}."
    )

    def filter_pass_count(sample):
        pass_count = sample.get("pass_count", -1)
        return (
            pass_count >= remove_pass_count_below
            and pass_count <= remove_pass_count_above
        )

    num_samples_before = len(dataset)
    dataset = dataset.filter(
        filter_pass_count, num_proc=num_proc, keep_in_memory=keep_in_memory
    )
    num_samples_after = len(dataset)
    print(f"Filtered dataset: {num_samples_before} -> {num_samples_after} samples")
    print(f"Remaining pass counts: {set(dataset['pass_count'])}")

    # order
    shuffle_before_ordering = True
    if shuffle_before_ordering:
        print("Shuffling dataset before ordering.")
        dataset = dataset.shuffle(seed=args.seed, keep_in_memory=keep_in_memory)

    order_type = args.order_type
    if order_type not in ["easy_to_hard", "hard_to_easy"]:
        raise ValueError(
            f"Invalid order_type: {order_type}. Must be 'easy_to_hard' or 'hard_to_easy'."
        )

    # sort api in datasets: https://huggingface.co/docs/datasets/v3.6.0/en/package_reference/main_classes#datasets.Dataset.sort
    print(f"Ordering dataset {order_type}.")
    if order_type == "easy_to_hard":
        # from large to small, reversed=True
        dataset = dataset.sort(
            "pass_count", reverse=True, keep_in_memory=keep_in_memory
        )
    else:
        dataset = dataset.sort(
            "pass_count", reverse=False, keep_in_memory=keep_in_memory
        )

    # save
    save_to_disk_path = args.save_to_disk_path
    dataset.save_to_disk(
        save_to_disk_path,
        num_proc=num_proc,
    )
    print(f"Saved ordered dataset to {save_to_disk_path}.")


if __name__ == "__main__":
    main()
