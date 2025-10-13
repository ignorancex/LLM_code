"""
Copied from https://github.com/open-thoughts/open-thoughts/blob/main/open_thoughts/decontaminate.py
"""

import dotenv

dotenv.load_dotenv()


import json
import multiprocessing as mp
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import click
import datasets
from datasets import Dataset, concatenate_datasets, load_dataset
from rapidfuzz import fuzz, process
from tqdm import tqdm


def fuzz_string_pair(
    str1: str, values2: List[str], similarity_threshold: float
) -> List[Tuple]:
    matches_with_scores = process.extract(
        str1, values2, scorer=fuzz.ratio, score_cutoff=similarity_threshold
    )
    return [
        (str1, match_tuple[0], match_tuple[1]) for match_tuple in matches_with_scores
    ]


def decontaminate(
    dataset: Dataset,
    column="question",
    eval_json_list=["misc/m1_eval_data.json"],
    eval_columns=["question"],
    threshold=95.0,
) -> Dataset:
    """Remove rows from dataset that have similar strings in eval_datasets based on fuzzy matching."""
    n_processes = mp.cpu_count()

    # Get values from input dataset
    dataset_strings = [str(x) for x in dataset[column] if x is not None]
    indices_to_remove = set()

    for eval_json in eval_json_list:
        with open(eval_json, "r") as f:
            eval_data = json.load(f)

        eval_strings = [
            str(sample[eval_column])
            for split in eval_data.values()
            for sample in split
            for eval_column in eval_columns
            if sample[eval_column] is not None
        ]

        # Track indices to remove
        process_pair = partial(
            fuzz_string_pair,
            values2=eval_strings,
            similarity_threshold=threshold,
        )

        with Pool(n_processes) as pool:
            matches = list(
                tqdm(
                    pool.imap(process_pair, dataset_strings, chunksize=100),
                    total=len(dataset_strings),
                    desc=f"Decontaminating against {eval_json}",
                )
            )

        # Find indices where matches were found
        for i, match_list in enumerate(matches):
            if any(score >= threshold for _, _, score in match_list):
                indices_to_remove.add(i)

    keep_mask = [i for i in range(len(dataset)) if i not in indices_to_remove]
    clean_dataset = dataset.select(keep_mask)

    print(f"Removed {len(indices_to_remove)} contaminated rows")
    print(f"Original size: {len(dataset)}, New size: {len(clean_dataset)}")

    return clean_dataset


@click.command()
@click.option(
    "--repo_id",
    type=str,
    help="The repo id to load the dataset from.",
    default="mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong",
)
@click.option(
    "--push_repo_id",
    type=str,
    default=None,
)
@click.option(
    "--eval_json_list",
    type=str,
    multiple=True,
    default=["misc/m1_eval_data.json"],
)
def main(repo_id=None, push_repo_id=None, eval_json_list=["misc/m1_eval_data.json"]):
    split = "train"
    print(f"Decontaminating {repo_id} on split {split}")
    print(f"Pushing to {push_repo_id}")
    print(f"Using eval jsons: {eval_json_list}")

    if push_repo_id is None:
        push_repo_id = f"{repo_id}-decon_eval"

    thinking_dataset = datasets.load_dataset(repo_id, split=split)

    decontaminated_dataset = decontaminate(
        thinking_dataset, eval_json_list=eval_json_list, column="prompt"
    )

    decontaminated_dataset.push_to_hub(push_repo_id)
    print(f"Decontaminated dataset: {decontaminated_dataset}")
    print(f"Pushed to {push_repo_id}")


if __name__ == "__main__":
    main()
