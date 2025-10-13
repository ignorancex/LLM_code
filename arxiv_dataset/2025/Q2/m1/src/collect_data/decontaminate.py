"""
Copied from https://github.com/open-thoughts/open-thoughts/blob/main/open_thoughts/decontaminate.py
"""

import multiprocessing as mp
from functools import partial
from multiprocessing import Pool

from datasets import Dataset, concatenate_datasets, load_dataset
from deduplicate import fuzz_string_pair
from tqdm import tqdm

EVALUATION_DATASETS = {
    # "HuggingFaceH4/MATH-500": {
    #     "eval_columns": ["problem"],
    #     "eval_splits": ["test"],
    # },
    # "Maxwell-Jia/AIME_2024": {
    #     "eval_columns": ["Problem"],
    #     "eval_splits": ["train"],
    # },
    # "AI-MO/aimo-validation-amc": {
    #     "eval_columns": ["problem"],
    #     "eval_splits": ["train"],
    # },
    # "livecodebench/code_generation_lite": {
    #     "eval_columns": ["question_content"],
    #     "eval_splits": ["test"],
    # },
    "Idavidrein/gpqa": {
        "eval_columns": ["Question"],
        "eval_splits": ["train"],
        "eval_subset": "gpqa_diamond",
    },
    # new
    "openlifescienceai/medmcqa": {
        "eval_columns": ["question"],
        "eval_splits": ["validation"],
    },
    "GBaker/MedQA-USMLE-4-options": {
        "eval_columns": ["question"],
        "eval_splits": ["test"],
    },
    "mmqm/pubmedqa_custom_test": {
        "eval_columns": ["prompt"],
        "eval_splits": ["train"],
    },
    # mmlu
    "cais/mmlu": {
        "eval_columns": ["question"],
        "eval_splits": ["test"],
        "eval_subset": [
            "clinical_knowledge",
            "college_biology",
            "college_medicine",
            "medical_genetics",
            "professional_medicine",
        ],
    },
    # mmlu pro
    "TIGER-Lab/MMLU-Pro": {
        "eval_columns": ["question"],
        "eval_splits": ["test"],
    },
}


def decontaminate(
    dataset: Dataset, column="question", evals=EVALUATION_DATASETS, threshold=95.0
) -> Dataset:
    """Remove rows from dataset that have similar strings in eval_datasets based on fuzzy matching."""
    n_processes = mp.cpu_count()

    # Get values from input dataset
    dataset_strings = [str(x) for x in dataset[column] if x is not None]
    indices_to_remove = set()

    for eval_name, eval_info in evals.items():
        eval_splits = eval_info["eval_splits"]
        eval_columns = eval_info["eval_columns"]
        eval_subset = eval_info.get("eval_subset", None)
        if eval_subset is not None:
            if isinstance(eval_subset, str):
                ds = load_dataset(
                    eval_name, eval_subset, split=eval_splits, trust_remote_code=True
                )
            elif isinstance(eval_subset, list):
                ds_list = []
                for _eval_subset in eval_subset:
                    _ds = load_dataset(
                        eval_name,
                        _eval_subset,
                        split=eval_splits,
                        trust_remote_code=True,
                    )
                    ds_list.extend(_ds)
                ds = concatenate_datasets(ds_list)
        else:
            ds = load_dataset(eval_name, split=eval_splits, trust_remote_code=True)

        # for each split, column, and value
        eval_strings = [
            str(x)
            for split in ds
            for column in eval_columns
            for x in split[column]
            if x is not None
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
                    desc=f"Decontaminating against {eval_name}",
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
