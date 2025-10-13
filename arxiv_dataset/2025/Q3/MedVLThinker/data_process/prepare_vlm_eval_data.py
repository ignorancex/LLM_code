"""
Reformat each row as:
- "images": List[Image]
- "question": str
- "options": Dict[str, str]
- "answer_label": str
- "answer": str
"""

import dotenv

dotenv.load_dotenv(override=True)
import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import datasets
import tqdm
from extract_options import extract_options


def get_str_hash(input_string: str) -> str:
    return sha256(input_string.encode()).hexdigest()[:16]


def process_row_pmc_vqa(row):
    """
    Process a single row from the PMC-VQA dataset.

    Question: What kind of images are displayed on the left and right in the given figure?
    The choices are:
    (A) CT and MRI
    (B) MRI and CT
    (C) X-ray and MRI
    (D) MRI and ultrasound
    """
    images = [row["image"]]
    question = row["input"]

    # remove leading "Question: " from the question
    question = question.replace("Question: ", "", 1).strip()
    # remove "\nThe choices are:" and all the following content
    question = question.split("\nThe choices are:")[0].strip()

    options = {k: row[k] for k in ["A", "B", "C", "D"]}
    answer_label = row["label"]
    answer = options[answer_label]

    return {
        "images": images,
        "question": question,
        "options": json.dumps(options, ensure_ascii=False),
        "answer_label": answer_label,
        "answer": answer,
    }


def get_formated_pmc_vqa(num_proc=32):
    path = "AdaptLLM/biomed-VQA-benchmark"
    name = "PMC-VQA"
    split = "test"

    dataset = datasets.load_dataset(path, name, split=split)
    # dataset = dataset.cast_column("image", datasets.Image(decode=False))
    print(f"Loaded {len(dataset)} samples from {path}/{name} ({split})")

    dataset = dataset.map(
        process_row_pmc_vqa,
        remove_columns=dataset.column_names,
        desc=f"Processing {name} dataset",
        keep_in_memory=True,
        num_proc=num_proc,
    )
    dataset = dataset.cast_column("images", datasets.Sequence(datasets.Image()))
    return dataset


# others
def get_formated_pathvqa_closed():
    path = "AdaptLLM/biomed-VQA-benchmark"
    name = "PathVQA"
    split = "test"

    dataset = datasets.load_dataset(path, name, split=split)
    # dataset = dataset.cast_column("image", datasets.Image(decode=False))
    print(f"Loaded {len(dataset)} samples from {path}/{name} ({split})")

    dataset = dataset.filter(lambda x: x["answer_type"] == "CLOSED")
    dataset = format_dataset(dataset, num_workers=32)

    return dataset


def get_formated_slake_closed():
    path = "AdaptLLM/biomed-VQA-benchmark"
    name = "SLAKE"
    split = "test"
    dataset = datasets.load_dataset(path, name, split=split)
    # dataset = dataset.cast_column("image", datasets.Image(decode=False))
    print(f"Loaded {len(dataset)} samples from {path}/{name} ({split})")

    dataset = dataset.filter(lambda x: x["answer_type"] == "CLOSED")
    dataset = format_dataset(dataset, num_workers=32)

    return dataset


def get_formated_vqa_rad_closed():
    path = "AdaptLLM/biomed-VQA-benchmark"
    name = "VQA_RAD"
    split = "test"
    dataset = datasets.load_dataset(path, name, split=split)
    # dataset = dataset.cast_column("image", datasets.Image(decode=False))
    print(f"Loaded {len(dataset)} samples from {path}/{name} ({split})")

    dataset = dataset.filter(lambda x: x["answer_type"] == "CLOSED")
    dataset = format_dataset(dataset, num_workers=32)

    return dataset


def extract_save_options_for_dataset(ds, cache_dir=None, num_workers=32):
    if cache_dir is None:
        raise ValueError("cache_dir must be specified")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    num_hashes = len(ds)
    existing_hashes = set(i.stem for i in cache_dir.glob("*.json"))
    num_missing_hashes = num_hashes - len(existing_hashes)
    print(f"Number of rows in dataset: {num_hashes}")
    print(f"Found {len(existing_hashes)} existing hashes in cache directory.")
    print(f"Potentially missing hashes: {num_missing_hashes}")

    def _extract_options(row):
        question = row["input"]
        answer = row["label"]
        row_hash = get_str_hash(f"{question}{answer}")
        if row_hash not in existing_hashes:
            options = extract_options(question=question, answer=answer)
            save_path = cache_dir / f"{row_hash}.json"
            save_obj = {
                "question": question,
                "answer": answer,
                "options": options,
            }
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(save_obj, f, ensure_ascii=False)
        return row_hash

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_extract_options, row) for row in ds]
        results = [
            future.result() for future in tqdm.tqdm(futures, desc="Extracting options")
        ]
        existing_hashes.update(results)

    def load_json(row_hash):
        save_path = cache_dir / f"{row_hash}.json"
        if not save_path.exists():
            raise FileNotFoundError(f"Cache file {save_path} does not exist.")
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {save_path}: {e}")
        return {row_hash: data}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_json, row_hash) for row_hash in existing_hashes]
        results = [
            future.result() for future in tqdm.tqdm(futures, desc="Loading options")
        ]
    options_dict = {}
    for result in results:
        options_dict.update(result)
    return options_dict


def format_dataset(dataset, num_workers=32):
    options_dict = extract_save_options_for_dataset(
        dataset, cache_dir="misc/extract_options_cache/"
    )

    def _process_row(row):
        question = row["input"]
        answer = row["label"]
        row_hash = get_str_hash(f"{question}{answer}")
        if row_hash in options_dict:
            options = options_dict[row_hash]["options"]

            options = {chr(ord("A") + i): v for i, v in enumerate(options)}

            # reversely get the answer label
            answer_label = next(
                (k for k, v in options.items() if v.lower() == answer.lower()),
                None,
            )
            if answer_label is None:
                raise ValueError(
                    f"Answer '{answer}' not found in options for hash {row_hash}."
                )
            answer = options[answer_label]
            return {
                "images": [row["image"]],
                "question": question,
                "options": json.dumps(options, ensure_ascii=False),
                "answer_label": answer_label,
                "answer": answer,
                "hash": row_hash,
            }
        else:
            raise ValueError(f"Options for hash {row_hash} not found in cache.")

    dataset = dataset.map(
        _process_row,
        remove_columns=dataset.column_names,
        desc="Reformatting dataset",
        num_proc=num_workers,
        keep_in_memory=True,
    )
    dataset = dataset.cast_column("images", datasets.Sequence(datasets.Image()))
    return dataset


def main():
    datasets_mapping = {
        "pmc_vqa": get_formated_pmc_vqa,
        "pathvqa_closed": get_formated_pathvqa_closed,
        "slake_closed": get_formated_slake_closed,
        "vqa_rad_closed": get_formated_vqa_rad_closed,
    }
    processed_datasets = []
    for name, func in datasets_mapping.items():
        print(f"Processing dataset: {name}")
        dataset = func()
        print(f"Processed {len(dataset)} samples from {name}")

        # add a column of the dataset name
        dataset = dataset.add_column("dataset_name", [name] * len(dataset))

        processed_datasets.append(dataset)

    combined_dataset = datasets.concatenate_datasets(processed_datasets)
    # Now we add `dataset_index` column to identify each sample for visualization
    combined_dataset = combined_dataset.add_column(
        "dataset_index", list(range(len(combined_dataset)))
    )

    print(f"Combined dataset has {len(combined_dataset)} samples.")
    hf_repo = "med-vlrm/med-vlm-eval"
    combined_dataset.push_to_hub(hf_repo, split="test")
    print(f"Combined dataset pushed to {hf_repo}.")


if __name__ == "__main__":
    main()
