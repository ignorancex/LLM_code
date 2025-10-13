"""
We add MedXpertQA MM and MMMU Medical part

## MMMU

MMMU: https://huggingface.co/datasets/MMMU/MMMU
Available subsets in MMMU/MMMU: ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']

accoring to https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/MMMU_Health&Medicine.md
The subsets are:
["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"]


Since MMMU does not provide the answer for `test` split. We only use `validation` and `dev` splits


## MedXpertQA MM

MedXpertQA: https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA/


Download the images

```bash
REPO_TYPE=dataset # model, dataset
LOCAL_DIR=data/raw/medxpertqa_mm/

REPO_URL=TsinghuaC3I/MedXpertQA

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL} images.zip

unzip -d data/raw/medxpertqa_mm data/raw/medxpertqa_mm/images.zip
```

images are in `data/raw/medxpertqa_mm/images/`
"""

import dotenv

dotenv.load_dotenv(override=True)

import ast
import json
import re
from pathlib import Path
from pprint import pformat, pprint

import datasets
from PIL import Image

# dataset_name = "MMMU/MMMU"
# subsets = get_dataset_config_names(dataset_name)
# print(f"Available subsets in {dataset_name}: {subsets}")


def load_mmmu_subset(subset_name=None):
    dataset_name = "MMMU/MMMU"

    # subset_name = "Basic_Medical_Science"  # Change this to the desired subset
    if subset_name is None:
        raise ValueError("Please provide a subset name for the MMMU dataset.")

    dataset_list = []
    for split in ["validation", "dev"]:
        dataset = datasets.load_dataset(dataset_name, subset_name, split=split)
        print(
            f"Loaded {len(dataset)} samples from {dataset_name}/{subset_name} ({split})"
        )
        dataset_list.append(dataset)

    dataset = datasets.concatenate_datasets(dataset_list)

    question_type = set(dataset["question_type"])
    print(f"Question types: {question_type}")

    # only keep question_type == 'multiple-choice
    print("Filtering dataset for multiple-choice questions...")
    num_orginal_samples = len(dataset)
    dataset = dataset.filter(lambda x: x["question_type"] == "multiple-choice")
    num_kept_samples = len(dataset)
    print(f"Kept {num_kept_samples} out of {num_orginal_samples} samples.")

    return dataset


MMMU_SUBSETS = [
    "Basic_Medical_Science",
    "Clinical_Medicine",
    "Diagnostics_and_Laboratory_Medicine",
    "Pharmacy",
    "Public_Health",
]


def load_mmmu_medical():
    dataset_list = []
    for subset_name in MMMU_SUBSETS:
        dataset = load_mmmu_subset(subset_name)
        dataset_list.append(dataset)
    dataset = datasets.concatenate_datasets(dataset_list)

    return dataset


def load_convert_mmmu_medical():
    dataset = load_mmmu_medical()

    converted_dataset = dataset.map(
        process_mmmu_sample,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc="Processing MMMU samples",
    )

    # Convert images to PIL images, otherwise it might be
    # {'bytes': Value(dtype='null', id=None), 'path': Value(dtype='string', id=None)}
    converted_dataset = converted_dataset.cast_column(
        "images", datasets.Sequence(datasets.Image())
    )

    return converted_dataset


# from image_1 to image_7

IMAGE_KEYS = [f"image_{i}" for i in range(1, 8)]


def process_mmmu_sample(row, idx):
    images = []
    for key in IMAGE_KEYS:
        if key in row and row[key] is not None:
            images.append(row[key])

    question = row["question"]
    # remove <image 1> and the spaces before/after it
    question = re.sub(r"\s*<image \d+>\s*", "", question).strip()

    # to parse options
    # https://github.com/MMMU-Benchmark/MMMU/blob/bb0b95a945998d91dfef37e969e07d49d1139438/mmmu/utils/data_utils.py#L58
    # https://github.com/MMMU-Benchmark/MMMU/blob/bb0b95a945998d91dfef37e969e07d49d1139438/mmmu-pro/evaluate.py#L3
    raw_options = row["options"]
    raw_options = ast.literal_eval(str(raw_options))
    options = {chr(i + ord("A")): value for i, value in enumerate(raw_options)}
    option_str = json.dumps(options, ensure_ascii=False)

    answer_label = row["answer"]
    answer = options.get(answer_label, None)
    if answer is None:
        raise ValueError(
            f"Answer label '{answer_label}' not found in options for row {idx}."
        )

    reasoning = None
    dataset_name = "MMMU-medical"
    dataset_index = idx

    hash = None
    misc = {
        "id": row["id"],
        "explanation": row["explanation"],
        "img_type": row["img_type"],
        "question_type": row["question_type"],
        "subfield": row["subfield"],
        "topic_difficulty": row["topic_difficulty"],
    }
    misc = json.dumps(misc, ensure_ascii=False)

    return {
        "dataset_name": dataset_name,
        "dataset_index": dataset_index,
        "question": question,
        "images": images,
        "options": option_str,
        "answer": answer,
        "answer_label": answer_label,
        "reasoning": reasoning,
        "hash": hash,
        "misc": misc,
    }


# medxpertqa mm


def main():
    dataset_list = []
    dataset_list.append(load_convert_mmmu_medical())
    dataset_list.append(load_convert_medxpertqa_mm())

    dataset = datasets.concatenate_datasets(dataset_list)
    print(f"Total samples in the combined dataset: {len(dataset)}")

    hf_repo_v1 = "med-vlrm/med-vlm-eval"
    prev_dataset = datasets.load_dataset(hf_repo_v1, split="test")
    print(
        f"Loaded previous dataset from {hf_repo_v1} with {len(prev_dataset)} samples."
    )

    dataset = datasets.concatenate_datasets([prev_dataset, dataset])

    # after combine, re-assign "dataset_index"
    dataset = dataset.remove_columns(["dataset_index"])
    dataset = dataset.add_column("dataset_index", list(range(len(dataset))))

    hf_repo = "med-vlrm/med-vlm-eval-v2"
    dataset.push_to_hub(hf_repo, split="test")
    print(f"Combined dataset pushed to {hf_repo}.")


def load_convert_medxpertqa_mm():
    dataset = load_medxpertqa_mm()

    converted_dataset = dataset.map(
        process_medxpertqa_mm_sample,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc="Processing MedXpertQA-MM samples",
    )

    # Convert images to PIL images, otherwise it might be
    # {'bytes': Value(dtype='null', id=None), 'path': Value(dtype='string', id=None)}
    converted_dataset = converted_dataset.cast_column(
        "images", datasets.Sequence(datasets.Image())
    )

    return converted_dataset


def load_medxpertqa_mm():
    dataset_name = "TsinghuaC3I/MedXpertQA"
    dataset_subset = "MM"
    # `dev` and `test`
    split = "test"

    dataset = datasets.load_dataset(dataset_name, dataset_subset, split=split)

    return dataset


IMAGE_DIR = "data/raw/medxpertqa_mm/images/"
IMAGE_DIR = Path(IMAGE_DIR)


def process_medxpertqa_mm_sample(row, idx):
    image_paths = row["images"]
    images = []
    for image_path in image_paths:
        image = Image.open(IMAGE_DIR / image_path)
        images.append(image)

    question = row["question"]
    # remove the trailing newline and `\nAnswer Choices: ` part
    question = question.split("\nAnswer Choices: ")[0].strip()

    raw_options = row["options"]
    option_str = json.dumps(raw_options, ensure_ascii=False)

    answer_label = row["label"]
    answer = raw_options.get(answer_label, None)
    if answer is None:
        raise ValueError(
            f"Answer label '{answer_label}' not found in options for row {idx}."
        )

    reasoning = None
    dataset_name = "MedXpertQA-MM"
    dataset_index = idx

    hash = None
    misc = {
        "body_system": row["body_system"],
        "id": row["id"],
        "medical_task": row["medical_task"],
        "question_type": row["question_type"],
    }
    misc = json.dumps(misc, ensure_ascii=False)

    return {
        "dataset_name": dataset_name,
        "dataset_index": dataset_index,
        "question": question,
        "images": images,
        "options": option_str,
        "answer": answer,
        "answer_label": answer_label,
        "reasoning": reasoning,
        "hash": hash,
        "misc": misc,
    }


if __name__ == "__main__":
    main()
