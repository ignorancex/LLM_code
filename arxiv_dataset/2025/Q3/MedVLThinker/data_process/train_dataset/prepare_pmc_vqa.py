import dotenv

dotenv.load_dotenv(override=True)
import json
import os
import string
from pathlib import Path

import click
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from datasets import Image as ImageData
from PIL import Image
from resize_image import resize_image_to_qwen2_5_vl


def generate_data(data_csv_path, img_dir):
    data_df = pd.read_csv(data_csv_path, dtype=str, keep_default_na=False)

    img_dir = Path(img_dir)
    custom_whitespace = string.whitespace + "'\"\xa0\u200a"

    num_errors = 0
    num_success = 0
    dataset_index = 0

    for idx, row in data_df.iterrows():
        image_path = row["Figure_path"]
        image_path = img_dir / image_path
        images = [str(image_path)]
        # image = Image.open(image_path).convert("RGB")

        answer = row["Answer"]
        answer_label = row["Answer_label"]

        answer_label = answer_label.strip(custom_whitespace)
        answer = answer.strip(custom_whitespace)

        choice_a = row["Choice A"].strip(custom_whitespace)
        choice_b = row["Choice B"].strip(custom_whitespace)
        choice_c = row["Choice C"].strip(custom_whitespace)
        choice_d = row["Choice D"].strip(custom_whitespace)
        options = {"A": choice_a, "B": choice_b, "C": choice_c, "D": choice_d}

        # convert ' B:Focal uptake pattern ' to 'Focal uptake pattern'
        try:
            rewrite_choices(custom_whitespace, row, options)
        except ValueError as e:
            num_errors += 1
            continue

        if answer_label not in options:
            num_errors += 1
            continue
            raise ValueError(
                f"Answer label [{answer_label}] not found in choices: [{options}]\n\noriginal row: [{row}]"
            )
        if options[answer_label] != answer:
            num_errors += 1
            continue
            raise ValueError(
                f"Answer [{answer}] does not match the choice for label [{answer_label}]: [{options[answer_label]}]\n\noriginal row: [{row}]"
            )

        question = row["Question"]
        question = question.strip(custom_whitespace)

        options = json.dumps(options, ensure_ascii=False)

        dataset_name = "pmc_vqa"
        hash = None
        yield {
            "images": images,
            "question": question,
            "options": options,
            "answer_label": answer_label,
            "answer": answer,
            "dataset_name": dataset_name,
            "hash": hash,
            "dataset_index": dataset_index,
        }

        dataset_index += 1
        num_success += 1

    print(f"Total errors: {num_errors}, Total successes: {num_success}")


def rewrite_choices(custom_whitespace, row, choices):
    for label, choice in choices.items():
        choice_label, choice = choice.split(":", 1)

        choice_label = choice_label.strip(custom_whitespace)
        choice = choice.strip(custom_whitespace)
        if choice_label != label:
            raise ValueError(
                f"Choice label [{choice_label}] does not match expected label [{label}] in choice: [{choice}]\n\noriginal row: [{row}]"
            )
        choices[label] = choice


def test():
    data_csv_path = "data/raw/PMC-VQA/train.csv"
    img_dir = "data/PMC-VQA/images/images"

    cnt = 0
    data = []
    for item in generate_data(data_csv_path, img_dir):
        if cnt == 5:
            break
        cnt += 1
        data.append(item)

        print(f"Item {cnt}: {item}")

    dummy_dataset = Dataset.from_generator(
        lambda: (item for item in data),
        features=Features(
            {
                "images": Sequence(ImageData()),
                "question": Value("string"),
                "options": Value("string"),
                "answer_label": Value("string"),
                "answer": Value("string"),
                "dataset_name": Value("string"),
                "hash": Value("string", id=None),
                "dataset_index": Value("int32"),
            }
        ),
    )
    dummy_dataset.push_to_hub("med-vlrm/pmc_vqa")
    breakpoint()


def main():
    data_csv_path = "data/raw/PMC-VQA/train.csv"
    img_dir = "data/PMC-VQA/images/images"
    num_proc = 32
    hf_hub_repo = "med-vlrm/med-vlm-pmc_vqa"

    features = Features(
        {
            "images": Sequence(ImageData()),  # This will load images in parallel
            "question": Value("string"),
            "options": Value("string"),
            "answer_label": Value("string"),
            "answer": Value("string"),
            "dataset_name": Value("string"),
            "hash": Value("string", id=None),
            "dataset_index": Value("int32"),
        }
    )

    train_dataset = Dataset.from_generator(
        generate_data,
        features=features,
        gen_kwargs={
            "data_csv_path": data_csv_path,
            "img_dir": img_dir,
        },
        num_proc=num_proc,
    )

    # resize the images
    train_dataset = train_dataset.map(
        resize_image_to_qwen2_5_vl,
        num_proc=num_proc,
        desc="Resizing images to Qwen2-VL-2B requirements",
    )

    train_dataset.push_to_hub(hf_hub_repo)
    print(f"Dataset pushed to {hf_hub_repo} with {len(train_dataset)} items.")


if __name__ == "__main__":
    main()
