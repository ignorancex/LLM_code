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


def generate_data(data_csv_path, img_dir):
    data_df = pd.read_csv(data_csv_path, dtype=str, keep_default_na=False)

    img_dir = Path(img_dir)
    custom_whitespace = string.whitespace + "'\"\xa0\u200a"

    num_errors = 0
    num_success = 0

    for idx, row in data_df.iterrows():
        image_path = row["Figure_path"]
        image_path = img_dir / image_path
        # image = Image.open(image_path).convert("RGB")

        answer = row["Answer"]
        answer_label = row["Answer_label"]

        answer_label = answer_label.strip(custom_whitespace)
        answer = answer.strip(custom_whitespace)

        choice_a = row["Choice A"].strip(custom_whitespace)
        choice_b = row["Choice B"].strip(custom_whitespace)
        choice_c = row["Choice C"].strip(custom_whitespace)
        choice_d = row["Choice D"].strip(custom_whitespace)
        choices = {"A": choice_a, "B": choice_b, "C": choice_c, "D": choice_d}

        # convert ' B:Focal uptake pattern ' to 'Focal uptake pattern'
        try:
            rewrite_choices(custom_whitespace, row, choices)
        except ValueError as e:
            num_errors += 1
            continue

        if answer_label not in choices:
            num_errors += 1
            continue
            raise ValueError(
                f"Answer label [{answer_label}] not found in choices: [{choices}]\n\noriginal row: [{row}]"
            )
        if choices[answer_label] != answer:
            num_errors += 1
            continue
            raise ValueError(
                f"Answer [{answer}] does not match the choice for label [{answer_label}]: [{choices[answer_label]}]\n\noriginal row: [{row}]"
            )

        question = row["Question"]
        question = question.strip(custom_whitespace)

        choices_str = "\n\n".join(
            [f"{label}: {choice}" for label, choice in choices.items()]
        )
        problem = f"{question}\n\n{choices_str}"

        num_success += 1
        yield {
            # "images": [image],
            "image": str(image_path),
            "problem": "<image>" + problem,
            "answer": answer_label,
        }
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
    for item in generate_data(data_csv_path, img_dir):
        if cnt == 5:
            break
        cnt += 1

        print(item["problem"])
        print("Answer:", item["answer"])
        print("Image size:", item["images"][0].size)

        breakpoint()


def iter_error():
    data_csv_path = "data/raw/PMC-VQA/train.csv"
    img_dir = "data/PMC-VQA/images/images"
    for _ in generate_data(data_csv_path, img_dir):
        pass


def main():
    data_csv_path = "data/raw/PMC-VQA/train.csv"
    img_dir = "data/PMC-VQA/images/images"

    features = Features(
        {
            "image": ImageData(),  # This will load images in parallel
            "problem": Value("string"),
            "answer": Value("string"),
        }
    )

    train_dataset = Dataset.from_generator(
        generate_data,
        gen_kwargs={
            "data_csv_path": data_csv_path,
            "img_dir": img_dir,
        },
        features=features,
        num_proc=32,
    )
    dataset = DatasetDict({"train": train_dataset})  # type: ignore
    dataset.push_to_hub("med-vlrm/PMC-VQA-EasyR1")


if __name__ == "__main__":
    # iter_error()
    main()
