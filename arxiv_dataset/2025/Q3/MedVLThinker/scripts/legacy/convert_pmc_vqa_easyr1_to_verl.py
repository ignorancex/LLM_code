import dotenv

dotenv.load_dotenv(override=True)
import json
import os
import string
from pathlib import Path

import click
import datasets
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from datasets import Image as ImageData
from PIL import Image


@click.command()
@click.option("--data_source", type=str, default="med-vlrm/PMC-VQA-EasyR1")
@click.option("--local_dir", type=Path, default="data/verl/pmc_vqa")
@click.option("--is_local", is_flag=True, default=False, help="Use local data source")
@click.option(
    "--num_proc", type=int, default=16, help="Number of processes to use for mapping"
)
@click.option(
    "--shuffle_seed", type=int, default=42, help="Seed for shuffling the dataset"
)
@click.option(
    "--num_samples", type=int, default=None, help="Number of samples to process"
)
def main(data_source, local_dir, is_local, num_proc, shuffle_seed, num_samples):
    if is_local:
        dataset = datasets.load_from_disk(data_source)
    else:
        dataset = datasets.load_dataset(data_source)

    dataset = dataset["train"]

    dataset = dataset.shuffle(seed=shuffle_seed)

    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    # split train and validation after shuffling
    dataset = dataset.train_test_split(test_size=0.1)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # X-Reasoner's training prompt template.
    # Also see `examples/format_prompt/r1v_format.jinja` in EasyR1
    instruction_following = r"You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags. Here is the question:\n\n"

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            # How interleave multi-modal inputs work in verl: https://github.com/volcengine/verl/blob/2aed8d0a4570166ff107b7af2c035c3a46a78101/verl/utils/dataset/rl_dataset.py#L169-L175
            prompt = instruction_following + problem
            answer = example.pop("answer")
            # only a simple image
            image = example.pop("image")
            images = [image]

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": images,
                "ability": "med_vqa_mcqa",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(
        function=make_map_fn("train"), with_indices=True, num_proc=num_proc
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test"), with_indices=True, num_proc=num_proc
    )

    train_dataset.to_parquet(local_dir / "train.parquet")
    test_dataset.to_parquet(local_dir / "test.parquet")
    print(f"Train dataset saved to {local_dir / 'train.parquet'}")
    print(f"Test dataset saved to {local_dir / 'test.parquet'}")


if __name__ == "__main__":
    main()
