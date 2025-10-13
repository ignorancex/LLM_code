import dotenv

dotenv.load_dotenv()


import json
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import click
import datasets
from transformers import AutoTokenizer

INSTRUCTION_PROMPT = r"You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags."


def tokenize_sample(sample, tokenizer):
    question = sample["question"]
    raw_options = sample["options"]
    options = json.loads(raw_options)

    prompt = f"Question: {question}\n\nOptions:"
    for letter, option in options.items():
        prompt += f"\n\n{letter}. {option}"
    prompt = INSTRUCTION_PROMPT + "\n\n" + prompt

    answer_label = sample["answer_label"]
    answer = sample["answer"]
    reasoning = sample["reasoning"]

    response = f"<think> {reasoning.strip()} </think>\n<answer> {answer_label.strip()} </answer>"

    message = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
    )
    return {"text": text}


@click.command()
@click.option(
    "--local_data_dir",
    "-d",
    type=str,
    default="data/local/med-vlm-m23k-qwen2_5_vl_3b-easy_to_hard",
    help="Local data directory containing the dataset.",
)
@click.option("--is_local", type=bool, default=False, help="Use local data source.")
# remote
@click.option(
    "--dataset_path",
    type=str,
    default="med-vlrm/med-vlm-m23k",
    help="Remote dataset path if not using local data.",
)
@click.option(
    "--dataset_subset", type=str, default=None, help="Subset of the dataset to use."
)
@click.option(
    "--dataset_split", type=str, default="train", help="Split of the dataset to use."
)
@click.option(
    "--num_proc",
    "-n",
    type=int,
    default=16,
    help="Number of processes to use for mapping.",
)
@click.option(
    "--tokenizer_name",
    type=str,
    default="Qwen/Qwen2.5-VL-7B-Instruct",
    help="Tokenizer name.",
)
@click.option(
    "--hf_repo_url",
    type=str,
    default="med-vlrm/med-vlm-m23k-tokenized",
    help="Hugging Face repo URL.",
)
@click.option(
    "--keep_in_memory",
    type=bool,
    default=True,
    help="Whether to keep the dataset in memory after tokenization.",
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    print(f"Arguments: {args}")

    is_local = args.is_local
    local_data_dir = args.local_data_dir
    num_proc = args.num_proc

    if is_local:
        local_data_dir = Path(local_data_dir)
        dataset = datasets.load_from_disk(local_data_dir)
        print(f"Loaded dataset from {local_data_dir}")
    else:
        dataset_path = args.dataset_path
        dataset_subset = args.dataset_subset
        dataset_split = args.dataset_split
        dataset = datasets.load_dataset(
            dataset_path,
            subset=dataset_subset,
            split=dataset_split,
        )
        print(f"Loaded dataset from {dataset_path}")
        local_data_dir = Path(dataset_path)

    tokenizer_name = args.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    keep_in_memory = args.keep_in_memory
    dataset = dataset.map(
        partial(tokenize_sample, tokenizer=tokenizer),
        num_proc=num_proc,
        desc="Tokenizing dataset",
        keep_in_memory=keep_in_memory,
    )

    hf_repo_url = args.hf_repo_url
    dataset.push_to_hub(repo_id=hf_repo_url)
    print(f"Pushed tokenized dataset to {hf_repo_url}")


if __name__ == "__main__":
    main()
