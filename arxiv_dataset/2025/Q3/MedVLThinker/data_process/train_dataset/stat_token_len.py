"""
python data_process/train_dataset/check_pass_rate_dataset.py \
    -d data/local/med-vlm-m23k-qwen2_5_vl_3b-easy_to_hard

python data_process/train_dataset/check_pass_rate_dataset.py \
    -d data/local/med-vlm-pmc_vqa-qwen2_5_vl_3b-easy_to_hard

python data_process/train_dataset/check_pass_rate_dataset.py \
    --is_local False
"""

import dotenv

dotenv.load_dotenv(override=True)

import json
from pathlib import Path
from types import SimpleNamespace

import click
import datasets
import matplotlib.pyplot as plt
from transformers import AutoProcessor


def build_prompt(row, processor):
    messages = build_messages(row)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(prompt[0])


def build_messages(row):
    options = row["options"]
    options = json.loads(options)

    question = row["question"]

    prompt_lines = [
        r"You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags. Here is the question:\n\n"
    ]
    qtxt = f"Question: {question}\nOptions:"
    for letter, option in options.items():
        qtxt += f"\n\n{letter}. {option}"

    qtxt = "\n".join(prompt_lines) + "\n\n" + qtxt

    images = row.get("images", None)
    if images is None:
        images = []
    else:
        images = [{"type": "image", "image": img} for img in images]

    return [
        {
            "role": "user",
            "content": [
                *images,
                {"type": "text", "text": qtxt},
            ],
        }
    ]


def stat_token_len(row, stats_column_name, tokenzier):
    text = row.get(stats_column_name, None)
    if text is None:
        return 0
    return len(tokenzier(text, add_special_tokens=True)["input_ids"])


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
    default="UCSC-VLAA/MedVLThinker-m23k-tokenized",
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
    "--stats_column_name",
    type=str,
    default="reasoning",
    help="Column name for statistics.",
)
@click.option(
    "--output_dir", type=str, default="misc/stat_token_len", help="Output directory."
)
def main(**kargs):
    args = SimpleNamespace(**kargs)
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

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    tokenizer = processor.tokenizer

    # Check build_prompt
    stats_column_name = args.stats_column_name
    _ = build_prompt(dataset[0], processor)
    _ = stat_token_len(dataset[0], stats_column_name, tokenizer)

    keep_in_memory = True
    dataset = dataset.map(
        lambda row: {
            "prompt_length": build_prompt(row, processor),
            "stats_token_len": stat_token_len(row, stats_column_name, tokenizer),
        },
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Building prompts and messages",
        keep_in_memory=keep_in_memory,
    )

    df = dataset.to_pandas()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = local_data_dir.stem

    # plot sorted prompt length
    sorted_prompt_length = df["prompt_length"].sort_values().reset_index(drop=True)

    plot_fig(output_dir, file_name, sorted_prompt_length, "prompt_length")

    sorted_stats_token_len = df["stats_token_len"].sort_values().reset_index(drop=True)
    plot_fig(output_dir, file_name, sorted_stats_token_len, stats_column_name)

    both_sorted = (
        (df["prompt_length"] + df["stats_token_len"])
        .sort_values()
        .reset_index(drop=True)
    )
    plot_fig(output_dir, file_name, both_sorted, f"prompt+{stats_column_name}")


def plot_fig(output_dir, file_name, sorted_prompt_length, name):
    fig, ax = plt.subplots()
    ax.plot(sorted_prompt_length)
    ax.set_title(f"Sorted {name} Length: {file_name}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Length")

    prompt_length_plot_path = output_dir / f"{file_name}-{name}_length.png"
    fig.savefig(prompt_length_plot_path, bbox_inches="tight", dpi=300)
    print(f"Saved {name} length plot to {prompt_length_plot_path}")


if __name__ == "__main__":
    main()
