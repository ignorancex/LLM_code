from datasets import load_dataset
from typing import Dict
import torch


DATASETS = {
    "c4": ("allenai/c4", "text", "realnewslike", "validation", True),
    "dolly": ("databricks/databricks-dolly-15k", "instruction", None, "train", False),
    "AdvBench": ("walledai/AdvBench", "prompt", None, "train", False),
}


def get_prompts(
    tokenizer,
    n_samples=100,
    batch_size: int = 16,
    dataset: str = "c4",
    selected_split: str = None,
) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_name, text_column, name, split, truncate = DATASETS[dataset]

    if selected_split is not None:
        split = selected_split

    dataset = load_dataset(dataset_name, name=name, split=split, streaming=True)

    max_length = 50 + 200
    min_length = 50 + 200

    def filter_length(example):
        return (
            len(
                tokenizer(example[text_column], truncation=True, max_length=max_length)[
                    "input_ids"
                ]
            )
            >= min_length
        )

    def encode(examples):
        trunc_tokens = tokenizer(
            examples[text_column], truncation=True, padding=True, return_tensors="pt"
        ).to(device)
        examples["text"] = tokenizer.batch_decode(
            trunc_tokens["input_ids"], skip_special_tokens=True
        )
        prompt = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=50,
            return_tensors="pt",
        ).to(device)
        examples["prompt_text"] = tokenizer.batch_decode(
            prompt["input_ids"], skip_special_tokens=True
        )
        examples["input_ids"] = prompt["input_ids"]
        examples["attention_mask"] = prompt["attention_mask"]
        examples["text_completion"] = tokenizer.batch_decode(
            trunc_tokens["input_ids"][:, 50:], skip_special_tokens=True
        )
        return examples

    if truncate:
        dataset = dataset.filter(filter_length)

    dataset = dataset.map(encode, batched=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    prompts = []
    human_text = []
    prompt_text = []
    full_human_text = []
    for batch in dataloader:
        if len(human_text) >= n_samples:
            break
        if type(batch["input_ids"]) == list:
            batch["input_ids"] = torch.stack(batch["input_ids"], dim=1).to(device)
        if type(batch["attention_mask"]) == list:
            batch["attention_mask"] = torch.stack(batch["attention_mask"], dim=1).to(
                device
            )
        prompts.append(batch)
        human_text.extend(batch["text_completion"])
        prompt_text.extend(batch["prompt_text"])
        full_human_text.extend(batch["text"])
    human_text = human_text[:n_samples]
    prompt_text = prompt_text[:n_samples]
    full_human_text = full_human_text[:n_samples]
    return {
        "prompts": prompts,
        "human_text": human_text,
        "prompt_text": prompt_text,
        "full_human_text": full_human_text,
    }
