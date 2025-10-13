# https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M

import dotenv

dotenv.load_dotenv(override=True)

import json
from pathlib import Path

import datasets

dataset_path = "open-thoughts/OpenThoughts3-1.2M"
dataset_split = "train"

dataset = datasets.load_dataset(
    dataset_path,
    split=dataset_split,
    streaming=True,
)

sample = next(iter(dataset))

conversations = sample["conversations"]
save_dir = "misc/explore_format_of_openthoughts"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / f"{dataset_split}.json"
with open(save_path, "w") as f:
    json.dump(conversations, f, indent=2, ensure_ascii=False)
print(f"Saved sample conversations to {save_path}")
