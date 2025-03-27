import os, sys
import argparse
import yaml
import torch

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(os.environ["HF_AUTH_TOKEN"])

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to save the merged model"
    )
    parser.add_argument(
        "--lora_merge_cache",
        type=str,
        default="/tmp",
        help="Path to save the merged model",
    )
    parser.add_argument(
        "--copy_tokenizer", action="store_true", help="Copy the tokenizer", default=True
    )
    parser.add_argument(
        "--lazy_unpickle",
        action="store_true",
        help="Lazy unpickle the model",
        default=False,
    )
    parser.add_argument(
        "--low_cpu_memory", action="store_true", help="Use low CPU memory", default=True
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fp:
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_merge(
        merge_config,
        out_path=args.out_path,
        options=MergeOptions(
            lora_merge_cache=args.lora_merge_cache,
            cuda=torch.cuda.is_available(),
            copy_tokenizer=args.copy_tokenizer,
            lazy_unpickle=args.lazy_unpickle,
            low_cpu_memory=args.low_cpu_memory,
            trust_remote_code=True,
        ),
    )
    print("Done!")
