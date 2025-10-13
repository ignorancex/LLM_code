import argparse
from watermark_stealing.config.meta_config import get_pydantic_models_from_path
from src.ngram_counter import load_ngram_counter_from_cfg

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Generate distribution of n-gram statistics")
    parser.add_argument("--n_gram", type=int, required=True, help="Size of the n-gram")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--ordered", type=str, default="N", help="Y/N")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ordered = True if args.ordered == "Y" else False
    cfg = get_pydantic_models_from_path(args.cfg_path)[0]
    load_ngram_counter_from_cfg(cfg, args.n_gram, ordered=ordered)
