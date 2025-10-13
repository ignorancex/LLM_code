import os
import sys
import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "audioldm_eval"))
from audioldm_eval import EvaluationHelper


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics.")
    parser.add_argument("--target_dir", type=str, default=None, help="path to target.")
    parser.add_argument("--source_dir", type=str, default=None, help="path to generated results.")
    args = parser.parse_args()

    device = torch.device(f"cuda:{0}")

    evaluator = EvaluationHelper(16000, device)

    # Perform evaluation, result will be print out and saved as json
    metrics = evaluator.main(
        args.source_dir,
        args.target_dir,
    )

    print(f"FAD = {metrics}")


if __name__ == "__main__":
    main()
