"""Main script for training the FontClassifier model."""

import warnings

import torch

from truetype_vs_postscript_transformer.scripts.classification.weight.weight import (
    weight_classifier,
)

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    weight_classifier("truetype_segment")
