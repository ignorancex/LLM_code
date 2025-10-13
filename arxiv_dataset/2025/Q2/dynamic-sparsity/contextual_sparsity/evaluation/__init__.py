# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from .hooks import (
    CROSS_ENTROPY,
    MEMORY,
    MLP_DENSITY,
    MLP_MEMORY,
    PERPLEXITY,
    Memory,
    Perplexity,
)
from .lm_eval import run_lm_eval
from .perplexity import evaluate_sparse_perplexity
