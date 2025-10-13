import os
from typing import Any, Optional
import torch

import sae_bench.evals.core.main as core
import sae_bench.evals.absorption.main as absorption
import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.custom_saes.custom_sae_config as custom_sae_config
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex
import sae_bench.custom_saes.run_all_evals_custom_saes as run_all_evals_custom_saes


import json
import jax
import jax.numpy as jnp
import equinox as eqx
import treescope
#treescope.basic_interactive_setup()
#import tensorflow as tf
import sys
import numpy as np


from get_trained import get_trained

output_folders = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    "unlearning": "eval_results/unlearning",
}

# Select your eval types here.
eval_types = [
    "absorption",
    # "autointerp",
    #"core",
    #"scr",
    #"tpp",
    #"sparse_probing",
    # "unlearning",
]


if "autointerp" in eval_types:
    raise ValueError("autointerp must be ran using a python script")

device = general_utils.setup_environment()

model_name = "gemma-2-2b"
llm_batch_size = 512
torch_dtype = torch.float32

# Currently all evals take str_dtype instead of torch_dtype. We did this for serialization purposes, but it was probably a mistake.
# For now we will just use the str_dtype. TODO: Fix this
str_dtype = torch_dtype.__str__().split(".")[-1]


# If evaluating multiple SAEs on the same layer, set save_activations to True
# This will require at least 100GB of disk space
save_activations = True


sae = get_trained("MOE", "32k_64", 20, torch_dtype, device, top_only=True)

sae.cfg.architecture = sae.cfg.architecture = "32k_64"
sae.cfg.training_tokens = sae.cfg.training_tokens = 2_000_000_000

trainer_markers = {
    "our_sae": "o",
}

trainer_colors = {
    "our_sae": "blue",
}

# We do a subset of the sparse probing datasets here for shorter runtime
# TODO: Add a verbose flag
_ = absorption.run_eval(
    absorption.AbsorptionEvalConfig(
        model_name=model_name,
        random_seed=42,
        llm_batch_size=10,
        llm_dtype=str_dtype,
    ),
    [("32k_64", sae)],
    device,
    "eval_results/absorption"
)
