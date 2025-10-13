from __future__ import annotations

import os
import gc
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint
from jax.experimental import mesh_utils
import sentencepiece as spm
import penzai
from penzai import pz
from penzai.toolshed import token_visualization, sharding_util
import pandas as pd
import pyarrow as pa
import subprocess
from penzai.models.transformer import variants
import tensorstore as ts
import queue
import tensorstore as ts
import numpy as np

gemma_path = "" # Where are the gemma model weights stored? e.g. "/net/projects2/interp/gemma2"
def load_gemma(precise_floats=True,gemma_dir=""):
    ckpt_path = os.path.join(gemma_dir, 'gemma2-2b')

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    metadata = checkpointer.metadata(ckpt_path)
    restore_args = orbax.checkpoint.checkpoint_utils.construct_restore_args(metadata)

    flax_params_dict = checkpointer.restore(ckpt_path, restore_args=restore_args)
    model = variants.gemma.gemma_from_pretrained_checkpoint(
        flax_params_dict,
        upcast_activations_to_float32=precise_floats
    )
    model = (
        pz.select(model)
        .at_instances_of(
            pz.nn.ApplyCausalAttentionMask
            | pz.nn.ApplyCausalSlidingWindowAttentionMask
        )
        .apply(lambda old: pz.nn.ApplyExplicitAttentionMask(
            mask_input_name="attn_mask",
            masked_out_value=old.masked_out_value,
        ))
    )

    unembed_matrix = flax_params_dict['transformer/embedder']['input_embedding']
    del flax_params_dict
    gc.collect()

    return model, unembed_matrix.T

model, unembed_matrix = load_gemma()
rest, main_model = (
    pz.select(model)
    .at(lambda m: list(m.body.sublayers)[20:],multiple=True)
    .partition()
)

def get_logits(model,inputs):
    wrapped_acts = pz.nx.wrap(inputs).tag("batch","embedding")
    return model(wrapped_acts)
