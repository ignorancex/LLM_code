# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from contextual_sparsity.data.data_processing import move_dict_to_device
from contextual_sparsity.data.dummy import get_dummy_dataloader
from contextual_sparsity.dense_models import DummyModel
from contextual_sparsity.utils.sparsify import build_sparse_model

N_SEQUENCES = 2
SEQUENCE_LENGTH = 100
PROMPT_LENGTH = 20
KEEP = 0.5


@pytest.mark.parametrize("masking_hook", ["glu_pruning", "dip_free_params"])
def test_masking_hooks(tmpdir, masking_hook):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    overrides = [
        "experiment=evaluate_llm",
        "dense_model=dummy",
        f"masking_hooks={masking_hook}",
        "data=wikitext",
        f"data.test.sequence_length={SEQUENCE_LENGTH}",
        f"data.test.prompt_length={PROMPT_LENGTH}",
        "+hw_simulator=default",
        "hw_simulator.cache_strategy=lfu",
        "cache_hooks=write_only",
        f"hardware.device={device}",
        f"hardware.paths.log={tmpdir}",
    ]
    if masking_hook == "glu_pruning":
        overrides.append(f"masking_hooks.keep={KEEP}")
    if masking_hook == "dip_free_params":
        overrides.append(f"masking_hooks.up_keep={KEEP}")
        overrides.append(f"masking_hooks.down_keep={KEEP}")

    with initialize(version_base="1.3", config_path="pkg://scripts/config"):
        conf = compose("config.yaml", overrides)
        os.chdir(
            tmpdir
        )  # Calling compose to set Hydra config does not have the same side effects as @hydra.main.

        # Initialize model with hooks and simulator
        model = DummyModel(device=device)
        dataloader = get_dummy_dataloader(
            sequence_length=SEQUENCE_LENGTH - PROMPT_LENGTH,
            n_sequences=N_SEQUENCES,
            batch_size=1,
        )
        masking_hooks = instantiate(conf.masking_hooks, dense_model=model)
        hardware = instantiate(conf.hw_simulator, model=model, masking_hooks=masking_hooks)
        masking_hooks = instantiate(
            conf.cache_hooks, masking_hooks=masking_hooks, hw_simulator=hardware
        )
        model = build_sparse_model(
            masking_hooks=masking_hooks,
            dense_model=model,
        )
        model.eval()

        # Smoke test inference with sample data
        for hook in masking_hooks:
            hook.set_sparse()

        hardware.reset_hook.set_active()
        for batch in dataloader:
            batch = move_dict_to_device(batch, device=device)
            model(**batch)
