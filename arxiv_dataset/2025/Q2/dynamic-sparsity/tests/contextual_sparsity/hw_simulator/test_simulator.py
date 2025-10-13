# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from contextual_sparsity.hw_simulator.cache import BeladyCache
from contextual_sparsity.hw_simulator.constants import MODEL_ID_TO_DIMS
from contextual_sparsity.utils.layer_names import FC_DOWN, get_layer_id


def test_dynamic_token_generation(tmpdir):
    device = f'{"cuda:0" if torch.cuda.is_available() else "cpu"}'

    model_id = "opt-350M"
    model_dims = MODEL_ID_TO_DIMS[model_id]
    keep = 0.5
    cache_strategy = "lru"
    with initialize(version_base="1.3", config_path="pkg://scripts/config"):
        overrides = [
            "experiment=evaluate_llm",
            "+hw_simulator=default",
            "data=dummy",
            "masking_hooks=glu_pruning",
            f"masking_hooks.keep={keep}",
            f"dense_model={model_id}",
            "hw_simulator.sequence_length=2",
            "hw_simulator.prompt_length=1",
            f"hw_simulator.device={device}",
            f"hw_simulator.cache_strategy={cache_strategy}",
        ]
        conf = compose("config.yaml", overrides)
        os.chdir(tmpdir)

        dense_model = instantiate(conf.dense_model)
        masking_hooks = instantiate(conf.masking_hooks, dense_model=dense_model)

        hardware = instantiate(conf.hw_simulator, model=dense_model, masking_hooks=masking_hooks)

        layer_key = get_layer_id(model_id=model_id, layer_type=FC_DOWN, layer_name=1)
        mask = torch.zeros(model_dims["intermediate_size"], dtype=torch.bool, device=device)
        k = int(keep * model_dims["intermediate_size"])
        mask[:k] = True

        # for generation of the first token, all necessary weights will be read from Flash, no matter cache strategy
        transfer_footprint = k * model_dims["hidden_size"] * hardware.precision["mlp"]
        first_token_time = transfer_footprint / hardware.flash_io_speed
        hardware.write_to_memory(layer_key=layer_key, cur_mask=mask)

        # first token generation time should be equal to moving selected weights of the selected layer from Flash
        torch.testing.assert_close(first_token_time, hardware.current_token_generation_dynamic)

        hardware.write_to_memory(layer_key=layer_key, cur_mask=mask)
        # For the second token, all weights are in DRAM already
        second_token_time = transfer_footprint / hardware.dram_io_speed
        torch.testing.assert_close(
            first_token_time + second_token_time,
            hardware.current_token_generation_dynamic,
        )

        hardware._reset()
        assert (
            hardware.current_token_generation_dynamic == 0
        ), "token generation time should be 0 after reset"


def test_static_elapsed_time(tmpdir):
    device = f'{"cuda:0" if torch.cuda.is_available() else "cpu"}'

    model_id = "opt-350M"
    with initialize(version_base="1.3", config_path="pkg://scripts/config"):
        overrides = [
            "experiment=evaluate_llm",
            "+hw_simulator=default",
            "data=dummy",
            "masking_hooks=glu_pruning",
            "masking_hooks.keep=1.0",
            f"dense_model={model_id}",
            "hw_simulator.sequence_length=2",
            "hw_simulator.prompt_length=1",
            f"hw_simulator.device={device}",
            "hw_simulator.dram.layers_dynamic=[]",
        ]
        conf = compose("config.yaml", overrides)
        os.chdir(tmpdir)

        dense_model = instantiate(conf.dense_model)
        masking_hooks = instantiate(conf.masking_hooks, dense_model=dense_model)

        hardware = instantiate(conf.hw_simulator, model=dense_model, masking_hooks=masking_hooks)

        # prompt encoding and static token generation should increase corresponding elapsed time and NOT reset (currently it is fixed)
        assert hardware.current_prompt_encoding > 0
        assert hardware.current_token_generation_fixed > 0
        hardware._reset()
        assert hardware.current_prompt_encoding > 0
        assert hardware.current_token_generation_fixed > 0


@pytest.mark.parametrize(
    "seq_mask, expected_cache",
    [
        (
            [
                [0, 0, 1, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            [0, 1, 1, 0, 1],
        ),
        (
            [
                [0, 0, 1, 1, 0],
                [0, 1, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            [0, 1, 0, 1, 1],
        ),
    ],
)
def test_playback_seq_masking(tmpdir, seq_mask, expected_cache):
    device = f'{"cuda:0" if torch.cuda.is_available() else "cpu"}'

    model_id = "opt-350M"
    with initialize(version_base="1.3", config_path="pkg://scripts/config"):
        overrides = [
            "experiment=evaluate_llm",
            "+hw_simulator=default",
            "data=dummy",
            "masking_hooks=glu_pruning",
            "masking_hooks.keep=1.0",
            f"dense_model={model_id}",
            f"hw_simulator.sequence_length={len(seq_mask)}",
            "hw_simulator.prompt_length=1",
            f"hw_simulator.device={device}",
            "hw_simulator.cache_strategy=belady",
        ]
        conf = compose("config.yaml", overrides)
        os.chdir(tmpdir)

        dense_model = instantiate(conf.dense_model)
        masking_hooks = instantiate(conf.masking_hooks, dense_model=dense_model)

        hardware = instantiate(conf.hw_simulator, model=dense_model, masking_hooks=masking_hooks)

        hw_cache = BeladyCache(
            size_per_idx=1, precision=1, max_cache_size=3, max_index=5, device=device
        )

        layer_key = get_layer_id(model_id=model_id, layer_type=FC_DOWN, layer_name=1)

        hardware.seq_mask[layer_key] = torch.tensor(seq_mask, dtype=torch.bool, device=device)
        hardware.caches[layer_key] = hw_cache
        hardware.layer_call_counter[layer_key] = 5
        hardware._counter_forward_calls[layer_key] = 5

        hardware._reset()

        assert all(torch.eq(hw_cache.cache, torch.tensor(expected_cache, device=device))), (
            hw_cache.cache,
            expected_cache,
        )
