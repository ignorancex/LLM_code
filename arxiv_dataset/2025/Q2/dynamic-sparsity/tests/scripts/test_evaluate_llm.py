# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os

import pytest
from hydra import compose, initialize

from scripts.run_experiment import parse

HARDWARE_ID = "ci_node"
BASE_CONFIG_PATH = "scripts/config"
if not os.path.exists(BASE_CONFIG_PATH):
    BASE_CONFIG_PATH = os.path.join("..", BASE_CONFIG_PATH)
BASE_CONFIG_PATH = os.path.abspath(BASE_CONFIG_PATH)


@pytest.mark.parametrize("use_simulator", [False, True])
def test_evaluate_llm_perplexity(tmpdir, use_simulator):
    log_dir = os.path.join(tmpdir, "log")
    cache_dir = os.path.join(tmpdir, "cache")
    overrides = [
        "dense_model=opt-350M",
        "experiment=evaluate_llm",
        "evaluation=perplexity",
        "data=wikitext",
        "data.test.take_n_sequences=1",
        "data.test.sequence_length=5",
        "data.test.prompt_length=0",
        "masking_hooks=glu_pruning",
        "masking_hooks.layers_to_sparsify=all",
        "+masking_hooks.k=128",
        f"hardware.paths.log={log_dir}",
        f"hardware.paths.cache={cache_dir}",
    ]
    if use_simulator:
        overrides += [
            "+hw_simulator=default",
            "cache_hooks=write_only",
        ]

    with initialize(version_base="1.3", config_path="pkg://scripts/config"):
        cfg = compose("config.yaml", overrides=overrides)

    parse(cfg)

    assert os.path.exists("results.csv")
    if use_simulator:
        assert os.path.exists("results_hwsim.csv")


@pytest.mark.parametrize("use_simulator", [False, True])
def test_evaluate_llm_lmeval(tmpdir, use_simulator):
    log_dir = os.path.join(tmpdir, "log")
    cache_dir = os.path.join(tmpdir, "cache")
    overrides = [
        "dense_model=opt-350M",
        "experiment=evaluate_llm",
        "evaluation=arc_easy",
        "evaluation.arc_easy.limit=1",
        "masking_hooks=glu_pruning",
        "masking_hooks.layers_to_sparsify=all",
        "+masking_hooks.k=128",
        f"hardware.paths.log={log_dir}",
        f"hardware.paths.cache={cache_dir}",
    ]
    if use_simulator:
        overrides += [
            "+hw_simulator=default",
            "cache_hooks=write_only",
            "hw_simulator.sequence_length=5",
            "hw_simulator.prompt_length=0",
        ]

    with initialize(version_base="1.3", config_path="pkg://scripts/config"):
        cfg = compose("config.yaml", overrides=overrides)

    parse(cfg)

    assert os.path.exists("results.csv")
    if use_simulator:
        assert os.path.exists("results_hwsim.csv")
