# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import pandas as pd
import pytest
import torch

from contextual_sparsity.data.dummy import get_dummy_dataloader
from contextual_sparsity.dense_models import DummyModel
from contextual_sparsity.evaluation import (
    CROSS_ENTROPY,
    Memory,
    Perplexity,
    evaluate_sparse_perplexity,
)
from contextual_sparsity.mask import MaskingHook
from contextual_sparsity.nn import ThresholdMask
from contextual_sparsity.utils.sparsify import sparsify_model

N_FEATURES = 101
SEQUENCE_LENGTH = 50
N_SEQUENCES = 129
BATCH_SIZE = 64


@pytest.mark.parametrize("threshold", [-1, 1, N_FEATURES // 2, N_FEATURES])
def test_evaluation(threshold: float):
    # Define a dummy architecture with two identity transformations
    dense_model = DummyModel()
    dataloader = get_dummy_dataloader(
        n_features=N_FEATURES,
        n_sequences=N_SEQUENCES,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
    )
    dataset = dataloader.dataset

    # Masking hooks
    masking_hooks = [
        MaskingHook(
            masking_func=ThresholdMask(threshold=threshold),
            input_from="layers.0.up",
            mask_rows_of=["layers.0.up"],
            mask_cols_of=["layers.0.down"],
        ),
        MaskingHook(
            masking_func=ThresholdMask(threshold=threshold),
            input_from="layers.1.up",
            mask_rows_of=["layers.1.up"],
            mask_cols_of=["layers.1.down"],
        ),
    ]

    sparse_model = sparsify_model(dense_model=dense_model, masking_hooks=masking_hooks)

    evaluation_hooks = [
        Perplexity(),
        Memory(
            model_id="dummy",
            precision={
                "embedding": 8,
                "lm_head": 8,
                "attention": 4,
                "mlp": 4,
                "activations": 16,
                "kv_cache": 8,
                "predictors": 16,
            },
            sequence_length=2048,
        ),
    ]

    evaluate_sparse_perplexity(
        model=sparse_model, test_data=dataloader, evaluation_hooks=evaluation_hooks
    )

    results = pd.read_csv("results.csv")

    #######################
    # Check Cross-entropy #
    #######################
    true_cross_entropy = torch.cat(
        [sparse_model(**dataset[i]).loss.unsqueeze(0) for i in range(len(dataset))], 0
    ).view(-1)

    measured_cross_entropy = results[
        (results["computed_at"] == ".") & (results["quantity"] == CROSS_ENTROPY)
    ].pivot_table(columns="stat", values="value")
    assert len(measured_cross_entropy) == 1
    measured_cross_entropy = measured_cross_entropy.iloc[0]

    assert np.isclose(
        measured_cross_entropy["mean"], true_cross_entropy.mean(), atol=1e-6
    ), f"{measured_cross_entropy['mean']} != {true_cross_entropy.mean()}"
    # The standard deviation can't be checked properly since the LLM loss function returns one value per batch instead
    # Of one per element. As a result, the standard deviation depends on the batches created by the DataLoader.
