# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os
from typing import Optional

from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from contextual_sparsity.hw_simulator.constants import MODEL_ID_TO_DIMS
from contextual_sparsity.utils.tokenizers import tokenize_for_language_modeling

# A logger for this file
log = logging.getLogger(__name__)


def get_dataloader(
    dataset_id: str,
    tokenizer: PreTrainedTokenizer,
    dataset_path: str,
    model_id: str,
    name: Optional[str],
    split: Optional[str],
    take_n_tokens: Optional[int] = None,
    take_n_sequences: Optional[int] = None,
    num_workers: Optional[int] = None,
    sequence_length: int = 2048,
    prompt_length: Optional[int] = None,
    batch_size: int = 1,
    shuffle: bool = True,
    keep_in_memory: bool = True,
) -> DataLoader:
    """
    Builds a dataloader for a huggingface dataset. Note that dataset_path corresponds to a local path in which
    the dataset is stored.
    """
    log.info(f"Loading {split} {dataset_id} dataset from {dataset_path}")
    assert sequence_length <= MODEL_ID_TO_DIMS[model_id]["max_position_embeddings"], (
        f'Selected sequence_length "{sequence_length}" is larger than the context_length '
        f'"{MODEL_ID_TO_DIMS[model_id]["max_position_embeddings"]}" for this model! '
    )

    if name is not None:
        dataset_path = os.path.join(dataset_path, name)
    if split is not None:
        dataset_path = os.path.join(dataset_path, split)

    # Load the dataset
    dataset = load_from_disk(dataset_path=dataset_path)
    log.info(f"Loaded {split} {dataset_id} dataset from {dataset_path}. Size: {len(dataset)}")

    # How many tokens to subsample
    if take_n_tokens is not None:
        dataset = dataset.take(take_n_tokens)

    # Tokenize and make sequences of sequence length
    log.info(f"Tokenizing the {split} {dataset_id} dataset.")
    dataset = tokenize_for_language_modeling(
        tokenizer,
        dataset,
        sequence_length=sequence_length,
        keep_in_memory=keep_in_memory,
    )

    # How many sequences/datapoints to subsample
    if take_n_sequences is not None:
        dataset = dataset.take(take_n_sequences)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )

    return dataloader
