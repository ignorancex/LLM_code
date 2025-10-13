# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from functools import partial
from glob import glob
from typing import Optional

import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

log = logging.getLogger(__name__)


def data_collator_with_truncation(features, seq_length, bos_token):
    batch = {}

    first = features[0]
    for k in first.keys():
        values = [f[k][:seq_length] for f in features]
        if bos_token is not None and k == "input_ids":
            values = [[bos_token] + v[:-1] for v in values]
        batch[k] = torch.tensor(values)

    if "labels" not in batch:
        batch["labels"] = batch["input_ids"].clone()

    return batch


def get_slimpajama_dataloader(
    tokenized_dataset_path: str,
    sequence_length: int,
    batch_size: int = 1,
    shuffle: bool = False,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model_id: Optional[str] = None,
    bos_token: Optional[str] = None,
    num_workers: int = 0,
    device: str = "cpu",
    take_n: Optional[int] = None,
):
    """
     Creates a dataloader object for a pre-tokenized SlimPajama dataset consisting of multiple .arrow files.
     The tokenized dataset is assumed to also come with sequences of length >= sequence_length that will be sliced
     if needed.

     Args:
        tokenized_dataset_path: The path to the tokenized .arrow files. Use '*" to match all the .arrow files.
        sequence_length: The desired sequence length for each batch produced by the dataloader.
        take_n: Slice the SlimPajama dataset at a specified number of sequences.
        batch_size: The batch size for the dataloader.
        model_id: The id of the model to load (this is used only to maintain a consistent interface).
        shuffle: Flag to enable shuffling.
        tokenizer: The tokenizer used to process the data.
            This argument is ignored in this function since the data is pre-tokenized.
        bos_token: Beginning of Sequence token (if any).
        device: The device to use for computation (this is used only to maintain a consistent interface).
        num_workers: Number of workers used for the dataloader.
    Returns:
        DataLoader: A dataloader object for the SlimPajama dataset.
    """

    # Determine all the files that match to the specified path
    files = glob(tokenized_dataset_path)

    if not files:
        raise RuntimeError("SlimPajama dataset not found.")

    log.info(f"Loading the tokenized SlimPajama dataset from {tokenized_dataset_path}")
    # Concatenate all the .arrow fragments
    tokenized_dataset = concatenate_datasets([Dataset.from_file(fpath) for fpath in sorted(files)])
    # Define a data collator that makes sequences of the specified length
    custom_collate_fn = partial(
        data_collator_with_truncation, seq_length=sequence_length, bos_token=bos_token
    )

    # If specified, slice the dataset to keep only a subset
    if take_n is not None:
        log.info(f"Taking a subset of {take_n} samples from SlimPajama")
        tokenized_dataset = tokenized_dataset.select(range(take_n))

    # Make a dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        collate_fn=custom_collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    batch = next(iter(dataloader))
    dataset_seq_length = batch["input_ids"].shape[1]
    assert dataset_seq_length == sequence_length, (
        f"Dataset does not support the requested seq_length of {sequence_length}. "
        f"Sequences up to {dataset_seq_length} are supported."
    )
    dataloader.seq_length = sequence_length

    return dataloader
