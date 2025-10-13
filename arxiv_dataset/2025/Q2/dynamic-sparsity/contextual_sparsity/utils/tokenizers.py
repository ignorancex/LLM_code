# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from typing import Any, Dict, List, Optional, Type, Union

import torch
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase

from contextual_sparsity.utils.misc import parse_dtype

# A logger for this file
log = logging.getLogger(__name__)


def tokenize_for_language_modeling(
    tokenizer: PreTrainedTokenizer,
    data: Dataset,
    sequence_length: int = 1024,
    sliding_window_length: int = -1,
    data_key: str = "text",
    batch_size: Optional[int] = None,
    num_workers: int = 1,
    separator: str = "\n\n",
    keep_in_memory: bool = True,
) -> Dataset:
    """Tokenization for perplexity evaluation.
    Tokenize all sentences in ``experiment`` and concatenate them to form blocks
    of size `sequence_length`. Optionally, the concatenation can be done in a sliding
    window approach such that overlapping tokens are masked in the ground-truth labels.
    This is inspired from this Huggingface post on computing perplexity of fixe-length models
    https://huggingface.co/docs/transformers/perplexity

    Args:
        tokenizer: Pretrained tokenizer
        data: Huggingface dataset
        sequence_length: Length of each block in the final dataset
        sliding_window_length: Length of sliding window. If None, this will be set to the
            length of the context; i.e. blocks will be non-overlapping
        batch_size: Batch size for processing. If None, this will be set to the
            length of the dataset, i.e. proper concatenation. This should be preferred
            for small datasets.
        num_workers: Number of processes to use in map functions.
        data_key: Key that indexes the input sentence in ``experiment``
        separator: String separator used to join/concat the input sentences
        keep_in_memory: Keep the dataset in memory instead of writing it to a cache file.
    Returns:
        Dataset: A dataset containing:
          * `input_ids`: A block of token of size `sequence_length`, possibly overlapping
          with neighboring blocks when `striding_window_length` < `sequence_length`
          * `attention_mask`: Corresponding attention mask
          * `labels`: Corresponding ground-truth labels (*not* shifted) where tokens
          overlapping between windows are masked
    """
    # Tokenize the dataset
    try:
        tokenized_data = data.map(
            lambda samples: tokenizer(separator.join(samples[data_key])),
            remove_columns=[data_key],
            batched=True,
            batch_size=batch_size,
            num_proc=num_workers,
            keep_in_memory=keep_in_memory,
        )
    except:
        tokenized_data = data.map(
            lambda samples: tokenizer(separator.join(samples[data_key])),
            remove_columns=[data_key],
            batched=True,
            batch_size=batch_size,
            keep_in_memory=keep_in_memory,
        )
    tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Concat to get windows of size context length
    if sliding_window_length < 0:
        sliding_window_length = sequence_length
    assert 1 <= sliding_window_length <= sequence_length

    def __get_sliding_windows__(tokenized_samples: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, List[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for start in range(0, len(tokenized_samples["input_ids"]), sliding_window_length):
            # if the last window is too small, all the tokens are already
            # covered by previous windows, and we can ignore this sample
            if (
                len(tokenized_samples["input_ids"][start:])
                <= sliding_window_length
                < sequence_length
            ):
                continue

            # collect a window of size context length
            for key in ["input_ids", "attention_mask"]:
                out[key].append(tokenized_samples[key][start : start + sequence_length])
            out["labels"].append(out["input_ids"][-1].clone())

            # pad if smaller than context length
            if len(out["input_ids"][-1]) < sequence_length:
                padding = sequence_length - len(out["input_ids"][-1])
                out["input_ids"][-1] = torch.nn.functional.pad(
                    out["input_ids"][-1], (0, padding), value=0
                )
                out["attention_mask"][-1] = torch.nn.functional.pad(
                    out["attention_mask"][-1], (0, padding), value=0
                )
                out["labels"][-1] = torch.nn.functional.pad(
                    out["labels"][-1], (0, padding), value=-100
                )

            # ignore loss for tokens overlapping in the previous window
            if start > 0:
                out["labels"][-1][:-sliding_window_length] = -100
        return out

    assert (
        batch_size is None or batch_size >= sequence_length
    ), "batch size too small, all generated sequences will have trailing zero-padding!"

    # gather
    try:
        tokenized_data = tokenized_data.map(
            __get_sliding_windows__,
            batched=True,
            batch_size=batch_size or len(tokenized_data),
            num_proc=num_workers,
        )
    except:
        tokenized_data = tokenized_data.map(
            __get_sliding_windows__,
            batched=True,
            batch_size=batch_size or len(tokenized_data),
        )
    assert tokenized_data[0]["input_ids"].shape[0] == sequence_length, (
        tokenized_data[0]["input_ids"].shape[0],
        sequence_length,
    )

    return tokenized_data


def load_tokenizer(
    pretrained_model_path: str,
    use_fast_tokenizer: bool = True,
    dtype: Optional[Union[str, torch.dtype]] = None,
    tokenizer_type: Type[PreTrainedTokenizerBase] = AutoTokenizer,
) -> PreTrainedTokenizerBase:
    """
    Load the specified pretrained tokenizer.
    """

    if dtype is None:
        torch_dtype = torch.float16
    else:
        # Parse the data type
        torch_dtype = parse_dtype(dtype)

    tokenizer = tokenizer_type.from_pretrained(
        pretrained_model_path,
        use_fast=use_fast_tokenizer,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.vocab_size - 1

    return tokenizer
