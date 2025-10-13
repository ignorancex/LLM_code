# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os.path
from typing import Any, List, Optional, Set, Tuple, Type, Union

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# A logger for this file
log = logging.getLogger(__name__)


def pairwise_disjoint(sets: List[Set]) -> bool:
    """
    Checks if a collection of sets is pairwise disjoint.
    A collection of sets is pairwise disjoint if any two sets in the collection are disjoint.
    """
    union = set().union(*sets)
    return len(union) == sum(map(len, sets))


def parse_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    """
    Parse a dtype string to a torch.dtype if necessary.
    """
    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    elif isinstance(dtype, str):
        torch_dtype = getattr(torch, dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    assert isinstance(torch_dtype, torch.dtype)
    return torch_dtype


def download_tokenizer(
    repo_id: str,
    download_dir: str,
):
    """
    Download pre-trained tokenizer
    """
    log.info(f"Loading a pretrained {repo_id} tokenizer from huggingface")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, force_download=True)

    log.info(f"Tokenizer downloaded and stored to {download_dir}")
    tokenizer.save_pretrained(download_dir)


def download_hf_model(
    repo_id: str,
    download_dir: str,
    max_num_downloads: int = 5,
    model_type: Type[PreTrainedModel] = AutoModelForCausalLM,
):
    """
    Download pre-trained HuggingFace model
    """
    model = None

    while max_num_downloads > 0:
        try:
            log.info(f"Loading a pretrained {repo_id} from huggingface with type {model_type}")
            model = model_type.from_pretrained(
                pretrained_model_name_or_path=repo_id,
                resume_download=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cpu",
                torch_dtype="auto",
            )
            max_num_downloads = 0
        except OSError:
            max_num_downloads -= 1
    if model is None:
        raise NotImplementedError(f"Could not download the model {repo_id}")

    log.info(f"Model {repo_id} loaded and stored in {download_dir}")
    model.save_pretrained(download_dir)


def download_hf_model_and_tokenizer(
    repo_id: str,
    download_dir: str,
    max_num_downloads: int = 5,
    model_type: Optional[Type[PreTrainedModel]] = None,
):
    """
    Download pre-trained HuggingFace model and tokenizer
    """
    download_hf_model(repo_id, download_dir, max_num_downloads, model_type)
    download_tokenizer(repo_id, download_dir)


def download_dataset(
    dataset_id: str,
    download_dir: str,
    name: Optional[str],
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """
    Download dataset from huggingface
    """
    log.info(f"Downloading the {dataset_id} dataset from huggingface")
    dataset = load_dataset(path=dataset_id, name=name, split=split, cache_dir=cache_dir)

    if name is not None:
        download_dir = os.path.join(download_dir, name)
    if split is not None:
        download_dir = os.path.join(download_dir, split)

    log.info(f"Dataset downloaded and stored in {download_dir}")
    dataset.save_to_disk(download_dir)


def move_to_device(batch: Any, device: Union[str, torch.device]) -> Any:
    """
    Move a batch to a specified device
    """
    if isinstance(batch, dict):
        batch = {k: v.to(device) for k, v in batch.items()}
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    elif isinstance(batch, list):
        batch = [v.to(device) for v in batch]
    else:
        raise NotImplementedError()
    return batch


def cast_to(batch: Any, dtype: torch.dtype) -> Any:
    """
    Cast a batch to a specified dtype
    """
    if isinstance(batch, dict):
        batch = {k: v.type(dtype) for k, v in batch.items()}
    elif isinstance(batch, torch.Tensor):
        batch = batch.type(dtype)
    elif isinstance(batch, list):
        batch = [v.type(dtype) for v in batch]
    else:
        raise NotImplementedError()
    return batch


def get_batch_size(batch: Any) -> int:
    """
    Determine the batch size of a batch
    """
    if isinstance(batch, dict):
        batch_size = next(iter(batch.values())).shape[0]
    elif isinstance(batch, torch.Tensor):
        batch_size = batch.shape[0]
    elif isinstance(batch, list):
        batch_size = next(iter(batch)).shape[0]
    else:
        raise NotImplementedError()
    return batch_size


def split_gate_up_layer(gate_up_layer: nn.Linear) -> Tuple[nn.Linear, nn.Linear]:
    """
    Split a gate-up linear layer into two linear layers
    """
    # Make the two linear layers out of the one up_gate_layer
    w_gate, w_up = torch.chunk(gate_up_layer.weight, 2, dim=0)
    if gate_up_layer.bias is not None:
        b_gate, b_up = torch.chunk(gate_up_layer.bias, 2, dim=0)
    else:
        b_gate, b_up = None, None

    fc_gate = nn.Linear(w_gate.shape[1], w_gate.shape[0], bias=b_gate is not None)
    fc_gate.weight.data = w_gate.data
    if b_gate is not None:
        fc_gate.bias.data = b_gate.data

    fc_up = nn.Linear(w_up.shape[1], w_up.shape[0], bias=b_up is not None)
    fc_up.weight.data = w_up.data
    if b_up is not None:
        fc_up.bias.data = b_up.data

    return fc_gate, fc_up
