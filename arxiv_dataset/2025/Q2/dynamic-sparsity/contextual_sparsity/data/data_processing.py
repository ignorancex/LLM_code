# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch


def separate_prompt(
    batch: Dict[str, torch.Tensor], prompt_length: int, sequence_length: int
) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """
    Separate batch in prompt and rest (generated part of the sentence).
    If prompt_length == 0, return None for the prompt.
    """
    assert 0 <= prompt_length < sequence_length, (prompt_length, sequence_length)

    if prompt_length == 0:
        assert batch["input_ids"].shape[1] == sequence_length
        return None, batch

    prompt, rest = dict(), dict()
    for k, v in batch.items():
        if k == "attention_mask":
            continue
        assert v.shape[1] == sequence_length
        prompt[k] = v[:, :prompt_length]
        rest[k] = v[:, prompt_length:]
    return prompt, rest


def move_dict_to_device(
    batch: Dict[str, torch.Tensor], device: Union[str, torch.device]
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def sequential_preprocessing(batch: Any, functions: List[Callable]) -> Any:
    # Applies a sequence of preprocessing functions to the input batch
    for fn in functions:
        batch = fn(batch)
    return batch
