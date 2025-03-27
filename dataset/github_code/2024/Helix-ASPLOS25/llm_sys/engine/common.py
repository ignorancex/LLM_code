from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from vllm.sampling_params import SamplingParams
from vllm.sequence import (Sequence, SequenceData, SequenceGroupMetadata, SequenceGroupState, MultiModalData)

TensorSlice = Tuple[torch.Tensor, int, int]
InputActivationType = Union[torch.Tensor, TensorSlice]


class PipelineSequenceData(SequenceData):
    def __init__(self,
                 prompt_token_ids: List[int],
                 input_act: InputActivationType,
                 output_token_ids: Optional[List[int]] = None):
        super().__init__(prompt_token_ids, output_token_ids)
        self.input_act: InputActivationType = input_act

    def set_input_hidden(self, val: torch.Tensor, is_prompt: bool):
        self.input_act = val
        if is_prompt:
            assert len(self.output_token_ids) == 0

    def get_input_activation(self) -> torch.Tensor:
        return self.input_act


class PipelineSequence(Sequence):
    def __init__(self, seq_id: int,
                 prompt: str,
                 prompt_token_ids: List[int],
                 block_size: int,
                 input_act: InputActivationType,
                 eos_token_id: Optional[int] = None,
                 lora_request=None, ):
        super().__init__(seq_id, prompt, prompt_token_ids, block_size,
                         eos_token_id, lora_request)

        self.data = PipelineSequenceData(prompt_token_ids, input_act)

    def set_input_hidden(self, val: InputActivationType, is_prompt: bool):
        # Call this at where we do append token
        self.data.set_input_hidden(val, is_prompt)


class PipelineSequenceGroupMetadata(SequenceGroupMetadata):
    def __init__(self, request_id: str, is_prompt: bool,
                 seq_data: Dict[int, PipelineSequenceData],
                 sampling_params: SamplingParams,
                 block_tables: Dict[int, List[int]],
                 token_chunk_size: Optional[int] = None,
                 lora_request=None,
                 computed_block_nums: Optional[List[int]] = None,
                 state: Optional[SequenceGroupState] = None,
                 multi_modal_data: Optional[MultiModalData] = None, ):
        super().__init__(request_id, is_prompt, seq_data, sampling_params,
                         block_tables, token_chunk_size, lora_request,
                         computed_block_nums, state, multi_modal_data)
        self.seq_data: Dict[int, PipelineSequenceData]


@dataclass
class PipelineStageOut:
    output_tensor: torch.Tensor
    prompt_lens: List[int]
    is_prompt: bool
