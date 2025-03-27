from typing import List, Optional, Sequence, Tuple, Union

import torch
from vllm.attention import AttentionMetadata
from vllm.config import (DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceData
from vllm.utils import CudaMemoryProfiler, make_tensor_with_pad
from vllm.worker.model_runner import ModelRunner, _PAD_SLOT_ID

from llm_sys.engine.common import (PipelineStageOut, PipelineSequenceGroupMetadata, PipelineSequenceData)


class LayerwiseModelRunner(ModelRunner):
    def __init__(
            self,
            model_config: ModelConfig,
            parallel_config: ParallelConfig,
            scheduler_config: SchedulerConfig,
            device_config: DeviceConfig,
            layer_ids: List[int],
            lora_config: Optional[LoRAConfig] = None,
            kv_cache_dtype: Optional[str] = "auto",
            is_driver_worker: bool = False,
            vision_language_config: Optional[VisionLanguageConfig] = None,
    ):
        super().__init__(model_config, parallel_config, scheduler_config,
                         device_config, lora_config, kv_cache_dtype,
                         is_driver_worker, vision_language_config)
        self.last_layer_id = self.model_config.get_num_layers(self.parallel_config) - 1
        self.layer_ids = layer_ids
        self.num_total_layer = self.model_config.hf_text_config.num_hidden_layers
        self.models = {layer_id: None for layer_id in self.layer_ids}

    def load_model(self):
        with CudaMemoryProfiler() as m:
            for layer_id in self.layer_ids:
                self.model_config.hf_config.layer_id = layer_id
                self.models[layer_id] = get_model(
                    self.model_config,
                    self.device_config,
                    lora_config=self.lora_config,
                    vision_language_config=self.vision_language_config,
                    parallel_config=self.parallel_config,
                    scheduler_config=self.scheduler_config)

        self.model_memory_usage = m.consumed_memory

    def _prepare_prompt(
            self,
            seq_group_metadata_list: List[PipelineSequenceGroupMetadata],
            is_first_layer: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, List[int],
    List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_hidden_states: List[torch.Tensor] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        prompt_lens: List[int] = []
        context_lens: List[int] = []
        subquery_lens: List[int] = []
        prefix_block_tables: List[List[int]] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            computed_block_nums = seq_group_metadata.computed_block_nums

            token_chunk_size = seq_group_metadata.token_chunk_size
            seq_data = seq_group_metadata.seq_data[seq_id]
            computed_len = seq_data.get_num_computed_tokens()
            # We should use get_len here because in case of preemption
            # it contains output tokens.
            prefill_end = min(seq_data.get_len(),
                              computed_len + token_chunk_size)
            if is_first_layer:
                # TODO(sang): Rename it after chunked prefill is introduced.
                prompt_tokens = seq_data.get_token_ids()[computed_len:prefill_end]
                prompt_len = len(prompt_tokens)
            else:
                prompt_token_hidden = seq_data.get_input_activation()
                if not isinstance(prompt_token_hidden, torch.Tensor):
                    prompt_token_hidden = prompt_token_hidden[0][prompt_token_hidden[1]:prompt_token_hidden[2]]
                prompt_token_hidden = prompt_token_hidden[computed_len:prefill_end]
                prompt_len = prompt_token_hidden.shape[0]
            # Right now, the prefill_end is always same as the length of
            # sequence. However, once chunked prefill is introduced, this
            # assumption can be changed.
            assert prefill_end == seq_data.get_len()
            prompt_lens.append(prompt_len)

            # NOTE: This only works for oooooooxxx style attention.
            if computed_block_nums is not None and len(
                    computed_block_nums) > 0 and self.sliding_window is None:
                # Prefix is not supported with sliding_window
                computed_len = len(computed_block_nums) * self.block_size
                if is_first_layer:
                    prompt_tokens = prompt_tokens[computed_len:]
                else:
                    prompt_token_hidden = prompt_token_hidden[computed_len:]
                prefix_block_tables.append(computed_block_nums)
            else:
                prefix_block_tables.append([])
                # Right now, prefill start is always 0. However, this
                # assumption can be changed once chunked prefill is introduced.
                assert computed_len == 0

            # actual prompt lens
            context_lens.append(computed_len)
            subquery_lens.append(prompt_len - computed_len)

            if is_first_layer:
                input_tokens.extend(prompt_tokens)
            else:
                # FIXME: this is slow. Do it in cpp to avoid torch tensor metadata
                #  creation. Or at least merge neighbor slices.
                if isinstance(prompt_token_hidden, tuple):
                    prompt_token_hidden = prompt_token_hidden[0][prompt_token_hidden[1]:prompt_token_hidden[2]]
                input_hidden_states.append(prompt_token_hidden)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(computed_len, prefill_end)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                assert computed_len == 0, (
                    "Prefix caching is currently not supported with "
                    "sliding window attention")
                start_idx = max(0, prompt_len - self.sliding_window)

            for i in range(computed_len, prefill_end):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        max_subquery_len = max(subquery_lens)
        max_prompt_len = max(prompt_lens)
        assert max_subquery_len > 0

        if is_first_layer:
            input_tokens = torch.tensor(input_tokens,
                                        dtype=torch.long,
                                        device=self.device)
            num_prompt_tokens = len(input_tokens)
        else:
            input_hidden_states = torch.concatenate(tuple(input_hidden_states),
                                                    dim=0)
            num_prompt_tokens = input_hidden_states.shape[0]
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)

        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device=self.device)

        # Prepare prefix block tables
        max_prompt_block_table_len = max(len(t) for t in prefix_block_tables)
        block_tables = make_tensor_with_pad(
            prefix_block_tables,
            max_len=max_prompt_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )

        # Query length can be shorter than key (i.e., prompt) when prefill
        # is chunked or prefix cached.
        subquery_lens_tensor = torch.tensor(subquery_lens,
                                            dtype=torch.long,
                                            device=self.device)
        subquery_start_loc = torch.zeros(subquery_lens_tensor.shape[0] + 1,
                                         dtype=torch.int32,
                                         device=self.device)

        prompt_lens_tensor = torch.tensor(prompt_lens,
                                          dtype=torch.long,
                                          device=self.device)
        seq_start_loc = torch.zeros(prompt_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)

        torch.cumsum(subquery_lens_tensor,
                     dim=0,
                     dtype=subquery_start_loc.dtype,
                     out=subquery_start_loc[1:])

        torch.cumsum(prompt_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            slot_mapping=slot_mapping,
            prompt_lens=prompt_lens,
            prompt_lens_tensor=prompt_lens_tensor,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=0,
            max_subquery_len=max_subquery_len,
            max_context_len=None,
            max_prompt_len=max_prompt_len,
            subquery_start_loc=subquery_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        if is_first_layer:
            input_hidden_states = input_tokens
        return (input_hidden_states, input_positions, attn_metadata,
                prompt_lens, subquery_lens,)

    def _prepare_decode(
            self,
            seq_group_metadata_list: List[PipelineSequenceGroupMetadata],
            is_first_layer: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata,]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_hidden_states: List[torch.Tensor] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                if is_first_layer:
                    generation_token = seq_data.get_last_token_id()
                    input_tokens.append(generation_token)
                else:
                    # FIXME: this is slow. Do it in cpp to avoid torch tensor metadata
                    #  creation. Or at least merge neighbor slices.
                    generated_state = seq_data.get_input_activation()
                    if isinstance(generated_state, tuple):
                        generated_state = generated_state[0][generated_state[1]:generated_state[2]]
                    input_hidden_states.append(generated_state)

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append(position)

                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        max_context_len = max(context_lens)

        if is_first_layer:
            input_tokens = torch.tensor(input_tokens,
                                        dtype=torch.long,
                                        device=self.device)
        else:
            input_hidden_states = torch.concatenate(input_hidden_states, dim=0)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=self.device)

        max_block_table_len = max(
            len(block_table) for block_table in block_tables)
        block_tables = make_tensor_with_pad(
            block_tables,
            max_len=max_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            prompt_lens=None,
            prompt_lens_tensor=None,
            num_prompt_tokens=0,
            num_generation_tokens=len(input_tokens),
            max_subquery_len=None,
            max_context_len=max_context_len,
            max_prompt_len=None,
            subquery_start_loc=None,
            seq_start_loc=None,
            context_lens=context_lens,
            block_tables=block_tables,
            use_cuda_graph=False,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        if is_first_layer:
            input_hidden_states = input_tokens
        return (input_hidden_states, input_positions, attn_metadata)

    def prepare_input_tensors(
            self,
            seq_group_metadata_list: Optional[List[PipelineSequenceGroupMetadata]],
            is_first_layer: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata]:
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_hidden_states, input_positions, attn_metadata, prompt_lens,
             subquery_lens,) = self._prepare_prompt(seq_group_metadata_list,
                                                    is_first_layer)
        else:
            (input_hidden_states, input_positions,
             attn_metadata) = self._prepare_decode(seq_group_metadata_list,
                                                   is_first_layer)
            prompt_lens = []
            subquery_lens = None
        sampling_metadata = self._prepare_sample(seq_group_metadata_list,
                                                 prompt_lens, subquery_lens)

        return (input_hidden_states, input_positions, attn_metadata,
                sampling_metadata,)

    @torch.inference_mode()
    def execute_model(
            self,
            seq_group_metadata_list: Optional[List[PipelineSequenceGroupMetadata]],
            kv_cache: torch.Tensor,
            layer_id: int,
    ) -> Optional[Union[SamplerOutput, PipelineStageOut]]:
        is_first_layer = layer_id == 0
        (input_hidden_states, input_positions, attn_metadata, sampling_metadata,
         ) = self.prepare_input_tensors(seq_group_metadata_list, is_first_layer)

        # Execute the model.
        model_executable = self.models[layer_id]
        execute_model_kwargs = {
            "positions": input_positions,
            "kv_cache": kv_cache,
            "attn_metadata": attn_metadata,
            "residual": None,
        }
        if is_first_layer:
            execute_model_kwargs["input_ids"] = input_hidden_states
        else:
            execute_model_kwargs["hidden_states"] = input_hidden_states
        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        if layer_id == self.num_total_layer - 1:
            logits = model_executable.compute_logits(hidden_states, sampling_metadata)

            # Only perform sampling in the driver worker.
            if not sampling_metadata.perform_sampling:
                return None

            # Sample the next token.
            output = model_executable.sample(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
            return output
        else:
            is_prompt = seq_group_metadata_list[0].is_prompt
            return PipelineStageOut(hidden_states,
                                    sampling_metadata.prompt_lens, is_prompt)

    @torch.inference_mode()
    def profile_run(self) -> None:
        # FIXME: fix this function.
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests_per_seq = []

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[PipelineSequenceGroupMetadata] = []
        # Additional GPU memory may be needed for vision encoding, which needs
        # to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        if self.vision_language_config:
            max_num_seqs = min(
                max_num_seqs,
                int(max_num_batched_tokens /
                    self.vision_language_config.image_feature_size))

        layer_id = self.layer_ids[0]
        # If it is the last stage, profile with the last layer to include
        # all post main body computations
        if self.last_layer_id == self.layer_ids[-1]:
            layer_id = self.layer_ids[-1]
        is_first_layer = layer_id == 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data, fake_multi_modal_input = _prepare_fake_inputs(
                seq_len, self.vision_language_config, is_first_layer,
                self.model_config.get_hidden_size())
            seq = PipelineSequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=fake_multi_modal_input,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        self.execute_model(seqs, None, layer_id)
        torch.cuda.synchronize()
        return


def _prepare_fake_inputs(
        seq_len: int, vision_language_config: Optional[VisionLanguageConfig],
        is_first_stage: bool, hidden_size: int = None):
    """Prepare fake inputs for profile run."""
    assert vision_language_config is None
    prompt_tokens = [0] * seq_len
    if is_first_stage:
        seq_data = SequenceData(prompt_tokens)
    else:
        # FIXME: send it to the correct cuda device if there is TP
        seq_data = PipelineSequenceData(
            prompt_tokens,
            torch.rand((seq_len, hidden_size), dtype=torch.float16).to("cuda"))
    fake_image_input = None
    return seq_data, fake_image_input


if __name__ == "__main__":
    import llama
    from vllm.utils import get_distributed_init_method, get_ip, get_open_port
    from vllm.worker.worker import init_distributed_environment

    model_config = ModelConfig(model="/root/fault_tol/mrprt/test/model_config",
                               tokenizer="georgesung/llama2_7b_chat_uncensored",
                               tokenizer_mode="auto",
                               trust_remote_code=True,
                               download_dir=None,
                               load_format="dummy",
                               dtype="auto",
                               seed=0,
                               max_model_len=2048,
                               enforce_eager=True)
    parallel_config = ParallelConfig(1, 1, False)
    scheduler_config = SchedulerConfig(max_num_batched_tokens=2048,
                                       max_num_seqs=128,
                                       max_model_len=2048)
    device_config = DeviceConfig("auto")
    layer_ids = [0, 1]

    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())
    init_distributed_environment(parallel_config, 0, distributed_init_method)
    runner = LayerwiseModelRunner(model_config, parallel_config,
                                  scheduler_config, device_config, layer_ids,
                                  is_driver_worker=True)
    runner.load_model()
    runner.profile_run()
    for layer in layer_ids:
        runner.execute_model(seq_group_metadata_list, kv_cache, layer)
