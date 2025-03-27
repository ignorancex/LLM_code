import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.core.scheduler import SchedulerOutputs
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (Logprob, MultiModalData, SamplerOutput, Sequence,
                           SequenceGroup, SequenceGroupMetadata, SequenceStatus)
from vllm.usage.usage_lib import (UsageContext)
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port)

logger = init_logger(__name__)

from vllm import LLMEngine
from vllm.executor.gpu_executor import GPUExecutor
from llm_sys.engine.common import PipelineSequence, PipelineStageOut
from llm_sys.engine.scheduler import LayerwiseScheduler
from llm_sys.engine.worker import LayerwiseWorker

_dummy_token_id = 2
_dummy_log_prob = Logprob(1.)


class PipelineStageGpuExecutor(GPUExecutor):
    def _init_worker(self):
        self.layer_ids = self.model_config.layer_ids

        # FIXME: fix this for TP
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        self.driver_worker = LayerwiseWorker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            self.layer_ids,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def execute_model(self,
                      layer_id: int,
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      blocks_to_swap_in: Dict[int, int],
                      blocks_to_swap_out: Dict[int, int],
                      blocks_to_copy: Dict[int, List[int]]) -> SamplerOutput:
        output = self.driver_worker.execute_model(
            layer_id=layer_id,
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        return output


class PipelineStageEngine(LLMEngine):

    def __init__(
            self,
            model_config: ModelConfig,
            cache_config: CacheConfig,
            parallel_config: ParallelConfig,
            scheduler_config: SchedulerConfig,
            device_config: DeviceConfig,
            lora_config: Optional[LoRAConfig],
            vision_language_config: Optional["VisionLanguageConfig"],
            layer_ids: Tuple[int],
            executor_class,
            log_stats: bool,
            usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> None:
        model_config.layer_ids = layer_ids
        super().__init__(model_config, cache_config, parallel_config,
                         scheduler_config, device_config, lora_config,
                         vision_language_config, executor_class, log_stats,
                         usage_context, )
        self.scheduler = LayerwiseScheduler(scheduler_config, cache_config,
                                            lora_config, layer_ids)
        # disable the local seq counter because we need to keep a consistency
        # across pipeline stages and replicas.
        self.seq_counter = None

    @classmethod
    def from_engine_args(
            cls,
            engine_args,
            layer_ids: Tuple[int],
            usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()

        # Create the LLM engine.
        engine = cls(
            *engine_configs,
            layer_ids=layer_ids,
            executor_class=PipelineStageGpuExecutor,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
        )
        return engine

    def add_request(
            self,
            request_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            local_layers: Tuple[int, int],
            seq_id: int,
            input_tensor: Optional[torch.Tensor] = None,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: Optional[float] = None,
            lora_request: Optional[LoRARequest] = None,
            multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        max_logprobs = self.get_model_config().max_logprobs
        if (sampling_params.logprobs
            and sampling_params.logprobs > max_logprobs) or (
                sampling_params.prompt_logprobs
                and sampling_params.prompt_logprobs > max_logprobs):
            raise ValueError(f"Cannot request more than "
                             f"{max_logprobs} logprobs.")
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = self.encode_request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request)

        # Create the sequences.
        block_size = self.cache_config.block_size
        eos_token_id = self.tokenizer.get_lora_tokenizer(
            lora_request).eos_token_id
        seq = PipelineSequence(seq_id, prompt, prompt_token_ids, block_size,
                               input_tensor, eos_token_id, lora_request)

        # Defensive copy of SamplingParams, which are used by the sampler,
        # this doesn't deep-copy LogitsProcessor objects
        sampling_params = sampling_params.clone()
        # inject the eos token id into the sampling_params to support min_tokens
        # processing
        sampling_params.eos_token_id = seq.eos_token_id

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                  arrival_time, lora_request, multi_modal_data)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group, local_layers[0], local_layers)

    def step(self, layer_id: int, finished_seq_infos, force_decode: bool):
        self.scheduler.context_switch(layer_id)
        self.scheduler.finish_requests(finished_seq_infos)
        (seq_group_metadata_list, scheduler_outputs,
         finished_infos) = self.scheduler.schedule(force_decode=force_decode)

        if not scheduler_outputs.is_empty():
            output = self.model_executor.execute_model(
                layer_id, seq_group_metadata_list,
                scheduler_outputs.blocks_to_swap_in,
                scheduler_outputs.blocks_to_swap_out,
                scheduler_outputs.blocks_to_copy)
        else:
            output = []

        ret = self._process_model_outputs(output, scheduler_outputs)
        self.scheduler.finish_requests(finished_infos)
        self.scheduler.free_finished_seq_groups()
        return ret

    def update_seq_data(self, layer_id: int, req_id: str, seq_datas: Dict[int, torch.Tensor]):
        self.scheduler.update_req_data(layer_id, req_id, seq_datas)

    def _process_model_outputs(
            self, output: Union[SamplerOutput, PipelineStageOut],
            scheduler_outputs: SchedulerOutputs):
        now = time.time()
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups

        if isinstance(output, SamplerOutput):
            self.scheduler.last_layer_post_step(scheduler_outputs)
            return super()._process_model_outputs(output, scheduler_outputs), scheduler_outputs.prompt_run
        elif isinstance(output, list):
            assert len(output) == 0, "Bad output!"
            return None, None, None
        else:
            tensors_to_send = self.scheduler.post_step(scheduler_outputs, output)

            for scheduled_seq_group in scheduled_seq_groups:
                seq_group = scheduled_seq_group.seq_group
                token_chunk_size = scheduled_seq_group.token_chunk_size
                seq_group.update_num_computed_tokens(token_chunk_size)
                seq_group.maybe_set_first_token_time(now)
                # FIXME: this is a tmp hack to keep the output length.
                #  The correct impl should instead send the info at the next
                #  iter.
                for seq in seq_group.get_seqs(SequenceStatus.RUNNING):
                    seq.append_token_id(_dummy_token_id,
                                        {_dummy_token_id: _dummy_log_prob})
            return tensors_to_send, output.output_tensor, output.is_prompt
