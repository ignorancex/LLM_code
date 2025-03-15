from functools import cached_property
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import copy

from vllm.config import (
    EnsembleDecodingConfig,
    VllmConfig,
)
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (
    ExecuteModelRequest,
    SequenceGroupMetadata,
)

from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase
from vllm.ensemble_decode.ensemble_sampler import EnsembleSampler

logger = init_logger(__name__)


def create_ensemble_worker(*args, **kwargs) -> "EnsembleDecodeWorker":
    vllm_config: VllmConfig = kwargs.get("vllm_config")
    ensemble_decoding_config: EnsembleDecodingConfig = vllm_config.ensemble_decoding_config
    assert ensemble_decoding_config is not None

    workers = [Worker(*args, **kwargs)]

    for model_config, parallel_config in zip(
            ensemble_decoding_config.model_configs[1:],
            ensemble_decoding_config.parallel_configs[1:]):
        new_vllm_config = copy.deepcopy(vllm_config)
        new_vllm_config.model_config = model_config
        new_vllm_config.parallel_config = parallel_config
        ensemble_worker_kwargs = kwargs.copy()
        ensemble_worker_kwargs.update(vllm_config=new_vllm_config)
        workers.append(Worker(**ensemble_worker_kwargs))

    return EnsembleDecodeWorker(
        workers=workers,
        ensemble_fn=ensemble_decoding_config.ensemble_fn,
        ensemble_target=ensemble_decoding_config.ensemble_target)


class EnsembleDecodeWorker(LoraNotSupportedWorkerBase):

    def __init__(
        self,
        workers: List[Worker],
        ensemble_fn: Callable,
        ensemble_target: Literal["raw-logits", "logits", "probs"],
    ):
        self.workers = workers
        self.sampler = EnsembleSampler(ensemble_fn=ensemble_fn,
                                       ensemble_target=ensemble_target)

    def init_device(self) -> None:
        for worker in self.workers:
            worker.init_device()

    def load_model(self, *args, **kwargs):
        for worker in self.workers:
            worker.load_model()

    def get_cache_block_size_bytes(self) -> int:
        pass

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of cache blocks to use.
        This is done by profiling the base model (which is typically the
        larger of the two). Then the total memory which would be used by the
        base model KV is divided evenly between the positive and negative model KV,
        such that the number of blocks is equal in both KV caches.
        """
        num_gpu_blocks, num_cpu_blocks = self.workers[
            0].determine_num_available_blocks()
        base_cache_block_size_bytes = self.workers[
            0].get_cache_block_size_bytes()
        total_memory = num_gpu_blocks * base_cache_block_size_bytes
        new_num_gpu_blocks = total_memory // sum(
            worker.get_cache_block_size_bytes() for worker in self.workers)

        logger.info(
            f"Num GPU blocks: {new_num_gpu_blocks}, Num CPU blocks: {num_cpu_blocks}"
        )
        return new_num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the cache engine of the scorer and proposer workers."""
        for worker in self.workers:
            worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                    num_cpu_blocks=num_cpu_blocks)

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Perform ensemble decoding on the input batch."""
        if self.rank != self._driver_rank:
            return []

        if execute_model_req is None:
            """
            This signals that there's no more requests to process for now.
            All workers are running infinite loop with broadcast_tensor_dict,
            and it stops the loop when the driver broadcasts an empty input.
            Send an empty input to notify all other workers to stop their
            execution loop.
            """
            broadcast_tensor_dict({}, src=0)
            return []

        return self._run_ensemble_decoding(execute_model_req)

    def _run_ensemble_decoding(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        """
        Run the model with ensemble decoding.
        """
        sampler_outputs = [
            worker.execute_model(execute_model_req) for worker in self.workers
        ]

        generators = self.workers[0].model_runner.get_generators(
            execute_model_req.finished_requests_ids)

        input_tokens_tensor, seq_lens, query_lens = self._prepare_input_tensors(
            execute_model_req.seq_group_metadata_list)

        sampling_metadata = SamplingMetadata.prepare(
            execute_model_req.seq_group_metadata_list,
            seq_lens,
            query_lens,
            self.device,
            self.workers[0].model_runner.pin_memory,
            generators,
        )

        # Sample the next token.
        logits = [output[0].logits for output in sampler_outputs]
        output: SamplerOutput = self.sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return [output]

    def _prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        if not seq_group_metadata_list:
            return torch.empty(0, device=self.device), [], []

        input_tokens: List[int] = []
        seq_lens: List[int] = []
        query_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            is_prompt = seq_group_metadata.is_prompt

            for seq_data in seq_group_metadata.seq_data.values():
                seq_data_len = seq_data.get_len()
                if is_prompt:
                    context_len = seq_data.get_num_computed_tokens()
                    seq_len = min(
                        seq_data_len,
                        context_len + seq_group_metadata.token_chunk_size)
                    tokens = seq_data.get_token_ids()[context_len:seq_len]
                    seq_lens.append(seq_len)
                    input_tokens.extend(tokens)
                    query_lens.append(seq_len - context_len)
                else:
                    seq_lens.append(seq_data_len)
                    input_tokens.append(seq_data.get_last_token_id())
                    query_lens.append(1)

        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device=self.device)
        return input_tokens_tensor, seq_lens, query_lens

    @cached_property
    def vocab_size(self) -> int:
        return self.workers[0].vocab_size

    @property
    def rank(self) -> int:
        return self.workers[0].rank

    @property
    def device(self) -> torch.device:
        return self.workers[0].device

    @property
    def _driver_rank(self) -> int:
        return 0
