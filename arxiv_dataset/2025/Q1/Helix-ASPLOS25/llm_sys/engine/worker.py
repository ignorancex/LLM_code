import copy
import os
from typing import Dict, List, Optional

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.model_executor import set_random_seed
from vllm.sequence import SamplerOutput
from vllm.worker.worker import (Worker, _check_if_gpu_supports_dtype,
                                init_distributed_environment)

from llm_sys.engine.model_runner import LayerwiseModelRunner
from llm_sys.engine.common import PipelineSequenceGroupMetadata


class LayerwiseWorker(Worker):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        layer_ids: List[int],
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool=True,
    ) -> None:
        super().__init__(model_config, parallel_config, scheduler_config,
                         device_config, local_rank, rank,
                         distributed_init_method, lora_config,
                         vision_language_config, kv_cache_dtype,
                         is_driver_worker)
        self.model_runner = LayerwiseModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            layer_ids,
            lora_config=self.lora_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config)

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # # Initialize the distributed environment.
        init_distributed_environment(self.parallel_config, self.rank,
                                     self.distributed_init_method,
                                     self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def init_cache_engine(self, cache_config: CacheConfig):
        model_config = copy.deepcopy(self.model_config)
        # monkey patch model config to mimic that only one layer exists
        backup_model_config = self.model_config
        model_config.get_num_layers = lambda _: 1
        self.model_config = model_config
        super().init_cache_engine(cache_config)
        self.model_config = backup_model_config

        self.gpu_cache = self.cache_engine.gpu_cache[0]

    def warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def execute_model(
        self,
        layer_id: int,
        seq_group_metadata_list: Optional[List[PipelineSequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[SamplerOutput]:
        assert seq_group_metadata_list is not None
        num_seq_groups = len(seq_group_metadata_list)
        assert blocks_to_swap_in is not None
        assert blocks_to_swap_out is not None
        assert blocks_to_copy is not None

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache, layer_id)
        return output

    def get_cache_block_size_bytes(self, block_size, cache_dtype):
        tot_size = super().get_cache_block_size_bytes(block_size, cache_dtype)
        # remove this factor in the parent class's method
        return tot_size / self.model_config.get_num_layers(self.parallel_config)


if __name__ == "__main__":
    import llama
    from vllm.config import CacheConfig
    from vllm.executor.utils import check_block_size_valid
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
    device_config = DeviceConfig("cuda")
    layer_ids = [0, 1]

    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())
    rank = local_rank = 0
    worker = LayerwiseWorker(model_config, parallel_config, scheduler_config,
                             device_config, layer_ids, rank, local_rank,
                             distributed_init_method)

    worker.init_device()
    worker.load_model()

    block_size = 16
    gpu_mem_util = 0.1
    cpu_swap_space = 1 << 30 # 1GB
    cache_dtype = "auto"
    cache_config = CacheConfig(block_size, gpu_mem_util, cpu_swap_space,
                               cache_dtype)

    num_gpu_blocks, num_cpu_blocks = (
        worker.profile_num_available_blocks(
            block_size=cache_config.block_size,
            gpu_memory_utilization=cache_config.
            gpu_memory_utilization,
            cpu_swap_space=cache_config.swap_space_bytes,
            cache_dtype=cache_config.cache_dtype,
        ))

    check_block_size_valid(num_gpu_blocks, cache_config.block_size,
                           model_config.max_model_len)

    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = num_cpu_blocks
    worker.init_cache_engine(cache_config)
