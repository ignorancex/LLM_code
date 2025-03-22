import enum
from typing import List, Optional

import torch

from vllm.model_executor.layers.sampler import SamplerOutput

from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.worker.model_runner_base import (ModelRunnerBase,
                                           ModelRunnerInputBase)
from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner
from vllm.spec_decode.target_model_runner import TargetModelRunner

logger = init_logger(__name__)

# A flag to enable debug prints for the updated input tensors
# before each step.
debug_advance_input = False
# A flag to allow GPU advance step for draft model runner.
# Set to False for debugging.
allow_gpu_advance_step = True

class StationChefStage(enum.Enum):
    """The stage of the station chef.
    Since the station chef inherits from both TP1DraftModelRunner and TargetModelRunner,
    it should use this class to decide which execute_model to call.
    """
    PROPOSAL = enum.auto()
    SCORING = enum.auto()

class StationChefRunner(TP1DraftModelRunner, TargetModelRunner):
    """Specialized model runner for chef decoding member models, i.e., station chef.
    Since the station chef always execute k forward passes consecutively to
    generate k speculative tokens in a single speculative decoding step,
    we could get rid of most CPU-GPU synchronization and data transfer
    overheads by keeping model input and output tensors on GPU all the time.

    TODOs:
    1. Currently supports only flash-attn, add support for other attn_backends.
    2. Support TP > 1 (this requires some designs because we do not expect
       any broadcasting inside execute_model).
    """

    def __init__(self, model_runner: ModelRunnerBase):
        super().__init__(model_runner)
        TargetModelRunner.__init__(self, model_runner=model_runner)

        # Lazy initialization
        self.num_speculative_tokens: int
        self.stage: StationChefStage

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelRunnerInputBase,
        kv_caches: List[torch.Tensor],
        previous_hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if self.stage == StationChefStage.PROPOSAL:
            return super().execute_model(
                model_input=model_input,
                kv_caches=kv_caches,
                previous_hidden_states=previous_hidden_states,
                intermediate_tensors=intermediate_tensors,
                num_steps=num_steps)
        elif self.stage == StationChefStage.SCORING:
            return self.model_runner.execute_model(
                model_input=model_input,
                kv_caches=kv_caches,
                intermediate_tensors=intermediate_tensors,
                num_steps=num_steps)
        else:
            raise ValueError(f"Unsupported stage: {self.stage}")
