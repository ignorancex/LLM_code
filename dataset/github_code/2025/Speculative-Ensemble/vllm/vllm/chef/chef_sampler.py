from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.jit

from vllm.logger import init_logger
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.ensemble_decode.ensemble_sampler import EnsembleSampler

logger = init_logger(__name__)


class ChefSampler(RejectionSampler):
    """Apply modified rejection sampling as described in "Accelerating Large
    Language Model Decoding with Speculative Sampling"
    https://arxiv.org/pdf/2302.01318.pdf.
    """

    def __init__(
        self,
        strict_mode: bool = False,
        use_flashinfer: Optional[bool] = None,
        ensemble_fn: Optional[Callable] = None,
        ensemble_target: Optional[str] = None,
        **kwargs,
    ):
        """Create a rejection sampler.

        Args:
            strict_mode: Whether or not to perform shape/device/dtype checks
            during sampling. This catches correctness issues but adds
            nontrivial latency.
            use_falshinfer: We will use this parameter to determine whether
            to use the FlashInfer rejection sampling kernel or not. If it's
            None, we will use the default value from the environment variable.
            This parameter is only used for testing purposes.
        """
        super().__init__(strict_mode=strict_mode, use_flashinfer=use_flashinfer)

        self.sampler = EnsembleSampler(ensemble_fn, ensemble_target)
        self.sampler.include_gpu_probs_tensor = True
        self.sampler.should_modify_greedy_probs_inplace = True

    def forward(
        self,
        station_chef_logits: List[torch.Tensor],  # list of [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        sampling_metadata: SamplingMetadata,
        seeded_seqs: Optional[Dict[int, torch.Generator]] = None,
        return_target_logprobs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform rejection sampling.
        Args:
            station_chef_logits: List of logits from the station chef model.
            draft_probs: Probabilities of the draft model.
            draft_token_ids: Token ids from the draft model.
            sampling_metadata: Sampling metadata.
            seeded_seqs: Seeded sequences.
            return_target_logprobs: Whether or not to return target logprobs.
                This is used in verification tokens of chef.
        """
        assert station_chef_logits[0].shape[:2] == draft_token_ids.shape, (
            f"Logits shape {station_chef_logits[0].shape} must match token_ids shape "
            f"{draft_token_ids.shape}."
        )

        batch_size, k, _ = draft_probs.shape

        # batch_size = 0 when all requests in the batch are
        # non_spec requests. In this case, output_token_ids is
        # just an empty tensor.
        if batch_size == 0:
            return torch.empty(0, k, device=draft_probs.device, dtype=int)

        # calculate ensemble probs
        target_probs_list, target_logprobs_list = [], []
        for idx in range(k):
            probs_list, logprobs_list = [], []
            for batch_idx in range(batch_size):
                # [1, vocab_size]
                sampler_output: Optional[SamplerOutput] = self.sampler(
                    [
                        logits[batch_idx, idx].unsqueeze(0)
                        for logits in station_chef_logits
                    ],
                    sampling_metadata,
                )
                assert sampler_output is not None, "Sampler output is None."
                probs_list.append(sampler_output.sampled_token_probs)
                logprobs_list.append(sampler_output.logprobs)
            
            # [batch_size, 1, vocab_size]
            target_probs_list.append(
                torch.vstack(probs_list)
                if len(probs_list) > 1
                else probs_list[0].unsqueeze(0)
            )
            target_logprobs_list.append(
                torch.vstack(logprobs_list)
                if len(logprobs_list) > 1
                else logprobs_list[0].unsqueeze(0)
            )
        
        # [batch_size, k, vocab_size]
        target_probs = torch.cat(target_probs_list, dim=1)
        target_logprobs = torch.cat(target_logprobs_list, dim=1)

        if self.use_flashinfer:
            raise NotImplementedError("FlashInfer is not supported yet.")
        else:
            accepted, recovered_token_ids = self._batch_modified_rejection_sampling(
                target_probs,
                draft_probs,
                draft_token_ids,
                seeded_seqs,
            )

            output_token_ids = self._create_output(
                accepted,
                recovered_token_ids,
                draft_token_ids,
            )

        if return_target_logprobs:
            return output_token_ids, target_logprobs

        return output_token_ids