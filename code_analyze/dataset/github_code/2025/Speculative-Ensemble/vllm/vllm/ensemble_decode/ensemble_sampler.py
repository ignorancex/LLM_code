from typing import Callable, List, Literal, Optional
import torch
from vllm.model_executor.layers.sampler import SamplerOutput, Sampler
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SamplingTensors)
from vllm.model_executor.layers.sampler import (
    _apply_min_tokens_penalty, _apply_penalties, _apply_top_k_top_p,
    _apply_min_p, get_logprobs, SampleResultArgsType, _build_sampler_output,
    _sample, flashinfer_top_k_top_p_sampling)


class EnsembleSampler(Sampler):

    def __init__(self, ensemble_fn: Callable,
                 ensemble_target: Literal["raw-logits", "logits", "probs"]):
        super().__init__()
        self.ensemble_fn = ensemble_fn
        self.ensemble_target = ensemble_target

    def _init_sampling_tensors(self, logits: List[torch.Tensor],
                               sampling_metadata):
        """The goal here is to reuse sampling tensors between similar decode
        runs. This is possible because sampling logic does not change between
        decodes of the same sequences.
        """
        _, vocab_size = logits[0].shape

        # First free any existing stored sampling tensors.
        # This is necessary because some sampling tensors may
        # have pinned memory.
        self._sampling_tensors = None

        # Initialize new sampling tensors

        self._sampling_tensors = []
        for logit in logits:
            (sampling_tensors, do_penalties, do_top_p_top_k,
             do_min_p) = SamplingTensors.from_sampling_metadata(
                 sampling_metadata, vocab_size, logit.device, logit.dtype)
            self._sampling_tensors.append(sampling_tensors)

        self._do_penalties = do_penalties
        self._do_top_p_top_k = do_top_p_top_k
        self._do_min_p = do_min_p

    def forward(
        self,
        logits: List[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """
            Single-step scheduling:
            * Perform GPU-side sampling computation & compute
            GPU-side logprobs tensor
            * Pythonize sampling result & logprobs tensor

            Multi-step scheduling:
            * Perform GPU-side sampling computation & compute
            GPU-side logprobs tensor
            * Defer Pythonization of sampling result & logprobs
            tensor
            * Encapsulate arguments required for deferred Pythonization
            in the :class:`SamplerOutput` structure

            Args:
                logits: (num_tokens, vocab_size).
                sampling_metadata: Metadata for sampling.
            """
        assert logits is not None
        seq_len, vocab_size = logits[0].shape

        # Prepare sampling tensors with pinned memory to avoid blocking.
        if not sampling_metadata.reuse_sampling_tensors:
            self._init_sampling_tensors(logits, sampling_metadata)
        elif self._do_penalties:
            # In this case, the sampling tensors logic depends on
            # "output_tokens" of a sequence. As a result, we cannot
            # reuse sampling tensors, since "output_tokens" changes
            # between decode runs.
            self._init_sampling_tensors(logits, sampling_metadata)

        assert self._sampling_tensors is not None
        sampling_tensors = self._sampling_tensors
        do_penalties = self._do_penalties
        do_top_p_top_k = self._do_top_p_top_k
        do_min_p = self._do_min_p

        def process_single_logits(logits: torch.Tensor,
                                  sampling_tensors) -> torch.Tensor:
            logits = _apply_min_tokens_penalty(logits, sampling_metadata)

            # Apply presence and frequency penalties.
            if do_penalties:
                logits = _apply_penalties(
                    logits, sampling_tensors.prompt_tokens,
                    sampling_tensors.output_tokens,
                    sampling_tensors.presence_penalties,
                    sampling_tensors.frequency_penalties,
                    sampling_tensors.repetition_penalties)

            # Use float32 to apply temperature scaling.
            # Use in-place division to avoid creating a new tensor.
            logits = logits.to(torch.float)
            logits.div_(sampling_tensors.temperatures.unsqueeze(dim=1))

            if do_top_p_top_k and flashinfer_top_k_top_p_sampling is None:
                logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps,
                                            sampling_tensors.top_ks)

            if do_min_p:
                logits = _apply_min_p(logits, sampling_tensors.min_ps)
            return logits

        if self.ensemble_target == "probs":
            logits = [
                process_single_logits(logits[idx], sampling_tensors[idx])
                for idx in range(len(logits))
            ]
            probs = [
                torch.softmax(logits[idx], dim=-1, dtype=torch.float)
                for idx in range(len(logits))
            ]
            logprobs = [
                torch.log_softmax(logits[idx], dim=-1, dtype=torch.float)
                for idx in range(len(logits))
            ]
            probs, logprobs = self.ensemble_fn(probs, logprobs)
        elif self.ensemble_target == "raw-logits":
            logits = process_single_logits(self.ensemble_fn(logits), sampling_tensors[0])
            probs = torch.softmax(logits, dim=-1, dtype=torch.float)
            logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        else:
            logits = [
                process_single_logits(logits[idx], sampling_tensors[idx])
                for idx in range(len(logits))
            ]
            logits = self.ensemble_fn(logits)
            probs = torch.softmax(logits, dim=-1, dtype=torch.float)
            logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        assert probs.shape == logprobs.shape == torch.Size(
            [seq_len, vocab_size])

        # Sample the next tokens.
        maybe_deferred_sample_results, maybe_sampled_tokens_tensor = _sample(
            probs,
            logprobs,
            sampling_metadata,
            sampling_tensors,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
            modify_greedy_probs=self._should_modify_greedy_probs_inplace,
        )

        if self.include_gpu_probs_tensor:
            # Since we will defer sampler result Pythonization,
            # preserve GPU-side tensors in support of later
            # deferred pythonization of logprobs
            assert maybe_sampled_tokens_tensor is not None
            on_device_tensors = (probs, logprobs, maybe_sampled_tokens_tensor)
        else:
            # Since Pythonization has already happened, don't preserve
            # GPU-side tensors.
            on_device_tensors = None

        # Get the logprobs query results.
        prompt_logprobs = None
        sample_logprobs = None
        if not sampling_metadata.skip_sampler_cpu_output:
            # Pythonize logprobs now (GPU -> CPU); do not defer.
            assert not isinstance(maybe_deferred_sample_results,
                                  SampleResultArgsType)
            prompt_logprobs, sample_logprobs = get_logprobs(
                logprobs, sampling_metadata, maybe_deferred_sample_results)

        sampler_output = _build_sampler_output(
            maybe_deferred_sample_results,
            sampling_metadata,
            prompt_logprobs,
            sample_logprobs,
            on_device_tensors=on_device_tensors,
            skip_sampler_cpu_output=sampling_metadata.skip_sampler_cpu_output)
        sampler_output.processed_logits = logits
        return sampler_output
