import copy
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple, Type
import weakref

import torch

from vllm.chef.chef_sampler import ChefSampler
from vllm.config import ChefConfig, VllmConfig
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler,
    SpecDecodeStochasticBaseSampler,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import (
    VLLM_INVALID_TOKEN_ID,
    ExecuteModelRequest,
    HiddenStates,
    SequenceGroupMetadata,
)

from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner
from vllm.spec_decode.interfaces import (
    SpeculativeProposals,
    SpeculativeScorer,
    SpeculativeScores,
)
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.chef.station_chef_worker import StationChefWorker
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.spec_decode_worker import (
    SpecDecodeWorker,
    prepare_prefill_hidden_states,
)

from vllm.spec_decode.target_model_runner import TargetModelRunner
from vllm.spec_decode.util import (
    Timer,
    nvtx_range,
    split_batch_by_proposal_len,
)
from vllm.worker.worker import Worker
from vllm.worker.worker_base import (
    WorkerBase,
)

logger = init_logger(__name__)


def create_ensemble_spec_worker(*args, **kwargs) -> "EnsembleSpecWorker":
    """Helper method that is the entrypoint for Executors which use
    WorkerWrapper. It constructs a EnsembleSdWorker from the chef config.
    """
    vllm_config: VllmConfig = kwargs.get("vllm_config")
    ensemble_spec_config: ChefConfig = vllm_config.ensemble_decoding_config
    assert ensemble_spec_config is not None

    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError(
            "Ensemble speculative decoding is currently incompatible with pipeline parallelism"
        )

    kwargs["vllm_config"].parallel_config.worker_cls = (
        vllm_config.parallel_config.sd_worker_cls
    )

    workers = []

    for model_config, parallel_config in zip(
        ensemble_spec_config.model_configs[:-1],
        ensemble_spec_config.parallel_configs[:-1],
    ):
        new_vllm_config = copy.deepcopy(vllm_config)
        new_vllm_config.model_config = model_config
        new_vllm_config.parallel_config = parallel_config
        new_vllm_config.parallel_config.worker_cls = (
            vllm_config.parallel_config.sd_worker_cls
        )
        scorer_worker_kwargs = kwargs.copy()
        scorer_worker_kwargs.update(
            vllm_config=new_vllm_config, model_runner_cls=TargetModelRunner
        )
        scorer_worker = Worker(*args, **scorer_worker_kwargs)
        scorer_worker.model_runner.disable_logprobs = (
            ensemble_spec_config.disable_logprobs
        )
        workers.append(scorer_worker)

    draft_worker_config = copy.deepcopy(vllm_config)
    draft_worker_config.model_config = ensemble_spec_config.model_configs[-1]
    draft_worker_config.parallel_config = ensemble_spec_config.parallel_configs[-1]
    draft_worker_config.parallel_config.worker_cls = (
        vllm_config.parallel_config.sd_worker_cls
    )
    kwargs.update(model_runner_cls=TP1DraftModelRunner, vllm_config=draft_worker_config)
    workers.append(MultiStepWorker(*args, **kwargs))

    chef_decode_sampler = ChefSampler(
        ensemble_fn=ensemble_spec_config.ensemble_fn,
        ensemble_target=ensemble_spec_config.ensemble_target,
    )

    logger.info(
        "[Ensemble Speculative Decoding] Configuring with sampler=%s",
        type(chef_decode_sampler),
    )
    return EnsembleSpecWorker(
        workers,
        disable_logprobs=ensemble_spec_config.disable_logprobs,
        disable_log_stats=ensemble_spec_config.disable_log_stats,
        chef_decode_sampler=chef_decode_sampler,
        disable_by_batch_size=ensemble_spec_config.speculative_disable_by_batch_size,
        allow_zero_draft_token_step=True,
    )


# Reminder: Please update docs/source/usage/compatibility_matrix.rst
# If the feature combo become valid
class EnsembleSpecWorker(SpecDecodeWorker):
    """Worker which implements naive ensemble spculative decoding."""

    def __init__(
        self,
        workers: List[StationChefWorker],
        chef_decode_sampler: SpecDecodeBaseSampler,
        disable_logprobs: bool = False,
        disable_log_stats: bool = False,
        metrics_collector: Optional[AsyncMetricsCollector] = None,
        disable_by_batch_size: Optional[int] = None,
        allow_zero_draft_token_step: Optional[bool] = True,
    ):
        """Create a EnsembleSpecWorker.

        Args:
            workers: Workers that can both produce speculative tokens probabilities of speculative
                tokens.
            spec_decode_sampler: A Torch module used to perform acceptance
                sampling of the draft tokens in the verification step of
                speculative decoding. Currently we support two different
                types of sampler namely RejectionSampler and
                TypicalAcceptanceSampler. 'spec_decode_sampler' is either an
                instance of RejectionSampler or TypicalAcceptanceSampler.
            disable_logprobs: If set to True, token log probabilities will
                not be output in both the draft worker and the target worker.
                If set to False, log probabilities will be output by both.
            disable_log_stats: If set to True, disable periodic printing of
                speculative stage times.
            disable_by_batch_size: If the batch size is larger than this,
                disable speculative decoding for new incoming requests.
            metrics_collector: Helper class for collecting metrics; can be set
                for testing purposes.
            allow_zero_draft_token_step: whether to allow a step where the draft
                model generates no draft token; should disallow when the tp of
                draft model is larger than 1 (TODO: #5814)
        """

        super().__init__(
            proposer_worker=workers[-1],
            scorer_worker=None,
            spec_decode_sampler=chef_decode_sampler,
            disable_mqa_scorer=True,
            disable_logprobs=disable_logprobs,
            disable_log_stats=disable_log_stats,
            metrics_collector=metrics_collector,
            disable_by_batch_size=disable_by_batch_size,
            allow_zero_draft_token_step=allow_zero_draft_token_step,
        )

        self.scorer_workers = workers[:-1]
        self.workers = workers
        scorer_runner = getattr(workers[-1], "model_runner", None)
        self.generators = scorer_runner.get_generators() if scorer_runner else None

        # Lazy initialization.
        self.scorers: List[SpeculativeScorer]

        # Remove unused variables to avoid mistakenly calling functions that were not properly overridden
        del self.disable_mqa_scorer

    def init_device(self) -> None:
        for worker in self.workers:
            worker.init_device()
            worker.load_model()

        self._metrics.init_tensors(self.rank, device_type=self.device)
        self.spec_decode_sampler.init_tensors(self.rank, device_type=self.device)

        self.scorers = [
            BatchExpansionTop1Scorer(
                scorer_worker=scorer_worker,
                device=self.device,
                vocab_size=self._vocab_size,
            )
            for scorer_worker in self.scorer_workers
        ]

        self._configure_model_sampler_for_spec_decode()

    def load_model(self, *args, **kwargs):
        pass

    def _configure_model_sampler_for_spec_decode(self):
        """Configure model sampler to emit GPU tensors. This allows spec decode
        to keep data on device without transferring to CPU and serializing,
        which significantly reduces overhead of sampling during verification.

        NOTE(cade): This breaks abstraction boundaries pretty badly. The better
        design is to have the "move to CPU and serialize" sampling decision be
        done outside of the model/sampler; this way the "last-mile" worker
        object which interfaces with the scheduler can serialize and incur the
        performance hit as necessary. This allows us to run the worker several
        iterations in a row without incurring the "move to CPU and serialize"
        performance penalty.

        Since this requires a large change to vLLM, we defer it to later and
        temporarily accept this broken abstraction boundary.

        NOTE(cade): This will require a special check if the proposer worker
        does not have a sampler (e.g. ngram speculation).
        """
        self.proposer_worker.set_include_gpu_probs_tensor()
        self.proposer_worker.set_should_modify_greedy_probs_inplace()
        for scorer_worker in self.scorer_workers:
            scorer_worker.model_runner.model.sampler.include_gpu_probs_tensor = True
            scorer_worker.model_runner.model.sampler.should_modify_greedy_probs_inplace = (
                True
            )

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of cache blocks to use.

        This is done by profiling the a station chef. Then the total memory which would be used by the
        scorer cache is divided among all models' KV,
        such that the number of blocks is equal in both KV caches.
        """
        num_gpu_blocks, num_cpu_blocks = self.workers[
            0
        ].determine_num_available_blocks()

        total_memory = self.workers[0].get_cache_block_size_bytes() * num_gpu_blocks

        new_num_gpu_blocks = total_memory // sum(
            worker.get_cache_block_size_bytes() for worker in self.workers
        )
        return new_num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Initialize the cache engine of all workers."""
        for worker in self.workers:
            worker.initialize_cache(
                num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=num_cpu_blocks
            )

    @nvtx_range("ensemble_spec_worker._run_no_spec")
    def _run_no_spec(
        self, execute_model_req: ExecuteModelRequest, skip_proposer: bool
    ) -> List[SamplerOutput]:
        """Run a single generation step without any speculation. The input is
        sent to the proposer and scorer model so that the KV cache is consistent
        between the two. When skip_proposer is True, the proposer model is
        not called, meaning that the kv-cache in proposer for requests is not
        updated, so they cannot enable spec decode in the rest decoding.
        """

        sampler_output = self.workers[0].execute_model(execute_model_req)
        assert len(sampler_output) == 1
        sampler_output = sampler_output[0]

        # Store hidden states from target model execution.
        hidden_states = sampler_output.hidden_states
        if hidden_states is not None:
            # remove hidden_states for prompt tokens
            # TODO Enable `return_hidden_states`: prefill chunks hidden states
            # are pruned by the logits processor. Also, they should be arranged
            # back into full-prefill latent. Address it to enable MLPSpeculator.
            if any(seq.is_prompt for seq in execute_model_req.seq_group_metadata_list):
                hidden_states = hidden_states[
                    torch.where(
                        sampler_output.sampled_token_ids - VLLM_INVALID_TOKEN_ID
                    )[0]
                ]
            if self.previous_hidden_states is None:
                self.previous_hidden_states = HiddenStates(
                    hidden_states, execute_model_req.seq_group_metadata_list
                )
            else:
                self.previous_hidden_states.update(
                    hidden_states, execute_model_req.seq_group_metadata_list
                )

        if not skip_proposer:
            # We prepare the prefill hidden states here so that there no
            # additional complexity in worker for spec_decode vs non_spec_decode
            # flow and execute_model doesn't need additional modifications.
            execute_model_req.previous_hidden_states = prepare_prefill_hidden_states(
                sampler_output.prefill_hidden_states
            )

            sample_logits = []

            for worker in self.workers[1:]:
                station_chef_output = worker.execute_model(execute_model_req)
                assert len(station_chef_output) == 1
                sample_logits.append(station_chef_output[0].logits)
            sample_logits.append(sampler_output.logits)

            # directly sample from ensemble distribution, since we only have one token
            sampler_output = self.spec_decode_sampler.sampler(
                sample_logits,
                self.scorers[-1]
                ._scorer_worker.prepare_input(execute_model_req)[0]
                .sampling_metadata,
            )

        sampler_output_to_return = (
            self._serialize_sampler_output_no_logprobs(
                execute_model_req=execute_model_req, sampler_output=sampler_output
            )
            if self._disable_logprobs
            else [sampler_output]
        )

        # Clear device tensors from sampler output. This reduces communication
        # overhead when the engine runs in a different process than the workers.
        sampler_output.sampled_token_probs = None
        sampler_output.sampled_token_ids = None
        sampler_output.logprobs = None
        return sampler_output_to_return

    def _run_non_driver_rank(self) -> bool:
        """Run proposer and verifier model in non-driver workers. This is used
        for both speculation cases (num_lookahead_slots>0) and non-speculation
        cases (e.g. prefill).

        Returns True if there are remaining sequences to process.
        """
        assert self.rank != self._driver_rank

        data = broadcast_tensor_dict(src=self._driver_rank)
        if not data:
            return False
        num_lookahead_slots = data["num_lookahead_slots"]

        # In case of prefill, scorer_worker has to be run before proposer so
        # that the hidden states can be propagated to proposer when needed.
        if data["no_spec"]:
            for worker in self.scorer_workers:
                worker.execute_model()

        if not data["disable_all_speculation"]:
            # Even if num_lookahead_slots is zero, we want to run the
            # proposer model as it may have KV.
            #
            # We run the proposer once per lookahead slot. In the future we
            # should delegate how many times it runs to the proposer.
            for _ in range(max(num_lookahead_slots, 1)):
                self.proposer_worker.execute_model()

        if not data["no_spec"]:
            for worker in self.scorer_workers:
                worker.execute_model()
            if data["run_spec_proposer_for_prefill"]:
                self.proposer_worker.execute_model()

        return True

    @nvtx_range("ensemble_spec_worker._run_speculative_decoding_step")
    def _run_speculative_decoding_step(
        self, execute_model_req: ExecuteModelRequest, num_lookahead_slots: int
    ) -> List[SamplerOutput]:
        """Execute a single step of speculative decoding.

        This invokes the proposer worker to get k speculative tokens for each
        sequence, then scores each speculative token using the scoring worker.

        When `enable_chunked_prefill` is set, scorer will batch decodes and
        prefills, while proposer will sync its KV-cache by running an extra
        forward on prefills.

        Returns a list of SamplerOutput, each containing a single token per
        sequence.
        """

        def prefill_seq_if_need(
            seq_group_metadata_list: List[SequenceGroupMetadata],
            proposer: StationChefWorker,
            proposal_lens: torch.Tensor,
            proposal_scores: SpeculativeScores,
        ):
            """
            Prefill sequences that run in non-speculative decoding.
            """
            _, (non_spec_seqs, non_spec_indices) = split_batch_by_proposal_len(
                seq_group_metadata_list, proposal_lens
            )
            # With prefill chunking enabled, `non_spec_seqs` contains prefills too:
            # discard decodes that have already been processed by proposer.
            non_spec_indices = [
                idx
                for idx in non_spec_indices
                if seq_group_metadata_list[idx].is_prompt
            ]
            if len(non_spec_indices):
                all_hidden_states = proposal_scores.hidden_states
                # TODO fix `return_hidden_states`, same as in `_run_no_spec`
                if all_hidden_states is not None:
                    prefill_hidden_states = all_hidden_states[non_spec_indices]
                    execute_model_req.previous_hidden_states = (
                        prepare_prefill_hidden_states(prefill_hidden_states)
                    )
                # Sync proposer KV cache for prefills.
                prefill_req = execute_model_req.clone(non_spec_seqs)
                proposer.execute_model(prefill_req)

        # With prefill chunking, expect requests to have prompts first
        # so that backend gets prefill|decode.
        assert num_lookahead_slots == execute_model_req.num_lookahead_slots

        # Pass last hidden states from target model to proposer
        execute_model_req.previous_hidden_states = self.previous_hidden_states
        self.previous_hidden_states = None

        with Timer() as proposal_timer:
            # Generate proposals using draft worker.
            proposals = self.proposer_worker.get_spec_proposals(
                execute_model_req, self._seq_with_bonus_token_in_last_step
            )

        if not self._allow_zero_draft_token_step and proposals.no_proposals:
            # TODO: Fix it #5814
            raise RuntimeError(
                "Cannot handle cases where distributed draft "
                "workers generate no tokens"
            )

        execute_model_req.previous_hidden_states = None

        with Timer() as scoring_timer:
            proposal_scores_list = [
                scorer.score_proposals(
                    execute_model_req,
                    proposals,
                )
                for scorer in self.scorers
            ]

        prefill_seq_if_need(
            execute_model_req.seq_group_metadata_list,
            self.proposer_worker,
            proposals.proposal_lens,
            proposal_scores_list[0],
        )

        sampling_metadata = (
            self.scorers[-1]
            ._scorer_worker.prepare_input(execute_model_req)[0]
            .sampling_metadata
        )

        verifi_seq_logits = [proposals.proposal_logits]
        verifi_seq_logits.extend(
            proposal_scores.logits for proposal_scores in proposal_scores_list
        )

        with Timer() as verification_timer:
            accepted_token_ids, target_logprobs = self._verify_tokens(
                execute_model_req.seq_group_metadata_list,
                verifi_seq_logits,
                proposals,
                sampling_metadata,
            )

        stage_times = (
            proposal_timer.elapsed_time_ms / num_lookahead_slots,
            scoring_timer.elapsed_time_ms,
            verification_timer.elapsed_time_ms,
        )

        return self._create_output_sampler_list(
            execute_model_req.seq_group_metadata_list,
            accepted_token_ids,
            target_logprobs=target_logprobs,
            k=execute_model_req.num_lookahead_slots,
            stage_times=stage_times,
        )

    @nvtx_range("ensemble_spec_worker._verify_tokens")
    def _verify_tokens(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        verifi_seq_logits: List[torch.Tensor],
        draft_proposals: SpeculativeProposals,
        sampling_metadata: SamplingMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine which speculative tokens are accepted using the
        probabilities of each token according to the proposer and scorer models.

        Returns a tuple of Tensors, one for the accepted token ids and one for
        the logprobs according to the scoring model.
        """

        # vLLM currently only supports proposal lens equal to zero or the batch
        # proposal len. This adds some complexity (splitting the batch into spec
        # and non spec sequences) and should be removed in the future. It can be
        # done by supporting per-sequence proposal lens.
        (_, spec_indices), (_, non_spec_indices) = split_batch_by_proposal_len(
            seq_group_metadata_list, draft_proposals.proposal_lens.tolist()
        )
        original_indices = spec_indices + non_spec_indices

        # select sequences that used in the speculative decoding
        station_chef_logits = [logits[spec_indices] for logits in verifi_seq_logits]
        draft_probs = draft_proposals.proposal_probs[spec_indices]
        draft_token_ids = draft_proposals.proposal_token_ids[spec_indices]

        # Sampler arguments
        sampler_extra_kwargs: Dict[str, Any] = {}
        if self.generators and isinstance(
            self.spec_decode_sampler, SpecDecodeStochasticBaseSampler
        ):
            sampler_extra_kwargs["seeded_seqs"] = {
                idx: self.generators[sgm.request_id]
                for idx, sgm in enumerate(seq_group_metadata_list)
                if sgm.sampling_params.seed is not None
            }

        accepted_token_ids, target_logprobs = self.spec_decode_sampler(
            station_chef_logits=station_chef_logits,
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            sampling_metadata=sampling_metadata,
            return_target_logprobs=True,
            **sampler_extra_kwargs,
        )

        # Append output tokens from non-speculative sequences to
        # the accepted token ids tensor.
        max_proposal_len = torch.max(draft_proposals.proposal_lens).item()
        non_spec_token_ids = draft_proposals.proposal_token_ids[non_spec_indices]
        non_spec_token_ids = non_spec_token_ids.expand(-1, max_proposal_len).clone()
        non_spec_token_ids[:, 1:] = -1
        accepted_token_ids = torch.cat([accepted_token_ids, non_spec_token_ids])
        # Rearrange so that results are in the order of the original seq group
        # metadata.
        accepted_token_ids[original_indices] = accepted_token_ids.clone()

        return accepted_token_ids, target_logprobs

    @cached_property
    def _vocab_size(self) -> int:
        """Get the vocab size of the model and make sure it's consistent between
        all workers.
        """
        vocab_sizes = set(worker.vocab_size for worker in self.workers)
        assert len(vocab_sizes) == 1
        return vocab_sizes.pop()

    @property
    def rank(self):
        rank = set(worker.rank for worker in self.workers)
        assert len(rank) == 1
        return rank.pop()

    @property
    def device(self):
        return self.workers[0].device

    @property
    def _driver_rank(self) -> int:
        return 0

    def start_profile(self):
        if isinstance(self.workers[-1], WorkerBase):
            self.workers[-1].start_profile()

    def stop_profile(self):
        if isinstance(self.workers[-1], WorkerBase):
            self.workers[-1].stop_profile()
