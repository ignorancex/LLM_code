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
    CompletionSequenceGroupOutput,
    ExecuteModelRequest,
    HiddenStates,
    SequenceGroupMetadata,
    get_all_seq_ids_and_request_ids,
)

from vllm.spec_decode.interfaces import (
    SpeculativeProposals,
    SpeculativeScores,
)
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.chef.station_chef_worker import StationChefWorker
from vllm.spec_decode.spec_decode_worker import (
    SpecDecodeWorker,
    prepare_prefill_hidden_states,
)
from vllm.chef.station_chef_runner import StationChefRunner, StationChefStage
from vllm.spec_decode.util import (
    create_sequence_group_output,
    get_all_num_logprobs,
    nvtx_range,
    split_batch_by_proposal_len,
)
from vllm.worker.worker_base import (
    WorkerBase,
)

logger = init_logger(__name__)


def create_chef_worker(*args, **kwargs) -> "ChefWorker":
    """Helper method that is the entrypoint for Executors which use
    WorkerWrapper. It constructs a ChefWorker from the chef config.
    """
    vllm_config: VllmConfig = kwargs.get("vllm_config")
    chef_config: ChefConfig = vllm_config.chef_config
    assert chef_config is not None

    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError(
            "Chef decoding is currently incompatible with pipeline parallelism"
        )

    kwargs["model_runner_cls"] = StationChefRunner
    kwargs["vllm_config"].parallel_config.worker_cls = (
        vllm_config.parallel_config.sd_worker_cls
    )
    kwargs["num_speculative_tokens"] = chef_config.num_speculative_tokens[0]

    workers = [StationChefWorker(*args, **kwargs)]

    for model_config, parallel_config, num_speculative_tokens in zip(
        chef_config.model_configs[1:],
        chef_config.parallel_configs[1:],
        chef_config.num_speculative_tokens[1:],
    ):
        new_vllm_config = copy.deepcopy(vllm_config)
        new_vllm_config.model_config = model_config
        new_vllm_config.parallel_config = parallel_config
        new_vllm_config.parallel_config.worker_cls = (
            vllm_config.parallel_config.sd_worker_cls
        )
        station_worker_kwargs = kwargs.copy()
        station_worker_kwargs.update(
            vllm_config=new_vllm_config, num_speculative_tokens=num_speculative_tokens
        )
        workers.append(StationChefWorker(**station_worker_kwargs))

    for worker in workers:
        worker.model_runner.disable_logprobs = chef_config.disable_logprobs
        worker.model_runner.num_speculative_tokens = worker.num_speculative_tokens

    chef_decode_sampler = ChefSampler(
        ensemble_target=chef_config.ensemble_target, ensemble_fn=chef_config.ensemble_fn
    )

    logger.info(
        "[Chef Decoding] Configuring with sampler=%s",
        type(chef_decode_sampler),
    )
    return ChefWorker(
        workers,
        disable_logprobs=chef_config.disable_logprobs,
        disable_log_stats=chef_config.disable_log_stats,
        chef_decode_sampler=chef_decode_sampler,
        disable_by_batch_size=chef_config.speculative_disable_by_batch_size,
        allow_zero_draft_token_step=True,
    )


# Reminder: Please update docs/source/usage/compatibility_matrix.rst
# If the feature combo become valid
class ChefWorker(SpecDecodeWorker):
    """Worker which implements chef decoding."""

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
        """Create a ChefWorker.

        Args:
            workers: Workers that can both produce speculative tokens probabilities of speculative
                tokens.
            chef_decode_sampler: A Torch module used to perform acceptance
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

        # Here, sorting by num_speculative_tokens in descending order is necessary
        # We assume that the descending order of num_speculative_tokens is the ascending order of model size.
        # For example, in contrastive decoding, the sorted workers[-1] is target model
        # and the sorted workers[0] is the draft model.
        self.workers = sorted(
            workers, key=lambda worker: worker.num_speculative_tokens, reverse=True
        )

        # In ensemble_fn, the input logits are ordered as [model, extra_model1, extra_model2, ...]
        # but `sorted` above breaks this order. Therefore, we need to use _apply_rev_sorted_order to restore
        # the logits generated from workers to the original order and use them as the sampler input.
        sorted_indices = sorted(
            range(len(workers)),
            key=lambda x: workers[x].num_speculative_tokens,
            reverse=True,
        )
        reverse_map = {sorted_indices[i]: i for i in range(len(sorted_indices))}
        self._apply_rev_sorted_order = lambda inputs: [
            inputs[reverse_map[i]] for i in range(len(inputs))
        ]

        super().__init__(
            proposer_worker=None,
            scorer_worker=None,
            spec_decode_sampler=chef_decode_sampler,
            disable_mqa_scorer=True,
            disable_logprobs=disable_logprobs,
            disable_log_stats=disable_log_stats,
            metrics_collector=metrics_collector,
            disable_by_batch_size=disable_by_batch_size,
            allow_zero_draft_token_step=allow_zero_draft_token_step,
        )

        scorer_runner = getattr(self.workers[-1], "model_runner", None)
        self.generators = scorer_runner.get_generators() if scorer_runner else None

        # Remove unused variables to avoid mistakenly calling functions that were not properly overridden
        del self.proposer_worker
        del self.scorer_worker
        del self.disable_mqa_scorer
        del self._metrics

        # CHEF parameters
        self.spec_cache: weakref.WeakKeyDictionary[
            StationChefWorker, Optional[torch.Tensor]
        ] = weakref.WeakKeyDictionary({worker: None for worker in self.workers})
        self.current_seq_proposals: Optional[SpeculativeProposals] = None

    def _clear_spec_cache(self):
        for worker in self.spec_cache:
            self.spec_cache[worker] = None
        self.current_seq_proposals = None

    def get_current_worker(self):
        """Get the worker that doesn't have cached proposals."""
        for worker in self.workers:
            if self.spec_cache[worker] is None:
                return worker

        # never reach here
        raise RuntimeError("all workers have cached proposals, but one should not have")

    def init_device(self) -> None:
        for worker in self.workers:
            worker.init_device()
            worker.load_model()

        self.spec_decode_sampler.init_tensors(self.rank, device_type=self.device)

        for worker in self.workers:
            worker.init_scorer(device=self.device, vocab_size=self._vocab_size)

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
        for worker in self.workers:
            worker.set_include_gpu_probs_tensor()
            worker.set_should_modify_greedy_probs_inplace()

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

    @nvtx_range("chef_decode_worker._run_no_spec")
    def _run_no_spec(
        self, execute_model_req: ExecuteModelRequest, skip_proposer: bool
    ) -> List[SamplerOutput]:
        """Run a single generation step without any speculation. The input is
        sent to the proposer and scorer model so that the KV cache is consistent
        between the two. When skip_proposer is True, the proposer model is
        not called, meaning that the kv-cache in proposer for requests is not
        updated, so they cannot enable spec decode in the rest decoding.
        """

        # clear cached proposals
        self._clear_spec_cache()
        for worker in self.workers:
            worker.stage = StationChefStage.PROPOSAL
        sampler_output = self.workers[-1].execute_model(execute_model_req)
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

            for worker in self.workers[:-1]:
                station_chef_output = worker.execute_model(execute_model_req)
                assert len(station_chef_output) == 1
                sample_logits.append(station_chef_output[0].logits)
            sample_logits.append(sampler_output.logits)

            # directly sample from ensemble distribution, since we only have one token
            sampler_output = self.spec_decode_sampler.sampler(
                self._apply_rev_sorted_order(sample_logits),
                self.workers[-1]
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

        # In case of prefill, scorer_worker has to be run before proposer so
        # that the hidden states can be propagated to proposer when needed.
        if data["no_spec"]:
            self.workers[-1].execute_model()

        if not data["disable_all_speculation"]:
            # Even if num_lookahead_slots is zero, we want to run the
            # proposer model as it may have KV.
            #
            # We run the proposer once per lookahead slot. In the future we
            # should delegate how many times it runs to the proposer.
            for worker in self.workers[:-1]:
                for _ in range(max(worker.num_lookahead_slots, 1)):
                    worker.execute_model()

        if not data["no_spec"]:
            self.workers[-1].execute_model()
            if data["run_spec_proposer_for_prefill"]:
                for worker in self.workers[:-1]:
                    worker.execute_model()

        return True

    @nvtx_range("chef_decode_worker._run_speculative_decoding_step")
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

        def score_current_seq(
            worker: StationChefWorker,
            num_verifi_tokens: int,
        ) -> torch.Tensor:
            """
            Score the current proposals with worker
            Returns:
                verifi_seq_logits: The logits of the sequences to be verified.
            """
            # scoring proposals, this step generated proposal_lens + 1 logits and tokens
            proposal_scores = worker.score_proposals(
                execute_model_req, self.current_seq_proposals
            )
            verifi_seq_logits = proposal_scores.logits[:, :num_verifi_tokens, :]
            prefill_seq_if_need(
                execute_model_req.seq_group_metadata_list,
                worker,
                self.current_seq_proposals.proposal_lens,
                proposal_scores,
            )

            # update cache
            previous_cache = (
                self.spec_cache[worker]
                if self.spec_cache[worker] is not None
                else torch.tensor([], device=self.device)
            )
            self.spec_cache[worker] = torch.cat(
                [previous_cache, proposal_scores.logits[:, num_verifi_tokens:, :]],
                dim=1,
            )

            bonus_token_logits = proposal_scores.logits[:, -1:, :]
            bonus_token_logprobs = proposal_scores.logprobs[:, -1:, :]
            bonus_token_probs = proposal_scores.probs[:, -1:, :]
            bonus_token_ids = proposal_scores.token_ids[:, -1:]

            # Append bonus tokens to the proposals
            self.current_seq_proposals = SpeculativeProposals(
                proposal_lens=self.current_seq_proposals.proposal_lens + 1,
                proposal_logits=torch.cat(
                    [self.current_seq_proposals.proposal_logits, bonus_token_logits],
                    dim=1,
                ),
                proposal_logprobs=torch.cat(
                    [
                        self.current_seq_proposals.proposal_logprobs,
                        bonus_token_logprobs,
                    ],
                    dim=1,
                ),
                proposal_probs=torch.cat(
                    [self.current_seq_proposals.proposal_probs, bonus_token_probs],
                    dim=1,
                ),
                proposal_token_ids=torch.cat(
                    [self.current_seq_proposals.proposal_token_ids, bonus_token_ids],
                    dim=1,
                ),
            )

            return verifi_seq_logits

        def spec_on_current_seq(
            worker: StationChefWorker,
        ) -> SpeculativeProposals:
            """Generate new proposals based on the current sequence proposals."""

            new_proposals = worker.get_spec_proposals(
                execute_model_req,
                {},
                self.current_seq_proposals,
                with_bonus_token=True,
            )

            # update cache
            previous_cache = (
                self.spec_cache[worker]
                if self.spec_cache[worker] is not None
                else torch.tensor([], device=self.device)
            )
            self.spec_cache[worker] = torch.cat(
                [previous_cache, new_proposals.proposal_logits], dim=1
            )

            self.current_seq_proposals = SpeculativeProposals(
                proposal_lens=self.current_seq_proposals.proposal_lens
                + new_proposals.proposal_lens,
                proposal_logits=torch.cat(
                    [
                        self.current_seq_proposals.proposal_logits,
                        new_proposals.proposal_logits,
                    ],
                    dim=1,
                ),
                proposal_probs=torch.cat(
                    [
                        self.current_seq_proposals.proposal_probs,
                        new_proposals.proposal_probs,
                    ],
                    dim=1,
                ),
                proposal_token_ids=torch.cat(
                    [
                        self.current_seq_proposals.proposal_token_ids,
                        new_proposals.proposal_token_ids,
                    ],
                    dim=1,
                ),
                proposal_logprobs=torch.cat(
                    [
                        self.current_seq_proposals.proposal_logprobs,
                        new_proposals.proposal_logprobs,
                    ],
                    dim=1,
                ),
            )

            return new_proposals

        # With prefill chunking, expect requests to have prompts first
        # so that backend gets prefill|decode.
        assert num_lookahead_slots == execute_model_req.num_lookahead_slots

        # Pass last hidden states from target model to proposer
        execute_model_req.previous_hidden_states = self.previous_hidden_states
        if self.previous_hidden_states is not None:
            raise NotImplementedError(
                "!!! CHEF not supported for self.previous_hidden_states is not None !!!"
            )
        self.previous_hidden_states = None

        verifi_seq_logits = (
            []
        )  # logits of the sequences from all workers to be verified
        current_worker = None  # the worker that doesn't have cached proposals
        verifi_proposals = None  # proposals from current_worker used as draft to verify
        num_verifi_tokens = None  # number of tokens to verify in this step

        if all(v is None for v in self.spec_cache.values()):
            # the first chef process, we iterate all models to obtain initial cache proposals
            current_worker = self.workers[0]
            verifi_proposals = self.current_seq_proposals = (
                current_worker.get_spec_proposals(execute_model_req, {})
            )
            num_verifi_tokens = current_worker.num_speculative_tokens
            verifi_seq_logits.append(self.current_seq_proposals.proposal_logits)

            for worker in self.workers[1:]:
                verifi_logits = score_current_seq(
                    worker, num_verifi_tokens=num_verifi_tokens
                )
                verifi_seq_logits.append(verifi_logits)
                # for the last worker, we generate new proposals only after
                # all tokens are accepted in the verification.
                if worker is not self.workers[-1]:
                    spec_on_current_seq(worker)

            # we set the current_worker to the last worker, because we need to
            # run generate_on_proposals on this worker if all tokens are accepted
            current_worker = self.workers[-1]
        else:
            # in next steps, we find the shortest and longest proposal
            # and verify the shortest proposal to get the final prediction
            shortest_worker = min(
                filter(lambda w: self.spec_cache[w] is not None, self.workers),
                key=lambda w: self.spec_cache[w].size(1),
            )

            current_worker = self.get_current_worker()
            assert current_worker is not shortest_worker

            num_verifi_tokens = self.spec_cache[shortest_worker].size(1)
            verifi_logits = score_current_seq(
                current_worker,
                num_verifi_tokens=num_verifi_tokens,
            )

            for worker in self.workers:
                if worker is current_worker:
                    verifi_seq_logits.append(verifi_logits)
                else:
                    if worker is shortest_worker:
                        # extract proposals of shortest_worker
                        verifi_proposals = SpeculativeProposals(
                            proposal_lens=torch.tensor(
                                [num_verifi_tokens]
                                * self.current_seq_proposals.proposal_logits.size(0),
                                device=self.device,
                                dtype=torch.long,
                            ),
                            proposal_logits=self.current_seq_proposals.proposal_logits[
                                :, :num_verifi_tokens, :
                            ],
                            proposal_probs=self.current_seq_proposals.proposal_probs[
                                :, :num_verifi_tokens, :
                            ],
                            proposal_token_ids=self.current_seq_proposals.proposal_token_ids[
                                :, :num_verifi_tokens
                            ],
                            proposal_logprobs=self.current_seq_proposals.proposal_logprobs[
                                :, :num_verifi_tokens, :
                            ],
                        )
                    # extract num_verifi_tokens logits from the cache
                    verifi_seq_logits.append(
                        self.spec_cache[worker][:, :num_verifi_tokens, :]
                    )
                    if num_verifi_tokens == self.spec_cache[worker].size(1):
                        self.spec_cache[worker] = None
                    else:
                        self.spec_cache[worker] = self.spec_cache[worker][
                            :, num_verifi_tokens:, :
                        ]

        sampling_metadata = (
            self.workers[-1]
            ._scorer_worker.prepare_input(execute_model_req)[0]
            .sampling_metadata
        )
        accepted_token_ids, target_logprobs = self._verify_tokens(
            execute_model_req.seq_group_metadata_list,
            verifi_seq_logits=verifi_seq_logits,
            draft_proposals=verifi_proposals,
            sampling_metadata=sampling_metadata,
        )

        if torch.any(accepted_token_ids != verifi_proposals.proposal_token_ids):
            # there are some tokens are rejected, we need to clean the cache
            self._clear_spec_cache()
        else:
            # all tokens are accepted, we need to remove tokens in proposals that
            # are already verified, and generate new proposals
            self.current_seq_proposals = SpeculativeProposals(
                proposal_lens=self.current_seq_proposals.proposal_lens
                - num_verifi_tokens,
                proposal_logits=self.current_seq_proposals.proposal_logits[
                    :, num_verifi_tokens:, :
                ],
                proposal_probs=self.current_seq_proposals.proposal_probs[
                    :, num_verifi_tokens:, :
                ],
                proposal_token_ids=self.current_seq_proposals.proposal_token_ids[
                    :, num_verifi_tokens:
                ],
                proposal_logprobs=self.current_seq_proposals.proposal_logprobs[
                    :, num_verifi_tokens:, :
                ],
            )
            spec_on_current_seq(current_worker)

        return self._create_output_sampler_list(
            execute_model_req.seq_group_metadata_list,
            accepted_token_ids,
            target_logprobs=target_logprobs,
        )

    @nvtx_range("chef_decode_worker._verify_tokens")
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
        station_chef_logits = [
            logits[spec_indices]
            for logits in self._apply_rev_sorted_order(verifi_seq_logits)
        ]
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

    def _create_output_sampler_list(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        accepted_token_ids: torch.Tensor,  # shape: [batch_size, k+1]
        target_logprobs: torch.Tensor,  # shape: [batch_size, k+1, vocab_size]
    ) -> List[SamplerOutput]:
        """Given the accepted token ids, create a list of SamplerOutput.

        The output is padded with -1 tokens such that each sequence has
        the same number of outputs.
        """
        batch_size, num_steps = accepted_token_ids.shape
        accepted_token_ids_by_step = accepted_token_ids.transpose(0, 1)
        if self._disable_logprobs:
            # We are skipping the logprobs. Hence don't serialize the
            # logprobs related tensors from the GPU. Instead create
            # empty/dummy lists.
            (
                accepted_token_id_ranks_by_step,
                accepted_token_id_logprobs_by_step,
                topk_logprobs_by_step,
                topk_indices_by_step,
            ) = self._create_dummy_logprob_lists(
                batch_size, num_steps, self.workers[-1].model_config.max_logprobs
            )
        else:
            # Organize input tensors by step instead of by sequence.
            target_logprobs_by_step = target_logprobs.transpose(0, 1)
            # Serialize all tensors into Python lists.
            (
                accepted_token_id_ranks_by_step,
                accepted_token_id_logprobs_by_step,
                topk_logprobs_by_step,
                topk_indices_by_step,
            ) = self._create_logprob_lists_from_tensors(
                target_logprobs_by_step,
                accepted_token_ids_by_step,
                self.workers[-1].model_config.max_logprobs,
            )

        # Get the sequence ids and num_logprobs (sampling parameter) in the
        # batch.
        seq_ids, request_ids_seq_ids_mapping = get_all_seq_ids_and_request_ids(
            seq_group_metadata_list
        )

        num_logprobs_per_seq = get_all_num_logprobs(seq_group_metadata_list)

        # Serialize tensor to CPU Python list.
        accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()

        # Construct the output on a per-step, per-sequence basis.
        # Non-terminal prefill chunks will end up here as rows with just -1s
        # i.e mixed-batch [[-1, 1576], [-1, 29884], [-1, -1], [-1, -1]]
        sampler_output_list: List[SamplerOutput] = []
        for step_index in range(num_steps):
            if all(
                token_id == -1 for token_id in accepted_token_ids_by_step[step_index]
            ):
                break

            step_output_token_ids: List[CompletionSequenceGroupOutput] = []
            for sequence_index in range(batch_size):
                # Each sequence may have a different num_logprobs; retrieve it.
                num_logprobs = num_logprobs_per_seq[sequence_index]
                step_output_token_ids.append(
                    create_sequence_group_output(
                        token_id=accepted_token_ids_by_step[step_index][sequence_index],
                        token_id_logprob_rank=accepted_token_id_ranks_by_step[
                            step_index
                        ][sequence_index],
                        token_id_logprob=accepted_token_id_logprobs_by_step[step_index][
                            sequence_index
                        ],
                        seq_id=seq_ids[sequence_index],
                        topk_token_ids=topk_indices_by_step[step_index][sequence_index][
                            :num_logprobs
                        ],
                        topk_logprobs=topk_logprobs_by_step[step_index][sequence_index][
                            :num_logprobs
                        ],
                    )
                )
            sampler_output_list.append(SamplerOutput(outputs=step_output_token_ids))

        # Populate the data structures needed to keep track of sequences with
        # bonus tokens.
        self._track_sequences_with_bonus_tokens(
            seq_ids, request_ids_seq_ids_mapping, accepted_token_ids_by_step
        )
        return sampler_output_list

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
