import copy
import weakref
from typing import Dict, List, Optional, Set, Tuple

import torch

from vllm.chef.station_chef_runner import StationChefStage
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (
    ExecuteModelRequest,
    SequenceDataDelta,
)

from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.spec_decode.multi_step_worker import MultiStepWorker


class StationChefWorker(MultiStepWorker, BatchExpansionTop1Scorer):
    """The MultiStepWorker is equivalent to a Worker except that it allows
    multiple forward passes in a single call, assuming the scheduler has
    allocated enough space to store the additional KV. This reduces overhead
    by invoking the scheduler less.

    The MultiStepWorker does not support cache swap operations, or beam search.
    Cache swap operations do not require large modifications. On the other hand,
    beam search requires memory allocations during sequence forks and thus
    requires more thought for MultiStepWorker support.
    """

    def __init__(self, *args, num_speculative_tokens: int, **kwargs):
        super().__init__(*args, **kwargs)

        if num_speculative_tokens is None or not isinstance(
            num_speculative_tokens, int
        ):
            raise TypeError(
                f"num_speculative_tokens should be an int, but got {num_speculative_tokens}."
            )

        self.num_speculative_tokens = num_speculative_tokens

    @property
    def num_lookahead_slots(self) -> int:
        return self.num_speculative_tokens

    @property
    def stage(self) -> StationChefStage:
        return self.worker.model_runner.stage

    @stage.setter
    def stage(self, stage: StationChefStage):
        if not isinstance(stage, StationChefStage):
            raise TypeError(
                f"stage should be an instance of StationChefStage, but got {stage}."
            )
        self.worker.model_runner.stage = stage

    def init_scorer(self, device, vocab_size: int):
        BatchExpansionTop1Scorer.__init__(
            self, weakref.proxy(self), device=device, vocab_size=vocab_size
        )

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: set[int],
        proposals: Optional[SpeculativeProposals] = None,
        with_bonus_token: bool = False,
    ) -> SpeculativeProposals:
        """
        Generates speculative proposals for the given model execution request.
        Args:
            execute_model_req (ExecuteModelRequest): The request object containing
                the model execution details.
            seq_ids_with_bonus_token_in_last_step (set[int]): A set of sequence IDs
                that had a bonus token in the last step.
            proposals (SpeculativeProposals, optional): Existing speculative
                proposals to be updated. Defaults to None.
            with_bonus_token (bool, optional): Flag indicating whether to include
                bonus tokens in the proposals. Defaults to False. If True, this method
                will only propose num_speculative_tokens - 1 tokens.
        Returns:
            SpeculativeProposals: The generated speculative proposals.
        """
        self.stage = StationChefStage.PROPOSAL

        if proposals is None:
            # since chef_config.num_lookahead_slots is sum(all gammas) - min(all gammas),
            # execute_model_req.num_lookahead_slots may not be equal to current worker.num_speculative_tokens
            execute_model_req.num_lookahead_slots = self.num_speculative_tokens - int(
                with_bonus_token
            )

            return super().get_spec_proposals(
                execute_model_req, seq_ids_with_bonus_token_in_last_step
            )

        if self.num_speculative_tokens <= 1 and with_bonus_token:
            return SpeculativeProposals(
                proposal_lens=torch.zeros_like(proposals.proposal_lens),
                proposal_token_ids=torch.tensor(
                    [], device=self.device, dtype=proposals.proposal_token_ids.dtype
                ),
                proposal_logits=torch.tensor(
                    [], device=self.device, dtype=proposals.proposal_logits.dtype
                ),
                proposal_logprobs=torch.tensor(
                    [], device=self.device, dtype=proposals.proposal_logprobs.dtype
                ),
                proposal_probs=torch.tensor(
                    [], device=self.device, dtype=proposals.proposal_probs.dtype
                ),
                no_proposals=True,
            )

        # updata execute_model_req
        new_execute_model_req = execute_model_req.clone(
            copy.deepcopy(execute_model_req.seq_group_metadata_list)
        )
        new_execute_model_req.num_lookahead_slots = self.num_lookahead_slots - int(
            with_bonus_token
        )

        for sgm in new_execute_model_req.seq_group_metadata_list:
            assert len(sgm.seq_data) == 1
            for i, sd in enumerate(sgm.seq_data.values()):
                sd.apply_delta(
                    SequenceDataDelta(
                        new_output_token_ids=proposals.proposal_token_ids[i].tolist(),
                        new_cumulative_logprob=sd.cumulative_logprob,  # i don't know what this is
                        new_num_computed_tokens=sd._num_computed_tokens
                        + proposals.proposal_lens[i],
                        new_stage=sd.stage,
                    )
                )
                sd._new_appended_tokens.extend(proposals.proposal_token_ids[i].tolist())

        # get new proposals
        return self._proposer.get_spec_proposals(
            new_execute_model_req, seq_ids_with_bonus_token_in_last_step
        )

    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.stage = StationChefStage.SCORING
        return super().score_proposals(execute_model_req, proposals)
