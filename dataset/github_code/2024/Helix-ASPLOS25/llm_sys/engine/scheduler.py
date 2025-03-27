"""Layerwise scheduler adapted from vllm."""
import time
from collections import deque
import copy
from typing import Deque, Dict, Iterable, List, Optional, Tuple, Union, Any

import torch
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.logger import init_logger
from vllm.sequence import (SequenceData, SequenceGroup, SequenceGroupMetadata,
                           SequenceStatus)

from vllm.core.scheduler import (PreemptionMode, ScheduledSequenceGroup,
                                 SchedulerOutputs, Scheduler)

from llm_sys.engine.common import (InputActivationType, PipelineSequence, PipelineSequenceData,
                                   PipelineSequenceGroupMetadata, PipelineStageOut)

logger = init_logger(__name__)

_SeqGroupIdType = Tuple[str, int]
_SeqIdType = Tuple[int, int]
_SeqStopInfo = Tuple[int, SequenceStatus]
_SeqGroupStopInfo = Tuple[str, Tuple[_SeqStopInfo]]
SeqGroupInputMetadata = Tuple[str, int, int]


class _WrappedQueue:
    def __init__(self):
        self.queue: Deque[SequenceGroup] = deque()
        self.ids = set()

    def appendleft(self, seq_group: SequenceGroup):
        self.queue.appendleft(seq_group)
        self.ids.add(seq_group.request_id)

    def popleft(self):
        seq_group = self.queue.popleft()
        self.ids.remove(seq_group.request_id)
        return seq_group

    def extendleft(self, seq_groups: Iterable[SequenceGroup]):
        self.queue.extendleft(seq_groups)
        self.ids.update(seq_group.request_id for seq_group in seq_groups)

    def append(self, seq_group: SequenceGroup):
        self.queue.append(seq_group)
        self.ids.add(seq_group.request_id)

    def extend(self, seq_groups: Iterable[SequenceGroup]):
        self.queue.extend(seq_groups)
        self.ids.update(seq_group.request_id for seq_group in seq_groups)

    def pop(self):
        seq_group = self.queue.pop()
        self.ids.remove(seq_group.request_id)
        return seq_group

    def __contains__(self, request_id: _SeqGroupIdType):
        return request_id in self.ids

    def __iter__(self):
        return iter(self.queue)

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, idx):
        return self.queue[idx]

    def remove(self, request_id: _SeqGroupIdType):
        self.ids.remove(request_id)
        self.queue = deque(filter(lambda sg: sg.request_id != request_id, self.queue))


class LayerwiseScheduler(Scheduler):

    def __init__(
            self,
            scheduler_config: SchedulerConfig,
            cache_config: CacheConfig,
            lora_config: Optional[LoRAConfig],
            layer_ids: List[int],
    ) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)
        self.next_seq_mapping: Dict[SequenceGroup, SequenceGroup] = {}
        self.layer_ids = layer_ids
        self.waitings: Dict[int, Deque[SequenceGroup]] = {i: deque() for i in layer_ids}
        self.runnings: Dict[int, Deque[SequenceGroup]] = {i: deque() for i in layer_ids}
        self.swappeds: Dict[int, Deque[SequenceGroup]] = {i: deque() for i in layer_ids}
        self.sleeps_cpu: Dict[int, Deque[_SeqGroupIdType]] = {i: deque() for i in layer_ids}
        self.sleeps_gpu: Dict[int, Deque[_SeqGroupIdType]] = {i: deque() for i in layer_ids}
        # req_id -> [begin_layer, end_layer)
        self.req_local_layers: Dict[int, Tuple[int, int]] = {}
        self.cur_layer = -1
        self.seq_groups: Dict[_SeqGroupIdType, SequenceGroup] = {}
        self.sleep_gpu_ids: Deque[_SeqGroupIdType] = _WrappedQueue()
        self.sleep_cpu_ids: Deque[_SeqGroupIdType] = _WrappedQueue()

    def context_switch(self, layer_id: int):
        if self.cur_layer >= 0:
            self.runnings[self.cur_layer] = self.running
            self.waitings[self.cur_layer] = self.waiting
            self.swappeds[self.cur_layer] = self.swapped
            self.sleeps_cpu[self.cur_layer] = self.sleep_cpu_ids
            self.sleeps_gpu[self.cur_layer] = self.sleep_gpu_ids
        self.running = self.runnings[layer_id]
        self.waiting = self.waitings[layer_id]
        self.swapped = self.swappeds[layer_id]
        self.sleep_cpu_ids = self.sleeps_cpu[layer_id]
        self.sleep_gpu_ids = self.sleeps_gpu[layer_id]
        self.cur_layer = layer_id

    def add_seq_group(self, seq_group: SequenceGroup, layer_id: int,
                      local_layers: Optional[Tuple[int, int]] = None) -> None:
        # Add sequence groups to the waiting queue.
        if not isinstance(seq_group.request_id, tuple):
            assert isinstance(seq_group.request_id, str)
            seq_group.request_id = (seq_group.request_id, layer_id)
        for seq in list(seq_group.seqs_dict.values()):
            if not isinstance(seq.seq_id, tuple):
                assert isinstance(seq.seq_id, int)
                seq_group.seqs_dict.pop(seq.seq_id)
                seq.seq_id = (seq.seq_id, layer_id)
                seq_group.seqs_dict[seq.seq_id] = seq
        logger.debug(f"add_seq_group {seq_group.request_id} layer: {layer_id}")
        self.waitings[layer_id].append(seq_group)
        self.seq_groups[seq_group.request_id] = seq_group
        if local_layers is None:
            assert seq_group.request_id[0] in self.req_local_layers
        else:
            self.req_local_layers[seq_group.request_id[0]] = local_layers

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        for layer_id in self.layer_ids:
            self.context_switch(layer_id)
            super().abort_seq_group([(request_id, layer_id)])

    def _schedule_waiting(self):
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.time()

        ignored_seq_groups: List[SequenceGroup] = []
        scheduled: List[SequenceGroup] = []
        # The total number of sequences on the fly, including the
        # requests in the generation phase.
        num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                            for seq_group in self.running)
        curr_loras = set(
            seq_group.lora_int_id
            for seq_group in self.running) if self.lora_enabled else None

        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        leftover_waiting_sequences = deque()
        num_batched_tokens = 0
        while self._passed_delay(now) and self.waiting:
            seq_group = self.waiting[0]
            waiting_seqs = seq_group.get_seqs(
                status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            # get_len includes output tokens if the request has been
            # preempted.
            num_prefill_tokens = waiting_seqs[0].get_len()
            if num_prefill_tokens > self.prompt_limit:
                logger.warning(
                    f"Input prompt ({num_prefill_tokens} tokens) is too "
                    f"long and exceeds limit of {self.prompt_limit}")
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            # try to preempt sleeping requests from gpu to cpu
            # TODO: also preempt sleeping requests in other layers.
            while not self._can_append_slots(seq_group):
                if self.sleep_gpu_ids:
                    victim_seq_group = self.seq_groups[self.sleep_gpu_ids.pop()]
                    self._preempt(victim_seq_group, blocks_to_swap_out,
                                  sleep=True)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    f"Input prompt ({num_prefill_tokens} tokens) is too "
                    f"long and exceeds the capacity of block_manager")
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                if (lora_int_id > 0 and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    self.waiting.popleft()
                    continue

            # If the number of batched tokens exceeds the limit, stop.
            num_batched_tokens += num_prefill_tokens
            if (num_batched_tokens >
                    self.scheduler_config.max_num_batched_tokens):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            if lora_int_id > 0:
                curr_loras.add(lora_int_id)
            self.waiting.popleft()
            self._allocate(seq_group)
            self.running.append(seq_group)
            num_curr_seqs += num_new_seqs
            scheduled.append(
                ScheduledSequenceGroup(
                    seq_group=seq_group,
                    token_chunk_size=num_prefill_tokens))
        self.waiting.extendleft(leftover_waiting_sequences)

        if scheduled or ignored_seq_groups:
            self.prev_prompt = True
            scheduler_outputs = SchedulerOutputs(
                scheduled_seq_groups=scheduled,
                prompt_run=True,
                num_batched_tokens=num_batched_tokens,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                ignored_seq_groups=ignored_seq_groups,
                num_lookahead_slots=self._get_num_lookahead_slots(
                    is_prefill=True),
            )
            return scheduler_outputs
        return None

    def _schedule_running(self, blocks_to_copy: Dict[int, List[int]],
                          blocks_to_swap_out: Dict[int, int], now):
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: Deque[SequenceGroup] = deque()
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.popleft()
            while not self._can_append_slots(seq_group):
                # try to preempt sleeping requests from gpu to cpu first
                if self.sleep_gpu_ids:
                    victim_seq_group = self.seq_groups[self.sleep_gpu_ids.pop()]
                    self._preempt(victim_seq_group, blocks_to_swap_out,
                                  sleep=True)
                    # FIXME: maintain a separate preempted_sleeping
                    preempted.append(victim_seq_group)
                elif self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop()
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slots(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running
        return preempted

    def _schedule_swapped(self, blocks_to_copy: Dict[int, List[int]],
                          blocks_to_swap_in: Dict[int, int]):
        num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                            for seq_group in self.running)
        curr_loras = set(
            seq_group.lora_int_id
            for seq_group in self.running) if self.lora_enabled else None

        leftover_swapped = deque()

        while self.swapped:
            seq_group = self.swapped[0]
            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                if (lora_int_id > 0 and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    self.swapped.popleft()
                    continue

            # If the sequence group cannot be swapped in, stop.
            if not self._can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            if lora_int_id > 0:
                curr_loras.add(lora_int_id)
            self.swapped.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            num_curr_seqs += num_new_seqs
            self.running.append(seq_group)

        self.swapped.extendleft(leftover_swapped)

    def _schedule(self, force_decode: bool) -> SchedulerOutputs:
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.time()

        # Join waiting sequences if possible.
        if not self.swapped and not force_decode:
            scheduler_outputs = self._schedule_waiting()
            if scheduler_outputs is not None:
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        preempted = self._schedule_running(blocks_to_copy, blocks_to_swap_out,
                                           now)

        # Swap in the sequence groups in the SWAPPED state if possible.
        # FIXME: preempted_sleeping
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted:
            self._schedule_swapped(blocks_to_copy, blocks_to_swap_in)

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=[
                ScheduledSequenceGroup(seq_group=running_group,
                                       token_chunk_size=1)
                for running_group in self.running
            ],
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False),
        )
        return scheduler_outputs

    def schedule(self, force_decode: bool) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, List[Any]]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule(force_decode=force_decode)
        now = time.time()

        # FIXME: fix this, let host send this
        finished_reqs_info = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            sg = seq_group.seq_group
            sp = sg.sampling_params
            has_termination = False
            term_info = []
            for seq in sg.get_seqs(SequenceStatus.RUNNING):
                if sp.max_tokens <= seq.get_output_len() + 1:
                    term_info.append((seq.seq_id[0], SequenceStatus.FINISHED_STOPPED))
                    has_termination = True
            if has_termination:
                term_info = tuple(term_info)
                finished_reqs_info.append((sg.request_id[0], term_info))

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                assert isinstance(seq.data, PipelineSequenceData) or self.cur_layer == 0
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            if self.cur_layer == 0:
                cls = SequenceGroupMetadata
            else:
                cls = PipelineSequenceGroupMetadata
            seq_group_metadata = cls(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                token_chunk_size=token_chunk_size,
                lora_request=seq_group.lora_request,
                computed_block_nums=common_computed_block_nums,
                state=seq_group.state,
                # `multi_modal_data` will only be present for the 1st comm
                # between engine and worker.
                # the subsequent comms can still use delta, but
                # `multi_modal_data` will be None.
                multi_modal_data=seq_group.multi_modal_data
                if scheduler_outputs.prompt_run else None,
            )
            seq_group_metadata_list.append(seq_group_metadata)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group)

        return seq_group_metadata_list, scheduler_outputs, finished_reqs_info

    def post_step(self, scheduler_outputs: SchedulerOutputs,
                  output: PipelineStageOut):
        """
        if there is a next layer, mark its value as ready, and move from
        sleeping to running/swapped. Otherwise, collect metadata for sending to
        the next stage.
        """
        # TODO: early exit if it is the last layer
        next_layer_id = self.cur_layer + 1
        new_running_reqs: List[SequenceGroup] = []
        new_swapped_reqs: List[SequenceGroup] = []
        cur_idx = 0
        cur_prompt_idx = 0
        tensors_to_send: List[SeqGroupInputMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            req_id = seq_group.seq_group.request_id

            # free useless input activation tensors
            for seq in seq_group.seq_group.get_seqs(SequenceStatus.RUNNING):
                if isinstance(seq, PipelineSequence):
                    seq.set_input_hidden(None, False)

            if self.req_local_layers[req_id[0]][1] > self.cur_layer:
                # has a next layer executed locally, perform the move
                next_layer_req_id = _get_next_layer_id(req_id)
                if next_layer_req_id in self.seq_groups:
                    # the next sequence group is already created
                    mapped_seq_group = self.seq_groups[next_layer_req_id]
                    # if it is sleeping on gpu, move it to the running.
                    if next_layer_req_id in self.sleeps_gpu[next_layer_id]:
                        new_running_reqs.append(mapped_seq_group)
                        self.sleeps_gpu[next_layer_id].remove(next_layer_req_id)
                    # else, it should be sleeping on cpu, move it to swapped.
                    else:
                        assert next_layer_req_id in self.sleeps_cpu[next_layer_id]
                        new_swapped_reqs.append(mapped_seq_group)
                        self.sleeps_cpu[next_layer_id].remove(next_layer_req_id)

                    # Set the input activation of corresponding requests
                    for seq in seq_group.seq_group.get_seqs(status=SequenceStatus.RUNNING):
                        next_layer_seq = mapped_seq_group.seqs_dict[
                            _get_next_layer_id(seq.seq_id)]
                        next_layer_seq.set_input_hidden((output.output_tensor,
                                                         cur_idx, cur_idx + 1),
                                                        False)
                        cur_idx += 1
                else:
                    # the next layer is never computed, because it's in the
                    # prompt phase
                    assert output.is_prompt
                    num_token = output.prompt_lens[cur_prompt_idx]
                    input_act = (output.output_tensor, cur_idx, cur_idx + num_token)
                    cur_idx += num_token
                    cur_prompt_idx += 1

                    cloned_seq_group = _clone_for_nxt_layer(seq_group.seq_group,
                                                            input_act)
                    self.add_seq_group(cloned_seq_group, self.cur_layer + 1)
            else:
                if output.is_prompt:
                    # here we adapt the assumption that prompt is shared.
                    num_tokens = output.prompt_lens[cur_prompt_idx]
                    tensors_to_send.append((seq_group.seq_group.request_id,
                                            cur_idx, cur_idx + num_tokens))
                    cur_idx += num_tokens
                    cur_prompt_idx += 1
                else:
                    num_seq = len(seq_group.seq_group.get_seqs(status=SequenceStatus.RUNNING))
                    tensors_to_send.append((seq_group.seq_group.request_id,
                                            cur_idx, cur_idx + num_seq))
                    cur_idx += num_seq

        if next_layer_id <= self.layer_ids[-1]:
            self.runnings[next_layer_id].extend(new_running_reqs)
            self.swappeds[next_layer_id].extend(new_swapped_reqs)

        # the input activation of this layer is consumed. requests are now slept
        ids_to_move = [sg.seq_group.request_id for sg in scheduler_outputs.scheduled_seq_groups]
        self.sleep_gpu_ids.extend(ids_to_move)
        ids_to_move = set(ids_to_move)
        self.running = [sg for sg in self.running if sg.request_id not in ids_to_move]
        return tensors_to_send

    def last_layer_post_step(self, scheduler_outputs):
        ids_to_move = [sg.seq_group.request_id for sg in scheduler_outputs.scheduled_seq_groups]
        self.sleep_gpu_ids.extend(ids_to_move)
        ids_to_move = set(ids_to_move)
        self.running = [sg for sg in self.running if sg.request_id not in ids_to_move]

    def _preempt(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: Dict[int, int],
            preemption_mode: Optional[PreemptionMode] = None,
            sleep: bool = False
    ) -> PreemptionMode:
        assert preemption_mode is None or preemption_mode == PreemptionMode.SWAP
        preemption_mode = PreemptionMode.SWAP
        self._preempt_by_swap(seq_group, blocks_to_swap_out, sleep)
        return preemption_mode

    def _preempt_by_swap(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: Dict[int, int],
            sleep: bool = False
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        if sleep:
            assert seq_group.request_id not in self.sleep_cpu_ids
            self.sleep_cpu_ids.append(seq_group.request_id)
        else:
            self.swapped.append(seq_group)

    def free_finished_seq_groups(self) -> None:
        # we need to clear all reference
        new_running = deque()
        for seq_group in self.running:
            if not seq_group.is_finished():
                new_running.append(seq_group)
            else:
                self.seq_groups.pop(seq_group.request_id)
        self.running = new_running

    def finish_requests(self, finished_seq_infos: List[_SeqGroupStopInfo]):
        if not finished_seq_infos:
            return
        layer_id = self.cur_layer
        for req_id, stop_infos in finished_seq_infos:
            if (req_id, layer_id) in self.seq_groups:
                seq_group = self.seq_groups[(req_id, layer_id)]
                for seq_id, status in stop_infos:
                    seq = seq_group.seqs_dict[(seq_id, layer_id)]
                    seq.status = status
                    self.free_seq(seq)
                if seq_group.is_finished():
                    self.seq_groups.pop(seq_group.request_id)
                    # this function is called only for requests finished an iter
                    # and waiting for next iter's input, while the next iter it
                    # is finished.
                    if seq_group.request_id in self.sleep_gpu_ids:
                        self.sleep_gpu_ids.remove(seq_group.request_id)
                    else:
                        self.sleep_cpu_ids.remove(seq_group.request_id)
        self.running = deque(filter(lambda sg: not sg.is_finished(), self.running))

    def update_req_data(self, layer_id: int, req_id: str, seq_datas: Dict[int, torch.Tensor]):
        # Move seq group
        if layer_id == self.cur_layer:
            sleep_on_gpus = self.sleep_gpu_ids
            sleep_on_cpus = self.sleep_cpu_ids
            running = self.running
            swapped = self.swapped
        else:
            sleep_on_gpus = self.sleeps_gpu[layer_id]
            sleep_on_cpus = self.sleeps_cpu[layer_id]
            running = self.runnings[layer_id]
            swapped = self.swappeds[layer_id]

        if isinstance(req_id, str):
            req_id = (req_id, layer_id)
        else:
            assert isinstance(req_id, Tuple)
        seq_group = self.seq_groups[req_id]
        if req_id in sleep_on_gpus:
            sleep_on_gpus.remove(req_id)
            running.append(seq_group)
        else:
            assert req_id in sleep_on_cpus
            sleep_on_cpus.remove(req_id)
            swapped.append(seq_group)

        if layer_id != 0:  # layer 0 does not have input activation.
            assert len(seq_datas) == 1, "advanced sampling not supported"
            for seq_id in seq_datas:
                data = seq_datas[seq_id]
                if isinstance(seq_id, int):
                    seq_id = (seq_id, layer_id)
                seq: PipelineSequence = seq_group.seqs_dict[seq_id]
                seq.set_input_hidden(data, False)


def _get_next_layer_id(cur_id: Union[_SeqGroupIdType, _SeqIdType]) -> int:
    return cur_id[0], cur_id[1] + 1


def _clone_for_nxt_layer(seq_group: SequenceGroup,
                         input_act: InputActivationType) -> SequenceGroup:
    new_req_id = _get_next_layer_id(seq_group.request_id)
    cloned_seqs = []
    for seq_id in seq_group.seqs_dict:
        seq = seq_group.seqs_dict[seq_id]
        new_seq_id = _get_next_layer_id(seq.seq_id)
        # clone a pipeline sequence data. Its input is not ready yet.
        cloned_seq = PipelineSequence(new_seq_id, seq.prompt,
                                      copy.copy(seq.get_prompt_token_ids()),
                                      seq.block_size, input_act,
                                      seq.eos_token_id, seq.lora_request)
        cloned_seqs.append(cloned_seq)
    return SequenceGroup(new_req_id, cloned_seqs, seq_group.sampling_params,
                         seq_group.metrics.arrival_time, seq_group.lora_request,
                         seq_group.multi_modal_data)
