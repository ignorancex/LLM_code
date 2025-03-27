# 2024.04.04 Yixuan Mei

import math
from typing import List

from simulator.event_simulator.request import InferenceRequest, RequestPhase
from simulator.event_simulator.compute_node import ComputeNode, InferenceSettings
from simulator.scheduler.base_scheduler import ExecutionSchedule


def execution_policy(node: ComputeNode, executable_requests: List[InferenceRequest]) -> ExecutionSchedule:
    """
    Schedule execution for a given node (in FIFO order).

    :param node: the node that needs execution
    :param executable_requests: all executable requests on this node
    :return: an execution schedule
    """
    # get inference settings
    assert isinstance(node.inference_settings, InferenceSettings), "No inference settings found!"
    prompt_max_requests = node.inference_settings.prompt_max_requests
    prompt_max_tokens = node.inference_settings.prompt_max_tokens
    decode_max_context = node.inference_settings.decode_max_context
    decode_max_tokens = node.inference_settings.decode_max_tokens

    # get the list of requests that can be executed
    requests_to_infer: List[InferenceRequest] = []
    cur_prompt_requests, cur_prompt_tokens, cur_decode_context, cur_decode_tokens = 0, 0, 0, 0
    for request in executable_requests:
        if request.phase == RequestPhase.Initialization:
            # prompt phase
            assert request.token_seq_length <= prompt_max_tokens, "Found request that is too long!"
            if (cur_prompt_requests + 1 <= prompt_max_requests and
                    cur_prompt_tokens + request.token_seq_length <= prompt_max_tokens):
                requests_to_infer.append(request)
                cur_prompt_requests += 1
                cur_prompt_tokens += request.token_seq_length

        elif request.phase == RequestPhase.Increment:
            # decode phase
            assert request.prev_num_tokens <= decode_max_context, "Found request that is too long!"
            if (cur_decode_tokens + 1 <= decode_max_tokens and
                    cur_decode_context + request.prev_num_tokens <= decode_max_context):
                requests_to_infer.append(request)
                cur_decode_context += request.prev_num_tokens
                cur_decode_tokens += 1

        else:
            assert False, "Found unknown request phase!"

    # compatibility check
    for request in requests_to_infer:
        assert request.get_current_pipeline_stage().node_uid == node.node_uid, "Node uid does not match!"
        if len(request.inference_history) == 0:
            next_layer_id = 0
        else:
            next_layer_id = request.inference_history[-1].layer_id + 1
        assert next_layer_id == node.get_current_inference_layer(), f"Found incompatible inference request!"

    # return
    return ExecutionSchedule(node_uid=node.node_uid, requests=requests_to_infer)
