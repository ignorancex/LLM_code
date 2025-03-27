# 2023.01.04 Yixuan Mei
import math
import random
from typing import List, Dict, Tuple

from simulator.event_simulator.request import InferenceRequest, RequestPhase, PipelineStage
from simulator.event_simulator.compute_node import ComputeNode
from simulator.event_simulator.coordinator_node import SourceNode, SinkNode
from simulator.event_simulator.utils import TOKEN_SLOW_LINK
from simulator.scheduler.execution_policy import execution_policy
from simulator.scheduler.base_scheduler import BaseScheduler, TransmissionSchedule, ExecutionSchedule, TransmissionType


class ShortestQueueScheduler(BaseScheduler):
    """ Select the node with the shortest queue. """
    def schedule_transmission(self, node: ComputeNode or SourceNode) -> Tuple[List[TransmissionSchedule], List[int]]:
        """
        Schedule transmission for a given node. Assign request to the link with shortest queue and use all link
        capacity for transmission.

        :param node: the node that initializes the transmission.
        :return: a list of transmission schedule
        """
        # get all links that are not occupied on the node
        free_links: Dict[int, float] = {}  # link_uid -> available bandwidth
        for link_uid, link in node.outbound_links.items():
            flow_control: float = (node.outbound_nic_speed / len(node.outbound_links) -
                                   (link.bandwidth - link.available_bandwidth))
            available_bandwidth: float = min(flow_control, link.get_available_bandwidth())
            assert available_bandwidth >= 0, "Found bad available bandwidth!"
            if available_bandwidth > TOKEN_SLOW_LINK:
                free_links[link_uid] = available_bandwidth

        # get all nodes with shortest queue
        shortest_queue_length, shortest_queue_links = math.inf, []
        for link_uid, link in node.outbound_links.items():
            next_node = link.node_out
            if isinstance(next_node, SinkNode):
                # sink node always has the shortest queue (as it has no queue)
                shortest_queue_links.append(link_uid)
                break
            elif isinstance(next_node, ComputeNode):
                # compute nodes
                queue_length = len(next_node.inbound_request_queue)
                if queue_length < shortest_queue_length:
                    shortest_queue_length = queue_length
                    shortest_queue_links = [link_uid]
                elif queue_length == shortest_queue_length:
                    shortest_queue_links.append(link_uid)
            else:
                assert False, "Unknown node type!"

        # get free links with shortest queue
        free_links_with_shortest_queue = []
        for link_uid in shortest_queue_links:
            if link_uid in free_links:
                free_links_with_shortest_queue.append(link_uid)

        # assign requests to links
        link_uid_to_request: Dict[int, InferenceRequest or None] = {link_uid: None for link_uid in free_links}
        for request in node.outbound_request_dict.values():
            if request.phase == RequestPhase.Initialization:
                # Case 1: initialization phase
                # assign a random path
                _free_links = list(link_uid_to_request.items())
                random.shuffle(_free_links)
                for link_uid, cur_request in _free_links:
                    # can not assign request to link without shortest queue
                    if link_uid not in free_links_with_shortest_queue:
                        continue

                    if cur_request is None:
                        # get layers to infer on next node
                        next_node = node.outbound_links[link_uid].node_out
                        if isinstance(node, SourceNode):
                            layers_to_infer = sorted(list(next_node.in_vram_model_layers.keys()))
                        elif isinstance(next_node, SinkNode):
                            layers_to_infer = None
                        else:
                            layers_on_cur_node = sorted(list(node.in_vram_model_layers.keys()))
                            layers_on_next_node = sorted(list(next_node.in_vram_model_layers.keys()))
                            cur_last_layer = max(layers_on_cur_node)
                            layers_to_infer = sorted([x for x in layers_on_next_node if x > cur_last_layer])
                            assert not len(layers_to_infer) == 0, "Can not infer any layer on next node!"

                        # set pipeline
                        request.add_pipeline_stage(PipelineStage(link_uid=link_uid,
                                                                 bandwidth_usage=-1,
                                                                 node_uid=next_node.node_uid,
                                                                 layers_to_infer=layers_to_infer))
                        if not request.pipeline_set:
                            request.mark_pipeline_set()
                        request.march_pipeline_stage()

                        # assign request to link
                        link_uid_to_request[link_uid] = request
                        break

            elif request.phase == RequestPhase.Increment:
                # Case 2: increment phase
                # follow path in pipeline
                cur_pipeline_stage = request.get_next_pipeline_stage()
                link_to_use = cur_pipeline_stage.link_uid
                if link_to_use in free_links and link_uid_to_request[link_to_use] is None:
                    request.march_pipeline_stage()
                    link_uid_to_request[link_to_use] = request

            else:
                assert False, "Unknown request type!"

        # schedule
        transmission_schedule: List[TransmissionSchedule] = []
        for link_uid in free_links.keys():
            # get bandwidth and scheduled request
            available_bandwidth = free_links[link_uid]
            request = link_uid_to_request[link_uid]

            # send the request
            if request is not None:
                new_schedule = TransmissionSchedule(link_uid=link_uid,
                                                    bandwidth_usage=available_bandwidth,
                                                    requests=[request],
                                                    transmission_type=TransmissionType.NormalExecution)
                transmission_schedule.append(new_schedule)
        return transmission_schedule, []

    def schedule_execution(self, node: ComputeNode, executable_requests: List[InferenceRequest]) -> ExecutionSchedule:
        """
        Schedule execution for a given node. Run the first max_batch_size tokens.

        :param node: the node that needs execution
        :param executable_requests: all executable requests on this node
        :return: an execution schedule
        """
        return execution_policy(node=node, executable_requests=executable_requests)

    def schedule_model_loading(self, ):
        pass
