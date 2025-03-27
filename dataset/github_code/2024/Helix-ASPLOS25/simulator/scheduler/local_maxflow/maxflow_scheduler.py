# 2023.01.04 Yixuan Mei
from typing import List, Dict, Tuple

from simulator.event_simulator.request import InferenceRequest
from simulator.event_simulator.network_link import NetworkLink
from simulator.event_simulator.compute_node import ComputeNode
from simulator.event_simulator.coordinator_node import SourceNode
from simulator.event_simulator.cluster_simulator import ClusterSimulator
from simulator.scheduler.base_scheduler import BaseScheduler, TransmissionSchedule, ExecutionSchedule
from simulator.scheduler.local_maxflow.maxflow_utils import ClusterTopology, MaxFlowParameters


class MaxFlowScheduler(BaseScheduler):
    def __init__(self, parameters: MaxFlowParameters) -> None:
        """
        A MaxFlow-based scheduler.

        :param parameters: parameters for this scheduler
        :return: None
        """
        self.parameters: MaxFlowParameters = parameters
        self.cluster_topology: ClusterTopology or None = None

    def update_topology_and_flow(self, cluster: ClusterSimulator) -> None:
        """
        Update cluster topology and compute max flow.

        :return: None
        """
        # create cluster topology and compute max flow
        self.cluster_topology = ClusterTopology()
        self.cluster_topology.create_from_cluster(cluster=cluster, parameters=self.parameters)
        self.cluster_topology.compute_max_flow()

    def schedule_transmission(self, node: ComputeNode or SourceNode) -> Tuple[List[TransmissionSchedule], List[int]]:
        """
        Schedule transmission for a given node. Assign requests to idle links based on the flow computed
        and use all link capacity for transmission.

        :param node: the node that initializes the transmission.
        :return: a list of transmission schedule
        """
        # check that flow has already been computed
        assert self.cluster_topology is not None, "Cluster topology must be set before scheduling!"

        # use IWRR to schedule all requests on this node (the list may contain requests that have been
        # scheduled and those will be ignored)
        self.cluster_topology.schedule_all_requests(node_uid=node.node_uid,
                                                    request_uids=list(node.outbound_request_dict.keys()))

        # determine the free links and bandwidth usage over them
        free_links: Dict[int, float] = {}
        normalized_flow: Dict[int, float] = self.cluster_topology.get_normalized_flow(node_uid=node.node_uid)
        for link_uid in node.outbound_links:
            link: NetworkLink = node.outbound_links[link_uid]
            flow_control: float = (normalized_flow[link_uid] * node.outbound_nic_speed -
                                   (link.bandwidth - link.available_bandwidth))
            available_bandwidth: float = min(flow_control, link.get_available_bandwidth())
            assert available_bandwidth >= 0, "Found bad available bandwidth!"
            if available_bandwidth > 0:
                free_links[link_uid] = available_bandwidth

        # extract requests for free links and build schedule
        transmission_schedule: List[TransmissionSchedule] = []
        next_requests: Dict[int, int] = self.cluster_topology.get_next_requests(node_uid=node.node_uid,
                                                                                link_uids=list(free_links.keys()))
        for free_link_uid in free_links:
            if next_requests[free_link_uid] is not None:
                next_request_uid: int = next_requests[free_link_uid]
                new_schedule_entry = TransmissionSchedule(link_uid=free_link_uid,
                                                          bandwidth_usage=free_links[free_link_uid],
                                                          requests=[node.outbound_request_dict[next_request_uid]])
                transmission_schedule.append(new_schedule_entry)
        return transmission_schedule, []

    def schedule_execution(self, node: ComputeNode, executable_requests: List[InferenceRequest]) -> ExecutionSchedule:
        """
        Schedule execution for a given node. Run the first max_batch_size tokens.

        :param node: the node that needs execution
        :param executable_requests: all executable requests on this node
        :return: an execution schedule
        """
        max_batch_size: int = node.inference_settings.max_batch_size
        current_batch_size: int = 0
        num_requests_to_execute: int = len(executable_requests)
        for idx, request in enumerate(executable_requests):
            assert request.token_seq_length <= max_batch_size, "Found request that is too long!"
            if current_batch_size + request.token_seq_length > max_batch_size:
                num_requests_to_execute = idx
                break
            current_batch_size += request.token_seq_length
        return ExecutionSchedule(node_uid=node.node_uid, requests=executable_requests[:num_requests_to_execute])

    def schedule_model_loading(self):
        pass
