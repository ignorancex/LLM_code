# 2023.12.12 Yixuan Mei

from typing import Dict, List

from simulator.event_simulator.base_node import BaseNode, NodeType
from simulator.event_simulator.network_link import NetworkLink, TransmissionObject
from simulator.event_simulator.request import InferenceRequest


class SourceNode(BaseNode):
    def __init__(self, node_uid: int, outbound_nic_speed: float) -> None:
        """
        Abstraction of coordinator node's request issuing part.

        :param node_uid: unique identifier of this node
        :param outbound_nic_speed: network speed of sending packages
        :return: None
        """
        # basic info
        super().__init__(node_uid=node_uid, node_type=NodeType.Source)
        self.outbound_nic_speed: float = outbound_nic_speed

        # network management
        # nic and connections
        self.outbound_links: Dict[int, NetworkLink] = {}
        self.outbound_available_bandwidth: float = outbound_nic_speed
        self.outbound_requests_on_the_fly: Dict[str, TransmissionObject] = {}
        # local outbound request queue
        self.outbound_request_dict: Dict[int, InferenceRequest] = {}

        # logging
        self.entity_name: str = f"Source-{self.node_uid}"
        self.link_request_counters: Dict[int, int] = {}
        self.link_token_counters: Dict[int, int] = {}
        self.link_backup_request_counters: Dict[int, int] = {}
        self.link_backup_token_counters: Dict[int, int] = {}

    def add_outbound_link(self, outbound_link: NetworkLink) -> None:
        """
        Add an outbound link to this node.

        :param outbound_link: the link to be added
        :return: None
        """
        # a few topology checks
        assert outbound_link.link_uid not in self.outbound_links
        assert outbound_link.node_in_type == NodeType.Source
        assert outbound_link.node_in.node_uid == self.node_uid
        assert not outbound_link.node_out.node_uid == self.node_uid

        # add the link
        self.outbound_links[outbound_link.link_uid] = outbound_link
        self.link_request_counters[outbound_link.link_uid] = 0
        self.link_token_counters[outbound_link.link_uid] = 0
        self.link_backup_request_counters[outbound_link.link_uid] = 0
        self.link_backup_token_counters[outbound_link.link_uid] = 0

    def issue_request(self, request: InferenceRequest) -> None:
        """
        Issues a request by putting it into the outbound queue of source node.

        :param request: request to issue
        :return: None
        """
        # issue the request
        assert request.request_uid not in self.outbound_request_dict, "Duplicate requests!"
        self.outbound_request_dict[request.request_uid] = request


class SinkNode(BaseNode):
    def __init__(self, node_uid: int, inbound_nic_speed: float, total_num_layers: int) -> None:
        """
        Abstraction of coordinator node's request gathering part.

        :param node_uid: unique identifier of this node
        :param inbound_nic_speed: network speed of receiving packages
        :param total_num_layers: total number of layers that should be inferred (used for sanity check)
        :return: None
        """
        # basic info
        super().__init__(node_uid=node_uid, node_type=NodeType.Sink)
        self.inbound_nic_speed: float = inbound_nic_speed

        # network management
        # nic transmission bottleneck
        self.inbound_links: Dict[int, NetworkLink] = {}
        self.inbound_available_bandwidth: float = inbound_nic_speed
        self.inbound_requests_on_the_fly: Dict[str, TransmissionObject] = {}
        # local inbound and outbound queue
        self.inbound_request_queue: List[InferenceRequest] = []

        # total number of layers, we use this to check that the request has finished inference
        self.total_num_layers: int = total_num_layers

        # logging
        self.entity_name: str = f"Sink-{self.node_uid}"

    def add_inbound_link(self, inbound_link: NetworkLink) -> None:
        """
        Add an inbound link to this node.

        :param inbound_link: the link to be added
        :return: None
        """
        # a few topology checks
        assert inbound_link.link_uid not in self.inbound_links
        assert inbound_link.node_out_type == NodeType.Sink
        assert inbound_link.node_out.node_uid == self.node_uid
        assert not inbound_link.node_in.node_uid == self.node_uid

        # add the link
        self.inbound_links[inbound_link.link_uid] = inbound_link

    def receive_request(self, request: InferenceRequest) -> None:
        """
        Receives a request from one inbound link after sending is finished.

        :param request: the request to receive
        :return: None
        """
        # check whether the request has been properly inferred
        layers_inferred: List[int] = [entry.layer_id for entry in request.inference_history]
        assert layers_inferred == list(range(self.total_num_layers)), "Found request not fully inferred!"

        # add the request to inbound request queue
        self.inbound_request_queue.append(request)
