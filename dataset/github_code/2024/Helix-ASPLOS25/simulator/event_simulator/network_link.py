# 2023.12.12 Yixuan Mei

from typing import Dict, List, TYPE_CHECKING
from enum import Enum

from simulator.event_simulator.base_node import NodeType
from simulator.event_simulator.request import InferenceRequest, RequestLocation

if TYPE_CHECKING:
    from simulator.event_simulator.compute_node import ComputeNode
    from simulator.event_simulator.coordinator_node import SourceNode, SinkNode


class TransmissionType(Enum):
    """ Type of transmission """
    NormalExecution = "TransmissionType.NormalExecution"
    ActivationBackup = "TransmissionType.ActivationBackup"


class TransmissionObject:
    def __init__(self, requests: List[InferenceRequest], duration: float, bandwidth_usage: float,
                 transmission_type: TransmissionType, link_uid: int) -> None:
        """
        Represent a request on the fly.

        :param requests: the list of inference requests on the fly
        :param duration: how long this transmission takes
        :param bandwidth_usage: how much bandwidth this transmission uses
        :param transmission_type: type of transmission
        :param link_uid: uid of the link this transmission is on
        :return: None
        """
        self.requests: List[InferenceRequest] = requests
        self.duration: float = duration
        self.bandwidth_usage: float = bandwidth_usage
        self.send_finished: bool = False
        self.transmission_type: TransmissionType = transmission_type
        self.link_uid: int = link_uid

    def get_handle(self) -> str:
        """
        Get a handle of this TransmissionObject. We use uid of the first request as the handle. (This guarantees
        that there won't be duplicate handles within any link)

        :return: handle of this TransmissionObject
        """
        assert not len(self.requests) == 0, "Can not return handle for empty TransmissionObject!"
        if self.transmission_type == TransmissionType.NormalExecution:
            # in normal execution, this handle str helps us identify duplicate transmissions
            handle_str: str = f"{self.transmission_type}-{self.requests[0].request_uid}"
        elif self.transmission_type == TransmissionType.ActivationBackup:
            # in activation backup, one request might be sent on multiple links, need link uid as identifier
            handle_str: str = f"{self.transmission_type}-{self.link_uid}-{self.requests[0].request_uid}"
        else:
            assert False, "Unknown transmission type!"
        return handle_str


class LinkStatus(Enum):
    """ Status of current link. """
    Available = "LinkAvailability.Available"
    SenderBounded = "LinkAvailability.SenderBounded"
    LinkBounded = "LinkAvailability.LinkBounded"
    ReceiverBounded = "LinkAvailability.ReceiverBounded"


class NetworkLink:
    def __init__(self, link_uid: int, node_in: "ComputeNode" or "SourceNode", node_out: "ComputeNode" or "SinkNode",
                 latency: float, bandwidth: float) -> None:
        """
        Abstraction of a link between two nodes in the cluster.

        :param link_uid: unique identifier of this node
        :param node_in: one end of the link
        :param node_out: another end of the link
        :param latency: latency of this link
        :param bandwidth: bandwidth of this link
        :return: None
        """
        # basic info
        self.link_uid: int = link_uid
        self.latency: float = latency
        self.bandwidth: float = bandwidth

        # connectivity
        self.node_in: "ComputeNode" or "SourceNode" = node_in
        self.node_in_type: NodeType = node_in.node_type
        self.node_out: "ComputeNode" or "SinkNode" = node_out
        self.node_out_type: NodeType = node_out.node_type

        # transmission
        self.available_bandwidth: float = bandwidth
        self.requests_on_the_fly: Dict[str, TransmissionObject] = {}
        self.finishing_requests: Dict[str, TransmissionObject] = {}

        # logging
        self.entity_name: str = f"Link-{self.link_uid}"

    def calculate_transmission_time(self, requests: List[InferenceRequest], planned_bandwidth: float) -> float:
        """
        Calculate the transmission time of requests over this link using the planned bandwidth.

        :param requests: the list of requests to be transferred
        :param planned_bandwidth: planned bandwidth
        :return: transmission time
        """
        # check that the planned bandwidth is smaller than available
        assert planned_bandwidth <= self.available_bandwidth, "Demand more bandwidth than possible!"

        # calculate the time needed and return
        total_time: float = self.latency
        if self.node_in_type == NodeType.Source or self.node_out_type == NodeType.Sink:
            # transmission between coordinator and compute nodes contains no activation
            total_size: float = 0
            for request in requests:
                total_size += request.token_size * request.token_seq_length
            total_time += total_size / planned_bandwidth
        else:
            # transmission between compute nodes contains activation
            total_size: float = 0
            for request in requests:
                total_size += request.token_size * request.token_seq_length
                total_size += request.activation_size * request.token_seq_length
            total_time += total_size / planned_bandwidth
        return total_time

    def get_available_bandwidth(self) -> float:
        """
        Get available transmission bandwidth over this link.

        :return: available bandwidth
        """
        return min(self.node_in.outbound_available_bandwidth,
                   self.available_bandwidth,
                   self.node_out.inbound_available_bandwidth)

    def check_availability(self, planned_bandwidth: float) -> LinkStatus:
        """
        Check whether we can send with the required bandwidth over this link.

        :param planned_bandwidth: planned bandwidth
        :return: status of the link
        """

        # first check whether the output node has enough nic bandwidth for output
        if self.node_in.outbound_available_bandwidth < planned_bandwidth:
            return LinkStatus.SenderBounded

        # then we check whether the current link can allocate the bandwidth
        if self.available_bandwidth < planned_bandwidth:
            return LinkStatus.LinkBounded

        # finally we check the outbound node's inbound bandwidth
        if self.node_out.inbound_available_bandwidth < planned_bandwidth:
            return LinkStatus.ReceiverBounded

        # we are good
        return LinkStatus.Available

    def start_sending(self, requests: List[InferenceRequest], bandwidth_requirement: float,
                      transmission_type: TransmissionType) -> TransmissionObject:
        """
        Start sending a list of requests over current link. (Allocate resources)

        :param requests: the list of requests to send
        :param bandwidth_requirement: the amount of bandwidth allocated
        :param transmission_type: type of transmission
        :return: a TransmissionObject that represents this send
        """
        # check whether we have enough bandwidth
        _link_status = self.check_availability(planned_bandwidth=bandwidth_requirement)
        assert _link_status == LinkStatus.Available, f"Can not allocate send over link with status {_link_status}!"

        # calculate end time and build the transmission object
        duration: float = self.calculate_transmission_time(requests=requests, planned_bandwidth=bandwidth_requirement)
        transmission_object: TransmissionObject = TransmissionObject(requests=requests, duration=duration,
                                                                     bandwidth_usage=bandwidth_requirement,
                                                                     transmission_type=transmission_type,
                                                                     link_uid=self.link_uid)
        transmission_object_handle: str = transmission_object.get_handle()

        # then we put the transmission object into on-the-fly queues
        assert transmission_object_handle not in self.node_in.outbound_requests_on_the_fly
        assert transmission_object_handle not in self.requests_on_the_fly
        assert transmission_object_handle not in self.node_out.inbound_requests_on_the_fly
        self.node_in.outbound_requests_on_the_fly[transmission_object_handle] = transmission_object
        self.node_in.outbound_available_bandwidth -= bandwidth_requirement
        self.requests_on_the_fly[transmission_object_handle] = transmission_object
        self.available_bandwidth -= bandwidth_requirement
        self.node_out.inbound_requests_on_the_fly[transmission_object_handle] = transmission_object
        self.node_out.inbound_available_bandwidth -= bandwidth_requirement

        # return the transmission object to event handler
        return transmission_object

    def finish_sending(self, transmission_object_handle: str) -> TransmissionObject:
        """
        Finish sending a list of requests over the link. (Deallocate Resources)

        :param transmission_object_handle: handle of the transmission object to finish
        :return: the transmission object that finishes
        """
        # first we check that the transmission object did exist
        assert transmission_object_handle in self.requests_on_the_fly
        assert transmission_object_handle in self.node_in.outbound_requests_on_the_fly
        assert transmission_object_handle in self.node_out.inbound_requests_on_the_fly

        # save a reference to the transmission object
        cur_transmission_object: TransmissionObject = self.requests_on_the_fly[transmission_object_handle]
        assert not cur_transmission_object.send_finished, "Found a transmission object that has already finished!"
        cur_transmission_object.send_finished = True

        # remove from sender node
        bandwidth_used = self.node_in.outbound_requests_on_the_fly[transmission_object_handle].bandwidth_usage
        self.node_in.outbound_available_bandwidth += bandwidth_used
        del self.node_in.outbound_requests_on_the_fly[transmission_object_handle]

        # remove from link
        assert bandwidth_used == self.requests_on_the_fly[transmission_object_handle].bandwidth_usage
        self.available_bandwidth += bandwidth_used
        del self.requests_on_the_fly[transmission_object_handle]

        # remove from receiver
        # TODO: changing to a more accurate model. In reality, receiver nic bandwidth should be
        #  occupied during [latency, finish] (here is [0, finish - latency])
        assert bandwidth_used == self.node_out.inbound_requests_on_the_fly[transmission_object_handle].bandwidth_usage
        self.node_out.inbound_available_bandwidth += bandwidth_used
        del self.node_out.inbound_requests_on_the_fly[transmission_object_handle]

        # put into finishing requests
        assert transmission_object_handle not in self.finishing_requests, "Duplicate transmission object!"
        self.finishing_requests[transmission_object_handle] = cur_transmission_object

        # return the finished transmission object
        return cur_transmission_object

    def finish_transmission(self, finish_time: float, transmission_object_handle: str) -> TransmissionObject:
        """
        Finish transmission and put the requests into receiver's inbound queue.

        :param finish_time: time when transmission finishes
        :param transmission_object_handle: handle of the transmission object to finish
        :return: the transmission object that finishes
        """
        # pop the transmission object from finishing_requests
        assert transmission_object_handle in self.finishing_requests, "Unknown transmission object!"
        cur_transmission_object: TransmissionObject = self.finishing_requests[transmission_object_handle]
        assert cur_transmission_object.send_finished, "Send must be finished before transmission finishes!"
        del self.finishing_requests[transmission_object_handle]

        # determine what to do based on transmission type
        if cur_transmission_object.transmission_type == TransmissionType.NormalExecution:
            # move requests to receiver's inbound queue and update the requests
            for request in cur_transmission_object.requests:
                # update request routing history and location
                request.add_routing_history(link=self, bandwidth_usage=cur_transmission_object.bandwidth_usage)
                assert request.current_location == RequestLocation.Link, "Mis-routed request!"
                assert request.current_location_uid == self.link_uid, "Mis-routed request!"
                if self.node_out_type == NodeType.Sink:
                    new_location = RequestLocation.SinkNode
                else:
                    new_location = RequestLocation.ComputeNode
                request.update_location(new_location=new_location, new_location_uid=self.node_out.node_uid,
                                        arrive_time=finish_time)

                # put the request into receiver's inbound queue
                self.node_out.receive_request(request=request)

        elif cur_transmission_object.transmission_type == TransmissionType.ActivationBackup:
            # only compute nodes can receive activation backup
            assert self.node_out_type == NodeType.Compute, "Receive backup at sink!"

            # backup the activations in local activation backup cache
            for request in cur_transmission_object.requests:
                self.node_out.receive_backup(request=request)
        else:
            assert False, "Unknown transmission type!"

        # return the transmission object
        return cur_transmission_object
