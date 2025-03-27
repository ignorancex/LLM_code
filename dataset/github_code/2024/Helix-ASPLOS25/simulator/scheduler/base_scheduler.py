# 2023.12.16 Yixuan Mei

from typing import List, Tuple
from enum import Enum
from abc import ABC, abstractmethod

from simulator.event_simulator.request import InferenceRequest
from simulator.event_simulator.network_link import TransmissionType
from simulator.event_simulator.compute_node import ComputeNode
from simulator.event_simulator.coordinator_node import SourceNode


class TransmissionSchedule:
    def __init__(self, link_uid: int, bandwidth_usage: float, requests: List[InferenceRequest],
                 transmission_type: TransmissionType) -> None:
        """
        Describes a transmission schedule.

        :param link_uid: uid of the link involved
        :param bandwidth_usage: bandwidth usage
        :param requests: the requests to be sent together in this schedule
        :param transmission_type: the type of transmission (NormalExecution or ActivationBackup)
        :return: None
        """
        self.link_uid: int = link_uid
        self.bandwidth_usage: float = bandwidth_usage
        self.requests: List[InferenceRequest] = requests
        self.transmission_type: TransmissionType = transmission_type

    def get_description(self) -> str:
        """
        Get description of current transmission schedule.

        :return: description string
        """
        attributes = {"transmission_type": self.transmission_type,
                      "link_uid": self.link_uid,
                      "bandwidth_usage": self.bandwidth_usage,
                      "request_uids": [request.request_uid for request in self.requests]}
        return f"{attributes}"


class ExecutionSchedule:
    def __init__(self, node_uid: int, requests: List[InferenceRequest]) -> None:
        """
        Describes an execution schedule.

        :param node_uid: uid of node involved
        :param requests: the requests to be executed together as a batch
        :return: None
        """
        self.node_uid: int = node_uid
        self.requests: List[InferenceRequest] = requests

    def get_description(self) -> str:
        """
        Get description of current execution schedule.

        :return: description string
        """
        attributes = {"node_uid": self.node_uid,
                      "request_uids": [request.request_uid for request in self.requests]}
        return f"{attributes}"


class SchedulingMethod(Enum):
    MaxFlow = "SchedulingMethod.MaxFlow"
    Swarm = "SchedulingMethod.Swarm"
    Naive = "SchedulingMethod.Naive"
    ShortestQueue = "SchedulingMethod.ShortestQueue"


class BaseScheduler(ABC):
    # ********************************* Normal Execution ********************************* #
    @abstractmethod
    def schedule_transmission(self, node: ComputeNode or SourceNode) -> Tuple[List[TransmissionSchedule], List[int]]:
        """
        Schedule transmission for a given node.
        Note: 1. if no requests are to be transmitted, return an empty list
              2. schedule_transmission may trigger new start_transmission event because in some schedulers
                 (e.g. SwarmScheduler), we have hard #on_the_fly_requests limits. schedule_transmission may
                 reduce the number of requests on the fly for previous nodes thus allowing new transmissions.
              3. transmission scheduler may contain two types of transmissions: (1) transmission of activation
                 for normal execution, (2) transmission for activation backup. The two types of schedules are
                 distinguished by the transmission_type field.

        :param node: the node that initializes the transmission.
        :return: a list of transmission schedule, list of uids of nodes that needs transmission scheduling
        """
        pass

    @abstractmethod
    def schedule_execution(self, node: ComputeNode, executable_requests: List[InferenceRequest]) -> ExecutionSchedule:
        """
        Schedule execution for a given node.
        Note: 1. executable_requests is guaranteed to be non-empty
              2. if no requests are to be executed, set requests field of ExecutionSchedule as []

        :param node: the node that needs execution
        :param executable_requests: all executable requests on this node
        :return: an execution schedule
        """
        pass

    # ********************************* Model Placement ********************************** #
    # TODO: the following part is subject to changes
    @abstractmethod
    def schedule_model_loading(self, ):
        pass
