# 2023.01.25 Yixuan Mei
import math
from typing import List, Tuple, Dict

from simulator.event_simulator.request import InferenceRequest, PipelineStage
from simulator.event_simulator.network_link import NetworkLink
from simulator.event_simulator.compute_node import ComputeNode
from simulator.event_simulator.coordinator_node import SourceNode
from simulator.event_simulator.cluster_simulator import ClusterSimulator
from simulator.scheduler.base_scheduler import BaseScheduler, TransmissionSchedule, ExecutionSchedule, TransmissionType
from simulator.scheduler.execution_policy import execution_policy
from simulator.scheduler.global_maxflow.network_flow import FlowGraph, FlowParameters
from simulator.scheduler.global_maxflow.scheduler_core import SchedulerCore, SchedulingMode
from simulator.scheduler.global_maxflow.kv_expectation import KVParameters


class GlobalFlowScheduler(BaseScheduler):
    def __init__(self, parameters: FlowParameters, simulator: ClusterSimulator,
                 kv_param: KVParameters, scheduling_mode: SchedulingMode) -> None:
        """
        A global MaxFlow-based scheduler. It assigns routes to requests at the beginning. Scheduling
        on the fly just follows the predefined paths with bandwidth designated by the pre-computed flow.

        :param parameters: parameters for this scheduler
        :param simulator: the cluster simulator
        :param kv_param: kv cache expectation parameters
        :return: None
        """
        # initialize flow graph and scheduler core.
        self.parameters: FlowParameters = parameters
        self.flow_graph: FlowGraph = FlowGraph(cluster_simulator=simulator, parameters=parameters)
        self.core: SchedulerCore = SchedulerCore(cluster=simulator, flow_graph=self.flow_graph, kv_param=kv_param,
                                                 scheduling_mode=scheduling_mode)

        # scheduling mode
        self.scheduling_mode: SchedulingMode = scheduling_mode

    def update_scheduler(self, time_stamp: float) -> None:
        """
        Update flow graph based on current topology of the simulator. Then update scheduler core
        accordingly.

        :param time_stamp: time at which the flow is updated (simulation time)
        :return: None
        """
        # update flow graph and scheduler core
        self.flow_graph.update_flow(time_stamp=time_stamp)
        self.core.update(time_stamp=time_stamp)

    def generate_schedule(self, request: InferenceRequest) -> bool:
        """
        Generate schedule for a request from a global perspective. Behavior of this function depends
        on the phase of the request:
            1. initialization: create an inference pipeline (a route through the cluster that finishes
                               inference)
            2. increment: expect that the request already carries a valid route (i.e. the route used in
                          its initialization phase), only updates the scheduler's IWRR loads

        :param request: the request to schedule
        :return: scheduling succeeded or not
        """
        return self.core.schedule(request=request)

    def schedule_transmission(self, node: ComputeNode or SourceNode) -> Tuple[List[TransmissionSchedule], List[int]]:
        """
        Schedule transmission for a given node, send out requests based on the globally computed schedule.

        :param node: the node that initializes the transmission.
        :return: a list of transmission schedule, list of uids of nodes that needs transmission scheduling
        """
        node: ComputeNode

        # find all requests that can be scheduled
        scheduled_request_uids: List[int] = []
        link_unavailable: Dict[int, bool] = {link_uid: False for link_uid in node.outbound_links}
        for request_uid, request in node.outbound_request_dict.items():
            # get information about next pipeline stage for current request
            next_pipeline_stage: PipelineStage = request.get_next_pipeline_stage()
            next_link_uid: int = next_pipeline_stage.link_uid
            next_link: NetworkLink = node.outbound_links[next_link_uid]

            # if we have not marked the target link as unavailable, check whether we can send the request
            if not link_unavailable[next_link_uid]:
                # check whether this link is busy
                # TODO: when there are backup objects, need to check type
                if len(next_link.requests_on_the_fly) == 0:
                    scheduled_request_uids.append(request_uid)

                # whether this link was free or not, it will not be free after this check
                link_unavailable[next_link_uid] = True

        # build transmission schedule
        transmission_schedule: List[TransmissionSchedule] = []
        for request_uid in scheduled_request_uids:
            # find the request and march pipeline stage forward
            request: InferenceRequest = node.outbound_request_dict[request_uid]
            request.march_pipeline_stage()

            # generate schedule entry
            current_stage: PipelineStage = request.get_current_pipeline_stage()
            new_schedule_entry = TransmissionSchedule(link_uid=current_stage.link_uid,
                                                      bandwidth_usage=math.floor(current_stage.bandwidth_usage),
                                                      requests=[request],
                                                      transmission_type=TransmissionType.NormalExecution)
            transmission_schedule.append(new_schedule_entry)
        return transmission_schedule, []

    def schedule_execution(self, node: ComputeNode, executable_requests: List[InferenceRequest]) -> ExecutionSchedule:
        """
        Schedule execution for a given node.

        :param node: the node that needs execution
        :param executable_requests: all executable requests on this node
        :return: an execution schedule
        """
        return execution_policy(node=node, executable_requests=executable_requests)

    def schedule_model_loading(self):
        pass
