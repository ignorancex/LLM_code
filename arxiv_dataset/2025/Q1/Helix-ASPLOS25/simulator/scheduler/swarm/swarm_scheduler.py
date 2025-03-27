# 2023.01.18 Yixuan Mei

from typing import List, Tuple, Dict, Set
from queue import PriorityQueue

from simulator.event_simulator.request import InferenceRequest, RequestPhase, PipelineStage
from simulator.event_simulator.base_node import NodeType
from simulator.event_simulator.compute_node import ComputeNode
from simulator.event_simulator.coordinator_node import SourceNode
from simulator.event_simulator.cluster_simulator import ClusterSimulator
from simulator.event_simulator.utils import TOKEN_SLOW_LINK
from simulator.scheduler.execution_policy import execution_policy
from simulator.scheduler.base_scheduler import BaseScheduler, TransmissionSchedule, ExecutionSchedule, TransmissionType
from simulator.scheduler.local_maxflow.maxflow_utils import RequestDestinationCache


class SwarmParameters:
    def __init__(self, initial_priority: float, smoothing_parameter: float, max_on_the_fly_request: int) -> None:
        """
        Parameters for swarm scheduler.
        Note: See Appendix C of SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient
        (https://proceedings.mlr.press/v202/ryabinin23a/ryabinin23a.pdf)

        :param initial_priority: (epsilon) initial priority of each machine
        :param smoothing_parameter: (gamma) smoothing parameter in priority update
        :param max_on_the_fly_request: max number of requests scheduled before any of them finish
        :return: None
        """
        self.initial_priority: float = initial_priority
        self.smoothing_parameter: float = smoothing_parameter
        self.max_on_the_fly_requests: int = max_on_the_fly_request


class SwarmNode:
    def __init__(self, node_uid: int, outbound_link_uids: List[int], outbound_node_uids: Dict[int, int],
                 initial_priority: float) -> None:
        """
        Swarm trainer, which resides on workers (compute nodes) in the cluster and schedule the batches.
        Note: See Appendix C of SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient
        (https://proceedings.mlr.press/v202/ryabinin23a/ryabinin23a.pdf)

        :param node_uid: uid of the node this trainer resides on
        :param outbound_link_uids: uids of all outbound links
        :param outbound_node_uids: a dict from outbound link uid -> node uid
        :param initial_priority: initial priority of each outbound link
        :return: None
        """
        # basic information
        self.node_uid: int = node_uid
        self.outbound_link_uids: List[int] = outbound_link_uids
        self.outbound_node_uids: Dict[int, int] = outbound_node_uids
        self.initial_priority: float = initial_priority

        # scheduling
        self.ema = {link_uid: initial_priority for link_uid in self.outbound_link_uids}
        self.queue: PriorityQueue[Tuple[float, int]] = PriorityQueue()
        for link_uid in self.outbound_link_uids:
            self.queue.put((initial_priority, link_uid))
        self.logged_request_uids: Set[int] = set()
        self.transmission_scheduling_cache = RequestDestinationCache(all_link_uids=outbound_link_uids)
        self.on_the_fly_requests: Set[int] = set()

        # history (number of requests scheduled to each link)
        self.traces: Dict[int, int] = {link_uid: 0 for link_uid in self.outbound_link_uids}

    def choose_server(self) -> int:
        """
        Select a server for request execution using IWRR. Note that we return link_uid here as one link
        corresponds to one server. The weights of the selected server will be updated after selection.

        :return: link_uid selected
        """
        priority, link_uid = self.queue.get()
        new_priority = priority + self.ema[link_uid]
        self.queue.put((new_priority, link_uid))
        self.traces[link_uid] += 1
        return link_uid

    def update_weights(self, link_uid: int, request_uid: int, delta_t: float, smoothing_factor: float) -> None:
        """
        Update weights for server connected to link_uid.

        :param link_uid: uid of the link to update
        :param request_uid: uid of the request to report
        :param delta_t: time used for last request
        :param smoothing_factor: smoothing factor used in update
        :return: None
        """
        assert link_uid in self.ema, "Unknown link found!"
        assert request_uid not in self.logged_request_uids, "Duplicate request reported!"
        self.ema[link_uid] = smoothing_factor * delta_t + (1 - smoothing_factor) * self.ema[link_uid]
        self.logged_request_uids.add(request_uid)
        assert request_uid in self.on_the_fly_requests, "Unknown request found!"
        self.on_the_fly_requests.remove(request_uid)

    def check_logged(self, request_uid: int) -> bool:
        """
        Check whether a request_uid has been used to update the weights.

        :param request_uid: uid of the request to check
        :return: whether the request has been used
        """
        return request_uid in self.logged_request_uids


class SwarmScheduler(BaseScheduler):
    def __init__(self, parameters: SwarmParameters) -> None:
        """
        Swarm scheduler.
        Note: Based on paper "SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient"
        (https://proceedings.mlr.press/v202/ryabinin23a/ryabinin23a.pdf)

        :param parameters: swarm scheduler parameters
        :return: None
        """
        # basic info
        self.parameters = parameters

        # scheduling
        self.cluster: ClusterSimulator or None = None
        self.nodes: Dict[int, SwarmNode] = {}

    def reset_topology(self, cluster: ClusterSimulator) -> None:
        """
        Update cluster topology and initialize the IWRR.

        :return: None
        """
        self.cluster = cluster
        self.nodes = {}

        # add source node
        assert cluster.source_node is not None, "No source node found in cluster!"
        source_node: SourceNode = cluster.source_node
        source_outbound_link_uids = []
        source_outbound_node_uids = {}
        for link_uid, link in source_node.outbound_links.items():
            source_outbound_link_uids.append(link_uid)
            source_outbound_node_uids[link_uid] = link.node_out.node_uid
        self.nodes[source_node.node_uid] = SwarmNode(node_uid=source_node.node_uid,
                                                     outbound_link_uids=source_outbound_link_uids,
                                                     outbound_node_uids=source_outbound_node_uids,
                                                     initial_priority=self.parameters.initial_priority)

        # add compute nodes
        for compute_node_uid, compute_node in cluster.compute_nodes.items():
            assert compute_node_uid not in self.nodes, "Duplicate node found!"
            outbound_link_uids = []
            outbound_node_uids = {}
            for link_uid, link in compute_node.outbound_links.items():
                outbound_link_uids.append(link_uid)
                outbound_node_uids[link_uid] = link.node_out.node_uid
            self.nodes[compute_node_uid] = SwarmNode(node_uid=compute_node_uid,
                                                     outbound_link_uids=outbound_link_uids,
                                                     outbound_node_uids=outbound_node_uids,
                                                     initial_priority=self.parameters.initial_priority)

    def schedule_transmission(self, node: ComputeNode or SourceNode) -> Tuple[List[TransmissionSchedule], List[int]]:
        """
        Schedule transmission for a given node.
        Note: 1. In SWARM scheduler, we use the pipeline filed of a request to record the path the query takes
                 during initialization.

        :param node: the node that initializes the transmission.
        :return: a list of transmission schedule
        """
        # update statistics for incoming nodes using new requests
        retransmission_node_uids: Set[int] = set()
        if node.node_type == NodeType.Compute:
            for request_uid, request in node.outbound_request_dict.items():
                # find the node and link this request comes from
                last_node_uid, last_link_uid = request.get_last_node_and_link_uid()
                assert last_node_uid in self.nodes, "Unknown node found!"
                last_swarm_node: SwarmNode = self.nodes[last_node_uid]

                # update weights if request not updated
                if not last_swarm_node.check_logged(request_uid=request_uid):
                    # get effective delta-t (delta time / #tokens)
                    # Note: since request is updated the first time it appears in outbound queue, the
                    # following computation of delta time is correct.
                    last_transmission_start_time: float = request.location_history[-2][1]
                    current_time: float = self.cluster.current_time
                    num_layers_inferred: int = request.get_num_layers_on_node(node_uid=node.node_uid)
                    delta_time: float = current_time - last_transmission_start_time
                    assert delta_time > 0, "Bad execution time!"

                    # update weights
                    last_swarm_node.update_weights(link_uid=last_link_uid,
                                                   request_uid=request_uid,
                                                   delta_t=delta_time,
                                                   smoothing_factor=self.parameters.smoothing_parameter)

                    # the node may transmit a new request
                    retransmission_node_uids.add(last_node_uid)

        # use IWRR to schedule the requests
        # determine the max number of requests that should be scheduled
        current_swarm_node: SwarmNode = self.nodes[node.node_uid]
        if not len(current_swarm_node.outbound_link_uids) == 1:
            # if there are multiple outbound links, limit max_new_send to make sure the performance
            # will not drop a lot when there is a burst of requests coming in
            max_new_sends: int = self.parameters.max_on_the_fly_requests - len(current_swarm_node.on_the_fly_requests)
        else:
            # if only one outbound link, no limit is imposed
            max_new_sends: int = len(node.outbound_request_dict) * 2
        max_new_schedules: int = max_new_sends - len(current_swarm_node.transmission_scheduling_cache.cache)
        assert max_new_schedules >= 0, "Bad new schedule count!"

        # schedule the requests
        scheduled_count: int = 0
        for request_uid, request in node.outbound_request_dict.items():
            if not current_swarm_node.transmission_scheduling_cache.contains(request_uid=request_uid):
                if scheduled_count >= max_new_schedules:
                    break

                if request.phase == RequestPhase.Initialization:
                    # initialization chooses a path that balances workload
                    next_link_uid: int = current_swarm_node.choose_server()
                elif request.phase == RequestPhase.Increment:
                    # increment phase follows the path set during initialization
                    next_pipeline_stage: PipelineStage = request.get_next_pipeline_stage()
                    request.march_pipeline_stage()
                    next_link_uid = next_pipeline_stage.link_uid
                else:
                    assert False, "Unknown request phase!"

                current_swarm_node.transmission_scheduling_cache.add_request(request_uid=request_uid,
                                                                             link_uid=next_link_uid)
                scheduled_count += 1

        # determine the free links and available bandwidth over them
        free_links: Dict[int, float] = {}
        for link_uid, link in node.outbound_links.items():
            flow_control: float = (node.outbound_nic_speed / len(node.outbound_links) -
                                   (link.bandwidth - link.available_bandwidth))
            available_bandwidth: float = min(flow_control, link.get_available_bandwidth())
            assert available_bandwidth >= 0, "Found bad available bandwidth!"
            if available_bandwidth > TOKEN_SLOW_LINK:
                free_links[link_uid] = available_bandwidth

        # extract requests for free links and build schedule
        transmission_schedule: List[TransmissionSchedule] = []
        next_requests: Dict[int, int or None] = {}
        for link_uid in free_links.keys():
            next_request_uid = current_swarm_node.transmission_scheduling_cache.get_next_request(link_uid=link_uid)
            next_requests[link_uid] = next_request_uid
        for free_link_uid in free_links:
            if next_requests[free_link_uid] is not None:
                # get the request
                next_request_uid: int = next_requests[free_link_uid]
                next_request: InferenceRequest = node.outbound_request_dict[next_request_uid]

                # if the request is in initialization phase, record the route it takes
                if next_request.phase == RequestPhase.Initialization:
                    # get layers to infer
                    current_node_uid = current_swarm_node.node_uid
                    next_node_uid = current_swarm_node.outbound_node_uids[free_link_uid]
                    self.cluster: ClusterSimulator
                    if current_node_uid == self.cluster.source_node.node_uid:
                        layers_on_cur_node: List[int] or None = None
                    else:
                        layers_on_cur_node: List[int] or None = sorted(
                            self.cluster.compute_nodes[current_node_uid].in_vram_model_layers.keys()
                        )
                    if next_node_uid == self.cluster.sink_node.node_uid:
                        layers_on_next_node: List[int] or None = None
                    else:
                        layers_on_next_node: List[int] or None = sorted(
                            self.cluster.compute_nodes[next_node_uid].in_vram_model_layers.keys()
                        )
                    assert not (layers_on_cur_node is None and layers_on_next_node is None), \
                        "Source is connected to sink!"
                    if layers_on_cur_node is None:
                        # current node is source, infer all layers on next node
                        layers_to_infer = layers_on_next_node
                    elif layers_on_next_node is None:
                        # next node is sink, layers_to_infer is None
                        layers_to_infer = None
                    else:
                        # both current node and next node are compute nodes
                        cur_last_layer = max(layers_on_cur_node)
                        layers_to_infer = sorted([x for x in layers_on_next_node if x > cur_last_layer])
                        assert not len(layers_to_infer) == 0, "Can not infer any layer on next node!"

                    next_stage = PipelineStage(link_uid=free_link_uid,
                                               bandwidth_usage=-1,
                                               node_uid=next_node_uid,
                                               layers_to_infer=layers_to_infer)
                    next_request.add_pipeline_stage(pipeline_stage=next_stage)
                    if not next_request.pipeline_set:
                        next_request.mark_pipeline_set()
                    next_request.march_pipeline_stage()

                # generate schedule and mark the request as on the fly
                current_swarm_node.on_the_fly_requests.add(next_request_uid)
                new_schedule_entry = TransmissionSchedule(link_uid=free_link_uid,
                                                          bandwidth_usage=free_links[free_link_uid],
                                                          requests=[next_request],
                                                          transmission_type=TransmissionType.NormalExecution)
                transmission_schedule.append(new_schedule_entry)
        return transmission_schedule, list(retransmission_node_uids)

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
