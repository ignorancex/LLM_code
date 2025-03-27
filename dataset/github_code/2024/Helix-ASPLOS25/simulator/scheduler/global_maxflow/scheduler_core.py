# 2023.01.28 Yixuan Mei

from typing import Dict, List, Tuple
from enum import Enum

from simulator.event_simulator.request import InferenceRequest, RequestPhase, PipelineStage
from simulator.event_simulator.base_node import NodeType, BaseNode
from simulator.event_simulator.compute_node import ComputeNode, InferenceSettings
from simulator.event_simulator.coordinator_node import SourceNode, SinkNode
from simulator.event_simulator.cluster_simulator import ClusterSimulator
from simulator.event_simulator.kv_cache import KVCache
from simulator.event_simulator.utils import TOKEN_SLOW_LINK, ACT_SLOW_LINK, ATOL, AVG_OUTPUT_LEN
from simulator.scheduler.global_maxflow.network_flow import FlowGraph
from simulator.scheduler.global_maxflow.interleaved_weighted_round_robin import IWRR
from simulator.scheduler.global_maxflow.kv_expectation import KVExpectation, KVParameters


class SchedulingMode(Enum):
    Online = "SchedulingMode.Online"
    Offline = "SchedulingMode.Offline"


class SchedulerNode:
    def __init__(self, node: BaseNode, flow_graph: FlowGraph, scheduler_core: "SchedulerCore",
                 scheduling_mode: SchedulingMode) -> None:
        """
        Represent a node (source node / sink node / compute node) in the cluster.

        :param node: the node to create from
        :param flow_graph: current flow graph
        :return: None
        """
        # basic node information
        self.node_uid: int = node.node_uid
        self.node_type: NodeType = node.node_type
        self.simulator_node: BaseNode = node

        # scheduler core
        self.scheduler_core: "SchedulerCore" = scheduler_core

        # scheduling mode
        self.scheduling_mode: SchedulingMode = scheduling_mode

        # flow version
        assert flow_graph.flow_graph_timestamp is not None, "SchedulerNode must create with valid flow!"
        self.flow_graph_timestamp: float = flow_graph.flow_graph_timestamp

        # node capacity and flow (in #tokens/s)
        capacity_dict: Dict[str, float or None] = flow_graph.get_node_capacity(node_uid=node.node_uid)
        flow_dict: Dict[str, float or None] = flow_graph.get_node_flow(node_uid=node.node_uid)

        # inbound-related statistics
        if self.node_type == NodeType.Compute or self.node_type == NodeType.Sink:
            node: SinkNode
            # inbound network interface card
            # full speed (byte/s) and used speed (byte/s)
            self.inbound_nic_speed: float = node.inbound_nic_speed
            self.inbound_nic_used_speed: float = (self.inbound_nic_speed * flow_dict["inbound"] /
                                                  capacity_dict["inbound"])
            # token throughput (#tokens/s) and used token throughput (#tokens/s)
            self.inbound_nic_token_throughput: float = capacity_dict["inbound"]
            self.inbound_nic_used_token_throughput: float = flow_dict["inbound"]
            # inbound link uids (and nodes)
            self.inbound_link_uids: List[int] = list(node.inbound_links.keys())
            self.inbound_node_uids: List[int] = [node.inbound_links[link_uid].node_in.node_uid for link_uid in
                                                 self.inbound_link_uids]

            # statistics of inbound links
            # full speed (byte/s) and used speed (byte/s)
            self.inbound_links_latency: List[float] = []
            self.inbound_links_speed: List[float] = []
            self.inbound_links_used_speed: List[float] = []
            # token throughput (#tokens/s) and used token throughput (#tokens/s)
            self.inbound_links_token_throughput: List[float] = []
            self.inbound_links_used_token_throughput: List[float] = []
            for prev_link_uid, prev_node_uid in zip(self.inbound_link_uids, self.inbound_node_uids):
                latency: float = node.inbound_links[prev_link_uid].latency
                token_throughput: float = flow_graph.get_link_capacity(prev_node_uid=prev_node_uid,
                                                                       next_node_uid=self.node_uid)["transmission"]
                used_token_throughput: float = flow_graph.get_link_flow(prev_node_uid=prev_node_uid,
                                                                        next_node_uid=self.node_uid)["transmission"]
                speed: float = node.inbound_links[prev_link_uid].bandwidth
                used_speed: float = speed * used_token_throughput / token_throughput
                self.inbound_links_latency.append(latency)
                self.inbound_links_speed.append(speed)
                self.inbound_links_used_speed.append(used_speed)
                self.inbound_links_token_throughput.append(token_throughput)
                self.inbound_links_used_token_throughput.append(used_token_throughput)

        elif self.node_type == NodeType.Source:
            node: SourceNode
            # inbound network interface card
            # full speed (byte/s) and used speed (byte/s)
            self.inbound_nic_speed = None
            self.inbound_nic_used_speed = None
            # token throughput (#tokens/s) and used token throughput (#tokens/s)
            self.inbound_nic_token_throughput = None
            self.inbound_nic_used_token_throughput = None
            # inbound link uids (and nodes)
            self.inbound_link_uids = None
            self.inbound_node_uids = None

            # statistics of inbound links
            # full speed (byte/s) and used speed (byte/s)
            self.inbound_links_latency = None
            self.inbound_links_speed = None
            self.inbound_links_used_speed = None
            # token throughput (#tokens/s) and used token throughput (#tokens/s)
            self.inbound_links_token_throughput = None
            self.inbound_links_used_token_throughput = None

        else:
            assert False, "Unknown node type!"

        # outbound-related statistics
        if self.node_type == NodeType.Compute or self.node_type == NodeType.Source:
            node: SourceNode
            # outbound network interface card
            # full speed (byte/s) and used speed (byte/s)
            self.outbound_nic_speed: float = node.outbound_nic_speed
            self.outbound_nic_used_speed: float = (self.outbound_nic_speed * flow_dict["outbound"] /
                                                   capacity_dict["outbound"])
            # token throughput (#tokens/s) and used token throughput (#tokens/s)
            self.outbound_nic_token_throughput: float = capacity_dict["outbound"]
            self.outbound_nic_used_token_throughput: float = flow_dict["outbound"]
            # outbound link uids (and nodes)
            self.outbound_link_uids: List[int] = list(node.outbound_links.keys())
            self.outbound_node_uids: List[int] = [node.outbound_links[link_uid].node_out.node_uid for link_uid in
                                                  self.outbound_link_uids]

            # inference settings & model layers of outbound nodes (None if is sink node)
            self.outbound_simulator_nodes: List[BaseNode] = []
            self.outbound_node_inference_settings: List[InferenceSettings or None] = []
            self.outbound_node_model_layers: List[List[int] or None] = []
            for link_uid in self.outbound_link_uids:
                next_node = node.outbound_links[link_uid].node_out
                self.outbound_simulator_nodes.append(next_node)
                if next_node.node_type == NodeType.Compute:
                    self.outbound_node_inference_settings.append(next_node.inference_settings)
                    self.outbound_node_model_layers.append(sorted(list(next_node.in_vram_model_layers.keys())))
                else:
                    self.outbound_node_inference_settings.append(None)
                    self.outbound_node_model_layers.append(None)

            # statistics of outbound links
            # full speed (byte/s) and used speed (byte/s)
            self.outbound_links_latency: List[float] = []
            self.outbound_links_speed: List[float] = []
            self.outbound_links_used_speed: List[float] = []
            # token throughput (#tokens/s) and used token throughput (#tokens/s)
            self.outbound_links_token_throughput: List[float] = []
            self.outbound_links_used_token_throughput: List[float] = []
            for next_link_uid, next_node_uid in zip(self.outbound_link_uids, self.outbound_node_uids):
                latency: float = node.outbound_links[next_link_uid].latency
                token_throughput: float = flow_graph.get_link_capacity(prev_node_uid=self.node_uid,
                                                                       next_node_uid=next_node_uid)["transmission"]
                used_token_throughput: float = flow_graph.get_link_flow(prev_node_uid=self.node_uid,
                                                                        next_node_uid=next_node_uid)["transmission"]
                speed: float = node.outbound_links[next_link_uid].bandwidth
                used_speed: float = speed * used_token_throughput / token_throughput
                self.outbound_links_latency.append(latency)
                self.outbound_links_speed.append(speed)
                self.outbound_links_used_speed.append(used_speed)
                self.outbound_links_token_throughput.append(token_throughput)
                self.outbound_links_used_token_throughput.append(used_token_throughput)

        elif self.node_type == NodeType.Sink:
            node: SinkNode
            # outbound network interface card
            # full speed (byte/s) and used speed (byte/s)
            self.outbound_nic_speed = None
            self.outbound_nic_used_speed = None
            # token throughput (#tokens/s) and used token throughput (#tokens/s)
            self.outbound_nic_token_throughput = None
            self.outbound_nic_used_token_throughput = None
            # outbound link uids (and nodes)
            self.outbound_link_uids = None
            self.outbound_node_uids = None

            # inference settings of outbound nodes
            self.outbound_simulator_nodes = None
            self.outbound_node_inference_settings = None
            self.outbound_node_model_layers = None

            # statistics of outbound links
            # full speed (byte/s) and used speed (byte/s)
            self.outbound_links_latency = None
            self.outbound_links_speed = None
            self.outbound_links_used_speed = None
            # token throughput (#tokens/s) and used token throughput (#tokens/s)
            self.outbound_links_token_throughput = None
            self.outbound_links_used_token_throughput = None

        else:
            assert False, "Unknown node type!"

        # inference-related statistics
        if self.node_type == NodeType.Compute:
            node: ComputeNode
            # token throughput (#tokens/s) and used token throughput (#tokens/s)
            self.inference_token_throughput: float = capacity_dict["inference"]
            self.inference_used_token_throughput: float = flow_dict["inference"]
            self.inference_model_layer_indices: List[int] = sorted(list(node.in_vram_model_layers.keys()))
            self.inference_settings: InferenceSettings = node.inference_settings

        elif self.node_type == NodeType.Source or self.node_type == NodeType.Sink:
            # token throughput (#tokens/s) and used token throughput (#tokens/s)
            self.inference_token_throughput = None
            self.inference_used_token_throughput = None
            self.inference_model_layer_indices = None
            self.inference_settings = None

        else:
            assert False, "Unknown node type!"

        # scheduling of request execution
        if self.node_type == NodeType.Source or self.node_type == NodeType.Compute:
            initial_loads: List[float] = [0 for _ in self.outbound_links_used_token_throughput]
            self.execution_scheduler: IWRR = IWRR(capacities=self.outbound_links_used_token_throughput,
                                                  initial_loads=initial_loads)
        else:
            self.execution_scheduler = None

    def schedule_initialization(self, reqeust: InferenceRequest) -> PipelineStage:
        """
        Schedule initialization of a request using IWRR. In initialization phase, the request can
        take any route. Therefore, we follow the MaxFlow.

        :param reqeust: request to schedule.
        :return: the next pipeline stage (a link and a node)
        """
        # check whether we can schedule this request
        assert isinstance(self.execution_scheduler, IWRR), "No execution scheduler on this node!"
        assert reqeust.phase == RequestPhase.Initialization, "Request must be in initialization phase"

        # prepare the masks
        # we mask out next nodes that: 1. has zero flow from current node
        #                              2. has max_batch_size smaller than token_seq_length
        #                              3. token throughput <= 0.05 * total used token throughput
        #                              4. kv-cache is not enough at the moment
        mask: List[bool] = []
        reason: List[str] = []   # for debugging purposes
        sum_of_used_token_throughput = sum(self.outbound_links_used_token_throughput)
        for inference_setting, token_throughput, simulator_node in zip(self.outbound_node_inference_settings,
                                                                       self.outbound_links_used_token_throughput,
                                                                       self.outbound_simulator_nodes):
            # filter 1 & 3
            if token_throughput < 0.05 * sum_of_used_token_throughput:
                mask.append(False)
                reason.append("Fail-Flow")
                continue

            # filter 2
            if inference_setting is not None and reqeust.token_seq_length > inference_setting.prompt_max_tokens:
                mask.append(False)
                reason.append("Fail-Length")
                continue

            # filter 4
            if isinstance(simulator_node, ComputeNode):
                if not self.scheduler_core.kv_expectation.check_can_add(node_uid=simulator_node.node_uid,
                                                                        input_seq_length=reqeust.token_seq_length):
                    mask.append(False)
                    reason.append("Fail-KV")
                    continue

            # pass all filters
            mask.append(True)
            reason.append("Pass")

        assert len(mask) == len(self.outbound_simulator_nodes), "Length mismatch!"

        # check whether there are feasible candidates
        if self.scheduling_mode == SchedulingMode.Online:
            # raise an error, as we must execute the request
            assert any(mask), f"No next level node meets scheduling requirement at compute {self.node_uid}!"
        elif self.scheduling_mode == SchedulingMode.Offline:
            # reject the request if no next level nodes can do the inference
            if not any(mask):
                # FIXME: restore the log here
                # print(f"[SchedulerNode-{self.node_uid}] Reject scheduler - out_nodes={self.outbound_node_uids}, "
                #       f"reasons={reason}")
                return PipelineStage(link_uid=-1, bandwidth_usage=-1, node_uid=-1, layers_to_infer=[])
        else:
            assert False, "Unknown scheduling mode!"

        # scheduling with IWRR
        chosen_index: int = self.execution_scheduler.choose_one(workload=reqeust.token_seq_length, mask=mask)
        next_link_uid = self.outbound_link_uids[chosen_index]
        next_node_uid = self.outbound_node_uids[chosen_index]

        # determine which layers to be inferred on next node
        layers_on_cur_node: List[int] or None = self.inference_model_layer_indices
        layers_on_next_node: List[int] or None = self.outbound_node_model_layers[chosen_index]
        assert not (layers_on_cur_node is None and layers_on_next_node is None), "Source is connected to sink!"
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

        # determine the max bandwidth we can use
        # limit by current link
        link_max_bandwidth: float = self.outbound_links_speed[chosen_index]
        # limit by outbound nic
        out_flow: float = self.outbound_links_used_speed[chosen_index]
        out_flow_percentage: float = out_flow / sum(self.outbound_links_used_speed)
        out_nic_limit: float = self.outbound_nic_speed * out_flow_percentage
        # limit by inbound nic on next node
        next_scheduler_node: SchedulerNode = self.scheduler_core.scheduler_nodes[next_node_uid]
        link_pos_in_next: int = next_scheduler_node.inbound_link_uids.index(next_link_uid)
        next_in_flow: float = next_scheduler_node.inbound_links_used_speed[link_pos_in_next]
        next_in_flow_percentage: float = next_in_flow / sum(next_scheduler_node.inbound_links_used_speed)
        next_in_nic_limit: float = next_in_flow_percentage * next_scheduler_node.inbound_nic_speed
        # final max bandwidth and some checks
        bandwidth_to_use: float = min(link_max_bandwidth, out_nic_limit, next_in_nic_limit)
        assert next_in_flow == self.outbound_links_used_speed[chosen_index], "Flow mismatch!"
        assert bandwidth_to_use >= next_in_flow - ATOL, "Bandwidth to use must be large than flow!"

        # construct the next pipeline stage
        next_pipeline_stage = PipelineStage(link_uid=next_link_uid, bandwidth_usage=bandwidth_to_use,
                                            node_uid=next_node_uid, layers_to_infer=layers_to_infer)

        # check that network flow does not create a very slow link by accident
        if layers_on_cur_node is None or layers_on_next_node is None:
            assert bandwidth_to_use > TOKEN_SLOW_LINK, "Found a very slow bandwidth in network flow scheduling!"
        else:
            assert bandwidth_to_use > ACT_SLOW_LINK, "Found a very slow bandwidth in network flow scheduling!"

        return next_pipeline_stage

    def reject_initialization(self, request: InferenceRequest, pipeline_stage: PipelineStage) -> None:
        """
        Reject a request before at some later point, the request can not be scheduled.

        :param request: the request to reject
        :param pipeline_stage: the pipeline stage generated by schedule_initialization on this node
        :return: None
        """
        assert self.scheduling_mode == SchedulingMode.Offline, "We can only reject requests in offline mode!"
        assert request.phase == RequestPhase.Initialization, "Can only reject initialization phase requests!"
        workload = request.token_seq_length
        index = self.outbound_node_uids.index(pipeline_stage.node_uid)
        assert index == self.outbound_link_uids.index(pipeline_stage.link_uid), "Index mismatch!"
        self.execution_scheduler.restore_one(workload=workload, index=index)

    def schedule_increment(self, request: InferenceRequest, link_uid: int, node_uid: int) -> None:
        """
        Scheduling increment (autoregressive generation) of a request. In increment phase, the request
        must follow its path during initialization, because of the KV cache.

        :param request: the request to schedule
        :param link_uid: uid of the dedicated link
        :param node_uid: uid of the dedicated node
        :return: None
        """
        # update loads over the corresponding link
        index: int = self.outbound_link_uids.index(link_uid)
        assert index == self.outbound_node_uids.index(node_uid), "Index mismatch!"
        self.execution_scheduler.update_loads(workload=request.token_seq_length, index=index)


class SchedulerCore:
    def __init__(self, cluster: ClusterSimulator, flow_graph: FlowGraph, kv_param: KVParameters,
                 scheduling_mode: SchedulingMode) -> None:
        """
        Core of global maxflow scheduler. All functionalities are implemented here.

        :param cluster: the cluster that this scheduler is bind to
        :param flow_graph: flow graph of the cluster
        :param kv_param: kv cache parameters
        :return: None
        """
        # basic dependencies
        self.cluster: ClusterSimulator = cluster
        self.flow_graph: FlowGraph = flow_graph

        # kv cache expectation
        self.kv_expectation: KVExpectation = KVExpectation(kv_param=kv_param)
        self.kv_param: KVParameters = kv_param

        # scheduling mode
        self.scheduling_mode: SchedulingMode = scheduling_mode

        # scheduler core's flow graph version
        self.creation_time_stamp: float or None = None
        self.scheduler_nodes: Dict[int, SchedulerNode] or None = None

    def update(self, time_stamp) -> None:
        """
        Update the scheduler based on the latest cluster and flow graph.

        :param time_stamp: time stamp of this update (simulation time)
        :return: None
        """
        # check flow graph compatibility
        assert time_stamp == self.flow_graph.flow_graph_timestamp, "Incompatible flow graph!"
        assert self.creation_time_stamp is None or self.creation_time_stamp < self.flow_graph.flow_graph_timestamp, \
            "Error: trying to update scheduler core using an older version of flow graph!"

        # rebuild topology
        if self.creation_time_stamp is not None:
            # TODO: in this branch, we are updating an existing scheduler in node failure, need to
            #  implement weight / loads migration
            self.creation_time_stamp = time_stamp
            raise NotImplementedError

        else:
            # build for the first time
            self.creation_time_stamp = time_stamp
            self.scheduler_nodes: Dict[int, SchedulerNode] = {}

            # add source
            source_node = SchedulerNode(node=self.cluster.source_node, flow_graph=self.flow_graph,
                                        scheduler_core=self, scheduling_mode=self.scheduling_mode)
            self.scheduler_nodes[self.cluster.source_node.node_uid] = source_node

            # add sink
            sink_node = SchedulerNode(node=self.cluster.sink_node, flow_graph=self.flow_graph,
                                      scheduler_core=self, scheduling_mode=self.scheduling_mode)
            self.scheduler_nodes[self.cluster.sink_node.node_uid] = sink_node

            # add compute nodes
            for compute_node_uid, compute_node in self.cluster.compute_nodes.items():
                scheduler_node = SchedulerNode(node=compute_node, flow_graph=self.flow_graph,
                                               scheduler_core=self, scheduling_mode=self.scheduling_mode)
                self.scheduler_nodes[compute_node_uid] = scheduler_node

            # update kv-cache expectation
            self.kv_expectation.initialize(simulator=self.cluster)

    def schedule(self, request: InferenceRequest) -> bool:
        """
        Schedule a request based on its phase.

        :param request: the request to schedule
        :return: scheduling succeeded or not
        """
        if request.phase == RequestPhase.Initialization:
            # in initialization phase, we allocate a path for this request based on MaxFlow
            pipeline: List[PipelineStage] = []
            current_node_uid: int = self.cluster.source_node.node_uid
            scheduling_succeeded = True
            while not current_node_uid == -1:
                # scheduler on current node
                current_scheduler_node: SchedulerNode = self.scheduler_nodes[current_node_uid]
                next_stage: PipelineStage = current_scheduler_node.schedule_initialization(reqeust=request)

                # check scheduling status
                if self.scheduling_mode == SchedulingMode.Online:
                    # online mode: scheduling must succeed
                    assert not next_stage.link_uid == -1, "Can not discard request in online mode!"
                    pipeline.append(next_stage)
                elif self.scheduling_mode == SchedulingMode.Offline:
                    if not next_stage.link_uid == -1:
                        # scheduling succeeded
                        pipeline.append(next_stage)
                    else:
                        # scheduling fails
                        scheduling_succeeded = False
                        break
                else:
                    assert False, "Unknown scheduling mode!"

                # switch to next node
                if not next_stage.node_uid == self.cluster.sink_node.node_uid:
                    current_node_uid = next_stage.node_uid
                else:
                    current_node_uid = -1

            # check if route scheduling is successful
            if not scheduling_succeeded:
                reject_node_uid = self.cluster.source_node.node_uid
                for pipeline_stage in pipeline:
                    reject_scheduler_node: SchedulerNode = self.scheduler_nodes[reject_node_uid]
                    reject_scheduler_node.reject_initialization(request=request,
                                                                pipeline_stage=pipeline_stage)
                    reject_node_uid = pipeline_stage.node_uid
                return False

            # route scheduling is successful, set pipeline
            request.set_pipeline(pipeline=pipeline)

            # after we have determined the pipeline, register it in kv expectation
            # need to exclude last stage, as it must lead to sink node
            nodes_used_in_pipeline: List[int] = []
            start_layer_idx_list: List[int] = []
            end_layer_idx_list: List[int] = []
            for pipeline_stage in pipeline[:-1]:
                nodes_used_in_pipeline.append(pipeline_stage.node_uid)
                start_layer_idx_list.append(pipeline_stage.layers_to_infer[0])
                end_layer_idx_list.append(pipeline_stage.layers_to_infer[-1] + 1)
            self.kv_expectation.add_request(input_seq_length=request.token_seq_length,
                                            route=nodes_used_in_pipeline,
                                            start_idx_list=start_layer_idx_list,
                                            end_idx_list=end_layer_idx_list)

        elif request.phase == RequestPhase.Increment:
            # in increment phase, the request follows the same route in its initialization phase,
            # we need to update loads of all nodes that it passes
            assert request.pipeline_set, "Request in increment phase must have an allocated pipeline!"
            current_node_uid: int = self.cluster.source_node.node_uid
            for pipeline_stage in request.mini_pipeline:
                # update loads of current node
                self.scheduler_nodes[current_node_uid].schedule_increment(request=request,
                                                                          link_uid=pipeline_stage.link_uid,
                                                                          node_uid=pipeline_stage.node_uid)
                current_node_uid = pipeline_stage.node_uid

        else:
            assert False, "Found request with unknown phase!"

        # check scheduling makes a continuous model inference
        expected_layer_idx = 0
        for pipeline_stage in request.mini_pipeline:
            if pipeline_stage.layers_to_infer is None:
                break
            assert expected_layer_idx == pipeline_stage.layers_to_infer[0], "Model not continuous!"
            expected_layer_idx = pipeline_stage.layers_to_infer[-1] + 1
        assert expected_layer_idx == len(self.cluster.model), "Scheduling is incomplete!"
        return True

    def remove_from_kv_expectation(self, input_seq_length: int, route: List[int],
                                   start_idx_list: List[int], end_idx_list: List[int]) -> None:
        """
        Remove a request. (Call after last decode finished)

        :param input_seq_length: input sequence length
        :param route: a list of node uids
        :param start_idx_list: start layer idx list (inclusive)
        :param end_idx_list: end layer idx list (exclusive)
        :return: None
        """
        self.kv_expectation.remove_request(input_seq_length=input_seq_length, route=route,
                                           start_idx_list=start_idx_list, end_idx_list=end_idx_list)
