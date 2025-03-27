# 2023.12.16 Yixuan Mei

import copy
import configparser
import math
import os.path

import networkx as nx
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Set, Any, Optional, TYPE_CHECKING
from queue import PriorityQueue

from simulator.event_simulator.utils import BASE_NODE_UID, BASE_LINK_UID, BASE_EVENT_UID, BASE_REQUEST_UID
from simulator.event_simulator.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec
from simulator.event_simulator.model import ModelLayer, create_model
from simulator.event_simulator.logger import Logger
from simulator.event_simulator.kv_cache import KVTracker, KVCache
from simulator.event_simulator.coordinator_node import SourceNode, SinkNode
from simulator.event_simulator.compute_node import ComputeNode, InferenceBatch
from simulator.event_simulator.network_link import NetworkLink, LinkStatus, TransmissionObject, TransmissionType
from simulator.event_simulator.request import InferenceRequest, RequestPhase, RequestLocation, PipelineStage
from simulator.event_simulator.event import EventDescription, Event, EventHandler
from simulator.event_simulator.base_node import NodeType
from simulator.event_simulator.query_manager import QueryManager, QueryManagerParameters
from simulator.model_manager.model_manager import ModelName, ModelManager
from simulator.scheduler.base_scheduler import BaseScheduler, TransmissionSchedule, ExecutionSchedule, SchedulingMethod

if TYPE_CHECKING:
    from simulator.trace_generator.simulator_query_feeder import OfflineRequestFeeder


class ClusterSimulator:
    def __init__(self, model_name: ModelName, machine_num_dict: Dict[str, int]) -> None:
        """
        Create an empty cluster simulator.

        :param model_name: name of the LLM to simulate
        :param machine_num_dict: {machine_name -> num of machine}
        :return: None
        """
        # global uid record
        self.next_node_uid: int = BASE_NODE_UID
        self.next_link_uid: int = BASE_LINK_UID
        self.next_event_uid: int = BASE_EVENT_UID
        self.next_request_uid: int = BASE_REQUEST_UID

        # simulation time
        self.ready_to_simulate: bool = False
        self.current_time: float = 0

        # nodes and links
        self.machine_types: List[str] = []
        self.source_node: SourceNode or None = None
        self.sink_node: SinkNode or None = None
        self.compute_nodes: Dict[int, ComputeNode] = {}
        self.links: Dict[int, NetworkLink] = {}

        # a mapping from name to nodes and links (only available when loading from a cluster file)
        self.name_2_compute_node: Dict[str, ComputeNode] = {}
        self.name_2_link: Dict[str, NetworkLink] = {}

        # model
        # model_manager: stores the profiling results, etc.
        # model: the real model (full)
        self.model_name: ModelName = model_name
        self.model_manager: ModelManager = ModelManager(model_name=model_name, machine_num_dict=machine_num_dict)
        self.model: Dict[int, ModelLayer] = {}

        # request tracker
        self.requests_on_the_fly: Dict[int, InferenceRequest] = {}
        self.finished_requests: Dict[int, Tuple[float, InferenceRequest]] = {}

        # event queue
        self.event_queue: PriorityQueue[Tuple[float, Event]] = PriorityQueue()
        self.previous_events_list: List[Tuple[float, Event]] = []
        self.previous_events_dict: Dict[int, Event] = {}

        # scheduler
        self.scheduler: BaseScheduler or None = None

        # query manager
        self.query_manager: QueryManager or None = None

        # offline query feeder
        self.use_offline_mode = False
        self.offline_query_feeder: Optional["OfflineRequestFeeder"] = None

        # logger
        self.last_watch_time: Optional[float] = None
        self.logger: Logger = Logger()

    # ********************************* Uid Management ********************************* #
    def get_next_node_uid(self) -> int:
        """
        Get a new node uid for creating new nodes.

        :return: next node uid
        """
        new_node_uid = self.next_node_uid
        self.next_node_uid += 1
        return new_node_uid

    def get_next_link_uid(self) -> int:
        """
        Get a new link uid for creating new links.

        :return: next link uid
        """
        new_link_uid = self.next_link_uid
        self.next_link_uid += 1
        return new_link_uid

    def get_next_event_uid(self) -> int:
        """
        Get a new event uid for creating new events.

        :return: next event uid
        """
        new_event_uid = self.next_event_uid
        self.next_event_uid += 1
        return new_event_uid

    def get_next_request_uid(self) -> int:
        """
        Get a new request uid for creating new requests.

        :return: next request uid
        """
        new_request_uid = self.next_request_uid
        self.next_request_uid += 1
        return new_request_uid

    # ********************************* Create Cluster ********************************* #
    # initialization:
    #   1. initialize cluster (nodes, links and model)
    #       1.1 call "initialize" + "add_compute_node" + "add_network_link"
    #       1.2 or call "from_ini_file" to initialize from an ini file
    #   2. set scheduler: call "set_scheduler" function
    #   3. set query manager: call "set_query_manager" function
    #   4. set machine statistics: call "set_machine_statistics"
    #   5. mark the cluster as ready to execute: call "mark_as_ready"
    #
    # after topology change:
    #   1. update scheduler: call "update_scheduler"
    #
    # simulation:
    #   1. issue new command: call "issue_command_new_request"
    #   2. issue model loading: call "issue_command_load_model"
    #   3. simulate: call "simulate"
    #

    def initialize(self, coordinator_inbound_nic_speed: float, coordinator_outbound_nic_speed: float,
                   full_model: Dict[int, ModelLayer], machine_types: List[str]) -> (SourceNode, SinkNode):
        """
        Initialize the cluster coordinator (i.e. source and sink node) and the full model used in the
        cluster. Note the model here does not inference statistics.

        :param coordinator_inbound_nic_speed: for receiving finished requests
        :param coordinator_outbound_nic_speed: for issuing requests to compute nodes
        :param full_model: a model with no inference statistics (layer id start from 0)
        :param machine_types: all types of machine in the cluster
        :return: a reference to source and sink node (i.e. two endpoints of the coordinator)
        """
        # set source and sink node uid
        source_uid = self.get_next_node_uid()
        sink_uid = self.get_next_node_uid()
        assert source_uid == BASE_NODE_UID and sink_uid == BASE_NODE_UID + 1, "Coordinator must be the first node!"

        # check that the model is consistent and store it in the simulator
        num_layers = len(full_model)
        assert sorted(list(full_model.keys())) == list(range(len(full_model))), "Inconsistent model found!"
        self.model = copy.deepcopy(full_model)

        # store machine type in cluster
        assert len(machine_types) == len(set(machine_types)), "Found duplicate machine types"
        self.machine_types = copy.deepcopy(machine_types)

        # create source and sink node
        self.source_node = SourceNode(node_uid=source_uid, outbound_nic_speed=coordinator_outbound_nic_speed)
        self.sink_node = SinkNode(node_uid=sink_uid, inbound_nic_speed=coordinator_inbound_nic_speed,
                                  total_num_layers=num_layers)

        # return reference
        return self.source_node, self.sink_node

    def add_compute_node(self, vram_size: float, inbound_nic_speed: float, outbound_nic_speed: float,
                         disk_speed: float, machine_type: str, kv_cache_capacity: int,
                         activation_backup_capacity: int) -> ComputeNode:
        """
        Initialize a compute node in the cluster.

        :param vram_size: total memory size of GPUs on this node
        :param inbound_nic_speed: network speed of receiving packages
        :param outbound_nic_speed: network speed of sending packages
        :param disk_speed: disk speed for loading models
        :param machine_type: type of this machine (e.g. A100, T4, etc.)
        :param kv_cache_capacity: how many tokens can be stored in the kv cache on this node
        :param activation_backup_capacity: how many tokens can be stored in the activation backup cache on this node
        :return: a reference to the new compute node
        """
        # get node uid and create node
        new_node_uid: int = self.get_next_node_uid()
        assert new_node_uid not in self.compute_nodes, "Duplicate node uid found!"
        assert machine_type in self.machine_types, "Unknown machine type found!"
        new_compute_node: ComputeNode = ComputeNode(node_uid=new_node_uid, vram_size=vram_size,
                                                    inbound_nic_speed=inbound_nic_speed,
                                                    outbound_nic_speed=outbound_nic_speed,
                                                    disk_speed=disk_speed,
                                                    machine_type=machine_type,
                                                    kv_cache_capacity=kv_cache_capacity,
                                                    activation_backup_capacity=activation_backup_capacity)

        # put into node list and return
        self.compute_nodes[new_node_uid] = new_compute_node
        return new_compute_node

    def add_network_link(self, node_in_uid: int, node_out_uid: int, latency: float, bandwidth: float) -> NetworkLink:
        """
        Add a network link that connects two nodes.

        :param node_in_uid: uid of input node
        :param node_out_uid: uid of output node
        :param latency: latency of the link
        :param bandwidth: bandwidth of the link
        :return: a reference to the new network link
        """
        # find inbound node
        assert not node_in_uid == self.sink_node.node_uid, "Sink can have no output network link!"
        if node_in_uid == self.source_node.node_uid:
            node_in = self.source_node
        else:
            assert node_in_uid in self.compute_nodes, "Input node not found when creating link!"
            node_in = self.compute_nodes[node_in_uid]

        # find outbound node
        assert not node_out_uid == self.source_node.node_uid, "Source can have no input network link!"
        if node_out_uid == self.sink_node.node_uid:
            node_out = self.sink_node
        else:
            assert node_out_uid in self.compute_nodes, "Output node not found when creating link!"
            node_out = self.compute_nodes[node_out_uid]

        # create link
        new_link_uid = self.get_next_link_uid()
        assert new_link_uid not in self.links, "Duplicate links found"
        new_link = NetworkLink(link_uid=new_link_uid, node_in=node_in, node_out=node_out,
                               latency=latency, bandwidth=bandwidth)

        # connect the link to two endpoints and store it in simulator
        node_in.add_outbound_link(outbound_link=new_link)
        node_out.add_inbound_link(inbound_link=new_link)
        self.links[new_link_uid] = new_link

        # return a reference
        return new_link

    def from_ini_file(self, config_file_name: str) -> \
            (SourceNode, SinkNode, Dict[str, ComputeNode], Dict[str, NetworkLink]):
        """
        Load configuration of a cluster from an ini file.

        :param config_file_name: name of the config file
        :return: ref to source node, sink node, all compute nodes and links.
        """
        # make sure that we start from a new simulator
        assert self.current_time == 0 and not self.ready_to_simulate, "Cluster should be empty!"
        assert self.source_node is None and self.sink_node is None and len(self.model) == 0, "Cluster should be empty!"
        assert len(self.compute_nodes) == 0 and len(self.links) == 0, "Cluster should be empty!"

        # construct the parser
        config = configparser.ConfigParser()
        config.read(config_file_name)

        # parse the coordinator and the model
        _coordinator_inbound_nic_speed: float = eval(config["Coordinator"]["inbound_nic_speed"])
        _coordinator_outbound_nic_speed: float = eval(config["Coordinator"]["outbound_nic_speed"])
        _model_param_sizes: List[float] = self.model_manager.get_model_params()
        _full_model: Dict[int, ModelLayer] = create_model(layer_parameter_sizes=_model_param_sizes)
        _machine_types: List[str] = eval(config["MachineTypes"]["types"])
        source_node, sink_node = self.initialize(coordinator_inbound_nic_speed=_coordinator_inbound_nic_speed,
                                                 coordinator_outbound_nic_speed=_coordinator_outbound_nic_speed,
                                                 full_model=_full_model,
                                                 machine_types=_machine_types)

        # parse compute nodes
        compute_node_names: List[str] = eval(config["ComputeNodes"]["names"])
        compute_nodes: Dict[str, ComputeNode] = {}
        for compute_node_name in compute_node_names:
            assert compute_node_name not in compute_nodes, "Found duplicate compute node definitions!"

            # parameters
            _vram_size: float = eval(config[compute_node_name]["vram_size"])
            _inbound_nic_speed: float = eval(config[compute_node_name]["inbound_nic_speed"])
            _outbound_nic_speed: float = eval(config[compute_node_name]["outbound_nic_speed"])
            _disk_speed: float = eval(config[compute_node_name]["disk_speed"])
            _machine_type: str = eval(config[compute_node_name]["machine_type"])
            _kv_cache_capacity: int = eval(config[compute_node_name]["kv_cache_capacity"])
            _activation_backup_capacity: int = eval(config[compute_node_name]["activation_backup_capacity"])

            # construct
            _new_compute_node = self.add_compute_node(vram_size=_vram_size,
                                                      inbound_nic_speed=_inbound_nic_speed,
                                                      outbound_nic_speed=_outbound_nic_speed,
                                                      disk_speed=_disk_speed,
                                                      machine_type=_machine_type,
                                                      kv_cache_capacity=_kv_cache_capacity,
                                                      activation_backup_capacity=_activation_backup_capacity)
            compute_nodes[compute_node_name] = _new_compute_node

        # parse links
        link_names: List[str] = eval(config["Links"]["names"])
        links: Dict[str, NetworkLink] = {}
        for link_name in link_names:
            assert link_name not in links, "Found duplicate link definitions!"

            # parameters
            _node_in_name = config[link_name]["in"]
            _node_out_name = config[link_name]["out"]
            _latency = eval(config[link_name]["latency"])
            _bandwidth = eval(config[link_name]["bandwidth"])

            def parse_node_name(node_name):
                if node_name == "source":
                    return source_node
                elif node_name == "sink":
                    return sink_node
                else:
                    assert node_name in compute_nodes, "Found reference to undefined compute nodes!"
                    return compute_nodes[node_name]

            _node_in = parse_node_name(node_name=_node_in_name)
            _node_out = parse_node_name(node_name=_node_out_name)

            # construct
            _new_link = self.add_network_link(node_in_uid=_node_in.node_uid,
                                              node_out_uid=_node_out.node_uid,
                                              latency=_latency,
                                              bandwidth=_bandwidth)
            links[link_name] = _new_link

        # set mapping from name to compute nodes and links
        self.name_2_compute_node = compute_nodes
        self.name_2_link = links

        # return refs to source, sink, all compute nodes and links
        return source_node, sink_node, compute_nodes, links

    def set_scheduler(self, scheduler: BaseScheduler) -> None:
        """
        Set scheduler that will be used in the simulator.
        Note: use init_scheduler for simpler scheduler initialization

        :param scheduler: the scheduler to use
        :return: None
        """
        self.scheduler = scheduler

    def init_scheduler(self, scheduling_method: SchedulingMethod, args: Optional[Dict[str, Any]] = None) -> None:
        """
        Set scheduler that will be used in the simulator.
        SchedulingMethod.MaxFlow:
            1. "kv_param": KVParameters
            2. "scheduling_mode": SchedulingMode
        SchedulingMethod.Swarm:
            /
        SchedulingMethod.Naive:
            /

        :param scheduling_method: scheduling method
        :param args: arguments for the scheduler
        :return: None
        """
        assert self.scheduler is None, "Trying the init scheduler when there is one already!"
        if scheduling_method == SchedulingMethod.MaxFlow:
            # MaxFlow
            from simulator.scheduler.global_maxflow.global_maxflow_scheduler import FlowParameters
            from simulator.scheduler.global_maxflow.global_maxflow_scheduler import GlobalFlowScheduler
            flow_params = FlowParameters(token_size=self.model_manager.get_model_token_size(),
                                         token_activation_size=self.model_manager.get_model_activation_size())
            self.scheduler = GlobalFlowScheduler(parameters=flow_params, simulator=self, kv_param=args["kv_param"],
                                                 scheduling_mode=args["scheduling_mode"])

        elif scheduling_method == SchedulingMethod.Swarm:
            # Swarm
            from simulator.scheduler.swarm.swarm_scheduler import SwarmParameters
            from simulator.scheduler.swarm.swarm_scheduler import SwarmScheduler
            swarm_params = SwarmParameters(initial_priority=0.05,
                                           smoothing_parameter=0.8,
                                           max_on_the_fly_request=99999)
            self.scheduler = SwarmScheduler(parameters=swarm_params)

        elif scheduling_method == SchedulingMethod.Naive:
            # Naive
            from simulator.scheduler.naive_scheduler import NaiveScheduler
            self.scheduler = NaiveScheduler()

        elif scheduling_method == SchedulingMethod.ShortestQueue:
            # ShortestQueue
            from simulator.scheduler.shortest_queue.shortest_queue_scheduler import ShortestQueueScheduler
            self.scheduler = ShortestQueueScheduler()

        else:
            assert False, "Found unknown scheduling method!"

    def update_scheduler(self) -> None:
        """
        Update the scheduler. Behavior depends on scheduler type:
        MaxFlowScheduler: construct cluster topology and compute max flow from current simulator

        :return: None
        """
        # import names here to avoid circular import
        from simulator.scheduler.global_maxflow.global_maxflow_scheduler import GlobalFlowScheduler
        from simulator.scheduler.local_maxflow.maxflow_scheduler import MaxFlowScheduler
        from simulator.scheduler.swarm.swarm_scheduler import SwarmScheduler

        # update scheduler based on type
        if isinstance(self.scheduler, MaxFlowScheduler):
            # update topology and flow
            self.scheduler.update_topology_and_flow(cluster=self)

            # logging
            self.logger.add_log(log_time=self.current_time,
                                entity_name="MaxFlow Scheduler",
                                activity="Update topology and flow.",
                                description=f"Max flow: {self.scheduler.cluster_topology.max_flow}, "
                                            f"flow dict: {self.scheduler.cluster_topology.flow_dict}")
        elif isinstance(self.scheduler, SwarmScheduler):
            # update topology
            self.scheduler.reset_topology(cluster=self)

            # logging
            self.logger.add_log(log_time=self.current_time,
                                entity_name="Swarm Scheduler",
                                activity="Update topology.",
                                description=f"Number of source + compute nodes: {len(self.scheduler.nodes)}.")
        elif isinstance(self.scheduler, GlobalFlowScheduler):
            # update topology
            self.scheduler.update_scheduler(time_stamp=self.current_time)

            # logging
            self.logger.add_log(log_time=self.current_time,
                                entity_name="Global MaxFlow Scheduler",
                                activity="Update scheduler.",
                                description=f"Max flow: {self.scheduler.flow_graph.flow_value}, "
                                            f"flow dict: {self.scheduler.flow_graph.flow_dict}")

        else:
            # logging (warning)
            self.logger.add_log(log_time=self.current_time,
                                entity_name="Unknown Scheduler",
                                activity="Warning: Nothing done in scheduler update!",
                                description="")

    def set_query_manager(self, query_manager: QueryManager) -> None:
        """
        Set scheduler that will be used in the simulator.
        Note: use init_query_manager for simpler scheduler initialization

        :param query_manager: the query manager
        :return: None
        """
        self.query_manager = query_manager

    def init_query_manager(self) -> QueryManager:
        """
        Init query manager in the cluster.

        :return: QueryManager
        """
        assert self.query_manager is None, "Trying to init query manager when there is one already!"
        params = QueryManagerParameters(token_size=self.model_manager.get_model_token_size(),
                                        token_activation_size=self.model_manager.get_model_activation_size(),
                                        total_num_layers=self.model_manager.get_num_layers())
        self.query_manager = QueryManager(param=params, simulator=self)
        return self.query_manager

    def register_offline_query_feeder(self, offline_query_feeder: "OfflineRequestFeeder") -> None:
        """
        Register offline query feeder.

        :param offline_query_feeder: the offline query feeder
        :return: None
        """
        self.use_offline_mode = True
        self.offline_query_feeder = offline_query_feeder

    def mark_as_ready(self) -> None:
        """
        Mark the simulator as ready to simulate.

        :return: None
        """
        # a few quick checks
        assert self.ready_to_simulate is False, "Simulator is already ready"
        assert self.current_time == 0, "Bad timer at start!"
        assert not len(self.model) == 0, "No model found in simulator!"
        assert self.source_node is not None and self.sink_node is not None, "Source / sink not initialized!"
        assert not len(self.compute_nodes) == 0, "No compute node in cluster!"
        assert self.scheduler is not None, "No scheduler found in simulator!"
        assert self.query_manager is not None, "No query manager found in simulator!"

        # mark as ready
        self.ready_to_simulate = True

    # ****************************** Commands and Events ******************************* #
    def handle_command_new_request(self, event: Event) -> bool:
        """
        Handle event: new request arrives at the cluster. event.args should have fileds:
            "request": InferenceReqeust

        :param event: the event to handle
        :return: whether the new request is submitted or not
        """
        # build the request
        new_request: InferenceRequest = event.args["request"]
        new_request_uid: int = new_request.request_uid

        # set global routing
        from simulator.scheduler.global_maxflow.global_maxflow_scheduler import GlobalFlowScheduler, SchedulingMode
        if isinstance(self.scheduler, GlobalFlowScheduler):
            succeeded = self.scheduler.generate_schedule(request=new_request)
            if self.scheduler.scheduling_mode == SchedulingMode.Online:
                assert succeeded, "Found request with failed scheduling in online mode (potential bug)!"
            elif self.scheduler.scheduling_mode == SchedulingMode.Offline:
                assert succeeded or new_request.phase == RequestPhase.Initialization, \
                    "Found request in increment phase with failed scheduling (potential bug)!"
                if not succeeded:
                    # the request must be in initialization phase
                    # scheduling fails because of kv cache, need to inform query manager to remove the query
                    self.query_manager.reject_query(request=new_request)
                    return False
            else:
                assert False, "Unknown scheduling mode!"

        # put the request in source node's outbound queue
        self.source_node.issue_request(request=new_request)

        # track the request in simulator
        assert new_request_uid not in self.requests_on_the_fly and new_request_uid not in self.finished_requests, \
            "Duplicate request found!"
        self.requests_on_the_fly[new_request_uid] = new_request

        # logging
        assert event.event_time == self.current_time, "Time discrepancy found!"
        self.logger.add_log(log_time=self.current_time,
                            entity_name=self.source_node.entity_name,
                            activity="Issue new request at coordinator.",
                            description=new_request.get_description())
        return True

    def handle_command_load_model(self, event: Event) -> bool:
        """
        Handle event: load model for a compute node in cluster. event.args should have fileds:
            "node_uid": int
            "new_layer_ids": List[int]
            "request_uids_to_wait": List[int]
        Note: "new_layer_statistics": layer_id -> bs2time, bs2vram

        :param event: the event to handle
        :return: whether we can start loading model right away
        """
        # extract input parameters
        node_uid: int = event.args["node_uid"]
        new_layer_ids: List[int] = event.args["new_layer_ids"]
        request_uids_to_wait: List[int] = event.args["request_uids_to_wait"]

        # locate the node we want to load model
        assert node_uid in self.compute_nodes, "Unknown compute node uid!"
        compute_node: ComputeNode = self.compute_nodes[node_uid]

        # check whether the layers are continuous
        assert new_layer_ids == list(range(min(new_layer_ids), max(new_layer_ids) + 1)), "Model should be continuous"

        # check whether the requests we want to wait exist
        for request_uid in request_uids_to_wait:
            assert request_uid in self.requests_on_the_fly or request_uid in self.finished_requests, \
                "Found request that does not exist!"

        # create the model that will be used for loading (setting statistics)
        new_model_dict: Dict[int, ModelLayer] = {}
        for layer_id in sorted(new_layer_ids):
            # get the layer and copy it
            assert layer_id in self.model, "Try to load a layer that does not exist!"
            cur_layer = copy.deepcopy(self.model[layer_id])

            # set statistics
            machine_profile = self.model_manager.get_profiling_results(machine_type=compute_node.machine_type)
            cur_layer.set_layer_statistics(machine_profile=machine_profile)

            # save into new model
            new_model_dict[layer_id] = cur_layer

        # get new inference settings
        new_inference_settings = self.model_manager.get_inference_settings(machine_type=compute_node.machine_type,
                                                                           num_on_node_layers=len(new_layer_ids))

        # load model
        ready_to_load: bool = compute_node.prepare_loading_model(new_model_layers=new_model_dict,
                                                                 request_uids_to_wait=request_uids_to_wait,
                                                                 new_inference_settings=new_inference_settings)

        # logging
        assert event.event_time == self.current_time, "Time discrepancy found!"
        self.logger.add_log(log_time=self.current_time,
                            entity_name=compute_node.entity_name,
                            activity="Receive load model command.",
                            description=f"New layers: {new_layer_ids}, request_uids to wait: {request_uids_to_wait}, "
                                        f"inference settings: {new_inference_settings.get_description()}.")

        # return
        return ready_to_load

    def handle_start_transmission(self, event: Event) -> Tuple[Dict[str, Tuple[int, float]], List[int]]:
        """
        Handle event: start transmission for a node. event.args should have fileds:
            "node": ComputeNode or SourceNode
        This function calls the scheduler for determining which requests to send over which link.

        :param event: the event to handle
        :return: a dict of transmission object handle -> (link_uid, finish_sending_time), a list of node uids
                 that needs to call start_transmission again
        """
        # determine which node initiates the transmission
        transmission_node: ComputeNode or SourceNode = event.args["node"]
        assert not transmission_node.node_type == NodeType.Sink, "Coordinator's receiver (sink) can not send!"

        # call scheduler to find out which inference requests will be sent
        schedules: List[TransmissionSchedule]
        retransmission_node_uids: List[int]
        schedules, retransmission_node_uids = self.scheduler.schedule_transmission(node=transmission_node)

        # check that there are no backups made when sending from source node
        if isinstance(transmission_node, SourceNode):
            for schedule in schedules:
                assert schedule.transmission_type == TransmissionType.NormalExecution, \
                    "Found backup attempts when trying to transmit from source node!"

        # check whether the schedule is feasible (based on bandwidth)
        link_bandwidth_requirement: Dict[int, float] = {link_uid: 0 for link_uid in transmission_node.outbound_links}
        for schedule in schedules:
            assert schedule.link_uid in link_bandwidth_requirement, "Found schedule using unknown link!"
            link_bandwidth_requirement[schedule.link_uid] += schedule.bandwidth_usage
        for link_uid in link_bandwidth_requirement:
            link: NetworkLink = transmission_node.outbound_links[link_uid]
            total_bandwidth: float = link_bandwidth_requirement[link_uid]
            link_status: LinkStatus = link.check_availability(planned_bandwidth=total_bandwidth)
            assert link_status == LinkStatus.Available, f"Unable to schedule send over link (status: {link_status})!"

        # define some data structures that will be used in transmission
        # uids of all requests that are scheduled for normal execution
        scheduled_request_uids: Set[int] = set()
        # request uid -> link uid for requests in execution and backup
        execution_request_to_link: Dict[int, int] = {}
        backup_request_to_link: Dict[int, List[int]] = {}
        # transmission object handle -> (link uid, finishing time)
        transmission_object_end_time: Dict[str, Tuple[int, float]] = {}

        # send the requests over outbound links based on the schedules
        for schedule in schedules:
            # gather information about requests
            if schedule.transmission_type == TransmissionType.NormalExecution:
                # for normal execution, each request should only appear once
                for request in schedule.requests:
                    assert request.request_uid not in scheduled_request_uids, \
                        "Duplicate request scheduled in transmission!"
                    scheduled_request_uids.add(request.request_uid)
                    execution_request_to_link[request.request_uid] = schedule.link_uid
            elif schedule.transmission_type == TransmissionType.ActivationBackup:
                # for activation backup, a request may appear multiple times
                for request in schedule.requests:
                    if request.request_uid not in backup_request_to_link:
                        backup_request_to_link[request.request_uid] = [schedule.link_uid]
                    else:
                        backup_request_to_link[request.request_uid].append(schedule.link_uid)
            else:
                assert False, "Unknown transmission type!"

            # find the target link and count passed requests & tokens
            assert schedule.link_uid in transmission_node.outbound_links, "Unknown link found in schedule!"
            cur_link: NetworkLink = transmission_node.outbound_links[schedule.link_uid]
            if schedule.transmission_type == TransmissionType.NormalExecution:
                transmission_node.link_request_counters[schedule.link_uid] += len(schedule.requests)
                transmission_node.link_token_counters[schedule.link_uid] += sum(
                    [request.token_seq_length for request in schedule.requests])
            elif schedule.transmission_type == TransmissionType.ActivationBackup:
                transmission_node.link_backup_request_counters[schedule.link_uid] += len(schedule.requests)
                transmission_node.link_backup_token_counters[schedule.link_uid] += sum(
                    [request.token_seq_length for request in schedule.requests])
            else:
                assert False, "Unknown transmission type!"

            # send scheduled requests over link
            cur_trans: TransmissionObject = cur_link.start_sending(requests=schedule.requests,
                                                                   bandwidth_requirement=schedule.bandwidth_usage,
                                                                   transmission_type=schedule.transmission_type)

            # update request locations if the schedule is normal execution
            if schedule.transmission_type == TransmissionType.NormalExecution:
                for request in schedule.requests:
                    assert request.current_location_uid == transmission_node.node_uid, "Mis-routed request!"
                    assert request.current_location == RequestLocation.SourceNode \
                           or request.current_location == RequestLocation.ComputeNode, "Mis-routed request!"
                    request.update_location(new_location=RequestLocation.Link, new_location_uid=cur_link.link_uid,
                                            arrive_time=self.current_time)

            # record handle and link uid & finish sending time
            cur_trans_handle: str = cur_trans.get_handle()
            assert cur_trans_handle not in transmission_object_end_time, "Duplicate requests scheduled!"
            finish_sending_time: float = self.current_time + cur_trans.duration - cur_link.latency
            transmission_object_end_time[cur_trans_handle] = (cur_link.link_uid, finish_sending_time)

        # check that backup requests are not scheduled through the same path as normal execution
        # Note: the check here can not block all such attempts (since normal execution might be sent
        # in a previous round). However, the receive_backup function in compute node will capture all
        # such attempts.
        for request_uid, backup_link_uids in backup_request_to_link.items():
            if request_uid in execution_request_to_link:
                assert execution_request_to_link[request_uid] not in backup_link_uids, \
                    "Trying to backup a request using the same link and node as normal execution"

        # update the outbound request dict of sender
        for scheduled_request_uid in scheduled_request_uids:
            assert scheduled_request_uid in transmission_node.outbound_request_dict, "Unknown request!"
            del transmission_node.outbound_request_dict[scheduled_request_uid]

        # logging
        assert event.event_time == self.current_time, "Time discrepancy found!"
        self.logger.add_log(log_time=self.current_time,
                            entity_name=transmission_node.entity_name,
                            activity="Start transmission.",
                            description=f"{[schedule.get_description() for schedule in schedules]}",
                            is_empty=len(schedules) == 0)

        # return transmission object handle and corresponding end time for creating finish transmission event
        return transmission_object_end_time, retransmission_node_uids

    def handle_finish_sending(self, event: Event) -> Tuple[str, int, float]:
        """
        Handle event: the node finishes sending, link resource can be deallocated. event.args should have fileds:
            "transmission_node": ComputeNode or SourceNode
            "transmission_object_handle": str
            "link_uid": int
            "send_end_time": float

        :param event: the event to handle
        :return: transmission_object_handle, link_uid, finish_transmission_time
        """
        # extract parameters
        transmission_node: ComputeNode or SourceNode = event.args["transmission_node"]
        handle: str = event.args["transmission_object_handle"]
        link_uid: int = event.args["link_uid"]
        send_end_time: float = event.args["send_end_time"]
        assert send_end_time == event.event_time and send_end_time == self.current_time, "Time mismatch!"

        # deallocate the resources and put into finishing requests
        assert link_uid in transmission_node.outbound_links, "Unknown link!"
        cur_link: NetworkLink = transmission_node.outbound_links[link_uid]
        cur_transmission_object: TransmissionObject = cur_link.finish_sending(transmission_object_handle=handle)

        # logging
        assert event.event_time == self.current_time, "Time discrepancy found!"
        _finished_request_uids: List[int] = [request.request_uid for request in cur_transmission_object.requests]
        self.logger.add_log(log_time=self.current_time,
                            entity_name=transmission_node.entity_name,
                            activity="Finish sending.",
                            description=f"transmission type: {cur_transmission_object.transmission_type}, "
                                        f"request_uids: {_finished_request_uids}")

        # return transmission_object_handle, link_uid, finish_transmission_time
        return handle, link_uid, self.current_time + cur_link.latency

    def handle_finish_transmission(self, event: Event) -> None:
        """
        Handle event: finish transmission on a given link. event.args should have fileds:
            "transmission_object_handle": str
            "link_uid": int
            "finish_transmission_time": float

        :param event: the event to handle
        :return: None
        """
        # extract parameters
        handle: str = event.args["transmission_object_handle"]
        link_uid: int = event.args["link_uid"]
        end_time: float = event.args["finish_transmission_time"]
        assert end_time == event.event_time and end_time == self.current_time, "Time mismatch!"

        # get the link and finish transmission
        assert link_uid in self.links, "Unknown link!"
        cur_link: NetworkLink = self.links[link_uid]
        cur_transmission_object: TransmissionObject = cur_link.finish_transmission(transmission_object_handle=handle,
                                                                                   finish_time=self.current_time)

        # logging
        assert event.event_time == self.current_time, "Time discrepancy found!"
        _node_in_uid: int = cur_link.node_in.node_uid
        _node_out_uid: int = cur_link.node_out.node_uid
        _finished_request_uids: List[int] = [request.request_uid for request in cur_transmission_object.requests]
        self.logger.add_log(log_time=self.current_time,
                            entity_name=cur_link.entity_name,
                            activity="Finish transmission.",
                            description=f"From node {_node_in_uid} to node {_node_out_uid}, "
                                        f"type: {cur_transmission_object.transmission_type}, "
                                        f"request_uids: {_finished_request_uids}")

    def handle_gather_finished(self, event: Event) -> None:
        """
        Handle event: cluster coordinator gather finished requests. event.args should have fileds:
            /

        :param event: the event to handle
        :return: None
        """
        # move the finished requests into finished requests dict
        assert self.current_time == event.event_time, "Time mismatch!"
        gathered_request_uids: List[int] = []
        for request in self.sink_node.inbound_request_queue:
            # update cluster
            assert request.request_uid in self.requests_on_the_fly, "Unknown request found!"
            assert request.request_uid not in self.finished_requests, "Duplicate finished request!"
            del self.requests_on_the_fly[request.request_uid]
            self.finished_requests[request.request_uid] = (self.current_time, request)
            gathered_request_uids.append(request.request_uid)

            # update the query manager
            query_finished: bool = self.query_manager.collect_finished_request(current_time=self.current_time,
                                                                               request=request)

            # delete kv cache & activation backup for finished queries
            if query_finished:
                # delete kv cache on all compute node
                for layer_id in range(len(self.model)):
                    # get the locations of kv cache
                    locations: List[int] = request.kv_tracker_ref.get_kv_cache_locations(layer_id=layer_id)
                    assert len(locations) == 1, "Found KV cache on multiple nodes!"

                    # delete from these nodes
                    for node_uid in locations:
                        kv_cache_node: ComputeNode = self.compute_nodes[node_uid]
                        kv_cache_node.kv_cache.remove_query_kv_cache(layers=[layer_id],
                                                                     query_uid=request.base_query_uid)

                # delete activation backup on all compute nodes
                for layer_id in range(len(self.model)):
                    # get the locations of activation backup
                    # note that for some layers there are no activation backup (e.g. a layer in the middle of
                    # several layers on a node), and for some there might be multiple backups
                    locations: List[int] = request.kv_tracker_ref.get_activation_backup_locations(layer_id=layer_id)

                    # delete from these nodes
                    for node_uid in locations:
                        backup_node: ComputeNode = self.compute_nodes[node_uid]
                        backup_node.activation_backup_cache.remove_activation_backup(layer_id=layer_id,
                                                                                     query_uid=request.base_query_uid)

            # if we are in offline mode also need to check whether we need to launch new queries
            if self.use_offline_mode:
                assert not (request.phase == RequestPhase.Initialization and query_finished), \
                    "Found a request that finished after prompt phase!"
                if request.phase == RequestPhase.Initialization or query_finished:
                    self.offline_query_feeder.check_launch_new_query(finished_request_type=request.phase)

        # clear inbound request queue of coordinator (sink)
        self.sink_node.inbound_request_queue.clear()

        # logging
        assert event.event_time == self.current_time, "Time discrepancy found!"
        self.logger.add_log(log_time=self.current_time,
                            entity_name=self.sink_node.entity_name,
                            activity="Gather finished requests.",
                            description=f"Requests finished (t={self.current_time}): {gathered_request_uids}")

    def handle_start_execution(self, event: Event) -> Tuple[int, int, float]:
        """
        Handle event: start execution on a node. event.args should have fileds:
            "node": ComputeNode

        :param event: the event to handle
        :return: InferenceBatch handle, node uid, end time
        """
        # determine which node needs execution
        execution_node: ComputeNode = event.args["node"]
        assert execution_node.node_type == NodeType.Compute, "Only compute node can execute requests!"

        # check whether the node is busy, if it is, we will not execute anything
        if execution_node.is_node_busy():
            # logging
            assert event.event_time == self.current_time, "Time discrepancy found!"
            self.logger.add_log(log_time=self.current_time,
                                entity_name=execution_node.entity_name,
                                activity="Start execution.",
                                description=f"No request executed because node is busy.",
                                is_empty=True)

            # return
            return -1, -1, -1

        # get the list of executable requests
        # if executable requests are empty, we will march and see whether other layers needs inference
        # this is to prevent a deadlock situation where finish_transmission puts requests in a different
        # layer than the current one
        executable_requests: List[InferenceRequest] = execution_node.get_executable_requests()
        if len(executable_requests) == 0:
            execution_node.march_to_next_layer()
            executable_requests = execution_node.get_executable_requests()

        # if no request is executable, return directly
        if len(executable_requests) == 0:
            # logging
            assert event.event_time == self.current_time, "Time discrepancy found!"
            self.logger.add_log(log_time=self.current_time,
                                entity_name=execution_node.entity_name,
                                activity="Start execution.",
                                description=f"No request executed because no request is executable.",
                                is_empty=True)

            # return
            return -1, -1, -1

        # call scheduler to generate schedule
        schedule: ExecutionSchedule = self.scheduler.schedule_execution(node=execution_node,
                                                                        executable_requests=executable_requests)

        # logging
        assert event.event_time == self.current_time, "Time discrepancy found!"
        self.logger.add_log(log_time=self.current_time,
                            entity_name=execution_node.entity_name,
                            activity="Start execution.",
                            description=f"Layer id = {execution_node.get_current_inference_layer()}, "
                                        f"request uids = {schedule.get_description()}",
                            is_empty=len(schedule.requests) == 0)

        # execute the requests if schedule is not empty
        if not len(schedule.requests) == 0:
            # execute the requests in the schedule
            inference_batch: InferenceBatch = execution_node.start_execution(requests=schedule.requests)

            # return
            inference_batch_handle: int = inference_batch.get_handle()
            node_uid: int = execution_node.node_uid
            end_time: float = self.current_time + inference_batch.duration
            return inference_batch_handle, node_uid, end_time

        else:
            # return
            return -1, -1, -1

    def handle_finish_execution(self, event: Event) -> bool:
        """
        Handle event: finish execution on a node. event.args should have fileds:
            "inference_batch_handle": int,
            "node_uid": int,
            "end_time": float

        :param event: the event to handle
        :return: whether we need to trigger network send to send the finished requests to next node
        """
        # unpack parameters
        inference_batch_handle: int = event.args["inference_batch_handle"]
        node_uid: int = event.args["node_uid"]
        end_time: float = event.args["end_time"]

        # get execution node
        assert node_uid in self.compute_nodes, "Found unknown compute node!"
        execution_node: ComputeNode = self.compute_nodes[node_uid]
        finished_layer_id: int = execution_node.get_current_inference_layer()

        # finish execution of the batch of requests
        assert end_time == event.event_time and end_time == self.current_time, "Time mismatch!"
        current_inference_batch, trigger_network_send = execution_node.finish_execution(
            inference_batch_handle=inference_batch_handle
        )

        # logging
        assert event.event_time == self.current_time, "Time discrepancy found!"
        finished_request_uids: List[int] = [request.request_uid for request in current_inference_batch.requests]
        self.logger.add_log(log_time=self.current_time,
                            entity_name=execution_node.entity_name,
                            activity="Finish execution.",
                            description=f"Layer id = {finished_layer_id}, "
                                        f"request uids = {finished_request_uids}")
        return trigger_network_send

    def handle_start_loading_model(self, event: Event) -> float:
        """
        Handle event: a node starts to load model. event.args should have fileds:
            "node": ComputeNode

        :param event: the event to handle
        :return: how long it takes to finish model loading
        """
        # get the node that needs to load model
        compute_node: ComputeNode = event.args["node"]
        assert compute_node.node_type == NodeType.Compute, "Only compute node can load model!"

        # load model
        loading_time: float = compute_node.start_loading_model()

        # logging
        assert event.event_time == self.current_time, "Time discrepancy found!"
        new_model_layers: List[int] = list(compute_node.new_model_layers.keys())
        self.logger.add_log(log_time=self.current_time,
                            entity_name=compute_node.entity_name,
                            activity="Start loading model.",
                            description=f"New layers: {new_model_layers}, loading time: {loading_time}")

        # return
        return loading_time

    def handle_finish_loading_model(self, event: Event) -> None:
        """
        Handle event: a node finishes loading new model. event.args should have fileds:
            "node": ComputeNode

        :param event: the event to handle
        :return: None
        """
        # get the node that finishes loading
        compute_node: ComputeNode = event.args["node"]
        assert compute_node.node_type == NodeType.Compute, "Only compute node can finish loading model!"

        # finish loading model
        compute_node.finish_loading_model()

        # logging
        assert event.event_time == self.current_time, "Time discrepancy found!"
        new_model_layers: List[int] = list(compute_node.in_vram_model_layers.keys())
        self.logger.add_log(log_time=self.current_time,
                            entity_name=compute_node.entity_name,
                            activity="Finish loading model.",
                            description=f"New layers: {new_model_layers}.")

    def handle_event(self, event: Event) -> None:
        """
        Handle the given event. This is the core of the cluster simulator.

        :param event: event to handle
        :return: None
        """
        # first some sanity checks
        assert self.ready_to_simulate, "Cluster has not been marked as ready!"
        assert event.event_time >= self.current_time, f"Bad event time: {event.event_time:.3f}<{self.current_time:.3f}!"
        assert event.event_uid not in self.previous_events_dict, f"Duplicate event found!"

        # march timer forward
        self.current_time = event.event_time

        if event.event_handler == EventHandler.CommandNewRequest:
            # new request arrives at cluster coordinator
            succeeded = self.handle_command_new_request(event=event)

            # create a new event to start transmission of request to compute nodes
            # in offline mode, if the scheduling fails, then we don't need to start transmission
            if succeeded:
                new_event_args: Dict[str, Any] = {"node": self.source_node}
                new_event_description = EventDescription(who=self.source_node.entity_name,
                                                         at_when=self.current_time,
                                                         does_what="Start transmission")
                new_event = Event(event_uid=self.get_next_event_uid(),
                                  event_time=self.current_time,
                                  event_handler=EventHandler.StartTransmission,
                                  args=new_event_args,
                                  background=event.description,
                                  description=new_event_description)
                self.event_queue.put((new_event.event_time, new_event))

        elif event.event_handler == EventHandler.CommandLoadModel:
            # mark a compute node for loading model
            ready_to_load: bool = self.handle_command_load_model(event=event)

            # create a load model event if the model can be loaded right away
            if ready_to_load:
                node_to_load_model: ComputeNode = self.compute_nodes[event.args["node_uid"]]
                start_loading_event_args: Dict[str, Any] = {"node": node_to_load_model}
                start_loading_event_description = EventDescription(who=node_to_load_model.entity_name,
                                                                   at_when=self.current_time,
                                                                   does_what="Start loading model")
                start_loading_model_event = Event(event_uid=self.get_next_event_uid(),
                                                  event_time=self.current_time,
                                                  event_handler=EventHandler.StartLoadingModel,
                                                  args=start_loading_event_args,
                                                  background=event.description,
                                                  description=start_loading_event_description)
                self.event_queue.put((start_loading_model_event.event_time, start_loading_model_event))

        elif event.event_handler == EventHandler.StartTransmission:
            # start transmission for a node
            # transmission_object_end_time: transmission object handle -> (link_uid, finish_sending_time)
            transmission_object_end_time: Dict[int, Tuple[int, float]]
            retransmission_node_uids: List[int]
            transmission_object_end_time, retransmission_node_uids = self.handle_start_transmission(event=event)
            transmission_node: ComputeNode or SourceNode = event.args["node"]

            # create new events to handle finish_sending for each transmission
            for transmission_object_handle in transmission_object_end_time:
                link_uid, send_end_time = transmission_object_end_time[transmission_object_handle]
                new_event_args: Dict[str, Any] = {"transmission_node": transmission_node,
                                                  "transmission_object_handle": transmission_object_handle,
                                                  "link_uid": link_uid,
                                                  "send_end_time": send_end_time}
                new_event_description = EventDescription(who=transmission_node.entity_name,
                                                         at_when=send_end_time,
                                                         does_what="Finish sending")
                new_event = Event(event_uid=self.get_next_event_uid(),
                                  event_time=send_end_time,
                                  event_handler=EventHandler.FinishSending,
                                  args=new_event_args,
                                  background=event.description,
                                  description=new_event_description)
                self.event_queue.put((new_event.event_time, new_event))

            # create new events to call start_transmission again on specific nodes
            for retransmission_node_uid in retransmission_node_uids:
                # get the node
                if retransmission_node_uid == self.source_node.node_uid:
                    retransmission_node = self.source_node
                else:
                    retransmission_node = self.compute_nodes[retransmission_node_uid]
                retransmission_event_args: Dict[str, Any] = {"node": retransmission_node}
                retransmission_event_description = EventDescription(who=retransmission_node.entity_name,
                                                                    at_when=self.current_time,
                                                                    does_what="Start transmission (Re)")
                retransmission_event = Event(event_uid=self.get_next_event_uid(),
                                             event_time=self.current_time,
                                             event_handler=EventHandler.StartTransmission,
                                             args=retransmission_event_args,
                                             background=event.description,
                                             description=retransmission_event_description)
                self.event_queue.put((retransmission_event.event_time, retransmission_event))

        elif event.event_handler == EventHandler.FinishSending:
            # the node finishes sending, link resource can be deallocated
            handle, link_uid, finish_transmission_time = self.handle_finish_sending(event=event)
            transmission_node: ComputeNode or SourceNode = event.args["transmission_node"]
            cur_link: NetworkLink = transmission_node.outbound_links[link_uid]

            # create new events for start new transmission and finish current transmission
            # start new transmission
            transmission_event_args: Dict[str, Any] = {"node": transmission_node}
            transmission_event_description = EventDescription(who=transmission_node.entity_name,
                                                              at_when=self.current_time,
                                                              does_what="Start transmission")
            transmission_event = Event(event_uid=self.get_next_event_uid(),
                                       event_time=self.current_time,
                                       event_handler=EventHandler.StartTransmission,
                                       args=transmission_event_args,
                                       background=event.description,
                                       description=transmission_event_description)
            self.event_queue.put((transmission_event.event_time, transmission_event))
            # finish current transmission
            finish_event_args: Dict[str, Any] = {"transmission_object_handle": handle,
                                                 "link_uid": link_uid,
                                                 "finish_transmission_time": finish_transmission_time}
            finish_event_description = EventDescription(who=cur_link.entity_name,
                                                        at_when=finish_transmission_time,
                                                        does_what="Finish transmission")
            finish_event = Event(event_uid=self.get_next_event_uid(),
                                 event_time=finish_transmission_time,
                                 event_handler=EventHandler.FinishTransmission,
                                 args=finish_event_args,
                                 background=event.description,
                                 description=finish_event_description)
            self.event_queue.put((finish_event.event_time, finish_event))

        elif event.event_handler == EventHandler.FinishTransmission:
            # finish transmission over a link
            self.handle_finish_transmission(event=event)

            # create receiver events based on the receiver's type
            receiver: ComputeNode or SinkNode = self.links[event.args["link_uid"]].node_out
            if receiver.node_type == NodeType.Compute:
                # compute node will start execution when it receives a new request
                # coordinator node's receiver will gather the received requests
                receiver_event_args: Dict[str, Any] = {"node": receiver}
                receiver_event_description = EventDescription(who=receiver.entity_name,
                                                              at_when=self.current_time,
                                                              does_what="Start execution")
                receiver_event = Event(event_uid=self.get_next_event_uid(),
                                       event_time=self.current_time,
                                       event_handler=EventHandler.StartExecution,
                                       args=receiver_event_args,
                                       background=event.description,
                                       description=receiver_event_description)
                self.event_queue.put((receiver_event.event_time, receiver_event))
            elif receiver.node_type == NodeType.Sink:
                # coordinator node's receiver will gather the received requests
                receiver_event_args: Dict[str, Any] = {}
                receiver_event_description = EventDescription(who=self.sink_node.entity_name,
                                                              at_when=self.current_time,
                                                              does_what="Gather finished requests")
                receiver_event = Event(event_uid=self.get_next_event_uid(),
                                       event_time=self.current_time,
                                       event_handler=EventHandler.GatherFinished,
                                       args=receiver_event_args,
                                       background=event.description,
                                       description=receiver_event_description)
                self.event_queue.put((receiver_event.event_time, receiver_event))
            else:
                assert False, "Unknown node type!"

        elif event.event_handler == EventHandler.GatherFinished:
            # cluster coordinator gathers a finished request
            self.handle_gather_finished(event=event)

        elif event.event_handler == EventHandler.StartExecution:
            # start execution on a compute node
            inference_batch_handle, node_uid, end_time = self.handle_start_execution(event=event)

            # create a new event to handle the end of execution (if we actually executes something)
            if not inference_batch_handle == -1:
                new_event_args: Dict[str, Any] = {"inference_batch_handle": inference_batch_handle,
                                                  "node_uid": node_uid,
                                                  "end_time": end_time}
                new_event_description = EventDescription(who=self.compute_nodes[node_uid].entity_name,
                                                         at_when=end_time,
                                                         does_what="Finish execution")
                new_event = Event(event_uid=self.get_next_event_uid(),
                                  event_time=end_time,
                                  event_handler=EventHandler.FinishExecution,
                                  args=new_event_args,
                                  background=event.description,
                                  description=new_event_description)
                self.event_queue.put((new_event.event_time, new_event))

        elif event.event_handler == EventHandler.FinishExecution:
            # finish execution on a compute node
            trigger_network_send = self.handle_finish_execution(event=event)
            event_node: ComputeNode = self.compute_nodes[event.args["node_uid"]]

            # new event 1: start transmission (only when the finished layer is the last layer)
            if trigger_network_send:
                transmission_event_args: Dict[str, Any] = {"node": event_node}
                transmission_event_description = EventDescription(who=event_node.entity_name,
                                                                  at_when=self.current_time,
                                                                  does_what="Start transmission")
                transmission_event = Event(event_uid=self.get_next_event_uid(),
                                           event_time=self.current_time,
                                           event_handler=EventHandler.StartTransmission,
                                           args=transmission_event_args,
                                           background=event.description,
                                           description=transmission_event_description)
                self.event_queue.put((transmission_event.event_time, transmission_event))

            # new event 2: start execution again
            execution_event_args: Dict[str, Any] = {"node": event_node}
            execution_event_description = EventDescription(who=event_node.entity_name,
                                                           at_when=self.current_time,
                                                           does_what="Start execution")
            execution_event = Event(event_uid=self.get_next_event_uid(),
                                    event_time=self.current_time,
                                    event_handler=EventHandler.StartExecution,
                                    args=execution_event_args,
                                    background=event.description,
                                    description=execution_event_description)
            self.event_queue.put((execution_event.event_time, execution_event))

            # new event 3: start loading model if necessary
            if event_node.ready_to_load_model():
                # this means that the node is in flushing mode and dependencies are cleared
                load_model_event_args: Dict[str, Any] = {"node": event_node}
                load_model_event_description = EventDescription(who=event_node.entity_name,
                                                                at_when=self.current_time,
                                                                does_what="Start loading model")
                load_model_event = Event(event_uid=self.get_next_event_uid(),
                                         event_time=self.current_time,
                                         event_handler=EventHandler.StartLoadingModel,
                                         args=load_model_event_args,
                                         background=event.description,
                                         description=load_model_event_description)
                self.event_queue.put((load_model_event.event_time, load_model_event))

        elif event.event_handler == EventHandler.StartLoadingModel:
            # start loading model for a compute node
            loading_time: float = self.handle_start_loading_model(event=event)

            # create a new event for finish loading
            load_model_node: ComputeNode = event.args["node"]
            loading_end_time: float = self.current_time + loading_time
            finish_loading_event_args: Dict[str, Any] = {"node": load_model_node}
            finish_loading_event_description = EventDescription(who=load_model_node.entity_name,
                                                                at_when=loading_end_time,
                                                                does_what="Finish loading model")
            finish_loading_event = Event(event_uid=self.get_next_event_uid(),
                                         event_time=loading_end_time,
                                         event_handler=EventHandler.FinishLoadingModel,
                                         args=finish_loading_event_args,
                                         background=event.description,
                                         description=finish_loading_event_description)
            self.event_queue.put((finish_loading_event.event_time, finish_loading_event))

        elif event.event_handler == EventHandler.FinishLoadingModel:
            # finish loading model for a compute node
            self.handle_finish_loading_model(event=event)

            # create a new event to start execution after model loading is finished
            load_model_node: ComputeNode = event.args["node"]
            start_execution_event_args: Dict[str, Any] = {"node": load_model_node}
            start_execution_event_description = EventDescription(who=load_model_node.entity_name,
                                                                 at_when=self.current_time,
                                                                 does_what="Start execution")
            start_execution_event = Event(event_uid=self.get_next_event_uid(),
                                          event_time=self.current_time,
                                          event_handler=EventHandler.StartExecution,
                                          args=start_execution_event_args,
                                          background=event.description,
                                          description=start_execution_event_description)
            self.event_queue.put((start_execution_event.event_time, start_execution_event))

        elif event.event_handler == EventHandler.Unknown:
            # unknown event
            assert False, "Found an unknown event!"
        else:
            assert False, f"Found unknown event handler name: {event.event_handler}!"

    def simulate_next_event(self) -> Tuple[bool, float]:
        """
        Simulate next event in the event queue.

        :return: whether an event is simulated, time after simulation of next event
        """
        # return if event queue is empty
        if self.event_queue.empty():
            return False, self.current_time

        # pop an event from queue and execute
        event_time, cur_event = self.event_queue.get()  # type: float, Event
        assert event_time == cur_event.event_time, "Event time mismatch!"
        self.handle_event(event=cur_event)

        # logging
        assert cur_event.event_uid not in self.previous_events_dict, "Duplicate event found!"
        self.previous_events_list.append((event_time, cur_event))
        self.previous_events_dict[cur_event.event_uid] = cur_event

        # return
        assert event_time == self.current_time, "Time mismatch!"
        return True, event_time

    def watch(self, items: List[str]) -> None:
        """
        Watch some properties of the cluster.
        Watchable items: 1. "active_queries": number of queries on the fly
                         2. "kv-cache": kv cache status

        :param items: items to watch
        :return: None
        """
        if len(items) == 0:
            return
        if "all" in items:
            items = ["active_queries", "kv-cache"]

        print(f"# -------------- Watch -------------- #")
        print(f"Last event time = {self.current_time}")
        print(f"Next event time = {self.event_queue.queue[0][0]}")
        if "active_queries" in items:
            # since one active query has one request on the fly at any time
            num_active_queries = len(self.requests_on_the_fly)
            finished_queries = len(self.query_manager.finished_queries)
            print(f"[Item] active queries: {num_active_queries}, finished queries {finished_queries}.")
        if "kv-cache" in items:
            from simulator.scheduler.global_maxflow.global_maxflow_scheduler import GlobalFlowScheduler
            if isinstance(self.scheduler, GlobalFlowScheduler):
                print("[Item] KV-Cache:")
                print("Node Name: Real Used / Real Total | Expected Used / Expected Total | Expected > Real")
                for compute_node_uid, compute_node in self.compute_nodes.items():
                    node_name = compute_node.entity_name
                    node_kv_cache: KVCache = compute_node.kv_cache
                    kv_capacity = node_kv_cache.max_capacity
                    used_kv_capacity = kv_capacity - node_kv_cache.available_capacity
                    expected_used, expected_total = self.scheduler.core.kv_expectation.get_node_usage(
                        node_uid=compute_node_uid
                    )
                    print(f"\t{node_name}: {used_kv_capacity}/{kv_capacity} "
                          f"({round(used_kv_capacity / kv_capacity * 100)}%) | "
                          f"{expected_used} / {expected_total} ({round(expected_used / expected_total * 100)}%) "
                          f"| {expected_used >= used_kv_capacity}")
                print(f"Realtime bottleneck usage: {self.get_bottleneck_kv_cache_usage()}")
                print(f"Expected bottleneck usage: {self.scheduler.core.kv_expectation.bottleneck_usage()}")
            else:
                print("[Item] KV-Cache:")
                for _, compute_node in self.compute_nodes.items():
                    node_name = compute_node.entity_name
                    node_kv_cache: KVCache = compute_node.kv_cache
                    kv_capacity = node_kv_cache.max_capacity
                    used_kv_capacity = kv_capacity - node_kv_cache.available_capacity
                    print(f"\t{node_name}: {used_kv_capacity}/{kv_capacity} "
                          f"({round(used_kv_capacity / kv_capacity * 100)}%)")
                print(f"Realtime bottleneck usage: {self.get_bottleneck_kv_cache_usage()}")
        print(f"# ------------ End Watch ------------ #")

    def get_bottleneck_kv_cache_usage(self) -> float:
        """
        Get an estimation of bottleneck kv-cache usage.

        :return: bottleneck kv cache usage
        """
        free_kv_entries = [0 for _ in range(len(self.model))]
        total_kv_entries = [0 for _ in range(len(self.model))]
        for _, compute_node in self.compute_nodes.items():
            node_kv_cache: KVCache = compute_node.kv_cache
            free_entries = node_kv_cache.available_capacity / len(compute_node.in_vram_model_layers)
            total_entries = node_kv_cache.max_capacity / len(compute_node.in_vram_model_layers)
            for layer_id in compute_node.in_vram_model_layers:
                free_kv_entries[layer_id] += free_entries
                total_kv_entries[layer_id] += total_entries
        kv_entry_usage = [1 - free / total for free, total in zip(free_kv_entries, total_kv_entries)]
        return max(kv_entry_usage)

    # ************************************** APIs ************************************** #
    def simulate(self, until: Optional[float] = None, watch_items: Optional[List[str]] = None,
                 watch_interval: Optional[float] = None) -> None:
        """
        Simulate all events until the time specified. The timer will stop at 'until' after calling this.
        If 'until' is not specified, the simulator will simulate all events.

        :param until: (optional) simulate all events before this time
        :param watch_items: items to watch
        :param watch_interval: watch interval
        :return: None
        """
        # set simulation end time
        if until is not None:
            simulation_end_time: float = until
        else:
            simulation_end_time: float = math.inf

        while True:
            # if there are no events remaining, then march directly to end time
            if self.event_queue.empty():
                self.current_time = simulation_end_time
                return

            # if the next event happen at or after end time, march to end time and return
            next_event_time, _ = self.event_queue.queue[0]
            if next_event_time >= simulation_end_time:
                self.current_time = simulation_end_time
                return

            # print watch items
            if watch_interval is not None and watch_items is not None:
                if math.ceil(self.current_time / watch_interval) * watch_interval <= next_event_time:
                    if not self.last_watch_time == self.current_time:
                        self.watch(items=watch_items)
                        self.last_watch_time = self.current_time

            # otherwise, we can execute next event
            succeeded, event_time = self.simulate_next_event()  # type: bool, float
            assert succeeded and event_time < simulation_end_time, "Bad simulation result!"

    def issue_command_new_request(self, base_query_uid: int, arrive_time: float, phase: RequestPhase,
                                  token_seq_length: int, prev_num_tokens: int, token_size: float,
                                  activation_size: float, pipeline: List[PipelineStage] or None,
                                  kv_tracker_ref: KVTracker) -> int:
        """
        Issue command: a new request arrives at cluster at arrive_time.
        Notes: 1. if the scheduler is not Global MaxFlow Scheduler, pipeline can be set to None
               2. for Global MaxFlow Scheduler, if the request is in initialization phase, pipeline should be
                  set to None, otherwise it should be a valid pipeline (from initialization phase)

        :param base_query_uid: uid of the query this reqeust belongs to
        :param arrive_time: when the new request arrives
        :param phase: which phase is the current request in (initialization / increment)
        :param token_seq_length: how many tokens this request contains (increment must be 1)
        :param prev_num_tokens: number of previous tokens in the query before this iteration
        :param token_size: how much space token transmission needs
        :param activation_size: how much space activation transmission needs
        :param pipeline: the pipeline to use (see above)
        :param kv_tracker_ref: a reference to the kv_tracker in base query of the new request
        :return: uid of the request
        """
        # check arrive time
        assert arrive_time >= self.current_time, "Can not issue an request to a past time!"

        # build the request
        new_request_uid = self.get_next_request_uid()
        new_request = InferenceRequest(base_query_uid=base_query_uid,
                                       request_uid=new_request_uid,
                                       phase=phase,
                                       token_seq_length=token_seq_length,
                                       prev_num_tokens=prev_num_tokens,
                                       token_size=token_size,
                                       activation_size=activation_size,
                                       request_creation_time=arrive_time,
                                       kv_tracker_ref=kv_tracker_ref)

        # set pipeline for this request
        if pipeline is not None:
            new_request.set_pipeline(pipeline=pipeline)

        # creat the event at arrive time to represent the arrival of the request
        new_event_args: Dict[str, Any] = {"request": new_request}
        new_event_description = EventDescription(who=self.source_node.entity_name,
                                                 at_when=arrive_time,
                                                 does_what="New request arrives")
        new_event = Event(event_uid=self.get_next_event_uid(),
                          event_time=arrive_time,
                          event_handler=EventHandler.CommandNewRequest,
                          args=new_event_args,
                          background=new_event_description,
                          description=new_event_description)
        self.event_queue.put((new_event.event_time, new_event))
        return new_request_uid

    def issue_command_load_model(self, load_time: float, node_uid: int, new_layers: List[int],
                                 request_uids_to_wait: List[int]) -> None:
        """
        Load model on a given node.

        :param load_time: when shall the load command be sent to cluster
        :param node_uid: node uid
        :param new_layers: new layers to be loaded
        :param request_uids_to_wait: requests to finish before loading the model
        :return: None
        """
        # get machine type of this compute node
        assert load_time >= self.current_time, "Can not issue model loading to a past time!"
        assert node_uid in self.compute_nodes, "Unknown compute node!"
        target_node: ComputeNode = self.compute_nodes[node_uid]
        machine_type: str = target_node.machine_type

        # check whether the layers are continuous
        assert new_layers == list(range(min(new_layers), max(new_layers) + 1)), "Bad model!"
        assert machine_type in self.machine_types, "Unknown machine type!"

        # build new event
        new_event_args: Dict[str, Any] = {"node_uid": node_uid,
                                          "new_layer_ids": new_layers,
                                          "request_uids_to_wait": request_uids_to_wait}
        new_event_description = EventDescription(who=target_node.entity_name,
                                                 at_when=load_time,
                                                 does_what="Receive command load model")
        new_event = Event(event_uid=self.get_next_event_uid(),
                          event_time=load_time,
                          event_handler=EventHandler.CommandLoadModel,
                          args=new_event_args,
                          background=new_event_description,
                          description=new_event_description)
        self.event_queue.put((new_event.event_time, new_event))

    # ******************************** Profile and Plot ******************************** #
    def get_connection_info(self) -> Dict[str, int or float]:
        """
        Get connection information of the cluster.
        Info includes:
                  name                                  meaning
            avg_num_inbound            average number of inbound links per node
            max_num_inbound            maximum number of inbound links per node
            avg_num_outbound           average number of outbound links per node
            max_num_outbound           maximum number of outbound links per node
            avg_num_inbound_partial    average number of inbound links per node that uses partial inference
            max_num_inbound_partial    maximum number of inbound links per node that uses partial inference
            avg_num_outbound_partial   average number of outbound links per node that uses partial inference
            max_num_outbound_partial   maximum number of outbound links per node that uses partial inference

        :return: a dictionary of statistics
        """
        # gather statistics
        num_inbound: List[int] = []
        num_outbound: List[int] = []
        num_inbound_partial: List[int] = []
        num_outbound_partial: List[int] = []
        for _, compute_node in self.compute_nodes.items():
            # num inbound and outbound links
            num_inbound.append(len(compute_node.inbound_links))
            num_outbound.append(len(compute_node.outbound_links))

            # num inbound links that uses partial inference
            cur_num_partial_inbound = 0
            for _, network_link in compute_node.inbound_links.items():
                prev_node = network_link.node_in
                if isinstance(prev_node, ComputeNode):
                    prev_next_layer = max(prev_node.in_vram_model_layers.keys()) + 1
                    cur_first_layer = min(compute_node.in_vram_model_layers.keys())
                    cur_last_layer = max(compute_node.in_vram_model_layers.keys())
                    assert cur_first_layer <= prev_next_layer <= cur_last_layer, "Bad model layout!"
                    if cur_first_layer < prev_next_layer:
                        cur_num_partial_inbound += 1
            num_inbound_partial.append(cur_num_partial_inbound)

            # num outbound links that uses partial inference
            cur_num_partial_outbound = 0
            for _, network_link in compute_node.outbound_links.items():
                next_node = network_link.node_out
                if isinstance(next_node, ComputeNode):
                    cur_next_layer = max(compute_node.in_vram_model_layers.keys()) + 1
                    next_first_layer = min(next_node.in_vram_model_layers.keys())
                    next_last_layer = max(next_node.in_vram_model_layers.keys())
                    assert next_first_layer <= cur_next_layer <= next_last_layer, "Bad model layout!"
                    if next_first_layer < cur_next_layer:
                        cur_num_partial_outbound += 1
            num_outbound_partial.append(cur_num_partial_outbound)

        # return statistics
        return {"avg_num_inbound": round(sum(num_inbound) / len(num_inbound), 1),
                "max_num_inbound": max(num_inbound),
                "avg_num_outbound": round(sum(num_outbound) / len(num_outbound), 1),
                "max_num_outbound": max(num_outbound),
                "avg_num_inbound_partial": round(sum(num_inbound_partial) / len(num_inbound_partial), 1),
                "max_num_inbound_partial": max(num_inbound_partial),
                "avg_num_outbound_partial": round(sum(num_outbound_partial) / len(num_outbound_partial), 1),
                "max_num_outbound_partial": max(num_outbound_partial)}

    def plot_inference_speed(self, max_time: int or None = None, save_path: str or None = None) -> None:
        """
        Plot inference speed of the cluster. This function only uses finished requests to plot the figure.

        :param max_time: max_time
        :param save_path: save path of the figure
        :return: None
        """
        # gather data from finished requests
        time_bins: Dict[int, int] = {}
        last_finished_time: float = 0
        total_num_tokens: int = 0
        for request_uid, (finishing_time, finished_request) in self.finished_requests.items():
            last_finished_time = max(last_finished_time, finishing_time)
            total_num_tokens += finished_request.token_seq_length
            if int(finishing_time) in time_bins:
                time_bins[int(finishing_time)] += finished_request.token_seq_length
            else:
                time_bins[int(finishing_time)] = finished_request.token_seq_length

        # process the data for plotting
        if max_time is None:
            max_time = int(last_finished_time) + 10
        results: List[int] = []
        for i in range(max_time):
            if i in time_bins:
                results.append(time_bins[i])
            else:
                results.append(0)

        # plot the figure
        plt.plot(results)
        plt.xlabel("Time (s)")
        plt.ylabel("Inference Speed (#tokens/s)")
        plt.title("Inference Speed")

        # plot additional items based on scheduler type
        from simulator.scheduler.global_maxflow.global_maxflow_scheduler import GlobalFlowScheduler
        from simulator.scheduler.swarm.swarm_scheduler import SwarmScheduler
        if isinstance(self.scheduler, GlobalFlowScheduler):
            # print inference speed vs optimal inference speed
            max_flow: float = self.scheduler.flow_graph.flow_value
            print("# -------------------- MaxFlow Scheduler -------------------- #")
            print(f"Total time usage: {last_finished_time:.2f}s ({total_num_tokens / last_finished_time:.2f} tokens/s)")
            print(f"Theoretical optimal: {total_num_tokens / max_flow:.2f}s ({max_flow:.2f} tokens/s)")
            print("# ----------------------------------------------------------- #")

            # plot the line of optimal throughput
            plt.plot([max_flow for _ in range(max_time)])
            plt.legend(["MaxFlow Scheduler", "Optimal throughput"])

        elif isinstance(self.scheduler, SwarmScheduler):
            print("# --------------------- SWARM Scheduler --------------------- #")
            print(f"Total time usage: {last_finished_time:.2f}s ({total_num_tokens / last_finished_time:.2f} tokens/s)")
            print("# ----------------------------------------------------------- #")

            # plot the line of optimal throughput
            plt.legend(["SWARM Scheduler"])

        else:
            print("# -------------------- Unknown Scheduler -------------------- #")
            print(f"Total time usage: {last_finished_time:.2f}s ({total_num_tokens / last_finished_time:.2f} tokens/s)")
            print("# ----------------------------------------------------------- #")

        # show this figure and save
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def plot_request_latency(self, ignore_initialize: bool, save_path: str or None = None) -> None:
        """
        Plot request latency of the cluster. This function only uses finished requests to plot the figure.

        :param ignore_initialize: whether to ignore the initialization phase
        :param save_path: save path of the figure
        :return: None
        """
        self.query_manager: QueryManager
        self.query_manager.latency_analyzer.visualize_request_latency(ignore_initialize=ignore_initialize,
                                                                      save_file_path=save_path)

    def visualize_cluster(self, title: str, save_path: Optional[str] = None, show_fig: bool = True) -> None:
        """
        Visualize the cluster.

        :param title: title of the figure
        :param save_path: save path of the figure (a directory)
        :param show_fig: whether to show the figure
        :return: None
        """
        # check scheduler type and simulator condition
        from simulator.scheduler.global_maxflow.global_maxflow_scheduler import GlobalFlowScheduler
        from simulator.scheduler.global_maxflow.scheduler_core import SchedulerNode
        assert isinstance(self.scheduler, GlobalFlowScheduler), "Visualization needs GlobalFlowScheduler"
        assert self.ready_to_simulate, "Simulator must be ready to simulate"

        # construct graph topology
        graph = nx.DiGraph()
        graph.add_node(node_for_adding=self.source_node.entity_name, label="source", color=(0.0, 0.0, 0.0))
        graph.add_node(node_for_adding=self.sink_node.entity_name, label="sink", color=(0.0, 0.0, 0.0))
        for compute_node_uid, compute_node in self.compute_nodes.items():
            # node label
            start_layer_idx = min(compute_node.in_vram_model_layers)
            end_layer_idx = max(compute_node.in_vram_model_layers)
            if start_layer_idx == end_layer_idx:
                model_info = f"{start_layer_idx}"
            else:
                model_info = f"{start_layer_idx}-{end_layer_idx}"

            # node color
            scheduler_node: SchedulerNode = self.scheduler.core.scheduler_nodes[compute_node_uid]
            node_used_throughput = scheduler_node.inference_used_token_throughput
            node_total_throughput = scheduler_node.inference_token_throughput
            node_load_coefficient = node_used_throughput / node_total_throughput
            node_color = (round(node_load_coefficient, 3),
                          round(1 - node_load_coefficient, 3),
                          0)

            # add node
            graph.add_node(node_for_adding=compute_node.entity_name,
                           label=f"{compute_node.machine_type}\n{model_info}",
                           color=node_color)

            # inbound and outbound link statistics
            inbound_link_uids: List[int] = scheduler_node.inbound_link_uids
            inbound_link_used_throughput: List[int] = scheduler_node.inbound_links_used_token_throughput
            inbound_link_throughput: List[int] = scheduler_node.inbound_links_token_throughput
            assert len(inbound_link_uids) == len(inbound_link_throughput) == len(inbound_link_used_throughput), \
                "Bad link statistics, length mismatch!"
            inbound_link_loads: Dict[int, float] = {}
            for link_uid, used_tp, tp in zip(inbound_link_uids, inbound_link_used_throughput, inbound_link_throughput):
                inbound_link_loads[link_uid] = used_tp / tp
            outbound_link_uids: List[int] = scheduler_node.outbound_link_uids
            outbound_link_used_throughput: List[int] = scheduler_node.outbound_links_used_token_throughput
            outbound_link_throughput: List[int] = scheduler_node.outbound_links_token_throughput
            assert len(outbound_link_uids) == len(outbound_link_throughput) == len(outbound_link_used_throughput), \
                "Bad link statistics, length mismatch!"
            outbound_link_loads: Dict[int, float] = {}
            for link_uid, used_tp, tp in zip(outbound_link_uids, outbound_link_used_throughput,
                                             outbound_link_throughput):
                outbound_link_loads[link_uid] = used_tp / tp

            # add links
            for link_uid, link in compute_node.inbound_links.items():
                assert 0 <= inbound_link_loads[link_uid] <= 1, "Link load should be in [0, 1]!"
                link_color = (round(inbound_link_loads[link_uid], 3),
                              round(1 - inbound_link_loads[link_uid], 3),
                              0)
                graph.add_edge(u_of_edge=link.node_in.entity_name, v_of_edge=compute_node.entity_name,
                               color=link_color)
            for link_uid, link in compute_node.outbound_links.items():
                assert 0 <= outbound_link_loads[link_uid] <= 1, "Link load should be in [0, 1]!"
                link_color = (round(outbound_link_loads[link_uid], 3),
                              round(1 - outbound_link_loads[link_uid], 3),
                              0)
                graph.add_edge(u_of_edge=compute_node.entity_name, v_of_edge=link.node_out.entity_name,
                               color=link_color)

        # visualize cluster using topological ordering
        # set layer of each node and get position
        for layer, nodes in enumerate(nx.topological_generations(graph)):
            for node in nodes:
                graph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(graph, subset_key="layer")
        # get features
        custom_labels = {node: data['label'] for node, data in graph.nodes(data=True)}
        node_colors = [graph.nodes[n]['color'] for n in graph.nodes()]
        edge_colors = [graph[u][v]['color'] for u, v in graph.edges()]
        # draw the figure
        plt.figure(1, figsize=(40, 15))
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), edge_color=edge_colors, width=10)
        nx.draw_networkx_nodes(graph, pos, node_size=15000, node_color=node_colors, edgecolors='black',
                               linewidths=10)
        nx.draw_networkx_labels(graph, pos, labels=custom_labels, font_size=40, font_family='sans-serif',
                                font_color='white')
        plt.title(title, fontsize=60)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f"{title}.jpg"))
        if show_fig:
            plt.show()
