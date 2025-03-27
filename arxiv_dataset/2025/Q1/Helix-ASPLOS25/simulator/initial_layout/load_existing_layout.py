# 2023.03.16 Yixuan Mei

import math
import random
from configparser import ConfigParser
from typing import Dict, Tuple, List

from simulator.event_simulator.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec
from simulator.event_simulator.cluster_simulator import ClusterSimulator
from simulator.event_simulator.query_manager import QueryManagerParameters
from simulator.model_manager.model_manager import ModelManager
from simulator.initial_layout.ilp_layout.ilp_layout import MachineProfile, ModelCard, ILPNode, ILPLink
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import FlowParameters


class LoadExistingLayout:
    def __init__(self, model_manager: ModelManager) -> None:
        """
        Load an existing layout found by some layout method.

        :return: None
        """
        # loaded problem information
        self.machine_profiles: Dict[str, MachineProfile] = {}
        self.model_card: ModelCard or None = None
        self.model_manager: ModelManager = model_manager

        # topology information
        self.source: ILPNode or None = None
        self.sink: ILPNode or None = None
        self.nodes: Dict[int, ILPNode] = {}
        self.links: Dict[Tuple[int or str, int or str], ILPLink] = {}

        # states
        self.cluster_loaded: bool = False
        self.solution_loaded: bool = False
        self.node_idx_offset: int = 2

        # solution
        self.nodes_in_use: List[int] = []

    def from_ini(self, cluster_file_name: str, machine_profile_name: str) -> None:
        """
        Load cluster topology and machine profiles.

        :param cluster_file_name: name of the file that stores cluster topology
        :param machine_profile_name: name of the file that stores machine profiling results
        :return: None
        """
        # clear the dicts
        self.machine_profiles.clear()
        self.nodes.clear()
        self.links.clear()

        # load machine statistics
        machine_profile_parser = ConfigParser()
        machine_profile_parser.read(machine_profile_name)
        for machine_name in machine_profile_parser.sections():
            self.machine_profiles[machine_name] = MachineProfile(machine_name=machine_name,
                                                                 config=machine_profile_parser)

        # load cluster topology
        cluster_file_parser = ConfigParser()
        cluster_file_parser.read(cluster_file_name)

        # check that the topology is a complete graph
        total_num_compute_nodes = eval(cluster_file_parser["NodeNames"]["total_compute_nodes"])
        assert sorted(eval(cluster_file_parser["SourceNode"]["connected_nodes"])) == list(
            range(total_num_compute_nodes)), "Not a complete graph!"
        assert sorted(eval(cluster_file_parser["SinkNode"]["connected_nodes"])) == list(
            range(total_num_compute_nodes)), "Not a complete graph!"
        for i in range(total_num_compute_nodes):
            connected_nodes: List[int or str] = ["source"] + list(range(total_num_compute_nodes)) + ["sink"]
            connected_nodes.remove(i)
            assert eval(cluster_file_parser[f"ComputeNode-{i}"]["connected_nodes"]) == connected_nodes, \
                "Not a complete graph!"

        # model
        self.model_card = ModelCard(model_manager=self.model_manager)

        # source and sink
        self.source = ILPNode(node_index=-1, machine_type=self.machine_profiles["SourceNode"], max_num_layers=-1,
                              connected_node_indices=eval(cluster_file_parser["SourceNode"]["connected_nodes"]),
                              layer_count_2_throughput={})
        self.sink = ILPNode(node_index=-1, machine_type=self.machine_profiles["SinkNode"], max_num_layers=-1,
                            connected_node_indices=eval(cluster_file_parser["SinkNode"]["connected_nodes"]),
                            layer_count_2_throughput={})

        # compute nodes
        total_compute_nodes: int = eval(cluster_file_parser["NodeNames"]["total_compute_nodes"])
        for node_idx in range(total_compute_nodes):
            # extract machine name, type and connected nodes from file
            machine_name: str = f"ComputeNode-{node_idx}"
            machine_type: MachineProfile = self.machine_profiles[cluster_file_parser[machine_name]["type"]]
            connected_nodes: List[int] = eval(cluster_file_parser[machine_name]["connected_nodes"])

            # compute max number of layers that can be stored on this node
            # Note: max # layers = (VRAM size / 2) / layer size
            max_num_layers: int = self.model_manager.get_max_num_layers(machine_type=machine_type.type_name)
            assert max_num_layers * max(self.model_manager.get_model_params()) <= machine_type.vram_size + 1, \
                "Trying to use more than the vram to load model parameters!"

            # compute layer count to throughput
            # Note: 1. inference throughput is computed under typical batch size
            #       2. total throughput is the min of inference throughput and nic throughput
            bottleneck_nic_speed: float = min(machine_type.inbound_nic_speed, machine_type.outbound_nic_speed)
            bottleneck_nic_throughput: float = bottleneck_nic_speed / self.model_card.activation_size
            layer_count_2_throughput: Dict[int, float] = {}
            for layer_count in range(1, max_num_layers + 1):
                inference_throughput: float = self.model_manager.get_typical_token_throughput(
                    machine_type=machine_type.type_name, num_on_node_layers=layer_count
                )
                layer_count_2_throughput[layer_count] = min(inference_throughput, bottleneck_nic_throughput)

            # add node
            self.nodes[node_idx] = ILPNode(node_index=node_idx,
                                           machine_type=machine_type,
                                           max_num_layers=max_num_layers,
                                           connected_node_indices=connected_nodes,
                                           layer_count_2_throughput=layer_count_2_throughput)

        # links
        # Note: links here are bidirectional
        for entity_name in cluster_file_parser.sections():
            if "Link-" in entity_name:
                # end points
                from_idx: int or str = entity_name.split("-")[1]
                if not from_idx == "source":
                    from_idx = int(from_idx)
                to_idx: int or str = entity_name.split("-")[2]
                if not to_idx == "sink":
                    to_idx = int(to_idx)

                # bandwidth and latency
                bandwidth: float = eval(cluster_file_parser[entity_name]["bandwidth"])
                latency: float = eval(cluster_file_parser[entity_name]["latency"])
                if from_idx == "source" or to_idx == "sink":
                    throughput: float = bandwidth / self.model_card.token_size
                else:
                    assert isinstance(from_idx, int) and isinstance(to_idx, int), "Bad index!"
                    throughput: float = bandwidth / self.model_card.activation_size
                self.links[(from_idx, to_idx)] = ILPLink(from_index=from_idx,
                                                         to_index=to_idx,
                                                         throughput=throughput,
                                                         bandwidth=bandwidth,
                                                         latency=latency)

        # mark cluster as loaded
        self.cluster_loaded = True

    def load_solution(self, solution_file_name: str) -> None:
        """
        Load a solution from existing solution file.

        :param solution_file_name: name of the solution file
        :return: None
        """
        # initialize
        assert self.cluster_loaded, "Cluster must be loaded before loading solutions!"
        self.nodes_in_use.clear()

        # load solution
        solution_parser = ConfigParser()
        solution_parser.read(solution_file_name)

        # get base offset
        base_idx_offset: int = eval(solution_parser["Settings"]["offset"])
        assert base_idx_offset == self.node_idx_offset, "Base index offset mismatch!"

        # for each compute node, load data
        for compute_node_name in solution_parser["Solution"]:
            # get node idx
            idx_before_offset = int(compute_node_name.split("_")[2]) - self.node_idx_offset
            assert idx_before_offset >= 0, "Found node index smaller than 0!"

            # get list of layer held by this node
            held_layers = eval(solution_parser["Solution"][compute_node_name])
            if not len(held_layers) == 0:
                self.nodes_in_use.append(idx_before_offset)
                start_layer_idx = min(held_layers)
                end_layer_idx = start_layer_idx + len(held_layers)
                assert sorted(held_layers) == list(range(start_layer_idx, end_layer_idx)), "Bad held layers!"
                self.nodes[idx_before_offset].start_layer_idx = start_layer_idx
                self.nodes[idx_before_offset].end_layer_idx = end_layer_idx

        self.solution_loaded = True

    def set_initial_layout(self, simulator: ClusterSimulator) -> float:
        """
        Load the initial model layout into the simulator.

        :param simulator: the cluster simulator to load model into
        :return: expected loading time in simulation
        """
        assert self.solution_loaded, "Must synthesize a solution before setting initial layout for simulator!"
        assert simulator.current_time == 0, "Initial layout can only be set at the beginning!"

        max_load_time: float = 0
        for ilp_node_idx, ilp_node in self.nodes.items():
            # if the node is not in the loaded solution, continue
            if ilp_node_idx not in self.nodes_in_use:
                continue

            # get the corresponding compute node in the simulator
            compute_node_name = f"compute_node_{self.node_idx_offset + ilp_node_idx}"
            compute_node = simulator.name_2_compute_node[compute_node_name]

            # get the model layers to load and corresponding loading time
            new_layers = list(range(ilp_node.start_layer_idx, ilp_node.end_layer_idx))
            new_layers_size = sum(self.model_manager.get_model_params()[ilp_node.start_layer_idx:
                                                                        ilp_node.end_layer_idx])
            loading_time = new_layers_size / compute_node.disk_speed
            max_load_time = max(max_load_time, loading_time)

            # issue load command
            simulator.issue_command_load_model(load_time=simulator.current_time,
                                               node_uid=compute_node.node_uid,
                                               new_layers=new_layers,
                                               request_uids_to_wait=[])

        # advance simulator
        max_load_time = math.ceil(max_load_time) + 1
        simulator.simulate(until=max_load_time)
        return max_load_time

    def get_flow_upper_bound(self) -> float:
        """
        Get the upper bound of max flow over this cluster, which is defined as the max flow when all network
        transmissions are instant.

        :return: flow upper bound
        """
        assert self.cluster_loaded, "Cluster must be loaded before we can compute flow upper bound!"
        total_compute_throughput: float = 0
        for node_idx, compute_node in self.nodes.items():
            cur_node_max = -1
            for i in range(1, compute_node.max_num_layers + 1):
                cur_node_max = max(cur_node_max, compute_node.layer_count_2_throughput[i] * i)
            assert not cur_node_max == -1, "Bad max throughput!"
            total_compute_throughput += cur_node_max
        return total_compute_throughput / self.model_card.num_layers

    def get_flow_parameters(self) -> FlowParameters:
        """
        Get flow parameters based on the loaded cluster file.

        :return: FlowParameters
        """
        assert self.cluster_loaded, "Cluster must be loaded before FlowParameters can be returned!"
        return FlowParameters(token_size=self.model_card.token_size,
                              token_activation_size=self.model_card.activation_size)

    def get_query_manager_parameters(self) -> QueryManagerParameters:
        """
        Get query manager parameters based on the loaded cluster file.

        :return: QueryManagerParameters
        """
        assert self.cluster_loaded, "Cluster must be loaded before QueryManagerParameters can be returned!"
        return QueryManagerParameters(token_size=self.model_card.token_size,
                                      token_activation_size=self.model_card.activation_size,
                                      total_num_layers=self.model_card.num_layers)
