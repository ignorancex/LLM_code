# 2023.01.26 Yixuan Mei

import networkx as nx
from typing import Dict, List, Tuple

from simulator.event_simulator.base_node import NodeType
from simulator.event_simulator.cluster_simulator import ClusterSimulator
from simulator.event_simulator.utils import is_close


class FlowParameters:
    def __init__(self, token_size: float, token_activation_size: float) -> None:
        """
        Parameters used for computing max flow.

        :param token_size: size to store a token
        :param token_activation_size: size to store activation for a token
        """
        self.token_size: float = token_size
        self.token_activation_size: float = token_activation_size


class FlowGraph:
    def __init__(self, cluster_simulator: ClusterSimulator, parameters: FlowParameters) -> None:
        """
        Flow graph of the cluster.

        :param cluster_simulator: the cluster to build flow graph on
        :return: None
        """
        # the cluster to build flow graph on
        self.cluster_simulator: ClusterSimulator = cluster_simulator
        self.parameters: FlowParameters = parameters

        # the flow graph
        self.flow_graph_timestamp: float or None = None
        self.flow_graph: nx.DiGraph or None = None
        self.flow_value: float or None = None
        self.flow_dict: Dict[str, Dict[str, float]] or None = None

        # past flow graphs
        # timestamp -> (flow_graph, flow_value, flow_dict)
        self.flow_graph_history: Dict[float, Tuple[nx.DiGraph, float, Dict[str, Dict[str, float]]]] = {}

    def add_source(self, source_node_uid: int, source_outbound_nic_token_throughput: float) -> None:
        """
        Add source node into the flow graph.

        :param source_node_uid: uid of source node
        :param source_outbound_nic_token_throughput: outbound nic throughput in #tokens/s
        :return: None
        """
        #          outbound TP
        # source --------------> source_out
        assert isinstance(self.flow_graph, nx.DiGraph), "Graph type not supported!"
        self.flow_graph.add_node(node_for_adding=f"source")
        self.flow_graph.add_node(node_for_adding=f"{source_node_uid}_out")
        self.flow_graph.add_edge(u_of_edge=f"source", v_of_edge=f"{source_node_uid}_out",
                                 capacity=source_outbound_nic_token_throughput)

    def add_sink(self, sink_node_uid: int, sink_inbound_nic_token_throughput: float) -> None:
        """
        Add sink node into the flow graph.

        :param sink_node_uid: uid of the sink node
        :param sink_inbound_nic_token_throughput: inbound nic throughput in #tokens/s
        :return: None
        """
        #          inbound TP
        # sink_in -----------> sink
        assert isinstance(self.flow_graph, nx.DiGraph), "Graph type not supported!"
        self.flow_graph.add_node(node_for_adding=f"{sink_node_uid}_in")
        self.flow_graph.add_node(node_for_adding=f"sink")
        self.flow_graph.add_edge(u_of_edge=f"{sink_node_uid}_in", v_of_edge=f"sink",
                                 capacity=sink_inbound_nic_token_throughput)

    def add_compute_node(self, node_uid: int, inference_throughput: float, inbound_nic_token_throughput: float,
                         outbound_nic_token_throughput: float) -> None:
        """
        Add compute node into the flow graph.

        :param node_uid: uid of the node
        :param inference_throughput: inference throughput in #tokens/s
        :param inbound_nic_token_throughput: inbound nic throughput in #tokens/s
        :param outbound_nic_token_throughput: outbound nic throughput in #tokens/s
        :return: None
        """
        #      inbound TP          inference TP         outbound TP
        # _in -----------> _start --------------> _end -------------> _out
        assert isinstance(self.flow_graph, nx.DiGraph), "Graph type not supported!"
        self.flow_graph.add_node(node_for_adding=f"{node_uid}_in")
        self.flow_graph.add_node(node_for_adding=f"{node_uid}_start")
        self.flow_graph.add_node(node_for_adding=f"{node_uid}_end")
        self.flow_graph.add_node(node_for_adding=f"{node_uid}_out")
        self.flow_graph.add_edge(u_of_edge=f"{node_uid}_in", v_of_edge=f"{node_uid}_start",
                                 capacity=inbound_nic_token_throughput)
        self.flow_graph.add_edge(u_of_edge=f"{node_uid}_start", v_of_edge=f"{node_uid}_end",
                                 capacity=inference_throughput)
        self.flow_graph.add_edge(u_of_edge=f"{node_uid}_end", v_of_edge=f"{node_uid}_out",
                                 capacity=outbound_nic_token_throughput)

    def add_link(self, prev_node_uid: int, next_node_uid: int, throughput: float) -> None:
        """
        Add a link between two nodes in the flow graph.

        :param prev_node_uid: uid of previous node (input to this link)
        :param next_node_uid: uid of next node (output of this link)
        :param throughput: throughput of this link in #tokens/s
        :return: None
        """
        #               TP
        # prev_out -----------> next_in
        assert isinstance(self.flow_graph, nx.DiGraph), "Graph type not supported!"
        self.flow_graph.add_edge(u_of_edge=f"{prev_node_uid}_out", v_of_edge=f"{next_node_uid}_in", capacity=throughput)

    def update_flow(self, time_stamp: float) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """
        Create flow graph at current timestamp.

        :param time_stamp: cluster simulator timestamp when flow is computed
        :return: None
        """
        # backup previous network flow results
        if self.flow_graph_timestamp:
            assert time_stamp > self.flow_graph_timestamp, "Time inconsistency found!"
            self.flow_graph_history[self.flow_graph_timestamp] = (self.flow_graph, self.flow_value, self.flow_dict)

        # initialize current network flow graph
        self.flow_graph_timestamp = time_stamp
        self.flow_graph = nx.DiGraph()

        # iterate through the cluster simulator to construct flow graph
        # add source node
        source_nic_throughput = self.cluster_simulator.source_node.outbound_nic_speed / self.parameters.token_size
        self.add_source(source_node_uid=self.cluster_simulator.source_node.node_uid,
                        source_outbound_nic_token_throughput=source_nic_throughput)
        # add sink node
        sink_nic_throughput = self.cluster_simulator.sink_node.inbound_nic_speed / self.parameters.token_size
        self.add_sink(sink_node_uid=self.cluster_simulator.sink_node.node_uid,
                      sink_inbound_nic_token_throughput=sink_nic_throughput)
        # add compute nodes
        for compute_node_uid, compute_node in self.cluster_simulator.compute_nodes.items():
            # calculate inbound nic throughput
            # if this node may take input from some other compute nodes, we will consider
            # token activation size in transmission.
            input_node_uids: List[int] = []
            for link_uid, link in compute_node.inbound_links.items():
                input_node_uids.append(link.node_in.node_uid)
            if input_node_uids == [self.cluster_simulator.source_node.node_uid]:
                inbound_transmission_size: float = self.parameters.token_size
            else:
                inbound_transmission_size: float = self.parameters.token_size + self.parameters.token_activation_size
            inbound_nic_throughput: float = compute_node.inbound_nic_speed / inbound_transmission_size

            # calculate outbound nic throughput
            # if this node may send output to some other compute nodes, we will consider
            # token activation size in transmission.
            output_node_uids: List[int] = []
            for link_uid, link in compute_node.outbound_links.items():
                output_node_uids.append(link.node_out.node_uid)
            if output_node_uids == [self.cluster_simulator.sink_node.node_uid]:
                outbound_transmission_size: float = self.parameters.token_size
            else:
                outbound_transmission_size: float = self.parameters.token_size + self.parameters.token_activation_size
            outbound_nic_throughput: float = compute_node.outbound_nic_speed / outbound_transmission_size

            # calculate inference throughput
            num_layers_on_node = len(compute_node.in_vram_model_layers)
            inference_throughput: float = self.cluster_simulator.model_manager.get_typical_token_throughput(
                machine_type=compute_node.machine_type, num_on_node_layers=num_layers_on_node
            )
            assert is_close(inference_throughput, compute_node.get_typical_token_throughput()), \
                "Typical inference throughput mismatch!"

            # add node
            self.add_compute_node(node_uid=compute_node_uid, inference_throughput=inference_throughput,
                                  inbound_nic_token_throughput=inbound_nic_throughput,
                                  outbound_nic_token_throughput=outbound_nic_throughput)
        # add links
        for link_uid, link in self.cluster_simulator.links.items():
            # calculate link throughput
            if link.node_in_type == NodeType.Source or link.node_out_type == NodeType.Sink:
                transmission_size: float = self.parameters.token_size
            else:
                transmission_size: float = self.parameters.token_size + self.parameters.token_activation_size
            link_throughput: float = link.bandwidth / transmission_size

            # add link
            self.add_link(prev_node_uid=link.node_in.node_uid, next_node_uid=link.node_out.node_uid,
                          throughput=link_throughput)

        # compute network flow and return
        self.flow_value, self.flow_dict = nx.maximum_flow(flowG=self.flow_graph, _s="source", _t="sink")
        return self.flow_value, self.flow_dict

    def get_node_capacity(self, node_uid: int) -> Dict[str, float or None]:
        """
        Get capacity of a given node.
                                        source          compute         sink
        Return dict: 1. "inbound"       None            #tokens/s       #tokens/s
                     2. "inference"     None            #tokens/s       None
                     3. "outbound"      #tokens/s       #tokens/s       None

        :param node_uid: uid of the node
        :return: a dictionary, see above
        """
        assert self.flow_graph_timestamp is not None, "No valid graph found!"
        capacity_dict: Dict[str, float or None] = {}

        if node_uid == self.cluster_simulator.source_node.node_uid:
            capacity_dict["inbound"] = None
            capacity_dict["inference"] = None
            capacity_dict["outbound"] = self.flow_graph["source"][f"{node_uid}_out"]["capacity"]
        elif node_uid == self.cluster_simulator.sink_node.node_uid:
            capacity_dict["inbound"] = self.flow_graph[f"{node_uid}_in"]["sink"]["capacity"]
            capacity_dict["inference"] = None
            capacity_dict["outbound"] = None
        else:
            capacity_dict["inbound"] = self.flow_graph[f"{node_uid}_in"][f"{node_uid}_start"]["capacity"]
            capacity_dict["inference"] = self.flow_graph[f"{node_uid}_start"][f"{node_uid}_end"]["capacity"]
            capacity_dict["outbound"] = self.flow_graph[f"{node_uid}_end"][f"{node_uid}_out"]["capacity"]

        return capacity_dict

    def get_link_capacity(self, prev_node_uid: int, next_node_uid: int) -> Dict[str, float]:
        """
        Get capacity of a given link.
                                            link
        Return dict: 1. "transmission"   #tokens/s

        :param prev_node_uid: uid of previous node
        :param next_node_uid: uid of next node
        :return: a dictionary, see above
        """
        assert self.flow_graph_timestamp is not None, "No valid graph found!"
        capacity_dict: Dict[str, float] = {
            "transmission": self.flow_graph[f"{prev_node_uid}_out"][f"{next_node_uid}_in"]["capacity"]}
        return capacity_dict

    def get_node_flow(self, node_uid: int) -> Dict[str, float or None]:
        """
        Get flow on a given node.
                                        source          compute         sink
        Return dict: 1. "inbound"       None            #tokens/s       #tokens/s
                     2. "inference"     None            #tokens/s       None
                     3. "outbound"      #tokens/s       #tokens/s       None

        :param node_uid: uid of the node
        :return: a dictionary, see above
        """
        assert self.flow_graph_timestamp is not None, "No valid graph found!"
        flow_dict: Dict[str, float or None] = {}

        if node_uid == self.cluster_simulator.source_node.node_uid:
            flow_dict["inbound"] = None
            flow_dict["inference"] = None
            flow_dict["outbound"] = self.flow_dict["source"][f"{node_uid}_out"]
        elif node_uid == self.cluster_simulator.sink_node.node_uid:
            flow_dict["inbound"] = self.flow_dict[f"{node_uid}_in"]["sink"]
            flow_dict["inference"] = None
            flow_dict["outbound"] = None
        else:
            flow_dict["inbound"] = self.flow_dict[f"{node_uid}_in"][f"{node_uid}_start"]
            flow_dict["inference"] = self.flow_dict[f"{node_uid}_start"][f"{node_uid}_end"]
            flow_dict["outbound"] = self.flow_dict[f"{node_uid}_end"][f"{node_uid}_out"]

        return flow_dict

    def get_link_flow(self, prev_node_uid: int, next_node_uid: int) -> Dict[str, float]:
        """
        Get flow on a given link.
                                            link
        Return dict: 1. "transmission"   #tokens/s

        :param prev_node_uid: uid of previous node
        :param next_node_uid: uid of next node
        :return: a dictionary, see above
        """
        assert self.flow_graph_timestamp is not None, "No valid graph found!"
        flow_dict: Dict[str, float] = {
            "transmission": self.flow_dict[f"{prev_node_uid}_out"][f"{next_node_uid}_in"]}
        return flow_dict
