# 2023.01.06 Yixuan Mei
import copy
import networkx as nx

from typing import List, Dict, Tuple

from simulator.event_simulator.cluster_simulator import ClusterSimulator
from simulator.event_simulator.network_link import NetworkLink
from simulator.event_simulator.base_node import NodeType
from simulator.event_simulator.compute_node import ComputeNode
from simulator.event_simulator.coordinator_node import SourceNode, SinkNode


class MaxFlowParameters:
    def __init__(self, token_size: float, token_activation_size: float) -> None:
        """
        Parameters used for computing max flow.

        :param token_size: size to store a token
        :param token_activation_size: size to store activation for a token
        """
        self.token_size: float = token_size
        self.token_activation_size: float = token_activation_size


class RequestDestinationCache:
    def __init__(self, all_link_uids: List[int]) -> None:
        """
        A cache for request destination.

        :param all_link_uids: uids of all outbound links from this node
        :return: None
        """
        # all link uids
        self.all_link_uids = all_link_uids

        # cache: request_uid -> link_uid
        self.cache: Dict[int, int] = {}
        # inverse_cache: link_uid -> the list of request_uids waiting on this link
        self.inverse_cache: Dict[int, List[int]] = {link_uid: [] for link_uid in all_link_uids}

    def contains(self, request_uid: int) -> bool:
        """
        Check whether the request_uid has been cached.

        :param request_uid: request_uid to check
        :return: whether the request_uid has been cached
        """
        return request_uid in self.cache

    def add_request(self, request_uid: int, link_uid: int) -> None:
        """
        Add a request into cache

        :param request_uid: uid of the request
        :param link_uid: uid of the link
        :return: None
        """
        # check
        assert link_uid in self.all_link_uids, "Unknown link uid!"
        assert request_uid not in self.cache, "Request already cached!"

        # save to cache
        self.cache[request_uid] = link_uid
        self.inverse_cache[link_uid].append(request_uid)

    def get_next_request(self, link_uid: int) -> int or None:
        """
        Get a request_uid for given link, return None if cache for that link is empty.

        :param link_uid: uid of the link
        :return: request_uid or None
        """
        assert link_uid in self.all_link_uids, "Unknown link uid!"

        if not len(self.inverse_cache[link_uid]) == 0:
            # pop a request and return
            request_uid: int = self.inverse_cache[link_uid].pop(0)
            assert request_uid in self.cache, "Cache mismatch!"
            del self.cache[request_uid]
            return request_uid
        else:
            # return None
            return None


class TopologyNode:
    def __init__(self, node_uid: int, inbound_network_throughput: float, outbound_network_throughput: float,
                 inference_throughput: float, outbound_link_uids: List[int]) -> None:
        """
        TopologyNode corresponds to a compute node in the cluster, used for computing max flow.

        :param node_uid: compute node's node_uid
        :param inbound_network_throughput: throughput of the node for receiving requests (in #tokens/s)
        :param outbound_network_throughput: throughput of the node for sending requests (in #tokens/s)
        :param inference_throughput: throughput of the node for inference (in #tokens/s)
        :param outbound_link_uids: uids of all outbound links
        :return: None
        """
        # basic information
        self.node_uid: int = node_uid
        self.inbound_network_throughput: float = inbound_network_throughput
        self.outbound_network_throughput: float = outbound_network_throughput
        self.inference_throughput: float = inference_throughput
        self.outbound_link_uids: List[int] = outbound_link_uids

        # transmission scheduling: interleaved weighted round-robin
        self.link_weights: Dict[int, float] = {}
        self.link_effective_weights: Dict[int, float] = {}

        # transmission scheduling cache
        self.transmission_scheduling_cache = RequestDestinationCache(all_link_uids=outbound_link_uids)

    def add_to_graph(self, graph: nx.DiGraph) -> None:
        """
        Add this node to a NetworkX directed graph.

        :param graph: the graph to add this node to
        :return: None
        """
        # add this node to graph
        #      inbound TP          inference TP         outbound TP
        # _in -----------> _start --------------> _end -------------> _out
        graph.add_node(node_for_adding=f"{self.node_uid}_in")
        graph.add_edge(u_of_edge=f"{self.node_uid}_in", v_of_edge=f"{self.node_uid}_start",
                       capacity=self.inbound_network_throughput)
        graph.add_node(node_for_adding=f"{self.node_uid}_start")
        graph.add_edge(u_of_edge=f"{self.node_uid}_start", v_of_edge=f"{self.node_uid}_end",
                       capacity=self.inference_throughput)
        graph.add_node(node_for_adding=f"{self.node_uid}_end")
        graph.add_edge(u_of_edge=f"{self.node_uid}_end", v_of_edge=f"{self.node_uid}_out",
                       capacity=self.outbound_network_throughput)
        graph.add_node(node_for_adding=f"{self.node_uid}_out")

    def set_weights(self, weights: Dict[int, float]) -> None:
        """
        Set weights for this node.

        :param weights: weights dict
        :return: None
        """
        # check that weights are set for each link
        assert sorted(self.outbound_link_uids) == sorted(list(weights.keys())), "Weights mismatch!"

        # set weights
        self.link_weights = copy.deepcopy(weights)
        self.link_effective_weights = copy.deepcopy(weights)

    def interleaved_weighted_round_robin(self) -> int:
        """
        Perform interleaved weighted round-robin for transmission.

        :return: link_uid to use
        """
        # check weights are not empty
        assert not len(self.link_weights) == 0, "Must set weights before performing round-robin!"

        # select the link
        selected_link_uid = max(self.link_effective_weights, key=self.link_effective_weights.get)

        # update effective weights
        total_weights: float = sum(self.link_weights.values())
        for link_uid in self.link_effective_weights:
            self.link_effective_weights[link_uid] += self.link_weights[link_uid]
            if link_uid == selected_link_uid:
                self.link_effective_weights[link_uid] -= total_weights

        # return
        return selected_link_uid

    def get_normalized_flow(self) -> Dict[int, float]:
        """
        Get normalized flow from this node to each outbound link.

        :return: normalized flow
        """
        # check weights are not empty
        assert not len(self.link_weights) == 0, "Must set weights before computing normalized flow!"

        # get normalized flow
        sum_weights: float = sum(self.link_weights.values())
        normalized_flow_dict: Dict[int, float] = {link_uid: self.link_weights[link_uid] / sum_weights for link_uid in
                                                  self.link_weights}
        return normalized_flow_dict


class TopologySource:
    def __init__(self, node_uid: int, outbound_network_throughput: float, outbound_link_uids: List[int]) -> None:
        """
        TopologySource corresponds to source node (i.e. coordinator) in the cluster, used for computing max flow.

        :param node_uid: source node_uid
        :param outbound_network_throughput: throughput of the node for sending requests (in #tokens/s)
        :param outbound_link_uids: uids of all outbound links
        :return: None
        """
        # basic information
        self.node_uid: int = node_uid
        self.outbound_network_throughput: float = outbound_network_throughput
        self.outbound_link_uids: List[int] = outbound_link_uids

        # transmission scheduling: interleaved weighted round-robin
        self.link_weights: Dict[int, float] = {}
        self.link_effective_weights: Dict[int, float] = {}

        # transmission scheduling cache
        self.transmission_scheduling_cache = RequestDestinationCache(all_link_uids=outbound_link_uids)

    def add_to_graph(self, graph: nx.DiGraph) -> None:
        """
        Add this node to a NetworkX directed graph.

        :param graph: the graph to add this node to
        :return: None
        """
        # add this node to graph
        #          outbound TP
        # source --------------> source_out
        graph.add_node(node_for_adding=f"source")
        graph.add_edge(u_of_edge=f"source", v_of_edge=f"{self.node_uid}_out",
                       capacity=self.outbound_network_throughput)
        graph.add_node(node_for_adding=f"{self.node_uid}_out")

    def set_weights(self, weights: Dict[int, float]) -> None:
        """
        Set weights for this node.

        :param weights: weights dict
        :return: None
        """
        # check that weights are set for each link
        assert sorted(self.outbound_link_uids) == sorted(list(weights.keys())), "Weights mismatch!"

        # set weights
        self.link_weights = copy.deepcopy(weights)
        self.link_effective_weights = copy.deepcopy(weights)

    def interleaved_weighted_round_robin(self) -> int:
        """
        Perform interleaved weighted round-robin for transmission.

        :return: link_uid to use
        """
        # check weights are not empty
        assert not len(self.link_weights) == 0, "Must set weights before performing round-robin!"

        # select the link
        selected_link_uid = max(self.link_effective_weights, key=self.link_effective_weights.get)

        # update effective weights
        total_weights: float = sum(self.link_weights.values())
        for link_uid in self.link_effective_weights:
            self.link_effective_weights[link_uid] += self.link_weights[link_uid]
            if link_uid == selected_link_uid:
                self.link_effective_weights[link_uid] -= total_weights

        # return
        return selected_link_uid

    def get_normalized_flow(self) -> Dict[int, float]:
        """
        Get normalized flow from this node to each outbound link.

        :return: normalized flow
        """
        # check weights are not empty
        assert not len(self.link_weights) == 0, "Must set weights before computing normalized flow!"

        # get normalized flow
        sum_weights: float = sum(self.link_weights.values())
        normalized_flow_dict: Dict[int, float] = {link_uid: self.link_weights[link_uid] / sum_weights for link_uid in
                                                  self.link_weights}
        return normalized_flow_dict


class TopologySink:
    def __init__(self, node_uid: int, inbound_network_throughput: float) -> None:
        """
        TopologySource corresponds to sink node (i.e. coordinator) in the cluster, used for computing max flow.

        :param node_uid: sink node_uid
        :param inbound_network_throughput: throughput of the node for receiving requests (in #tokens/s)
        :return: None
        """
        # basic information
        self.node_uid: int = node_uid
        self.inbound_network_throughput: float = inbound_network_throughput

    def add_to_graph(self, graph: nx.DiGraph) -> None:
        """
        Add this node to a NetworkX directed graph.

        :param graph: the graph to add this node to
        :return: None
        """
        # add this node to graph
        #          inbound TP
        # sink_in -----------> sink
        graph.add_node(node_for_adding=f"{self.node_uid}_in")
        graph.add_edge(u_of_edge=f"{self.node_uid}_in", v_of_edge=f"sink",
                       capacity=self.inbound_network_throughput)
        graph.add_node(node_for_adding=f"sink")


class TopologyLink:
    def __init__(self, link_uid: int, prev_node_uid: int, next_node_uid: int, throughput: float) -> None:
        """
        TopologyLink corresponds to a network link in the cluster, used for computing max flow.

        :param link_uid: network link's link_uid
        :param prev_node_uid: node_uid of previous node
        :param next_node_uid: node_uid of next node
        :param throughput: throughput of this link (in #tokens/s)
        :return: None
        """
        # basic information
        self.link_uid: int = link_uid
        self.prev_node_uid: int = prev_node_uid
        self.next_node_uid: int = next_node_uid
        self.throughput: float = throughput

    def add_to_graph(self, graph: nx.DiGraph) -> None:
        """
        Add this link to a NetworkX directed graph.

        :param graph: the graph to add this node to
        :return: None
        """
        # add this link to graph
        graph.add_edge(u_of_edge=f"{self.prev_node_uid}_out", v_of_edge=f"{self.next_node_uid}_in",
                       capacity=self.throughput)

    def get_flow(self, flow_dict: dict) -> float:
        """
        Get flow over this link in flow dict.

        :param flow_dict: flow dict
        :return: flow
        """
        return flow_dict[f"{self.prev_node_uid}_out"][f"{self.next_node_uid}_in"]


class ClusterTopology:
    def __init__(self) -> None:
        """
        Topology of a cluster
        """
        # topology
        self.graph: nx.DiGraph or None = None
        self.source_node: TopologySource or None = None
        self.sink_node: TopologySink or None = None
        self.compute_nodes: Dict[int, TopologyNode] = {}
        self.network_links: Dict[int, TopologyLink] = {}

        # max flow
        self.flow_computed: bool = False
        self.max_flow: float = -1
        self.flow_dict: dict = {}

    def create_from_cluster(self, cluster: ClusterSimulator, parameters: MaxFlowParameters) -> None:
        """
        Create a topology graph from given cluster.
        Note: each compute node in the cluster must have models (and inference settings) set before calling
        this function.

        :param cluster: the cluster to create from
        :param parameters: max flow related parameters
        :return: None
        """
        # create a graph
        assert self.graph is None, "Can not load into non-empty cluster topology!"
        self.graph: nx.DiGraph = nx.DiGraph()

        # extract parameters
        token_size: float = parameters.token_size
        token_activation_size: float = parameters.token_activation_size

        # construct source and sink
        _source: SourceNode = cluster.source_node
        _source_network_throughput: float = _source.outbound_nic_speed / token_size
        self.source_node: TopologySource = TopologySource(node_uid=_source.node_uid,
                                                          outbound_network_throughput=_source_network_throughput,
                                                          outbound_link_uids=list(_source.outbound_links.keys()))
        self.source_node.add_to_graph(graph=self.graph)
        _sink: SinkNode = cluster.sink_node
        _sink_network_throughput: float = _sink.inbound_nic_speed / token_size
        self.sink_node: TopologySink = TopologySink(node_uid=_sink.node_uid,
                                                    inbound_network_throughput=_sink_network_throughput)
        self.sink_node.add_to_graph(graph=self.graph)

        # construct compute nodes
        for _compute_node_uid in cluster.compute_nodes:
            _compute_node: ComputeNode = cluster.compute_nodes[_compute_node_uid]

            # network throughput
            # _TODO: if we support partial inference, inbound of layer {1, 2} may contain both source and {1},
            #  we need to handle throughput accordingly
            _first_layer_id: int = 0
            _last_layer_id: int = max(cluster.model.keys())
            if _first_layer_id in _compute_node.in_vram_model_layers:
                _inbound_throughput: float = _compute_node.inbound_nic_speed / token_size
            else:
                _inbound_throughput: float = _compute_node.inbound_nic_speed / (token_size + token_activation_size)
            if _last_layer_id in _compute_node.in_vram_model_layers:
                _outbound_throughput: float = _compute_node.outbound_nic_speed / token_size
            else:
                _outbound_throughput: float = _compute_node.outbound_nic_speed / (token_size + token_activation_size)

            # inference throughput
            assert _compute_node.inference_settings is not None, "Node must have inference settings to compute flow!"
            _typical_batch_size: int = _compute_node.inference_settings.typical_batch_size
            _inference_time: float = 0
            for layer_id in _compute_node.in_vram_model_layers:
                _inference_time += _compute_node.in_vram_model_layers[layer_id].bs2time[_typical_batch_size]
            _inference_throughput: float = _typical_batch_size / _inference_time

            # add node
            assert _compute_node_uid not in self.compute_nodes, "Duplicate node found!"
            new_compute_node = TopologyNode(node_uid=_compute_node_uid,
                                            inbound_network_throughput=_inbound_throughput,
                                            outbound_network_throughput=_outbound_throughput,
                                            inference_throughput=_inference_throughput,
                                            outbound_link_uids=list(_compute_node.outbound_links.keys()))
            self.compute_nodes[_compute_node_uid] = new_compute_node
            new_compute_node.add_to_graph(graph=self.graph)

        # construct network links
        for _link_uid in cluster.links:
            _link: NetworkLink = cluster.links[_link_uid]

            # throughput
            if _link.node_in_type == NodeType.Source or _link.node_out_type == NodeType.Sink:
                _link_throughput: float = _link.bandwidth / token_size
            else:
                _link_throughput: float = _link.bandwidth / (token_size + token_activation_size)

            # add link
            new_link = TopologyLink(link_uid=_link_uid,
                                    prev_node_uid=_link.node_in.node_uid,
                                    next_node_uid=_link.node_out.node_uid,
                                    throughput=_link_throughput)
            self.network_links[_link_uid] = new_link
            new_link.add_to_graph(graph=self.graph)

    def compute_max_flow(self) -> Tuple[float, dict]:
        """
        Compute max flow and set the values for each node.

        :return: max flow value, flow dict
        """
        # check and compute flow
        assert self.graph is not None, "Must load from cluster before computing maxflow"
        self.max_flow, self.flow_dict = nx.maximum_flow(flowG=self.graph, _s="source", _t="sink")
        self.flow_computed = True

        # set round-robin weights for source node
        source_weights: Dict[int, float] = {}
        for link_uid in self.source_node.outbound_link_uids:
            link: TopologyLink = self.network_links[link_uid]
            flow: float = link.get_flow(flow_dict=self.flow_dict)
            source_weights[link_uid] = flow
        self.source_node.set_weights(weights=source_weights)

        # set round-robin weights for compute nodes
        for compute_node_uid in self.compute_nodes:
            compute_node: TopologyNode = self.compute_nodes[compute_node_uid]
            compute_node_weights: Dict[int, float] = {}
            for link_uid in compute_node.outbound_link_uids:
                link: TopologyLink = self.network_links[link_uid]
                flow: float = link.get_flow(flow_dict=self.flow_dict)
                compute_node_weights[link_uid] = flow
            compute_node.set_weights(weights=compute_node_weights)

        return self.max_flow, self.flow_dict

    def schedule_all_requests(self, node_uid: int, request_uids: List[int]) -> None:
        """
        Schedule all requests provided and write into cache. The "request_uids" can contain scheduled
        request uids and those scheduled ones will be ignored.

        :param node_uid: node uid
        :param request_uids: the list of request uids to schedule
        :return: None
        """
        assert self.flow_computed, "Max flow must be computed before scheduling"

        # locate the node
        if node_uid == self.source_node.node_uid:
            target_node = self.source_node
        else:
            assert node_uid in self.compute_nodes, "No matching node!"
            target_node = self.compute_nodes[node_uid]

        # find all unscheduled request uids
        unscheduled_request_uids: List[int] = []
        for request_uid in request_uids:
            if not target_node.transmission_scheduling_cache.contains(request_uid=request_uid):
                unscheduled_request_uids.append(request_uid)

        # schedule each unscheduled request using IWRR
        for unscheduled_request_uid in unscheduled_request_uids:
            cur_link_uid: int = target_node.interleaved_weighted_round_robin()
            target_node.transmission_scheduling_cache.add_request(request_uid=unscheduled_request_uid,
                                                                  link_uid=cur_link_uid)

    def get_next_requests(self, node_uid: int, link_uids: List[int]) -> Dict[int, int or None]:
        """
        Get next requests for each link from given node.

        :param node_uid: node uid
        :param link_uids: a list of links from this node
        :return: link_uid -> request_uid or None (if empty)
        """
        assert self.flow_computed, "Max flow must be computed before scheduling"

        # locate the node
        if node_uid == self.source_node.node_uid:
            target_node = self.source_node
        else:
            assert node_uid in self.compute_nodes, "No matching node!"
            target_node = self.compute_nodes[node_uid]

        # construct the dict
        schedule_dict: Dict[int, int or None] = {}
        for link_uid in link_uids:
            schedule_dict[link_uid] = target_node.transmission_scheduling_cache.get_next_request(link_uid=link_uid)
        return schedule_dict

    def get_normalized_flow(self, node_uid: int) -> Dict[int, float]:
        """
        Get normalized flow from the target node to each outbound link.

        :return: normalized flow
        """
        assert self.flow_computed, "Max flow must be computed before scheduling"

        # get normalized flow
        if node_uid == self.source_node.node_uid:
            return self.source_node.get_normalized_flow()
        else:
            assert node_uid in self.compute_nodes, "No matching node!"
            return self.compute_nodes[node_uid].get_normalized_flow()
