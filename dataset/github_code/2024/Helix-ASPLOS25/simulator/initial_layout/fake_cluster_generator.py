# 2023.02.22 Yixuan Mei
import os.path
import random
import itertools

from typing import Dict, List, Tuple
from configparser import ConfigParser
from simulator.event_simulator.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec


def create_weighted_list(strings: List[str], probabilities: List[float], m: int) -> List[str]:
    """
    Create a weighted list of strings based on the given probabilities and the total number of strings to generate.

    :param strings: the list of strings to be weighted
    :param probabilities: the list of probabilities for each string
    :param m: the total number of strings to generate
    :return: a list of strings (in random order)
    """
    # normalize the probabilities
    total_probability = sum(probabilities)
    probabilities = [p / total_probability for p in probabilities]

    # Calculate the exact counts for each string and adjust counts if rounding caused a discrepancy
    counts = [round(p * m) for p in probabilities]
    while sum(counts) != m:
        difference = m - sum(counts)
        adjustment = 1 if difference > 0 else -1
        for i in range(abs(difference)):
            counts[i % len(counts)] += adjustment

    # Generate the list by repeating each string the calculated number of times
    weighted_list = [string for string, count in zip(strings, counts) for _ in range(count)]

    # Shuffle the list to ensure random order
    random.shuffle(weighted_list)

    return weighted_list


class FakeClusterGenerator:
    def __init__(self) -> None:
        """
        FakeClusterGenerator generates a full cluster topology file based on given parameters. This
        cluster generated can be used to test initial layout algorithms.
        """
        # node statistics
        self.node_statistics_set: bool = False
        self.num_compute_nodes: int = -1
        self.avg_degree: int = -1
        self.source_degree: int = -1
        self.sink_degree: int = -1
        self.node_type_percentage: Dict[str, float] = {}

        # edge statistics
        self.edge_statistics_set: bool = False
        self.avg_bandwidth: float = -1
        self.var_bandwidth: float = -1
        self.avg_latency: float = -1
        self.var_latency: float = -1
        self.fill_with_slow_link: bool = False
        self.slow_link_avg_bandwidth: float = -1
        self.slow_link_var_bandwidth: float = -1
        self.slow_link_avg_latency: float = -1
        self.slow_link_var_latency: float = -1

    def set_node_statistics(self, num_compute_nodes: int, avg_degree: int, source_degree: int, sink_degree: int,
                            node_type_percentage: Dict[str, float]) -> None:
        """
        Set node statistics for this fake cluster generator.

        :param num_compute_nodes: number of compute nodes in the cluster
        :param avg_degree: average degree of compute nodes (< num compute nodes)
        :param source_degree: degree of source (<= num compute nodes)
        :param sink_degree: degree of sink (<= num compute nodes)
        :param node_type_percentage: type name -> percentage
        :return: None
        """
        # check input
        assert avg_degree < num_compute_nodes, "Avg degree must be smaller than the number of nodes!"
        assert source_degree <= num_compute_nodes and sink_degree <= num_compute_nodes, "Source/Sink degree too large!"

        # set node statistics
        self.num_compute_nodes = num_compute_nodes
        self.avg_degree = avg_degree
        self.source_degree = source_degree
        self.sink_degree = sink_degree
        self.node_type_percentage = node_type_percentage
        self.node_statistics_set = True

    def set_link_statistics(self, avg_bandwidth: float, var_bandwidth: float, avg_latency: float,
                            var_latency: float, fill_with_slow_link: bool = False,
                            slow_link_avg_bandwidth: float = -1, slow_link_var_bandwidth: float = -1,
                            slow_link_avg_latency: float = -1, slow_link_var_latency: float = -1) -> None:
        """
        Set edge statistics for this fake cluster generator.

        :param avg_bandwidth: average bandwidth
        :param var_bandwidth: variance of bandwidth
        :param avg_latency: average latency
        :param var_latency: variance of latency
        :param fill_with_slow_link: if true, we will use slow links to connect unconnected nodes with slow links
        :param slow_link_avg_bandwidth: average bandwidth of slow links
        :param slow_link_var_bandwidth: variance of bandwidth of slow links
        :param slow_link_avg_latency: average latency of slow links
        :param slow_link_var_latency: variance of latency of slow links
        :return: None
        """
        self.avg_bandwidth = avg_bandwidth
        self.var_bandwidth = var_bandwidth
        self.avg_latency = avg_latency
        self.var_latency = var_latency
        self.fill_with_slow_link = fill_with_slow_link
        self.slow_link_avg_bandwidth = slow_link_avg_bandwidth
        self.slow_link_var_bandwidth = slow_link_var_bandwidth
        self.slow_link_avg_latency = slow_link_avg_latency
        self.slow_link_var_latency = slow_link_var_latency
        self.edge_statistics_set = True

    def generator_fake_cluster(self, file_name: str, seed: int = 0) -> None:
        """
        Generate a fake cluster and write into the given file.
        File format convention:
            1. connected_nodes is sorted (source < compute node < sink)
            2. each link is bidirectional and appear in sorted order (i.e. source-i, i-j with i < j and j-sink)

        :param file_name: name of the file
        :param seed: random seed
        :return: None
        """
        # make sure the generator is properly initialized and set random seed
        assert self.node_statistics_set and self.edge_statistics_set, "Not initialized!"
        random.seed(seed)

        # create a partial graph from the complete graph of compute nodes based on avg degree
        # edges_to_keep: list of (begin, end), where begin < end
        num_edges_to_keep: int = self.num_compute_nodes * self.avg_degree // 2
        full_edge_list: List[Tuple[int, int]] = list(itertools.combinations(range(self.num_compute_nodes), 2))
        edges_to_keep: List[Tuple[int, int]] = random.sample(full_edge_list, num_edges_to_keep)

        # check that the degree of each node is no smaller than 2
        degree_list: List[int] = [0] * self.num_compute_nodes
        for edge in edges_to_keep:
            degree_list[edge[0]] += 1
            degree_list[edge[1]] += 1
        if not self.fill_with_slow_link:
            assert all([degree >= 2 for degree in degree_list]), "Found node with degree smaller than 2!"

        # determine the nodes that are connected to source and sink
        source_connected_nodes: List[int] = sorted(random.sample(range(self.num_compute_nodes), self.source_degree))
        sink_connected_nodes: List[int] = sorted(random.sample(range(self.num_compute_nodes), self.sink_degree))

        # determine the type of each node (strictly follows the percentage)
        compute_node_types: List[str] = create_weighted_list(strings=list(self.node_type_percentage.keys()),
                                                             probabilities=list(self.node_type_percentage.values()),
                                                             m=self.num_compute_nodes)

        # # open output file and generate cluster into the file
        with open(file_name, "w") as f:
            # output the parameters that generates the fake cluster
            f.write(f"# ***************** Fake cluster generated by FakeClusterGenerator ***************** #\n")
            f.write(f"# seed: {seed}\n")
            f.write(f"# Node Settings:\n")
            f.write(f"#     num_compute_nodes: {self.num_compute_nodes}\n")
            f.write(f"#     avg_degree: {self.avg_degree}\n")
            f.write(f"#     source_degree: {self.source_degree}\n")
            f.write(f"#     sink_degree: {self.sink_degree}\n")
            f.write(f"#     node_type_percentage: {self.node_type_percentage}\n")
            f.write(f"# Edge Settings:\n")
            f.write(f"#     avg_bandwidth: {self.avg_bandwidth} ({self.avg_bandwidth / mbps} mbps)\n")
            f.write(f"#     var_bandwidth: {self.var_bandwidth} ({self.var_bandwidth / mbps} mbps)\n")
            f.write(f"#     avg_latency: {self.avg_latency} ({self.avg_latency * 1000} MilliSec)\n")
            f.write(f"#     var_latency: {self.var_latency} ({self.var_latency * 1000} MilliSec)\n")
            f.write(f"#     fill_with_slow_link: {self.fill_with_slow_link}\n")
            f.write(f"#     slow_link_avg_bandwidth: {self.slow_link_avg_bandwidth} "
                    f"({self.slow_link_avg_bandwidth / kbps} kbps)\n")
            f.write(f"#     slow_link_var_bandwidth: {self.slow_link_var_bandwidth} "
                    f"({self.slow_link_var_bandwidth / kbps} kbps)\n")
            f.write(f"#     slow_link_avg_latency: {self.slow_link_avg_latency} "
                    f"({self.slow_link_avg_latency * 1000} MilliSec)\n")
            f.write(f"#     slow_link_var_latency: {self.slow_link_var_latency} "
                    f"({self.slow_link_var_latency * 1000} MilliSec)\n")
            f.write(f"# **********************************************************************************\n")
            f.write(f"\n")

            # write the list of nodes in the cluster
            f.write(f"[NodeNames]\n")
            f.write(f"total_compute_nodes={self.num_compute_nodes}\n")
            f.write(f"\n")

            # write type and connectivity for source and sink
            if not self.fill_with_slow_link:
                f.write(f"[SourceNode]\n")
                f.write(f"connected_nodes={source_connected_nodes}\n")
                f.write(f"\n")
                f.write(f"[SinkNode]\n")
                f.write(f"connected_nodes={sink_connected_nodes}\n")
                f.write(f"\n")
            else:
                f.write(f"[SourceNode]\n")
                f.write(f"connected_nodes={list(range(self.num_compute_nodes))}\n")
                f.write(f"\n")
                f.write(f"[SinkNode]\n")
                f.write(f"connected_nodes={list(range(self.num_compute_nodes))}\n")
                f.write(f"\n")

            # write type and connectivity for each compute node
            if not self.fill_with_slow_link:
                for i in range(self.num_compute_nodes):
                    # get the connected nodes
                    connected_nodes: List[int or str] = [edge[1] for edge in edges_to_keep if edge[0] == i]
                    connected_nodes += [edge[0] for edge in edges_to_keep if edge[1] == i]
                    connected_nodes = sorted(connected_nodes)

                    if i in source_connected_nodes:
                        connected_nodes = ["source"] + connected_nodes
                    if i in sink_connected_nodes:
                        connected_nodes = connected_nodes + ["sink"]

                    # write the properties of the compute node
                    f.write(f"[ComputeNode-{i}]\n")
                    f.write(f"type={compute_node_types[i]}\n")
                    f.write(f"connected_nodes={connected_nodes}\n")
                    f.write(f"\n")
            else:
                for i in range(self.num_compute_nodes):
                    # connected to all other nodes
                    connected_nodes: List[int or str] = ["source"] + list(range(self.num_compute_nodes)) + ["sink"]
                    connected_nodes.remove(i)

                    # write the properties of the compute node
                    f.write(f"[ComputeNode-{i}]\n")
                    f.write(f"type={compute_node_types[i]}\n")
                    f.write(f"connected_nodes={connected_nodes}\n")
                    f.write(f"\n")

            # define a function to write the properties of the network link between two nodes
            def write_link(from_node: str, to_node: str, is_slow: bool) -> None:
                """
                Write the properties of the network link between two nodes.

                :param from_node: the name of the from_node
                :param to_node: the name of the to_node
                :param is_slow: whether this link is a slow link
                :return: None
                """
                # get bandwidth and latency
                if not is_slow:
                    raw_bandwidth = self.avg_bandwidth + random.uniform(-self.var_bandwidth, self.var_bandwidth)
                    latency = int((self.avg_latency + random.uniform(-self.var_latency, self.var_latency)) * 1000)
                else:
                    raw_bandwidth = self.slow_link_avg_bandwidth + random.uniform(-self.slow_link_var_bandwidth,
                                                                                  self.slow_link_var_bandwidth)
                    latency = int((self.slow_link_avg_latency + random.uniform(-self.slow_link_var_latency,
                                                                               self.slow_link_var_latency)) * 1000)
                if raw_bandwidth > 10 * mbps:
                    bandwidth = int(raw_bandwidth / mbps)
                elif raw_bandwidth > 1 * mbps:
                    bandwidth = round(raw_bandwidth / mbps, 1)
                else:
                    bandwidth = round(raw_bandwidth / mbps, 3)

                # write the properties of the link
                f.write(f"[Link-{from_node}-{to_node}]\n")
                f.write(f"bandwidth={bandwidth} * mbps\n")
                f.write(f"latency={latency} * MilliSec\n")
                f.write(f"\n")

            # write properties of the network links
            if not self.fill_with_slow_link:
                for i in source_connected_nodes:
                    write_link(from_node="source", to_node=f"{i}", is_slow=False)
                for i in sink_connected_nodes:
                    write_link(from_node=f"{i}", to_node="sink", is_slow=False)
                for edge in edges_to_keep:
                    write_link(from_node=f"{edge[0]}", to_node=f"{edge[1]}", is_slow=False)
            else:
                # source
                for i in range(self.num_compute_nodes):
                    if i in source_connected_nodes:
                        write_link(from_node="source", to_node=f"{i}", is_slow=False)
                    else:
                        write_link(from_node="source", to_node=f"{i}", is_slow=True)

                # sink
                for i in range(self.num_compute_nodes):
                    if i in sink_connected_nodes:
                        write_link(from_node=f"{i}", to_node="sink", is_slow=False)
                    else:
                        write_link(from_node=f"{i}", to_node="sink", is_slow=True)

                # compute nodes
                for edge in full_edge_list:
                    if edge in edges_to_keep:
                        write_link(from_node=f"{edge[0]}", to_node=f"{edge[1]}", is_slow=False)
                    else:
                        write_link(from_node=f"{edge[0]}", to_node=f"{edge[1]}", is_slow=True)


class PartitionedClusterGenerator:
    def __init__(self) -> None:
        """
        PartitionedClusterGenerator generates a full cluster topology file based on given parameters. This
        cluster generated can be used to test initial layout algorithms.
        """
        # partition statistics
        self.partition_nodes_lists: List[List[str]] = []

        # network statistics
        self.in_partition_avg_bandwidth: float = -1
        self.in_partition_var_bandwidth: float = -1
        self.in_partition_avg_latency: float = -1
        self.in_partition_var_latency: float = -1
        self.cross_partition_avg_bandwidth: float = -1
        self.cross_partition_var_bandwidth: float = -1
        self.cross_partition_avg_latency: float = -1
        self.cross_partition_var_latency: float = -1

    def add_partition(self, nodes_list: List[str]) -> None:
        """
        Add a partition to the cluster.

        :param nodes_list: list of nodes in the partition
        :return: None
        """
        self.partition_nodes_lists.append(nodes_list)

    def set_network_statistics(self, in_partition_avg_bandwidth: float, in_partition_var_bandwidth: float,
                               in_partition_avg_latency: float, in_partition_var_latency: float,
                               cross_partition_avg_bandwidth: float, cross_partition_var_bandwidth: float,
                               cross_partition_avg_latency: float, cross_partition_var_latency: float) -> None:
        """
        Set network statistics for this fake cluster generator.

        :param in_partition_avg_bandwidth: average bandwidth within a partition
        :param in_partition_var_bandwidth: variance of bandwidth within a partition
        :param in_partition_avg_latency: average latency within a partition
        :param in_partition_var_latency: variance of latency within a partition
        :param cross_partition_avg_bandwidth: average bandwidth between partitions
        :param cross_partition_var_bandwidth: variance of bandwidth between partitions
        :param cross_partition_avg_latency: average latency between partitions
        :param cross_partition_var_latency: variance of latency between partitions
        :return: None
        """
        self.in_partition_avg_bandwidth = in_partition_avg_bandwidth
        self.in_partition_var_bandwidth = in_partition_var_bandwidth
        self.in_partition_avg_latency = in_partition_avg_latency
        self.in_partition_var_latency = in_partition_var_latency
        self.cross_partition_avg_bandwidth = cross_partition_avg_bandwidth
        self.cross_partition_var_bandwidth = cross_partition_var_bandwidth
        self.cross_partition_avg_latency = cross_partition_avg_latency
        self.cross_partition_var_latency = cross_partition_var_latency

    def generator_fake_cluster(self, file_name: str, seed: int = 0, create_separate: bool = True,
                               separate_path: str = "./") -> None:
        """
        Generate a fake cluster and write into the given file.
        File format convention:
            1. connected_nodes is sorted (source < compute node < sink)
            2. each link is bidirectional and appear in sorted order (i.e. source-i, i-j with i < j and j-sink)

        :param file_name: name of the file
        :param seed: random seed
        :param create_separate: whether to create separate cluster files for each type
        :param separate_path: the path to store the separate cluster files
        :return: None
        """
        # set random seed
        random.seed(seed)

        # join the list of nodes in each partition
        node_types: List[str] = []
        node_partition_id: List[int] = []
        for i, nodes_list in enumerate(self.partition_nodes_lists):
            node_types += nodes_list
            node_partition_id += [i] * len(nodes_list)

        # get the id of node in its own type
        node_type_count: Dict[str, int] = {}
        in_type_node_id: List[int] = []
        for i, node_type in enumerate(node_types):
            if node_type not in node_type_count:
                node_type_count[node_type] = 0
            in_type_node_id.append(node_type_count[node_type])
            node_type_count[node_type] += 1

        # edges to keep store the edges that are within the same partition
        # edges_to_keep: list of (begin, end), where begin < end
        full_edge_list: List[Tuple[int, int]] = list(itertools.combinations(range(len(node_types)), 2))
        edges_to_keep: List[Tuple[int, int]] = []
        for begin in range(len(node_types)):
            for end in range(begin + 1, len(node_types)):
                if node_partition_id[begin] == node_partition_id[end]:
                    edges_to_keep.append((begin, end))

        # get in type network edges
        in_type_edges: Dict[str, List[Tuple[int or str, int or str, float, float]]] = {
            machine_type: [] for machine_type in node_type_count.keys()}

        # # open output file and generate cluster into the file
        with open(file_name, "w") as f:
            # output the parameters that generates the fake cluster
            f.write(f"# ***************** Fake cluster generated by FakeClusterGenerator ***************** #\n")
            f.write(f"# seed: {seed}\n")
            f.write(f"# Node Settings:\n")
            f.write(f"#     num_compute_nodes: {len(node_types)}\n")
            for i, nodes_list in enumerate(self.partition_nodes_lists):
                f.write(f"#     partition-{i}: {nodes_list}\n")
            f.write(f"# Edge Settings:\n")
            f.write(f"#     in_partition_avg_bandwidth: {self.in_partition_avg_bandwidth} "
                    f"({self.in_partition_avg_bandwidth / mbps} mbps)\n")
            f.write(f"#     in_partition_var_bandwidth: {self.in_partition_var_bandwidth} "
                    f"({self.in_partition_var_bandwidth / mbps} mbps)\n")
            f.write(f"#     in_partition_avg_latency: {self.in_partition_avg_latency} "
                    f"({self.in_partition_avg_latency * 1000} MilliSec)\n")
            f.write(f"#     in_partition_var_latency: {self.in_partition_var_latency} "
                    f"({self.in_partition_var_latency * 1000} MilliSec)\n")
            f.write(f"#     cross_partition_avg_bandwidth: {self.cross_partition_avg_bandwidth} "
                    f"({self.cross_partition_avg_bandwidth / mbps} mbps)\n")
            f.write(f"#     cross_partition_var_bandwidth: {self.cross_partition_var_bandwidth} "
                    f"({self.cross_partition_var_bandwidth / mbps} mbps)\n")
            f.write(f"#     cross_partition_avg_latency: {self.cross_partition_avg_latency} "
                    f"({self.cross_partition_avg_latency * 1000} MilliSec)\n")
            f.write(f"#     cross_partition_var_latency: {self.cross_partition_var_latency} "
                    f"({self.cross_partition_var_latency * 1000} MilliSec)\n")
            f.write(f"# **********************************************************************************\n")
            f.write(f"\n")

            # write the list of nodes in the cluster
            f.write(f"[NodeNames]\n")
            f.write(f"total_compute_nodes={len(node_types)}\n")
            f.write(f"\n")

            # write type and connectivity for source and sink
            f.write(f"[SourceNode]\n")
            f.write(f"connected_nodes={list(range(len(node_types)))}\n")
            f.write(f"\n")
            f.write(f"[SinkNode]\n")
            f.write(f"connected_nodes={list(range(len(node_types)))}\n")
            f.write(f"\n")

            # write type and connectivity for each compute node
            for i in range(len(node_types)):
                # connected to all other nodes
                connected_nodes: List[int or str] = ["source"] + list(range(len(node_types))) + ["sink"]
                connected_nodes.remove(i)

                # write the properties of the compute node
                f.write(f"[ComputeNode-{i}]\n")
                f.write(f"type={node_types[i]}\n")
                f.write(f"connected_nodes={connected_nodes}\n")
                f.write(f"\n")

            # define a function to write the properties of the network link between two nodes
            def write_link(from_node: str, to_node: str, in_partition: bool) -> Tuple[float, float]:
                """
                Write the properties of the network link between two nodes.

                :param from_node: the name of the from_node
                :param to_node: the name of the to_node
                :param in_partition: whether this link is within the same partition
                :return: None
                """
                # get bandwidth and latency
                if not in_partition:
                    raw_bandwidth = self.cross_partition_avg_bandwidth + random.uniform(
                        -self.cross_partition_var_bandwidth, self.cross_partition_var_bandwidth)
                    latency = int((self.cross_partition_avg_latency + random.uniform(
                        -self.cross_partition_var_latency, self.cross_partition_var_latency)) * 1000)
                else:
                    raw_bandwidth = self.in_partition_avg_bandwidth + random.uniform(
                        -self.in_partition_var_bandwidth, self.in_partition_var_bandwidth)
                    latency = int((self.in_partition_avg_latency + random.uniform(
                        -self.in_partition_var_latency, self.in_partition_var_latency)) * 1000)
                if raw_bandwidth > 100 * mbps:
                    bandwidth = int(raw_bandwidth / mbps)
                elif raw_bandwidth > 1 * mbps:
                    bandwidth = round(raw_bandwidth / mbps, 1)
                else:
                    bandwidth = round(raw_bandwidth / mbps, 3)

                # write the properties of the link
                f.write(f"[Link-{from_node}-{to_node}]\n")
                f.write(f"bandwidth={bandwidth} * mbps\n")
                f.write(f"latency={latency} * MilliSec\n")
                f.write(f"\n")
                return bandwidth, latency

            # write properties of the network links
            # source and sink
            for i in range(len(node_types)):
                cur_bandwidth, cur_latency = write_link(from_node="source", to_node=f"{i}", in_partition=False)

                # recording
                cur_node_type = node_types[i]
                cur_in_type_id = in_type_node_id[i]
                in_type_edges[cur_node_type].append(("source", cur_in_type_id, cur_bandwidth, cur_latency))
            for i in range(len(node_types)):
                cur_bandwidth, cur_latency = write_link(from_node=f"{i}", to_node="sink", in_partition=False)

                # recording
                cur_node_type = node_types[i]
                cur_in_type_id = in_type_node_id[i]
                in_type_edges[cur_node_type].append((cur_in_type_id, "sink", cur_bandwidth, cur_latency))

            # compute nodes
            for edge in full_edge_list:
                if edge in edges_to_keep:
                    cur_bandwidth, cur_latency = write_link(
                        from_node=f"{edge[0]}", to_node=f"{edge[1]}", in_partition=True)
                else:
                    cur_bandwidth, cur_latency = write_link(
                        from_node=f"{edge[0]}", to_node=f"{edge[1]}", in_partition=False)

                node_id_1, node_id_2 = edge[0], edge[1]
                if node_types[node_id_1] == node_types[node_id_2]:
                    in_type_id_1 = in_type_node_id[node_id_1]
                    in_type_id_2 = in_type_node_id[node_id_2]
                    in_type_edges[node_types[node_id_1]].append(
                        (in_type_id_1, in_type_id_2, cur_bandwidth, cur_latency))

        # use in_type_edges to write separate files
        if create_separate:
            for type_name, all_edges in in_type_edges.items():
                with open(os.path.join(separate_path, f"./{type_name.lower()}.ini"), "w") as sub_file:
                    # node names
                    sub_file.write(f"[NodeNames]\n")
                    sub_file.write(f"total_compute_nodes={node_type_count[type_name]}\n")
                    sub_file.write(f"\n")

                    # source and sink
                    sub_file.write(f"[SourceNode]\n")
                    sub_file.write(f"connected_nodes={list(range(node_type_count[type_name]))}\n")
                    sub_file.write(f"\n")
                    sub_file.write(f"[SinkNode]\n")
                    sub_file.write(f"connected_nodes={list(range(node_type_count[type_name]))}\n")
                    sub_file.write(f"\n")

                    # compute nodes
                    for i in range(node_type_count[type_name]):
                        # connected to all other nodes
                        connected_nodes: List[int or str] = ["source"] + list(range(node_type_count[type_name])) + [
                            "sink"]
                        connected_nodes.remove(i)

                        # write the properties of the compute node
                        sub_file.write(f"[ComputeNode-{i}]\n")
                        sub_file.write(f"type={type_name}\n")
                        sub_file.write(f"connected_nodes={connected_nodes}\n")
                        sub_file.write(f"\n")

                    # edges
                    for edge_info in all_edges:
                        src_id, dst_id, bandwidth, latency = edge_info
                        sub_file.write(f"[Link-{src_id}-{dst_id}]\n")
                        sub_file.write(f"bandwidth={bandwidth} * mbps\n")
                        sub_file.write(f"latency={latency} * MilliSec\n")
                        sub_file.write(f"\n")


def prune_cluster(complete_cluster_file_name: str, pruned_cluster_file_name: str,
                  min_keep: int, max_keep: int, keep_bandwidth_threshold: float) -> None:
    """
    Prune a cluster file.

    :param complete_cluster_file_name: name of the complete cluster file (a complete graph)
    :param pruned_cluster_file_name: name of the pruned cluster file
    :param min_keep: keep at least first min_keep links for each node (sorted by bandwidth)
    :param max_keep: keep at most first max_keep links for each node (sorted by bandwidth)
    :param keep_bandwidth_threshold: for links between [min_keep, max_keep), only those faster than
           the threshold will be kept.
    :return: None
    """
    # check inputs
    assert min_keep <= max_keep, "min_keep must be smaller than or equal to max_keep!"

    # load the ini and check whether it is a complete graph
    complete_config = ConfigParser()
    complete_config.read(complete_cluster_file_name)
    total_num_compute_nodes = eval(complete_config["NodeNames"]["total_compute_nodes"])
    assert sorted(eval(complete_config["SourceNode"]["connected_nodes"])) == list(range(total_num_compute_nodes)), \
        "Not a complete graph!"
    assert sorted(eval(complete_config["SinkNode"]["connected_nodes"])) == list(range(total_num_compute_nodes)), \
        "Not a complete graph!"
    for i in range(total_num_compute_nodes):
        connected_nodes: List[int or str] = ["source"] + list(range(total_num_compute_nodes)) + ["sink"]
        connected_nodes.remove(i)
        assert eval(complete_config[f"ComputeNode-{i}"]["connected_nodes"]) == connected_nodes, "Not a complete graph!"

    # prune the source connections
    source_link_speed: List[Tuple[float, Tuple[str, int]]] = []
    for i in range(total_num_compute_nodes):
        link_bandwidth = eval(complete_config[f"Link-source-{i}"]["bandwidth"])
        source_link_speed.append((link_bandwidth, ("source", i)))
    source_link_speed.sort(reverse=True)
    pruned_source_connections: List[str] = []
    # keep the first min_keep links (need to check no overflow)
    for i in range(min(min_keep, len(source_link_speed))):
        pruned_source_connections.append(f"Link-source-{source_link_speed[i][1][1]}")
    # keep the next max_keep links that are faster than the threshold
    for i in range(min(min_keep, len(source_link_speed)), min(max_keep, len(source_link_speed))):
        if source_link_speed[i][0] > keep_bandwidth_threshold:
            pruned_source_connections.append(f"Link-source-{source_link_speed[i][1][1]}")
    pruned_source_connections.sort()

    # prune the sink connections
    sink_link_speed: List[Tuple[float, Tuple[int, str]]] = []
    for i in range(total_num_compute_nodes):
        link_bandwidth = eval(complete_config[f"Link-{i}-sink"]["bandwidth"])
        sink_link_speed.append((link_bandwidth, (i, "sink")))
    sink_link_speed.sort(reverse=True)
    pruned_sink_connections: List[str] = []
    # keep the first min_keep links (need to check no overflow)
    for i in range(min(min_keep, len(sink_link_speed))):
        pruned_sink_connections.append(f"Link-{sink_link_speed[i][1][0]}-sink")
    # keep the next max_keep links that are faster than the threshold
    for i in range(min(min_keep, len(sink_link_speed)), min(max_keep, len(sink_link_speed))):
        if sink_link_speed[i][0] >= keep_bandwidth_threshold:
            pruned_sink_connections.append(f"Link-{sink_link_speed[i][1][0]}-sink")
    pruned_sink_connections.sort()

    # prune the compute node connections
    pruned_compute_connections: List[str] = []
    for i in range(total_num_compute_nodes):
        # get link speed
        link_speed: List[Tuple[float, str]] = []
        for j in range(total_num_compute_nodes):
            if i != j:
                if f"Link-{i}-{j}" in complete_config:
                    assert f"Link-{j}-{i}" not in complete_config, "Duplicate edge found!"
                    assert i < j, "Wrong link idx order!"
                    link_speed.append((eval(complete_config[f"Link-{i}-{j}"]["bandwidth"]), f"Link-{i}-{j}"))
                else:
                    assert j < i, "Wrong link idx order!"
                    link_speed.append((eval(complete_config[f"Link-{j}-{i}"]["bandwidth"]), f"Link-{j}-{i}"))
        link_speed.sort(reverse=True)

        # keep the first min_keep links (need to check no overflow)
        cur_keep_links: List[str] = []
        for idx in range(min(min_keep, len(link_speed))):
            cur_keep_links.append(link_speed[idx][1])

        # keep the next max_keep links that are faster than the threshold
        for idx in range(min(min_keep, len(link_speed)), min(max_keep, len(link_speed))):
            if link_speed[idx][0] >= keep_bandwidth_threshold:
                cur_keep_links.append(link_speed[idx][1])

        # insert the links into the pruned_compute_connections
        for link_name in cur_keep_links:
            if link_name not in pruned_compute_connections:
                pruned_compute_connections.append(link_name)
    pruned_compute_connections.sort()

    # write the pruned file
    with open(pruned_cluster_file_name, "w") as pruned_file:
        # write header into the pruned file
        pruned_file.write(f"# ********************************************************************************** #\n")
        pruned_file.write(f"# This file is pruned from {complete_cluster_file_name}.\n")
        pruned_file.write(f"# Pruning Settings:\n")
        pruned_file.write(f"#     min_keep: {min_keep}\n")
        pruned_file.write(f"#     max_keep: {max_keep}\n")
        pruned_file.write(f"#     keep_bandwidth_threshold: {keep_bandwidth_threshold} "
                          f"({keep_bandwidth_threshold / mbps} mbps)\n")
        pruned_file.write(f"# Original file heading is as follows:\n")
        with open(complete_cluster_file_name, "r") as complete_file:
            ori_lines = complete_file.readlines()
            for line in ori_lines:
                if line.startswith("#"):
                    pruned_file.write(line)
        pruned_file.write(f"\n")

        # write node names
        pruned_file.write(f"[NodeNames]\n")
        pruned_file.write(f"total_compute_nodes={total_num_compute_nodes}\n")
        pruned_file.write(f"\n")

        # write source node
        pruned_file.write(f"[SourceNode]\n")
        source_connected_nodes = []
        for i in eval(complete_config["SourceNode"]["connected_nodes"]):
            if f"Link-source-{i}" in pruned_source_connections:
                source_connected_nodes.append(i)
        pruned_file.write(f"connected_nodes={source_connected_nodes}\n")
        pruned_file.write(f"\n")

        # write sink node
        pruned_file.write(f"[SinkNode]\n")
        sink_connected_nodes = []
        for i in eval(complete_config["SinkNode"]["connected_nodes"]):
            if f"Link-{i}-sink" in pruned_sink_connections:
                sink_connected_nodes.append(i)
        pruned_file.write(f"connected_nodes={sink_connected_nodes}\n")
        pruned_file.write(f"\n")

        # write compute nodes
        for i in range(total_num_compute_nodes):
            pruned_file.write(f"[ComputeNode-{i}]\n")
            connected_nodes = []
            for j in eval(complete_config[f"ComputeNode-{i}"]["connected_nodes"]):
                if j == "source":
                    if f"Link-source-{i}" in pruned_source_connections:
                        connected_nodes.append(j)
                elif j == "sink":
                    if f"Link-{i}-sink" in pruned_sink_connections:
                        connected_nodes.append(j)
                else:
                    if f"Link-{i}-{j}" in pruned_compute_connections or f"Link-{j}-{i}" in pruned_compute_connections:
                        connected_nodes.append(j)
            pruned_file.write(f"type={complete_config[f'ComputeNode-{i}']['type']}\n")
            pruned_file.write(f"connected_nodes={connected_nodes}\n")
            pruned_file.write(f"\n")

        # write links
        for link_name in pruned_source_connections:
            pruned_file.write(f"[{link_name}]\n")
            pruned_file.write(f"bandwidth={complete_config[link_name]['bandwidth']}\n")
            pruned_file.write(f"latency={complete_config[link_name]['latency']}\n")
            pruned_file.write(f"\n")
        for link_name in pruned_sink_connections:
            pruned_file.write(f"[{link_name}]\n")
            pruned_file.write(f"bandwidth={complete_config[link_name]['bandwidth']}\n")
            pruned_file.write(f"latency={complete_config[link_name]['latency']}\n")
            pruned_file.write(f"\n")
        for link_name in pruned_compute_connections:
            pruned_file.write(f"[{link_name}]\n")
            pruned_file.write(f"bandwidth={complete_config[link_name]['bandwidth']}\n")
            pruned_file.write(f"latency={complete_config[link_name]['latency']}\n")
            pruned_file.write(f"\n")
