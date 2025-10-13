import os
from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import torch
from torch_geometric.data import Batch, Data, Dataset  # type: ignore
from torch_geometric.datasets import ZINC, TUDataset  # type: ignore


class WeisfeilerLemanLabelingTree:
    """Weisfeiler Leman Labeling Tree

    Attributes:
        depth (int): Number of layers in WILT
        norm (str): Normalization method. "size" or "dummy"
        n_nodes (int): Number of nodes in WILT
        parent (list[int]): Parent node of each node in WILT
        attr2label (dict[int, int]): Mapping from attribute to label (for layer 1)
        labeling_hash (dict[tuple[int, tuple[int, ...]], int]): Hash table of labelings (for layer 1, layer 2, ..., layer depth)
        parameter (torch.Tensor): Parameter of each edge in WILT

    Properties:
        weight (torch.Tensor): Weight of each edge in WILT

    Structure:
        __init__
        |-- _build_tree
            |-- _adjancy_list
            |-- reset_parameter

        weight

        calc_distance_between_graphs
        |-- calc_subtree_weights
        |-- calc_distance_between_subtree_weights

        test_data_wwo_unseen_nodes

        reset_parameter
        load_parameter
        train
        eval
    """

    ##################
    # initialization #
    ##################
    def __init__(
        self,
        data: Dataset,
        depth: int,
        norm: str = "dummy",
        edgelabel: Optional[bool] = None,
    ) -> None:
        """initialize WILT

        Args:
            data (Dataset): Dataset
            depth (int): Number of layers in WILT
            norm (str, optional): Normalization method. "size" or "dummy". Defaults to "dummy".
            edgelabel (Optional[bool], optional): Whether to consider edge labels. If None, consider edge labels when data have them. Defaults to None.
        """
        self.depth = depth
        self.norm = norm
        assert self.depth > 0, "Depth should be greater than 0"
        if edgelabel is True:
            assert data[0].edge_attr is not None, "Data does not have edge labels"
        if edgelabel is None:
            self.edgelabel = data[0].edge_attr is not None
        else:
            self.edgelabel = edgelabel
        if type(data) == TUDataset:
            self.onehot_node_label = True
            if self.edgelabel:
                self.onehot_edge_label = True
        elif type(data) == ZINC:
            self.onehot_node_label = False
            self.onehot_edge_label = False
        else:
            self.onehot_node_label = self._check_node_label_encoding(data)
            if self.edgelabel:
                self.onehot_edge_label = self._check_edge_label_encoding(data)
        self._build_tree(data)

    def _check_node_label_encoding(self, data: Dataset) -> bool:
        """return whether node label is one-hot

        Args:
            data (Dataset): Dataset

        Returns:
            bool: whether node label is one-hot
        """
        if len(data[0].x.shape) != 2:
            return False
        if data[0].x.shape[1] == 1:
            return False
        for g in data:
            for v in g.x:
                if torch.sum(v) != 1 or len(torch.nonzero(v)) != len(v) - 1:
                    return False
        return True

    def _check_edge_label_encoding(self, data: Dataset) -> bool:
        """return whether edge label is one-hot

        Args:
            data (Dataset): Dataset

        Returns:
            bool: whether edge label is one-hot
        """
        if len(data[0].edge_attr.shape) != 2:
            return False
        if data[0].edge_attr.shape[1] == 1:
            return False
        for g in data:
            for e in g.edge_attr:
                if torch.sum(e) != 1 or len(torch.nonzero(e)) != len(e) - 1:
                    return False
        return True

    def _adjancy_list(self, graph: Data) -> list[list[tuple[int, int]]]:
        """convert edge_index to adjancy list

        Args:
            graph (Data): graph

        Returns:
            list[list[tuple[int, int]]]: adjancy list
        """
        adj_list: list[list[tuple[int, int]]] = [[] for _ in range(graph.num_nodes)]
        for idx, (u, v) in enumerate(graph.edge_index.T):
            adj_list[u.item()].append((v.item(), idx))
        return adj_list

    def _build_tree(self, data) -> None:
        """Build WILT tree

        Args:
            data (Dataset): Dataset
        """
        cnt_nodes: list[int] = [
            0 for _ in range(self.depth + 1)
        ]  # cnt_nodes[i] = number of nodes in layer [i] (The layer of root node is 0)
        parent: list[list[int]] = [
            [] for _ in range(self.depth + 1)
        ]  # parent[i][j] = parent node of node j in layer i (parent of root node is -1)
        attr2label: dict[int, int] = {}  # attr2label[i] = label of attribute i
        labeling_hash: list[dict[tuple[int, tuple[tuple[int, int], ...]], int]] = [
            {} for _ in range(self.depth + 1)
        ]  # labeling_hash[i] = dict of labelings in layer i (labeling_hash[0] is not used)

        # root node
        cnt_nodes[0] = 1
        parent[0].append(-1)

        # iterate over dataset # O(|G| * (|E| + self.depth * |V| * log|E|))
        for g in data:
            # initial labeling # O(|V|)
            current_labeling: list[int] = [-1 for _ in range(g.num_nodes)]
            for node_idx, node_attr in enumerate(
                torch.argmax(g.x, dim=1) if self.onehot_node_label else g.x.reshape(-1)
            ):
                if attr2label.get(node_attr.item()) is None:
                    attr2label[node_attr.item()] = cnt_nodes[1]
                    cnt_nodes[1] += 1
                    parent[1].append(0)  # root node
                current_labeling[node_idx] = attr2label[node_attr.item()]
            # adjancy_list # O(|V| + |E|)
            adj_list: list[list[tuple[int, int]]] = self._adjancy_list(g)
            # iterative labeling # O(self.depth * |V| * log|E|)
            for d in range(2, self.depth + 1):
                new_labeling: list[int] = [-1 for _ in range(g.num_nodes)]
                for node_idx in range(g.num_nodes):
                    # TODO: linear time sort
                    idx: tuple[int, tuple[tuple[int, int], ...]] = (
                        current_labeling[node_idx],
                        tuple(
                            sorted(
                                [
                                    (
                                        current_labeling[nv],
                                        (
                                            int(
                                                torch.argmax(
                                                    g.edge_attr[edge_idx]
                                                ).item()
                                                if self.onehot_edge_label
                                                else g.edge_attr[edge_idx].item()
                                            )
                                            if self.edgelabel
                                            else 0
                                        ),
                                    )
                                    for nv, edge_idx in adj_list[node_idx]
                                ]
                            )
                        ),
                    )
                    if labeling_hash[d].get(idx) is None:
                        labeling_hash[d][idx] = cnt_nodes[d]
                        cnt_nodes[d] += 1
                        parent[d].append(current_labeling[node_idx])
                    new_labeling[node_idx] = labeling_hash[d][idx]
                current_labeling = new_labeling

        # relabeling
        tuple2int: dict[tuple[int, int], int] = {}
        n_nodes = 0
        for layer_idx in range(self.depth + 1):
            for label_idx in range(cnt_nodes[layer_idx]):
                tuple2int[(layer_idx, label_idx)] = n_nodes
                n_nodes += 1
        self.n_nodes = n_nodes  # sum(cnt_nodes)
        self.parent: list[int] = [-1 for _ in range(self.n_nodes)]
        for layer_idx in range(1, self.depth + 1):
            for label_idx in range(cnt_nodes[layer_idx]):
                self.parent[tuple2int[(layer_idx, label_idx)]] = tuple2int[
                    (layer_idx - 1, parent[layer_idx][label_idx])
                ]
        self.attr2label: dict[int, int] = {
            k: tuple2int[(1, v)] for k, v in attr2label.items()
        }
        self.labeling_hash: dict[tuple[int, tuple[tuple[int, int], ...]], int] = {}
        for layer_idx in range(2, self.depth + 1):
            for key, val in labeling_hash[layer_idx].items():
                label_idx = tuple2int[(layer_idx - 1, key[0])]
                neighbors_idx = tuple(
                    [
                        (tuple2int[(layer_idx - 1, nv)], edge_val)
                        for nv, edge_val in key[1]
                    ]
                )  # already sorted
                self.labeling_hash[(label_idx, neighbors_idx)] = tuple2int[
                    (layer_idx, val)
                ]

        # weight
        self.reset_parameter()

    ########################
    # distance calculation #
    ########################
    def calc_subtree_weights(self, graph: Data) -> torch.Tensor:
        """calculate distribution on WILT

        Args:
            graph (Data): graph

        Returns:
            torch.Tensor: distribution on WILT
        """
        dist = torch.zeros(self.n_nodes, dtype=torch.float32)
        current_labeling: list[int] = [-1 for _ in range(graph.num_nodes)]
        # root node
        dist[0] = graph.num_nodes
        # initial labeling
        for node_idx, node_attr in enumerate(
            torch.argmax(graph.x, dim=1)
            if self.onehot_node_label
            else graph.x.reshape(-1)
        ):
            current_labeling[node_idx] = self.attr2label[node_attr.item()]
            dist[self.attr2label[node_attr.item()]] += 1
        # adjancy_list
        adj_list: list[list[tuple[int, int]]] = self._adjancy_list(graph)
        # iterative labeling
        for _ in range(2, self.depth + 1):
            new_labeling: list[int] = [-1 for _ in range(graph.num_nodes)]
            for node_idx in range(graph.num_nodes):
                # TODO: linear time sort
                idx: tuple[int, tuple[tuple[int, int], ...]] = (
                    current_labeling[node_idx],
                    tuple(
                        sorted(
                            [
                                (
                                    current_labeling[nv],
                                    (
                                        int(
                                            torch.argmax(
                                                graph.edge_attr[edge_idx]
                                            ).item()
                                            if self.onehot_edge_label
                                            else graph.edge_attr[edge_idx].item()
                                        )
                                        if self.edgelabel
                                        else 0
                                    ),
                                )
                                for nv, edge_idx in adj_list[node_idx]
                            ]
                        )
                    ),
                )
                new_labeling[node_idx] = self.labeling_hash[idx]
                dist[self.labeling_hash[idx]] += 1
            current_labeling = new_labeling
        # normalize
        if self.norm == "size":
            dist /= graph.num_nodes
        return dist

    def calc_distance_between_subtree_weights(
        self,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
    ) -> torch.Tensor:
        """calculate distance between two distributions on WILT

        Args:
            weight1 (torch.Tensor): subtree_weight vector(s)
            weight2 (torch.Tensor): subtree_weight vector(s)

        Returns:
            torch.Tensor: distance(s)
        """
        if len(weight1.shape) == 1:
            return torch.dot(
                torch.abs(weight1 - weight2),
                self.weight,
            )  # (1, )
        else:
            # batch execution
            return torch.abs(weight1 - weight2) @ self.weight  # (batch_size,)

    def calc_distance_between_graphs(
        self,
        graph1: Data | Batch | list[Data],
        graph2: Data | Batch | list[Data],
    ) -> torch.Tensor:
        """calculate distance between two graphs

        Args:
            graph1 (Data | Batch): graph(s)
            graph2 (Data | Batch): graph(s)

        Returns:
            torch.Tensor: distance(s)
        """
        if isinstance(graph1, Data) and isinstance(graph2, Data):
            graph1 = [graph1]
            graph2 = [graph2]
        elif isinstance(graph1, Batch) and isinstance(graph2, Batch):
            graph1 = graph1.to_data_list()
            graph2 = graph2.to_data_list()
        dist1 = torch.stack(
            [self.calc_subtree_weights(g) for g in graph1], dim=0
        )  # (batch_size, self.n_nodes)
        dist2 = torch.stack(
            [self.calc_subtree_weights(g) for g in graph2], dim=0
        )  # (batch_size, self.n_nodes)
        return self.calc_distance_between_subtree_weights(dist1, dist2)

    ########################
    # test data separation #
    ########################
    def identify_nodes_convered_by_subset(
        self, data: Dataset, indices: torch.Tensor
    ) -> torch.Tensor:
        """identify nodes covered by subset of data

        Args:
            data (Dataset): Dataset
            indices (torch.Tensor): indices of subset

        Returns:
            torch.Tensor: nodes covered by subset
        """
        subset = data[indices]
        seen = torch.zeros(self.n_nodes, dtype=torch.bool)
        for graph in subset:
            dist = self.calc_subtree_weights(graph)
            seen = torch.logical_or(seen, dist > 0)
        return seen

    def test_data_wwo_unseen_nodes(
        self, data: Dataset, train_indices: torch.Tensor, test_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """classify test data into ones with and without unseen nodes

        Args:
            data (Dataset): Dataset
            train_indices (torch.Tensor): indices of training data
            test_indices (torch.Tensor): indices of test data

        Returns:
            tuple[torch.Tensor, torch.Tensor]: indices for test data with/without unseen nodes
        """
        train_seen = self.identify_nodes_convered_by_subset(data, train_indices)
        train_unseen = torch.logical_not(train_seen)
        test_seen_indices = []
        test_unseen_indices = []
        for i in test_indices:
            test_dist = self.calc_subtree_weights(data[i])
            if torch.any(torch.logical_and(test_dist > 0, train_unseen)):
                test_unseen_indices.append(i)
            else:
                test_seen_indices.append(i)
        return (
            torch.tensor(test_seen_indices),
            torch.tensor(test_unseen_indices),
        )

    ##############
    # parameters #
    ##############
    @property
    def weight(self) -> torch.Tensor:
        return self.parameter

    def reset_parameter(self) -> None:
        self.parameter = torch.ones(self.n_nodes, dtype=torch.float32)

    def load_parameter(self, path: str) -> None:
        self.parameter = torch.load(path)
        self.parameter = self.parameter.to(torch.float32)

    def train(self) -> None:
        self.parameter.requires_grad = True

    def eval(self) -> None:
        self.parameter.requires_grad = False

    #################
    # visualization #
    #################
    def get_node_attr(self, node: int) -> int:
        """get attribute of node

        Args:
            node (int): node of WILT

        Returns:
            int: attribute of node
        """
        assert node >= 1
        if node in self.label2attr:
            return self.label2attr[node]
        else:
            return self.get_node_attr(self.labeling_hash_inv[node][0])

    def _prepare_inverse_dict(self) -> None:
        """prepare inverse dictionary if not exists"""
        if not hasattr(self, "label2attr"):
            self.label2attr: dict[int, int] = {v: k for k, v in self.attr2label.items()}
        if not hasattr(self, "labeling_hash_inv"):
            self.labeling_hash_inv: dict[
                int, tuple[int, tuple[tuple[int, int], ...]]
            ] = {v: k for k, v in self.labeling_hash.items()}

    def unfolding_tree(self, node: int, path: str) -> None:
        """Unfolding tree visualization

        Args:
            node (int): node of WILT to unfold
            path (str): Path to save image
        """
        if node == 0:
            print("No corresponding unfolding tree for node 0")
            return

        self._prepare_inverse_dict()

        unfolded = nx.Graph()
        queue = [(-1, node, -1)]  # parent, now, edge_attr
        node_cnt = 0
        while len(queue) > 0:
            pa, now, edge_attr = queue.pop(0)
            if now in self.labeling_hash_inv:
                unfolded.add_node(
                    node_cnt,
                    label=self.get_node_attr(now),
                    color="tab:red" if node_cnt == 0 else "tab:blue",
                )
                if pa != -1:
                    if self.edgelabel:
                        unfolded.add_edge(pa, node_cnt, label=edge_attr)
                    else:
                        unfolded.add_edge(pa, node_cnt)
                _, neighbors = self.labeling_hash_inv[now]
                for nv, edge_attr in neighbors:
                    queue.append((node_cnt, nv, edge_attr))
            else:
                unfolded.add_node(
                    node_cnt,
                    label=self.label2attr[now],
                    color="tab:red" if node_cnt == 0 else "tab:blue",
                )
                if pa != -1:
                    if self.edgelabel:
                        unfolded.add_edge(pa, node_cnt, label=edge_attr)
                    else:
                        unfolded.add_edge(pa, node_cnt)
            node_cnt += 1
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(unfolded)
        nx.draw(unfolded, pos)
        nx.draw_networkx_nodes(
            unfolded,
            pos,
            node_color=[node[1]["color"] for node in unfolded.nodes(data=True)],
        )
        nx.draw_networkx_labels(
            unfolded,
            pos,
            labels=nx.get_node_attributes(unfolded, "label"),
        )
        if self.edgelabel:
            nx.draw_networkx_edge_labels(
                unfolded,
                pos,
                edge_labels=nx.get_edge_attributes(unfolded, "label"),
            )
        else:
            nx.draw_networkx_edges(unfolded, pos)
        plt.savefig(path)
        plt.close()

    def color_unfolding_tree_in_graph(
        self, node_wilt: int, graph: Data, path: str
    ) -> None:
        """Color unfolding tree in graph

        Args:
            node_wilt (int): node of WILT to unfold
            graph (Data): graph
            path (str): Path to the directory to save images
        """
        if node_wilt == 0:
            print("No corresponding unfolding tree for node 0")
            return

        self._prepare_inverse_dict()

        # identify root node
        current_labeling: list[int] = [-1 for _ in range(graph.num_nodes)]
        root_candidate: list[bool] = [False for _ in range(graph.num_nodes)]
        root_depth = -1
        # initial labeling
        for node_idx, node_attr in enumerate(
            torch.argmax(graph.x, dim=1)
            if self.onehot_node_label
            else graph.x.reshape(-1)
        ):
            current_labeling[node_idx] = self.attr2label[node_attr.item()]
            if self.attr2label[node_attr.item()] == node_wilt:
                root_candidate[node_idx] = True
                root_depth = 1
        # adjancy_list
        adj_list: list[list[tuple[int, int]]] = self._adjancy_list(graph)
        # iterative labeling
        for d in range(2, self.depth + 1):
            new_labeling: list[int] = [-1 for _ in range(graph.num_nodes)]
            for node_idx in range(graph.num_nodes):
                # TODO: linear time sort
                idx: tuple[int, tuple[tuple[int, int], ...]] = (
                    current_labeling[node_idx],
                    tuple(
                        sorted(
                            [
                                (
                                    current_labeling[nv],
                                    (
                                        int(
                                            torch.argmax(
                                                graph.edge_attr[edge_idx]
                                            ).item()
                                            if self.onehot_edge_label
                                            else graph.edge_attr[edge_idx].item()
                                        )
                                        if self.edgelabel
                                        else 0
                                    ),
                                )
                                for nv, edge_idx in adj_list[node_idx]
                            ]
                        )
                    ),
                )
                new_labeling[node_idx] = self.labeling_hash[idx]
                if self.labeling_hash[idx] == node_wilt:
                    root_candidate[node_idx] = True
                    root_depth = d
            current_labeling = new_labeling
        if root_depth == -1:
            # no subgraph corresponding to node_wilt
            return

        for root_idx in range(graph.num_nodes):
            if not root_candidate[root_idx]:
                continue
            # identify nodes that are in d-hop neighborhood of root node
            d_hop_neighborhood: list[bool] = [False for _ in range(graph.num_nodes)]
            d_hop_neighborhood[root_idx] = True
            current_neighbors: list[int] = [root_idx]
            for _ in range(root_depth - 1):
                next_neighbors: list[int] = []
                for v in current_neighbors:
                    for nv, _ in adj_list[v]:
                        if not d_hop_neighborhood[nv]:
                            d_hop_neighborhood[nv] = True
                            next_neighbors.append(nv)
                current_neighbors = next_neighbors
            # color the nodes and edges
            colored_graph = nx.Graph()
            for node_idx, node_attr in enumerate(
                torch.argmax(graph.x, dim=1)
                if self.onehot_node_label
                else graph.x.reshape(-1)
            ):
                colored_graph.add_node(
                    node_idx,
                    label=node_attr.item(),
                    color="tab:red" if d_hop_neighborhood[node_idx] else "tab:blue",
                )
            if self.edgelabel:
                for (u, v), edge_attr in zip(
                    graph.edge_index.T,
                    (
                        torch.argmax(graph.edge_attr, dim=1)
                        if self.onehot_edge_label
                        else graph.edge_attr
                    ),
                ):
                    colored_graph.add_edge(
                        u.item(),
                        v.item(),
                        label=edge_attr.item(),
                        color=(
                            "tab:red"
                            if d_hop_neighborhood[u.item()]
                            and d_hop_neighborhood[v.item()]
                            else "tab:blue"
                        ),
                    )
            else:
                for u, v in graph.edge_index.T:
                    colored_graph.add_edge(
                        u.item(),
                        v.item(),
                        color=(
                            "tab:red"
                            if d_hop_neighborhood[u.item()]
                            and d_hop_neighborhood[v.item()]
                            else "tab:blue"
                        ),
                    )

            # save the image
            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(colored_graph)
            nx.draw(colored_graph, pos)
            nx.draw_networkx_nodes(
                colored_graph,
                pos,
                node_color=[
                    node[1]["color"] for node in colored_graph.nodes(data=True)
                ],
            )
            nx.draw_networkx_edges(
                colored_graph,
                pos,
                edge_color=[
                    edge[2]["color"] for edge in colored_graph.edges(data=True)
                ],
            )
            nx.draw_networkx_labels(
                colored_graph,
                pos,
                labels=nx.get_node_attributes(colored_graph, "label"),
            )
            if self.edgelabel:
                nx.draw_networkx_edge_labels(
                    colored_graph,
                    pos,
                    edge_labels=nx.get_edge_attributes(colored_graph, "label"),
                )
            else:
                nx.draw_networkx_edges(colored_graph, pos)
            plt.savefig(os.path.join(path, f"{root_idx}.png"))
            plt.close()
