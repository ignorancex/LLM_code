import os
import sys
from typing import Optional

import pytest
import torch
from torch_geometric.data import Batch  # type: ignore
from torch_geometric.datasets import TUDataset  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from path import DATA_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, depth, n_nodes, edgelabel",
    [
        ("MUTAG", 3, 1 + 7 + 64 + 277, True),
        ("MUTAG", 4, 1 + 7 + 33 + 174 + 572, False),
        ("MUTAG", 5, 1 + 7 + 64 + 277 + 796 + 1453, None),
        ("Mutagenicity", 3, 1 + 14 + 334 + 4997, True),
        ("Mutagenicity", 3, 1 + 14 + 274 + 4327, False),
        ("Mutagenicity", 5, 1 + 14 + 334 + 4997 + 21118 + 43750, None),
        ("NCI1", 3, 1 + 37 + 292 + 4058, False),
        ("NCI1", 5, 1 + 37 + 292 + 4058 + 22948 + 44508, None),
    ],
)
def test_WeisfeilerLemanLabelingTree_init(
    dataset_name: str, depth: int, n_nodes: int, edgelabel: Optional[bool]
):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth, edgelabel=edgelabel)
    assert tree.n_nodes == n_nodes


@pytest.mark.parametrize(
    "dataset_name, depth",
    [
        ("MUTAG", 1),
        ("MUTAG", 3),
        ("Mutagenicity", 1),
        ("Mutagenicity", 3),
        ("NCI1", 1),
        ("NCI1", 3),
    ],
)
def test_WeisfeilerLemanLabelingTree_calc_distance_between_subtree_weights(
    dataset_name: str, depth: int
):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    graph1 = torch.rand(3, tree.n_nodes)
    graph2 = torch.rand(3, tree.n_nodes)
    distance = tree.calc_distance_between_subtree_weights(graph1, graph2)
    for d in distance:
        assert d >= 0
    graph1 = torch.rand(tree.n_nodes)
    graph2 = torch.rand(tree.n_nodes)
    distance = tree.calc_distance_between_subtree_weights(graph1, graph2)
    assert d >= 0


@pytest.mark.parametrize(
    "dataset_name, depth",
    [
        ("MUTAG", 1),
        ("MUTAG", 3),
        ("Mutagenicity", 1),
        ("Mutagenicity", 3),
        ("NCI1", 1),
        ("NCI1", 3),
    ],
)
def test_WeisfeilerLemanLabelingTree_calc_distance_between_graphs(
    dataset_name: str, depth: int
):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    graph1 = [data[0], data[1], data[2]]
    graph2 = [data[3], data[4], data[5]]
    distance = tree.calc_distance_between_graphs(graph1, graph2)
    for d in distance:
        assert d >= 0
    graph1 = Batch.from_data_list([data[6], data[7], data[8]])
    graph2 = Batch.from_data_list([data[9], data[10], data[11]])
    distance = tree.calc_distance_between_graphs(graph1, graph2)
    for d in distance:
        assert d >= 0


@pytest.mark.parametrize(
    "dataset_name, depth, edgelabel",
    [
        ("MUTAG", 3, True),
        ("MUTAG", 4, False),
        ("MUTAG", 5, None),
        ("Mutagenicity", 3, True),
        ("Mutagenicity", 3, False),
        ("Mutagenicity", 5, None),
        ("NCI1", 3, False),
        ("NCI1", 5, None),
    ],
)
def test_unfolding_tree(
    tmpdir, dataset_name: str, depth: int, edgelabel: Optional[bool]
):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth, edgelabel=edgelabel)
    tree.unfolding_tree(10, os.path.join(tmpdir, "test_unfolding_tree.png"))
    assert os.path.exists(os.path.join(tmpdir, "test_unfolding_tree.png"))


@pytest.mark.parametrize(
    "dataset_name, depth, edgelabel",
    [
        ("MUTAG", 3, True),
        ("MUTAG", 4, False),
        ("MUTAG", 5, None),
        ("Mutagenicity", 3, True),
        ("Mutagenicity", 3, False),
        ("Mutagenicity", 5, None),
        ("NCI1", 3, False),
        ("NCI1", 5, None),
    ],
)
def test_color_unfolding_tree_in_graph(
    tmpdir, dataset_name: str, depth: int, edgelabel: Optional[bool]
):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth, edgelabel=edgelabel)
    embedding = tree.calc_subtree_weights(data[0])
    nonzero = torch.nonzero(embedding)
    os.mkdir(os.path.join(tmpdir, "test_color_unfolding_tree_in_graph"))
    tree.color_unfolding_tree_in_graph(
        nonzero[-1],
        data[0],
        os.path.join(tmpdir, "test_color_unfolding_tree_in_graph"),
    )
