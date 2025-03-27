
# add code to import path
from pathlib import Path
from scipy.sparse import lil_array, csc_array
import numpy as np
import networkx as nx
import graph_tool as gt
import graph_tool.topology as top

import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

def cell_boundary_matrix(edges: list[tuple[int,int]], cells: list[tuple]) -> csc_array:
    edge_index = {idx: edge for edge, idx in enumerate(edges)}
    cell_boundary = lil_array((len(edges), len(cells)), dtype=np.float64)
    for upper_idx, cell in enumerate(cells):
        for i, _ in enumerate(cell):
            j = (i + 1) % len(cell)
            lower_idx = edge_index[tuple(sorted([cell[i], cell[j]]))]
            orientation = 1 if cell[i] < cell[j] else -1
            cell_boundary[lower_idx, upper_idx] = orientation
    return csc_array(cell_boundary)

def get_bodnar_lifting(G, max_k=7):
    """
    Returns rings used in the lifting procedure of [1].

    Adapted from method `get_rings(edge_index, max_k=7)` in:
    https://github.com/twitter-research/cwn/blob/aac2f3b/data/utils.py (line 300)
    (MIT License)

    [1] Cristian Bodnar, Fabrizio Frasca, Nina Otter, Yuguang Wang, Pietro Liò, Guido F Montufar, and Michael Bronstein. 2021.
        Weisfeiler and Lehman Go Cellular: CW Networks.
        In Advances in Neural Information Processing Systems, Curran Associates, Inc., 2625–2640.
        Retrieved from https://proceedings.neurips.cc/paper_files/paper/2021/file/157792e4abb490f99dbd738483e0d2d4-Paper.pdf 
    """
    edge_list = list(G.edges)
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)

    # We represent rings with their original node ordering
    # so that we can easily read out the boundaries
    # The use of the `sorted_rings` set allows to discard
    # different isomorphisms which are however associated
    # to the same original ring – this happens due to the intrinsic
    # symmetries of cycles
    rings = set()
    sorted_rings = set()
    for k in range(3, max_k+1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings