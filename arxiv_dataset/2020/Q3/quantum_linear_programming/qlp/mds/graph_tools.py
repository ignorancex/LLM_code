"""Routines which are useful to create and plot graphs
"""
from typing import Set, Tuple, Optional, List, Any, Text

from numpy import abs, ndarray
from numpy import random as rand
from math import factorial
import matplotlib.pyplot as plt
from seaborn import heatmap

import networkx as nx
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import (
    fast_gnp_random_graph,
    newman_watts_strogatz_graph,
)


def generate_graph(
    n_nodes: int, n_edges: int, n_edge_max: int = 5, seed: Optional[int] = None
) -> Set[Tuple[int]]:
    """Creates random edges for graph.

    Node labels are from 0 to n_nodes - 1

    Arguments:
        n_nodes: Number of nodes.
        n_edges: Number of edges.
        n_edge_max: Maximal number of edges for graph.
        seed: Random seed to set before graph is generated.

    Todo:
        What about connected / unconnected graphs?
        Am I creating the right graph?
        Think about ill input conditions.
    """
    assert n_edge_max > 0
    assert n_nodes * n_edge_max // 2 > n_edges

    if seed:
        rand.seed(42)

    nodes = {n: 0 for n in range(n_nodes)}
    edges = set()

    while len(edges) < n_edges:
        choices = [n for n, count in nodes.items() if count <= n_edge_max]
        e1, e2 = (int(e) for e in rand.choice(choices, size=2, replace=False))
        edge = (e2, e1) if e2 < e1 else (e1, e2)
        if edge not in edges:
            edges.add(edge)
            nodes[e1] += 1
            nodes[e2] += 2

    return edges


def get_plot(
    graph: Set[Tuple[int]],
    color_nodes: Optional[List[int]] = None,
    show_plot: bool = False,
    backend="mpl",
    seed: int = 42,
    directed: bool = False,
):
    """Creates graph plot from edge set.

    Arguments:
        graph: The graph to plot.
        color_nodes: Colors given nodes green, default is white.
        show_plot: Wether or not the plot should be rendered.
        backend: The plotting backend. Either "mpl" or "bokeh".
        directed: Plot graph as a directed graph
    """
    if backend == "mpl":
        fig = get_plot_mpl(
            graph,
            color_nodes=color_nodes,
            show_plot=show_plot,
            seed=seed,
            directed=directed,
        )
    elif backend == "bokeh":
        if directed:
            print("Directed graph not supported for bokeh")
        fig = get_plot_bokeh(graph, color_nodes=color_nodes, show_plot=show_plot)
    else:
        raise KeyError("Only implemented matplotlib ('mpl') or ")

    return fig


def _flatten(obj: Tuple[Any]) -> Tuple[Any]:
    """Flattens nested tuple of tuple to just a tuple.

    Returns iterator.

    Follows:
    https://symbiosisacademy.org/tutorial-index/python-flatten-nested-lists-tuples-sets/
    """
    for item in obj:
        if isinstance(item, tuple):
            yield from _flatten(item)
        else:
            yield item


def _v2str(v: Tuple[int]) -> str:
    """Maps nested tuple of ints to flat string.

    String is joined on '' if v_max < 9 else ','
    """
    if isinstance(v, tuple):
        vflat = list(_flatten(v))
        sep = "," if max(vflat) > 9 else ""
        out = sep.join(map(str, vflat))
    else:
        out = str(v)
    return out


def generate_hamming_graph(
    d: int, q: int, v_as_string: bool = False
) -> Tuple[Set[Tuple[str, str]], Text]:
    """Returns edges for hamming graph (caretesian product of complete graphs).

    See also http://mathworld.wolfram.com/HammingGraph.html

    Arguments:
        d: Dimension of graph (number of copies).
        q: Number of nodes for complete graph used in cartesian product.
        v_as_string: Convert vertices to strings of format "v10v11v12...".
    """
    kq = nx.complete_graph(q)
    graph = kq
    for _ in range(d - 1):
        graph = nx.cartesian_product(graph, kq)

    edges = set((_v2str(v1), _v2str(v2)) for v1, v2 in graph.edges)

    if not v_as_string:
        nodes = dict()
        n_nodes = 0
        int_edges = set()
        for v1, v2 in edges:
            if v1 not in nodes:
                nodes[v1] = n_nodes
                n_nodes += 1
            if v2 not in nodes:
                nodes[v2] = n_nodes
                n_nodes += 1

            int_edges.add((nodes[v1], nodes[v2]))
        edges = int_edges

    return edges, f"Hamming({d},{q})"


def generate_bipartite_graph(p: int, q: int) -> Tuple[Set[Tuple[int, int]], Text]:
    """Returns edges of a complete bipartite graph.

    K(4,4) is the fundamental unit of a Chimera graph.

    Arguments:
        p: Number of vertices in set 1
        q: Number of vertices in set 2
    """

    return set(bipartite.complete_bipartite_graph(p, q).edges), f"K({p},{q})"


def generate_erdos_renyi_graph(n: int, p: float) -> Tuple[Set[Tuple[int, int]], Text]:
    """Return edges of a Erdos Renyi graph.

    A generalized random graph.
    Can be disconnected

    The following statements are true on average

    ``
        np < 1 has O(log(n)) clusters
        np = 1 has O(n^2/3) clusters
        np -> c > 1 will cluster to one large component, sub clusters have O(log(n)) vertices

        p < ln(n)/n the graph will be disconnected
        p > ln(n)/n the graph will be connected
    ``

    G(n,p)

    Arguments:
        n: Number of vertices
        p: Probability of edge creation
    """
    G = fast_gnp_random_graph(n, p)
    return set(G.edges), f"G({n},{p})"


def generate_newman_watts_strogatz_graph(
    n: int, k: int, p: float
) -> Tuple[Set[Tuple[int, int]], Text]:
    """Returns edges of a NWS small-world graph.

    NWS graph is a knn ring graph + random connections.
    This is guaranteed to be connected since edges are not removed for shortcuts.
    The WS graph removes edges from the ring when spawning shortcuts.
    Has applications to modeling social networks.

    Arguments:
        n: Number of vertices
        k: Number of nearest neighbors
        p: Probability of spawning new edge on top of base ring

    """
    return set(newman_watts_strogatz_graph(n, k, p).edges), f"WS({n},{k},{p})"


def generate_nn_graph(v: int) -> Tuple[Set[Tuple[int, int]], Text]:
    """Returns edges of a 1 dimensional nearest neighbor graph.

    Arugments:
        v: Number of vertices
    """
    graph = {(i - 1, i) for i in range(1, v)}
    return graph, f"NN({v})"


def generate_corona_graph(k: int, n: int) -> Tuple[Set[Tuple[int, int]], Text]:
    """Returns edges of a n-shortcut 3k-ring graph with broken rotational symmetry.

    Has unique ground state solution with increased connectivity

    Arguments:
        k: number of 3 vertex segments
        n: number of shortcuts. must be < 3*k - 2 for this definition
    """

    if n >= factorial(3 * k - 2):
        raise ValueError(
            "n >= (3k-2)!: More shortcuts than allowed for graph.\n Choose a smaller number."
        )

    # construct base ring
    ring, _ = generate_nn_graph(3 * k)
    ring.add((3 * k - 1, 0))

    # add antigen
    antigen = ((3 * i + 1, 3 * k + i) for i in range(k))
    ring.update(antigen)

    # add shortcuts
    count = 0
    for nidx in range(k):
        for kidx in range(3 * k):
            if count < n:
                start = kidx
                end = (kidx + nidx + 2) % (3 * k)
                ring.add((start, end))
                count += 1
            else:
                break

    return ring, f"C19({k},{n})"


def generate_limited_corona_graph(k: int, n: int):
    """Returns edges of a n-shortcut 3k-ring graph with broken rotational symmetry.

    Has unique ground state solution with increased connectivity
    For limited corona, the shortcuts are added to keep max connections / vertex the smallest

    Arguments:
        k: number of 3 vertex segments
        n: number of shortcuts.
    """

    if n >= factorial(3 * k - 2):
        raise ValueError(
            "n >= (3k-2)!: More shortcuts than allowed for graph.\n Choose a smaller number."
        )

    # construct base ring
    ring, _ = generate_nn_graph(3 * k)
    ring.add((3 * k - 1, 0))

    # add antigen
    antigen = ((3 * i + 1, 3 * k + i) for i in range(k))
    ring.update(antigen)

    # connectivity
    connections = {idx: 2 for idx in range(3 * k)}
    for i in range(k):
        connections[3 * i + 1] += 1

    # add shortcuts
    shortcuts = [
        (kidx, (kidx + nidx + 2) % (3 * k))
        for nidx in range(k)
        for kidx in range(3 * k)
    ]
    count = 0
    minconnections = min(connections.values())
    while count < n:
        for idx, shortcut in enumerate(shortcuts):
            start = shortcut[0]
            end = shortcut[1]
            if connections[start] > minconnections or connections[end] > minconnections:
                deadend = True
                pass
            else:
                ring.add(shortcut)
                shortcuts.pop(idx)
                connections[start] += 1
                connections[end] += 1
                minconnections = min(connections.values())
                count += 1
                deadend = False
                break
        if idx == len(shortcuts) - 1 and deadend:
            minconnections += 1
    return ring, f"lC19({k},{n})"


def get_plot_mpl(
    graph: Set[Tuple[int]],
    color_nodes: Optional[List[int]] = None,
    show_plot: bool = False,
    ax: Optional["Axes"] = None,
    seed: int = 42,
    directed: bool = False,
) -> Optional["Figure"]:
    """Creates matplotlib graph plot from edge set.

    Arguments:
        graph: The graph to plot.
        color_nodes: Colors given nodes green, default is white.
        show_plot: Wether or not the plot should be rendered.
        ax: The axes to plot in
        directed: Plot graph as a directed graph
    """
    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = None

    G = nx.DiGraph(list(graph)) if directed else nx.Graph(list(graph))
    node_color = (
        ["lightgreen" if node in color_nodes else "white" for node in G.nodes]
        if color_nodes
        else "white"
    )

    if seed is not None:
        pos = nx.spring_layout(G, seed=seed)
        # pos = nx.spectral_layout(G)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_color,
        font_size=10,
        node_shape="o",
        linewidths=1.0,
        edgecolors="black",
        ax=ax,
    )

    if show_plot:
        plt.show()

    return fig


def get_plot_bokeh(  # pylint: disable=R0914
    graph: Set[Tuple[int]],
    color_nodes: Optional[List[int]] = None,
    show_plot: bool = False,
) -> "Plot":
    """Creates bokeh graph plot from edge set.

    Arguments:
        graph: The graph to plot.
        color_nodes: Colors given nodes green, default is white.
        show_plot: Wether or not the plot should be rendered.
    """
    from bokeh.io import show
    from bokeh.models import Plot, Range1d, Circle, HoverTool
    from bokeh.models.graphs import from_networkx

    color_nodes = color_nodes or []

    G = nx.Graph(list(graph))
    edge_attrs = {}
    node_attrs = {}

    for start_node, end_node, _ in G.edges(data=True):
        edge_attrs[(start_node, end_node)] = "black"

    for node in G.nodes:
        node_attrs[node] = "green" if node in color_nodes else "white"

    nx.set_edge_attributes(G, edge_attrs, "edge_color")
    nx.set_node_attributes(G, node_attrs, "node_color")

    # Show with Bokeh
    plot = Plot(
        plot_width=400,
        plot_height=400,
        x_range=Range1d(-1.1, 1.1),
        y_range=Range1d(-1.1, 1.1),
    )

    graph_renderer = from_networkx(G, nx.circular_layout, scale=1, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="node_color")
    plot.renderers.append(graph_renderer)  # pylint: disable=E1101

    node_hover_tool = HoverTool(tooltips=[("index", "@index")])
    plot.add_tools(node_hover_tool)

    if show_plot:
        show(plot)

    return plot


def plot_qubo(
    qubo, ax: Optional["Axes"] = None, axes: bool = False, fontsize: int = 4, **kwargs
) -> "Axes":
    """Plots qubo as heat map

    Kwargs are fed to heatmap.
    """
    if not ax:
        ax = plt.gca()

    m = qubo.todense() if not isinstance(qubo, ndarray) else qubo
    vmax = abs(m).max()

    data = dict(
        annot=True,
        annot_kws={"fontsize": fontsize},
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        mask=m == 0,
        cbar=False,
    )
    data.update(kwargs)

    heatmap(m, ax=ax, **data)

    if not axes:
        ax.set_yticks([])
        ax.set_xticks([])

    return ax


def main():
    """Creates a random graph (fixed seed) and plots it.
    """

    n_nodes = 5
    n_edges = 5
    n_edge_max = 3

    test_graph = generate_graph(n_nodes, n_edges, n_edge_max=n_edge_max, seed=42)
    get_plot(test_graph, show_plot=True)


if __name__ == "__main__":
    main()
