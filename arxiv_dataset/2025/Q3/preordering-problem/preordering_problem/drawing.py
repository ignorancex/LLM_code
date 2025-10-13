import numpy as np
import matplotlib.pyplot as plt
from preordering_problem.decompose_preorder import decompose_preorder
import networkx as nx
from matplotlib.patches import Circle, FancyArrow
from matplotlib.backend_bases import KeyEvent, MouseEvent, MouseButton


class PreorderPlot:
    """
    Interactive drawing of preorder represented as partial order on clusters.
    This tools allows to:

    - Drag and drop clusters to reposition them.
    - Toggle/un-toggle display of disagreements by pressing `d`.
    - Recompute the axis limits by pressing `a`.
    - Exporting the plot as a tikz graphic by right-clicking on it.
    """

    def __init__(self, adjacency, costs, ax=None):

        assert adjacency.ndim == 2
        assert adjacency.shape[0] == adjacency.shape[1]
        assert adjacency.shape == costs.shape

        self.adjacency = adjacency
        self.costs = costs

        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = ax.get_figure()
        self.ax.set_aspect('equal')

        self.transitive_reduction, self.clustering = decompose_preorder(adjacency)
        self.node2cluster = {i: ci for ci, cluster in enumerate(self.clustering) for i in cluster}
        self.disagreement_graph = nx.DiGraph()
        self.disagreement_graph.add_nodes_from(range(self.adjacency.shape[0]))
        for ci, cluster_i in enumerate(self.clustering):
            for cj, cluster_j in enumerate(self.clustering):
                for ii, i in enumerate(cluster_i):
                    for jj, j in enumerate(cluster_j):
                        if i == j:
                            continue
                        dis = None
                        if self.adjacency[i, j] == 1 and self.costs[i, j] < 0:
                            dis = "pos"
                        elif self.adjacency[i, j] == 0 and self.costs[i, j] > 0:
                            dis = "neg"
                        if dis is None:
                            continue
                        self.disagreement_graph.add_edge(i, j, dis=dis)

        self.fig.canvas.mpl_connect("key_press_event", self.toggle_dis)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

        self.r = 1
        self.selected_cluster = None
        self.selected_node = None
        self.name = None

    def on_click(self, event: MouseEvent):
        if event.inaxes != self.ax:
            return
        if event.button == MouseButton.RIGHT:
            self.generate_tikz()
            return
        coord = np.array([event.xdata, event.ydata])
        for i in self.transitive_reduction.nodes:
            handle = self.transitive_reduction.nodes[i]['handle']
            if np.linalg.norm(coord - handle.center) < handle.radius:
                self.selected_cluster = i
                for n in self.clustering[i]:
                    node_handle = self.disagreement_graph.nodes[n]["node_handle"]
                    if np.linalg.norm(coord - node_handle.center) < node_handle.radius:
                        self.selected_node = n

    def on_release(self, event: MouseEvent):
        self.selected_cluster = None
        self.selected_node = None

    def on_move(self, event: MouseEvent):
        if self.selected_cluster is None:
            return
        if event.inaxes != self.ax:
            return
        coords = np.array([event.xdata, event.ydata])
        if self.selected_node is None:
            self.transitive_reduction.nodes[self.selected_cluster]["handle"].center = coords
            for i in self.transitive_reduction.successors(self.selected_cluster):
                x, y, dx, dy = self.get_cluster_arc(self.selected_cluster, i)
                self.transitive_reduction[self.selected_cluster][i]["handle"].set_data(x=x, y=y, dx=dx, dy=dy)
            for i in self.transitive_reduction.predecessors(self.selected_cluster):
                x, y, dx, dy = self.get_cluster_arc(i, self.selected_cluster)
                self.transitive_reduction[i][self.selected_cluster]["handle"].set_data(x=x, y=y, dx=dx, dy=dy)
            for n in self.clustering[self.selected_cluster]:
                self.update_node_pos(n)

        else:
            cluster_center = self.transitive_reduction.nodes[self.selected_cluster]["handle"].center
            direction = coords - cluster_center
            angle = np.arctan2(direction[1], direction[0])
            if angle < 0:
                angle += 2*np.pi
            cluster = self.clustering[self.selected_cluster]
            i = cluster.index(self.selected_node)
            angles = np.arange(len(cluster)) / len(cluster) * 2 * np.pi
            diff = np.abs(angles - angle)
            diff[diff > np.pi] = 2*np.pi - diff[diff > np.pi]
            new_i = np.argmin(diff)
            if i != new_i:
                cluster.insert(new_i, cluster.pop(i))
                for n in cluster:
                    self.update_node_pos(n)

        self.fig.canvas.draw_idle()

    def update_node_pos(self, n):
        pos = self.get_node_pos(n)
        self.disagreement_graph.nodes[n]["node_handle"].center = pos
        self.disagreement_graph.nodes[n]["text_handle"].set(x=pos[0], y=pos[1])
        for s in self.disagreement_graph.successors(n):
            coords = self.get_arc_coords(n, s)
            self.disagreement_graph[n][s]["line_handle"].set_data(coords)
            self.disagreement_graph[n][s]["arrow_handle"].set_data(x=coords[0, -1], y=coords[1, -1],
                                                                   dx=coords[0, -1] - coords[0, -2],
                                                                   dy=coords[1, -1] - coords[1, -2])
        for p in self.disagreement_graph.predecessors(n):
            coords = self.get_arc_coords(p, n)
            self.disagreement_graph[p][n]["line_handle"].set_data(coords)
            self.disagreement_graph[p][n]["arrow_handle"].set_data(x=coords[0, -1], y=coords[1, -1],
                                                                   dx=coords[0, -1] - coords[0, -2],
                                                                   dy=coords[1, -1] - coords[1, -2])

    def toggle_dis(self, event: KeyEvent):
        if event.key == "d":
            for i, j in self.disagreement_graph.edges:
                for h in [self.disagreement_graph[i][j]["line_handle"], self.disagreement_graph[i][j]["arrow_handle"]]:
                    h.set_visible(not h.get_visible())
            self.fig.canvas.draw_idle()
        elif event.key == "a":
            self.reset_ax_lim()
            self.fig.canvas.draw_idle()

    def get_node_pos(self, n):
        c = self.node2cluster[n]
        cluster = self.clustering[c]
        i = cluster.index(n)
        a = 2 * np.pi * i / len(cluster)
        cluster_handle = self.transitive_reduction.nodes[c]["handle"]
        co = self.compute_cluster_orbit(len(cluster))
        offset = np.array([np.cos(a), np.sin(a)]) * co
        return cluster_handle.center + offset

    def compute_cluster_orbit(self, n):
        if n == 1:
            return 0
        return 2.5 * self.r / np.sqrt(2 - 2 * np.cos(2*np.pi / n))

    def plot_clusters(self):
        pos = nx.nx_pydot.graphviz_layout(self.transitive_reduction, prog="dot")
        pos = np.array(list(pos.values()))
        pos /= np.max(pos) / 2

        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        dist[np.diag_indices_from(dist)] = np.inf
        self.r = dist.min() / 8

        for i, cluster in enumerate(self.clustering):
            co = self.compute_cluster_orbit(len(cluster))
            cluster_circle = Circle(pos[i], radius=co + self.r * 3 / 2,
                                    facecolor="lightgray", edgecolor="black")
            self.transitive_reduction.nodes[i]["handle"] = cluster_circle
            self.ax.add_artist(cluster_circle)
            for j, n in enumerate(cluster):
                node_pos = self.get_node_pos(n)
                node_circle = Circle(node_pos, radius=self.r, facecolor="white", edgecolor="black")
                self.ax.add_artist(node_circle)
                text_handle = self.ax.text(*node_pos, str(n), va="center", ha="center")
                self.disagreement_graph.nodes[n]["node_handle"] = node_circle
                self.disagreement_graph.nodes[n]["text_handle"] = text_handle

    def get_cluster_arc(self, i, j):
        handle_i = self.transitive_reduction.nodes[i]["handle"]
        handle_j = self.transitive_reduction.nodes[j]["handle"]
        start = np.array(handle_i.center)
        end = np.array(handle_j.center)
        direction = end - start
        direction /= np.linalg.norm(direction)
        start += handle_i.radius * direction
        end -= handle_j.radius * direction
        return start[0], start[1], end[0] - start[0], end[1] - start[1]

    def plot_transitive_reduction(self):
        for i, j in self.transitive_reduction.edges:
            data = self.get_cluster_arc(i, j)
            arrow = FancyArrow(*data, width=self.r / 15, color='black',
                               length_includes_head=True, head_width=self.r / 3)
            self.ax.add_artist(arrow)
            self.transitive_reduction[i][j]["handle"] = arrow

    def get_arc_coords(self, i, j):
        handle_i = self.disagreement_graph.nodes[i]["node_handle"]
        handle_j = self.disagreement_graph.nodes[j]["node_handle"]
        start = np.array(handle_i.center)
        end = np.array(handle_j.center)
        direction = end - start
        dist = np.linalg.norm(direction)
        if dist != 0:
            direction /= dist
        start += handle_i.radius * direction
        end -= handle_j.radius * direction

        t = np.linspace(0, 1, 101)
        coords = start + (end-start)[None, :] * t[:, None]
        orthogonal_direction = np.array([direction[1], -direction[0]])
        coords += orthogonal_direction[None, :] * (t*(1-t))[:, None] * dist / 8
        return coords.T

    def plot_disagreement(self):
        for i, j in self.disagreement_graph.edges:
            dis = self.disagreement_graph[i][j]["dis"]
            coords = self.get_arc_coords(i, j)
            line = self.ax.plot(coords[0], coords[1], color="red",
                                linestyle="-" if dis == "pos" else "-", alpha=0.5)[0]
            arrow = FancyArrow(coords[0, -2], coords[1, -2], coords[0, -1] - coords[0, -2],
                               coords[1, -1] - coords[1, -2], color="red",
                               head_width=self.r / 4, length_includes_head=True,
                               head_length=self.r / 3)
            self.ax.add_artist(arrow)
            self.disagreement_graph[i][j]["arrow_handle"] = arrow
            self.disagreement_graph[i][j]["line_handle"] = line

    def reset_ax_lim(self):
        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf
        for i in self.transitive_reduction.nodes:
            handle = self.transitive_reduction.nodes[i]["handle"]
            if min_x > handle.center[0] - handle.radius:
                min_x = handle.center[0] - handle.radius
            if max_x < handle.center[0] + handle.radius:
                max_x = handle.center[0] + handle.radius
            if min_y > handle.center[1] - handle.radius:
                min_y = handle.center[1] - handle.radius
            if max_y < handle.center[1] + handle.radius:
                max_y = handle.center[1] + handle.radius
        pad = self.r
        self.ax.set_xlim(min_x - pad, max_x + pad)
        self.ax.set_ylim(min_y - pad, max_y + pad)

    def generate_tikz(self, f=None):
        """
        Export the plot as a tikz graphic.
        Define the following tikz styles in order to use the build the exported tikz figure:

        \tikzstyle{cluster}=[circle, inner sep=0pt, outer sep=0pt, draw, fill=black!20!white]
        \tikzstyle{cluster_arc}=[-{Latex[length=5pt, width=3pt]}, line width=0.8pt]
        \tikzstyle{disagreement_arc}=[-{Latex[length=4pt, width=3pt]}, line width=0.5pt, red!70!white]
        :param f: file to write to. If none, prints to console
        :return:
        """

        print("\\begin{tikzpicture}", file=f)
        print(f"\\def\\r{{{self.r:.3f}}}", file=f)

        for i in self.transitive_reduction.nodes:
            handle = self.transitive_reduction.nodes[i]["handle"]
            print(f"\\node[cluster, minimum size={2 * handle.radius / self.r:.3f}*\\r cm] (C{i}) at "
                  f"({handle.center[0] / self.r:.3f}*\\r, {handle.center[1] / self.r:.3f}*\\r) {{}};", file=f)

        for ci, cluster in enumerate(self.clustering):
            for i, n in enumerate(cluster):
                a = i / len(cluster) * 360
                d = np.linalg.norm(self.disagreement_graph.nodes[n]["node_handle"].center -
                                   self.transitive_reduction.nodes[ci]["handle"].center)
                s = self.disagreement_graph.nodes[n]["node_handle"].radius
                print(f"\\node[vertex, fill=white, shift=({a:.3f}:{d/self.r:.3f}*\\r cm), "
                      f"minimum size={2*s/self.r}*\\r cm] ({n})"
                      f" at (C{ci}) {{\\scriptsize {n}}};", file=f)

        for i, j in self.transitive_reduction.edges:
            print(f"\\draw[cluster_arc] (C{i}) -- (C{j});", file=f)

        for i, j in self.disagreement_graph.edges:
            print(f"\\draw[disagreement_arc] ({i}) -- ({j});", file=f)

        print("\\end{tikzpicture}", file=f)

    def plot(self):
        self.plot_clusters()
        self.plot_transitive_reduction()
        self.plot_disagreement()
        self.reset_ax_lim()


def example():
    adjacency = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1]
    ])
    cost = np.array([
        [1, -1, 1, -1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1]
    ])
    plotter = PreorderPlot(adjacency, cost)
    plotter.plot()
    plt.show()

if __name__ == '__main__':
    example()
