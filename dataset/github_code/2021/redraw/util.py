import requests
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.decomposition import PCA
import sys


def to_tuple(x):
    if isinstance(x, list):
        return tuple([to_tuple(y) for y in x])
    else:
        return x


def compute_lattice(filename):
    # Request the lattice from conexp-clj
    req = {'id': 'req',
           'lat': {'type': 'function',
                   'name': 'concept-lattice',
                   'args': ['file']},
           'file': {'type': 'context_file',
                    'data': open(filename).read()}
           }
    answ = requests.post("http://127.0.0.1:8080", json=req).json()
    lat = answ['lat']['result']
    nodes = to_tuple(lat['nodes'])
    edges = set(to_tuple(lat['edges']))

    # lattice to networkx
    G = nx.DiGraph()
    for x in nodes:
        G.add_node(x)
    for x in edges:
        G.add_edge(x[0], x[1])
    return G


def transitive_reduction(Gplus):
    G = deepcopy(Gplus)
    for x in Gplus.nodes:
        for y in Gplus.nodes:
            if x != y and Gplus.has_edge(x, y):
                for z in Gplus.nodes:
                    if (y != z and x != z and Gplus.has_edge(y, z) and
                            G.has_edge(x, z)):
                        G.remove_edge(x, z)
    return G


class Order:
    def __init__(self, filename, dimension):
        self.Gplus = compute_lattice(filename)
        self.G = transitive_reduction(self.Gplus)
        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        self.revG = self.G.reverse()
        self.dimension = dimension
        self.init_drawing()
        self.reset_forces()

    def init_drawing(self):
        linext = []
        G_cpy = deepcopy(self.Gplus)
        while True:
            A = [x for x in G_cpy.nodes if G_cpy.out_degree(x) == 1]
            if len(A) == 0:
                break
            linext.append(A[0])
            G_cpy.remove_node(A[0])

        self.drawing = {x: np.append(y, np.random.rand(self.dimension - 1)
                                     - 0.5)
                        for x, y in zip(linext, range(len(linext)))}

    def reset_forces(self):
        self.forces = {x: np.zeros(self.dimension) for x in self.G.nodes}

    def apply_forces(self):
        sum_forces = 0
        for x in self.G.nodes:
            tmp = self.drawing[x][0]
            self.drawing[x] += self.forces[x]
            sum_forces += sum(np.absolute(self.forces[x]))
            if self.forces[x][0] > 0:
                upperb = min({self.drawing[x][0] for x in self.revG.adj[x]}.union(
                    {sys.float_info.max}))-1/10
                if self.drawing[x][0] > upperb:
                    self.drawing[x][0] = (tmp+upperb)/2
            else:
                lowerb = max({self.drawing[x][0] for x in self.G.adj[x]}.union(
                    {sys.float_info.min}))+1/10
                if self.drawing[x][0] < lowerb:
                    self.drawing[x][0] = (tmp+lowerb)/2

        self.reset_forces()
        return sum_forces

    def nodes(self):
        return self.G.nodes

    def edges(self):
        return self.G.edges

    def add_h_force(self, x, f):
        self.forces[x][1:] += f

    def add_v_force(self, x, f):
        self.forces[x][0] += f

    def add_force(self, x, f):
        self.forces[x] += f

    def pos(self, x):
        return self.drawing[x]

    def has_edge(self, x, y):
        return self.G.has_edge(x, y)

    def show(self):
        drawing = {x: [self.drawing[x][1], -self.drawing[x][0]] for x in self.G.nodes}
        nx.draw(self.G, drawing)
        plt.show()

    def save(self, path):
        drawing = {x: [self.drawing[x][1], -self.drawing[x][0]/2] for x in self.G.nodes}
        nx.draw(self.G, drawing)
        plt.gca().set_aspect('equal')
        plt.savefig(path)
        plt.clf()

    def is_comp(self, x, y):
        return self.Gplus.has_edge(x, y)

    def correct_offset(self):
        offset = np.mean(list(self.drawing.values()), axis=0)[1:]
        for x in self.G.nodes:
            self.drawing[x][1:] -= offset

    def dimension_stepdown(self):
        pca = PCA(n_components=self.dimension-2)
        k = list(self.drawing.keys())
        v = np.array(list(self.drawing.values()))
        pca.fit(v[:, 1:])
        proj = pca.transform(v[:, 1:])
        f = np.concatenate((v[:, 0:1], proj), axis=1)
        self.drawing = dict(zip(k, f))
        self.dimension -= 1
        self.reset_forces()

    def write_tikz(self, filename):
        with open(filename, "w") as f:
            f.write("\\begin{tikzpicture}\n")
            drawing = {x: [self.drawing[x][1], -self.drawing[x][0]/2] for x in self.G.nodes}
            for x in self.G.edges():
                f.write("\\draw ({:f},{:f})--({:f},{:f});\n".format(drawing[x[0]][0],drawing[x[0]][1],drawing[x[1]][0],drawing[x[1]][1]))
            for x in drawing.values():
                f.write("\\draw ({:f},{:f}) node () {{}};\n".format(x[0],x[1]))
            f.write("\\end{tikzpicture}")
