
# external imports
from __future__ import annotations
import networkx as nx

# internal imports
from .World import *
from .GlobalPlanning import *
from PlotObjects.plotobjects.palettes import *
import math

class AbstractionOptions:
    def __init__(self):
        self.onlyDirectConnections : bool = True

class GraphAbstraction:
    def __init__(
            self, 
            world : World, 
            gpp : GlobalPathPlanner = None, 
            opts : AbstractionOptions = None):
        self.graph : nx.DiGraph = None
        self.world: World = world
        self.options : AbstractionOptions = None
        self.setOptions(opts)

    def setOptions(self, opts : AbstractionOptions):
        if opts is None:
            self.options = AbstractionOptions()
            return
        if not isinstance(opts, AbstractionOptions):
            raise ValueError("opts must be of type AbstractionOptions")        
        self.options = opts

    def abstract(self, world : World, gpp : GlobalPathPlanner):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(world.targets())
        for t1 in world.targets():
            for t2 in world.targets():
                if t1 == t2:
                    continue
                path, time = gpp.planPathToTarget(t1.p(), t2)
                
                # No connection if no path found
                if path is None or time >= np.inf:
                    continue

                # No connection if we choose to drop indirect connections
                if self.options.onlyDirectConnections:
                    if not gpp.isDirectConnection(t1, t2, path):
                        continue

                # otherwise add the edge
                self.graph.add_edge(t1, t2, weight=time, path=path)

    def plotAbstraction(self, ax : plt.Axes = None, **kwargs):
        pos = nx.spring_layout(self.graph)
        for target in self.graph.nodes:
            pos[target] = target.p()
        
        # draw nodes
        node_labels = {}
        target : Target
        for target in self.graph.nodes:
            node_labels[target] = target.name
        nx.draw(self.graph, pos, ax=ax, labels=node_labels, with_labels=True, font_size=10)
        
        # draw edges
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edges(self.graph, pos, ax=ax)

    def highlightNeighborhood(self, node, agents: List[np.ndarray] = [], covered: List[Target] = [], alpha_non_neighbor=0.1, ax : plt.Axes = None, **kwargs):
        
        # define alphas
        alpha_node = lambda n : 1.0 if n == node or (n not in covered and n in self.graph[node]) else alpha_non_neighbor 
        alpha_edge = lambda n, m: 1.0 if n == node and m in self.graph[n] and m not in covered else alpha_non_neighbor           

        pos = nx.spring_layout(self.graph)
        target: Target
        for target in self.graph.nodes:
            pos[target] = target.p()
            self.graph.nodes[target]['name'] = f"T{target.name}"
            self.graph.nodes[target]['color'] = pastel_warm[0]
            self.graph.nodes[target]['alpha'] = alpha_node(target)
        
        # add agents
        for i, a in enumerate(agents):
            self.graph.add_node(i, pos=a, name=f"A{i}", color=pastel_cold[1], alpha=1.0)
            pos[i] = a

        # draw targets
        for n in self.graph.nodes:
            # if n in enumerate(agents):
            #     continue
            nx.draw_networkx_nodes(
                self.graph, pos, 
                nodelist=[n], 
                label=self.graph.nodes[n]['name'], 
                node_color=self.graph.nodes[n]['color'], 
                alpha=self.graph.nodes[n]['alpha']
            )

        # Remove agents again
        for i, a in enumerate(agents):
            self.graph.remove_node(i)

        for n in self.graph.nodes():
            for m in self.graph.nodes():
                if m not in self.graph[n]:
                    continue
                nx.draw_networkx_edges(
                    self.graph, 
                    pos, 
                    edgelist=[(n,m)],
                    alpha=alpha_edge(n, m)
                )
    
    def simpleGraph(self) -> nx.Graph:
        g = nx.Graph()
        t: Target
        for t in self.graph.nodes:
            g.add_node(t.name)
        for u, v, data in self.graph.edges(data=True):
            g.add_edge(u.name, v.name, weight=data['weight'])
        return g
        