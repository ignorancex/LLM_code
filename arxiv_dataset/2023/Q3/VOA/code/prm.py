import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from typing import List, Tuple
import numpy as np

map_size = 100


class Plotter:
    def __init__(self):
        self.fig = plt.figure(2)
        self.ax = self.fig.subplots()

    def add_obstacles(self, obstacles: List[Polygon]):
        for obstacle in obstacles:
            self.ax.add_patch(plt.Polygon(
                obstacle.exterior.coords, color='black'))

    def add_prm(self, prm):
        for i, node in enumerate(prm):
            # print('plot',i+1,'/',len(prm))
            for edge in node.edges_lines:
                self.ax.add_line(plt.Line2D(
                    list(edge.coords.xy[0]), list(edge.coords.xy[1]), 1, alpha=0.3))

        # plot the nodes after the edges so they appear above them
        for node in prm:
            # Do not plot isolated nodes
            if len(node.edges) > 0:
                plt.scatter(*node.pos.coords.xy, 4, color='red')

    def add_sol(self, sol):
        for i, node in enumerate(sol):
            if node != sol[-1]:
                next_node = sol[i+1]
                for j, edge in enumerate(node.edges):
                    if next_node == edge[0]:
                        line = node.edges_lines[j]
                        self.ax.add_line(plt.Line2D(list(line.coords.xy[0]), list(line.coords.xy[1]),
                                                    1, color='red'))

    def show_prm_graph(self, N_nodes, thd, edges, avg_node_degree):
        plt.autoscale()
        plt.title('$N_{nodes}=$'+str(N_nodes)+', $th_{d}=$'+str(thd)+', #edges=' +
                  str(int(edges))+', avg node degree=%.2f' % (avg_node_degree))
        plt.grid(color='lightgray', linestyle='--')

    def show_sol(self, N_nodes, thd, alg, cost):
        plt.autoscale()
        plt.title('$N_{nodes}=$'+str(N_nodes)+', $th_{d}=$' +
                  str(thd)+', Algorithm='+alg+', path cost=%.2f' % (cost))
        plt.grid(color='lightgray', linestyle='--')
        plt.show()


class NodePRM:
    def __init__(self, pos: Point):
        """Creates a single NodePRM containing only the nodes location with no edges

        Args:
            pos (Point): Location of the node in a 2D space
        """
        self.pos = pos
        self.edges = []
        self.edges_lines = []  # hold the lines for the plots

    def add_edge(self, neighbour, edge: LineString) -> None:
        """Adding an edge between 2 nodes in a PRM graph

        Args:
            neighbour (NodePRM): A prm node in the PRM graph other then self
            edge (LineString): edge connecting the self node and the neighbor node
        """
        # every edge in self.edges is described by the neighbour node and the weight
        self.edges.append((neighbour, edge.length))
        self.edges_lines.append(edge)


def GeneratePRM(C_obs: List[Polygon], N: float, init_location: List[float] = [0, 0], goal_location: List[float] = [map_size, map_size]) -> List[NodePRM]:
    """Creates a PRM graph including goven initial and goal location.

    Args:
        C_obs (List[Polygon]): Obstacles on the map
        N (float): Size of the environment NxN
        init_location (List[float], optional): Initial location on the map. Defaults to [0, 0].
        goal_location (List[float], optional): Goal location on themap. Defaults to [N, N].

    Returns:
        List[NodePRM]: _description_
    """
    graph = []

    thd = 2*N
    N_nodes = 30

    for i in range(N_nodes):
        # draw a sample from the 2D environment
        if i == 0:
            # initial location
            if init_location == None:
                sample = Point(0, 0)
            else:
                sample = Point(init_location[0], init_location[1])

        elif i == 1:
            # goal locarion
            if goal_location == None:
                sample = Point(N, N)
            else:
                sample = Point(goal_location[0], goal_location[1])

        else:
            sample = Point(np.random.uniform(0, N, 1),
                           np.random.uniform(0, N, 1))

        # check if the sample is collision free (in C_free)
        in_c_free = True
        for obstacle in C_obs:
            if sample.intersects(obstacle):
                in_c_free = False
                break

        if in_c_free:
            # add the new node to the graph
            new_vertex = NodePRM(sample)
            graph.append(new_vertex)

            for v in graph:
                if new_vertex == v:
                    continue

                edge = LineString([new_vertex.pos, v.pos])
                # consider all neighbours that are less than thd away
                if edge.length < thd:
                    # check if the edge intersects any obstacle
                    connect = True
                    for obstacle in C_obs:
                        if edge.intersects(obstacle):
                            connect = False
                            break

                    if connect:
                        new_vertex.add_edge(v, edge)
                        v.add_edge(new_vertex, edge)
    return graph


def ReGeneratePRM(prm: List[NodePRM], C_obs: List[Polygon], start=[0, 0], goal=[map_size, map_size]):
    """Regenerating an existing PRM graph with new initial and goal locations, while all other nodes remain the same.

    Args:
        prm (List[NodePRM]): A list of the nodes of the copied PRM graph including the connections between them
        thd (float): Maximal allowd length of edge
        C_obs (List[Polygon]): Obstacles on the map
        start (list, optional): Initial location in the 2D map. Defaults to [0, 0].
        goal (list, optional): Initial location in the 2D map. Defaults to [N, N].

    Returns:
        graph: A list of PRMNodes including the new initial and goal locations and edges connecting between them
    """
    # initialize nodes with new start and goal locations
    thd = 2*map_size

    if start == None:
        start_point = prm[0].pos
    else:
        start_point = Point(start[0], start[1])

    if goal == None:
        goal_point = prm[1].pos
    else:
        goal_point = Point(goal[0], goal[1])

    graph = [NodePRM(start_point),
             NodePRM(goal_point)]

    for vertex in prm:
        node_point = vertex.pos
        if node_point != start_point and node_point != goal_point:
            new_vertex = NodePRM(node_point)
            graph.append(new_vertex)

    for new_vertex in graph:
        for v in graph:
            if new_vertex == v:
                break
            edge = LineString([new_vertex.pos, v.pos])
            # consider all neighbours that are less than thd away
            if edge.length < thd:
                # check if the edge intersects any obstacle
                connect = True
                for obstacle in C_obs:
                    if edge.intersects(obstacle):
                        connect = False
                        break

                if connect:
                    new_vertex.add_edge(v, edge)
                    v.add_edge(new_vertex, edge)
    return graph


def GetPRMStatistics(prm: List[NodePRM]) -> Tuple[int, int]:
    """Extracting from a PRM graph the total amount of edges,
    and the avarage degree of a node.

    Args:
        prm (List[NodePRM]): A list of the nodes from a PRM graph including the connections between them

    Returns:
        Tuple[int, int]: 
            int - edges - total number of edges in the prm graph
            int - avg_node_degree - avarage degree of the nodes in the prm graph
    """

    degrees = 0
    connected_nodes = 0
    for node in prm:
        degrees += len(node.edges)
        # we do not count for isolated nodes in the prm (ofcourse this is only semantic)
        if len(node.edges) > 0:
            connected_nodes += 1

    edges = degrees/2
    avg_node_degree = degrees/connected_nodes

    return edges, avg_node_degree


def prmFindTopRight(prm: List[NodePRM]) -> NodePRM:
    """Finding the node in the prm graph located closer then all other nodes to the top right corner

    Args:
        prm (List[NodePRM]): A list of the nodes from a PRM graph including the connections between them

    Returns:
        NodePRM: The node in the prm graph located closer then all other nodes to the top right corner
    """
    goal = None
    top_right = Point(map_size, map_size)
    min_dist = np.inf

    for node in prm:
        edge = LineString([node.pos, top_right])
        dist = edge.length
        if dist < min_dist:
            min_dist = dist
            goal = node

    return goal


def prmFindBottomLeft(prm: List[NodePRM]) -> NodePRM:
    """Finding the node in the prm graph located closer then all other nodes to the bottom left corner

    Args:
        prm (List[NodePRM]): A list of the nodes from a PRM graph including the connections between them

    Returns:
        NodePRM: The node in the prm graph located closer then all other nodes to the bottom left corner
    """
    start = None
    bottom_left = Point(0, 0)
    min_dist = np.inf

    for node in prm:
        edge = LineString([bottom_left, node.pos])
        dist = edge.length
        if dist < min_dist:
            min_dist = dist
            start = node

    return start


def prmFindStart(prm: List[NodePRM]) -> NodePRM:
    """Extracting the node in the prm graph that was added as the initial location

    Args:
        prm (List[NodePRM]): A list of the nodes from a PRM graph including the connections between them

    Returns:
        NodePRM: The node in the PRM defined as the initial location
    """
    start = prm[0]
    return start


def prmFindGoal(prm: List[NodePRM]) -> NodePRM:
    """Extracting the node in the prm graph that was added as the goal location

    Args:
        prm (List[NodePRM]): A list of the nodes from a PRM graph including the connections between them

    Returns:
        NodePRM: The node in the PRM defined as the goal location
    """
    goal = prm[1]
    return goal


# def newPrmSolWithoutOneEdge(prm, node1, node2):

#     prm_new = prm
#     value_corrected = False

#     for i, prm_node in enumerate(prm_new):
#         if prm_node == node1:
#             for j, prm_edge in enumerate(prm_node.edges):
#                 if prm_edge[0] == node2:

#                     prm_node.edges.pop(j)
#                     prm_node.edges_lines.pop(j)
#                     prm_new[i] = prm_node
#                     value_corrected = True
#                     break

#         if value_corrected == True:
#             break

#     value_corrected = False

#     for i, prm_node in enumerate(prm_new):
#         if prm_node == node2:
#             for j, prm_edge in enumerate(prm_node.edges):
#                 if prm_edge[0] == node1:

#                     prm_node.edges.pop(j)
#                     prm_node.edges_lines.pop(j)
#                     value_corrected = True
#                     break

#         if value_corrected == True:
#             break

#     start = prmFindBottomLeft(prm_new)
#     goal = prmFindTopRight(prm_new)
#     AStar = AStarPlanner(prm_new, start, goal)
#     new_sol, new_cost = AStar.Plan()
#     return new_sol, new_cost
