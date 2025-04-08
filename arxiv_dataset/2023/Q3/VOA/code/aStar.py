import heapq
from typing import Optional
from shapely.geometry import LineString


class NodeAStar:
    """
    Helper class to hold the nodes of Astar.
    A node keeps track of potentially a parent node.
    A node has its cost (g_score , the cost of the cheapest path from start to it),
    and the f-score (cost + heuristic to goal).
    """

    def __init__(self, prm_node,
                 parent_node: Optional['Node'] = None,
                 g_score: float = 0,
                 f_score: Optional[float] = None):

        self.prm_node = prm_node
        self.parent_node: NodeAStar = parent_node
        self.g_score = g_score
        self.f_score = f_score

        if self.parent_node is not None:
            self.g_score += self.parent_node.g_score

    def __lt__(self, other):
        return self.f_score < other.f_score


class AStarPlanner(object):
    def __init__(self, prm, start, goal):
        self.start = start
        self.goal = goal
        self.prm = prm
        self.nodes = dict()

        self.open = AstarPriorityQueue()
        self.close = AstarPriorityQueue()

    def Plan(self):
        plan = []

        start_node = NodeAStar(self.start, None, 0)
        start_node.f_score = self._calc_node_f_score(start_node)
        self.open.Insert(start_node, start_node.f_score)

        count = 0
        while not self.open.IsEmpty():
            next_node = self.open.Pop()
            self.close.Insert(next_node, 1)  # priority has no meaning in close
            self.nodes[next_node.prm_node] = next_node
            plan.append(next_node.prm_node)
            count += 1

            if next_node.prm_node == self.goal:
                break

            edges = self._get_neighbours(next_node)
            for edge in edges:
                neighbour, cost = edge
                successor_node = NodeAStar(neighbour, next_node, cost)
                successor_node.f_score = self._calc_node_f_score(
                    successor_node)

                new_g = successor_node.g_score
                # the node is already in OPEN
                if self.open.Contains(successor_node.prm_node):
                    already_found_node = self.open.GetByPRMNode(
                        successor_node.prm_node)
                    if new_g < already_found_node.g_score:  # new parent is better
                        already_found_node.g_score = new_g
                        already_found_node.parent_node = successor_node.parent_node
                        already_found_node.f_score = self._calc_node_f_score(
                            already_found_node)

                        # f changed so need to reposition in OPEN
                        self.open.Remove(already_found_node.prm_node)
                        self.open.Insert(already_found_node,
                                         already_found_node.f_score)

                    else:  # old path is better - do nothing
                        pass
                else:  # state not in OPEN maybe in CLOSED
                    # this node exists in CLOSED
                    if self.close.Contains(successor_node.prm_node):
                        already_found_node = self.close.GetByPRMNode(
                            successor_node.prm_node)
                        if new_g < already_found_node.g_score:  # new parent is better
                            already_found_node.g_score = new_g
                            already_found_node.parent_node = successor_node.parent_node
                            already_found_node.f_score = self._calc_node_f_score(
                                already_found_node)

                            # move old node from CLOSED to OPEN
                            self.close.Remove(already_found_node.prm_node)
                            self.nodes.pop(already_found_node.prm_node)
                            self.open.Insert(
                                already_found_node, already_found_node.f_score)
                        else:  # old path is better - do nothing
                            pass
                    else:
                        # this is a new state - create a new node = insert new node to OPEN
                        self.open.Insert(
                            successor_node, successor_node.f_score)

        # print("Astar expanded", count, "nodes")

        return self._backtrace(plan)

    def _backtrace(self, plan):
        """
        backtrace from goal to start
        """
        cost = 0
        current = self.nodes[plan[-1]]
        sol = [current.prm_node]
        while current.prm_node != plan[0]:
            cost += self._get_distance_between_nodes(
                current, current.parent_node)
            current = self.nodes[current.parent_node.prm_node]
            sol.append(current.prm_node)

        return sol, cost

    def _calc_node_f_score(self, node: NodeAStar):
        return node.g_score + self._compute_heuristic(node)

    def _compute_heuristic(self, node: NodeAStar):
        """
        Heuristic is defined as Euclidean distance to goal
        """
        edge = LineString([node.prm_node.pos, self.goal.pos])
        return edge.length

    def _get_neighbours(self, node: NodeAStar):
        """
        Returns the edges in the PRM
        each edge is a tuple of (neighbour, cost)
        """
        return node.prm_node.edges

    def _get_distance_between_nodes(self, node1: NodeAStar, node2: NodeAStar):
        """
        Returns Euclidean distance between two nodes in the graph
        """
        edge = LineString([node1.prm_node.pos, node2.prm_node.pos])
        return edge.length


class AstarPriorityQueue:
    def __init__(self):
        """Creates an empty priority queue for A*
        """
        self.elements = []
        # just for performance (could probably used ordered_dict..)
        self.elements_dict = {}

    def IsEmpty(self) -> bool:
        """Check if there are any elements in the priopity queue

        Returns:
            bool: are there any elements in the priority queue
        """
        return len(self.elements) == 0

    def Insert(self, item, priority) -> None:
        """Insert a new item into the priority queue

        Args:
            item (Any): Item to insert
            priority (Any): The priority of the item 
        """
        heapq.heappush(self.elements, (priority, item))
        self.elements_dict[item.prm_node] = item

    def Pop(self):
        item = heapq.heappop(self.elements)
        self.elements_dict.pop(item[1].prm_node)
        return item[1]

    def TopKey(self):
        return heapq.nsmallest(1, self.elements)[0][0]

    def Remove(self, prm_node):
        self.elements = [e for e in self.elements if e[1].prm_node != prm_node]
        heapq.heapify(self.elements)

        self.elements_dict.pop(prm_node)

    def Contains(self, prm_node):
        return prm_node in self.elements_dict

    def GetByPRMNode(self, prm_node):
        return self.elements_dict[prm_node]
