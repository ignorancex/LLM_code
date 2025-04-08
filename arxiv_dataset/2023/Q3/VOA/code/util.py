import errno
import string
import numpy as np
from shapely.geometry import Polygon
from scipy.stats import norm
from prm import*
from typing import Tuple
from aStar import*
import os
from datetime import datetime
import math


def probNormalDistConfined(mean: float, cov: float, val1: float, val2: float) -> float:
    """Probability in a Gaussian distribution being between val1 and val2

    Args:
        mean (float): Mean of Gaussian distribution
        cov (float): Covariance of Gaussian Distribution
        val1 (float): First bound of value
        val2 (float): Second bound for value

    Returns:
        float: Probability in a Gaussian distribution being between val1 and val2
    """

    # Probability of value in normal distribution being between val1 and val2
    prob1 = norm(loc=mean, scale=cov).cdf(val1)
    prob2 = norm(loc=mean, scale=cov).cdf(val2)
    prob = (np.abs(prob1 - prob2))[0][0]

    return prob


def probNormalInSquare(mu: np.matrix, Sigma: List[float], x_left: float, x_right: float, y_left: float, y_right: float) -> float:
    """Calculate the probability of being inside a given square acording to a Gaussian distribution

    Args:
        mu (List[float]): 2D mean values in a Gaussian distribution
        Sigma (List[float]): 2D covariance values in a Gaussian distribution
        x_left (float): Left edge location on a square acording to x axis
        x_right (float): Right edge location on a square acording to x axis
        y_left (float): Left edge location on a square acording to y axis
        y_right (float): Right edge location on a square acording to y axis

    Returns:
        float: Probability of being inside a square acording to a Gaussian distribution
    """
    prob_x = probNormalDistConfined(mu[0], Sigma[0, 0], x_left, x_right)
    prob_y = probNormalDistConfined(mu[1], Sigma[1, 1], y_left, y_right)
    prob = prob_x * prob_y
    return prob


def euclideanDist(pos1: List[float], pos2: List[float]) -> float:
    """2 dimentional euclidian distance

    Args:
        pos1 (List[float]): Fisrt position
        pos2 (List[float]): Second position

    Returns:
        float: Euclidian distance
    """
    # calculate 2 dimentional euclidian distance
    return np.sqrt(((pos1[0] - pos2[0]) ** 2) +
                   ((pos1[1] - pos2[1]) ** 2))


def multiplyValuesInList(value_list: List[float]) -> float:
    """Multiply all value in a list of floats by each other

    Args:
        value_list (List[float]): A list of values

    Returns:
        float: The multiplication of all values on the list
    """
    tot_val = value_list[0]
    for i in range(len(value_list) - 1):
        tot_val = tot_val*value_list[i+1]
    return tot_val


def createScenario(C_obs: List[Polygon] = None, init_location: List[float] = None, goal_location: List[float] = None, map_size: float = 100) -> Tuple[List[NodePRM], float, List[Polygon], NodePRM, bool, List[NodePRM]]:
    """Creates a map using obstacles and an initial and goal location,
    then uses PRM to cennect random points on the map
    and then uses A* to find the shortest path from start to goal

    Args:
        C_obs (List[Polygon], optional): A list of all obstacles in the environment. Defaults to None.
        init_location (List[float], optional): Initial location of the agent. Defaults to None.
        goal_location (List[float], optional): Goal location for the agent. Defaults to None.

    Returns:
        Tuple[List[NodePRM], float, List[Polygon], NodePRM, bool, List[NodePRM]]:
            List[NodePRM] - sol - PRM nodes to follow to get from the initial to the goal location.
            float - cost - Length of the planned route.
            List[Polygon] - C_obs - A list of all obstacles in the environment. Defaults to None.
            NodePRM - start - PRMnode representing the initial location.
            bool - connected - Whether the initial and goal locations are connected in the PRM graph.
            List[NodePRM] - prm - The created PRM graph.
    """
    prm = GeneratePRM(C_obs, map_size, init_location, goal_location)

    start = prmFindStart(prm)
    goal = prmFindGoal(prm)

    # Solve Using AStar
    AStar = AStarPlanner(prm, start, goal)
    sol, cost = AStar.Plan()

    if start not in sol or goal not in sol:
        connected = False
    else:
        connected = True

    return sol, start, connected


def ReCreateScenario(prm: List[NodePRM], C_obs: List[Polygon], init_location=None, goal_location=None) -> Tuple[List[NodePRM], float, List[Polygon], NodePRM, bool, List[NodePRM]]:
    """Creates a map using obstacles and an initial and goal location,
    and an already established PRM graph to which th initial and goal location are added
    and then uses A* to find the shortest path from start to goal

    Args:
        prm (List[NodePRM]): The given PRM graph.
        C_obs (List[Polygon], optional): A list of all obstacles in the environment. Defaults to None.
        init_location (List[float], optional): Initial location of the agent. Defaults to None.
        goal_location (List[float], optional): Goal location for the agent. Defaults to None.

    Returns:
        Tuple[List[NodePRM], float, List[Polygon], NodePRM, bool, List[NodePRM]]: 
            List[NodePRM] - sol - PRM nodes to follow to get from the initial to the goal location.
            float - cost - Length of the planned route.
            List[Polygon] - C_obs - A list of all obstacles in the environment. Defaults to None.
            NodePRM - start - PRMnode representing the initial location.
            bool - connected - Whether the initial and goal locations are connected in the PRM graph.
            List[NodePRM] - new_prm - The newly created PRM graph using the existing prm graph and new initial and goal locations.
    """
    new_prm = ReGeneratePRM(prm, C_obs, init_location, goal_location)

    start = prmFindStart(new_prm)
    goal = prmFindGoal(new_prm)

    # Solve Using AStar
    AStar = AStarPlanner(new_prm, start, goal)
    sol, cost = AStar.Plan()

    if start not in sol or goal not in sol:
        connected = False
    else:
        connected = True

    return sol, cost, C_obs, start, connected, new_prm


def fromMat2LocationVal(X: List[np.matrix], t: int) -> Tuple[List[float], List[float]]:
    """Transfer from a list of matrices (each a length of 2) to 2 lists of values 

    Args:
        X (List[np.matrix]): List of matrices each the size of 2
        t (int): Length of the list of matrices

    Returns:
        Tuple[List[float], List[float]]: 
            List[float] - x_location - A list of all the first values in each matrix 
            List[float] - y_location - A list of all the second values in each matrix 
    """
    x_location = list()
    y_location = list()

    for i in range(t):
        X_t = (X[i]).A
        x_location.extend(X_t[0])
        y_location.extend(X_t[1])

    return x_location, y_location


def fileCreation(name_extra: string):
    mydir = os.path.join(
        os.getcwd(),
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S' + '_machine_' + name_extra))
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..

    return mydir


def createSlidingWindowVec(vec: List[float], size_of_window: int) -> List[float]:
    vec = [0] + vec
    new_vec = []
    for i in range(len(vec) - (size_of_window-1)):
        val = 0
        for j in range(size_of_window):
            val += vec[i+j]
        new_vec.append(val/size_of_window)
    return new_vec


def sameMaxValLocation(vec1: List[float], vec2: List[float]) -> bool:
    max_vec1_index = vec1.index(max(vec1))
    max_vec2_index = vec2.index(max(vec2))
    return (max_vec1_index == max_vec2_index)


def sameMaxValLocationMultiMax(vec_main: List[float], vec_compare: List[float], amount_of_max: int) -> bool:
    for i in range(amount_of_max):
        if sameMaxValLocation(vec_main, vec_compare) == True:
            return True
        vec_compare[vec_compare.index(max(vec_compare))] = - math.inf
    return False


def sameMaxValLocationNearMax(vec_main: List[float], vec_compare: List[float], how_close: int = 2) -> bool:
    index_main = vec_main.index(max(vec_main))
    index_compare = vec_compare.index(max(vec_compare))
    for i in range(how_close):
        if index_main == index_compare - i or index_main == index_compare + i:
            return True
    return False


def confidenceInterval(std: float, confidence_level: float = 0.99, sample_size: float = 5000) -> float:
    added_value = confidence_level*(std/math.sqrt(sample_size))
    return added_value
