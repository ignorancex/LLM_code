import random
from shapely.geometry import Polygon
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon
from typing import Tuple


## Creating obstacles for maps ##

def obstaclesForMapRandom(map_size: float) -> List[Polygon]:
    """Random n_obs obstacles all with width=n_x_obs, height=n_y_obs that fit in the environment with size NxN

    Args:
        N (float): Environment size NxN

    Returns:
        List[Polygon]: List containing all generated obstacles
    """

    n_obs = map_size/5
    n_x_obs = map_size/20
    n_y_obs = map_size/20
    C_obs = []

    while len(C_obs) < n_obs:
        bottom_left_x = np.random.uniform(0, map_size-n_x_obs)
        bottom_left_y = np.random.uniform(0, map_size-n_y_obs)
        new_obstacle = Polygon([(bottom_left_x, bottom_left_y),
                                (bottom_left_x, bottom_left_y+n_y_obs),
                                (bottom_left_x+n_x_obs, bottom_left_y+n_y_obs),
                                (bottom_left_x+n_x_obs, bottom_left_y)])

        add = True
        for obs in C_obs:
            if new_obstacle.intersects(obs):
                add = False
                break
        if add:
            C_obs.append(new_obstacle)

    return C_obs


def obstaclesForMap1(map_size: float) -> List[Polygon]:
    """A predetermined set of obstacles for that fit in the environment with size NxN
    1 rectangular obstacle in the bottom middle of the map

    Args:
        N (float): Environment size NxN

    Returns:
        List[Polygon]: List containing all generated obstacles
    """
    C_obs = []

    new_obstacle = Polygon([(map_size/3, 0),
                            (map_size*2/3, 0),
                            (map_size*2/3, map_size*2/3),
                            (map_size/3, map_size*2/3)])

    C_obs.append(new_obstacle)

    return C_obs


def obstaclesForMap2(map_size: float) -> List[Polygon]:
    """A predetermined set of obstacles for that fit in the environment with size NxN
    multiple rows of square obstacles with interchanging location between consecutive rows

    Args:
        N (float): Environment size NxN

    Returns:
        List[Polygon]: List containing all generated obstacles
    """
    C_obs = []
    width = map_size*2/24
    x_row1 = map_size*5/24
    x_row2 = map_size*9/24
    x_row3 = map_size*13/24
    x_row4 = map_size*17/24
    y_row_with_3_obs = [map_size*3/24, map_size*3 /
                        24 + map_size/3, map_size*3/24 + map_size*2/3]
    y_row_with_2_obs = [map_size/3 - map_size/24, map_size*2/3 - map_size/24]

    bot_left_x_vec_row1 = [x_row1, x_row1, x_row1]
    bot_left_y_vec_row1 = y_row_with_3_obs
    bot_left_x_vec_row2 = [x_row2, x_row2]
    bot_left_y_vec_row2 = y_row_with_2_obs
    bot_left_x_vec_row3 = [x_row3, x_row3, x_row3]
    bot_left_y_vec_row3 = bot_left_y_vec_row1
    bot_left_x_vec_row4 = [x_row4, x_row4]
    bot_left_y_vec_row4 = bot_left_y_vec_row2

    bot_left_x_vec = bot_left_x_vec_row1 + bot_left_x_vec_row2 + \
        bot_left_x_vec_row3 + bot_left_x_vec_row4
    bot_left_y_vec = bot_left_y_vec_row1 + bot_left_y_vec_row2 + \
        bot_left_y_vec_row3 + bot_left_y_vec_row4

    for i in range(len(bot_left_x_vec)):
        new_obstacle = Polygon([(bot_left_x_vec[i],
                                 bot_left_y_vec[i]),
                                (bot_left_x_vec[i],
                                 bot_left_y_vec[i]+width),
                                (bot_left_x_vec[i]+width,
                                 bot_left_y_vec[i]+width),
                                (bot_left_x_vec[i]+width,
                                 bot_left_y_vec[i])])

        C_obs.append(new_obstacle)

    return C_obs


def obstaclesForMap3(map_size: float) -> List[Polygon]:
    """A predetermined set of obstacles for that fit in the environment with size NxN
    4 rectangular obstacle in a line in the middle of the map
    with 3 corridors from the bottom part of the map to the top part

    Args:
        N (float): Environment size NxN

    Returns:
        List[Polygon]: List containing all generated obstacles
    """
    y_bot = map_size/3
    y_top = map_size*2/3
    obs_width = map_size/4
    gap_width = (map_size - 3 * obs_width) / 3

    obs1 = Polygon([(0, y_bot),
                    (obs_width/2, y_bot),
                    (obs_width/2, y_top),
                    (0, y_top)])

    obs2 = Polygon([(obs_width/2 + gap_width, y_bot),
                    (obs_width*3/2 + gap_width, y_bot),
                    (obs_width*3/2 + gap_width, y_top),
                    (obs_width/2 + gap_width, y_top)])

    obs3 = Polygon([(obs_width*3/2 + gap_width*2, y_bot),
                    (obs_width*5/2 + gap_width*2, y_bot),
                    (obs_width*5/2 + gap_width*2, y_top),
                    (obs_width*3/2 + gap_width*2, y_top)])

    obs4 = Polygon([(obs_width*5/2 + gap_width*3, y_bot),
                    (obs_width*6/2 + gap_width*3, y_bot),
                    (obs_width*6/2 + gap_width*3, y_top),
                    (obs_width*5/2 + gap_width*3, y_top)])

    C_obs = [obs1, obs2, obs3, obs4]

    return C_obs


def obstaclesForMap4(map_size: float) -> List[Polygon]:
    """A predetermined set of obstacles for that fit in the environment with size NxN
    3 rectangular obstacle in a line in the middle of the map
    with 3 corridors from the left part of the map to the right part with different widthes part

    Args:
        N (float): Environment size NxN

    Returns:
        List[Polygon]: List containing all generated obstacles
    """
    x_left = map_size/3
    x_right = map_size*2/3
    obs_width = map_size/6
    gap_width = (map_size - 4 * obs_width) / 3

    obs1 = Polygon([(x_left, map_size),
                    (x_left, map_size - obs_width),
                    (x_right, map_size - obs_width),
                    (x_right, map_size)])

    obs2 = Polygon([(x_left, map_size - obs_width - gap_width),
                    (x_left, map_size - 2*obs_width - gap_width),
                    (x_right, map_size - 2*obs_width - gap_width),
                    (x_right, map_size - obs_width - gap_width)])

    obs3 = Polygon([(x_left, map_size - 2*obs_width - 2*gap_width),
                    (x_left, map_size - 3*obs_width - 2*gap_width),
                    (x_right, map_size - 3*obs_width - 2*gap_width),
                    (x_right, map_size - 2*obs_width - 2*gap_width)])

    C_obs = [obs1, obs2, obs3]

    return C_obs


def obstaclesForMap6(map_size: float) -> List[Polygon]:

    C_obs = []

    new_obstacle1 = Polygon([(map_size*4/10, 0),
                            (map_size*6/10, 0),
                            (map_size*6/10, map_size*45/100),
                            (map_size*4/10, map_size*45/100)])

    new_obstacle2 = Polygon([(map_size*4/10, map_size),
                            (map_size*6/10, map_size),
                            (map_size*6/10, map_size*55/100),
                            (map_size*4/10, map_size*55/100)])

    C_obs.append(new_obstacle1)
    C_obs.append(new_obstacle2)

    return C_obs


def obstaclesForPrevMap(map_size: float, n_x_obs: float = 2, n_y_obs: float = 2) -> List[Polygon]:
    """A predetermined set of obstacles all with width=n_x_obs, height=n_y_obs that fit in the environment with size NxN

    Args:
        n_obs (int): Number of obstacles
        n_x_obs (float): Width of each obstacle
        n_y_obs (float): Hight of each obstacle
        N (float): Environment size NxN

    Returns:
        List[Polygon]: List containing all generated obstacles
    """
    C_obs = []

    bottom_left_x_vec = [
        map_size/2 - 2*(n_x_obs),
        map_size/2 + 1*(n_x_obs),
        map_size/2 - 0.5*(n_x_obs),
        map_size/2 - 2*(n_x_obs),
        map_size/2 + 1*(n_x_obs),
        map_size/2 - 0.5*(n_x_obs),
        map_size/2 - 3*(n_x_obs),
        map_size/2 + 2*(n_x_obs), ]
    bottom_left_y_vec = [
        0,
        0,
        map_size*1/4 - 0.5*(n_y_obs),
        map_size*2/4 - 0.5*(n_y_obs),
        map_size*2/4 - 0.5*(n_y_obs),
        map_size*3/4 - 0.5*(n_y_obs),
        map_size*3/4 - 0.5*(n_y_obs),
        map_size*3/4 - 0.5*(n_y_obs)]

    for i in range(len(bottom_left_x_vec)):
        new_obstacle = Polygon([(bottom_left_x_vec[i], bottom_left_y_vec[i]),
                                (bottom_left_x_vec[i],
                                 bottom_left_y_vec[i]+n_y_obs),
                                (bottom_left_x_vec[i]+n_x_obs,
                                 bottom_left_y_vec[i]+n_y_obs),
                                (bottom_left_x_vec[i]+n_x_obs, bottom_left_y_vec[i])])

        C_obs.append(new_obstacle)

    return C_obs


## Added functions for obstacles ##

def obstaclesPlotter(obstacles: List[Polygon]) -> None:
    """Plots obstacles

    Args:
        C_obs (List[Polygon]): A list of all obstacles in the environment
    """
    # plot obstacles
    for i in range(len(obstacles)):
        x, y = obstacles[i].exterior.xy
        plt.fill(x, y, 'k')


def initMaze(obstacles: List[Polygon]) -> MultiPolygon:
    """All obstacles (each type Polygon) from list combined into one variable 

    Args:
        C_obs (List[Polygon]): A list of all obstacles in the environment

    Returns:
        MultiPolygon: "maze" - all obstacles in the envionment in a single variable
    """
    maze = obstacles[0]
    for i in range(len(obstacles)):
        maze = maze.union(obstacles[i])

    return maze


def obstacleMapByNumber(map_size: float, map_num: int) -> Tuple[List[Polygon], List[float], List[float]]:
    if map_num == 1:
        C_obs = obstaclesForMap1(map_size)
        init_location = [map_size/6, map_size/6]
        goal_location = [map_size*5/6, map_size/6]

    elif map_num == 2:
        C_obs = obstaclesForMap2(map_size)
        init_location = [map_size/8, map_size/2]
        goal_location = [map_size*7/8, map_size/2]

    elif map_num == 3:
        C_obs = obstaclesForMap3(map_size)
        init_location = [map_size/6, map_size*5/6]
        goal_location = [map_size*5/6, map_size/6]

    elif map_num == 4:
        C_obs = obstaclesForMap4(100)
        init_location = [map_size/6, map_size/2]
        goal_location = [map_size*5/6, map_size/2]

    elif map_num == 6:
        C_obs = obstaclesForMap6(map_size)
        init_location = [map_size*3/10, map_size/2]
        goal_location = [map_size*6/10, map_size/2]

    elif map_num == -1:
        C_obs = []
        init_location = [map_size*(random.randrange(1, 3))/10,
                         map_size*(random.randrange(1, 3))/10]
        goal_location = [map_size*(random.randrange(6, 8))/10,
                         map_size*(random.randrange(6, 8))/10]

    elif map_num == 7:
        C_obs = []
        init_location = [map_size*2/10,
                         map_size*2/10]
        goal_location = [map_size*7/10,
                         map_size*7/10]

    return C_obs, init_location, goal_location


def obstacleMapPlotter(map_size: float, map_num: int):
    C_obs, init_location, goal_location = obstacleMapByNumber(
        map_size, map_num)
    obstaclesPlotter(C_obs)
    plt.plot(init_location[0], init_location[1], marker="o", markersize=20,
             markerfacecolor="green")
    plt.plot(goal_location[0], goal_location[1], marker="o", markersize=20,
             markerfacecolor="red")
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.show()
