import random
from array import array
import string
from scipy.ndimage import gaussian_filter
import math
import numpy as np
import matplotlib.pyplot as plt

M = 10e20  # Size multiplier for Gaussian Blur


## Real World Experiments maps ##

def randomArrayMapRealWorld(grid_size: int) -> array:
    x_grid = grid_size
    y_grid = round(2.5*grid_size)
    map_arr = [[0 for i in range(x_grid)] for j in range(y_grid)]

    for rand in range(grid_size):
        i = random.randint(0, x_grid - 1)
        j = random.randint(0, y_grid - 1)
        map_arr[j][i] += M

    return map_arr


def arrayMapRealWorld(grid_size: int) -> array:
    """Accurding to the experiments done in the lab in summer 2022,
    genorate a map representing the the real world experiments in the form of an array.

    Returns:
        array: a binary array with 1 representing
    """
    x_grid = grid_size
    y_grid = round(2.5*grid_size)
    map_arr = [[0 for i in range(x_grid)] for j in range(y_grid)]

    for j in range(round(y_grid * 1/5), round(y_grid * 1.5/5)):
        for i in range(0, round(x_grid * 1/5)):
            map_arr[j][i] = M

    for j in range(round(y_grid * 1/5), round(y_grid)):
        for i in range(0, round(x_grid * 1/5)):
            map_arr[j][i] = M

    for j in range(round(y_grid * 35/100), round(y_grid * 45/100)):
        for i in range(0, round(x_grid * 3/5)):
            map_arr[j][i] = M

    for j in range(round(y_grid * 3/5), round(y_grid * 70/100)):
        for i in range(round(x_grid * 2/5), x_grid):
            map_arr[j][i] = M

    for j in range(round(y_grid * 88/100), y_grid):
        for i in range(0, round(x_grid * 3/5)):
            map_arr[j][i] = M

    return map_arr


## Simulation Experiments maps ##

def arrayMapRandom(grid_size: int) -> array:
    rows, cols = (grid_size, grid_size)
    map_arr = [[0 for i in range(cols)] for j in range(rows)]

    for rand in range(round(grid_size/4)):
        i = random.randint(0, grid_size - 1)
        j = random.randint(0, grid_size - 1)
        map_arr[i][j] += M

    return map_arr


def arrayMap1(grid_size: int) -> array:
    """Genorate an array of float values representing a predetermined set of obstacles
     that fit in the environment with size NxN.
    1 rectangular obstacle in the bottom middle of the map.

    Returns:
        array: an array of floats representing the environment

    """
    rows, cols = (grid_size, grid_size)
    map_arr = [[0 for i in range(cols)] for j in range(rows)]

    for j in range(round(grid_size/3), round(grid_size*2/3)):
        for i in range(0, round(grid_size*2/3)):
            map_arr[j][i] = M

    return map_arr


def arrayMap2(grid_size: int) -> array:
    """Genorate an array of float values representing a predetermined set of obstacles
     that fit in the environment with size NxN.
    muliple rows of square obstacles with interchanging location between consecutove rows

    Returns:
        array: an array of floats representing the environment
    """

    rows, cols = (grid_size, grid_size)
    map_arr = [[0 for i in range(cols)] for j in range(rows)]

    width = round(grid_size*2/24)
    x_row1 = round(grid_size*5/24)
    x_row2 = round(grid_size*9/24)
    x_row3 = round(grid_size*13/24)
    x_row4 = round(grid_size*17/24)
    y_row_with_3_obs = [round(grid_size*3/24),
                        round(grid_size*3/24 + grid_size/3),
                        round(grid_size*3/24 + grid_size*2/3)]
    y_row_with_2_obs = [round(grid_size/3 - grid_size/24),
                        round(grid_size*2/3 - grid_size/24)]

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

    for k in range(len(bot_left_x_vec)):
        bot_left_x = bot_left_x_vec[k]
        bot_left_y = bot_left_y_vec[k]
        for j in range(width):
            for i in range(width):
                map_arr[j + bot_left_x][i + bot_left_y] = M

    return map_arr


def arrayMap3(grid_size: int) -> array:
    rows, cols = (grid_size, grid_size)
    map_arr = [[0 for i in range(cols)] for j in range(rows)]

    for i in range(round(grid_size/3), round(grid_size*2/3)):

        for j in range(0, round(grid_size/10) + 1):
            map_arr[j][i] = M

        for j in range(round(grid_size/5), round(grid_size*9/20)):
            map_arr[j][i] = M

        for j in range(round(grid_size*11/20) - 1, round(8/10) - 1):
            map_arr[j][i] = M

        for j in range(round(grid_size*9/10) - 1, round(grid_size) - 1):
            map_arr[j][i] = M

    return map_arr


def arrayMap6(grid_size: int) -> array:

    rows, cols = (grid_size, grid_size)
    map_arr = [[0 for i in range(cols)] for j in range(rows)]

    for j in range(round(grid_size*4/10), round(grid_size*5/10)):
        for i in range(0, round(grid_size*45/100)):
            map_arr[j][i] = M
        for i in range(round(grid_size*55/100), grid_size):
            map_arr[j][i] = M
    return map_arr


def arrayMap7(grid_size: int) -> array:

    rows, cols = (grid_size, grid_size)
    map_arr = [[0 for i in range(cols)] for j in range(rows)]

    for j in range(round(grid_size*2/10), round(grid_size*7/10)):
        for i in range(round(grid_size*2/10), round(grid_size*7/10)):
            if i == j:
                map_arr[j][i] = M

    return map_arr


## Added map functions ##

def filteredAndNormalizedArrayMap(map_arr: array, map_size: float, grid_size: int) -> array:
    """using an existing array containing float values,
    create a new array using Gaussian blur over the given values,
     then it is normalized to a maximal value of 1.
    There is a Sigma for that blur inside the function

    Args:
        map_arr (array): an array of float values

    Returns:
        array: An array of float values after going throgh Gaussian blur
    """

    proportion = map_size/grid_size
    sigma_val = math.sqrt(6/proportion)

    map_arr = map_arr
    map_arr = gaussian_filter(map_arr, sigma_val, truncate=4.0)
    map_arr = map_arr / map_arr.max()

    return map_arr


def arrayMapByNumber(grid_size: int, map_size: float, map_num: int = -1) -> array:
    if map_num == 1:
        map_arr_reg = arrayMap1(grid_size)

    elif map_num == 2:
        map_arr_reg = arrayMap2(grid_size)

    elif map_num == 3:
        map_arr_reg = arrayMap3(grid_size)

    elif map_num == 6:
        map_arr_reg = arrayMap6(grid_size)

    elif map_num == -1:
        map_arr_reg = arrayMapRandom(grid_size)

    elif map_num == 7:
        map_arr1_reg = arrayMapRandom(grid_size)
        map_arr1 = filteredAndNormalizedArrayMap(
            map_arr1_reg, map_size, grid_size)

        map_arr2_reg = arrayMap7(grid_size)
        map_arr2 = filteredAndNormalizedArrayMap(
            map_arr2_reg, map_size, grid_size)

        map_arr = np.subtract(map_arr1, map_arr2)
        return map_arr

    return filteredAndNormalizedArrayMap(
        map_arr_reg, map_size, grid_size)


def pltSavedMap(txt_location: string, txt_add: string, map_num: int):
    if map_num == 1:
        map_arr = arrayMap1()
    elif map_num == 2:
        map_arr = arrayMap2()
    elif map_num == 3:
        map_arr = arrayMap3()
    map_arr = filteredAndNormalizedArrayMap(map_arr)

    planned_path_array = np.loadtxt(
        txt_location + "\planned_path_array" + txt_add)
    full_path_array = np.add(np.multiply(map_arr, 1000), planned_path_array)

    plt.matshow(full_path_array)
    plt.show()
