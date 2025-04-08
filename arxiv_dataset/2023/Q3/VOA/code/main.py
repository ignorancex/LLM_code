from experiments_data import*
from array import array
from kalmanFilter import belief
from prm import*
from util import*
from tests import*
from points_csv import*
from typing import List
from maps import*
from obstacles import*

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import math
from math import floor
import os
import statistics

# note: @ is a symbol for multiplying matrices
M = 10e20  # Size multiplier for Gaussian Blur


def plotPlannedPath(planned_path, map_array: array, grid_size: int):
    path_array = [[0 for i in range(grid_size)] for j in range(grid_size)]
    for loc in planned_path:
        x_loc = min(floor((loc.A)[0][0]), grid_size - 1)
        y_loc = min(floor((loc.A)[1][0]), grid_size - 1)
        path_array[x_loc][y_loc] = 1

    full_path_array = np.add(map_array, path_array)
    plt.matshow(full_path_array)

## VOA ##


def estimatedSigmaAfterHelp(Sigma: np.matrix, mu: np.matrix, map_arr: array, map_size: float, grid_size: int, voa_Sigma: List[float]):
    proportion = map_size/grid_size
    bel = belief(mu, Sigma)
    prob_init_arr = extractGaussianMatFromBelief(bel, map_size, grid_size)

    p_correct_cost_tot = 0
    Sigma_final_tot = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if prob_init_arr[j][i] != 0:

                init_x = j * proportion
                init_y = i * proportion
                init = NodePRM(Point(init_x, init_y))
                goal = NodePRM(Point(((mu.A)[0]).max(), ((mu.A)[1]).max()))
                sol = [goal, init]

                Belief = VOAInPlanning(
                    sol, init, -1, voa_Sigma)[0]
                p_correct_cost = VOAGrade(
                    Belief, map_arr, map_size, grid_size)

                Sigma_final = ((Belief[-1]).Sigma.A)[0][0]
                Sigma_final_tot = Sigma_final_tot + \
                    Sigma_final * prob_init_arr[j][i]
                p_correct_cost_tot = p_correct_cost_tot + \
                    p_correct_cost * prob_init_arr[j][i]

    Sigma_return = np.matrix([[Sigma_final_tot, 0], [0, Sigma_final_tot]])

    return Sigma_return, p_correct_cost_tot


def extractGaussianMatFromBelief(bel: belief, map_size: float, grid_size: int) -> array:
    proportion = map_size/grid_size
    mu_val = (bel.mu).A
    mu_x = min(floor(mu_val[0][0]/proportion), grid_size - 1)
    mu_y = min(floor(mu_val[1][0]/proportion), grid_size - 1)
    bel_sigma = math.sqrt((((bel.Sigma).A)[0][0])/proportion)

    rows, cols = (grid_size, grid_size)
    arr = [[0 for i in range(cols)] for j in range(rows)]
    arr[mu_x][mu_y] = M

    arr = (gaussian_filter(arr, sigma=bel_sigma))/M
    return arr


def extractGaussianMatFromBeliefExperiment(x_mean: float, y_mean: float, x_var: float, y_var: float) -> array:
    arr = [[0 for i in range(40)] for j in range(100)]
    arr[round(x_mean)][round(y_mean)] = M
    arr = (gaussian_filter(arr, sigma=[x_var, y_var]))/M
    return arr


def VOAGradeRealWorldExperiments(x_mean: List[float], y_mean: List[float], x_var: List[float], y_var: List[float], map_arr: array) -> float:
    sum = 0
    for i in range(len(x_mean)):
        bel_arr = extractGaussianMatFromBeliefExperiment(
            x_mean[i], y_mean[i], x_var[i], y_var[i])

        for i in range(100):
            for j in range(40):
                sum = sum + ((bel_arr[i][j])*(map_arr[i][j]))

    return sum


def VOAGrade(Beliefs: List[belief], map_arr: array, map_size: float, grid_size: int) -> float:
    sum = 0

    for bel in Beliefs:
        bel_arr = extractGaussianMatFromBelief(bel, map_size, grid_size)
        sum += np.sum(np.multiply(bel_arr, map_arr))

    return sum


def VOAForGivenTimeStep(help_step: int, start: List[float], sol, map_arr: array, p_collide_score: float, map_size: float, grid_size: int, voa_Sigma: List[float]) -> float:
    Belief_before_help, mu_final, Sigma_final, route_point = VOAInPlanning(
        sol, start, help_step, voa_Sigma)

    relocation = True
    localization = False

    if relocation == True:
        if help_step >= 0:
            new_init_location = [mu_final[0][0], mu_final[1][0]]
            new_start = NodePRM(
                Point(new_init_location[0], new_init_location[1]))
            sol_new = [sol[0], new_start]
            new_Sigma = np.matrix('0 0 ; 0 0')
            Belief_after_help = VOAInPlanning(
                sol_new, new_start, -1, voa_Sigma, new_Sigma, route_point)[0]

        Belief_help = Belief_before_help + Belief_after_help
        p_collide_score_help = VOAGrade(
            Belief_help, map_arr, map_size, grid_size)

        return p_collide_score - (p_collide_score_help)

    if localization == True:
        if help_step >= 0:
            new_Sigma, p_collide_correction = estimatedSigmaAfterHelp(
                Sigma_final, mu_final, map_arr, map_size, grid_size, voa_Sigma)
            new_init_location = [mu_final[0][0], mu_final[1][0]]
            new_start = NodePRM(
                Point(new_init_location[0], new_init_location[1]))
            sol_new = [sol[0], new_start]
            Belief_after_help = VOAInPlanning(
                sol_new, new_start, -1, voa_Sigma, new_Sigma, route_point)[0]

        Belief_help = Belief_before_help + Belief_after_help
        p_collide_score_help = VOAGrade(
            Belief_help, map_arr, map_size, grid_size)

        return p_collide_score - (p_collide_score_help + p_collide_correction)


def VOAExperiments(map_arr: array, sol, start, map_size: float, grid_size: int, voa_Sigma: List[float]):
    Belief = VOAInPlanning(sol, start, -1, voa_Sigma)[0]
    p_collide_score = VOAGrade(Belief, map_arr, map_size, grid_size)
    if p_collide_score == 0:
        print(-1)

    planned_path = []
    for bel in Belief:
        planned_path.append(bel.mu)

    step_size = round(len(planned_path)/22)
    help_steps = []
    for i in range(15):
        help_step = (i + 3) * step_size
        help_steps.append(help_step)

    VOA_arr = []
    for i, help_step in enumerate(help_steps):

        VOAForOneTimeStep = VOAForGivenTimeStep(
            help_step, start, sol, map_arr, p_collide_score, map_size, grid_size, voa_Sigma)
        VOA_arr.append(VOAForOneTimeStep)
        print("Done: " + str((i+1)/(len(help_steps))))

    return help_steps, VOA_arr, planned_path, help_steps


## AECD ##


def AECDGradeExperiments(X_loc: List[float], Y_loc: List[float], map_arr: array) -> float:
    sum = 0
    max_x_size = 100
    max_y_size = 40

    for i in range(len(X_loc)):
        x_loc = min(floor(X_loc[i]), max_x_size - 1)
        y_loc = min(floor(Y_loc[i]), max_y_size - 1)
        sum = sum + map_arr[x_loc][y_loc]

    return sum


def AECDGrade(path, map_arr: array, map_size: float, grid_size: int) -> float:
    sum = 0
    proportion = map_size/grid_size

    for loc in path:
        x_loc = min(floor((loc.A)[0][0]/proportion), grid_size - 1)
        y_loc = min(floor((loc.A)[1][0]/proportion), grid_size - 1)
        sum = sum + map_arr[x_loc][y_loc]

    return sum


def AECDExperiments(map_arr: array, sol, start, num_of_tests: int, help_steps, voi_sigma, map_size: float, grid_size: int):
    VOI_mu = []
    VOI_std_plus = []
    VOI_std_min = []
    VOI_no_cor_sum_vec = []

    for i, help_step in enumerate(help_steps):
        collide_data = []
        sum = 0
        for j in range(num_of_tests):
            path_with_cor, path_no_cor = calcVOIScenarioT(
                sol, start, voi_sigma, help_step)
            VOI_no_cor = AECDGrade(path_no_cor, map_arr,
                                   map_size, grid_size)
            VOI_with_cor = AECDGrade(
                path_with_cor, map_arr, map_size, grid_size)
            p_collide_VOI = VOI_with_cor
            VOI_no_cor_sum_vec.append(VOI_no_cor)
            sum = sum + p_collide_VOI
            collide_data.append(p_collide_VOI)

        mu, std = norm.fit(collide_data)
        VOI_mu.append(mu)
        VOI_std_plus.append(mu + std)
        VOI_std_min.append(mu - std)

        print("Done: " + str((i+1)/(len(help_steps))))

    VOI_mu = statistics.mean(VOI_no_cor_sum_vec) - VOI_mu

    return VOI_mu, VOI_std_plus, VOI_std_min


## Main functions ##

def RealWorldExperiments(grid_size: int, map_size: float, random_map=False):
    x_mean, y_mean, x_var, y_var = VOADataExtractHelpAtI(0)

    # Map
    if random_map == True:
        map_arr = randomArrayMapRealWorld(grid_size)
    elif random_map == False:
        map_arr_with_path = arrayMapRealWorld(grid_size)
        map_arr_with_path = filteredAndNormalizedArrayMap(
            map_arr_with_path, map_size, grid_size)
        for i in range(len(x_mean)):
            x_val = round(x_mean[i])
            y_val = round(y_mean[i])
            map_arr_with_path[x_val][y_val] += 0.5
        map_arr = arrayMapRealWorld(grid_size)

    map_arr = filteredAndNormalizedArrayMap(map_arr, map_size, grid_size)

    # VOA
    e_val_no_help = VOAGradeRealWorldExperiments(
        x_mean, y_mean, x_var, y_var, map_arr)

    mean_plus_sigma = []
    mean_minus_sigma = []
    for i in range(len(y_mean)):
        mean_plus_sigma.append(y_mean[i] + 3*y_var[i])
        mean_minus_sigma.append(y_mean[i] - 3*y_var[i])

    VOAExperimentsResults = []
    for i in range(4):
        x_mean, y_mean, x_var, y_var = VOADataExtractHelpAtI(i+1)
        e_val_with_help = VOAGradeRealWorldExperiments(
            x_mean, y_mean, x_var, y_var, map_arr)
        VOA_val = e_val_no_help - e_val_with_help
        VOAExperimentsResults.append(VOA_val)

    # Validation
    mean_val_vec = [0, 0, 0, 0]
    std_val_vec = [0, 0, 0, 0]
    folder_location = os.getcwd() + '/real_world_experiments/teleportation/csv'
    for folder_name in os.listdir(folder_location):
        paths = folder2Paths(folder_name, folder_location)
        val_vec = []
        for path in paths:
            if folder_name == "full_path":
                X, Y = AECDDataLinear(path, folder_name)
                val = AECDGradeExperiments(X, Y, map_arr)
                val_vec.append(val)
            elif len(val_vec) < 27 and (not (folder_name == "full_path")):
                X, Y = AECDDataLinear(path, folder_name)
                val = AECDGradeExperiments(X, Y, map_arr)
                val_vec.append(val)

        mean_val, std_val = norm.fit(val_vec)

        if folder_name[0] == "A":
            mean_val_vec[0] += mean_val
            std_val_vec[0] += std_val
        elif folder_name[0] == "B":
            mean_val_vec[1] += mean_val
            std_val_vec[1] += std_val
        elif folder_name[0] == "C":
            mean_val_vec[2] += mean_val
            std_val_vec[2] += std_val
        elif folder_name[0] == "D":
            std_val_vec[3] += std_val
            mean_val_vec[3] += mean_val
        elif folder_name == "full_path":
            mean_val_full_path = mean_val
            std_val_full_path = std_val

    VOA_validation_mean = (mean_val_full_path - mean_val_vec).tolist()
    VOA_validation_std = []
    for val in std_val_vec:
        VOA_validation_std.append(math.sqrt(val**2 + std_val_full_path**2))

    confidence_interval = []
    for i in range(len(VOA_validation_mean)):
        confidence_interval.append(
            confidenceInterval(VOA_validation_std[i], 0.99, 27))

    location = [1/6, 2/6, 3.5/6, 4.5/6]
    plt.rcParams.update({'font.size': 14})
    plt.plot(location, VOA_validation_mean, color='k', label="AECD")
    plt.plot(location, VOAExperimentsResults,
             '<', color='g', label="VOA")
    plt.fill_between(location,
                     np.array(VOA_validation_mean) +
                     0.3*(np.array(VOA_validation_std)),
                     np.array(VOA_validation_mean) -
                     0.3*(np.array(VOA_validation_std)),
                     color='r', alpha=.1,
                     label="0.3 Standard Deviation")
    plt.fill_between(location,
                     np.array(VOA_validation_mean) +
                     np.array(confidence_interval),
                     np.array(VOA_validation_mean) -
                     np.array(confidence_interval),
                     color='b', alpha=.1,
                     label="Confidence Interval")
    plt.grid(color='lightgray', linestyle='--')
    # plt.legend()
    plt.ylim(-1, 1)
    plt.xlabel("path progression")
    plt.ylabel("cost difference")
    plt.show()

    return_vals = []
    for i in range(3):
        return_vals.append(sameMaxValLocationMultiMax(
            [0] + VOA_validation_mean, [0] + VOAExperimentsResults, i+1))
    return_vals.append(sameMaxValLocationNearMax(
        [0] + VOA_validation_mean, [0] + VOAExperimentsResults))

    # check if the VOA values are inside 1 std of the validation values
    VOA_inside_std = 0
    for i, val in enumerate(VOAExperimentsResults):
        mean = VOA_validation_mean[i]
        std = VOA_validation_std[i]
        if mean - std <= val <= mean + std:
            VOA_inside_std += 1

    return return_vals, VOA_inside_std


def PythonSimulations(map_num: int, num_of_tests: int, VOI_Sigma: List[float], map_size: float, grid_size_vec: List[int], name_extra: string, VOA_Sigma: List[float]):
    mydir = fileCreation(name_extra)
    C_obs, init_location, goal_location = obstacleMapByNumber(
        map_size, map_num)
    colors = ['b', 'g', 'red', 'c', 'm']

    connected = False
    while connected == False:
        sol, start, connected = createScenario(
            C_obs, init_location, goal_location, map_size)

    for grid_size in grid_size_vec:
        map_arr = arrayMapByNumber(grid_size, map_size, map_num)

        for j, voa_Sigma in enumerate(VOA_Sigma):
            help_steps, EVOI_arr, planned_path, help_steps = VOAExperiments(
                map_arr, sol, start, map_size, grid_size, voa_Sigma)

            text_indicator = 'map_num_' + \
                str(map_num) + 'VOA_Sigma_' + str(voa_Sigma)
            np.savetxt(mydir + '/help_steps' +
                       text_indicator + '.out', help_steps)
            np.savetxt(mydir + '/EVOI_arr' + text_indicator + '.out', EVOI_arr)

            if voa_Sigma == VOA_Sigma[0]:
                help_steps_plot = []
                for i in range(len(help_steps)):
                    help_steps_plot.append(help_steps[i]/help_steps[-1])

            plt.plot(help_steps_plot, EVOI_arr, 'o', color=colors[j],
                     label="VOA for movement error of" + str(voa_Sigma))

        for i, voi_sigma in enumerate(VOI_Sigma):
            VOI_mu, VOI_std_plus, VOI_std_min = AECDExperiments(
                map_arr, sol, start, num_of_tests, help_steps, voi_sigma, map_size, grid_size)
            text_indicator = 'map_num_' + \
                str(map_num) + 'VOI_Sigma_' + str(voi_sigma)
            np.savetxt(mydir + '/VOI_mu' + text_indicator + '.out', VOI_mu)
            np.savetxt(mydir + '/VOI_std_plus' +
                       text_indicator + '.out', VOI_std_plus)
            np.savetxt(mydir + '/VOI_std_min' +
                       text_indicator + '.out', VOI_std_min)
            plt.plot(help_steps_plot, VOI_mu, 'o', color='black',
                     label="Average cost difference for movement error of " + str(voi_sigma))

        plt.grid(color='lightgray', linestyle='--')
        plt.legend()
        plt.xlabel("s")
        plt.ylabel("Path cost")
        plt.show()


def runningPastResults(validation_Sigma: List[float], VOA_Sigma: List[float], colors, map_num: int, folder: string, multi_max_amount: int = 2, do_plot: bool = False):
    txt_map_num = 'map_num_' + str(map_num[0])
    txt_ending = '.out'
    did_calc = False

    txt_VOI_Sigma = 'VOI_Sigma_' + str(validation_Sigma[0])
    txt_add_new = txt_map_num + txt_VOI_Sigma + txt_ending
    VOI_mu = np.loadtxt(folder + "\VOI_mu" + txt_add_new)
    VOI_std_plus = np.loadtxt(folder + "\VOI_std_plus" + txt_add_new)
    VOI_1_std = VOI_std_plus - VOI_mu

    VOA_arr_all = []
    for i, voa_sigma in enumerate(VOA_Sigma):
        txt_VOA_Sigma = 'VOA_Sigma_' + str(voa_sigma)
        txt_add = txt_map_num + txt_VOA_Sigma + txt_ending

        VOA_arr_all.append(np.loadtxt(folder + "\EVOI_arr" + txt_add))
        help_steps = np.loadtxt(folder + "\help_steps" + txt_add)

    true_max_x_axis = [0, 0, 0, 0, 0]
    error_magnitude = [0, 0, 0, 0, 0]

    # NOTE: added code to filter out bad results
    VOI_mu_list = (VOI_mu).tolist()
    # VOI_mu_list.sort(reverse=True)
    did_calc = True

    for i, voa_sigma in enumerate(VOA_Sigma):
        true_max_x_axis[i] = sameMaxValLocationMultiMax(
            [0] + (VOI_mu).tolist(),
            [0] + (VOA_arr_all[i]).tolist(), multi_max_amount)
        if true_max_x_axis[i] == False:
            max_vec1_index = ((VOI_mu).tolist()).index(max((VOI_mu).tolist()))
            max_vec2_index = ((VOA_arr_all[i]).tolist()).index(
                max((VOA_arr_all[i]).tolist()))
            true_max_x_axis[i] = (VOI_mu_list[max_vec1_index]
                                  <= VOI_mu_list[max_vec2_index]*1.2)

        if multi_max_amount == 1:
            EVOI_list = (VOA_arr_all[i]).tolist()
            EVOI_list.sort(reverse=True)
            error_magnitude[i] = 100 * \
                ((EVOI_list[0] - EVOI_list[2]) /
                 (EVOI_list[0] - EVOI_list[-1]))
            if error_magnitude[i] > 100:
                print(error_magnitude[i])

    VOA_arr_all.append(VOI_mu)
    VOA_arr_all.append(VOI_1_std)
    help_steps_plot = []
    for i in range(len(help_steps)):
        help_steps_plot.append(help_steps[i]/help_steps[-1])

    if do_plot == True:
        for i, voi_sigma in enumerate(validation_Sigma):
            plt.plot(help_steps, VOI_mu, 'o', color=colors[i],
                     label="Average cost difference for movement error of " + str(voi_sigma))

            # plt.plot(help_steps, VOI_std_plus, 'o', color='black', label="VOI 1 std")
            # plt.plot(help_steps, VOI_std_min, 'o', color='black')

        plt.plot(help_steps, EVOI_arr, 'o', color='red',
                 label="VOA for movement error of 0.2")
        plt.grid(color='lightgray', linestyle='--')
        plt.legend()
        plt.xlabel("\u03C4")
        plt.ylabel("Path cost")

        plt.show()

    return true_max_x_axis, VOA_arr_all, help_steps_plot, did_calc, error_magnitude


def plotPastResults(VOA_Sigma: List[float], max_res_matrix_x, max_res_matrix_y, help_steps_plot):
    plt.rcParams.update({'font.size': 10})
    colors1 = ['dimgray', 'fuchsia', 'orangered']
    shapes1 = ['^', ' p', 's']
    shapes2 = ['ro', 'ms', 'g<', 'c>', 'bX', 'k-']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, max_res_vec in enumerate(max_res_matrix_x):
        ax.plot(VOA_Sigma, max_res_vec, shapes1[i], color=colors1[i],
                label="Cordiality = " + str(i+1))
    plt.grid(color='lightgray', linestyle='--')
    # plt.legend()
    plt.xlabel("\u0394\u03A3")
    plt.ylabel("%")
    plt.ylim(0, 100)
    plt.show()

    confidence_interval = []
    for i in range(len(max_res_matrix_y[5])):
        confidence_interval.append(confidenceInterval(max_res_matrix_y[6][i]))

    for i, max_res_vec in enumerate(max_res_matrix_y[0:5]):
        plt.plot(help_steps_plot, max_res_vec,  shapes2[i],
                 label="VOA, \u0394\u03A3 = " + str(VOA_Sigma[i]))
    plt.plot(help_steps_plot, max_res_matrix_y[5],  shapes2[i+1],
             label="AECD, \u0394\u03A3 = 0.2")
    plt.fill_between(help_steps_plot,
                     max_res_matrix_y[5] + confidence_interval,
                     max_res_matrix_y[5] - confidence_interval,
                     color='b', alpha=.1, label="Confidence Interval")
    plt.fill_between(help_steps_plot,
                     max_res_matrix_y[5] + max_res_matrix_y[6]*0.1,
                     max_res_matrix_y[5] - max_res_matrix_y[6]*0.1,
                     color='r', alpha=.1, label="0.1 Standard Deviation")
    plt.grid(color='lightgray', linestyle='--')
    # plt.legend()
    # plt.ylim(-0.045, 0.025)
    # plt.xlabel("Path progression")
    # plt.ylabel("VOA and average of tested values")
    plt.show()


def informationCorrectionResults():
    distance = []
    location_err = []
    folder_location = os.getcwd() + '/real_world_experiments/Information'
    for folder_name in os.listdir(folder_location):
        paths = folder2Paths(folder_name, folder_location)
    for path in paths:
        distance.append(path[0][0])
        location_err.append(euclideanDist(path[1], [0, 0]))

    plt.plot(distance, location_err, 'o', color='black')
    plt.grid(color='lightgray', linestyle='--')
    plt.xlabel("Initial distance from Goal [m]")
    plt.ylabel("Final distance after correction [m]")
    plt.show()


def RunPastResults():
    validation_Sigma = [0.2]
    VOA_Sigma = [0.15, 0.175, 0.2, 0.225, 0.25]
    colors = ['b', 'g', 'k', 'c', 'm']
    map_num = [-1]
    window_size = 1
    # main_folder = 'Information_5000_runs_changing_VOA_sigma'
    main_folder = 'Teleportation_1000_runs_changing_VOA_Sigma'
    # main_folder = 'info_extra_runs'
    folder_location = os.getcwd() + '\\' + main_folder + '\\'
    multi_max_vec = [1, 2, 3]
    max_result_matrix_x = []
    # [[[0] * 100] * 15] * len(VOA_Sigma)
    max_result_matrix_y = np.zeros((len(VOA_Sigma) + 2, 15, 100))

    error_mag_matrix_15 = []
    error_mag_matrix_175 = []
    error_mag_matrix_20 = []
    error_mag_matrix_225 = []
    error_mag_matrix_25 = []

    loc_vec = [23]  # list(range(100))
    for loc in loc_vec:
        print(loc)
        for j, multi_max_amount in enumerate(multi_max_vec):
            true_max_average_x = [0, 0, 0, 0, 0]
            valid_runs = 0

            for i, folder_name in enumerate((os.listdir(folder_location))[0:100]):
                true_max_x, true_max_y, help_steps_plot, did_calc, error_magnitude = runningPastResults(
                    validation_Sigma, VOA_Sigma, colors, map_num,
                    folder_location + folder_name, multi_max_amount)
                if did_calc == True:
                    valid_runs += 1
                true_max_average_x = [x + y for x,
                                      y in zip(true_max_average_x, true_max_x)]

                if error_magnitude[0] != 0:
                    error_mag_matrix_15.append(error_magnitude[0])
                if error_magnitude[1] != 0:
                    error_mag_matrix_175.append(error_magnitude[1])
                if error_magnitude[2] != 0:
                    error_mag_matrix_20.append(error_magnitude[2])
                if error_magnitude[3] != 0:
                    error_mag_matrix_225.append(error_magnitude[3])
                if error_magnitude[4] != 0:
                    error_mag_matrix_25.append(error_magnitude[4])

                if j == 0:
                    for index1 in range(len(true_max_y)):
                        for index2 in range(len(true_max_y[index1])):
                            max_result_matrix_y[index1][index2][i] = true_max_y[index1][index2]

            print(valid_runs)
            true_max_average_x = [i * (100/valid_runs)
                                  for i in true_max_average_x]
            max_result_matrix_x.append(true_max_average_x)

        max_result_matrix_y_mean = np.zeros((len(VOA_Sigma) + 2, 15))
        for index1 in range(len(max_result_matrix_y_mean)):
            for index2 in range(len(max_result_matrix_y_mean[index1])):
                max_result_matrix_y_mean[index1][index2] = statistics.mean(
                    max_result_matrix_y[index1][index2])

        mean15 = statistics.mean(error_mag_matrix_15)
        mean175 = statistics.mean(error_mag_matrix_175)
        mean20 = statistics.mean(error_mag_matrix_20)
        mean225 = statistics.mean(error_mag_matrix_225)
        mean25 = statistics.mean(error_mag_matrix_25)

        print(round(mean15, 2))
        print(round(mean175, 2))
        print(round(mean20, 2))
        print(round(mean225, 2))
        print(round(mean25, 2))

        plotPastResults(VOA_Sigma, max_result_matrix_x,
                        max_result_matrix_y_mean, help_steps_plot)


def RunSimulation():
    validation_Sigma = [0.2]
    VOA_Sigma = [0.15, 0.175, 0.2, 0.225, 0.25]
    python_maps = [-1]  # [-1]  # [1, 2, 3]
    map_size = 100
    grid_size_vec = [100]
    num_of_VOI_tests = 1000
    name_extra = input("enter name extra:")
    for map_num in python_maps:
        for i in range(1):
            ratio_vec = PythonSimulations(map_num, num_of_VOI_tests,
                                          validation_Sigma, map_size,
                                          grid_size_vec, name_extra, VOA_Sigma)


def RunRealWorld():
    map_size = 40
    grid_size_vec = [40]
    num_of_tests = 1
    num_of_true_max = 0
    num_of_VOA_inside_std = 0
    num_of_correct_2max = 0
    num_of_correct_3max = 0
    num_of_correct_near_max = 0
    for i in range(num_of_tests):
        correct_max_vec, VOA_inside_std = RealWorldExperiments(
            grid_size_vec[0], map_size, True)
        num_of_true_max += correct_max_vec[0]
        num_of_correct_2max += correct_max_vec[1]
        num_of_correct_3max += correct_max_vec[2]
        num_of_correct_near_max += correct_max_vec[3]
        num_of_VOA_inside_std += VOA_inside_std
        print("test num " + str(i))
        print("num of true max out of tests " + str(num_of_true_max/(i+1)))
        print("num of true max 2 maximals out of tests " +
              str(num_of_correct_2max/(i+1)))
        print("num of true max 3 maximals out of tests " +
              str(num_of_correct_3max/(i+1)))
        print("num of true max 3 near out of tests " +
              str(num_of_correct_near_max/(i+1)))
        print("num of VOA inside 1 std out of tests " +
              str(num_of_VOA_inside_std/(4*(i+1))))
    print(num_of_true_max/num_of_tests)


## Integrable planner ##
def integrablePlanner(mu=np.matrix([[50.5], [50.5]]), Sigma=np.matrix([[20, 0], [0, 20]]), grid_size: int = 100, map_size: float = 100, map_num: int = -1):
    map_arr = arrayMapByNumber(grid_size, map_size, map_num)
    rows, cols = (grid_size, grid_size)
    Sigma_array = [[0 for i in range(cols)] for j in range(rows)]
    cost_array = [[0 for i in range(cols)] for j in range(rows)]

    proportion = map_size/grid_size
    bel = belief(mu, Sigma)
    prob_init_arr = extractGaussianMatFromBelief(bel, map_size, grid_size)

    for i in range(grid_size):
        for j in range(grid_size):
            if prob_init_arr[j][i] != 0:

                init_x = j * proportion
                init_y = i * proportion
                init = NodePRM(Point(init_x, init_y))
                goal = NodePRM(Point(((mu.A)[0]).max(), ((mu.A)[1]).max()))
                sol = [goal, init]

                Belief = VOAInPlanning(
                    sol, init, help_step=-1)[0]
                p_correct_cost = VOAGrade(
                    Belief, map_arr, map_size, grid_size)

                Sigma_final = ((Belief[-1]).Sigma.A)[0][0]
                Sigma_array[j][i] = Sigma_final
                cost_array[j][i] = p_correct_cost

    np.savetxt('map_arr.out', map_arr)
    np.savetxt('prob_init_arr.out', prob_init_arr)
    np.savetxt('Sigma_array.out', Sigma_array)
    np.savetxt('cost_array.out', cost_array)


def integrablePlannerPlotResults(grid_size: int = 100):
    map_arr = np.loadtxt('map_arr.out')
    prob_init_arr = np.loadtxt('prob_init_arr.out')
    Sigma_array = np.loadtxt('Sigma_array.out')
    cost_array = np.loadtxt('cost_array.out')

    d_sigma_max_add = []
    d_sigma_mean = []
    d_sigma_min_add = []

    d_cost_max_add = []
    d_cost_mean = []
    d_cost_min_add = []

    d_x = list(range(0, 10))

    for x in d_x:
        sigma_vec = []
        cost_vec = []
        for i in range(grid_size):
            for j in range(grid_size):
                if prob_init_arr[j][i] != 0 and prob_init_arr[j+x][i] != 0:
                    sigma_vec.append(np.abs(Sigma_array[j+x][i] -
                                            Sigma_array[j][i]))
                    cost_vec.append(np.abs(cost_array[j+x][i] -
                                           cost_array[j][i]))
                if prob_init_arr[j][i] != 0 and prob_init_arr[j][i+x] != 0:
                    sigma_vec.append(np.abs(Sigma_array[j][i+x] -
                                            Sigma_array[j][i]))
                    cost_vec.append(np.abs(cost_array[j][i+x] -
                                           cost_array[j][i]))

        d_sigma_mean.append(sum(sigma_vec)/len(sigma_vec))
        d_sigma_max_add.append(max(sigma_vec) - sum(sigma_vec)/len(sigma_vec))
        d_sigma_min_add.append(sum(sigma_vec)/len(sigma_vec) - min(sigma_vec))

        d_cost_mean.append(sum(cost_vec)/len(cost_vec))
        d_cost_max_add.append(max(cost_vec) - sum(cost_vec)/len(cost_vec))
        d_cost_min_add.append(sum(cost_vec)/len(cost_vec) - min(cost_vec))

    plt.rcParams.update({'font.size': 18})
    plt.plot(d_x, d_cost_mean,  'b')
    plt.fill_between(d_x,
                     np.array(d_cost_mean) +
                     np.array(d_cost_max_add),
                     np.array(d_cost_mean) -
                     np.array(d_cost_min_add),
                     color='b', alpha=.1)
    plt.grid(color='lightgray', linestyle='--')
    plt.xlabel("Initial location distance")
    plt.ylabel("Cost difference")
    plt.show()

    plt.plot(d_x, d_sigma_mean,  'b')
    plt.fill_between(d_x,
                     np.array(d_sigma_mean) +
                     np.array(d_sigma_max_add),
                     np.array(d_sigma_mean) -
                     np.array(d_sigma_min_add),
                     color='b', alpha=.1)
    plt.grid(color='lightgray', linestyle='--')
    plt.xlabel("Initial location distance")
    plt.ylabel("Final location uncertainty")
    plt.show()


if __name__ == "__main__":
    # integrablePlannerPlotResults()
    RunRealWorld()
    # RunSimulation()
    # RunPastResults()
