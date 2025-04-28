#!/usr/bin/env python3

# command line program
import argparse
# deepcopy
import copy
# numpy
import numpy as np
# internal modules
import libpost
# plotting
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser(description='Plots 2D trajectories of all solutions over time, assuming pos_0 as x and pos_1 as y.')
    parser.add_argument('--equation-list', '-e', nargs='*', default=[], help='equations for which to plot trajectorys')
    parser.add_argument('--color-list', '-c', nargs='*', default=[], help='color for each equation')
    return parser.parse_args()

def main(input_equation_list, input_color_list):
    # equations
    if input_equation_list:
        equation_name_list = input_equation_list
    else:
        print("INFO: Reading equation names...", flush=True)
        equation_name_list = libpost.get_equation_names()
    print("INFO: Reading time...", flush=True)
    time_dir_array, time_array = libpost.get_time()
    print("INFO: Reading equation property over time...", flush=True)
    pos_1_over_time = {}
    for equation_name in equation_name_list:
        pos_1_over_time[equation_name] = np.array(libpost.get_equation_property_over_time(equation_name, ".*__pos_1", time_dir_array))
    print("INFO: Computing the vertical migration performance...", flush=True)
    average_performance = {}
    ci_performance = {}
    for equation_name in equation_name_list:
        tmp_vertical_displacement = (pos_1_over_time[equation_name][-1] - pos_1_over_time[equation_name][0])
        average_performance[equation_name] = tmp_vertical_displacement.mean() / (8.0 * np.pi) # (8.0 * np.pi) corresponds to the duration of the simulation, change it if our edited the param/run/parameters.h file
        ci_performance[equation_name] = 1.96 * tmp_vertical_displacement.std() / np.sqrt(len(tmp_vertical_displacement)) / (8.0 * np.pi) # (8.0 * np.pi) corresponds to the duration of the simulation, change it if our edited the param/run/parameters.h file
    print("INFO: Extracting parameters from equation names...", flush=True)
    average_performance_sorted = {}
    ci_performance_sorted = {}
    for equation_name in equation_name_list:
        parameters = libpost.get_properties_from_string(equation_name)
        if "TimeHorizon" in parameters and "SwimmingVelocity" in parameters:
            if not parameters["SwimmingVelocity"] in average_performance_sorted:
                average_performance_sorted[parameters["SwimmingVelocity"]] = [[parameters["TimeHorizon"], average_performance[equation_name]]]
                ci_performance_sorted[parameters["SwimmingVelocity"]] = [[parameters["TimeHorizon"], ci_performance[equation_name]]]
            else:
                average_performance_sorted[parameters["SwimmingVelocity"]].append([parameters["TimeHorizon"], average_performance[equation_name]])
                ci_performance_sorted[parameters["SwimmingVelocity"]].append([parameters["TimeHorizon"], ci_performance[equation_name]])
    # average
    for swimming_velocity in average_performance_sorted:
        average_performance_sorted[swimming_velocity] = np.array(average_performance_sorted[swimming_velocity])
        average_performance_sorted[swimming_velocity] = average_performance_sorted[swimming_velocity][average_performance_sorted[swimming_velocity][:, 0].argsort()]
    # ci
    for swimming_velocity in ci_performance_sorted:
        ci_performance_sorted[swimming_velocity] = np.array(ci_performance_sorted[swimming_velocity])
        ci_performance_sorted[swimming_velocity] = ci_performance_sorted[swimming_velocity][ci_performance_sorted[swimming_velocity][:, 0].argsort()]
    print("INFO: Plotting...", flush=True)
    # colors
    if input_color_list:
        color_list = input_color_list
    else:
        cmap = plt.get_cmap("plasma", len(average_performance_sorted))
        color_list = [cmap(index) for index in range(len(average_performance_sorted))]
    marker_list = ["o", "^", "s", "P", "*"]
    # plot
    plot_index = 0
    for swimming_velocity in sorted(average_performance_sorted):
        plt.fill_between(
            average_performance_sorted[swimming_velocity][:, 0],
            (average_performance_sorted[swimming_velocity][:, 1] - 0.5 * ci_performance_sorted[swimming_velocity][:, 1]) / swimming_velocity, 
            (average_performance_sorted[swimming_velocity][:, 1] + 0.5 * ci_performance_sorted[swimming_velocity][:, 1]) / swimming_velocity,
            linewidth=0.0,
            color=color_list[plot_index],
            alpha = 0.25
        )
        plot_index += 1
    plot_index = 0
    for swimming_velocity in sorted(average_performance_sorted):
        plt.plot(
            average_performance_sorted[swimming_velocity][:, 0], 
            average_performance_sorted[swimming_velocity][:, 1] / swimming_velocity, 
            color=color_list[plot_index], 
            label=r'$V_{\mathrm{swim}} = ' + str(swimming_velocity) + r'$', 
            marker=marker_list[plot_index % len(marker_list)]
        )
        plot_index += 1
    plt.xlabel(r'$\tau / \pi$')
    plt.ylabel(r'$\langle V_{\mathrm{eff.}} \rangle / V_{\mathrm{swim}}$')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    args = parse()
    main(args.equation_list, args.color_list)
