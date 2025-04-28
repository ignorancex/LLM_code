#!/usr/bin/env python3

# command line program
import argparse
# numpy
import numpy as np
# internal modules
import libpost
# plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# copy
import copy

def parse():
    parser = argparse.ArgumentParser(description='Plots 2D trajectories of all solutions over time, assuming pos_0 as x and pos_1 as y.')
    parser.add_argument('--equation-list', '-e', nargs='*', default=[], help='equations for which to plot trajectories')
    parser.add_argument('--color-list', '-c', nargs='*', default=[], help='color for each equation')
    parser.add_argument('--step', '-s', type=int, default=1, help='animation frame step')
    return parser.parse_args()

def main(input_equation_list, input_color_list, input_step):
    # equations
    if input_equation_list:
        equation_name_list = input_equation_list
    else:
        print("INFO: Reading equation names...", flush=True)
        equation_name_list = libpost.get_equation_names()
    # colors
    if input_color_list:
        color_list = input_color_list
    else:
        cmap = plt.get_cmap("plasma", len(equation_name_list))
        color_list = [cmap(index) for index in range(len(equation_name_list))]
    # process
    print("INFO: Reading time...", flush=True)
    time_dir_array, time_array = libpost.get_time()
    print("INFO: Reading equation property over time...", flush=True)
    pos_0_over_time = {}
    pos_1_over_time = {}
    vorticity_over_time = {}
    for equation_name in equation_name_list:
        pos_0_over_time[equation_name] = np.array(libpost.get_equation_property_over_time(equation_name, ".*__pos_0", time_dir_array))
        pos_1_over_time[equation_name] = np.array(libpost.get_equation_property_over_time(equation_name, ".*__pos_1", time_dir_array))
        vorticity_over_time[equation_name] = np.array(libpost.get_equation_property_over_time(equation_name, ".*__vorticity", time_dir_array))
    print("INFO: Animating...", flush=True)
    art_fig, art_ax = plt.subplots()
    art_ax.set_aspect('equal', 'box')
    art_ax.grid(False)
    art_ax.axis('off')
    art_ax.set_facecolor("black")
    art_fig.set_facecolor("black")
    trajectories = {}
    artists = []
    for time_index, time in enumerate(time_array):
        if time_index % input_step == 0:
            artists.append([])
            for equation_index, equation_name in enumerate(equation_name_list):
                arts = art_ax.plot(pos_0_over_time[equation_name][0:time_index+1:input_step], pos_1_over_time[equation_name][0:time_index+1:input_step], color=color_list[equation_index % len(color_list)], alpha=0.5)
                artists[-1] += arts
                art = art_ax.scatter(pos_0_over_time[equation_name][time_index], pos_1_over_time[equation_name][time_index], s=12, color=color_list[equation_index % len(color_list)])
                artists[-1].append(art)
    # legend
    for equation_index, equation_name in enumerate(equation_name_list):
        art_ax.text(
            pos_0_over_time[equation_name].max(), 
            pos_1_over_time[equation_name].max(),
            equation_name, 
            fontsize=8,
            #color=color_list[equation_index % len(color_list)],
            backgroundcolor=color_list[equation_index % len(color_list)]
        )
    # anim
    anim = animation.ArtistAnimation(art_fig, artists, interval=33)
    anim.save("trajectory_animation.mp4")

if __name__ == '__main__':
    args = parse()
    main(args.equation_list, args.color_list, args.step)
