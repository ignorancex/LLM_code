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
import matplotlib.animation as animation
# os
import os

def parse():
    parser = argparse.ArgumentParser(description='Chain post processing')
    parser.add_argument('--step', '-s', type=int, default=16, help='animation frame step')
    parser.add_argument('--plot', '-p', action='store_true', help='plot and show the last frame')
    parser.add_argument('--trajectories', '-t', action='store_true', help='add trajectories to the plot')
    return parser.parse_args()

def main(anim_step, is_plotting, is_processing_trajectories):
    if not os.path.exists('output'):
        os.makedirs('output')
    print("INFO: Post processing trajectories.", flush=True)
    print("INFO: Reading posts...", flush=True)
    post_names = libpost.get_post_names()
    print("INFO: Done. Post names are:", " ".join(post_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    print("INFO: Done.", flush=True)
    print("INFO: Processing trajectories...", flush=True)
    art_fig, art_ax = plt.subplots()
    art_ax.set_aspect('equal', 'box')
    art_ax.grid(False)
    art_ax.axis('off')
    art_ax.set_facecolor("black")
    art_fig.set_facecolor("black")
    trajectories = []
    for name in post_names:
        print("\tINFO: Processing post {name}...".format(name=name), flush=True)
        for time_index in range(0, time.size):
            current_time = time[time_index]
            print("\t\tINFO: Processing t={t}/{t_f}...".format(t=current_time, t_f=time[-1]), flush=True)
            print("\t\t\tINFO: Reading Data...", flush=True)
            post = libpost.get_post(time_dirs[time_list.index(current_time)], name)
            print("\t\t\tINFO: Done.", flush=True)
            if is_processing_trajectories:
                print("\t\t\tINFO: Reading trajectories...", flush=True)
                pos_0 = libpost.get_post_properties(post, ["source_of_points__.*__pos_0"])["value"]
                pos_1 = libpost.get_post_properties(post, ["source_of_points__.*__pos_1"])["value"]
                lifetime = libpost.get_post_properties(post, ["source_of_points_.*__lifetime"])
                print("\t\t\tINFO: Done.", flush=True)
                print("\t\t\t\tINFO: Sorting trajectories...", flush=True)
                sorted_index = np.empty(lifetime["value"].size)
                sorted_index[:]= np.nan
                for index in range(len(lifetime["info"])):
                    sorted_index[index] = float(libpost.get_properties_from_string(lifetime["info"][index])["particle"])
                sorted_index = np.argsort(sorted_index)
                sorted_index = sorted_index[np.logical_not(np.isnan(sorted_index))]
                # updating everything
                pos_0 = pos_0[sorted_index]
                pos_1 = pos_1[sorted_index]
                lifetime = lifetime["value"][sorted_index]
                print("\t\t\tINFO: Done.", flush=True)
                print("\t\t\tINFO: Processing trajectories...", flush=True)
                if lifetime.size > 0:
                    for index in range(lifetime.size):
                        if lifetime[index] == 1.0:
                            trajectories.append({"x":[pos_0[index]], "y":[pos_1[index]]})
                    dead_points_number = len(trajectories) - lifetime.size
                    for index in range(dead_points_number - 1, -1, -1):
                        trajectories[index]["x"].pop(0)
                        trajectories[index]["y"].pop(0)
                        if not trajectories[index]["x"]:
                            trajectories.pop(index)
                            dead_points_number -= 1;
                    for index in range(lifetime.size):
                        trajectories[dead_points_number + index]["x"].append(pos_0[index])
                        trajectories[dead_points_number + index]["y"].append(pos_1[index])
                print("\t\t\tINFO: Done.", flush=True)
                print("\t\t\tINFO: Plotting trajectories...", flush=True)
                if time_index > -1 and time_index % anim_step == 0:
                    for index in range(len(trajectories)):
                        art_ax.plot(trajectories[index]["x"], trajectories[index]["y"], color=(0.25, 0.25, 0.5))
                print("\t\t\tINFO: Done.", flush=True)
            curve_index = 0
            print("\t\t\tINFO: Reading reactive front curve 0...", flush=True)
            pos_0 = libpost.get_post_properties(post, ["reactive_front__curve_index_0__s_.*__pos_0"])
            pos_1 = libpost.get_post_properties(post, ["reactive_front__curve_index_0__s_.*__pos_1"])
            point_pos_0 = libpost.get_post_properties(post, ["reactive_front__curve_index_0__point_index_.*__pos_0"])
            point_pos_1 = libpost.get_post_properties(post, ["reactive_front__curve_index_0__point_index_.*__pos_1"])
            print("\t\t\tINFO: Done.", flush=True)
            while pos_0["value"].size > 0:
                print("\t\t\tINFO: Sorting reactive front curve {curve_index}...".format(curve_index=curve_index), flush=True)
                sorted_index = np.empty(pos_0["value"].size)
                sorted_index[:]= np.nan
                for index in range(len(pos_0["info"])):
                    sorted_index[index] = float(libpost.get_properties_from_string(pos_0["info"][index])["s"])
                sorted_index = np.argsort(sorted_index)
                sorted_index = sorted_index[np.logical_not(np.isnan(sorted_index))]
                # updating everything
                pos_0 = pos_0["value"][sorted_index]
                pos_1 = pos_1["value"][sorted_index]
                print("\t\t\tINFO: Done.", flush=True)
                print("\t\t\tINFO: Plotting reactive front curve {curve_index}...".format(curve_index=curve_index), flush=True)
                if time_index > -1 and time_index % anim_step == 0:
                    art_ax.scatter(point_pos_0["value"], point_pos_1["value"], color=(1.0, 0.5, 0.5), s=15)
                    art_ax.plot(pos_0, pos_1, color=(1.0, 0.5, 0.5))
                print("\t\t\tINFO: Done.", flush=True)
                curve_index += 1
                print("\t\t\tINFO: Reading reactive front curve {curve_index}...".format(curve_index=curve_index), flush=True)
                pos_0 = libpost.get_post_properties(post, ["reactive_front__curve_index_{curve_index}__s_.*__pos_0".format(curve_index=curve_index)])
                pos_1 = libpost.get_post_properties(post, ["reactive_front__curve_index_{curve_index}__s_.*__pos_1".format(curve_index=curve_index)])
                point_pos_0 = libpost.get_post_properties(post, ["reactive_front__curve_index_{curve_index}__point_index_.*__pos_0".format(curve_index=curve_index)])
                point_pos_1 = libpost.get_post_properties(post, ["reactive_front__curve_index_{curve_index}__point_index_.*__pos_1".format(curve_index=curve_index)])
                print("\t\t\tINFO: Done.", flush=True)
            print("\t\t\tINFO: Annoting current time...", flush=True)
            if time_index > -1 and time_index % anim_step == 0:
                art_ax.annotate('t = {:.1f}'.format(current_time), xy=(0.5, 1.0), xycoords='axes fraction', horizontalalignment='center', verticalalignment='top', color=(1.0, 1.0, 1.0))
            print("\t\t\tINFO: Done.", flush=True)
            print("\t\t\tINFO: Saving...", flush=True)
            if time_index > -1 and time_index % anim_step == 0:
                art_fig.savefig("output/frame_{0:0=6d}.png".format(time_index // anim_step))
            print("\t\t\tINFO: Done.", flush=True)
            art_ax.clear()
            print("\t\tINFO: Done processing t={t}/{t_f}.".format(t=current_time, t_f=time[-1]), flush=True)
        print("\tDone processing post {name}.".format(name=name), flush=True)
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main(args.step, args.plot, args.trajectories)
