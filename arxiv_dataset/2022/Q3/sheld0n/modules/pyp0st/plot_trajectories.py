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

def parse():
    parser = argparse.ArgumentParser(description='Chain post processing')
    parser.add_argument('--step', '-s', type=int, default=16, help='animation frame step')
    parser.add_argument('--plot', '-p', action='store_true', help='plot and show the last frame')
    return parser.parse_args()

def main(anim_step, is_plotting):
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
    artists = []
    for name in post_names:
        print("\tINFO: Processing post {name}...".format(name=name), flush=True)
        for time_index in range(0, time.size):
            current_time = time[time_index]
            print("\t\tINFO: Processing t={t}/{t_f}...".format(t=current_time, t_f=time[-1]), flush=True)
            print("\t\t\tINFO: Reading data...", flush=True)
            post = libpost.get_post(time_dirs[time_list.index(current_time)], name)
            info = libpost.get_post_properties(post, ["test_.*__pos_0"])["info"]
            pos_0 = libpost.get_post_properties(post, ["test_.*__pos_0"])["value"]
            pos_1 = libpost.get_post_properties(post, ["test_.*__pos_1"])["value"]
            print("\t\t\tINFO: Done.", flush=True)
            print("\t\t\t\tINFO: Sorting data...", flush=True)
            sorted_index = np.empty(pos_0.size)
            sorted_index[:] = np.nan
            for index in range(len(info)):
                sorted_index[index] = libpost.get_properties_from_string(info[index])["particle"]
            sorted_index = np.argsort(sorted_index)
            sorted_index = sorted_index[np.logical_not(np.isnan(sorted_index))]
            # updating everything
            pos_0 = pos_0[sorted_index]
            pos_1 = pos_1[sorted_index]
            print("\t\t\tINFO: Done.", flush=True)
            print("\t\t\tINFO: Processing data...", flush=True)
            if not trajectories:
                for index in range(pos_0.size):
                    trajectories.append({"x":[pos_0[index]], "y":[pos_1[index]]})
            else:
                for index in range(pos_0.size):
                    trajectories[index]["x"].append(pos_0[index])
                    trajectories[index]["y"].append(pos_1[index])
            print("\t\t\tINFO: Done.", flush=True)
            print("\t\t\tINFO: Animating...", flush=True)
            if time_index % anim_step == 0:
                artists.append([])
                for index in range(len(trajectories)):
                    art, = art_ax.plot(trajectories[index]["x"], trajectories[index]["y"], color=(0.5, 0.5, 0.75))
                    artists[-1].append(art)
            # art = art_ax.scatter(pos_0, pos_1, color=(0.75, 0.75, 1.0))
            # artists[-1].append(art)
            print("\t\t\tINFO: Done.", flush=True)
            print("\t\tINFO: Done processing t={t}/{t_f}.".format(t=current_time, t_f=time[-1]), flush=True)
        print("\tDone processing post {name}.".format(name=name), flush=True)
    print("\t\tINFO: Animating...", flush=True)
    anim = animation.ArtistAnimation(art_fig, artists, interval=30)
    anim.save("animation.mp4")
    print("\t\tINFO: Done.", flush=True)
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main(args.step, args.plot)
