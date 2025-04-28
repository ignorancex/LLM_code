#!/usr/bin/env python3

# command line program
import argparse
# deepcopy
import copy
# numpy
import numpy as np
# internal modules
import libpost

def parse():
    parser = argparse.ArgumentParser(description='Extracts particles trajectories.')
    parser.add_argument('nb', type=int, help='specify the number of trajectories to be extracted')
    parser.add_argument('-i', '--indexs', nargs='+', type=int, help='specify particles indexs (NOT IMPLEMENTED YET)')
    parser.add_argument('-s', '--split', action='store_true', help='splits each particle trajectory into separate files')
    return parser.parse_args()

def main(nb, split = False):
    # get positions
    trajectories = libpost.get_objects_properties(["pos_0", "pos_1", "pos_2"])
    for object_name in trajectories:
        trajectories[object_name]["value"] = trajectories[object_name]["value"][:, :3*nb]
        trajectories[object_name]["info"] = trajectories[object_name]["info"][:3*nb]
    if split:
        splited_trajectories = {}
        for object_name in trajectories:
            for i in range(nb):
                splited_trajectories[object_name + "__particle_{index}".format(index=i)] = {}
                splited_trajectories[object_name + "__particle_{index}".format(index=i)]["value"] = trajectories[object_name]["value"][:,3*i:3*(i+1)]
                splited_trajectories[object_name + "__particle_{index}".format(index=i)]["info"] = trajectories[object_name]["info"][3*i:3*(i+1)]
        libpost.save(splited_trajectories, "trajectories")
    else:
        libpost.save(trajectories, "trajectories")

if __name__ == '__main__':
    args = parse()
    main(args.nb, args.split)
