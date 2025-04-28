#!/usr/bin/env python3

# command line program
import argparse
# numpy
import numpy as np
# custom lib
import libpost

BIN_NB = 1000
BIN_RANGE = (0, 0.0028)

def parse():
    parser = argparse.ArgumentParser(description='Filament post processing')
    parser.add_argument('name', help='specify the filament object name')
    parser.add_argument('timestep', type=int, help='specify the time step to post process')
    parser.add_argument('distance', type=float, help='max distance to consider')
    return parser.parse_args()

def main(name, timestep, distance):
    time_dirs, time_list, time = libpost.get_time()
    obj = libpost.get_object(time_dirs[time_list.index(time[timestep])], name)
    # # get positions
    objects_pos_0 = libpost.get_object_properties(obj, ["l_.*__pos_0"])
    objects_pos_1 = libpost.get_object_properties(obj, ["l_.*__pos_1"])
    objects_pos_2 = libpost.get_object_properties(obj, ["l_.*__pos_2"])
    objects_scalar = libpost.get_object_properties(obj, ["l_.*__scal"])
    objects_l = libpost.get_object_properties(obj, ["l_.*__l"])
    # compute values
    pos = np.empty((objects_pos_0["value"].size, 3))
    pos[:, 0] = objects_pos_0["value"]
    pos[:, 1] = objects_pos_1["value"]
    pos[:, 2] = objects_pos_2["value"]
    scalar = objects_scalar["value"]
    l = objects_l["value"]
    # select scalars
    scalar_diff = []
    l_diff = []
    for i in range(pos.shape[0]):
        for j in range(i +1, pos.shape[0]):
            d = np.linalg.norm(pos[j, :] - pos[i, :])
            if d < distance:
                scalar_diff.append(np.abs(scalar[i] - scalar[j]))
                l_diff.append(np.abs(l[i] - l[j]))
    # compute scalar pdfs
    hist, bin_edges = np.histogram(scalar_diff, bins=np.histogram_bin_edges(scalar_diff, bins=BIN_NB, range=BIN_RANGE), range=BIN_RANGE, density=True)
    bins = 0.5*(bin_edges[1:] + bin_edges[:-1])
    np.savetxt("{name}__scalar_diff_pdf__d_{d}__t_{time}.csv".format(name=name, d=str(distance).replace(".", "o"), time=str(time[timestep]).replace(".", "o")), np.column_stack((bins, hist)), delimiter=",", header="scalar_diff,pdf")
    # compute l pdfs
    hist, bin_edges = np.histogram(l_diff, bins=np.histogram_bin_edges(l_diff, bins=BIN_NB), density=True)
    bins = 0.5*(bin_edges[1:] + bin_edges[:-1])
    np.savetxt("{name}__l_diff_pdf__d_{d}__t_{time}.csv".format(name=name, d=str(distance).replace(".", "o"), time=str(time[timestep]).replace(".", "o")), np.column_stack((bins, hist)), delimiter=",", header="l_diff,pdf")

if __name__ == '__main__':
    args = parse()
    main(args.name, args.timestep, args.distance)
