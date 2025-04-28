#!/usr/bin/env python3

# command line program
import argparse
# numpy
import numpy as np
# custom lib
import libpost

def parse():
    parser = argparse.ArgumentParser(description='Reverse post processing')
    parser.add_argument('names', nargs='*', default=libpost.get_object_names(), help='specify the object names')
    parser.add_argument('nb', nargs='?', default=0, type=int, help='specify the number of time steps to post process')
    return parser.parse_args()

def main(names, nb):
    # get time
    time_dirs, time_list, time = libpost.get_time()
    if nb == 0 or nb > time.size:
        nb = time.size
    # init
    time_nb = np.empty((nb, 1))
    distance_nb = np.empty((nb, 2))
    # stretch_nb = np.empty((nb, 2))
    # get object
    for name in names:
        print("INFO: Processing object: {name}...".format(name=name), flush=True)
        for i in range(nb):
            timestep = int(round((i / (nb - 1)) * (len(time) - 1)))
            print("\tINFO: Processing time: {timestep}/{time}...".format(timestep=time[timestep], time=time[-1]), flush=True)
            # get object
            obj = libpost.get_object(time_dirs[time_list.index(time[timestep])], name)
            # get properties
            ## a
            obj_a_pos_0 = libpost.get_object_properties(obj, ["a__pos_0"])
            obj_a_pos_1 = libpost.get_object_properties(obj, ["a__pos_1"])
            obj_a_pos_2 = libpost.get_object_properties(obj, ["a__pos_2"])
            a_pos = np.empty((obj_a_pos_0["value"].size, 3))
            a_pos[:, 0] = obj_a_pos_0["value"]
            a_pos[:, 1] = obj_a_pos_1["value"]
            a_pos[:, 2] = obj_a_pos_2["value"]
            obj_a_j_0_0 = libpost.get_object_properties(obj, ["a__j_0_0"])
            obj_a_j_0_1 = libpost.get_object_properties(obj, ["a__j_0_1"])
            obj_a_j_0_2 = libpost.get_object_properties(obj, ["a__j_0_2"])
            obj_a_j_1_0 = libpost.get_object_properties(obj, ["a__j_1_0"])
            obj_a_j_1_1 = libpost.get_object_properties(obj, ["a__j_1_1"])
            obj_a_j_1_2 = libpost.get_object_properties(obj, ["a__j_1_2"])
            obj_a_j_2_0 = libpost.get_object_properties(obj, ["a__j_2_0"])
            obj_a_j_2_1 = libpost.get_object_properties(obj, ["a__j_2_1"])
            obj_a_j_2_2 = libpost.get_object_properties(obj, ["a__j_2_2"])
            a_grads = np.empty((obj_a_j_0_0["value"].size, 3, 3))
            a_grads[:, 0, 0] = obj_a_j_0_0["value"]
            a_grads[:, 0, 1] = obj_a_j_0_1["value"]
            a_grads[:, 0, 2] = obj_a_j_0_2["value"]
            a_grads[:, 1, 0] = obj_a_j_1_0["value"]
            a_grads[:, 1, 1] = obj_a_j_1_1["value"]
            a_grads[:, 1, 2] = obj_a_j_1_2["value"]
            a_grads[:, 2, 0] = obj_a_j_2_0["value"]
            a_grads[:, 2, 1] = obj_a_j_2_1["value"]
            a_grads[:, 2, 2] = obj_a_j_2_2["value"]
            ## b
            obj_b_pos_0 = libpost.get_object_properties(obj, ["b__pos_0"])
            obj_b_pos_1 = libpost.get_object_properties(obj, ["b__pos_1"])
            obj_b_pos_2 = libpost.get_object_properties(obj, ["b__pos_2"])
            b_pos = np.empty((obj_b_pos_0["value"].size, 3))
            b_pos[:, 0] = obj_b_pos_0["value"]
            b_pos[:, 1] = obj_b_pos_1["value"]
            b_pos[:, 2] = obj_b_pos_2["value"]
            obj_b_j_0_0 = libpost.get_object_properties(obj, ["b__j_0_0"])
            obj_b_j_0_1 = libpost.get_object_properties(obj, ["b__j_0_1"])
            obj_b_j_0_2 = libpost.get_object_properties(obj, ["b__j_0_2"])
            obj_b_j_1_0 = libpost.get_object_properties(obj, ["b__j_1_0"])
            obj_b_j_1_1 = libpost.get_object_properties(obj, ["b__j_1_1"])
            obj_b_j_1_2 = libpost.get_object_properties(obj, ["b__j_1_2"])
            obj_b_j_2_0 = libpost.get_object_properties(obj, ["b__j_2_0"])
            obj_b_j_2_1 = libpost.get_object_properties(obj, ["b__j_2_1"])
            obj_b_j_2_2 = libpost.get_object_properties(obj, ["b__j_2_2"])
            b_grads = np.empty((obj_b_j_0_0["value"].size, 3, 3))
            b_grads[:, 0, 0] = obj_b_j_0_0["value"]
            b_grads[:, 0, 1] = obj_b_j_0_1["value"]
            b_grads[:, 0, 2] = obj_b_j_0_2["value"]
            b_grads[:, 1, 0] = obj_b_j_1_0["value"]
            b_grads[:, 1, 1] = obj_b_j_1_1["value"]
            b_grads[:, 1, 2] = obj_b_j_1_2["value"]
            b_grads[:, 2, 0] = obj_b_j_2_0["value"]
            b_grads[:, 2, 1] = obj_b_j_2_1["value"]
            b_grads[:, 2, 2] = obj_b_j_2_2["value"]
            # compute
            time_nb[timestep] = time[timestep]
            distance_nb[timestep, 0] = np.average(np.linalg.norm(b_pos - a_pos, axis=1))
            distance_nb[timestep, 1] = 1.96 * np.std(np.linalg.norm(b_pos - a_pos, axis=1))
            print("\tINFO: Processing time: {timestep}/{time} done.".format(timestep=time[timestep], time=time[-1]), flush=True)
        # save snapshot
        np.savetxt("distance__{name}.csv".format(name=name), np.column_stack((time_nb, distance_nb)), delimiter=",", header="time,average_distance,95CLI")
        print("INFO: Processing object: {name} done.".format(name=name), flush=True)

if __name__ == '__main__':
    args = parse()
    main(args.names, args.nb)
