#!/usr/bin/env python3

# command line program
import argparse
# numpy
import numpy as np
# scipy
import scipy as sp
import scipy.stats
# custom lib
import libpost

BINS_NB = 512
BINS_RANGE = (0.0, 2.0)

def parse():
    parser = argparse.ArgumentParser(description='Testing stuff')
    parser.add_argument('name', help='specify the object name')
    return parser.parse_args()

def main(name):
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    time = time[0:10]
    print("INFO: Done.", flush=True)
    print("INFO: Processing velocity as function of distance...", flush=True)
    print("\tINFO: Processing init...", flush=True)
    obj = libpost.get_object(time_dirs[time_list.index(time[0])], name)
    objects_pos_0 = libpost.get_object_properties(obj, ["particle_.*__pos_0"])
    objects_pos_1 = libpost.get_object_properties(obj, ["particle_.*__pos_1"])
    objects_pos_2 = libpost.get_object_properties(obj, ["particle_.*__pos_2"])
    objects_pos_init = np.empty((objects_pos_0["value"].size, 3))
    objects_pos_init[:, 0] = objects_pos_0["value"]
    objects_pos_init[:, 1] = objects_pos_1["value"]
    objects_pos_init[:, 2] = objects_pos_2["value"]
    # test
    objects_u_0 = libpost.get_object_properties(obj, ["particle_.*__u_0"])
    objects_u_1 = libpost.get_object_properties(obj, ["particle_.*__u_1"])
    objects_u_2 = libpost.get_object_properties(obj, ["particle_.*__u_2"])
    objects_u = np.empty((objects_u_0["value"].size, 3))
    objects_u[:, 0] = objects_u_0["value"]
    objects_u[:, 1] = objects_u_1["value"]
    objects_u[:, 2] = objects_u_2["value"]
    print(np.average(np.linalg.norm(objects_u, axis=1)))
    print("\tINFO: Done.", flush=True)
    distance = []
    for index in range(time.size):
        t = time[index]
        print("\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
        # read object
        obj = libpost.get_object(time_dirs[time_list.index(t)], name)
        # extract distance
        objects_pos = np.empty((objects_pos_init.shape[0], 3))
        objects_pos[:, 0] = libpost.get_object_properties(obj, ["particle_.*__pos_0"])["value"]
        objects_pos[:, 1] = libpost.get_object_properties(obj, ["particle_.*__pos_1"])["value"]
        objects_pos[:, 2] = libpost.get_object_properties(obj, ["particle_.*__pos_2"])["value"]
        objects_distance = np.linalg.norm(objects_pos - objects_pos_init, axis=1)
        # computes average distance
        distance.append(objects_distance)
        # print
        print("\tINFO: Done.", flush=True)
    distance = np.stack(distance)
    velocity = (np.diff(distance, axis=0).transpose() / np.diff(time)).transpose()
    print(np.average(distance[0, :]))
    print(np.average(velocity[0, :]))
    # end test

if __name__ == '__main__':
    args = parse()
    main(args.name)
