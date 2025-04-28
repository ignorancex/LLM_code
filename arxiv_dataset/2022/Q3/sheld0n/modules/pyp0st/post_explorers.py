#!/usr/bin/env python3

# command line program
import argparse
# deepcopy
import copy
# numpy
import numpy as np
import scipy as sp
import scipy.optimize
# internal modules
import libpost

def fit_func(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3

def parse():
    parser = argparse.ArgumentParser(description='Computes the average distance from original position over time.')
    parser.add_argument('--number-average', '-n', nargs='?', type=int, default=30, help='number of time steps for wich average quantities are computed, default: 30')
    return parser.parse_args()

def fit_func(t, d):
    return np.sqrt(6 * d * t)

def main(n_average):
    print("INFO: Post processing the dispersion.", flush=True)
    print("INFO: Reading objects...", flush=True)
    object_names = libpost.get_object_names()
    print("INFO: Done. Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    average_time_steps = np.round(np.linspace(0, time.size-1, n_average)).astype(int)
    print("INFO: Done.", flush=True)
    print("INFO: Processing init...", flush=True)
    objects_pos_init = {}
    for name in object_names:
        obj = libpost.get_object(time_dirs[time_list.index(0.0)], name)
        pos_0 = libpost.get_object_properties(obj, ["particle_.*__pos_0"])["value"]
        objects_pos_init[name] = np.empty((pos_0.size, 3))
        objects_pos_init[name][:, 0] = pos_0
        objects_pos_init[name][:, 1] = libpost.get_object_properties(obj, ["particle_.*__pos_1"])["value"]
        objects_pos_init[name][:, 2] = libpost.get_object_properties(obj, ["particle_.*__pos_2"])["value"]
    print("INFO: Done.", flush=True)
    print("INFO: Processing dispersion...", flush=True)
    objects_standard_deviation = {}
    objects_distance = {}
    for name in object_names:
        print("\tINFO: Processing object {name}...".format(name=name), flush=True)
        objects_standard_deviation[name] = np.empty((average_time_steps.shape[0], 3))
        objects_distance[name] = np.empty((average_time_steps.shape[0]))
        for index in range(average_time_steps.size):
            t = time[average_time_steps[index]]
            print("\t\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # read objects
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # effective velocity
            pos = np.empty(objects_pos_init[name].shape)
            pos[:, 0] = libpost.get_object_properties(obj, ["particle_.*__pos_0"])["value"]
            pos[:, 1] = libpost.get_object_properties(obj, ["particle_.*__pos_1"])["value"]
            pos[:, 2] = libpost.get_object_properties(obj, ["particle_.*__pos_2"])["value"]
            objects_standard_deviation[name][index, :] = np.std(pos - objects_pos_init[name], axis=0)
            objects_distance[name][index] = np.average(np.linalg.norm(pos - objects_pos_init[name], axis=1))
            print("\t\tINFO: Done processing t={t}/{t_f}.".format(t=t, t_f=time[-1]), flush=True)
        print("\tDone processing object {name}.".format(name=name), flush=True)
    print("INFO: Done.", flush=True)
    print("INFO: Saving...", flush=True)
    for name in object_names:
        np.savetxt("{name}__standard_deviation.csv".format(name=name), np.column_stack((time[average_time_steps], objects_standard_deviation[name])), delimiter=",", header="time,std_x,std_y,std_z")
        np.savetxt("{name}__distance.csv".format(name=name), np.column_stack((time[average_time_steps], objects_distance[name])), delimiter=",", header="time,distance")
    print("INFO: Done.", flush=True)
    print("INFO: Computing diffusion coefficients...", flush=True)
    merged_objects = {}
    for object_name in objects_standard_deviation:
        # compute diffusion coefficient by fitting
        diffusion_coefficient_x, cov = sp.optimize.curve_fit(fit_func, time[average_time_steps], objects_standard_deviation[object_name][:, 0])
        diffusion_coefficient_y, cov = sp.optimize.curve_fit(fit_func, time[average_time_steps], objects_standard_deviation[object_name][:, 1])
        diffusion_coefficient_z, cov = sp.optimize.curve_fit(fit_func, time[average_time_steps], objects_standard_deviation[object_name][:, 2])
        # merge
        reduced_object_name = object_name.split("__")[0]
        if not reduced_object_name in merged_objects:
            merged_objects[reduced_object_name] = {}
            merged_objects[reduced_object_name]["value"] = [np.array(list(libpost.get_properties_from_string(object_name).values()) + list(diffusion_coefficient_x) + list(diffusion_coefficient_y) + list(diffusion_coefficient_z))]
            merged_objects[reduced_object_name]["info"] = list(libpost.get_properties_from_string(object_name).keys()) + ["diffusion_coefficient_x,diffusion_coefficient_y,diffusion_coefficient_z"]
        else:
            merged_objects[reduced_object_name]["value"].append(np.array(list(libpost.get_properties_from_string(object_name).values()) + list(diffusion_coefficient_x) + list(diffusion_coefficient_y) + list(diffusion_coefficient_z)))
    for reduced_object_name in merged_objects:
        merged_objects[reduced_object_name]["value"] = np.stack(merged_objects[reduced_object_name]["value"])
        for column in range(merged_objects[reduced_object_name]["value"].shape[1] - 3):
            merged_objects[reduced_object_name]["value"] = merged_objects[reduced_object_name]["value"][np.argsort(merged_objects[reduced_object_name]["value"][:, column])]
    print("INFO: Done.", flush=True)
    print("INFO: Saving...", flush=True)
    for name in merged_objects:
        np.savetxt("{name}__diffusion_coefficient.csv".format(name=name), merged_objects[name]["value"], delimiter=",", header=",".join(merged_objects[name]["info"]))
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main(args.number_average)
