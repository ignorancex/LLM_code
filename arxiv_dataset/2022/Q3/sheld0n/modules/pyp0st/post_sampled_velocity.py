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

def parse():
    parser = argparse.ArgumentParser(description='Computes the average sampled flow velocity.')
    return parser.parse_args()

def main():
    print("INFO: Post processing of the flow velocity sampled by lagrangian objects.", flush=True)
    print("INFO: Reading objects...", flush=True)
    object_names = libpost.get_object_names()
    #object_names = [name for name in object_names if not "riser" in name]
    print("INFO: Done. Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    # time = time[::1000]
    print("INFO: Done.", flush=True)
    print("INFO: Processing sampled velocity...", flush=True)
    objects_average_sampled_flow_velocity = {}
    for name in object_names:
        print("\tINFO: Processing object {name}...".format(name=name), flush=True)
        # read for init
        obj = libpost.get_object(time_dirs[time_list.index(0.0)], name)
        obj_u_0 = libpost.get_object_properties(obj, ["particle_.*__u_0"])["value"]
        # then init
        objects_average_sampled_flow_velocity[name] = np.zeros((obj_u_0.size, 3))
        # compute
        for index in range(0, time.size):
            t = time[index]
            print("\t\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # read objects
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # effective velocity
            objects_average_sampled_flow_velocity[name][:, 0] += libpost.get_object_properties(obj, ["particle_.*__u_0"])["value"]
            objects_average_sampled_flow_velocity[name][:, 1] += libpost.get_object_properties(obj, ["particle_.*__u_1"])["value"]
            objects_average_sampled_flow_velocity[name][:, 2] += libpost.get_object_properties(obj, ["particle_.*__u_2"])["value"]
            print("\t\tINFO: Done processing t={t}/{t_f}.".format(t=t, t_f=time[-1]), flush=True)
        print("\tDone processing object {name}.".format(name=name), flush=True)
        objects_average_sampled_flow_velocity[name] /= time.size
        objects_average_sampled_flow_velocity[name] = np.array((
            time[-1], 
            np.average(objects_average_sampled_flow_velocity[name][:, 0]), 1.96 * np.std(objects_average_sampled_flow_velocity[name][:, 0]) / np.sqrt(objects_average_sampled_flow_velocity[name].shape[0]),
            np.average(objects_average_sampled_flow_velocity[name][:, 1]), 1.96 * np.std(objects_average_sampled_flow_velocity[name][:, 1]) / np.sqrt(objects_average_sampled_flow_velocity[name].shape[0]),
            np.average(objects_average_sampled_flow_velocity[name][:, 2]), 1.96 * np.std(objects_average_sampled_flow_velocity[name][:, 2]) / np.sqrt(objects_average_sampled_flow_velocity[name].shape[0]),
        ))
    print("INFO: Done.", flush=True)
    print("INFO: averaging...", flush=True)
    for name in object_names:
        np.savetxt("{name}__average_sampled_flow_velocity.csv".format(name=name), objects_average_sampled_flow_velocity[name], delimiter=",", header="time,u_0,95CLI,u_1,95CLI,u_2,95CLI")
    print("INFO: Done.", flush=True)
    print("INFO: Merging objects...", flush=True)
    merged_objects = {}
    for object_name in objects_average_sampled_flow_velocity:
        reduced_object_name = object_name.split("__")[0]
        if not reduced_object_name in merged_objects:
            merged_objects[reduced_object_name] = {}
            merged_objects[reduced_object_name]["value"] = [np.array(list(objects_average_sampled_flow_velocity[object_name]) + list(libpost.get_properties_from_string(object_name).values()))]
            merged_objects[reduced_object_name]["info"] = ["time", "u_0", "95CLI", "u_1", "95CLI", "u_2", "95CLI"] + list(libpost.get_properties_from_string(object_name).keys())
        else:
            merged_objects[reduced_object_name]["value"].append(np.array(list(objects_average_sampled_flow_velocity[object_name]) + list(libpost.get_properties_from_string(object_name).values())))
    for reduced_object_name in merged_objects:
        merged_objects[reduced_object_name]["value"] = np.stack(merged_objects[reduced_object_name]["value"])
        for column in range(7, merged_objects[reduced_object_name]["value"].shape[1]):
            merged_objects[reduced_object_name]["value"] = merged_objects[reduced_object_name]["value"][np.argsort(merged_objects[reduced_object_name]["value"][:, column])]
    print("INFO: Done.", flush=True)
    print("INFO: Saving...", flush=True)
    for name in merged_objects:
        np.savetxt("{name}__merge_average_sampled_flow_velocity.csv".format(name=name), merged_objects[name]["value"], delimiter=",", header=",".join(merged_objects[name]["info"]))
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main()
