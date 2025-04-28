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
    parser = argparse.ArgumentParser(description='Computes the average swimming power consumption.')
    return parser.parse_args()

def main():
    print("INFO: Post processing the average swimming power consumption.", flush=True)
    print("INFO: Reading objects...", flush=True)
    object_names = libpost.get_object_names()
    print("INFO: Done. Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    time = time[:-1]
    print("INFO: Done.", flush=True)
    print("INFO: Processing power...", flush=True)
    objects_average_swimming_power_consumption = {}
    for name in object_names:
        print("\tINFO: Processing object {name}...".format(name=name), flush=True)
        swimming_power_consumption = None
        for index in range(1, time.size):
            t = time[index]
            print("\t\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # read objects
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # swimming power consumption
            obj_us = libpost.get_object_properties(obj, ["particle_.*__us"])["value"]
            if swimming_power_consumption is None:
                swimming_power_consumption = np.zeros(obj_us.shape)
            swimming_power_consumption += np.power(obj_us, 2) * (t - time[index-1])
            print("\t\tINFO: Done processing t={t}/{t_f}.".format(t=t, t_f=time[-1]), flush=True)
        swimming_power_consumption /= time[-1]
        objects_average_swimming_power_consumption[name] = np.array((time[-1], np.average(swimming_power_consumption), 1.96 * np.std(swimming_power_consumption) / np.sqrt(swimming_power_consumption.size)))
        print("\tDone processing object {name}.".format(name=name), flush=True)
    print("INFO: Done.", flush=True)
    print("INFO: Merging objects...", flush=True)
    merged_objects = {}
    for object_name in objects_average_swimming_power_consumption:
        reduced_object_name = object_name.split("__")[0]
        if not reduced_object_name in merged_objects:
            merged_objects[reduced_object_name] = {}
            merged_objects[reduced_object_name]["value"] = [np.array(objects_average_swimming_power_consumption[object_name].tolist() + list(libpost.get_properties_from_string(object_name).values()))]
            merged_objects[reduced_object_name]["info"] = ["time", "us^2", "95CLI"] + list(libpost.get_properties_from_string(object_name).keys())
        else:
            merged_objects[reduced_object_name]["value"].append(np.array(objects_average_swimming_power_consumption[object_name].tolist() + list(libpost.get_properties_from_string(object_name).values())))
    for reduced_object_name in merged_objects:
        merged_objects[reduced_object_name]["value"] = np.stack(merged_objects[reduced_object_name]["value"])
        for column in range(3, merged_objects[reduced_object_name]["value"].shape[1]):
            merged_objects[reduced_object_name]["value"] = merged_objects[reduced_object_name]["value"][np.argsort(merged_objects[reduced_object_name]["value"][:, column])]
    print("INFO: Done.", flush=True)
    print("INFO: Saving...", flush=True)
    for name in merged_objects:
        np.savetxt("{name}__merge_swimming_power_consumption.csv".format(name=name), merged_objects[name]["value"], delimiter=",", header=",".join(merged_objects[name]["info"]))
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main()
