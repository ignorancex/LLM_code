#!/usr/bin/env python3

# command line program
import argparse
import copy
import numpy as np
import scipy as sp
import scipy.integrate
# internal modules
import libpost

objects_name_info = ["us/u_eta", "surftimeconst/t_eta", "reorientationtime/t_eta", "surftimeprefactor", "proportion", "swimnoise", "dirnoise", "jnoise", "omegamax * t_eta", "reacttime/t_eta"]
objects_name_properties = ["us", "surftimeconst", "reorientationtime", "surftimeprefactor", "proportion", "swimnoise", "dirnoise", "jnoise", "omegamax", "reacttime"]

# filtering
for prop in ["surftimeconst", "swimnoise", "dirnoise", "jnoise", "omegamax", "reacttime"]:
    i = objects_name_properties.index(prop)
    objects_name_info.pop(i)
    objects_name_properties.pop(i)

# core
def parse():
    parser = argparse.ArgumentParser(description='Computes particles swimming velocity statistics.')
    return parser.parse_args()

def main():
    print("INFO: Post processing swimming velocity statistics of lagrangian objects.", flush=True)
    object_names = libpost.get_object_names()
    print("INFO: Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time = libpost.get_time()
    print("INFO: Done.", flush=True)
    # get axis
    print("INFO: Reading objects properties...", flush=True)
    objects_us = libpost.get_objects_properties(["us"])
    objects_pos = libpost.get_objects_properties(["pos_0"])
    print("INFO: Done.", flush=True)
    print("INFO: Computing average.", flush=True)
    objects_us_average = {object_name:{"value":np.array([np.average(objects_us[object_name]["value"])]), "info":["average_us"]} for object_name in objects_us}
    print("INFO: Done.", flush=True)
    print("INFO: Saving...", flush=True)
    libpost.save(objects_us_average, "average_us")
    print("INFO: Done.", flush=True)
    print("INFO: Computing E_perf...", flush=True)
    merged_objects = {}
    for object_name in objects_us:
        reduced_object_name = object_name.split("__")[0]
        if not reduced_object_name in merged_objects:
            merged_objects[reduced_object_name] = {}
            merged_objects[reduced_object_name]["value"] = []
            merged_objects[reduced_object_name]["info"] = ["average_E_perf/c_d", "95CLI"] + objects_name_info
        e_eff = 2.0 * libpost.get_property_from_object_name(object_name, "us") * 0.066 * (objects_pos[object_name]["value"][-1, :] - objects_pos[object_name]["value"][0, :]) - np.trapz(objects_us[object_name]["value"]**2, x=time, axis=0)
        merged_objects[reduced_object_name]["value"].append(
            np.array(
                [np.average(e_eff), 1.96 * np.std(e_eff) / np.sqrt(e_eff.size)] + [libpost.get_property_from_object_name(object_name, prop) for prop in objects_name_properties]
            )
        )
    for reduced_object_name in merged_objects:
            merged_objects[reduced_object_name]["value"] = np.stack(merged_objects[reduced_object_name]["value"])
    print("INFO: Done.", flush=True)
    print("INFO: Saving...", flush=True)
    libpost.save(merged_objects, "merge_average_E_perf")
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main()
