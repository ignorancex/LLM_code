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
    parser = argparse.ArgumentParser(description='Computes the average active rotation power consumption.')
    return parser.parse_args()

def main():
    print("INFO: Post processing the average rotating power consumption.", flush=True)
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
        t = 0.0
        obj = libpost.get_object(time_dirs[time_list.index(t)], name)
        obj_dir_0 = libpost.get_object_properties(obj, ["particle_.*__dir_0"])["value"]
        obj_dir = np.empty((obj_dir_0.shape[0], 3))
        obj_dir[:, 0] = obj_dir_0
        obj_dir[:, 1] = libpost.get_object_properties(obj, ["particle_.*__dir_1"])["value"]
        obj_dir[:, 2] = libpost.get_object_properties(obj, ["particle_.*__dir_2"])["value"]
        swimming_power_consumption = np.zeros(obj_dir_0.shape)
        for index in range(1, time.size):
            previous_t = t
            t = time[index]
            print("\t\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # read objects
            previous_obj = obj
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # extract data
            previous_obj_dir = obj_dir
            obj_dir = np.empty((previous_obj_dir.shape[0], 3))
            obj_dir[:, 0] = libpost.get_object_properties(obj, ["particle_.*__dir_0"])["value"]
            obj_dir[:, 1] = libpost.get_object_properties(obj, ["particle_.*__dir_1"])["value"]
            obj_dir[:, 2] = libpost.get_object_properties(obj, ["particle_.*__dir_2"])["value"]
            flow_gradients = np.empty((previous_obj_dir.shape[0], 3, 3))
            flow_gradients[:, 0, 0] = libpost.get_object_properties(obj, [".*__j_0_0"])["value"]
            flow_gradients[:, 0, 1] = libpost.get_object_properties(obj, [".*__j_0_1"])["value"]
            flow_gradients[:, 0, 2] = libpost.get_object_properties(obj, [".*__j_0_2"])["value"]
            flow_gradients[:, 1, 0] = libpost.get_object_properties(obj, [".*__j_1_0"])["value"]
            flow_gradients[:, 1, 1] = libpost.get_object_properties(obj, [".*__j_1_1"])["value"]
            flow_gradients[:, 1, 2] = libpost.get_object_properties(obj, [".*__j_1_2"])["value"]
            flow_gradients[:, 2, 0] = libpost.get_object_properties(obj, [".*__j_2_0"])["value"]
            flow_gradients[:, 2, 1] = libpost.get_object_properties(obj, [".*__j_2_1"])["value"]
            flow_gradients[:, 2, 2] = libpost.get_object_properties(obj, [".*__j_2_2"])["value"]
            skew_flow_gradients = 0.5 * (flow_gradients - flow_gradients.transpose((0, 2, 1)))
            vorticity = np.empty((skew_flow_gradients.shape[0], 3))
            vorticity[:, 0] = 2.0 * -skew_flow_gradients[:, 1, 2]
            vorticity[:, 1] = 2.0 * skew_flow_gradients[:, 0, 2]
            vorticity[:, 2] = 2.0 * -skew_flow_gradients[:, 0, 1]
            # rotation power consumption
            axis = np.cross(previous_obj_dir, obj_dir, axis=1)
            mask = np.linalg.norm(axis, axis=1) > 0.0
            if np.sum(mask) > 0:
                axis[mask, :] = axis[mask, :] / np.linalg.norm(axis[mask, :], axis=1).reshape((obj_dir_0.shape[0], 1))
                angular_velocity = np.arccos(np.clip(np.sum(previous_obj_dir * obj_dir, axis=1), -1.0, 1.0)).reshape((obj_dir_0.shape[0], 1)) * axis / (t - previous_t)
                swimming_power_consumption += np.power(np.linalg.norm(angular_velocity - 0.5 * vorticity, axis=1), 2) * (t - previous_t)
            else:
                swimming_power_consumption += np.power(np.linalg.norm(0.5 * vorticity, axis=1), 2) * (t - previous_t)
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
            merged_objects[reduced_object_name]["info"] = ["time", "omega_active^2", "95CLI"] + list(libpost.get_properties_from_string(object_name).keys())
        else:
            merged_objects[reduced_object_name]["value"].append(np.array(objects_average_swimming_power_consumption[object_name].tolist() + list(libpost.get_properties_from_string(object_name).values())))
    for reduced_object_name in merged_objects:
        merged_objects[reduced_object_name]["value"] = np.stack(merged_objects[reduced_object_name]["value"])
        for column in range(3, merged_objects[reduced_object_name]["value"].shape[1]):
            merged_objects[reduced_object_name]["value"] = merged_objects[reduced_object_name]["value"][np.argsort(merged_objects[reduced_object_name]["value"][:, column])]
    print("INFO: Done.", flush=True)
    print("INFO: Saving...", flush=True)
    for name in merged_objects:
        np.savetxt("{name}__merge_rotating_power_consumption.csv".format(name=name), merged_objects[name]["value"], delimiter=",", header=",".join(merged_objects[name]["info"]))
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main()
