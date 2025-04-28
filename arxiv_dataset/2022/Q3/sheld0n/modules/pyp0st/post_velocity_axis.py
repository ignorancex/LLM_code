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
    parser = argparse.ArgumentParser(description='Computes the average velocity along a specific direction')
    parser.add_argument('axis', nargs='?', type=int, default=0, help='specify the axis')
    parser.add_argument('--sorting-property', '-s', nargs='?', default="surftimeconst", help='sorting property, default: surftimeconst')
    parser.add_argument('--fit-max', '-m', nargs='?', type=float, default=0.0, help='maximum interval value for the fit, 0 automatic selection, default: 0')
    parser.add_argument('--number-average', '-n', nargs='?', type=int, default=30, help='number of time steps for wich average quantities are computed, default: 30')
    parser.add_argument('--reverse', '-r', action='store_true', help='consider negative axis')
    return parser.parse_args()

def main(axis, sorting_property, fit_max, n_average, reverse):
    print("INFO: Post processing the effective velocity of lagrangian objects along the axis {axis}.".format(axis=axis), flush=True)
    print("INFO: Reading objects...", flush=True)
    object_names = libpost.get_object_names()
    #object_names = [name for name in object_names if not "riser" in name]
    print("INFO: Done. Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    time = time[:-1]
    average_time_steps = np.round(np.linspace(0, time.size-1, n_average)).astype(int)
    print("INFO: Done.", flush=True)
    print("INFO: Processing init...", flush=True)
    objects_pos_init = {}
    for name in object_names:
        obj = libpost.get_object(time_dirs[time_list.index(0.0)], name)
        objects_pos_init[name] = libpost.get_object_properties(obj, ["particle_.*__pos_{axis}".format(axis=axis)])["value"]
    print("INFO: Done.", flush=True)
    print("INFO: Processing effective velocity along axis...", flush=True)
    objects_average_effective_velocity = {}
    for name in object_names:
        print("\tINFO: Processing object {name}...".format(name=name), flush=True)
        objects_average_effective_velocity[name] = np.empty((average_time_steps.shape[0], 3))
        objects_average_effective_velocity[name][0, :] = (0.0, 0.0, 0.0)
        for index in range(1, average_time_steps.size):
            t = time[average_time_steps[index]]
            print("\t\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # read objects
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # effective velocity
            obj_pos = libpost.get_object_properties(obj, ["particle_.*__pos_{axis}".format(axis=axis)])["value"]
            if reverse:
                obj_effective_velocity = -(obj_pos - objects_pos_init[name]) / t
            else:
                obj_effective_velocity = (obj_pos - objects_pos_init[name]) / t
            objects_average_effective_velocity[name][index, :] = [t, np.average(obj_effective_velocity), 1.96 * np.std(obj_effective_velocity) / np.sqrt(obj_effective_velocity.size)]
            print("\t\tINFO: Done processing t={t}/{t_f}.".format(t=t, t_f=time[-1]), flush=True)
        print("\tDone processing object {name}.".format(name=name), flush=True)
    print("INFO: Done.", flush=True)
    print("INFO: Saving...", flush=True)
    for name in object_names:
        np.savetxt("{name}__average_velocity_axis_{axis}.csv".format(name=name, axis=axis), objects_average_effective_velocity[name], delimiter=",", header="time,average_velocity_axis,95CLI")
    print("INFO: Done.", flush=True)
    print("INFO: Merging objects...", flush=True)
    merged_objects = {}
    for object_name in objects_average_effective_velocity:
        reduced_object_name = object_name.split("__")[0]
        if not reduced_object_name in merged_objects:
            merged_objects[reduced_object_name] = {}
            merged_objects[reduced_object_name]["value"] = [np.array(list(objects_average_effective_velocity[object_name][-1, :]) + list(libpost.get_properties_from_string(object_name).values()))]
            merged_objects[reduced_object_name]["info"] = ["time", "average_velocity_axis_{axis}_t_inf".format(axis=axis), "95CLI"] + list(libpost.get_properties_from_string(object_name).keys())
        else:
            merged_objects[reduced_object_name]["value"].append(np.array(list(objects_average_effective_velocity[object_name][-1, :]) + list(libpost.get_properties_from_string(object_name).values())))
    for reduced_object_name in merged_objects:
        merged_objects[reduced_object_name]["value"] = np.stack(merged_objects[reduced_object_name]["value"])
        for column in range(3, merged_objects[reduced_object_name]["value"].shape[1]):
            merged_objects[reduced_object_name]["value"] = merged_objects[reduced_object_name]["value"][np.argsort(merged_objects[reduced_object_name]["value"][:, column])]
    print("INFO: Done.", flush=True)
    print("INFO: Saving...", flush=True)
    for name in merged_objects:
        np.savetxt("{name}__merge_average_velocity_axis_{axis}.csv".format(name=name, axis=axis), merged_objects[name]["value"], delimiter=",", header=",".join(merged_objects[name]["info"]))
    print("INFO: Done.", flush=True)
    print("INFO: Sorting and fitting...", flush=True)
    max_objects = {}
    fits_objects = {object_name:{} for object_name in merged_objects}
    fits_max_objects = {object_name:{} for object_name in merged_objects}
    for object_name in merged_objects:
        object_sorted = {}
        sorting_index = merged_objects[reduced_object_name]["info"].index(sorting_property)
        for row in merged_objects[reduced_object_name]["value"]:
            row_list = row.tolist()
            row_list.pop(sorting_index)
            row_tuple = tuple(row_list[3:])
            if not row_tuple in object_sorted:
                object_sorted[row_tuple] = [row]
            else:
                object_sorted[row_tuple].append(row)
        for row_tuple in object_sorted:
            object_sorted[row_tuple] = np.stack(object_sorted[row_tuple])
        # max
        max_objects[object_name] = {"value":[], "info":merged_objects[object_name]["info"]}
        for row_tuple in object_sorted:
            max_objects[object_name]["value"].append(object_sorted[row_tuple][np.argmax(object_sorted[row_tuple][:, 1]), :])
        max_objects[object_name]["value"] = np.stack(max_objects[object_name]["value"])
        # info
        fits_objects[object_name]["info"] = [sorting_property]
        fits_max_objects[object_name]["info"] = merged_objects[object_name]["info"][3:] + ["max_surftimeconst", "max_average_velocity_axis"]
        fits_max_objects[object_name]["info"].pop(sorting_index - 3)
        # value
        if fit_max == 0.0:
            fit_max = merged_objects[object_name]["value"][:, sorting_index].max()
        fits_objects[object_name]["value"] = [np.linspace(0.0, fit_max, num=50)]
        fits_max_objects[object_name]["value"] = []
        for row_tuple in object_sorted:
            # value
            fit, cov = sp.optimize.curve_fit(fit_func, object_sorted[row_tuple][:, sorting_index], object_sorted[row_tuple][:, 1])
            fits_objects[object_name]["info"] += ["average_velocity_axis_{axis}_t_inf__params_{params}".format(axis=axis, params="_".join([str(tmp) for tmp in row_tuple]).replace(".", "o"))]
            fits_objects[object_name]["value"] += [fit_func(fits_objects[object_name]["value"][0], *fit)]
            # max
            index_max = np.argmax(fits_objects[object_name]["value"][-1])
            fits_max_objects[object_name]["value"] += [[*row_tuple, fits_objects[object_name]["value"][0][index_max], fits_objects[object_name]["value"][-1][index_max]]]
        fits_objects[object_name]["value"] = np.column_stack(fits_objects[object_name]["value"])
        fits_max_objects[object_name]["value"] = np.stack(fits_max_objects[object_name]["value"])
    print("INFO: Done.", flush=True)
    # save
    print("INFO: Saving...", flush=True)
    for name in merged_objects:
        np.savetxt("{name}__max_average_velocity_axis_{axis}.csv".format(name=name, axis=axis), max_objects[name]["value"], delimiter=",", header=",".join(max_objects[name]["info"]))
    for name in fits_objects:
        np.savetxt("{name}__fits_average_velocity_axis_{axis}.csv".format(name=name, axis=axis), fits_objects[name]["value"], delimiter=",", header=",".join(fits_objects[name]["info"]))
    for name in fits_max_objects:
        np.savetxt("{name}__fits_max_average_velocity_axis_{axis}.csv".format(name=name, axis=axis), fits_max_objects[name]["value"], delimiter=",", header=",".join(fits_max_objects[name]["info"]))
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main(args.axis, args.sorting_property, args.fit_max, args.number_average, args.reverse)
