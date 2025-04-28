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

BIN_RANGE = (-1.0, 1.0)
BIN_NB = 12
BINS = np.logspace(-3.0, 0.0, num=BIN_NB+1)

SORTING_PARAMETER_STR = "surftimeconst"
#SORTING_PARAMETER_STR = "surftimeprefactor"

def parse():
    parser = argparse.ArgumentParser(description='Computes particle concentration along an axis')
    parser.add_argument('axis', nargs='?', type=int, default=0, help='specify the axis to process')
    parser.add_argument('-n', '--negative', action='store_true', help='consider axis as negative')
    return parser.parse_args()

def fit_func(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3

def main(axis, negative):
    print("INFO: Post processing the effective velocity of lagrangian objects along the axis {axis}.".format(axis=axis), flush=True)
    print("INFO: Reading objects...", flush=True)
    object_names = libpost.get_object_names()
    print("INFO: Done. Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    print("INFO: Done.", flush=True)
    print("INFO: Computing binned velocity...", flush=True)
    velocity_max = {} 
    parameter_max = {}
    for object_name in object_names:
        selecting_parameter = libpost.get_properties_from_string(object_name)["us"]
        if not selecting_parameter in velocity_max:
            velocity_max[selecting_parameter] = np.full(BIN_NB, -np.inf) 
            parameter_max[selecting_parameter] = np.full(BIN_NB, np.nan) 
        print("\tINFO:Processing object: {}".format(object_name), flush=True)
        # init
        obj = libpost.get_object(time_dirs[time_list.index(0.0)], object_name)
        previous_pos_1 = libpost.get_object_properties(obj, ["particle_.*__pos_1"])["value"]
        previous_pos_axis = libpost.get_object_properties(obj, ["particle_.*__pos_{axis}".format(axis=axis)])["value"]
        # compute binned veocity
        velocity_axis = np.zeros((BIN_NB, 3))
        for index in range(1, time.size):
            t = time[index]
            # read object
            obj = libpost.get_object(time_dirs[time_list.index(t)], object_name)
            obj_pos_1 = libpost.get_object_properties(obj, ["particle_.*__pos_1"])["value"]
            obj_pos_axis = libpost.get_object_properties(obj, ["particle_.*__pos_{axis}".format(axis=axis)])["value"]
            # modulo
            mod_pos_value = np.remainder(1.0 + 0.5 * (previous_pos_1 + obj_pos_1), 2.0) - 1.0
            # symmetry
            mask = (mod_pos_value > 0.0)
            mod_pos_value[mask] = -mod_pos_value[mask]
            mod_pos_value += 1.0
            # compute velocity
            if negative:
                velocity_axis_value = -(obj_pos_axis - previous_pos_axis) / (time[index] - time[index - 1])
            else:
                velocity_axis_value = (obj_pos_axis - previous_pos_axis) / (time[index] - time[index - 1])
            for i in range(BIN_NB):
                y_min = BINS[i]
                y_max = BINS[i+1]
                velocity_axis[i, 0] = 0.5 * (y_min + y_max)
                velocity_axis[i, 1] += np.sum(velocity_axis_value[np.logical_and(mod_pos_value > y_min, mod_pos_value < y_max)])
                velocity_axis[i, 2] += np.sum(np.logical_and(mod_pos_value > y_min, mod_pos_value < y_max))
            # update previous
            previous_pos_1 = obj_pos_1
            previous_pos_axis = obj_pos_axis
        print("\tINFO: Saving...", flush=True)
        velocity_axis = np.column_stack((velocity_axis[:, 0], velocity_axis[:, 1] / velocity_axis[:, 2]))
        np.savetxt("{name}__binned_velocity_axis_{axis}.csv".format(name=object_name, axis=axis), velocity_axis, delimiter=",", header="pos_1,average_velocity_axis")
        print("\tINFO: Done.", flush=True)
        print("\tINFO: Checking max...", flush=True)
        sorting_parameter = libpost.get_properties_from_string(object_name)[SORTING_PARAMETER_STR]
        for i in range(BIN_NB):
            if velocity_axis[i,1] > velocity_max[selecting_parameter][i]:
                velocity_max[selecting_parameter][i] = velocity_axis[i,1]
                parameter_max[selecting_parameter][i] = sorting_parameter
        print("\tINFO: Done.", flush=True)
    print("\tINFO: Saving...", flush=True)
    for selecting_parameter in velocity_max:
        np.savetxt("surfer__us_{selecting_parameter}__max_binned_velocity_axis_{axis}.csv".format(selecting_parameter=str(selecting_parameter).replace(".", "o"), axis=axis), np.column_stack((0.5 * (BINS[1:] + BINS[:-1]), velocity_max[selecting_parameter], parameter_max[selecting_parameter])), delimiter=",", header="pos_1,max_average_velocity_axis,parameter")
    print("\tINFO: Done.", flush=True)
    
if __name__ == '__main__':
    args = parse()
    main(args.axis, args.negative)
