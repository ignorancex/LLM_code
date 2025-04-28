#!/usr/bin/env python3

# command line program
import argparse
# deepcopy
import copy
# numpy
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.stats
# internal modules
import libpost

KOLMOGOROV_TIME_SCALE = 8.8e-2
TAU = 0.5 * KOLMOGOROV_TIME_SCALE
TARGET_DIRECTION = np.array([1.0, 0.0, 0.0]).reshape((3, 1))

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
    #time = time[::100]
    print("INFO: Done.", flush=True)
    print("INFO: Processing power...", flush=True)
    objects_average_swimming_power_consumption = {}
    for name in object_names:
        print("\tINFO: Processing object {name}...".format(name=name), flush=True)
        vorticity_z = []
        grad_x_y = []
        surfing_direction_x = []
        surfing_direction_y = []
        for index in range(0, time.size):
            t = time[index]
            print("\t\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # read objects
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # extract data
            flow_gradients_0_0 = libpost.get_object_properties(obj, [".*__j_0_0"])["value"]
            flow_gradients = np.empty((flow_gradients_0_0.size, 3, 3))
            flow_gradients[:, 0, 0] = flow_gradients_0_0
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
            skew_flow_gradients_horizontal = skew_flow_gradients.copy()
            skew_flow_gradients_horizontal[:, 1, 2] = 0.0
            skew_flow_gradients_horizontal[:, 2, 1] = 0.0
            # rotation power consumption
            surfing_direction = np.array([
                np.transpose(TAU * sp.linalg.expm(flow_gradients[k, :, :])) @ TARGET_DIRECTION
                for k in range(flow_gradients.shape[0])
            ])
            surfing_direction /= np.linalg.norm(surfing_direction)
            vorticity_z.append(vorticity[:, 2])
            grad_x_y.append(flow_gradients[:, 0, 1])
            surfing_direction_x.append(surfing_direction[:, 0])
            surfing_direction_y.append(surfing_direction[:, 1])
            print("\t\tINFO: Done processing t={t}/{t_f}.".format(t=t, t_f=time[-1]), flush=True)
        values_vort_z_dir_x, bins, bin_nb = sp.stats.binned_statistic(np.array(vorticity_z).flatten() * KOLMOGOROV_TIME_SCALE, np.array(surfing_direction_x).flatten(), "mean", bins=32, range=(-2,2))
        values_grad_x_y_dir_x, bins, bin_nb = sp.stats.binned_statistic(np.array(grad_x_y).flatten() * KOLMOGOROV_TIME_SCALE, np.array(surfing_direction_x).flatten(), "mean", bins=32, range=(-2,2))
        values_vort_z_dir_y, bins, bin_nb = sp.stats.binned_statistic(np.array(vorticity_z).flatten() * KOLMOGOROV_TIME_SCALE, np.array(surfing_direction_y).flatten(), "mean", bins=32, range=(-2,2))
        values_grad_x_y_dir_y, bins, bin_nb = sp.stats.binned_statistic(np.array(grad_x_y).flatten() * KOLMOGOROV_TIME_SCALE, np.array(surfing_direction_y).flatten(), "mean", bins=32, range=(-2,2))
        np.savetxt("{}__vorticity_z__n_x__tau_{}.csv".format(name, TAU/KOLMOGOROV_TIME_SCALE), np.column_stack((0.5 * (bins[1:] + bins[:-1]), values_vort_z_dir_x)), delimiter=",", header="omega_z*kolmogorov_time,n_x")
        np.savetxt("{}__duxdy__n_x__tau_{}.csv".format(name, TAU/KOLMOGOROV_TIME_SCALE), np.column_stack((0.5 * (bins[1:] + bins[:-1]), values_grad_x_y_dir_x)), delimiter=",", header="duxdy*kolmogorov_time,n_x")
        np.savetxt("{}__vorticity_z__n_y__tau_{}.csv".format(name, TAU/KOLMOGOROV_TIME_SCALE), np.column_stack((0.5 * (bins[1:] + bins[:-1]), values_vort_z_dir_y)), delimiter=",", header="omega_z*kolmogorov_time,n_y")
        np.savetxt("{}__duxdy__n_y__tau_{}.csv".format(name, TAU/KOLMOGOROV_TIME_SCALE), np.column_stack((0.5 * (bins[1:] + bins[:-1]), values_grad_x_y_dir_y)), delimiter=",", header="duxdy*kolmogorov_time,n_y")
        np.savetxt("{}__vorticity_z__angle_z_tau_{}.csv".format(name, TAU/KOLMOGOROV_TIME_SCALE), np.column_stack((0.5 * (bins[1:] + bins[:-1]), np.arctan2(values_vort_z_dir_y, values_vort_z_dir_x))), delimiter=",", header="omega_z*kolmogorov_time,angle_z")
        np.savetxt("{}__duxdy__angle_z_tau_{}.csv".format(name, TAU/KOLMOGOROV_TIME_SCALE), np.column_stack((0.5 * (bins[1:] + bins[:-1]), np.arctan2(values_grad_x_y_dir_y, values_grad_x_y_dir_x))), delimiter=",", header="omega_z*kolmogorov_time,angle_z")
        print("\tDone processing object {name}.".format(name=name), flush=True)
    print("INFO: Done.", flush=True)
    # print("INFO: Merging objects...", flush=True)
    # merged_objects = {}
    # for object_name in objects_average_swimming_power_consumption:
        # reduced_object_name = object_name.split("__")[0]
        # if not reduced_object_name in merged_objects:
            # merged_objects[reduced_object_name] = {}
            # merged_objects[reduced_object_name]["value"] = [np.array(objects_average_swimming_power_consumption[object_name].tolist() + list(libpost.get_properties_from_string(object_name).values()))]
            # merged_objects[reduced_object_name]["info"] = ["time", "omega_active^2", "95CLI"] + list(libpost.get_properties_from_string(object_name).keys())
        # else:
            # merged_objects[reduced_object_name]["value"].append(np.array(objects_average_swimming_power_consumption[object_name].tolist() + list(libpost.get_properties_from_string(object_name).values())))
    # for reduced_object_name in merged_objects:
        # merged_objects[reduced_object_name]["value"] = np.stack(merged_objects[reduced_object_name]["value"])
        # for column in range(3, merged_objects[reduced_object_name]["value"].shape[1]):
            # merged_objects[reduced_object_name]["value"] = merged_objects[reduced_object_name]["value"][np.argsort(merged_objects[reduced_object_name]["value"][:, column])]
    # print("INFO: Done.", flush=True)
    # print("INFO: Saving...", flush=True)
    # for name in merged_objects:
        # np.savetxt("{name}__merge_rotating_power_consumption.csv".format(name=name), merged_objects[name]["value"], delimiter=",", header=",".join(merged_objects[name]["info"]))
    # print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main()
