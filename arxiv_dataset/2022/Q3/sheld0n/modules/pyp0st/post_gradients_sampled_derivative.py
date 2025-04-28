#!/usr/bin/env python3

# command line program
import argparse
# numpy
import numpy as np
# scipy
import scipy as sp
import scipy.stats
import scipy.linalg
# custom lib
import libpost

def parse():
    parser = argparse.ArgumentParser(description='Tracers post processing')
    parser.add_argument('names', nargs="*", help='specify the object names')
    return parser.parse_args()

def main(names):
    if not names:
        print("INFO: Reading objects...", flush=True)
        names = libpost.get_object_names()
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    #time = time[:10]
    print("INFO: Init...", flush=True)
    averaged_norm_measured_gradient_derivative = {}
    std_norm_measured_gradient_derivative = {}
    averaged_correlation_time = {}
    std_correlation_time = {}
    print("INFO: Running...", flush=True)
    for name in names:
        print("INFO: Computing {} flow sampled statistics...".format(name), flush=True)
        # init
        obj = libpost.get_object(time_dirs[time_list.index(0.0)], name)
        flow_gradients_0_0 = libpost.get_object_properties(obj, [".*__j_0_0"])["value"]
        flow_gradients_index_m_1 = np.empty((flow_gradients_0_0.size, 3, 3))
        flow_gradients_index_m_1[:, 0, 0] = flow_gradients_0_0
        del flow_gradients_0_0;
        flow_gradients_index_m_1[:, 0, 1] = libpost.get_object_properties(obj, [".*__j_0_1"])["value"]
        flow_gradients_index_m_1[:, 0, 2] = libpost.get_object_properties(obj, [".*__j_0_2"])["value"]
        flow_gradients_index_m_1[:, 1, 0] = libpost.get_object_properties(obj, [".*__j_1_0"])["value"]
        flow_gradients_index_m_1[:, 1, 1] = libpost.get_object_properties(obj, [".*__j_1_1"])["value"]
        flow_gradients_index_m_1[:, 1, 2] = libpost.get_object_properties(obj, [".*__j_1_2"])["value"]
        flow_gradients_index_m_1[:, 2, 0] = libpost.get_object_properties(obj, [".*__j_2_0"])["value"]
        flow_gradients_index_m_1[:, 2, 1] = libpost.get_object_properties(obj, [".*__j_2_1"])["value"]
        flow_gradients_index_m_1[:, 2, 2] = libpost.get_object_properties(obj, [".*__j_2_2"])["value"]
        # init
        norm_measured_gradient_derivative = np.zeros(flow_gradients_index_m_1.shape[0])
        correlation_time = np.zeros(flow_gradients_index_m_1.shape[0])
        correlation_time_number = np.zeros(flow_gradients_index_m_1.shape[0])
        # running
        for index in range(1, time.size):
            t = time[index]
            t_m = time[index-1]
            print("\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # extract gradients index
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            flow_gradients_0_0 = libpost.get_object_properties(obj, [".*__j_0_0"])["value"]
            flow_gradients_index = np.empty((flow_gradients_0_0.size, 3, 3))
            flow_gradients_index[:, 0, 0] = flow_gradients_0_0
            del flow_gradients_0_0;
            flow_gradients_index[:, 0, 1] = libpost.get_object_properties(obj, [".*__j_0_1"])["value"]
            flow_gradients_index[:, 0, 2] = libpost.get_object_properties(obj, [".*__j_0_2"])["value"]
            flow_gradients_index[:, 1, 0] = libpost.get_object_properties(obj, [".*__j_1_0"])["value"]
            flow_gradients_index[:, 1, 1] = libpost.get_object_properties(obj, [".*__j_1_1"])["value"]
            flow_gradients_index[:, 1, 2] = libpost.get_object_properties(obj, [".*__j_1_2"])["value"]
            flow_gradients_index[:, 2, 0] = libpost.get_object_properties(obj, [".*__j_2_0"])["value"]
            flow_gradients_index[:, 2, 1] = libpost.get_object_properties(obj, [".*__j_2_1"])["value"]
            flow_gradients_index[:, 2, 2] = libpost.get_object_properties(obj, [".*__j_2_2"])["value"]
            # compute statistics
            tmp_gradients = 0.5 * (flow_gradients_index + flow_gradients_index_m_1)
            tmp_norm_measured_gradient_derivative = np.linalg.norm(flow_gradients_index - flow_gradients_index_m_1, axis=(1,2)) / (t - t_m)
            norm_measured_gradient_derivative += tmp_norm_measured_gradient_derivative
            tmp_norm_measured_gradient_derivative[tmp_norm_measured_gradient_derivative == 0.0] = np.inf
            tmp_correlation_time = np.linalg.norm(tmp_gradients, axis=(1,2)) / tmp_norm_measured_gradient_derivative
            expected_performance = np.empty(tmp_correlation_time.size)
            for i in range(expected_performance.size):
                expected_performance[i] = (sp.linalg.expm(0.5 * (tmp_gradients[i, :, :] + tmp_gradients[i, :, :].T) * tmp_correlation_time[i]) - sp.linalg.expm(tmp_gradients[i, :, :] * tmp_correlation_time[i]))[0, 0]
            correlation_time += expected_performance * tmp_correlation_time
            # correlation_increment = np.zeros(tmp_norm_measured_gradient_derivative.size)
            # correlation_increment[tmp_norm_measured_gradient_derivative != np.inf] = 1.0
            # correlation_time_number += correlation_increment
            correlation_time_number += expected_performance
            # extract gradients index - 1
            flow_gradients_index_m_1[:, :, :] = flow_gradients_index
            # print
            print("\tINFO: Done.", flush=True)
        averaged_norm_measured_gradient_derivative[name] = np.average(norm_measured_gradient_derivative / (time.size - 1))
        std_norm_measured_gradient_derivative[name] = np.std(norm_measured_gradient_derivative / (time.size - 1)) / np.sqrt(norm_measured_gradient_derivative.size)
        averaged_correlation_time[name] = np.average(correlation_time / correlation_time_number)
        std_correlation_time[name] = np.std(correlation_time / correlation_time_number) / np.sqrt(norm_measured_gradient_derivative.size)
        print("INFO: Done...", flush=True)
    print("INFO: Merging objects...", flush=True)
    merged_objects = {}
    for object_name in names:
        reduced_object_name = object_name.split("__")[0]
        if not reduced_object_name in merged_objects:
            merged_objects[reduced_object_name] = {}
            merged_objects[reduced_object_name]["value"] = [np.array([averaged_norm_measured_gradient_derivative[object_name], 1.96 * std_norm_measured_gradient_derivative[object_name], averaged_correlation_time[object_name], 1.96 * std_correlation_time[object_name]] + list(libpost.get_properties_from_string(object_name).values()))]
            merged_objects[reduced_object_name]["info"] = ["averaged_norm_measured_gradient_derivative", "95CLI", "correlation_time", "95CLI"] + list(libpost.get_properties_from_string(object_name).keys())
        else:
            merged_objects[reduced_object_name]["value"].append(np.array([averaged_norm_measured_gradient_derivative[object_name], 1.96 * std_norm_measured_gradient_derivative[object_name], averaged_correlation_time[object_name], 1.96 * std_correlation_time[object_name]] + list(libpost.get_properties_from_string(object_name).values())))
    for reduced_object_name in merged_objects:
        merged_objects[reduced_object_name]["value"] = np.stack(merged_objects[reduced_object_name]["value"])
        for column in range(1, merged_objects[reduced_object_name]["value"].shape[1]):
            merged_objects[reduced_object_name]["value"] = merged_objects[reduced_object_name]["value"][np.argsort(merged_objects[reduced_object_name]["value"][:, column])]
    print("INFO: Done.", flush=True)
    print("INFO: Saving...", flush=True)
    for name in merged_objects:
        np.savetxt("{name}__merge_averaged_norm_measured_gradient_derivative.csv".format(name=name), merged_objects[name]["value"], delimiter=",", header=",".join(merged_objects[name]["info"]))
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main(args.names)
