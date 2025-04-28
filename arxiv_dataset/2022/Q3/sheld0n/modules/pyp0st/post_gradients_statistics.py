#!/usr/bin/env python3

# command line program
import argparse
# deepcopy
import copy
# numpy
import numpy as np
# internal modules
import libpost

DIRECTION = np.array([[1.0, 0.0, 0.0]]).T

def parse():
    parser = argparse.ArgumentParser(description='Computes statistics of the lagrangian gradients matrix (computed along particle trajectories)')
    parser.add_argument('--key', '-k', nargs='?', default="j", help='key name of the gradient in the data')
    return parser.parse_args()

def main(key):
    print("INFO: Post processing flow velocity gradients statistics sampled by lagrangian objects.", flush=True)
    print("INFO: Reading objects...", flush=True)
    object_names = libpost.get_object_names()
    print("INFO: Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    print("INFO: Done.", flush=True)
    # get gradient matrix
    print("INFO: Reading data...", flush=True)
    objects_data = {}
    for object_name in object_names:
        print("\tINFO: Processing object {name}...".format(name=object_name), flush=True)
        # get object
        obj = libpost.get_object(time_dirs[time_list.index(0.0)], object_name)
        obj__j_0_0 = libpost.get_object_properties(obj, ["particle_.*__j_0_0"])["value"]
        particle_number = obj__j_0_0.size
        # read gradient
        obj_gradient = np.zeros((particle_number, 3, 3))
        obj_gradient[:, 0, 0] = obj__j_0_0
        obj_gradient[:, 0, 1] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_0_1"])["value"]
        obj_gradient[:, 0, 2] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_0_2"])["value"]
        obj_gradient[:, 1, 0] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_1_0"])["value"]
        obj_gradient[:, 1, 1] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_1_1"])["value"]
        obj_gradient[:, 1, 2] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_1_2"])["value"]
        obj_gradient[:, 2, 0] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_2_0"])["value"]
        obj_gradient[:, 2, 1] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_2_1"])["value"]
        obj_gradient[:, 2, 2] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_2_2"])["value"]
        obj_skew_gradients = 0.5 * (obj_gradient - obj_gradient.transpose((0, 2, 1)))
        # init
        objects_data[object_name] = {}
        objects_data[object_name]["sqrt(tr(J.J))"] = np.zeros((time.size, particle_number))
        objects_data[object_name]["vorticity_norm"] = np.zeros((time.size, particle_number))
        # compute
        objects_data[object_name]["sqrt(tr(J.J))"][0, :] = np.trace(np.matmul(obj_gradient, obj_gradient, axes=[(-2, -1), (-2,-1), (-2,-1)]), axis1=1, axis2=2)
        objects_data[object_name]["vorticity_norm"][0, :] = 2.0 * np.sqrt(obj_skew_gradients[:, 1, 2]**2 + obj_skew_gradients[:, 0, 2]**2 + obj_skew_gradients[:, 0, 1]**2)
        # time
        for index in range(1, time.size):
            print("\tINFO: Processing time {}/{}...".format(time[index], time[-1]), flush=True)
            # read
            obj = libpost.get_object(time_dirs[time_list.index(time[index])], object_name)
            obj_gradient[:, 0, 0] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_0_0"])["value"]
            obj_gradient[:, 0, 1] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_0_1"])["value"]
            obj_gradient[:, 0, 2] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_0_2"])["value"]
            obj_gradient[:, 1, 0] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_1_0"])["value"]
            obj_gradient[:, 1, 1] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_1_1"])["value"]
            obj_gradient[:, 1, 2] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_1_2"])["value"]
            obj_gradient[:, 2, 0] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_2_0"])["value"]
            obj_gradient[:, 2, 1] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_2_1"])["value"]
            obj_gradient[:, 2, 2] = libpost.get_object_properties(obj, ["particle_.*__"+key+"_2_2"])["value"]
            obj_skew_gradients = 0.5 * (obj_gradient - obj_gradient.transpose((0, 2, 1)))
            # combute
            objects_data[object_name]["sqrt(tr(J.J))"][index, :] = np.trace(np.matmul(obj_gradient, obj_gradient, axes=[(-2, -1), (-2,-1), (-2,-1)]), axis1=1, axis2=2)
            objects_data[object_name]["vorticity_norm"][index, :] = 2.0 * np.sqrt(obj_skew_gradients[:, 1, 2]**2 + obj_skew_gradients[:, 0, 2]**2 + obj_skew_gradients[:, 0, 1]**2)
    print("INFO: Done.", flush=True)
    # compute fft
    objects_fft_data = {}
    print("INFO: Computing Fast Fourrier Transforms...", flush=True)
    for object_name in object_names:
        objects_fft_data[object_name] = {}
        for invariant in objects_data[object_name]:
            objects_fft_data[object_name][invariant] = np.abs(np.fft.rfft(objects_data[object_name][invariant], axis=0)) / objects_data[object_name][invariant].shape[0]
    # additional treatement
    for object_name in object_names:
        objects_fft_data[object_name]["sqrt(tr(J.J))"] = np.sqrt(objects_fft_data[object_name]["sqrt(tr(J.J))"])
    # average fft and 95CLI
    objects_average_fft_data = {}
    for object_name in object_names:
            objects_average_fft_data[object_name] = {}
            for invariant in objects_data[object_name]:
                objects_average_fft_data[object_name][invariant] = np.column_stack([np.average(objects_fft_data[object_name][invariant], axis=1), 1.96 * np.std(objects_fft_data[object_name][invariant], axis=1) / np.sqrt(objects_fft_data[object_name][invariant].shape[1])])
    print("INFO: Done.", flush=True)
    # save
    print("INFO: Saving Fast Fourrier Transforms...", flush=True)
    angular_frequency = 2 * np.pi / (time[-1] - time[0]) * np.linspace(1, len(time)//2+2, num=len(time)//2+1)
    for object_name in objects_average_fft_data:
        for invariant in objects_data[object_name]:
            np.savetxt("{name}__average_fft_{invariant}.csv".format(name=object_name, invariant=invariant), np.column_stack((angular_frequency, objects_average_fft_data[object_name][invariant][:, 0], objects_average_fft_data[object_name][invariant][:, 1])), delimiter=",", header="pulsation,average_fft,95CLI")
            print("The correlation time of the invariant {} corresponding to the object {} is {}".format(invariant, object_name, np.trapz(objects_average_fft_data[object_name][invariant][:, 0] * 2 * np.pi / angular_frequency, x=angular_frequency) / np.trapz(objects_average_fft_data[object_name][invariant][:, 0], x=angular_frequency)))
    print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main(args.key)
