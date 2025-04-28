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
    parser = argparse.ArgumentParser(description='Computes statistics of the lagrangian flow velocity (computed along particle trajectories)')
    return parser.parse_args()

def main():
    print("INFO: Post processing flow velocity statistics sampled by lagrangian objects.")
    object_names = libpost.get_object_names()
    print("INFO: Object names are:", " ".join(object_names)[:-1])
    # get gradient matrix
    print("INFO: Reading objects properties...")
    time = libpost.get_time();
    objects_pos_0 = libpost.get_objects_properties(["pos_0"], object_names)
    objects_pos_1 = libpost.get_objects_properties(["pos_1"], object_names)
    objects_pos_2 = libpost.get_objects_properties(["pos_2"], object_names)
    objects_u_0 = libpost.get_objects_properties(["u_0"], object_names)
    objects_u_1 = libpost.get_objects_properties(["u_1"], object_names)
    objects_u_2 = libpost.get_objects_properties(["u_2"], object_names)
    print("INFO: Done.")
    print("INFO: Gradients reconstruction...")
    velocity_value = {}
    position_value = {}
    for object_name in objects_u_0:
        # flow velocity
        velocity_value[object_name] = np.empty((objects_u_0[object_name]["value"].shape[0], objects_u_0[object_name]["value"].shape[1], 3))
        velocity_value[object_name][:, :, 0] = objects_u_0[object_name]["value"]
        velocity_value[object_name][:, :, 1] = objects_u_1[object_name]["value"]
        velocity_value[object_name][:, :, 2] = objects_u_2[object_name]["value"]
        # position
        position_value[object_name] = np.empty((objects_pos_0[object_name]["value"].shape[0], objects_pos_0[object_name]["value"].shape[1], 3))
        position_value[object_name][:, :, 0] = objects_pos_0[object_name]["value"]
        position_value[object_name][:, :, 1] = objects_pos_1[object_name]["value"]
        position_value[object_name][:, :, 2] = objects_pos_2[object_name]["value"]
    print("INFO: Done.")
    print("INFO: Cleaning...")
    del objects_pos_0
    del objects_pos_1
    del objects_pos_2
    del objects_u_0
    del objects_u_1
    del objects_u_2
    print("INFO: Done.")
    print("INFO: Saving flow velocity statistics...")
    for object_name in velocity_value:
        np.savetxt("flow_velocity_statistics__"+object_name+".csv", np.array([
            np.average(np.linalg.norm(velocity_value[object_name], axis=2)),
            1.96 * np.std(np.linalg.norm(velocity_value[object_name], axis=2)) / np.sqrt(velocity_value[object_name].shape[0] * velocity_value[object_name].shape[1]),
        ]).reshape(1, 2), header="<|u|>, 95CLI", delimiter=",")
    print("INFO: Done.")
    print("INFO: Computing velocity error...")
    objects_velocity_error = {object_name:{"value":np.empty((velocity_value[object_name].shape[0], 2)), "info":["<|u_t - u_0|>", "95CLI"]} for object_name in velocity_value}
    objects_position_error = {object_name:{"value":np.empty((position_value[object_name].shape[0], 2)), "info":["<|X_t - X_0|>", "95CLI"]} for object_name in position_value}
    for object_name in velocity_value:
        print("INFO:    Processing " + object_name + "...")
        # velocity
        objects_velocity_error[object_name]["value"][0, 0] = 0.0
        objects_velocity_error[object_name]["value"][0, 1] = 0.0
        # position
        objects_position_error[object_name]["value"][0, 0] = 0.0
        objects_position_error[object_name]["value"][0, 1] = 0.0
        for k in range(1, velocity_value[object_name].shape[0]):
            # velocity error
            tmp = np.array([
                np.linalg.norm(velocity_value[object_name][:-k, :, :] - velocity_value[object_name][k:, :, :], axis=2),
                np.linalg.norm(velocity_value[object_name][k:, :, :] - velocity_value[object_name][:-k, :, :], axis=2)
            ])
            objects_velocity_error[object_name]["value"][k, 0] = np.average(tmp)
            objects_velocity_error[object_name]["value"][k, 1] = 1.96 * np.std(tmp) / np.sqrt(tmp.size)
            # position error
            tmp = np.array([
                np.linalg.norm(position_value[object_name][:-k, :, :] - position_value[object_name][k:, :, :], axis=2),
                np.linalg.norm(position_value[object_name][k:, :, :] - position_value[object_name][:-k, :, :], axis=2)
            ])
            objects_position_error[object_name]["value"][k, 0] = np.average(tmp)
            objects_position_error[object_name]["value"][k, 1] = 1.96 * np.std(tmp) / np.sqrt(tmp.size)
        print("INFO:    " + object_name + " done.")
    print("INFO: Done.")
    # save
    print("INFO: Saving...")
    libpost.savet(time, objects_velocity_error, "average_velocity_error")
    libpost.savet(time, objects_position_error, "average_position_error")
    print("INFO: Done.")

if __name__ == '__main__':
    args = parse()
    main()
