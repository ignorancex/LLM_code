#!/usr/bin/env python3

# command line program
import argparse
# deepcopy
import copy
# numpy
import numpy as np
# internal modules
import libpost

def parse():
    parser = argparse.ArgumentParser(description='Computes the average distance')
    return parser.parse_args()

def main():
    # get positions
    time = libpost.get_time();
    objects_distance = libpost.get_objects_properties(["distance"])
    # average distance and 95CLI
    objects_average_distance = copy.deepcopy(objects_distance)
    for object_name in objects_distance:
        # value
        objects_average_distance[object_name]["value"] = np.column_stack([np.average(objects_distance[object_name]["value"], axis=1), 1.96 * np.std(objects_distance[object_name]["value"], axis=1) / np.sqrt(objects_distance[object_name]["value"].shape[1])])
        # info
        objects_average_distance[object_name]["info"] = ["average_distance", "95CLI"]
    # save
    libpost.savet(time, objects_average_distance, "average_distance")
    # average velocity and 95CLI
    objects_average_velocity = copy.deepcopy(objects_average_distance)
    for object_name in objects_distance:
        # value
        objects_average_distance[object_name]["value"] = np.column_stack([-(objects_distance[object_name]["value"][1:, 0] - objects_distance[object_name]["value"][:-1, 0]) / (time[1:] - time[:-1]), objects_distance[object_name]["value"][1:, 1] / (time[1:] - time[:-1])])
        # info
        objects_average_distance[object_name]["info"] = ["average_velocity", "95CLI"]
    # save
    libpost.savet(time[:-1], objects_average_distance, "average_velocity")

if __name__ == '__main__':
    args = parse()
    main()
