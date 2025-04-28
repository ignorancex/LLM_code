#!/usr/bin/env python3

# command line program
import argparse
# numpy
import numpy as np
# custom lib
import libpost

def parse():
    parser = argparse.ArgumentParser(description='Flame post processing')
    parser.add_argument('name', help='specify the flame object name')
    parser.add_argument('timestep', type=int, help='specify the time step to post process')
    return parser.parse_args()

def main(name, timestep):
    time_dirs, time_list, time = libpost.get_time()
    obj = libpost.get_object(time_dirs[time_list.index(time[timestep])], name)
    # # get positions
    objects_pos_0 = libpost.get_object_properties(obj, ["s_.*__pos_0"])
    objects_pos_1 = libpost.get_object_properties(obj, ["s_.*__pos_1"])
    objects_pos_2 = libpost.get_object_properties(obj, ["s_.*__pos_2"])
    # sort stuff based on s
    # result = {}
    # for object_name in objects_pos:
        # for i in range(len(objects_pos[object_name]["info"])):
            # s = libpost.get_property_from_object_name(objects_pos[object_name]["info"][i], "s")
            # if s in result:
                # result[s].append({"info":objects_pos[object_name]["info"][i], "value":objects_pos[object_name]["value"][:, i]})
            # else:
                # result[s] = [{"info":objects_pos[object_name]["info"][i], "value":objects_pos[object_name]["value"][:, i]}]
    # # extract snapshot
    # for snapshot_id in range(len(time)):
        # snapshot = []
        # for s, data in result.items():
            # snapshot.append([s] +  [d["value"][snapshot_id] for d in data])
    # save snapshot
    np.savetxt("{name}__t_{time}.csv".format(name=name, time=str(time[timestep]).replace(".", "o")), np.column_stack((objects_pos_0["value"], objects_pos_1["value"], objects_pos_2["value"])), delimiter=",", header="pos_0,pos_1,pos_2")

if __name__ == '__main__':
    args = parse()
    main(args.name, args.timestep)
