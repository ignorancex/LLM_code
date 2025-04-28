#!/usr/bin/env python3

# command line program
import argparse
# deepcopy
import copy
# numpy
import numpy as np
# internal modules
import libpost
# plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## OUTPUT
OUTPUT = "{name}__animation.mp4"
LIM = 4.0 * np.pi

def parse():
    parser = argparse.ArgumentParser(description='Chain post processing')
    parser.add_argument('--plot', '-p', action='store_true', help='plot each frame')
    parser.add_argument('--save', '-s', action='store_true', help='save each frame as a csv')
    return parser.parse_args()

anim_pos_0 = []
anim_pos_1 = []
mesh_pos_0 = None
mesh_pos_1 = None
anim_u_0 = []
anim_u_1 = []

def main(plot, save):
    global anim_pos_0, anim_pos_1, mesh_pos_0, mesh_pos_1, anim_u_0, anim_u_1
    print("INFO: Post processing chain.", flush=True)
    print("INFO: Reading objects...", flush=True)
    object_names = libpost.get_object_names()
    print("INFO: Done. Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    print("INFO: Done.", flush=True)
    print("INFO: Reading mesh...", flush=True)
    mesh = libpost.get_mesh()
    mesh_pos_0 = libpost.get_object_properties(mesh, [".*__pos_0"])["value"]
    mesh_pos_1 = libpost.get_object_properties(mesh, [".*__pos_1"])["value"]
    print("INFO: Done.", flush=True)
    print("INFO: Processing chain...", flush=True)
    for name in object_names:
        print("\tINFO: Processing object {name}...".format(name=name), flush=True)
        anim_pos_0 = []
        anim_pos_1 = []
        anim_u_0 = []
        anim_u_1 = []
        for index in range(0, time.size):
            t = time[index]
            print("\t\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            print("\t\t\tINFO: Chain post processing...")
            # read objects
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # effective velocity
            obj_pos_0 = libpost.get_object_properties(obj, ["l_.*__pos_0"])
            print("\t\t\t\tINFO: Chain sorting...", flush=True)
            sorting_scalar = {}
            for i in range(len(obj_pos_0["info"])):
                managed_index = int(obj_pos_0["info"][i].split("__")[1].split("_")[1])
                if not managed_index in sorting_scalar:
                    sorting_scalar[managed_index] = np.empty(obj_pos_0["value"].size)
                    sorting_scalar[managed_index][:]= np.nan
                sorting_scalar[managed_index][i] = float(obj_pos_0["info"][i].split("__")[2].split("_")[1].replace("o", "."))
            sorting_index = {managed_index:np.argsort(sorting_scalar[managed_index]) for managed_index in sorting_scalar}
            sorting_index = {managed_index:sorting_index[managed_index][np.logical_not(np.isnan(sorting_scalar[managed_index][sorting_index[managed_index]]))] for managed_index in sorting_scalar}
            obj_pos_1 = libpost.get_object_properties(obj, ["l_.*__pos_1"])
            obj_pos_2 = libpost.get_object_properties(obj, ["l_.*__pos_2"])
            obj_axis_0 = libpost.get_object_properties(obj, ["l_.*__axis_0"])
            obj_axis_1 = libpost.get_object_properties(obj, ["l_.*__axis_1"])
            obj_axis_2 = libpost.get_object_properties(obj, ["l_.*__axis_2"])
            obj_scalar = libpost.get_object_properties(obj, ["l_.*__scal"])
            
            print("\t\t\t\tINFO: Done.", flush=True)
            managed_anim_pos_0 = []
            managed_anim_pos_1 = []
            for key in sorting_index:
                managed_obj_pos_0 = obj_pos_0["value"][sorting_index[key]]
                managed_anim_pos_0.append(managed_obj_pos_0)
                managed_obj_pos_1 = obj_pos_1["value"][sorting_index[key]]
                managed_anim_pos_1.append(managed_obj_pos_1)
                managed_obj_axis_0 = obj_axis_0["value"][sorting_index[key]]
                managed_obj_axis_1 = obj_axis_1["value"][sorting_index[key]]
                managed_obj_axis_2 = obj_axis_2["value"][sorting_index[key]]
                managed_obj_pos_2 = obj_pos_2["value"][sorting_index[key]]
                managed_obj_scalar = obj_scalar["value"][sorting_index[key]]
                if save:
                    print("\t\t\t\tINFO: Saving...", flush=True)
                    np.savetxt("{name}__managed_{index}__time_{time}.csv".format(name=name, index=key, time=str(t).replace(".", "o")), np.column_stack((managed_obj_pos_0, managed_obj_pos_1, managed_obj_pos_2, managed_obj_scalar)), delimiter=",", header="x,y,z,scalar")
                    print("\t\t\t\tINFO: Done.", flush=True)
                if plot:
                    plt.plot(managed_obj_pos_0, managed_obj_pos_1, color=(1.0, 0.75, 0.75))
                    plt.quiver(managed_obj_pos_0, managed_obj_pos_1, managed_obj_axis_0, managed_obj_axis_1, color=(1.0, 0.75, 0.75))
            anim_pos_0.append(managed_anim_pos_0)
            anim_pos_1.append(managed_anim_pos_1)
            print("\t\t\tDone.")
            print("\t\t\tINFO: Flow post processing...")
            flow = libpost.get_flow(time_dirs[time_list.index(t)])
            flow_u_0 = libpost.get_object_properties(flow, [".*__u_0"])["value"]
            flow_u_1 = libpost.get_object_properties(flow, [".*__u_1"])["value"]
            anim_u_0.append(flow_u_0)
            anim_u_1.append(flow_u_1)
            # extract velocity
            print("\t\t\tDone.")
            if plot:
                plt.show()
            print("\t\tINFO: Done processing t={t}/{t_f}.".format(t=t, t_f=time[-1]), flush=True)
        print("\t\tINFO: Creating animation...", flush=True)
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=time.size, interval=30)
        anim.save(OUTPUT.format(name=name))
        print("\t\tINFO: Done.", flush=True)
        print("\tDone processing object {name}.".format(name=name), flush=True)
    print("INFO: Done.", flush=True)

def animate(i):
    global anim_pos_0, anim_pos_1, mesh_pos_0, mesh_pos_1, anim_u_0, anim_u_1
    plt.clf()
    plt.xlim(-LIM, LIM)
    plt.ylim(-LIM, LIM)
    plt.gca().set_aspect('equal', 'box')
    plt.quiver(mesh_pos_0, mesh_pos_1, anim_u_0[i], anim_u_1[i], color=(0.75, 0.75, 1.0))
    for j in range(len(anim_pos_0[i])):
        if anim_pos_0[i][j].size and anim_pos_1[i][j].size:
            plt.plot(anim_pos_0[i][j], anim_pos_1[i][j], color=(1.0, 0.75, 0.75))

if __name__ == '__main__':
    args = parse()
    main(args.plot, args.save)
