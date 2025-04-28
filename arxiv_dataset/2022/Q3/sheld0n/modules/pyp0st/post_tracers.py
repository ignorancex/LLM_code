#!/usr/bin/env python3

# command line program
import argparse
# numpy
import numpy as np
# scipy
import scipy as sp
import scipy.stats
# custom lib
import libpost

BINS_NB = 16
BINS_RANGE = (0.0, 1.0)

def parse():
    parser = argparse.ArgumentParser(description='Tracers post processing')
    parser.add_argument('name', help='specify the object name')
    parser.add_argument('--number-inits', '-i', nargs='?', type=int, default=100, help='number of time steps considered as initial time steps')
    parser.add_argument('--number-average', '-a', nargs='?', type=int, default=30, help='number of time steps for wich average quantities are computed')
    parser.add_argument('--number-profiles', '-p', nargs='?', type=int, default=5, help='number of time steps for wich profiles are computed')
    parser.add_argument('--flow-kinematic-viscosity', '-f', nargs='?', type=float, default=0.0, help='if non zero, enables flow statistics computation based on the value of kinematic viscosity specified, default: 0')
    return parser.parse_args()

def main(name, n_inits, n_average, n_profiles, flow_kinematic_viscosity):
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    time = time[:-1]
    init_time_steps = np.round(np.linspace(0, time.size-1, n_inits+1)[:-1]).astype(int)
    average_time_steps = np.round(np.linspace(0, time.size-1, n_average)).astype(int)
    profiles_time_steps = np.round(np.linspace(0, time.size-1, n_profiles)).astype(int)
    print("INFO: Done.", flush=True)
    if flow_kinematic_viscosity > 0.0:
        print("INFO: Computing flow statistics...", flush=True)
        dissipation_rate = np.empty(time.size)
        rms_flow_velocity = np.empty(time.size)
        rms_vorticity = np.empty(time.size)
        average_flow_velocity = np.empty(time.size)
        average_streching = np.empty(time.size)
        average_vorticity = np.empty(time.size)
        integral_length_scale = np.empty(time.size)
        for index in range(time.size):
            t = time[index]
            print("\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # read flow
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # extract velocity
            flow_velocity_0 = libpost.get_object_properties(obj, [".*__u_0"])
            flow_velocity_1 = libpost.get_object_properties(obj, [".*__u_1"])
            flow_velocity_2 = libpost.get_object_properties(obj, [".*__u_2"])
            flow_velocity = np.empty((flow_velocity_0["value"].size, 3))
            flow_velocity[:, 0] = flow_velocity_0["value"]
            flow_velocity[:, 1] = flow_velocity_1["value"]
            flow_velocity[:, 2] = flow_velocity_2["value"]
            del flow_velocity_0;
            del flow_velocity_1;
            del flow_velocity_2;
            # extract gradients
            flow_gradients_0_0 = libpost.get_object_properties(obj, [".*__j_0_0"])
            flow_gradients_0_1 = libpost.get_object_properties(obj, [".*__j_0_1"])
            flow_gradients_0_2 = libpost.get_object_properties(obj, [".*__j_0_2"])
            flow_gradients_1_0 = libpost.get_object_properties(obj, [".*__j_1_0"])
            flow_gradients_1_1 = libpost.get_object_properties(obj, [".*__j_1_1"])
            flow_gradients_1_2 = libpost.get_object_properties(obj, [".*__j_1_2"])
            flow_gradients_2_0 = libpost.get_object_properties(obj, [".*__j_2_0"])
            flow_gradients_2_1 = libpost.get_object_properties(obj, [".*__j_2_1"])
            flow_gradients_2_2 = libpost.get_object_properties(obj, [".*__j_2_2"])
            flow_gradients = np.empty((flow_gradients_0_0["value"].size, 3, 3))
            flow_gradients[:, 0, 0] = flow_gradients_0_0["value"]
            flow_gradients[:, 0, 1] = flow_gradients_0_1["value"]
            flow_gradients[:, 0, 2] = flow_gradients_0_2["value"]
            flow_gradients[:, 1, 0] = flow_gradients_1_0["value"]
            flow_gradients[:, 1, 1] = flow_gradients_1_1["value"]
            flow_gradients[:, 1, 2] = flow_gradients_1_2["value"]
            flow_gradients[:, 2, 0] = flow_gradients_2_0["value"]
            flow_gradients[:, 2, 1] = flow_gradients_2_1["value"]
            flow_gradients[:, 2, 2] = flow_gradients_2_2["value"]
            del flow_gradients_0_0;
            del flow_gradients_0_1;
            del flow_gradients_0_2;
            del flow_gradients_1_0;
            del flow_gradients_1_1;
            del flow_gradients_1_2;
            del flow_gradients_2_0;
            del flow_gradients_2_1;
            del flow_gradients_2_2;
            # more gradients
            sym_flow_gradients = 0.5 * (flow_gradients + flow_gradients.transpose((0, 2, 1)))
            skew_flow_gradients = 0.5 * (flow_gradients - flow_gradients.transpose((0, 2, 1)))
            # vorticity
            vorticity = np.empty((skew_flow_gradients.shape[0], 3))
            vorticity[:, 0] = 2.0 * -skew_flow_gradients[:, 1, 2]
            vorticity[:, 1] = 2.0 * skew_flow_gradients[:, 0, 2]
            vorticity[:, 2] = 2.0 * -skew_flow_gradients[:, 0, 1]
            # compute statistics
            dissipation_rate[index] = 2.0 * flow_kinematic_viscosity * np.average(np.sum(np.square(sym_flow_gradients), axis=(1, 2)))
            rms_flow_velocity[index] = (1.0/np.sqrt(3.0)) * np.average(np.sqrt(np.sum(np.square(flow_velocity), axis=1))) # CAREFUL: NO MEAN FLOW IS ASSUMED
            rms_vorticity[index] = (1.0/np.sqrt(3.0)) * np.average(np.sqrt(np.sum(np.square(vorticity), axis=1))) # CAREFUL: NO MEAN FLOW IS ASSUMED
            average_flow_velocity[index] = np.average(np.linalg.norm(flow_velocity, axis=1))
            average_streching[index] = np.average(np.linalg.norm(sym_flow_gradients, axis=(1, 2)))
            average_vorticity[index] = np.average(np.linalg.norm(vorticity, axis=1))
            # print
            print("\tINFO: Done.", flush=True)
        dissipation_rate = np.average(dissipation_rate)
        rms_flow_velocity = np.average(rms_flow_velocity)
        rms_vorticity = np.average(rms_vorticity)
        average_flow_velocity = np.average(average_flow_velocity)
        average_streching = np.average(average_streching)
        average_vorticity = np.average(average_vorticity)
        taylor_micro_scale = np.sqrt(15 * flow_kinematic_viscosity * np.square(rms_flow_velocity) / dissipation_rate)
        taylor_scale_reynolds = rms_flow_velocity * taylor_micro_scale / flow_kinematic_viscosity
        kolmogorov_time_scale = np.sqrt(flow_kinematic_viscosity / dissipation_rate)
        kolmogorov_length_scale = flow_kinematic_viscosity**(3.0/4.0) * dissipation_rate**(-1.0/4.0)
        print("INFO: Done...", flush=True)
        print("INFO: Saving...", flush=True)
        np.savetxt("{name}__flow_statistics.csv".format(name=name), np.column_stack([dissipation_rate, rms_flow_velocity, rms_vorticity, average_flow_velocity, average_streching, average_vorticity, taylor_micro_scale, taylor_scale_reynolds, kolmogorov_time_scale, kolmogorov_length_scale]), delimiter=",", header="dissipation_rate,rms_flow_velocity,rms_vorticity,average_flow_velocity,average_streching,average_vorticity,taylor_micro_scale,taylor_scale_reynolds,kolmogorov_time_scale,kolmogorov_length_scale")
        print("INFO: Done.", flush=True)
    print("INFO: Computing dispersion velocity as function of distance...", flush=True)
    distance_velocity = []
    for index_init in init_time_steps:
        t_init = time[index_init]
        print("\tINFO: Processing init...", flush=True)
        obj = libpost.get_object(time_dirs[time_list.index(t_init)], name)
        objects_pos_0 = libpost.get_object_properties(obj, ["particle_.*__pos_0"])
        objects_pos_1 = libpost.get_object_properties(obj, ["particle_.*__pos_1"])
        objects_pos_2 = libpost.get_object_properties(obj, ["particle_.*__pos_2"])
        objects_pos_init = np.empty((objects_pos_0["value"].size, 3))
        objects_pos_init[:, 0] = objects_pos_0["value"]
        objects_pos_init[:, 1] = objects_pos_1["value"]
        objects_pos_init[:, 2] = objects_pos_2["value"]
        # tmp init
        tmp_distance = np.empty((objects_pos_init.shape[0], time.size - index_init))
        tmp_velocity = np.empty((objects_pos_init.shape[0], time.size - index_init))
        print("\tINFO: Done.", flush=True)
        for index in range(index_init, time.size):
            t = time[index]
            print("\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # read object
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # extract distance
            objects_pos = np.empty((objects_pos_init.shape[0], 3))
            objects_pos[:, 0] = libpost.get_object_properties(obj, ["particle_.*__pos_0"])["value"]
            objects_pos[:, 1] = libpost.get_object_properties(obj, ["particle_.*__pos_1"])["value"]
            objects_pos[:, 2] = libpost.get_object_properties(obj, ["particle_.*__pos_2"])["value"]
            objects_r = objects_pos - objects_pos_init
            objects_distance = np.linalg.norm(objects_r, axis=1)
            # extract velocity
            objects_velocity = np.empty((objects_pos_init.shape[0], 3))
            objects_velocity[:, 0] = libpost.get_object_properties(obj, ["particle_.*__u_0"])["value"]
            objects_velocity[:, 1] = libpost.get_object_properties(obj, ["particle_.*__u_1"])["value"]
            objects_velocity[:, 2] = libpost.get_object_properties(obj, ["particle_.*__u_2"])["value"]
            # computes average distance
            tmp_distance[:, index - index_init] = objects_distance
            if index > index_init:
                objects_direction = (objects_r.transpose() / objects_distance).transpose()
                tmp_velocity[:, index - index_init] = np.sum(objects_direction * objects_velocity, axis=1)
            else:
                tmp_velocity[:, index - index_init] = np.linalg.norm(objects_velocity, axis=1)
            # print
            print("\tINFO: Done.", flush=True)
        tmp_distance_velocity = []
        for i in range(objects_pos_init.shape[0]):
            values, bins, nb = sp.stats.binned_statistic(np.array(tmp_distance[i, :]), np.array(tmp_velocity[i, :]), statistic="mean", bins=BINS_NB, range=BINS_RANGE)
            tmp_distance_velocity.append(values)
        tmp_distance_velocity = np.column_stack(tmp_distance_velocity)
        distance_velocity.append(np.nanmean(tmp_distance_velocity, axis=1))
    distance_velocity = np.nanmean(np.column_stack(distance_velocity), axis=1)
    bins = 0.5 * (bins[1:] + bins[:-1])
    print("INFO: Saving...", flush=True)
    np.savetxt("{name}__average_dispersion_velocity.csv".format(name=name), np.column_stack((bins, distance_velocity)), delimiter=",", header="distance,velocity")
    print("INFO: Done.", flush=True)
    # computes pdfs
    # print("INFO: Processing pdfs...", flush=True)
    # pdfs_header = []
    # pdfs = []
    # for index in range(profiles_time_steps.size):
        # t = time[profiles_time_steps[index]]
        # print("\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
        # # computes distance pdf
        # hist, bins = np.histogram(distance, bins=BINS_NB, density=True)
        # bins = 0.5 * (bins[1:] + bins[:-1])
        # # append
        # pdfs_header.append("distance__t_{t},pdf__t_{t}".format(t=t))
        # pdfs += [bins, hist]
        # # print
        # print("\tINFO: Done.", flush=True)
    # print("INFO: Done.", flush=True)
    # # save snapshot
    # print("INFO: Saving...", flush=True)
    # np.savetxt("{name}__distance_pdf.csv".format(name=name), np.column_stack(pdfs), delimiter=",", header=",".join(pdfs_header))
    # print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    main(args.name, args.number_inits, args.number_average, args.number_profiles, args.flow_kinematic_viscosity)
