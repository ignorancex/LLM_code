#!/usr/bin/env python3

# command line program
import argparse
# numpy
import numpy as np
# custom lib
import libpost

BINS_NB = 100

def parse():
    parser = argparse.ArgumentParser(description='Tracker comparison post processing')
    parser.add_argument('--object-names', '-o', nargs='+', help='specify the object names to perform the post processing on')
    parser.add_argument('--arrival-distance', '-d', type=float, default=0.0, help='arrival distance')
    parser.add_argument('--number-average', '-n', nargs='?', type=int, default=30, help='number of time steps for wich average quantities are computed')
    parser.add_argument('--number-profiles', '-p', nargs='?', type=int, default=5, help='number of time steps for wich profiles are computed')
    return parser.parse_args()

def main(object_names, arrival_distance, n_average, n_profiles):
    if not object_names:
        print("INFO: Reading objects...", flush=True)
        object_names = libpost.get_object_names()
        print("INFO: Done. Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    average_time_steps = np.round(np.linspace(0, time.size-1, n_average)).astype(int)
    profiles_time_steps = np.round(np.linspace(0, time.size-1, n_profiles)).astype(int)
    print("INFO: Done.", flush=True)
    for name in object_names:
        if arrival_distance > 0.0:
            print("INFO: Processing arrival time...", flush=True)
            arrival_number_time = np.empty(average_time_steps.size)
            average_arrival_time = np.empty(average_time_steps.size)
            cli_arrival_time = np.empty(average_time_steps.size)
            arrival_times = []
            mask = None
            for t_index in range(time.size):
                t = time[t_index]
                print("\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
                # read object
                obj = libpost.get_object(time_dirs[time_list.index(t)], name)
                # extract distance
                objects_distance = libpost.get_object_properties(obj, ["particle_.*__distance"])
                # computes arrival time
                ## compute mask
                if mask is None:
                    mask = objects_distance["value"] < arrival_distance
                else:
                    mask = np.logical_or(mask, objects_distance["value"] < arrival_distance)
                ## append result
                arrival_number = np.sum(mask)
                print("\tINFO: Number of arrivals: {}".format(arrival_number), flush=True)
                if arrival_number > len(arrival_times):
                    arrival_times.extend([t] * (arrival_number - len(arrival_times)))
                    if arrival_number == objects_distance["value"].size:
                        print("\tINFO: All particles arrived.", flush=True)
                        print("\tINFO: Done.", flush=True)
                        break
                # compute average
                if t_index in average_time_steps:
                    arrival_number_time[np.where(average_time_steps == t_index)] = arrival_number
                    average_arrival_time[np.where(average_time_steps == t_index)] = 0.0
                    if (len(arrival_times) > 0.0):
                        average_arrival_time[np.where(average_time_steps == t_index)] = np.average(arrival_times)
                        cli_arrival_time[np.where(average_time_steps == t_index)] = 1.96 * np.std(arrival_times) / np.sqrt(len(arrival_times))
                    else:
                        cli_arrival_time[np.where(average_time_steps == t_index)] = 0.0
                # print
                print("\tINFO: Done.", flush=True)
            print("INFO: Done.", flush=True)
        print("INFO: Processing average distance...", flush=True)
        average_distance = np.empty(average_time_steps.size)
        cli_distance = np.empty(average_time_steps.size)
        for index in range(average_time_steps.size):
            t = time[average_time_steps[index]]
            print("\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # read object
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # extract distance
            objects_distance = libpost.get_object_properties(obj, ["particle_.*__distance"])
            # computes average distance
            average_distance[index] = np.average(objects_distance["value"])
            cli_distance[index] = 1.96 * np.std(objects_distance["value"]) / np.sqrt(objects_distance["value"].size)
            # print
            print("\tINFO: Done.", flush=True)
        print("INFO: Done.", flush=True)
        # computes average velocity
        print("INFO: Computing average velocity...", flush=True)
        average_effective_velocity = np.empty(average_time_steps.size)
        average_effective_velocity[0] = 0.0
        average_effective_velocity[1:] = -(average_distance[1:] - average_distance[0]) / time[average_time_steps][1:]
        cli_effective_velocity = np.empty(average_time_steps.size)
        cli_effective_velocity[0] = 0.0
        cli_effective_velocity[1:] = cli_distance[1:] / time[average_time_steps][1:]
        print("INFO: Done.", flush=True)
        # computes pdfs
        print("INFO: Processing pdfs...", flush=True)
        pdfs_header = []
        pdfs = []
        for index in range(profiles_time_steps.size):
            t = time[profiles_time_steps[index]]
            print("\tINFO: Processing t={t}/{t_f}...".format(t=t, t_f=time[-1]), flush=True)
            # read object
            obj = libpost.get_object(time_dirs[time_list.index(t)], name)
            # extract distance
            objects_distance = libpost.get_object_properties(obj, ["particle_.*__distance"])
            # computes average distance
            hist, bins = np.histogram(objects_distance["value"], bins=BINS_NB, density=True)
            bins = 0.5 * (bins[1:] + bins[:-1])
            # append
            pdfs_header.append("distance__t_{t},pdf__t_{t}".format(t=t))
            pdfs += [bins, hist]
            # print
            print("\tINFO: Done.", flush=True)
        print("INFO: Done.", flush=True)
        # save snapshot
        print("INFO: Saving...", flush=True)
        if arrival_distance > 0.0:
            np.savetxt("{name}__average_arrival_time__distance_{distance}.csv".format(name=name, distance=str(arrival_distance).replace(".", "o")), np.column_stack((time[average_time_steps], average_arrival_time, cli_arrival_time)), delimiter=",", header="time,average_arrival_time,95CLI")
            np.savetxt("{name}__arrival_number__distance_{distance}.csv".format(name=name, distance=str(arrival_distance).replace(".", "o")), np.column_stack((time[average_time_steps], arrival_number_time)), delimiter=",", header="time,arrival_number")
        np.savetxt("{name}__average_distance.csv".format(name=name), np.column_stack((time[average_time_steps], average_distance, cli_distance, average_effective_velocity, cli_effective_velocity)), delimiter=",", header="time,average_distance,95CLI,average_effective_velocity,95CLI")
        np.savetxt("{name}__distance_pdf.csv".format(name=name), np.column_stack(pdfs), delimiter=",", header=",".join(pdfs_header))
        print("INFO: Done.", flush=True)

if __name__ == '__main__':
    args = parse()
    #main(args.name, args.distance, args.distance_number, args.pdf_number)
    main(args.object_names, args.arrival_distance, args.number_average, args.number_profiles)
