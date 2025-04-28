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

def parse():
    parser = argparse.ArgumentParser(description='Computes particle concentration along an axis')
    parser.add_argument('nb', nargs='?', type=int, default=4, help='number of times to process')
    return parser.parse_args()

def fit_func(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3

def main(nb):
    axis = 1
    print("INFO: Post processing the concentration of lagrangian objects along the axis {axis}.".format(axis=axis), flush=True)
    print("INFO: Reading objects...", flush=True)
    object_names = libpost.get_object_names()
    print("INFO: Done. Object names are:", " ".join(object_names), flush=True)
    print("INFO: Reading time...", flush=True)
    time_dirs, time_list, time = libpost.get_time()
    print("INFO: Done.", flush=True)
    print("INFO: Computing concentration pdfs...", flush=True)
    for i in range(nb):
        print("\tINFO: Processing {}/{}...".format(i+1, nb), flush=True)
        index = int(round(float(i / (nb - 1.0)) * (len(time) - 1)))
        t = time[index]
        for object_name in object_names:
            print("\t\tINFO:Processing object: {}".format(object_name), flush=True)
            # read object
            obj = libpost.get_object(time_dirs[time_list.index(t)], object_name)
            obj_pos = libpost.get_object_properties(obj, ["particle_.*__pos_{axis}".format(axis=axis)])["value"]
            # modulo
            mod_pos_value = np.remainder(1.0 + obj_pos, 2.0) - 1.0
            # symmetry
            mask = (mod_pos_value > 0.0)
            mod_pos_value[mask] = -mod_pos_value[mask]
            mod_pos_value += 1.0
            # compute concentration
            hist, bin_edges = np.histogram(mod_pos_value, bins=BINS, range=BIN_RANGE, density=True)
            bins = (bin_edges[1:] + bin_edges[:-1])/2
            # info
            print("\t\tINFO: Saving...", flush=True)
            np.savetxt("{name}__concentration_axis_{axis}__t_{time}.csv".format(name=object_name, axis=axis, time=str(t).replace(".", "o")), np.column_stack((bins, hist)), delimiter=",", header="pos_1,concentration")
            print("\t\tINFO: Done.", flush=True)
    print("INFO: Done.", flush=True)
    
if __name__ == '__main__':
    args = parse()
    main(args.nb)
