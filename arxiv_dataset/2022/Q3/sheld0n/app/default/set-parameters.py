#!/usr/bin/env python3

# command line program
import argparse
# util
import math
# lib param
import libset


def parse():
    parser = argparse.ArgumentParser(description='Set simulation parameters.')
    return parser.parse_args()

def main():
    args = parse()
    # edit the following to set parameters
    ## parameters
    dt = 1.0/128.0 # time step
    ntime = int(round(16.0 * math.pi / dt)) # number of time steps
    number = 128 # number of particles
    npost = 64 # number of post processed steps
    ## set, the path to directories must be given following the glob syntax.
    libset.set_parameter("param/run", "Dt", dt)
    libset.set_parameter("param/run", "NTime", ntime)
    libset.set_parameter("param/post", "Number", npost)
    libset.set_parameter("param/solutions/*", "Number", number) # will edit all "parameters.h" files found in that directory (all equations)

if __name__ == '__main__':
    main()
