#!/usr/local/bin/python3

# --- Needed to tell python about the package in ../src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))
# ---

import argparse
from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
from src.beam import BeamExperiment
from src.mixing_type import MixingType
from src.signal_processor import SignalProcessor
from src.logger import Logger
from src.experiment import Experiment

def total_decays_less_than_observed(hnl_mass, mixing_squared, experiment, num_samples, mixing_type):
    beam = BeamExperiment(mixing_type=mixing_type,experiment=experiment, num_samples=num_samples).with_mixing(mixing_squared)

    beam.start_dump(hnl_mass)
    total_decays_less_than_observed = SignalProcessor(beam).total_decays_less_than_observed()

    return total_decays_less_than_observed

def main():
    hnl_masses, mass_range, num_samples, debug, mixing_type, plot, save, bound, experiment = parse_arguments()
    file = open(f"./{bound}_bound_data/{bound}_bounds_{mixing_type}_non_linear.csv", "a")
    logger = Logger(debug=debug)
    lower_bounds = []

    if mass_range is not None:
        lower_mass, upper_mass, num_points = mass_range
        hnl_masses = np.linspace(lower_mass, upper_mass, int(num_points))

    for hnl_mass in hnl_masses:
        lower_mixing = 1e-6
        upper_mixing = 1
        mixing = (upper_mixing + lower_mixing)/2
        prev_mixing = lower_mixing
        it = 0
        min_difference = inf
        closest_bound = 0
        while abs((prev_mixing - mixing)/mixing) > 0.0005 and it < 25: 
            logger.log(f"Mixing: {mixing}")
            less_decays_than_observed = total_decays_less_than_observed(hnl_mass, mixing, experiment, num_samples, mixing_type)

            diff = abs(less_decays_than_observed)
            if diff < min_difference:
                min_difference = diff
                closest_bound = mixing

            if less_decays_than_observed < 0:
                if bound == "lower":
                    upper_mixing = mixing
                else:
                    lower_mixing = mixing
            else:
                if bound == "lower":
                    lower_mixing = mixing
                else:
                    upper_mixing = mixing
            it += 1
            prev_mixing = mixing
            mixing = (upper_mixing + lower_mixing)/2
            logger.log(f"difference in mixing: {abs(prev_mixing - mixing)/mixing}")
        lower_bounds.append(mixing)
        if save:
            file.write(f"{hnl_mass},{mixing}\n")
        print("Mass: {}, Lower bound: {:e}".format(hnl_mass, mixing))    
    
    if plot:
        plt.plot(hnl_masses, lower_bounds)
        plt.yscale('log')
        plt.show()

    file.close()

def parse_arguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', type=float, nargs='+', help="Masses of the HNL in GeV")
    group.add_argument('--mass-range', type=float, nargs='+', help="Mass range '<lower> <higher> <number_of_points>' of the HNL in GeV")
    parser.add_argument('--bound', type=str, choices=['upper', 'lower'], help="Whether to calculate upper or lower bound", required=True)
    parser.add_argument('-n', type=int, help="Number of samples", required=True)
    parser.add_argument('--mixing', type=str, choices=['electron', 'tau'], help="valid values: 'electron' or 'tau'", required=True)
    parser.add_argument('--debug', action='store_true', help="Print debug logs")
    parser.add_argument('--plot', action='store_true', help="Plot upper bounds")
    parser.add_argument('--save', action='store_true', help="Save data to file")
    parser.add_argument('--exp', type=str, choices=['BEBC', 'NuTeV'], help="valid values: 'BEBC' or 'NuTeV'", default = 'BEBC')
    args = parser.parse_args()

    hnl_masses = args.m
    mass_range = args.mass_range
    bound = args.bound
    num_samples = args.n
    debug = args.debug
    plot = args.plot
    save = args.save
    mixing_type = MixingType[args.mixing]
    experiment = Experiment[args.exp]
    return hnl_masses,mass_range,num_samples,debug,mixing_type,plot,save,bound,experiment

if __name__ == "__main__":
    main()