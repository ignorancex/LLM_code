""" Run JAMA simulation grid search across mega-influencer reaches
"""
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)

import src as odyn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable

def main():
    new_dir = ("jama_simulation_results")
    check_dir = os.path.isdir("jama_simulation_results")
    if not check_dir:
            os.makedirs(new_dir)
    for left_reach in [0,.25,.5,.75,1]:
        for right_reach in [0,.25,.5,.75,1]:

            l = int(left_reach * 100)
            r = int(right_reach * 100)

            model = odyn.OpinionNetworkModel(probabilities = [.69, .31],
                                            power_law_exponent = 1.5,
                                            openness_to_neighbors = 1.5,
                                            openness_to_influencers = 1.5,
                                            distance_scaling_factor = 1/10,
                                            importance_of_weight = 2, 
                                            importance_of_distance = 8,
                                            left_reach = left_reach,
                                            right_reach = right_reach,
                                            include_opinion = True,
                                            include_weight = True,
                                            include_distance = True)

            model.agent_df = model.add_random_agents_to_triangle(num_agents = 1000,show_plot = False)

            hesitant = 0
            while hesitant != 31:
                model.belief_df = model.assign_weights_and_beliefs(model.agent_df)
                hesitant = int(np.sum(model.belief_df["belief"] > 0)/model.belief_df.shape[0] * 100)

            prob_df = model.compute_probability_array(model.belief_df)
            model.adjacency_df = model.compute_adjacency(prob_df)

            model.belief_df.to_csv(f"../data/simulation_results_from_paper/left_reach_{l:03}_right_reach_{r:03}/belief_df.csv")
            
            model.left_reach = left_reach
            model.right_reach = right_reach
            model.mega_influencer_df = model.connect_mega_influencers(model.belief_df)

            sim = odyn.NetworkSimulation()
            sim.run_simulation(model = model, stopping_thresh = 0.01, show_plot = False, store_results = False)

            sim.dynamic_belief_df.to_csv(f"../data/simulation_results_from_paper/left_reach_{l:03}_right_reach_{r:03}/dynamic_belief_df.csv")

if __name__ == '__main__':
    main()