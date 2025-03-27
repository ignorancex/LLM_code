from gerrychain.random import random
random.seed(2023)

from gerrychain import Graph, Partition, Election, MarkovChain, updaters, accept
from gerrychain.constraints import contiguous, UpperBound, within_percent_of_ideal_population
from gerrychain.tree import recursive_tree_part
from gerrychain.proposals import recom, swap
import pandas as pd
import networkx as nx
import geopandas
import maup
import math
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from pcompress import Record
import sys


_, user_state = sys.argv
state = str(user_state)

graphs = {"ReCom": {"OH": "OH_precincts_18.json",
               "WI": "WI_wards_20.json"},
        "Swap": {"OH": "OH_house_22.json",
               "WI": "WI_house_22.json"}}

parties = ["Rep", "None", "Dem"]
party_colors = ["red", "grey", "blue"]

election_names = {"OH": ["SEN18", "TRES18"],
            "WI": ["SEN18", "AG18"]}

burst_length = 10
# if bursts yield no new results after 10000 tries, stop
stability_bound = 10000
recom_ensemble_size = 1000000
swap_ensemble_size = 1000000


for election in election_names[state]:
    data_sets = []
    for party in parties:

        if state == "OH":
            epsilon = .06
        else:
            epsilon = .05
            
        
        graph = Graph.from_json(graphs["ReCom"][state])

        elections  = {"OH":[Election("SEN18", {"Dem": "G18USSDBRO", "Rep": "G18USSRREN"}),
                             Election("TRES18", {"Dem": "G18TREDRIC", "Rep": "G18TRERSPR"})],
                     "WI": [Election("SEN18", {"Dem": "SEN18D", "Rep": "SEN18R"}),
                        Election("AG18", {"Dem": "AG18D", "Rep": "AG18R"})]}
        election_updaters = {e.name: e for e in elections[state]}

        my_updaters= {"population": updaters.Tally("P0010001", alias="population")}
        my_updaters.update(election_updaters)


        # want to make a house map, so initial assignment is enacted house
        initial_partition = Partition(
                        graph, 
                        assignment="E_H_ID", 
                        updaters=my_updaters  
                    )

        # ideal pop is now divided by 99 districts
        ideal_population = sum(initial_partition["population"].values()) / 99

        proposal = partial(recom,
                           pop_col="P0010001",
                           pop_target=ideal_population,
                           epsilon=epsilon,
                           node_repeats=2
                          )

        compactness_bound = UpperBound(
            lambda p: len(p["cut_edges"]),
            2*len(initial_partition["cut_edges"])
        )

        pop_constraint = within_percent_of_ideal_population(initial_partition, epsilon)
        
        if party != "None":
            current_biased_map = initial_partition
            current_score = current_biased_map[election].seats(party)
            
            # short burst
            i=0
            while i < stability_bound:
                chain = MarkovChain(
                        proposal=proposal,
                        constraints=[
                            pop_constraint,
                            compactness_bound
                        ],
                        accept=accept.always_accept,
                        initial_state=current_biased_map,
                        total_steps=burst_length
                    )
                
                score_improved = False
                for p in chain:
                    new_score = p[election].seats(party)
                    
                    # trying to maximize for party
                    if new_score >= current_score:
                        current_biased_map = Partition(graph, assignment = p.assignment,
                                                        updaters = my_updaters)
                        
                        if new_score > current_score:
                            score_improved = True
                            
                        current_score = new_score
                
                if not score_improved:
                    i+=1
                
        
        
        else:
            chain = MarkovChain(
                    proposal=proposal,
                    constraints=[
                        pop_constraint,
                        compactness_bound
                    ],
                    accept=accept.always_accept,
                    initial_state=initial_partition,
                    total_steps=recom_ensemble_size
                )
            
            # just will use the last map as sample if not actually biasing
            for p in Record(chain, f"biased_house_maps_{party}_{election}_{state}.chain"):
                pass
            
            current_biased_map = p

        house_map_statistic = current_biased_map[election].seats("Dem")
        # saving the biased map
        for node, data in graph.nodes(data = True):
            data[f"H_bias_{party}_{election}"] = current_biased_map.assignment[node]


        def quotient_relationship(u,v):
            return graph.nodes[u][f"H_bias_{party}_{election}"]==graph.nodes[v][f"H_bias_{party}_{election}"]


        quotient_graph = nx.quotient_graph(graph, partition = quotient_relationship)
        
        # new house graph
        new_graph = Graph.from_networkx(quotient_graph)

        for node, data in new_graph.nodes(data = True):
            data["fake_population"] = 1
            # collating election data from precincts to new house districts

            dem_col = current_biased_map.updaters[election].parties_to_columns["Dem"]
            rep_col = current_biased_map.updaters[election].parties_to_columns["Rep"]

            data[rep_col]=sum([graph.nodes[precinct][rep_col] for precinct in node])
            data[dem_col]=sum([graph.nodes[precinct][dem_col] for precinct in node])

        # run swap on new house graph
        ideal_population = 3            
        epsilon = 0

        # while loop runs until cut is found
        while True:
            try:
                senate_assignment = recursive_tree_part(new_graph, 
                                                 parts = range(33),
                                                 pop_target = ideal_population,
                                                 pop_col = "fake_population", 
                                                 epsilon = epsilon, 
                                                 node_repeats=2)
                break
            except:
                pass



        
        elections  = {"OH":[Election("SEN18", {"Dem": "G18USSDBRO", "Rep": "G18USSRREN"}),
                             Election("TRES18", {"Dem": "G18TREDRIC", "Rep": "G18TRERSPR"})],
                     "WI": [Election("SEN18", {"Dem": "SEN18D", "Rep": "SEN18R"}),
                        Election("AG18", {"Dem": "AG18D", "Rep": "AG18R"})]}
        election_updaters = {e.name: e for e in elections[state]}


        initial_partition = Partition(
                        new_graph, 
                        assignment=senate_assignment, 
                        updaters=election_updaters  
                    )

        chain = MarkovChain(
                        proposal=swap,
                        constraints=[contiguous],
                        accept=accept.always_accept,
                        initial_state=initial_partition,
                        total_steps=swap_ensemble_size
                    )

        # regardless of bias, we will look at dem seats as the statistic
        seats_won = []
        for p in Record(chain, f"senate_maps_on_bias_house_{party}_{election}_{state}.chain"):
            seats_won.append(p[election].seats("Dem"))

        data_sets.append((seats_won, house_map_statistic))

    lower_range = [min(seats_won) for seats_won, house_map_statistic in data_sets]
    upper_range = [max(seats_won) for seats_won, house_map_statistic in data_sets]
    
    print("seats won ranges")
    print(state)
    print(election)
    print(parties)
    print(lower_range)
    print(upper_range)
    bin_min = min(lower_range)
    bin_max = max(upper_range)

    bins = np.arange(bin_min - 0.5, bin_max + 1.5, 1)

    plt.hist([seats_won for seats_won, house_map_statistic in data_sets], 
             bins = bins, 
             label=[f"{p}: {data_sets[i][1]}" for i, p in enumerate(parties)],
            color = party_colors)
    plt.xlabel('Senate Seats Won by Dems')
    plt.ylabel('Frequency')

    title = f"Biasing House Maps in {state} under {election}"        

    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(np.arange(bin_min-1, bin_max+2, 1))  


    file_name = title.replace(" ", "_")+".pdf"
    plt.savefig("Figures/"+file_name,bbox_inches='tight')
    plt.close()        

