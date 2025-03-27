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

# # Running Swap Chains
# swap runs
chain_type  = "Swap"
swap_ensemble_size = 1000000
graphs = {"OH":{"Swap": "OH_house_22.json",
               "ReCom": "OH_precincts_18.json"},
         "WI": {"Swap": "WI_house_22.json",
               "ReCom": "WI_wards_20.json"}}

elections  = {"OH":[Election("SEN18", {"Dem": "G18USSDBRO", "Rep": "G18USSRREN"}),
                     Election("TRES18", {"Dem": "G18TREDRIC", "Rep": "G18TRERSPR"})],
             "WI": [Election("SEN18", {"Dem": "SEN18D", "Rep": "SEN18R"}),
                Election("AG18", {"Dem": "AG18D", "Rep": "AG18R"})]}


# intializing dictionary to store data
df_dict = {}

for seed in ["E", "S1", "S2"]:
    for e in elections[state]:
        df_dict[state+chain_type+seed+e.name+"SEATS"] = []
        for i in range(33):
            df_dict[state+chain_type+seed+e.name+"VOTE SHARE"+str(i)] = []
            


graph = Graph.from_json(graphs[state][chain_type])
election_updaters = {e.name: e for e in elections[state]}

for seed in ["E", "S1", "S2"]:
    if seed == "E":
        initial_partition = Partition(
            graph,
            assignment="E_S_ID",
            updaters= election_updaters)
        
        
    else:
        epsilon = 0
        ideal_population = 3
        while True:
            try:
                senate_assignment = recursive_tree_part(graph, 
                                                 parts = range(33),
                                                 pop_target = ideal_population,
                                                 pop_col = "fake_population", 
                                                 epsilon = epsilon, 
                                                 node_repeats=2)
                
                break
            except:
                pass
            
        
            
        initial_partition = Partition(
            graph,
            assignment=senate_assignment,
            updaters= election_updaters)
    
    
    chain = MarkovChain(
        proposal=swap,
        constraints=[contiguous],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=swap_ensemble_size
    )
    
    
    for plan in Record(chain, f"{state}_{chain_type}_{seed}.chain"):
        for e in elections[state]:
            # store seats won and ranked percent dem vote share
            df_dict[state+chain_type+seed+e.name+"SEATS"].append(plan[e.name].seats("Dem"))
            for j, percent in enumerate(sorted(plan[e.name].percents("Dem"))):
                df_dict[state+chain_type+seed+e.name+"VOTE SHARE"+str(j)].append(percent)
    
            
    print(state, seed)
                    
swap_df = pd.DataFrame(df_dict)
print("swap done")



# In[25]:


# # Running ReCom chains
# recom runs
chain_type  = "ReCom"
recom_ensemble_size = 1000000

graphs = {"OH":{"Swap": "OH_house_22.json",
               "ReCom": "OH_precincts_18.json"},
         "WI": {"Swap": "WI_house_22.json",
               "ReCom": "WI_wards_20.json"}}

elections  = {"OH":[Election("SEN18", {"Dem": "G18USSDBRO", "Rep": "G18USSRREN"}),
                     Election("TRES18", {"Dem": "G18TREDRIC", "Rep": "G18TRERSPR"})],
             "WI": [Election("SEN18", {"Dem": "SEN18D", "Rep": "SEN18R"}),
                Election("AG18", {"Dem": "AG18D", "Rep": "AG18R"})]}


seed_assignments = {"OH": ["E", "D", "I", "C"],
                   "WI": ["E", "S1", "S2"]} 

epsilon = .05

# intializing dictionary to store data
df_dict = {}

for seed in seed_assignments[state]:
        for e in elections[state]:
            df_dict[state+chain_type+seed+e.name+"SEATS"] = []
            for i in range(33):
                df_dict[state+chain_type+seed+e.name+"VOTE SHARE"+str(i)] = []
                

graph = Graph.from_json(graphs[state][chain_type])
election_updaters = {e.name: e for e in elections[state]}
my_updaters= {"population": updaters.Tally("P0010001", alias="population")}
    
my_updaters.update(election_updaters)


for seed in seed_assignments[state]:
    print(seed)
    if seed != "I":
        epsilon = .05
    else:
        epsilon = .06
    
    # use real world seeds
    if state == "OH" or (state == "WI" and seed == "E"):
        initial_partition = Partition(
            graph,
            assignment=seed+"_S_ID",
            updaters= my_updaters)
        
    else:
        # use randomly generated seeds for wisconsin
        # just to compute ideal population
        initial_partition = Partition(
            graph, 
            assignment="E_S_ID", 
            updaters=my_updaters  
        )

        ideal_population = sum(initial_partition["population"].values()) / 33

        while True:
            try:
                assignment = recursive_tree_part(graph, parts = range(33), 
                                    pop_target = ideal_population, pop_col="P0010001", 
                                    epsilon=epsilon, node_repeats=2)
                
                break
            
            except:
                pass
        
        initial_partition = Partition(
            graph,
            assignment= assignment,
            updaters= my_updaters)
    
    
    ideal_population = sum(initial_partition["population"].values()) / 33

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
    
    constraints = [pop_constraint, compactness_bound]
        
    chain = MarkovChain(
        proposal=proposal,
        constraints=constraints,
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=recom_ensemble_size
    )


    for i, plan in enumerate(Record(chain, f"{state}_{chain_type}_{seed}.chain")):
        for e in elections[state]:
            
            # store seats won and ranked percent dem vote share
            df_dict[state+chain_type+seed+e.name+"SEATS"].append(plan[e.name].seats("Dem"))
            for j, percent in enumerate(sorted(plan[e.name].percents("Dem"))):
                df_dict[state+chain_type+seed+e.name+"VOTE SHARE"+str(j)].append(percent)
            
        
            
    print(state, seed)
            
recom_df = pd.DataFrame(df_dict)
print("recom done")

dfs = {"Swap": swap_df, "ReCom": recom_df}


# Creating Figures

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

seeds = {"OH": {"Swap" : ["E", "S1", "S2"],
              "ReCom": ["E", "D", "I", "C"]},
        "WI": {"Swap" : ["E", "S1", "S2"],
              "ReCom": ["E", "S1", "S2"]}}

pieces = {"Swap": [int(x*swap_ensemble_size) for x in [.1, .5, 1]],
         "ReCom" : [int(x*recom_ensemble_size) for x in [.1, .5, 1]]}

auto_corr_len = {"Swap":int(swap_ensemble_size/20),
                 "ReCom":int(recom_ensemble_size/20)}



# seats won
# comparing seeds

for e in elections[state]:
    for chain_type in ["Swap", "ReCom"]:
        data_sets = [dfs[chain_type][state+chain_type+seed+e.name+"SEATS"] for seed in seeds[state][chain_type]]
        
        print("seats won ranges")
        print(state)
        print(e.name)
        print(chain_type)
        print(seeds[state][chain_type])
        print([min(data) for data in data_sets])
        print([max(data) for data in data_sets])
        
        bin_min = min([min(data) for data in data_sets])
        bin_max = max([max(data) for data in data_sets])
        bins = np.arange(bin_min - 0.5, bin_max + 1.5, 1)

        plt.hist(data_sets, bins = bins, label=[f'{s}: {data_sets[i][0]}' for i,s in enumerate(seeds[state][chain_type])])
        plt.xlabel('Seats Won by Dems')
        plt.ylabel('Frequency')

        title = f"{state} {chain_type} {e.name}\nSeats Won Comparing Seeds".replace("  "," ")        

        plt.title(title)
        plt.legend(loc='upper left', bbox_to_anchor =(1,1))
        plt.xticks(np.arange(bin_min-1, bin_max+2, 1)) 

        # Show the plot
        file_name = "Figures/"+title.replace("\n", " ").replace(" ", "_")+".pdf"
        plt.savefig(file_name,bbox_inches='tight')
        plt.close()




seed = "E"
# comparing swap to recom

for e in elections[state]:
    data_sets = [dfs["Swap"][state+"Swap"+seed+e.name+"SEATS"],
                dfs["ReCom"][state+"ReCom"+seed+e.name+"SEATS"]]
    
    
    bin_min = min([min(data) for data in data_sets])
    bin_max = max([max(data) for data in data_sets])
    bins = np.arange(bin_min - 0.5, bin_max + 1.5, 1)

    plt.hist(data_sets, bins = bins, label=['Swap', "ReCom"]    )
    plt.xlabel('Seats Won by Dems')
    plt.ylabel('Frequency ')
    plt.legend(loc='upper left', bbox_to_anchor = (1,1))
    plt.xticks(np.arange(bin_min-1, bin_max+2, 1)) 
    
    title = f"{state} {seed} {e.name}\nSeats Won Swap v. ReCom".replace("  "," ")      
    plt.title(title)
    plt.savefig("Figures/"+title.replace("\n"," ").replace(" ", "_")+".pdf", bbox_inches='tight')
    plt.close()
            



# auto correlate plots
seed = "E"
for chain_type in ["Swap", "ReCom"]:
    for e in elections[state]:
        data_seat = dfs[chain_type][state+chain_type+seed+e.name+"SEATS"]
        
        ac_seats = [data_seat.autocorr(lag = n) for n in range(auto_corr_len[chain_type])]

        plt.plot(ac_seats, label = "Dem Seats", color = 'b')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        
        
        title = f"{state} {chain_type} {seed} \n{e.name} Autocorrelation".replace("  ", " ")
        file_name = f"{state} {chain_type} {seed}  {e.name} Autocorrelation.pdf".replace("  ", " ").replace(" ", "_")
        plt.title(title)
        lgd = plt.legend(loc='upper right')
        plt.savefig("Figures/"+file_name,
                  bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()



font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 22}

plt.rc('font', **font)

linewidth = 1.5
colors = ["r", "k","b", "g"]



# sliced ensemble boxplots 
seed = "E"
for chain_type in ["Swap", "ReCom"]:
    for e in elections[state]:
        fig, ax = plt.subplots(figsize=(20, 10))

        # Draw 50% line
        ax.axhline(0.5, color="#cccccc")

        # store different sliced boxplots
        bps = []

        for index, piece in enumerate(pieces[chain_type]):
            data = dfs[chain_type][[state+chain_type+seed+e.name+"VOTE SHARE"+str(j) for j in range(33)]][:piece]

            # offset the different ensembles
            positions = [i+index/6 for i in range(33)]
            bps.append(ax.boxplot(data, 
                                    positions=positions, 
                                    showfliers=False, 
                                    widths =.1, 
                                  patch_artist = True,
                                    boxprops=dict(color=colors[index], linewidth=linewidth, facecolor=(0,0,0,0)),
                                 capprops=dict(color = colors[index], linewidth=linewidth),
                                  whiskerprops=dict(color = colors[index],linewidth=linewidth),
                                  medianprops=dict(color = colors[index], linewidth=linewidth)
                                 )
                      )



        # Annotate
        title = f"{state} {chain_type} {seed}  {e.name} Democratic Vote Share Partial Ensembles".replace("  ", " ")
        ax.set_title(title)
        ax.set_ylabel("% Democratic vote")
        ax.set_xlabel("Sorted districts")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticks([])

        ax.legend([x["whiskers"][0] for x in bps], 
                  [str(piece) for piece in pieces[chain_type]], 
                  loc='upper left')

        plt.savefig("Figures/"+ title.replace(" ", "_")+ ".pdf",
                  bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.close()             
    

# different seeds boxplots
for chain_type in ["Swap", "ReCom"]:
    
    for e in elections[state]:
        
        fig, ax = plt.subplots(figsize=(20, 10))

        # Draw 50% line
        ax.axhline(0.5, color="#cccccc")

        # store different sliced boxplots
        bps = []

        for index, seed in enumerate(seeds[state][chain_type]):
            data = dfs[chain_type][[state+chain_type+seed+e.name+"VOTE SHARE"+str(j) for j in range(33)]][:piece]
            #data = data[[state+seed+title+election.name+"_"+str(j) for j in range(33)]]

            # offset the different ensembles
            positions = [i+index/6 for i in range(33)]
            bps.append(ax.boxplot(data, 
                                    positions=positions, 
                                    showfliers=False, 
                                    widths =.1, 
                                  patch_artist = True,
                                    boxprops=dict(color=colors[index], linewidth=linewidth, facecolor=(0,0,0,0)),
                                 capprops=dict(color = colors[index], linewidth=linewidth),
                                  whiskerprops=dict(color = colors[index],linewidth=linewidth),
                                  medianprops=dict(color = colors[index], linewidth=linewidth)
                                 )
                      )

        title = f"{state} {chain_type} {e.name} % Democratic Vote Share Different Seeds".replace("  "," ")
        ax.set_title(title)
        ax.set_ylabel("% Democratic vote")
        ax.set_xlabel("Sorted districts")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticks([])

        ax.legend([x["whiskers"][0] for x in bps], 
                  [str(seed) for seed in seeds[state][chain_type]], 
                  loc='upper left')

        plt.savefig("Figures/" + title.replace(" ", "_") + ".pdf",
                  bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.close()            
    

