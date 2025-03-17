from gerrychain.optimization import SingleMetricOptimizer
from gerrychain import Partition, MarkovChain, constraints, accept, Election, Graph, updaters
from gerrychain.updaters import Tally
from gerrychain.tree import recursive_tree_part, bipartition_tree
from gerrychain.proposals import ReCom, recom
from functools import partial
from swap_proposal import swap

import sys, csv
import random
import networkx as nx

random.seed(2024)


args = sys.argv[1:]
print(args)
state = str(args[0])
bias_party = str(args[1])
election = str(args[2])
opt_method = str(args[3])
opt_param = float(args[4])

ensemble_size = 1000000
print(ensemble_size)

graphs = {"ReCom": {"OH": "../json/OH_precincts_18.json",
               "WI": "../json/WI_wards_20.json"},
        "Swap": {"OH": "../json/OH_house_22.json",
               "WI": "../json/WI_house_22.json"}}


graph = Graph.from_json(graphs["ReCom"][state])
elections  = {"OH":[Election("SEN18", {"Dem": "G18USSDBRO", "Rep": "G18USSRREN"}),
                    Election("TRES18", {"Dem": "G18TREDRIC", "Rep": "G18TRERSPR"})],
            "WI": [Election("SEN18", {"Dem": "SEN18D", "Rep": "SEN18R"}),
                Election("AG18", {"Dem": "AG18D", "Rep": "AG18R"})]}

election_updaters = {e.name: e for e in elections[state]}

biased_house_file_paths = {"OH": {"Dem": {"SEN18": "OH_Dem_SEN18_burst_7.0_best_assignment.csv",
                                            "TRES18": "OH_Dem_TRES18_burst_8.0_best_assignment.csv"},
                                    "Rep":{"SEN18": "OH_Rep_SEN18_burst_9.0_best_assignment.csv",
                                            "TRES18": "OH_Rep_TRES18_burst_7.0_best_assignment.csv"}},
                            "WI": {"Dem": {"SEN18": "WI_Dem_SEN18_burst_8.0_best_assignment.csv",
                                            "AG18": "WI_Dem_AG18_burst_11.0_best_assignment.csv"},
                                    "Rep":{"SEN18": "WI_Rep_SEN18_burst_8.0_best_assignment.csv",
                                            "AG18": "WI_Rep_AG18_burst_7.0_best_assignment.csv"}}}

file_path = biased_house_file_paths[state][bias_party][election]
with open(f"data/house/{file_path}", mode='r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    assignment = {}
    for i, row in enumerate(reader):
        if i == 0:
            continue
        
        node, dist = row
        assignment[int(node)] = int(dist)

biased_map = Partition(graph, assignment, updaters = election_updaters)

# saving the biased map
for node, data in graph.nodes(data = True):
    data[f"H_bias_{bias_party}_{election}"] = assignment[node]


def quotient_relationship(u,v):
    return graph.nodes[u][f"H_bias_{bias_party}_{election}"]==graph.nodes[v][f"H_bias_{bias_party}_{election}"]


quotient_graph = nx.quotient_graph(graph, partition = quotient_relationship)

# new house graph
new_graph = Graph.from_networkx(quotient_graph)


for node, data in new_graph.nodes(data = True):
    data["fake_population"] = 1
    # collating election data from precincts to new house districts

    dem_col = biased_map.updaters[election].parties_to_columns["Dem"]
    rep_col = biased_map.updaters[election].parties_to_columns["Rep"]

    data[rep_col]=sum([graph.nodes[precinct][rep_col] for precinct in node])
    data[dem_col]=sum([graph.nodes[precinct][dem_col] for precinct in node])

# run swap on new house graph
ideal_population = 3            
epsilon = 0
my_updaters= {"population": updaters.Tally("fake_population", alias="population")}
my_updaters.update(election_updaters)


# random initial partition
for i in range(10):
    try:
        initial_partition = Partition.from_random_assignment(
                        graph=new_graph,
                        n_parts = 33,
                        epsilon = epsilon,
                        pop_col = "fake_population",
                        updaters = my_updaters,
                        method = partial(
                                        recursive_tree_part,
                                        node_repeats = 20,
                                        method=partial(
                                                bipartition_tree,
                                                max_attempts=10000,
                                                allow_pair_reselection=True
                                            )
                                    )
                    )
        break
    except:
        continue

if i == 9:
    raise ValueError("failed to find partition after 10 tries")

# TODO this is where i need to edit to swap
# proposal = partial(recom,
#                     pop_col="fake_population",
#                     pop_target=ideal_population,
#                     epsilon=epsilon,
#                     node_repeats=2,
#                     method = partial(
#                                 bipartition_tree,
#                                 max_attempts=100,
#                                 allow_pair_reselection=True
#                             )
#                     )

# pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, epsilon)
num_seats= lambda p: p[election].seats(bias_party)

# optimize to maximize number of seats for bias party
optimizer = SingleMetricOptimizer(
        proposal=swap,
        constraints=constraints.contiguous,
        initial_state=initial_partition,
        optimization_metric=num_seats,
        maximize=True
    )


file_str=f"data/senate/{state}_{bias_party}_{election}_{opt_method}_{opt_param}"

if opt_method == "tilt":
    for p in optimizer.tilted_run(ensemble_size, p=opt_param):
        pass
else:
    for p in optimizer.short_bursts(burst_length = opt_param, num_bursts = int(ensemble_size/opt_param)):
        pass



# Open the file in write mode
with open(f"{file_str}_score.txt", 'w') as file:
    print(optimizer.best_score, file=file)

best_assignment = optimizer.best_part.assignment.to_series()
best_assignment.to_csv(f"{file_str}_best_assignment.csv")

print("done")