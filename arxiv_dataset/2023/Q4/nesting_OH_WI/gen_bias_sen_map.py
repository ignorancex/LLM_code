from gerrychain.optimization import SingleMetricOptimizer
from gerrychain import Partition, MarkovChain, constraints, accept, Election, Graph, updaters
from gerrychain.updaters import Tally
from gerrychain.tree import recursive_tree_part, bipartition_tree
from gerrychain.proposals import ReCom, recom
from functools import partial
import sys 
import random

random.seed(2023)


args = sys.argv[1:]
print(args)
state = str(args[0])
bias_party = str(args[1])
election = str(args[2])
opt_method = str(args[3])
opt_param = float(args[4])

ensemble_size = 1000000
# ensemble_size = 10

graphs = {"ReCom": {"OH": "../json/OH_precincts_18.json",
               "WI": "../json/WI_wards_20.json"},
        "Swap": {"OH": "../json/OH_house_22.json",
               "WI": "../json/WI_house_22.json"}}

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


# want to make a senate map so seed is enacted senate
initial_partition = Partition(
                graph, 
                assignment="E_S_ID", 
                updaters=my_updaters  
            )

# ideal pop is now divided by 33 districts
ideal_population = sum(initial_partition["population"].values()) / 33

proposal = partial(recom,
                    pop_col="P0010001",
                    pop_target=ideal_population,
                    epsilon=epsilon,
                    node_repeats=2,
                    method = partial(
                                bipartition_tree,
                                max_attempts=100,
                                allow_pair_reselection=True
                            )
                    )


pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, epsilon)
num_seats= lambda p: p[election].seats(bias_party)

optimizer = SingleMetricOptimizer(
        proposal=proposal,
        constraints=pop_constraint,
        initial_state=initial_partition,
        optimization_metric=num_seats,
        maximize=True
    )

file_str=f"data/bias_senate_recom/{state}_{bias_party}_{election}_{opt_method}_{opt_param}"

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
