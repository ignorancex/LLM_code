import matplotlib.pyplot as plt
import sys
import numpy as np

args = sys.argv[1:]
print(args)
state = str(args[0])
election = str(args[1])


parties = ["Rep", "Neutral", "Dem"]
party_colors = ["red", "grey", "blue"]
data_sets = []
for party in parties:
    file_str = f"data/neutral_senate/{state}_{party}_{election}_neutral_ensemble_seats_won.txt"

    with open(file_str, "r") as file:
        lines = list(file)
        # the seats won vector, house map dem seats
        seats_won = lines[2].split(", ")
        seats_won[0] = seats_won[0].split("[")[1]
        seats_won[-1] = seats_won[-1].split("]")[0]
        data_sets.append(([int(x) for x in seats_won], lines[0].split(" ")[-1]))
        del seats_won
    
        

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

title = f"Neutral Senate Ensembles on Biased House Maps in {state} under {election}"        

plt.title(title)
plt.legend(loc='center left', bbox_to_anchor=(1, .5))
plt.xticks(np.arange(bin_min-1, bin_max+2, 1))  


file_name = title.replace(" ", "_")+".pdf"
plt.savefig("Figures/"+file_name, bbox_inches='tight')
plt.close() 