# TODO maybe i should save the stats and then reload them, seems to be very long 
# to rerun the chains


from pcompress import Replay
from gerrychain import Graph, Election
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

args = sys.argv[1:]
state = str(args[0])

TESTING = False

ensemble_size = 1000000

graphs = {"OH":{"Swap": "../json/OH_house_22.json",
               "ReCom": "../json/OH_precincts_18.json"},
         "WI": {"Swap": "../json/WI_house_22.json",
               "ReCom": "../json/WI_wards_20.json"}}

seeds = {"OH": {"Swap" : ["E", "S1", "S2"],
              "ReCom": ["E", "D", "I", "C"]},
        "WI": {"Swap" : ["E", "S1", "S2"],
              "ReCom": ["E", "S1", "S2"]}}

plot_relabeling = {"Swap": "Nested",
                "ReCom": "Unnested"}

elections  = {"OH":[Election("SEN18", {"Dem": "G18USSDBRO", "Rep": "G18USSRREN"}),
                     Election("TRES18", {"Dem": "G18TREDRIC", "Rep": "G18TRERSPR"})],
             "WI": [Election("SEN18", {"Dem": "SEN18D", "Rep": "SEN18R"}),
                Election("AG18", {"Dem": "AG18D", "Rep": "AG18R"})]}

election_updaters = {e.name: e for e in elections[state]}

#swap chain code
chain_type = "Swap"

graph = Graph.from_json(graphs[state][chain_type])
# intializing dictionary to store data
df_dict = {}

for seed in seeds[state][chain_type]:
    for e in elections[state]:
        df_dict[state+chain_type+seed+e.name+"SEATS"] = []
        for i in range(33):
            df_dict[state+chain_type+seed+e.name+"VOTE SHARE"+str(i)] = []

for seed in seeds[state][chain_type]:
    print("swap chain ended", seed)
    for i,plan in enumerate(Replay(graph, f"../old_code/{state}_{chain_type}_{seed}.chain", updaters = election_updaters)):
        for e in elections[state]:
            # store seats won and ranked percent dem vote share
            df_dict[state+chain_type+seed+e.name+"SEATS"].append(plan[e.name].seats("Dem"))
            for j, percent in enumerate(sorted(plan[e.name].percents("Dem"))):
                df_dict[state+chain_type+seed+e.name+"VOTE SHARE"+str(j)].append(percent)
        
        if TESTING and i == 100:
            break
    print("swap chain ended", seed)
    
                    
swap_df = pd.DataFrame(df_dict)




#recom chain code
chain_type = "ReCom"
graph = Graph.from_json(graphs[state][chain_type])
elections  = {"OH":[Election("SEN18", {"Dem": "G18USSDBRO", "Rep": "G18USSRREN"}),
                     Election("TRES18", {"Dem": "G18TREDRIC", "Rep": "G18TRERSPR"})],
             "WI": [Election("SEN18", {"Dem": "SEN18D", "Rep": "SEN18R"}),
                Election("AG18", {"Dem": "AG18D", "Rep": "AG18R"})]}

election_updaters = {e.name: e for e in elections[state]}


# intializing dictionary to store data
df_dict = {}

for seed in seeds[state][chain_type]:
    for e in elections[state]:
        df_dict[state+chain_type+seed+e.name+"SEATS"] = []
        for i in range(33):
            df_dict[state+chain_type+seed+e.name+"VOTE SHARE"+str(i)] = []

for seed in seeds[state][chain_type]:
    for i,plan in enumerate(Replay(graph, f"../old_code/{state}_{chain_type}_{seed}.chain", updaters = election_updaters)):
        for e in elections[state]:
            # store seats won and ranked percent dem vote share
            df_dict[state+chain_type+seed+e.name+"SEATS"].append(plan[e.name].seats("Dem"))
            for j, percent in enumerate(sorted(plan[e.name].percents("Dem"))):
                df_dict[state+chain_type+seed+e.name+"VOTE SHARE"+str(j)].append(percent)

        if TESTING and i == 100:
            break
    print("recom chain ended", seed)

recom_df = pd.DataFrame(df_dict)


dfs = {"Swap": swap_df, "ReCom": recom_df}


# Creating Figures

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)



pieces = {"Swap": [int(x*ensemble_size) for x in [.1, .5, 1]],
         "ReCom" : [int(x*ensemble_size) for x in [.1, .5, 1]]}

auto_corr_len = {"Swap":int(ensemble_size/20),
                 "ReCom":int(ensemble_size/20)}



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

        title = f"{state} {plot_relabeling[chain_type]} {e.name}\nSeats Won Comparing Seeds".replace("  "," ")        

        plt.title(title)
        plt.legend(loc='upper left', bbox_to_anchor =(1,1))
        plt.xticks(np.arange(bin_min-1, bin_max+2, 1)) 

        # Show the plot
        file_name = "Figures/"+title.replace("\n", " ").replace(" ", "_")+".pdf"
        plt.savefig(file_name,bbox_inches='tight')
        plt.close()


print("seats won fig doe")

seed = "E"
# comparing swap to recom

for e in elections[state]:
    data_sets = [dfs["Swap"][state+"Swap"+seed+e.name+"SEATS"],
                dfs["ReCom"][state+"ReCom"+seed+e.name+"SEATS"]]
    
    
    bin_min = min([min(data) for data in data_sets])
    bin_max = max([max(data) for data in data_sets])
    bins = np.arange(bin_min - 0.5, bin_max + 1.5, 1)

    plt.hist(data_sets, bins = bins, label=['Nested', "Unnested"]    )
    plt.xlabel('Seats Won by Dems')
    plt.ylabel('Frequency ')
    plt.legend(loc='upper left', bbox_to_anchor = (1,1))
    plt.xticks(np.arange(bin_min-1, bin_max+2, 1)) 
    
    title = f"{state} {seed} {e.name}\nSeats Won Nested v. Unnested".replace("  "," ")      
    plt.title(title)
    plt.savefig("Figures/"+title.replace("\n"," ").replace(" ", "_")+".pdf", bbox_inches='tight')
    plt.close()
            
print("comparing swpa to recom fig done")


# auto correlate plots
seed = "E"
for chain_type in ["Swap", "ReCom"]:
    for e in elections[state]:
        data_seat = dfs[chain_type][state+chain_type+seed+e.name+"SEATS"]
        
        ac_seats = [data_seat.autocorr(lag = n) for n in range(auto_corr_len[chain_type])]

        plt.plot(ac_seats, label = "Dem Seats", color = 'b')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        
        
        title = f"{state} {plot_relabeling[chain_type]} {seed} \n{e.name} Autocorrelation".replace("  ", " ")
        file_name = f"{state} {plot_relabeling[chain_type]} {seed}  {e.name} Autocorrelation.pdf".replace("  ", " ").replace(" ", "_")
        plt.title(title)
        lgd = plt.legend(loc='upper right')
        plt.savefig("Figures/"+file_name,
                  bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

print("autocorrelates done")

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
        title = f"{state} {plot_relabeling[chain_type]} {seed}  {e.name} Democratic Vote Share Partial Ensembles".replace("  ", " ")
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

print("sliced boxplot done")


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

        title = f"{state} {plot_relabeling[chain_type]} {e.name} % Democratic Vote Share Different Seeds".replace("  "," ")
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

print("diffferent seeds done")
