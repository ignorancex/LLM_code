#!/usr/bin/env python
# coding: utf-8

# In[34]:


from gerrychain.random import random
random.seed(2023)

from gerrychain import Graph, Partition, Election, MarkovChain, updaters, accept
from gerrychain.constraints import contiguous, UpperBound, within_percent_of_ideal_population, valid_county_splits_swap, valid_county_splits_recom
from gerrychain.tree import recursive_tree_part, bipartition_tree, random_spanning_tree
from gerrychain.proposals import recom, swap
import pandas as pd
import networkx as nx
import geopandas
import maup
import math
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

print("modules loaded")



# # Processing shapefiles

# In[3]:


seeds = {"OH": ["E", "D", "I", "C"],
        "WI": ["E"]}

h_maps = {"OH": {"E": "oh_sldl_adopted_2022.zip",
                "D": "SR House.zip",
                "I": "JM House.zip",
                "C": "OCRC House.zip"},
         "WI": {"E": "tl_2022_55_sldl.zip"}}

s_maps = {"OH": {"E": "oh_sldu_adopted_2022.zip",
                "D": "SR Senate.zip",
                "I": "JM Senate.zip",
                "C": "OCRC Senate.zip"},
         "WI": {"E": "tl_2022_55_sldu.zip"}}

c_blocks = {"OH": "oh_pl2020_p1_b.zip",
           "WI": "wi_pl2020_p1_b.zip"}

p_maps = {"OH": "oh_vest_18.zip",
            "WI": "WI_2020_wards.zip"}

elections = {"OH": {"TRES18": ['G18TRERSPR', 'G18TREDRIC'],
                    "SEN18": ['G18USSRREN' , 'G18USSDBRO' ]},
            "WI":{"AG18":["AG18R","AG18D"],
                 "SEN18":["SEN18R", "SEN18D"]}
            }


for state in ["WI", "OH"]:
    hds = geopandas.read_file(h_maps[state]["E"])
    sds = geopandas.read_file(s_maps[state]["E"])
    ps = geopandas.read_file(p_maps[state])
    print(state, "geopandas loaded")
    
    # mapping house to senate
    assignment = maup.assign(hds, sds)
    
    hds["E_S_ID"] = assignment
    h_graph = Graph.from_file(h_maps[state]["E"])
    h_graph.join(hds, columns = ["E_S_ID"], left_index = "E_H_ID", right_index = "E_H_ID")
    
    # aggregating election data to house districts
    hds.to_crs(ps.crs, inplace=True)
    assignment = maup.assign(ps, hds)
    for election in elections[state].keys():
        election_cols = elections[state][election]
        hds[election_cols]= ps[election_cols].groupby(assignment).sum()
        h_graph.join(hds, columns = election_cols, left_index="E_H_ID", right_index="E_H_ID")
        
    # lets us use ReCom on house graph
    for node, data in h_graph.nodes(data = True):
        data["fake_population"] = 1
    
    h_graph.to_json(state+"_house_22.json")
    print(state, "house graph saved")
    
    # this deals with the fact that the wisconsin shape file is not connected, so we load the repaired graph from MGGG
    if state == "OH":
        p_graph = Graph.from_file(p_maps[state])
    elif state == "WI":
        p_graph = Graph.from_json("wisconsin2020_graph.json")
        
        
    # mapping precincts to senate, house districts for any seeds
    for seed in seeds[state]:
        hds = geopandas.read_file(h_maps[state][seed])
        hds.to_crs(ps.crs, inplace=True)
        sds = geopandas.read_file(s_maps[state][seed])
        sds.to_crs(ps.crs, inplace=True)
        
        
        h_assignment = maup.assign(ps, hds)
        s_assignment = maup.assign(ps, sds)
    
        ps[seed+"_H_ID"] = h_assignment
        ps[seed+"_S_ID"] = s_assignment
        
        # the precinct tag
        if state == "OH":
            tag = "GEOID18"
        elif state == "WI":
            tag = "Code-2"
            
        p_graph.join(ps, columns=[seed+"_H_ID", seed+"_S_ID"], left_index=tag, right_index=tag)
        
    
    # adding population data to precincts
    bs = geopandas.read_file(c_blocks[state])
    bs.to_crs(ps.crs, inplace = True)
    assignment = maup.assign(bs, ps)
    
    # population column
    ps["P0010001"] = bs["P0010001"].groupby(assignment).sum()
    p_graph.join(ps, columns = ["P0010001"], left_index=tag, right_index=tag)
    
    if state == "OH":
        title = "OH_precincts_18.json"
    elif state == "WI":
        title = "WI_wards_20.json"
        
        # deals with na population errors by using 2010 value
        for node, data in list(p_graph.nodes(data=True)):
            if pd.isna(data["P0010001"]):
                data["P0010001"] = data["TOTPOP"]
        
        
    p_graph.to_json(title)
    print(state, "precinct graph saved")


# In[3]:







