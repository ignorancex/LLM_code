import pandas as pd
import scipy.io
import numpy as np
from pyvis.network import Network
import matplotlib.pyplot as plt
import collections
import json
import matlab.engine
from sklearn.linear_model import LinearRegression
import os
import sys
import julia
import subprocess

def modify_CATS(change_solar=True, rescale_imports=True, remove_imports=True):
    """
    Load the generator and bus data. Remove renewable generators from the generator dataset. All these datasets come fron CATS paper.
    """
    bus = scipy.io.loadmat('MATPOWER/bus.mat')['bus']
    power_demand = bus[:, 2]
    df = pd.read_csv("GIS/CATS_gens.csv").to_numpy()
    #df = df[df['Pmax'] != 0.0].to_numpy()
    #old_num_gen = len(df)
    #df = df.drop_duplicates(subset=['PlantCode','GenID']).to_numpy()
    indices = []
    idx = []
    impt_idx = []
    impt_indices = []
    renew_type = ["Solar Photovoltaic", "Solar Thermal without Energy Storage", "Onshore Wind Turbine", "Hydroelectric Pumped Storage", "Conventional Hydroelectric"]
    for i, r in enumerate(df):
        if r[3] in renew_type:
            indices.append(i+1)
            idx.append(i)
        if r[3] == "IMPORT":
            impt_idx.append(i)
            impt_indices.append(i+1)
    #bus_loc = pd.read_csv("GIS/CATS_buses.csv").to_numpy()[:, [4,5]]
    #gen_loc = df[:, -2:]
    eng = matlab.engine.start_matlab()
    if change_solar:
        df[idx, 5] = 0
        g = eng.removeSolar(indices)
    if rescale_imports:
        g = eng.rescaleImport(impt_idx)
    if remove_imports:
        g = eng.removeSolar(impt_indices)
        df[impt_idx, 5] = 0
    m = eng.loadcase('CaliforniaTestSystem.m')
    num_bus = np.array(m['bus']).shape[0]
    num_branch = np.array(m['branch']).shape[0]
    num_gen = np.array(m['gen']).shape[0]
    #gen_bus_temp = np.array(m['gen'])[:,0]
    eng.quit()
    return num_bus, num_branch, num_gen, df

def gen_emission():
    '''
    Dictionary to store fuel type to emission rate.
    '''
    type_to_emission = collections.defaultdict(float)
    #type_to_emission['Conventional Hydroelectric'] = 0
    #type_to_emission['Hydroelectric Pumped Storage'] = 0
    type_to_emission['Petroleum Liquids'] = 0.656
    type_to_emission['Natural Gas Internal Combustion Engine'] = 0.44
    type_to_emission['Natural Gas Fired Combined Cycle'] = 0.44
    type_to_emission['Natural Gas Steam Turbine'] = 0.44
    type_to_emission['Natural Gas Fired Combustion Turbine'] = 0.44
    type_to_emission['Nuclear'] = 0
    type_to_emission['Geothermal'] = 0.038
    #type_to_emission['Onshore Wind Turbine'] = 0
    type_to_emission['Other Waste Biomass'] = 0.23
    type_to_emission['Wood/Wood Waste Biomass'] = 0.23
    type_to_emission['Landfill Gas'] = 0.11
    #type_to_emission['Solar Photovoltaic'] = 0
    #type_to_emission['Solar Thermal without Energy Storage'] = 0
    type_to_emission['Conventional Steam Coal'] = 0.82
    type_to_emission['Other Gases'] = 0.776560
    type_to_emission['Batteries'] = 0
    type_to_emission['Petroleum Coke'] = 0.656
    type_to_emission['Municipal Solid Waste'] = 0.029702
    type_to_emission['Other Natural Gas'] = 0.44
    type_to_emission['All Other'] = 0.104261
    type_to_emission['IMPORT'] = 0.43
    return type_to_emission

def gen_fuel(df):
    '''
    Check the total power generation by fuel type.
    '''
    gen_by_fuel = collections.defaultdict(float)
    for row in df:
        gen_by_fuel[row[3]] += row[5]
    return gen_by_fuel

def branch_data():
    '''
    Load branch data. Create dictionarys that map branch to bus.
    '''
    branch_ = scipy.io.loadmat('MATPOWER/branch.mat')['brach']
    branch_from_bus = list(map(int, branch_[:, 0]-1))
    branch_to_bus = list(map(int, branch_[:, 1]-1))
    line_to_nodes = [list(map(int,i)) for i in branch_[:, 0:2]-1]
    return line_to_nodes

def match_caiso(gen_by_fuel):
    '''
    Match the fuel type in our data to CAISO fuel type.
    '''
    fuel_type_to_caiso = collections.defaultdict(str)
    for k, v in gen_by_fuel.items():
        if 'Natural Gas' in k:
            fuel_type_to_caiso[k] = 0 #'Natural Gas'
        elif 'Other Gases' in k or 'Landfill' in k:
            fuel_type_to_caiso[k] = 11 #'Biogas'
        elif 'Coal' in k:
            fuel_type_to_caiso[k] = 5 #'Coal'
        elif 'Biomass' in k or 'Waste' in k:
            fuel_type_to_caiso[k] = 10 #'Biomass'
        elif 'Geothermal' in k:
            fuel_type_to_caiso[k] = 9 #'Geothermal'
        elif 'Batteries' in k:
            fuel_type_to_caiso[k] = 3
        elif 'uclear' in k:
            fuel_type_to_caiso[k] = 4
        elif 'Solar' in k:
            fuel_type_to_caiso[k] = 7
        elif 'Wind' in k:
            fuel_type_to_caiso[k] = 8
        elif 'Hydro' in k:
            fuel_type_to_caiso[k] = 1
        elif 'IMPORT' in k:
            fuel_type_to_caiso[k] = 2
        else: #others
            fuel_type_to_caiso[k] = 6
    return fuel_type_to_caiso

'''
After finding the SCC and store it in c, turn the cycle to a super node and turn graph into DAG.
'''
def update_graph_DAG(graph, c, graph_reverse, branch_power_from):
    for b in c:
        for nei, line in graph[b]:
            if nei in c:
                branch_power_from[line] = 0
                graph[b].remove((nei, line))
        for nei, line in graph_reverse[b]:
            if nei in c:
                branch_power_from[line] = 0
                if (b, line) in graph[nei]:
                    graph[nei].remove((b, line))
        if b != c[0]:
            for nei, line in graph[b]:
                if nei not in c:
                    graph[c[0]].append((nei, line))
                    graph[b].remove((nei, line))
            for nei, line in graph_reverse[b]:
                if nei not in c:
                    graph[nei].append((c[0], line))
                    if (b, line) in graph[nei]:
                        graph[nei].remove((b, line))
            graph[b] = []
    return graph


def get_load(ng=True, ng_im=True, total=True):
    '''
    Get the CAISO load total power demand data at different timestamps. real_gen only includes non-renew while total_dem includes all.
    '''
    if ng_im:
        some_gen = pd.read_csv('data/201906/CAISO-netdemand-20190617.csv').iloc[[7]].to_numpy().reshape((290, ))[1:-1]
        ng_im_load = []
        for i in range(0,288):
            if i%12 == 0:
                ng_im_load.append(some_gen[i])
    if total:
        some_gen = pd.read_csv('data/201906/CAISO-netdemand-20190617.csv').iloc[[2]].to_numpy().reshape((290, ))[1:-1]
        total_load = []
        for i in range(0,288):
            if i%12 == 0:
                total_load.append(some_gen[i])
    if ng:
        some_gen = pd.read_csv('data/201906/CAISO-netdemand-20190617.csv').iloc[[5]].to_numpy().reshape((290, ))[1:-1]
        ng_load = []
        for i in range(0,288):
            if i%12 == 0:
                ng_load.append(some_gen[i])
    return ng_load, ng_im_load, total_load

def get_l_rn():
    num_bus, num_branch, num_gen, df = modify_CATS(change_solar=True, rescale_imports=True, remove_imports=False)
    gen_by_fuel = gen_fuel(df)
    fuel_type_to_caiso = match_caiso(gen_by_fuel)
    type_to_emission = gen_emission()
    line_to_nodes = branch_data()
    l_rn_time = []
    l_ori_time = []
    carbon_vec_time = []
    num_fuel_type = 12
    power_generation_time = []
    #res = []
    #emi_by_type = []
    pg_time_by_type = []
    eng = matlab.engine.start_matlab()
    h = 0
    _, ng_im_load, total_load = get_load(ng=True, ng_im=True, total=True)
    # counties = pd.read_csv("Result/carbon_intensity_0617_27.csv").to_numpy()[:,3]
    for i in range(len(ng_im_load)):
        #qd = np.concatenate((qd, np.zeros((1743,))))
        r = eng.myFunc(ng_im_load[i])
        #m = eng.loadcase('CaliforniaTestSystem.m')
        #power_d = np.array(m['bus'])[:,2]
        #np.array(m['bus'])[:,2] = power_demand
        #np.array(m['gen'])[:,1] = power_generation
        #eng.savecase('CaliforniaTestSystem.m', m)
        subprocess.run(["julia", "run_opf.jl"])
        f = open("pf_solution.json")
        sol = json.load(f)
        # gen to bus
        gen = [-1]*num_gen
        gen_cost = [0]*num_gen
        power_generation = [0]*num_gen
        carbon_emission = [0]*num_gen
        power_generation_by_type = [0]*num_fuel_type
        emission_by_type = [0]*num_fuel_type
        branch_power_to = [0]*num_branch
        branch_power_from = [0]*num_branch
        for line, val in sol['solution']['gen'].items():
            if val['pg'] > 0.0:
                gen[int(line)-1] = df[int(line)-1][2]-1
                power_generation_by_type[fuel_type_to_caiso[df[int(line)-1][3]]] += val['pg']*100
                carbon_emission[int(line)-1] = type_to_emission[df[int(line)-1][3]]
                gen_by_fuel[df[int(line)-1][3]] += val['pg']*100
                gen_cost[int(line)-1] = val['pg_cost']
                power_generation[int(line)-1] = val['pg']*100
                emission_by_type[fuel_type_to_caiso[df[int(line)-1][3]]] += val['pg']*100*type_to_emission[df[int(line)-1][3]]

        for line, val in sol['solution']['branch'].items():
            branch_power_from[int(line)-1] = round(val['pf']*100, 4)
            branch_power_to[int(line)-1] = round(val['pt']*100, 4)
        power_generation_time.append(power_generation)
        pg_time_by_type.append(power_generation_by_type)
        #emi_by_type.append(emission_by_type)
        f.close()
        r = eng.myFuncNoModi(total_load[i])
        m2 = eng.loadcase('CaliforniaTestSystem2.m')
        l_ori = np.array(m2['bus'])[:,2]
        power_demand = np.array(m2['bus'])[:,2]
        #print("power demand: " + str(sum(power_demand)))
        #print("power gen: " + str(sum(power_generation)))
        #print("real gen: " + str(total_load[i]))
        graph = collections.defaultdict(list) # from: (to, line)
        graph_reverse = collections.defaultdict(list) # to: (from, line)
        for i, (from_bus, to_bus) in enumerate(line_to_nodes):
            graph[from_bus].append((to_bus, i))
            graph_reverse[to_bus].append((from_bus, i))
        for i, f in enumerate(branch_power_from):
            if f < 0.0:
                from_node, to_node = line_to_nodes[i]
                graph[from_node].remove((to_node, i))
                graph[to_node].append((from_node, i))
                graph_reverse[to_node].remove((from_node, i))
                graph_reverse[from_node].append((to_node, i))
                branch_power_from[i] = -f
        UNVISITED = -1
        id = [0]
        sccCount = [0]
        ids = [0]*num_bus
        low = [0]*num_bus
        onStack = [False]*num_bus
        stack = []
        def findSccs():
            for i in range(num_bus): ids[i] = UNVISITED
            for i in range(num_bus):
                if ids[i] == UNVISITED:
                    tarjan_dfs(i)
            return low
        def tarjan_dfs(at):
            stack.append(at)
            onStack[at] = True
            ids[at] = id[0]
            low[at] = id[0]
            id[0] += 1
            for nei, _ in graph[at]:
                if ids[nei] == UNVISITED:
                    tarjan_dfs(nei)
                if onStack[nei]:
                    low[at] = min(low[nei], low[at])
            if ids[at] == low[at]:
                while stack:
                    node = stack.pop(-1)
                    onStack[node] = False
                    low[node] = ids[at]
                    if node == at: break
                sccCount[0] += 1
        #print(pg_time_by_type)
        sccs = collections.defaultdict(list)
        seen = []
        edges = findSccs()
        for i, v in enumerate(edges):
            if v in seen:
                sccs[v].append(i)
                sccs[v].append(seen.index(v))
            seen.append(v)
        cycles = set()
        for k, v in sccs.items():
            cycles.add(frozenset(v))
        for c in cycles:
            c = list(c)
            total_demand = 0
            total_gen = 0
            for b in c:
                total_demand += power_demand[b]
                idx = gen.index(b) if b in gen else -1
                if idx != -1:
                    total_gen += power_generation[idx]
            power_demand[c[0]] = total_demand
            idx = gen.index(b) if b in gen else -1
            if idx != -1:
                power_generation[gen.index(c[0])] = total_gen
            graph = update_graph_DAG(graph, c, graph_reverse, branch_power_from)
        graph_reverse = collections.defaultdict(list)
        for f, v in graph.items():
            for t, line in v:
                graph_reverse[t].append((f, line))
        # Since the power generation can be negative, we can convert them to power demand instead
        for i, v in enumerate(power_generation):
            if v < 0.0:
                power_demand[gen[i]] -= v
                power_generation[i] = 0.0
        line_to_gen = collections.defaultdict(set)
        node_to_gen = collections.defaultdict(set)
        def dfs(i, b, visited):
            if b in visited: return
            visited.add(b)
            node_to_gen[b].add(i)
            for nei, line in graph[b]:
                line_to_gen[line].add(i)
                dfs(i, nei, visited)
        for i, b in enumerate(gen):
            visited = set()
            dfs(i, b, visited)
        line_prop_mat=np.zeros((num_gen, num_branch), dtype=float)
        bus_prop_mat=np.zeros((num_gen, num_bus), dtype=float)
        in_degree = collections.defaultdict(int)
        for i, v in graph_reverse.items():
            in_degree[i] = len(v)
        q = set() # list of nodes with no inflow
        topo_order = []
        for b in range(num_bus):
            if in_degree[b] == 0:
                q.add(b)
        q = list(q)
        visited = set()
        while q:
            cur = int(q.pop(0))
            visited.add(cur)
            #if len(node_to_gen[cur]) > 1:
            out_total = power_demand[cur]
            for nei, out_line in graph[cur]:
                out_total += branch_power_from[out_line]
            
            for g in node_to_gen[cur]:
                if cur == gen[g]:
                    if out_total > 0.0: 
                        bus_prop_mat[g][cur] += power_generation[g]/out_total
                for nei, in_line in graph_reverse[cur]:
                    if out_total > 0.0:
                        bus_prop_mat[g][cur] += branch_power_from[in_line]*line_prop_mat[g][in_line]/out_total
            for g in node_to_gen[cur]:
                for nei, out_line in graph[cur]:
                    line_prop_mat[g][out_line] = bus_prop_mat[g][cur]
            topo_order.append(cur)
            for nei, line in graph[cur]:
                in_degree[nei] -= 1
                if in_degree[nei] == 0 and nei not in visited:
                    q.append(nei)
        #avg_carbon_emission_rate_node = carbon_emission @ bus_prop_mat
        #carbon_vec = np.zeros((num_bus, 1), dtype=float)
        '''bus_prop_mat_2 = np.zeros((num_gen, num_bus), dtype=float)
        for i in range(num_gen):
            s = np.sum(bus_prop_mat[i])
            for j in range(num_bus):
                if s != 0.0:
                    bus_prop_mat_2[i][j] = bus_prop_mat[i][j]/s
        carbon_vec = np.zeros((num_bus, 1), dtype=float)
        for i in range(num_bus):
            for j in range(num_gen):
                carbon_vec[i] += bus_prop_mat_2[j,i] * power_generation[j] * carbon_emission[j]'''
        l_ren = [0]*num_bus
        for n, v in graph.items():
            total_out = 0
            total_in = 0
            for (n_out, line_out) in v:
                total_out += branch_power_from[line_out]
            for (n_in, line_in) in graph_reverse[n]:
                total_in += branch_power_from[line_in]
            if total_out - total_in > 0.001 or total_out - total_in < -0.001:
                l_ren[n] = total_out - total_in
        #l_ren = np.reshape(l_ori, (1, num_bus)) - np.matmul(np.reshape(power_generation, (1, num_gen)), bus_prop_mat_2)
        #l_ren = np.reshape(l_ren, (1, num_bus)).tolist()[0]
        l_ori = np.reshape(l_ori, (1, num_bus)).tolist()[0]
        l_rn_time.append(l_ren)
        l_ori_time.append(l_ori)
        #carbon_vec_time.append(np.reshape(carbon_vec, (1, num_bus)).tolist()[0])
        h += 1
    eng.quit()
    return pg_time_by_type, power_generation_time, l_ori_time, l_rn_time#, carbon_vec_time