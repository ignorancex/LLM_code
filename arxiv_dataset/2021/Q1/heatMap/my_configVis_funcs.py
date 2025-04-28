import importlib
import setup_nx # your own module, setup.nx.py
import random

importlib.reload(setup_nx)
from setup_nx import *
from graphviz import render
import matplotlib.pyplot as plt
import my_feeder_funcs as ff
import my_heatmapSetup_funcs as hm
import re # regex

def markActuatorConfig(lst_act_locs, feeder, file_name):
    #creates diagram of actuator-configuration (though leaves out performance nodes)
    #lst_act_locs = list of node names where actuators are placed
    #file_name = string that will be used as the file name of the returned network graph
    ff.clear_graph(feeder)
    graph = feeder.network
    
    for loc in lst_act_locs:
        graph.nodes[loc]['style'] = 'filled'
        graph.nodes[loc]['fillcolor'] = 'indigo'
        
    write_formatted_dot(graph, 'actConfig_'+ file_name)

    return

def markMultiple_actConfig(group_of_lsts,feeder,file_name):
    #creates diagram of actuator-configuration (though leaves out performance nodes)
    # lst1 = actautors that will be colored first
    # lst2 = actutors that will be colored in second 
    #file_name = string that will be used as the file name of the returned network graph
    ff.clear_graph(feeder)
    graph = feeder.network

    colors=['indigo','tomato','webgreen']
    assert len(group_of_lsts)<=len(colors)

    for i in range(len(group_of_lsts)):
        lst=group_of_lsts[i]
        for loc in lst:
            graph.nodes[loc]['style'] = 'filled'
            graph.nodes[loc]['fillcolor'] = colors[i]
        
    write_formatted_dot(graph, 'actConfig_'+ file_name)
    return

def markActLoc(graph, act_loc):
    #changes color of nodes with set actuators to gray
    #graph = networkx graph object (feeder.network)
    #act_loc = node name as string where actuator is placed
    graph.nodes[act_loc]['style'] = 'filled'
    graph.nodes[act_loc]['fillcolor'] = 'indigo'
    graph.nodes[act_loc]['shape'] = 'circle'
    return

def markFeas(range_vec,test_value,test_act_loc, graph):
    #if controllability can be achieved with actuator at test_act_loc then mark green, if only a few controllable configurations (given by thresh_yellowgreen) exist mark yellow, otherwise mark red
    #feas = True or False
    #test_act_loc = node name as string
    #graph = networkx graph object (feeder.network)
    graph.nodes[test_act_loc]['style'] = 'filled'
    graph.nodes[test_act_loc]['shape'] = 'circle'

    assert range_vec[1]<=1 # gradient should be from 0 to 1
    level=(test_value-range_vec[0])/(range_vec[1]-range_vec[0]) # percentage indicating where you are in the range
    assert level>=0 and range_vec[1]>range_vec[0]

    thresh1,thresh2=0.34,0.67
    level_vals=[range_vec[0],round(range_vec[0]+thresh1*(range_vec[1]-range_vec[0]),4),round(range_vec[0]+thresh2*(range_vec[1]-range_vec[0]),4),range_vec[1]]

    # for more colors see https://graphviz.org/doc/info/colors.html
    if round(test_value,3) >= 1: # anything at 0.9995 and above
        graph.nodes[test_act_loc]['fillcolor'] = 'crimson'
    elif level<thresh1: #Lower half of stable domeigs
        graph.nodes[test_act_loc]['fillcolor'] = 'yellow'
    elif thresh1<level < thresh2:  # Lower half of stable domeigs
        graph.nodes[test_act_loc]['fillcolor'] = 'gold'
    else:
        graph.nodes[test_act_loc]['fillcolor'] = 'darkorange'
    return level_vals

def write_formatted_dot(graph, file_name):
    # fix fontsize to 20
    for node in graph.nodes:
        graph.nodes[node]['fontsize'] = 20

    for edge in graph.edges: # line, switch, or xfmr objects
        graph.edges[edge]['penwidth'] = 3 # increase thickness of all edges
        conn=graph.edges[edge]['connector']
        if hasattr(conn, 'from_phases'): # switches and lines, but not xfmrs, have from_phase attribute
            numph=min([len(conn.from_phases),len(conn.to_phases)])
            if numph>1: # make 2 and 3ph lines darker grey edges
                #graph.edges[edge]['style'] = 'dashed'
                graph.edges[edge]['color'] = "#585858"

    # specify node separation (horiz) and rank separation (vertical)
    nx.nx_pydot.write_dot(graph, 'generated_figs/'+file_name)
    f_old = open('generated_figs/'+file_name)
    f_new = open('generated_figs/'+file_name+'wide', "w")
    for line in f_old:
        f_new.write(line)
        if 'strict digraph  {' in line:
            f_new.write("nodesep=0.45;\nranksep=0.4;\n") # nodesep is horizontal separation, ranksep is vertical separation
    #
    # pos2 = dict((land, position[land]) for land in graph.nodes())
    # collection = nx.draw_networkx_edges(graph, pos2, width=2, edge_color='grey')
    # for patch in collection:
    #     patch.set_linestyle('dotted')
    #
    # nx.draw_networkx_nodes(G, pos, with_labels=True, node_size=[[200, 20, 20, 200, 20]], node_color='red')
    #
    # collection = nx.draw_networkx_edges(G, pos, color='white', width=[7, 3, 3, 3, 3], edge_color='b')
    # for patch in collection:
    #     patch.set_linestyle('dashdot')

    f_old.close()
    f_new.close()        
    
    # convert .dot to .png
    render('dot', 'png', 'generated_figs/'+file_name+'wide')
    
    return

def make_map(v,cur_act_locs,hm_dic,vals_range,tag_name):
        # make heatmap
        ff.clear_graph(v.feeder)  # clear any previous modifictions made to graph
        graph = v.feeder.network

        for act in cur_act_locs:  # mark placed acts in grey
            markActLoc(graph, act)
        for k, val in hm_dic.items():
            level_vals=markFeas(vals_range, val[1], k, graph)
        heatMapName = tag_name+'_heatmap_'+ v.file_name
        write_formatted_dot(graph, heatMapName)
        return level_vals # threshold domeig values between colormap colors

def write_formatted_dot2(graph, file_name): # special version, for ecoblock report
    # remove node labels, widen graph
    
    # fix fontsize to 20
    for node in graph.nodes:
        graph.nodes[node]['fontsize'] = 20
       
    # specify node separation (horiz) and rank separation (vertical)
    nx.nx_pydot.write_dot(graph, 'generated_figs/'+file_name)
    f_old = open('generated_figs/'+file_name)
    f_new = open('generated_figs/'+file_name+'wide', "w")
    for line in f_old: # go through every line of dot file
        f_new.write(re.sub("xlabel=n\d+", "",line)); # regex, remove node labels
        if 'strict digraph  {' in line:
            f_new.write("nodesep=0.7;\nranksep=0.25;\n") # nodesep is horizontal separation, ranksep is vertical separation
            
    f_old.close()
    f_new.close()        
    
    # convert .dot to .png
    render('dot', 'png', 'generated_figs/'+file_name+'wide')
    
    return

def markCommonFeasNodes(lst_feas_configs, feeder, file_name):
    #input a list of feasible actuator configurations where each configuration is its own list (ie a list of lists)
    #for actuator locations that appear in every configuration, mark node gray
    shared_locs = lst_feas_configs[0]
    num_configs = len(lst_feas_configs)
    graph = feeder.network
    
    for i in range(num_configs):
        if i <= num_configs - 2:
            cur_config = lst_feas_configs[i + 1]
            copy_shared_locs = shared_locs
            for loc in copy_shared_locs:
                if loc not in cur_config:
                    shared_locs.remove(loc)
    
    for act_loc in shared_locs:
        graph.nodes[act_loc]['style'] = 'filled'
        graph.nodes[act_loc]['fillcolor'] = 'indigo'
        
    write_formatted_dot(graph, 'shared_act_locs_'+ file_name)

    return shared_locs


def find_ActLoc_in_FeasConfigs(act_loc, feas_configs):
    #returns list of configurations in feas_configs that contain act_loc
    #act_loc = node name of potential actuator location
    #feas_configs = a list of feasible actuator configurations
    configs_with_actLoc = []
    
    for config in feas_configs:
        if act_loc in config:
            configs_with_actLoc += [config]
    return configs_with_actLoc
    

def plot_actuator_num_histogram(lst_feas_configs, file_name):
    #input list of lists of feasible actuator configurations
    #output histogram reflecting number of actuators in each configuration
    lst_num_acts = []
    for config in lst_feas_configs:
        lst_num_acts += [len(config)]
        
    bins_lst = range(max(lst_num_acts) + 2)
        
    plt.title("Number of Actuators per Configuration")
    plt.xlabel("Number of Actuators per Configuration")
    plt.ylabel("Number of Feasible Configurations")
    plt.hist(lst_num_acts, bins = bins_lst, edgecolor = 'black', align = 'left')
    plt.savefig(file_name + '_' + 'Histogram.png') 
    plt.show() # need to savefig before plt.show
    return


def assign_network_branches1(feeder, substation_name):
    #used in multiphase line loss func
    #feeder = initialized feeder
    #substation_name = name of substation node as string
    #devides network into branches
    cur_child_nodes = list(feeder.network.successors(substation_name))
    branches = []
    branch_builder = [substation_name]
    
    while len(cur_child_nodes) == 1:
        branch_builder += cur_child_nodes
        cur_child_nodes = list(feeder.network.successors(cur_child_nodes[0]))
            
    if len(cur_child_nodes) > 1:
        branches += [branch_builder]
        for child in cur_child_nodes:
            branches += assign_network_branches1(feeder, child)
            
    else:
        branches += [branch_builder]
        return branches
    
    return branches  


def assign_network_branches(feeder, substation_name):
    #used for general visualization
    #feeder = initialized feeder
    #substation_name = name of substation node as string
    #devides network into branches
    cur_child_node = substation_name
    branches = []
    branch_builder = []
    
    while cur_child_node:
        branch_builder += [cur_child_node]
        all_children = list(feeder.network.successors(cur_child_node)) # get nodes below current node
        
        if len(all_children) > 1: # if more than one node below
            if 'bus_28' in all_children: # if children contains 28, want 28 to start its own branch
                cur_child_node='bus_26'
            elif 'bus_62' in all_children: # if children contains 62, want 28 to start its own branch
                cur_child_node='bus_61'
            else:
                cur_child_node = max(all_children) # choose rightmost (max) branch
                
            all_children.remove(cur_child_node)
            new_branch_heads = all_children
            
            for head in new_branch_heads:
                branches += assign_network_branches(feeder, head)
                    
        elif len(all_children) == 1:
            cur_child_node = all_children[0]
            
        else:
            break
    
    branches += [branch_builder]
    return branches


def mark_network_branches(feeder, branch_lst, file_name, substation_name, depths):
    #marks network branches in different colors so they canbe visualized
    #feeder = initialized feeder object
    #branch_lst = a list of lists containing node names as strings of nodes that are in the same branch
    graph = feeder.network
    colors = ['turquoise', 'purple', 'limegreen', 'yellow','pink', 'orange', 'gray', 'red', 'orchid']
    node_colors = {}
    branch_heads = [branch[0] for branch in branch_lst]
    head_depths = [depths[h] for h in branch_heads]
    unique_depths = []
    
    for d in head_depths:
        if d not in unique_depths and d != 0:
            unique_depths += [d]
    
    subst_branch = [b for b in branch_lst if substation_name in b][0]  
    for node in subst_branch:
        node_colors[node] = colors[0]
        
    while unique_depths:
        cur_depth = min(unique_depths)
        unique_depths.remove(cur_depth)
        cur_heads = [head for head in branch_heads if depths[head] == cur_depth]
        used_colors = []
        for head in cur_heads:
            cur_branch = [b for b in branch_lst if head in b][0]
            parent_node = list(graph.predecessors(head))
            parent_color = node_colors[parent_node[0]]
            color_choices = list(colors)
            color_choices.remove(parent_color)
            for uc in used_colors:
                if uc in color_choices:
                    color_choices.remove(uc)
            cur_color = random.choice(color_choices)
            for node in cur_branch:
                node_colors[node] = cur_color
            used_colors += [color_choices[0]]
        
    for node in graph.nodes:
        graph.nodes[node]['style'] = 'filled'
        graph.nodes[node]['fillcolor'] = node_colors[node]
    
    write_formatted_dot(graph, 'branch_key:'+ file_name)

    
    return
        

def find_good_branches(lst_feas_configs, branch_lst, num_good_branches, placed):
    #branch_lst = list of branches for a certain feeder --> get by calling assign_network_branches()
    #lst_feas_configs = list of feasible actuator configurations --> get by calling runHeatMapProcess()
    #num_good_branches = number of 'good' branches you want returned by the function, argument should be an integer, branches returned in order from 'best' to 'worst'
    #placed is an ordered list of lists, where the k'th ele of the list is the actuators placed at that step

    #the 'best' branch is done here with respect to 2 metrics:
    #1) best_branches_unique, is where a branch is "upvoted" once when a feas config contains at least one act on that branch
    #2) best_branches_percent, is where a branch is best if it has highest percentage of green/yellow (excluding gray) in heatmap across all heatmaps used to create the config set
    
    branch_dic_unique,branch_dic_gray,percent_good = {},{},{}
    best_branches_unique,best_branches_percent = [],[]
    worst_branches_unique,worst_branches_percent=[],[]
    
    for branch in branch_lst:
        branch_head = branch[0]
        branch_dic_unique[branch_head] = 0
        branch_dic_gray[branch_head] = 0
        percent_good[branch_head] = 0
    
    num_hm=len(placed)-1 # number of heatmaps used to create config set
    flag=np.zeros(num_hm)
    
    for feas in lst_feas_configs:
        unique_tracker = 0 # tracks whether or not an a branch has already been represented in a certain configuration
        totact=len(feas)
        for act_loc in feas: 
            for branch in branch_lst:
                branch_head = branch[0]
                if (act_loc not in placed[totact-1]): # then green or yellow on heatmap
                    if (act_loc in branch) and (unique_tracker == 0): # exclude actuators already placed from the count on branch usage
                        branch_dic_unique[branch_head] += 1
                        unique_tracker = 1
                        break # found which branch the act is on

    for placed_act in placed: # for each act in list of placed actuators
        for branch in branch_lst:
            branch_head = branch[0]
        if (placed_act in branch):
            branch_dic_gray[branch_head]+=1 # increment the gray property for branch those placed actuators is on
            break # found which branch the act is on
            
     # Compute (green+yellow)/(green+red+yellow+firebrick)   
    for branch in branch_lst:
        branch_head = branch[0]
        gy=branch_dic_unique[branch_head]
        gry_gray=len(branch)*(num_hm) # gives us green+red+yellow+gray+firebrick
        gray=branch_dic_gray[branch_head]
        if (gy/(gry_gray-gray)<0):
            print('problem at branch ',branch)
            print('gry+gray=',gry_gray)
            print('gray=',gray)
        if len(branch)<4: # don't count branches that are too short (i.e. <4 nodes)
            percent_good.pop(branch_head) # remove
            #branch_dic_unique.pop(branch_head)
        else:
            percent_good[branch_head]=gy/(gry_gray-gray)
        
    print('number of branches with len>3: ',len(percent_good),'\n')
    print('percent good dictionary is=',percent_good,'\n')

    # common_branches = branches that have at least one actuator placed on them in every feas config
    common_branches = [k for k, v in branch_dic_unique.items() if v == len(lst_feas_configs)]
    
    counter = num_good_branches
    while (counter > 0 and len(percent_good)>0): # break out of loop if percent_good is empty
        max_unique = max(branch_dic_unique, key = branch_dic_unique.get)
        max_percent = max(percent_good, key = percent_good.get)
        min_unique = min(branch_dic_unique, key = branch_dic_unique.get)
        min_percent = min(percent_good, key = percent_good.get)
        
        best_branches_unique += [max_unique + ' = ' + str(branch_dic_unique[max_unique]) + ' actuators']
        best_branches_percent += [max_percent + ' = ' + str(100*percent_good[max_percent]) + ' percent']
        worst_branches_unique += [min_unique + ' = ' + str(branch_dic_unique[min_unique]) + ' actuators']
        worst_branches_percent += [min_percent + ' = ' + str(100*percent_good[min_percent]) + ' percent']
      
        
        branch_dic_unique.pop(max_unique) # remove these branches for next run because already labeled as good/bad
        percent_good.pop(max_percent)
        if min_unique in branch_dic_unique: # need has_key check in case max and min are same branch
            branch_dic_unique.pop(min_unique)
        if min_percent in percent_good:
            percent_good.pop(min_percent)
        
        counter -= 1
    
    print('Branches are represented by their first node (the node closest to the substation).')
    print('\nBranches Common Across Configs:')
    print(common_branches)
    print('\nThe ' + str(num_good_branches) + ' best branches when each actuator configuration is only considered once:')
    print(best_branches_unique) # "upvoted" once when a feas config contains at least one act on that branch
    print('\nThe ' + str(num_good_branches) + ' best branches by percentage of green/yellow (excluding <=3 len branches):')            
    print(best_branches_percent)
    print('\nThe ' + str(num_good_branches) + ' worst branches when each actuator configuration is only considered once:')
    print(worst_branches_unique) # "upvoted" once when a feas config contains at least one act on that branch
    print('\nThe ' + str(num_good_branches) + ' worst branches by percentage of green/yellow (excluding <=3 len branches):')            
    print(worst_branches_percent)
    return

def determine_good_or_bad_branch(branch, lst_feas_configs, num_configs_for_good_branch):
    # branch = list of nodes in a branch
    # lst_feas_configs = list of feasible actuator configurations
    # num_configs_for_good_branch = the minimum number of configurations that must use a branch for the branch to be 'good' (argument should be an integer)
    # a feas configuration 'uses' a branch if the configuration includes an actuator that is placed on one of the nodes within the branch
    configs_with_branch = []
    
    for config in lst_feas_configs:
        
        for node in config:
            
            if node in branch:
                configs_with_branch += [config]
                break
                
    if len(configs_with_branch) >= num_configs_for_good_branch:
        print('Branch ' + str(branch) + ' is good.')
        
    else:
        print('Branch ' + str(branch) + ' is bad.')
    
    print('\nNumber of configurations that use the branch: ' + str(len(configs_with_branch)))
    return configs_with_branch      


def getPhases(branch,feeder): # create list of phases A/B/C, one for each node of branch
    #    branch_heads = [branch[0] for branch in branch_lst]
    phase_list = []
    #print(branch)

    for node in branch:
        pred_list = list(feeder.network.predecessors(node))
        try:
            edge = feeder.network.get_edge_data(pred_list[0], node, default=None)['connector']
        except:
            print("pred_list was empty")
        
        try:
            phase_list+=[edge.to_phases]
        except:
            print("Node is not a line or shunt")
    
        #print('nodes phases are=',impedance.to_phases)
    return phase_list          

    
def find_branch_in_branch_list(node_in_branch, branch_lst):
    #node_in_branch = a node in the branch you want to select
    #branch_lst = list of branches in a network
    #returns branch containing node_in_branch
    for branch in branch_lst:
        
        if node_in_branch in branch:
            my_branch = branch
            break
    return my_branch
            
    
def phaseCouplingPerNode(feeder, depths, file_name):
    #phase coupling = mutual impedance/ self impedance on EACH phase
    graph = feeder.network
    coupling_ratios_3ph = {} 
    graphNodes_nosub = hm.remove_subst_nodes(feeder, file_name) #dont consider substation nodes, node 650 and 651 for 13NF
    
    for node in graphNodes_nosub:
        edge_path = hm.get_total_impedance_from_substation(feeder, node, depths)
        couplingRatio_runSum = np.zeros((1, 3), dtype = complex)
        #print('---------------new path-----------------')

        phA_count,phB_count,phC_count=0,0,0 # keep track of number of lines in each phase along path to subst
        for edge in edge_path:
            rat_A,rat_B,rat_C=0,0,0
            impedance_test = graph.get_edge_data(edge[1], edge[0], default=None)['connector']
            impedance = impedance_test.Z if isinstance(impedance_test, setup_nx.line) else np.zeros((3, 3))
            #print('Z=',impedance)
            self_imped_A = impedance[0][0] 
            self_imped_B = impedance[1][1]
            self_imped_C = impedance[2][2]

            mutual_imped_A = impedance[0][1] + impedance[0][2] 
            mutual_imped_B = impedance[1][0] + impedance[1][2]
            mutual_imped_C = impedance[2][0] + impedance[2][1]
            
            if self_imped_A != 0:
                rat_A = mutual_imped_A/self_imped_A
                phA_count+=1
            else:
                ratA=0
            if self_imped_B != 0:
                rat_B = mutual_imped_B/self_imped_B
                phB_count+=1
            else:
                ratB=0
            if self_imped_C != 0:
                rat_C = mutual_imped_C/self_imped_C
                phC_count+=1
            else:
                ratC=0
            
            couplingRatio_runSum += np.array([[rat_A, rat_B, rat_C]]) # 3x1 vec of complex
            #print('3ph_rat=',couplingRatio_runSum)

        #print('[phA_count,phB_count,phC_count]=',[phA_count,phB_count,phC_count])
        coupling_ratios_3ph[node] = np.divide(couplingRatio_runSum,[phA_count,phB_count,phC_count]) # take average of coupling ratios across edges to substation.
            
    #coupling ratios is a dictionary with node names as keys and phase coupling ratios per node as values
    return coupling_ratios_3ph


def createColorMap(feeder, values_dic, file_name):
    #takes in a dictionary of real values with node names as keys and devides the nodes into 8 color bins based on their values
    #generates png file with feeder nodes colored based on which bin they fall into
    graph = feeder.network
    ff.clear_graph(feeder)
    vals = list(values_dic.values())
    if isinstance(vals[0], np.ndarray):
        vals = [np.mean(val) for val in vals]
    if isinstance(vals[0], complex):
        vals = [(val.imag**2 + val.real**2)**(1/2) for val in vals]
    maxVal = max(vals)
    minVal = min(vals)
    bin_size = (maxVal - minVal)/8
    bin_edges = [[minVal + (i*bin_size), minVal + ((i + 1)*bin_size)] for i in range(8)]
    
    # color reference: https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    colors = ['darkgreen','limegreen', 'yellowgreen','yellow','gold', 'orange','tomato','firebrick']
    bins_with_clrs = [bin_edges[i] + [colors[i]] for i in range(8)]
    print('Color Bin Key for',file_name)
    for b in bins_with_clrs:
        print(str(b[2]) + ': (' + str(b[0]) + ' --> ' + str(b[1]) +')')
        
    for node, val in values_dic.items():
        if isinstance(val, np.ndarray):
            val = np.mean(val)
        if isinstance(val, complex):
            val = (val.imag**2 + val.real**2)**(1/2)
        color_num = val
        
        for b in bins_with_clrs:
            if color_num == minVal and b[0] == minVal:
                graph.nodes[node]['style'] = 'filled'
                graph.nodes[node]['fillcolor'] = b[2]
                break
            elif color_num > b[0] and color_num <= b[1]:
                graph.nodes[node]['style'] = 'filled'
                graph.nodes[node]['fillcolor'] = b[2]
                break
  
    write_formatted_dot(graph, 'colorMap_'+ file_name)

    return


def get_path_to_subst_as_list(feeder, node, substation_name):
    #returns list of nodes between node and substation
    #node = node name as string
    #feeder = initiaized feeder object
    graph = feeder.network
    node_path = [node]
    current_node = node
    
    while current_node != substation_name:
        pred_node = list(graph.predecessors(current_node)) #retrieves parent node of current_node
        node_path += pred_node
        current_node = pred_node[0]
    
    node_path += [substation_name]
    return node_path


def find_main_branch(feeder, substation_name, file_name):
    # returns the "main branch" of a feeder and creates a png file with the main branch marked in orange
    # main branch is the path from substation to an edge node that traces the maximum amount of intersection between the paths of every node in the network to the substation
    ff.clear_graph(feeder)
    main_branch = [substation_name]
    graph = feeder.network
    children = list(graph.successors(substation_name))
    nodes = graph.nodes
    paths_to_sub = []
    
    for node in nodes:
        paths_to_sub += [get_path_to_subst_as_list(feeder, node, substation_name)]
        
    while children != []:
        if len(children) == 1:
            main_branch += children
            children = list(graph.successors(children[0]))
        elif len(children) > 1:
            nodes_below = {}
            
            for child in children:
                num_nodes_below = len([1 for path in paths_to_sub if child in path])
                nodes_below[child] = num_nodes_below
            
            max_nodes_below = max(nodes_below, key = nodes_below.get)
            main_branch += [max_nodes_below]
            children = list(graph.successors(max_nodes_below))
    
    for node in main_branch:
        graph.nodes[node]['fillcolor'] = 'orange'
    
    write_formatted_dot(graph, 'main_branch_'+ file_name)

    return main_branch           



