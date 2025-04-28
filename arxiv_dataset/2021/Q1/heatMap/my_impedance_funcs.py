import importlib
import setup_nx # your own module, setup.nx.py
from itertools import combinations # Function which returns subset or r length from n
import random
import my_impedance_funcs as imp

importlib.reload(setup_nx)
from setup_nx import *
import matplotlib.pyplot as plt

"-----Notes on the following two methods-----"
#1)Does not account for the presence of artifically added transformer windings -- would have to separate out the line impedance information
#2)Will not function properly if there are cycles
#3)Assumes that switches and transformers have zero impedance; the latter case is due to unit discrepancy: transformer impedances use PU, line impedances use Ohms


def get_path_from_substation(feeder, node, depths):
    #returns list of edges (not impedances!) between node and substation
    #node = node name as string
    #feeder = initiaized feeder object
    graph = feeder.network
    node_path = []
    current_node = node

    for i in range(depths[node]): # start from given node and move node-by-node up to subst
        pred_node = list(graph.predecessors(current_node)) #retrieves parent node of current_node
        node_path += [(current_node, pred_node[0])]
        current_node = pred_node[0]
    return node_path

def get_total_impedance_from_substation(feeder, node_name, depths):
    #returns the 3x3 impedance matrix from a specified node to the substation 
    #Notice that since this is a tree, any node will only have at most one predecessor
    node_name_full = node_name
    total_impedance = np.zeros((3,3), dtype = complex)
    current_node = node_name_full
    pred_list = None
    
    try:
        pred_list = list(feeder.network.predecessors(node_name_full))
    except nx.NetworkXError:
        print("Bus with name " + current_node + " does not exist in the feeder.")
        return 0
    
    iter_depth = depths[current_node]
    for _ in range(iter_depth):
        #pred_list[0] is the parent node
        impedance = feeder.network.get_edge_data(pred_list[0], current_node, default=None)['connector']
        #print("Type is " + str(type(impedance)))
        if impedance == None:
            print("WARNING: No connection between nodes " + str(pred_list[0]) + " and " + str(current_node) + ".")
            return 0
        else:
            imp_dict = impedance.Z if isinstance(impedance, setup_nx.line) else np.zeros((3, 3), dtype = complex)
            total_impedance += imp_dict
            current_node = pred_list[0]
            pred_list = list(feeder.network.predecessors(current_node))
    
    return total_impedance


def get_common_node_impedance_two_buses(feeder,node_name_1,node_name_2,depths):
    # method returns the 3x3 complex impedance matrix of the z-path from common node (fork) to the substation
    # For example:
    #     F
    #     |
    #     A      Calculating common-node impedance between B and C yields Z_AF
    #    / \
    #   B   C
    bus_1 = node_name_1
    bus_2 = node_name_2
    common_z = np.zeros((3, 3), dtype=complex)

    pred_lst1=get_path_from_substation(feeder, bus_1, depths)
    pred_lst2=get_path_from_substation(feeder, bus_2, depths)
    # pred_lst is a list of tuples with deeper node first in tuple
    # in above example, first tuple is (bus_A,bus_F)

    def itct(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    common_lst=itct(pred_lst1,pred_lst2)

    return get_total_impedance_from_substation(feeder, common_lst[0][0], depths) # 3x3 complex mat
#
# def old_impedance_between_two_buses(feeder, node_name_1, node_name_2, depths):
#     #Method will return the distance sum of impedances to a common bus upstream if the two buses are not along the
#     #the same path. For example:
#     #     A      Calculating total impedance between B and C yields Z_AB + Z_CA
#     #    / \
#     #   B   C
#     bus_1 = node_name_1
#     bus_2 = node_name_2
#     total_impedance = np.zeros((3,3), dtype = complex)
#     depth_1 = 0
#     depth_2 = 0
#
#     try:
#         depth_1 = depths[bus_1]
#         depth_2 = depths[bus_2]
#     except KeyError:
#         print("Either the first bus, " + bus_1 + ", or the second bus, " + bus_2 + " is not a valid bus in the feeder.")
#         return 0
#
#     depth_dif = abs(depth_1 - depth_2)
#     max_depth_bus = bus_1 if depth_1 > depth_2 else bus_2
#     min_depth_bus = bus_1 if max_depth_bus == bus_2 else bus_2
#     pred_list_max = list(feeder.network.predecessors(max_depth_bus))
#     pred_list_min = list(feeder.network.predecessors(min_depth_bus))
#
#     for i in range(depth_dif):
#         impedance = feeder.network.get_edge_data(pred_list_max[0], max_depth_bus, default=None)['connector']
#         if impedance == None:
#             print("WARNING: No connection between nodes " + str(pred_list_max[0]) + " and " + str(max_depth_bus) + ".")
#             return 0
#         else:
#             imp_dict = impedance.Z if isinstance(impedance, setup_nx.line) else np.zeros((3, 3), dtype = complex)
#             total_impedance += imp_dict
#             #Case of where we the two buses are directly linked by purely upstream connections, allowing us to
#             #terminate our calculations earlier
#             if pred_list_max[0] == min_depth_bus:
#                 #print("Iterated " + str(i+1) + " times to get direct upstream connection total impedance.")
#                 return total_impedance
#
#             max_depth_bus = pred_list_max[0]
#             pred_list_max = list(feeder.network.predecessors(max_depth_bus))
#
#     assert(depths[max_depth_bus] == depths[min_depth_bus])
#     #print("Iterated " + str(depth_dif) + " times to reach equal depths."
#     common_parent = pred_list_max[0] == pred_list_min[0]
#     count_get_to_common = 0
#
#     #Here, we simultaneously shift both buses (after the max depth bus has been shifted to be of equal depth to the min
#     #depth bus) to a point where the parent bus is shared
#     while not common_parent:
#         count_get_to_common += 1
#         impedance_bus_min = feeder.network.get_edge_data(pred_list_min[0], min_depth_bus, default=None)['connector']
#         if impedance_bus_min == None:
#             print("WARNING: No connection between nodes " + str(pred_list_min[0]) + " and " + str(min_depth_bus) + ".")
#             return 0
#         else:
#             imp_dict_min = impedance_bus_min.Z if isinstance(impedance_bus_min, setup_nx.line) else np.zeros((3, 3), dtype = complex)
#
#             total_impedance += imp_dict_min
#             min_depth_bus = pred_list_min[0]
#             pred_list_min = list(feeder.network.predecessors(min_depth_bus))
#
#         impedance_bus_max = feeder.network.get_edge_data(pred_list_max[0], max_depth_bus, default=None)['connector']
#         if impedance_bus_max == None:
#             print("WARNING: No connection between nodes " + str(pred_list_max[0]) + " and " + str(max_depth_bus) + ".")
#             return 0
#         else:
#             imp_dict_max = impedance_bus_max.Z if isinstance(impedance_bus_max, setup_nx.line) else np.zeros((3, 3), dtype = complex)
#
#             total_impedance += imp_dict_max
#             max_depth_bus = pred_list_max[0]
#             pred_list_max = list(feeder.network.predecessors(max_depth_bus))
#
#         common_parent = pred_list_max[0] == pred_list_min[0]
#
#     #print("Total iterations to get to common parent is " + str(count_get_to_common))
#     #print("Common parent is " + str(pred_list_max[0]))
#     #Need to iterate one more time to account for "joining" node
#     impedance_bus_min = feeder.network.get_edge_data(pred_list_min[0], min_depth_bus, default=None)['connector']
#     if impedance_bus_min == None:
#         print("WARNING: No connection between nodes " + str(pred_list_min[0]) + " and " + str(min_depth_bus) + ".")
#         return 0
#     else:
#         imp_dict_min = impedance_bus_min.Z if isinstance(impedance_bus_min, setup_nx.line) else np.zeros((3, 3), dtype = complex)
#         total_impedance += imp_dict_min
#
#     impedance_bus_max = feeder.network.get_edge_data(pred_list_max[0], max_depth_bus, default=None)['connector']
#     if impedance_bus_max == None:
#         print("WARNING: No connection between nodes " + str(pred_list_max[0]) + " and " + str(max_depth_bus) + ".")
#         return 0
#     else:
#         imp_dict_max = impedance_bus_max.Z if isinstance(impedance_bus_max, setup_nx.line) else np.zeros((3, 3), dtype = complex)
#         total_impedance += imp_dict_max
#
#     return total_impedance

      
def get_RX_ratio_tosubst(feeder, node_name, depths):
    #Returns the R/X ratio from a node up to the substation 
    impedances_per_phase = get_total_impedance_from_substation(feeder, node_name,depths)
    
    p1_z = impedances_per_phase[0][0]
    p2_z = impedances_per_phase[1][1]
    p3_z = impedances_per_phase[2][2]
    
    p1_rx = np.real(p1_z) / np.imag(p1_z) if p1_z.imag != 0 else 0 # R/X ratio
    p2_rx = np.real(p2_z) / np.imag(p2_z) if p2_z.imag != 0 else 0
    p3_rx = np.real(p3_z) / np.imag(p3_z) if p3_z.imag != 0 else 0
    
    return {'Phase 1': np.around(p1_rx, 2), 'Phase 2': np.around(p2_rx, 2), 'Phase 3': np.around(p3_rx, 2)}

   
def get_RX_ratio_between_two_buses(feeder, node_name_1, node_name_2, depths):
    #Returns the R/X ratio between 2 nodes     
    impedances_per_phase = get_common_node_impedance_two_buses(feeder, node_name_1, node_name_2, depths)
    
    p1_z = impedances_per_phase[0][0]
    p2_z = impedances_per_phase[1][1]
    p3_z = impedances_per_phase[2][2]

    p1_rx = np.real(p1_z) / np.imag(p1_z) if p1_z.imag != 0 else 0 # R/X ratio
    p2_rx = np.real(p2_z) / np.imag(p2_z) if p2_z.imag != 0 else 0
    p3_rx = np.real(p3_z) / np.imag(p3_z) if p3_z.imag != 0 else 0
    
    return {'Phase 1': np.around(p1_rx,2), 'Phase 2': np.around(p2_rx,2), 'Phase 3': np.around(p3_rx,2)}

def get_phase_ratio_between_two_buses(feeder, node_name_1, node_name_2, depths):
    # this gives us total impedance, sum of z on each edge
    impedance= get_common_node_impedance_two_buses(feeder, node_name_1, node_name_2, depths) # 3x3 mat, ztot
    self_imped_A = impedance[0][0] 
    self_imped_B = impedance[1][1]
    self_imped_C = impedance[2][2]

    mutual_imped_A = impedance[0][1] + impedance[0][2] 
    mutual_imped_B = impedance[1][0] + impedance[1][2]
    mutual_imped_C = impedance[2][0] + impedance[2][1]

    rat_A = mutual_imped_A/self_imped_A if self_imped_A != 0 else 0
    rat_B = mutual_imped_B/self_imped_B if self_imped_B != 0 else 0
    rat_C = mutual_imped_C/self_imped_C if self_imped_C != 0 else 0

    coupling_ratios_3ph= np.array([[rat_A, rat_B, rat_C]])
    # returns 1x3 array
    return coupling_ratios_3ph

def plot_histogram_RX_ratios(feeder, leaves, slack_bus, depths, leaves_only = False):
    #generates histograms of the R/X ratios for all of the nodes or a specified set of nodes in a network to the substation
    #one histograms is generated for each phase (a total of 3) 
    nodes_to_use = leaves if leaves_only else list(feeder.network.nodes)
    
    #PL0001 Case
    #lst_remove = ['bus_N_L_22666_sec', 'bus_N_L_21316_sec', 'bus_N_L_22077_sec', 
                  #'bus_N_L_52586_sec', 'bus_N_L_38426_sec', slack_bus]
    #AL0001 Case
    #lst_remove = ['bus_N_L_87632_sec', 'bus_N_L_46793_sec', 'bus_N_L_17532_sec',
     #            'bus_N_L_108238_sec', 'bus_N_L_111610_sec', 'bus_N_L_147411_sec',
     #            'bus_N_L_6860_sec', 'bus_N_L_9709_sec', 'bus_N_L_110483_sec',
     #            'bus_N_L_132901_sec', 'bus_N_L_40688_sec', 'bus_N_L_138443_sec',
     #            'bus_N_L_108120_sec', 'bus_N_L_116563_sec', 'bus_N_L_113827_sec', slack_bus]
    
    #13NF case
    lst_remove=[]
    
    nodes_to_use = [nd for nd in nodes_to_use if nd not in lst_remove]
    
    RX_ratios_all = [get_RX_ratio_tosubst(feeder, node_name, depths) for node_name in nodes_to_use]
    
    phase_1_RX_ratios = [phase_dct['Phase 1'] for phase_dct in RX_ratios_all]
    phase_2_RX_ratios = [phase_dct['Phase 2'] for phase_dct in RX_ratios_all]
    phase_3_RX_ratios = [phase_dct['Phase 3'] for phase_dct in RX_ratios_all]
    
    plt.title("Phase 1 R/X Ratio Histogram")
    plt.xlabel("R/X Ratio")
    plt.ylabel("Number of Nodes")
    plt.hist(phase_1_RX_ratios, bins=20)
    plt.savefig('AL0001_RXhist_phA.png') # modify for each feeder
    plt.show() # need to savefig before plt.show

    plt.title("Phase 2 R/X Ratio Histogram")
    plt.xlabel("R/X Ratio")
    plt.ylabel("Number of Nodes")
    plt.hist(phase_2_RX_ratios, bins=20)
    plt.savefig('AL0001_RXhist_phB.png') # modify for each feeder
    plt.show()
    
    plt.title("Phase 3 R/X Ratio Histogram")
    plt.xlabel("R/X Ratio")
    plt.ylabel("Number of Nodes")
    plt.hist(phase_3_RX_ratios, bins=20)
    plt.savefig('AL0001_RXhist_phC.png') # modify for each feeder
    plt.show()
    
    return None


def get_largest_imped_to_sub(feeder, depths, num_impeds):
    #returns dictionary with the edge names and impedances for the num_impeds largest impedances from an edge to the substation
    edge_depths = {}
    max_edge_imps = {}
    edge_imps = {}
    edge_nodes = lzn.find_edge_nodes(feeder)
    for edge in edge_nodes:
        edge_depths[edge] = depths[edge]
    
    for edge in edge_depths.keys():
        imp_3by3 = get_total_impedance_from_substation(feeder, edge, depths)
        imp_lst = list(imp_3by3[0]) + list(imp_3by3[1]) + list(imp_3by3[2])
        avg_imp = np.mean(imp_lst)
        avg_imp_mag = ((avg_imp.real**2) + (avg_imp.imag**2))**(1/2)
        edge_imps[edge] = avg_imp_mag
        
    for _ in range(num_impeds):
        max_z_edge = max(edge_imps, key = edge_imps.get)
        max_edge_imps[max_z_edge] = edge_imps[max_z_edge]
        edge_imps.pop(max_z_edge)
    
    return max_edge_imps

def get_z2sub_for_config(feeder, act_locs,depths):
    # compute sum of z2subs for a given config
    mysum=np.zeros((1,3))
    for act_name in act_locs:
        z2sub=get_total_impedance_from_substation(feeder, act_name,depths) # returns 3x3 complex matrix
        zself=[[z2sub[0][0], z2sub[1][1], z2sub[2][2]]]
        #print('zzelf=',zself)
        abs_z=np.absolute(zself)
        #print('abs_z=',abs_z)
        mysum=mysum+np.array(abs_z)
        
    return mysum


def get_commonNodeZ_for_config(cfg,feeder,depths):
     # compute the sum of common-node impedances across all pairs of nodes in cfg

    def rSubset(arr, r):
        # return list of all subsets of length r
        return set(list(combinations(arr, r)))
    combo_lst=rSubset(cfg, 2) # n choose 2 combinations
    sum=0
    for pair in combo_lst:
        sum= sum + get_common_node_impedance_two_buses(feeder, pair[0], pair[1], depths)
    return sum # 3x3 array

def compute_z2sub_lstNodes(lstNodes,feeder,depths):
    # compute a 3x1 vector z_to_sub for all buses ins lstNodes
    # 3x1 vector is |z| on each phase
    myabsZs = np.array([], dtype=np.int64).reshape(0, 3)

    for busName in lstNodes:  # for all non-slack buses
        z2sub = np.around(imp.get_total_impedance_from_substation(feeder, busName, depths), 2)
        zself = [[z2sub[0][0], z2sub[1][1], z2sub[2][2]]]
       # print('zzelf=', zself)
        abs_z = np.absolute(zself)
       # print('abs_z=', abs_z)
        myabsZs = np.append(myabsZs, np.array(abs_z), axis=0)

    return myabsZs # list of 3x1 vectors