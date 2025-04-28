import importlib
import setup_nx # your own module, setup.nx.py

importlib.reload(setup_nx)
from setup_nx import *
import my_impedance_funcs as imp
import my_detControlMatExistence_funcs as ctrl
import numpy as np

def remove_subst_nodes(feeder, file_name):
    #remove the substation nodes/node from the network's node list, print(node_index_map) to determine the idx
    graph = feeder.network
    if '13NFbalanced' in file_name:
        substIdx = [6, 7] # substation index --> Note: idx 6 & 7 are MANUALLY PICKED OUT FOR 13NF
    elif '123NF' in file_name:
        substIdx = [22, 24]
    elif 'PL0001' in file_name:
        substIdx = [340] 
    elif 'oaklandJ' in file_name:
        substIdx=[163]
    elif '37NF' in file_name:
        substIdx=[0,36] # node 701 and node 799 respectively (701-799 is substation xfmr edge)
    elif '13NFunbalanced' in file_name:
        substIdx = [6, 7]
    else:
        print('error in hm.remove_subst_nodes: do not recognize file_name')
    graphNodes_nosub = list(np.delete(graph.nodes, substIdx)) # dont consider substation nodes, node 650 and 651 for 13NF
    return graphNodes_nosub # list of buses ['bus_X','bus_Y',...]
    
def createNodeIndexMap(feeder):
    #indexes assigned based on numerical value of node, not node's possition in the network
    #for example, the node named with the lowest numerical value will map to 0, not the substation
    #feeder = initiaized feeder object
    graph = feeder.network
    node_index_map = {} #node indices for indicMat and F matrix
    t = 0 #initializing first index for node_index_map
    
    for node in graph.nodes: #populating node_idex_map
        node_index_map[node] = t
        t += 1
    return node_index_map


def getKey(dictionary, value):
    #use to retrieve node name from node_index_map using node index
    #value = node index
    for k, v in dictionary.items(): 
         if value == v:
                key = k
    return key 


def createRXmatrices_3ph(feeder, depths,file_name):
    #returns 2 (3n)x(3n) matrices containing R and X values for the network 
    #includes all 3 phases
    #feeder = initiaized feeder object
    
    graph_noSub=remove_subst_nodes(feeder, file_name) # list of graph.nodes
#    graph = feeder.network

    n = len(graph_noSub) #number of nodes
    print('n=',n)
    R = np.zeros((3*n, 3*n)) #initializing R matrix
    X = np.zeros((3*n, 3*n)) #initializing X matrix
    P = {} #initializing line path dictionary
  
    for node in graph_noSub:
        P[node] = imp.get_path_from_substation(feeder, node, depths)
    
    index_outer=0
    for n_outer in graph_noSub: #outer loop
        index_outer+=1
        index_inner=0
        for n_inner in graph_noSub: #inner loop
            index_inner+=1
            intersection_set = set.intersection(set(P[n_outer]), set(P[n_inner])) #finds common edges in paths from node to substation
            intersection_list = list(intersection_set)
            #print('intersection_list=',intersection_list) # temp
            imped_sum = np.zeros((3,3))
            imped_sum = imped_sum.astype('complex128')
            
            for edge in intersection_list: #iterates through shared edges and sums their impedances
                impedance = feeder.network.get_edge_data(edge[1], edge[0], default = None)['connector']
                tot_edge_impedance = impedance.Z if isinstance(impedance, setup_nx.line) else np.zeros((3, 3))
                #print('impedance=',tot_edge_impedance) # temp
                imped_sum += tot_edge_impedance
            for i_row in range(3):    # goes 0-->1-->2
                for i_col in range(3):
                    R[(3*index_outer) - (3-i_row)][(3*index_inner) - (3-i_col)] = 2*imped_sum[i_row][i_col].real
                    X[(3*index_outer) - (3-i_row)][(3*index_inner) - (3-i_col)] = 2*imped_sum[i_row][i_col].imag
            
    return R, X



def setupStateSpace(parmObj,feeder, depths,file_name):
    #initializes state space matrices A and B
    #n = number of nodes in network
    #feeder = initiaized feeder object
    #node_index_map = dictionary of node indices with node names as keys
    #^not used
    R, X = createRXmatrices_3ph(feeder, depths,file_name)
    n=round(len(R)/3)
    concat_XR=np.concatenate((X, R), axis = 1)
    if parmObj.get_version()==1: # PBC       
        A = np.identity(6*n)
        concat_XR_halfs = np.concatenate(((-1/2) * R, (1/2) * X), axis = 1)
        B = np.concatenate((concat_XR, concat_XR_halfs))
    else: # volt-watt and volt-var
        A = np.zeros((3*n,3*n))
        B = concat_XR # (6n*3n) matrix
    
    return A, B,n


# correct version
def computeFeas_v1(parmObj,feeder, act_locs, A, B, indicMat, indicMat_table,substation_name, perf_nodes, depths, node_index_map, Vbase_ll, Sbase, printCurves,file_name):
    node_0 = list(feeder.network.successors(substation_name))
    node_1 = list(feeder.network.successors(node_0[0]))
    z12 = imp.get_total_impedance_from_substation(feeder, node_1[0],depths) # 3 phase, not pu
    B12=np.zeros((3,3)) # TEMPORARY, line susceptance, Yshunt=G+jB

    MYfeas,MYfeasFs,MYpercentfeas,MYnumTried,MYnumact,MYbestF_asvec,MYbestF_asmat,MYindicMat,min_domeig_mag = ctrl.detControlMatExistence(parmObj, feeder, A, B, indicMat,indicMat_table,act_locs,perf_nodes,node_index_map,depths,file_name)
    print('percent feas=',MYpercentfeas)
    print('num tried=',MYnumTried)

    #lzn_err_max, slopes = lzn.detLznRange(feeder, Vbase_ll, Sbase, z12, B12, act_locs, load_data, headerpath, substation_name, modelpath, depths,printCurves) # usually called by computeFeas
    lzn_err_max=[-1, -1, -1, -1] # workaround, for [PV, QV, Pdel,Qdel] lzn errors

    return MYfeas,lzn_err_max,MYpercentfeas,MYbestF_asvec,MYbestF_asmat,MYindicMat,min_domeig_mag


# workaround version
def computeFeas_v2(feeder, act_locs, perf_nodes, A, B, indicMat):
    random_bit = np.random.choice(2,1)
    if random_bit[0] == 0:
        feas = True
    else:
        feas = False
    maxError = 10
    return feas, maxError

class configParms: # used by updateStateSpace to determine whether each act is PBC, VVC, or VWC
    def __init__(self, default = {}): 
        self.ctrlTypeList = default 
        self.version=-1
    def get_ctrlTypes(self): 
        return self.ctrlTypeList 
    def set_ctrlTypes(self, mylist): 
        self.ctrlTypeList = mylist
    def get_version(self): 
        return self.version 
    def set_version(self, ver): 
        self.version = ver
        
def updateStateSpace(parmObj,feeder, n, act_locs, perf_nodes,file_name):
    # creates indicMat and indicMatTable
    #creates (6n*3n) matrix with 1 at (3i+ph)(3j+ph) for volt-watt control, and (3i+3n+ph)(3j+ph) for volt-var control
    #in the above description, ph is the integer representation (a=0, b=1, c=2) of the phase intersection between the actuator and performance nodes
    #if an actuator and performance node have no phases in common, a warning is printed
    #n = number of nodes in network
    #act_locs = list of actuators locations in network (list of strings)
    #perf_nodes = list of performance nodes 
    #node_index_map = list of bus names in order for indicMat and F matrix
    if parmObj.get_version()==1: # PBC
        indicMat = np.zeros((6*n,6*n))
    else: # volt-watt and volt-var
        indicMat = np.zeros((6*n,3*n))
    #print('act_locs=',act_locs)
    graph_noSub=remove_subst_nodes(feeder, file_name) # list of graph.nodes

    ctrlTypeList=parmObj.get_ctrlTypes()
    indicMat_table=np.array([], dtype=np.int64).reshape(0,3)
    for k in range(len(act_locs)): 
        act = act_locs[k]
        perf = perf_nodes[k]
        ctrlType=ctrlTypeList[k] # need to have 5 control types, 4 for existing and 1 for the test
        if not(ctrlType=='PBC' or ctrlType=='VVC' or ctrlType=='VWC'):
            raise Exception('Actuator node first 3 chars should be PBC, VVC, or VWC')
        
        act_phases = feeder.loaddict[act[4:]].phases # [7:] extracts the YYY bus number from 'XXXbus_YYY'
        perf_phases = feeder.busdict[perf[4:]].phases
        
        act_index=graph_noSub.index(act)
        perf_index=graph_noSub.index(perf)

        phase_intrsct = [ph for ph in act_phases if ph in perf_phases]
        if phase_intrsct == []: # disallow configs in which the act and perf node phases are not aligned. Results in kgain=0.0001 and thinks it's feasible
            print('WARNING: act_node ' + act + ' can NOT track perf_node ' + perf + ' --> no common phases')
            phase_loop_check=False
            break # if any actuator of the config has phase mismtach (between act and perf), don't evaluate the config
        else:
            phase_loop_check=True
            for i in range(len(phase_intrsct)):
                if phase_intrsct[i] == 'a':
                    phase_intrsct[i] = 0
                elif phase_intrsct[i] == 'b':
                    phase_intrsct[i] = 1
                elif phase_intrsct[i] == 'c':
                    phase_intrsct[i] = 2


        #print('act=',act,', perf=',perf,', ctrlType=',ctrlType,', phaseItrsct=',phase_intrsct)
        for ph in phase_intrsct:
            row1=(act_index*3)+ph # row will mark in indicMat upper left block
            col1=(perf_index*3)+ph # col will mark in indicMat upper left block
            row2=(act_index*3)+(3*n)+ph # row will mark in indicMat lower right block
            col2=(perf_index*3)+(3*n)+ph # col will mark in indicMat lower right block
            if ctrlType=='PBC':
                indicMat[row1][col1] = 1
                indicMat[row2][col2] = 1 
                indicMat_table=np.append(indicMat_table,np.array([[k,row1,col1]]),axis=0) 
                indicMat_table=np.append(indicMat_table,np.array([[k,row2,col2]]),axis=0) 
            elif ctrlType=='VVC':
                indicMat[row1][col1] = 1
                indicMat_table=np.append(indicMat_table,np.array([[k,row1,col1]]),axis=0) 
            elif ctrlType=='VWC': #volt-watt control
                indicMat[row2][col1] = 1 
                indicMat_table=np.append(indicMat_table,np.array([[k,row2,col1]]),axis=0) 
            
    #print('[updateStateSpace] indicMat_table=\n',indicMat_table,'<<  [APNP_number indicMat_row indicMat_col], 3ph nodes should have 6 rows')
            
    return indicMat,indicMat_table,phase_loop_check

