import importlib
import setup_nx # your own module, setup.nx.py
import random

importlib.reload(setup_nx)
from setup_nx import *
import my_feeder_funcs as ff
import my_configVis_funcs as vis
import my_heatmapSetup_funcs as hm
import my_impedance_funcs as imp

def eval_config(parmObj,feeder, all_act_locs, perf_nodes, node_index_map, substation_name, depths, file_name, Vbase_ll, Sbase):
    #returns whether controllability can be achieved for a given actuator performance node configuration
    #also returns the linearization error associated with the feasibility calculation
    #all_act_locs and perf_nodes = lists of node names as strings
    np.random.seed(2) # so that each time you run the whole code you get the same result
    printCurves = True # your choice on whether to print PVcurves
    graph = feeder.network
    A, B,n = hm.setupStateSpace(parmObj,feeder,depths,file_name)
    assert A.shape==(6*n,6*n), "issue: A (from setupStateSpace) or n have incorrect dims"
    indicMat,indicMat_table,phase_loop_check = hm.updateStateSpace(parmObj,feeder, n, all_act_locs, perf_nodes,file_name)
    if phase_loop_check:  # disallow configs in which the act and perf node phases are not aligned
        feas, maxError, percent_feas,bestF_asvec,bestF_asmat,indicMat,min_domeig_mag = hm.computeFeas_v1(parmObj,feeder, all_act_locs, A, B, indicMat, indicMat_table, substation_name, perf_nodes, depths, node_index_map, Vbase_ll, Sbase,  printCurves,file_name)
    else:
        raise Exception("configs has act and perf node phases not aligned")

    vis.markActuatorConfig(all_act_locs, feeder, file_name) # create diagram with actuator locs marked
    
    print('Actuator configuration is feasible') if feas else print('Actuator configuration is not feasible')
    return feas, maxError, percent_feas,min_domeig_mag,bestF_asvec,bestF_asmat, indicMat

    
def find_good_colocated(parmObj,feeder, set_acts, addon_acts, node_index_map,substation_name, depths, file_name, Vbase_ll, Sbase):
    #CPP process
    #almost the same as runheatmap process, but only runs once and shows one heatmap indicating which nodes are good to place a co-located act/perf node
    #return list of all "green" configs and the associated lzn errors
    #act_locs == a list of pre-set colocated act/perf locations --> to evaluate an empty network, pass in act_locs == []
    a = 0 # counter for adding new APNPs
    ff.clear_graph(feeder) # clear any previous modifictions made to graph
    graph = feeder.network
    cur_act_locs = set_acts
    heatMapNames = [] # collect heat map names as list of strings
    A, B,n = hm.setupStateSpace(parmObj,feeder,depths,file_name)
    assert A.shape==(6*n,6*n), "issue: A (from setupStateSpace) or n have incorrect dims"
    feas_configs = [] 
    printCurves=False # your choice on whether to print PVcurves
    random.seed(3)  # initialize random num generator so results are reproducable

    all_ctrlTypes=parmObj.get_ctrlTypes() # format is ctrl types of [set_acts addon_acts]
    set_ctrlTypes=all_ctrlTypes[:len(set_acts)] # get control types of set_acts
    cand_ctrlTypes=all_ctrlTypes[len(set_acts):] # get control types of addon_acts
    #print('set control types=',set_ctrlTypes)
    #print('cand control types=',cand_ctrlTypes)

    heatmap_dic = {} # hold results across process
    while a < len(addon_acts): #outer loop, a = number of actuators to place
        test_nodes = []
        graphNodes_nosub = hm.remove_subst_nodes(feeder, file_name) # dont consider co-located at substation nodes, node 650 and 651
    
        cand_ctrlType=cand_ctrlTypes[a] # for each step all candidate APNPs will be of this type
        parmObj.set_ctrlTypes([cand_ctrlType]+set_ctrlTypes) # to be used in updateStateSpace
        
        for act in cur_act_locs: # mark placed acts in grey
            vis.markActLoc(graph, act)
            
        for node in graphNodes_nosub: # try placing act/perf at all nodes of the network
            if node not in cur_act_locs:
                test_nodes.append(node)

        domeig_lst=[]
        for i in range(len(test_nodes)):
            test=test_nodes[i]
            print('------ running CPP: percent done=', np.round(100*i / len(test_nodes),1),flush=True)
            feas=False # default
            print('evaluating act and perf colocated at ',[test] + cur_act_locs) 
            indicMat,indicMat_table,phase_loop_check = hm.updateStateSpace(parmObj,feeder, n, [test] + cur_act_locs, [test] + cur_act_locs,file_name) # (n,act,perf,dictionary)
            if phase_loop_check:  # disallow configs in which the act and perf node phases are not aligned
                feas, maxError, percent_feas,bestF_asvec,bestF_asmat,indicMat,min_domeig_mag = hm.computeFeas_v1(parmObj,feeder, [test] + cur_act_locs, A, B, indicMat, indicMat_table,substation_name,[test] + cur_act_locs, depths, node_index_map,Vbase_ll, Sbase, printCurves,file_name) # pass in potential actual loc
                domeig_lst.append(min_domeig_mag)
            else:
                raise Exception("configs has act and perf node phases not aligned")

            heatmap_dic[test]=[percent_feas,min_domeig_mag,bestF_asvec,bestF_asmat]

        # plt.figure()
        domeig_lst_less1=[x for x in domeig_lst if x<1]
        domeig_range=[min(domeig_lst_less1),max(domeig_lst_less1)]
        print('domeigs that are <1 range from', round(domeig_range[0], 5), ' to ', round(domeig_range[1], 5))
        for i in range(len(test_nodes)):
            vis.markFeas(domeig_range,domeig_lst[i], test_nodes[i], graph)
            
        heatMapName='CPP_heatmap_step' + str(a+1) + '_' + file_name
        heatMapNames.append(heatMapName)
        vis.write_formatted_dot(graph, heatMapName)

        a += 1 # place actuator
        
        if a <= len(addon_acts): # choose actuator and finalize assoc perf node
            cur_act_locs = addon_acts[0:a]+set_acts # populate cur_act_locs with subset of all_act_locs
            set_ctrlTypes=[cand_ctrlType]+set_ctrlTypes # update control types for next step
            
    return heatmap_dic,heatMapNames


def runHeatMapProcess(parmObj,feeder, set_acts, set_perfs, addon_acts, addon_perfs, node_index_map, substation_name, depths, file_name, Vbase_ll, Sbase):
    #compute heatmap (assess feas and lzn error on every node of the feeder, color each red/green on diagram)
    #Then for each act-perf node pair, compute heatmap
    #return list of all "green" configs and the associated lzn errors
    #heatmap shows good places to placed act wrt given perf node, NOT good places to place colocated act-perf node
    #addon_acts and addon_perfs = lists of node names as strings
    a = 0
    graph = feeder.network
    cur_act_locs = set_acts
    cur_perf_nodes = set_perfs
    heatMapNames = [] # collect heat map names as list of strings
    A, B,n = hm.setupStateSpace(parmObj,feeder, depths,file_name)
    assert A.shape==(6*n,6*n), "issue: A (from setupStateSpace) or n have incorrect dims"
    heatmap_dic={}
    random.seed(3)  # initialize random num generator so results are reproducable

    while a < len(addon_acts): #outer loop, a = number of actuators
        for act in cur_act_locs: 
            vis.markActLoc(graph, act)

        test_nodes = []
        graphNodes_nosub = hm.remove_subst_nodes(feeder, file_name) # dont consider co-located at substation nodes, node 650 and 651
        
        for node in graphNodes_nosub:
            if node not in cur_act_locs:
                test_nodes.append(node)
        
        domeig_lst=[]
        for i in range(len(test_nodes)): #inner loop
            test = test_nodes[i]
            print('------ running RHP: percent done=', np.round(100 * i / len(test_nodes), 1), flush=True)
            feas=False # default
            # heatmap color indicates good places to place actuator given chosen loc of perf node (not necessarily colocated)          
            print('evaluating actuator node at ', [test] + cur_act_locs,',\n performance node at ', [addon_perfs[a]] + cur_perf_nodes)
            indicMat,indicMat_table,phase_loop_check = hm.updateStateSpace(parmObj,feeder, n, [test] + cur_act_locs, [addon_perfs[a]] + cur_perf_nodes,file_name)
            if phase_loop_check:  # disallow configs in which the act and perf node phases are not aligned
                feas, maxError, percent_feas,bestF_asvec,bestF_asmat,indicMat,min_domeig_mag = hm.computeFeas_v1(parmObj,feeder, [test] + cur_act_locs, A, B, indicMat, indicMat_table, substation_name,[addon_perfs[a]] + cur_perf_nodes, depths, node_index_map, Vbase_ll, Sbase, False,file_name) # false for printing PV curves
                domeig_lst.append(min_domeig_mag)
            else:
                raise Exception("configs has act and perf node phases not aligned")

            heatmap_dic[test]=[percent_feas,min_domeig_mag,bestF_asvec,bestF_asmat]

        
        # if have time, store domeig_lst and test_nodes into dictionary, then print the dictionary so that it can accompany the heatmap            
        # plt.figure()
        domeig_lst_less1=[x for x in domeig_lst if x<1]       
        # plt.hist(domeig_lst_less1, density=True, bins=15) # round to nearest hundredth
        # plt.xlabel('RHP: min domeig per config')
        domeig_range = [min(domeig_lst_less1), max(domeig_lst_less1)]
        print('domeigs that are <1 range from', round(domeig_range[0], 5), ' to ', round(domeig_range[1], 5))
        for i in range(len(test_nodes)):
            vis.markFeas(domeig_range,domeig_lst[i], test_nodes[i], graph)

            
        graph.nodes[addon_perfs[a]]['shape'] = 'square'
        # after generate data for heatmap..
        heatMapName = 'NPP_heatmap_step' + str(a+1) + '_' + file_name
        heatMapNames.append(heatMapName)
        vis.write_formatted_dot(graph, heatMapName)

        a += 1 # place actuator
        
        if a <= len(addon_acts): # choose actuator and finalize assoc perf node
            cur_act_locs = addon_acts[0:a]+set_acts # populate cur_act_locs with subset of addon_acts
            cur_perf_nodes = addon_perfs[0:a]+set_acts

        # end of while loop
    return heatmap_dic, heatMapNames

def design_config(parmObj,seedkey,numAct,feeder, file_name, node_index_map, depths, substation_name,Vbase_ll, Sbase):
    # randomly try groups of numAct actuators until find one that works 
    test_nodes = []
    ctrlTypes=[]

    graphNodes_nosub = hm.remove_subst_nodes(feeder, file_name) # dont consider substation nodes, node 650 and 651 for 13NF
    
    for node in graphNodes_nosub:
        test_nodes.append(node)

    random.seed(seedkey)  # initialize random num generator so results are reproducable 
    numtry=0
    while numtry<100: # try a max of 100 configs with numAct actuators
        # choose random set of control types
        if parmObj.get_version()==1:
            ctrlTypes_choosefrom=['PBC','PBC']
        else:
            ctrlTypes_choosefrom=['VWC','VVC']    
            
        rand_ctrlType=random.choices(ctrlTypes_choosefrom,k=numAct)
        ctrlTypes=rand_ctrlType # save into list of control types

        # choose random set of collocated APNP locations
        rand_test = random.sample(test_nodes,numAct) # Return a list of unique elements chosen from the population sequence or set. Used for random sampling without replacement.
        parmObj.set_ctrlTypes(ctrlTypes)
        print('control types=',ctrlTypes)
        print('evaluating actuator and performance node colocated at ',rand_test) 
        feas, maxError, percent_feas,min_domeig_mag, bestF_asvec,bestF_asmat, indicMat=eval_config(parmObj,feeder, rand_test, rand_test, node_index_map, substation_name, depths, file_name, Vbase_ll, Sbase)
        
        if feas:
            break # break out of loop
        else:
            numtry+=1
            
    if not feas:
        print('Algo failed: could not find config with ',numAct,' APNPs')
        bestFparm=-1
    act_locs=rand_test
    return parmObj,act_locs,bestFparm,indicMat

def eval_random_configs(parmObj, seedkey,numAct,numEval,feeder, file_name, node_index_map, depths, substation_name,Vbase_ll, Sbase):
    # randomly try numEval groups of numAct actuators and returns vector indicating which are feas
    test_nodes = []
    ctrlTypes=[]

    graphNodes_nosub = hm.remove_subst_nodes(feeder, file_name) # dont consider substation nodes, node 650 and 651 for 13NF
    
    for node in graphNodes_nosub:
        test_nodes.append(node)

    random.seed(seedkey)  # initialize random num generator so results are reproducable 
    if parmObj.get_version()==1:
        ctrlTypes_choosefrom=['PBC','PBC']
    else:
        ctrlTypes_choosefrom=['VWC','VVC']  
           
    feas_vec=[] # hold 1 if feas, 0 if not
    for i in range(1,numEval): # try a max of 100 configs with numAct actuators
        # choose random set of control types  
        rand_ctrlType=random.choices(ctrlTypes_choosefrom,k=numAct)
        ctrlTypes=rand_ctrlType # save into list of control types

        # choose random set of collocated APNP locations
        rand_test = random.sample(test_nodes,numAct) # Return a list of unique elements chosen from the population sequence or set. Used for random sampling without replacement.
        parmObj.set_ctrlTypes(ctrlTypes)
        print('control types=',ctrlTypes)
        print('evaluating actuator and performance node colocated at ',rand_test) 
        feas, maxError, percent_feas,min_domeig_mag, bestF_asvec,bestF_asmat, indicMat=eval_config(parmObj,feeder, rand_test, rand_test, node_index_map, substation_name, depths, file_name, Vbase_ll, Sbase)
        
        if feas:
            feas_vec.append(1)
        else: 
            feas_vec.append(0)

    return feas_vec


def placeMaxColocActs_stopAtInfeas(parmObj,feeder, file_name, node_index_map, depths, substation_name,Vbase_ll, Sbase):
    #place colocated actuators until an infeasible loc is tested, then call find_good_colocated and return 
    graph = feeder.network
    A, B,n = hm.setupStateSpace(parmObj,feeder,depths,file_name)
    assert A.shape==(6*n,6*n), "issue: A (from setupStateSpace) or n have incorrect dims"
    test_nodes = []
    act_locs = []
    ctrlTypes=[]
    printCurves=False # your choice on whether to print PVcurves
    graphNodes_nosub = hm.remove_subst_nodes(feeder, file_name) # dont consider substation nodes, node 650 and 651 for 13NF
    if parmObj.get_version()==1:
        ctrlTypes_choosefrom=['PBC','PBC']
    else:
        ctrlTypes_choosefrom=['VWC','VVC']
    rand_ctrlType=random.choice(ctrlTypes_choosefrom)
    ctrlTypes.append(rand_ctrlType) # save into list of control types
    
    for node in graphNodes_nosub:
        if node not in act_locs:
            test_nodes.append(node)
            
    random.seed(3)  # initialize random num generator so results are reproducable
    while test_nodes:         
        rand_test = random.choice(test_nodes)
        parmObj.set_ctrlTypes(ctrlTypes)
        print('control types=',ctrlTypes)
        print('evaluating actuator and performance node colocated at ',[rand_test] + act_locs) 
        indicMat,indicMat_table,phase_loop_check = hm.updateStateSpace(parmObj,feeder, n, [rand_test] + act_locs, [rand_test] + act_locs,file_name)
        if phase_loop_check:  # disallow configs in which the act and perf node phases are not aligned
            feas, maxError, numfeas,bestF,indicMat,min_domeig_mag = hm.computeFeas_v1(parmObj,feeder, [rand_test] + act_locs, A, B, indicMat, indicMat_table,substation_name, [rand_test] + act_locs, depths, node_index_map,Vbase_ll, Sbase, printCurves, file_name)
        else:
            raise Exception("configs has act and perf node phases not aligned")
                
        if feas:
            act_locs += [rand_test]
            test_nodes.remove(rand_test)
            rand_ctrlType=random.choice(ctrlTypes_choosefrom)  # choose control for next candidate APNP 
            ctrlTypes.append(rand_ctrlType) # save into list of control types
        else:
            print('Random choice of co-located APNP yields unstable  configuration. Generating heatmap by checking all remaining feeder nodes...')
            heatmap_dic, heatMapNames = find_good_colocated(parmObj,feeder,[],act_locs, node_index_map, substation_name, depths,file_name, Vbase_ll, Sbase) # makes a heatmap, assume set_acts=[]
            break
    
    ff.clear_graph(feeder)
    vis.markActuatorConfig(act_locs, feeder, 'nonauto-CPP')
    return act_locs, parmObj


def place_max_coloc_acts(parmObj,seedkey,feeder, file_name, node_index_map, depths, substation_name,Vbase_ll, Sbase, cby_cand=-1):
    #place maximum number of colocated actuators
    #if infeas loc tested, randomly select another test node and continue function run
    # cby_cand is set of all nodes that we want to consider placing an actuator; default set is all nodes without the substation node(s)
    
    graph = feeder.network
    A, B,n = hm.setupStateSpace(parmObj,feeder, depths,file_name)
    assert A.shape==(6*n,6*n), "issue: A (from setupStateSpace) or n have incorrect dims"
    test_nodes = []
    act_locs = []
    ctrlTypes=[]
    printCurves = False # your choice on whether to print PVcurves
    if cby_cand==-1:
        cby_cand=hm.remove_subst_nodes(feeder, file_name) #default set is all nodes without the substation node(s)
        
    print('parmObj.get_version=',parmObj.get_version())
    if parmObj.get_version()==1:
        ctrlTypes_choosefrom=['PBC','PBC']
    else:
        ctrlTypes_choosefrom=['VWC','VVC']
    rand_ctrlType=random.choice(ctrlTypes_choosefrom)
    ctrlTypes.append(rand_ctrlType) # save into list of control types

    for node in cby_cand:
        if node not in act_locs:
            test_nodes.append(node)
    
    random.seed(seedkey)  # random num generator seed so results are reproducable
    while test_nodes:      # while non-empty
        rand_test = random.choice(test_nodes) # choose node randomly among test_nodes
        parmObj.set_ctrlTypes(ctrlTypes)
        print('control types=',ctrlTypes)
        
        print('evaluating actuator and performance node colocated at ',[rand_test] + act_locs) 
        indicMat,indicMat_table,phase_loop_check = hm.updateStateSpace(parmObj,feeder, n, [rand_test] + act_locs, [rand_test] + act_locs,file_name)
        if phase_loop_check:  # disallow configs in which the act and perf node phases are not aligned
            feas, maxError, numfeas,bestF,indicMat,min_domeig_mag = hm.computeFeas_v1(parmObj,feeder, [rand_test] + act_locs, A, B,indicMat, indicMat_table,substation_name, [rand_test] + act_locs, depths, node_index_map,Vbase_ll, Sbase, printCurves, file_name)
        
        else:
            raise Exception("configs has act and perf node phases not aligned")

        if feas:
            act_locs += [rand_test] # store feas act_loc
            test_nodes = []
            rand_ctrlType=random.choice(ctrlTypes_choosefrom) # choose control for next candidate APNP 
            ctrlTypes.append(rand_ctrlType) # save into list of control types
            for node in cby_cand:
                if node not in act_locs:
                    test_nodes.append(node)
        else:
            test_nodes.remove(rand_test)
    print('Randomly placed ',len(act_locs),' actuators, among n=',len(cby_cand),' possible DER nodes of feeder')
    ff.clear_graph(feeder)
    vis.markActuatorConfig(act_locs, feeder, 'auto-CPP_seed'+str(seedkey))
    return act_locs, parmObj


def place_max_coloc_acts_v2(parmObj,seedkey,feeder,node_index_map,substation_name,depths,file_name,Vbase_ll,Sbase,select):
    # place maximum number of colocated actuators
    # place on only lower half of feeder, and place to maximize the common-node impedance
    random.seed(seedkey)  # initialize random num generator so results are reproducable

    # compute z2sub for all nodes, put into list
    graphNodes_nosub = hm.remove_subst_nodes(feeder,file_name)  # dont consider co-located at substation nodes, node 650 and 651
    myabsZs=imp.compute_z2sub_lstNodes(graphNodes_nosub, feeder, depths)
    z2sub_dic={}
    for i in range(len(myabsZs)):
        z2sub_dic[graphNodes_nosub[i]]=np.mean(myabsZs[i])

    # sort the list and select the lower half of the nodes --> set N
    sorted_dic={}
    sorted_keys=sorted(z2sub_dic,key=z2sub_dic.get,reverse=True)
    for w in sorted_keys:
        sorted_dic[w]=z2sub_dic[w]
    for i in range(round(len(graphNodes_nosub)/2)):
        sorted_dic.popitem()

    N=list(sorted_dic.keys()) # list of buses on lower half of feeder
    M = N.copy()  # initialize set of remaining DERs where you could place
    n = len(N)
    p = round(0.1 * n)
    actLocs = []  # initialize
    actLocs.append(random.sample(N, 1)[0])  # place first DER randomly
    M.remove(actLocs[0])
    while len(M) > 0:  # while there are remaining DERs to try
        m = len(M)
        if m < p:
            p = m
        P = random.sample(M, p)  # sample p candidate locations from M

        if select=='commonNodeZ':
            # select using commonNodeZ
            zvals = [] # holds 1 scalar per cand_loc
            for cand_loc in P:
                zabs_mat=np.absolute(imp.get_commonNodeZ_for_config(actLocs + [cand_loc],feeder,depths))
                z_scalar=max([zabs_mat[0][0],zabs_mat[1][1],zabs_mat[2][2]])
                zvals.append(z_scalar)
            minval = min(zvals)
            idx = zvals.index(minval)
            chosen_loc = P[idx]
        elif select=='rdm':
            # select at random
            chosen_loc = random.sample(P,1)[0]
        else:
            raise Exception('OCPP: select value not recognized')

        parmObj.set_ctrlTypes(['PBC'] * len(actLocs+[chosen_loc]))
        feas, maxError, percent_feas,min_domeig_mag, bestF_asvec,bestF_asmat, indicMat = \
            eval_config(parmObj, feeder, actLocs+[chosen_loc], actLocs+[chosen_loc],
                        node_index_map,substation_name, depths, file_name, Vbase_ll, Sbase)

        if feas:
            actLocs.append(chosen_loc) # place APNP
            print('OCPP: new stable config is ', actLocs)
        M.remove(chosen_loc) # remove the candidate whether you placed it or not

    ff.clear_graph(feeder)
    vis.markActuatorConfig(actLocs, feeder, file_name)  # create diagram with actuator locs marked

    return actLocs