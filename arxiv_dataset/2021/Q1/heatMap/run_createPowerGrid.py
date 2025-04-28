
class Grid_vars2:
    def __init__(self, R,X,depths,fin_feeder,graph,substation_name,node_index_map,Sbase,Vbase_ll,file_name,graphNodes_nosub):
        self.R = R
        self.X=X
        # self.Sbase=Sbase
        # self.Vbase=Vbase
        self.depths=depths
        self.feeder=fin_feeder
        self.graph=graph
        self.substation_name=substation_name
        self.node_index_map=node_index_map
        self.Sbase=Sbase
        self.Vbase_ll=Vbase_ll
        self.file_name=file_name
        self.graphNodes_nosub=graphNodes_nosub

def createPowerGrid_func():
    #region Import Statements

    import importlib
    import numpy as np
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

    import datetime
    import time

    import sys

    sys.path.append('py_modules') # below modules are in this folder
    #print(sys.path)
    import setup_nx # your own module, setup.nx.py
    importlib.reload(setup_nx)
    import my_feeder_funcs as ff
    import my_heatmapSetup_funcs as hm
    #import grid_vars_class

    print('finished importing packages...')

    # endregion

    #region [Essential] specify input feeder data for IEEE 123-node test feeder

    # Enter the path/name of the impedance model data (excel file)
    filepath = "feeder_impedance_models/"
    modelpath = filepath + "004_GB_IEEE123_OPAL_accur.xlsx"

    #==========================================================================================================

    #file_name = string specifying name of dot file created when make_graph() is called
    file_name = '123NF'

    # Specify substation kV, kVA bases, name, and the number of timesteps in the load data'
    Vbase_ll = 4160
    Vbase = Vbase_ll / np.sqrt(3)
    Sbase = 5000/3
    substation_name = 'bus_150'
    timesteps = 1

    # initialize some variables
    ts = time.time()
    print(datetime.datetime.fromtimestamp(ts))
    plot = 0 #turn plot on/off
    depths = {}
    leaves = []

    #endregion

    #region [ESSENTIAL] create feeder object

    fin_feeder = ff.feeder_init(modelpath, '', '', timesteps, Vbase_ll, Sbase, depths, leaves)
    print("Finished initializing feeder")
    ff.make_graph(fin_feeder, file_name)
    node_index_map = hm.createNodeIndexMap(fin_feeder)  # node indices for indicMat and F matrix
    R, X = hm.createRXmatrices_3ph(fin_feeder, depths, file_name)
    graphNodes_nosub = hm.remove_subst_nodes(fin_feeder,
                                             file_name)  # dont consider co-located at substation nodes, node 650 and 651

    # print('depths=',depths) # check this is populated, lists how far each node is from substation
    # print('depths length=',len(depths))

    # print list of first 10 buses in network
    print("First ten nodes are:")
    count = 0
    for i in fin_feeder.network:
        print(i)
        count += 1
        if count >= 10:
            break

    graph = fin_feeder.network
    graphNodes_nosub

    grid_vars_obj = Grid_vars2(R,X,depths,fin_feeder,graph,substation_name,node_index_map,Sbase,Vbase_ll,file_name,graphNodes_nosub)
    #endregion


    return grid_vars_obj