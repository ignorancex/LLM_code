#region Import statements
import sys
import numpy as np
import my_impedance_funcs as imp
import my_heatmapSetup_funcs as hm
import my_process_funcs as prc  # new
import my_configVis_funcs as vis
import pickle

#endregion

cfg1 = ['bus_300', 'bus_197', 'bus_104', 'bus_108', 'bus_106']  # bad config 1, cluster in one portion of feeder

def exp5(v):

    parmObj=hm.configParms()
    parmObj.set_version(1) # 1 for PBC
    seedkey=4 # you choose, so that each call to place_max_act results in same random seq
    select='commonNodeZ' # choose 'commonNodeZ' or 'rdm'

    cfg7=prc.place_max_coloc_acts_v2(parmObj, seedkey, v.feeder, v.node_index_map, v.substation_name, v.depths, v.file_name, v.Vbase_ll,
                            v.Sbase,select)

    with open('exp5_7.4_seed'+str(seedkey)+'_'+str(select)+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(cfg7, f)

    return