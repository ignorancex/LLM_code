#region Import statements
import sys
import numpy as np
import my_heatmapSetup_funcs as hm
import my_process_funcs as prc  # new
import matplotlib.pyplot as plt
import my_configVis_funcs as vis
import my_feeder_funcs as ff
import pickle

#endregion

CPP_dataFile_badBr="exp4_CPP_7.2_badBr.pkl"
CPP_dataFile_goodBr="exp4_CPP_7.2_goodBr.pkl"

def exp4a(v): # bad branch scenario
    # ----- good vs. bad branch scenario, 1-step heatmap
    set_acts = ['bus_8','bus_53','bus_57','bus_66'] # has bad branch
    addon_acts= ['bus_152'] # arbitrary

    parmObj=hm.configParms()
    parmObj.set_ctrlTypes(['PBC']*len(set_acts+addon_acts))
    parmObj.set_version(1) # 1 for PBC

    # here we set act_locs as the existing actuators
    heatmap_dic,heatMapNames=prc.find_good_colocated(parmObj,v.feeder, set_acts, addon_acts, v.node_index_map, v.substation_name, v.depths,v.file_name, v.Vbase_ll, v.Sbase)
    # result looks like: 'heatmap_colocated' + '_' + file_name

    # Saving results:
    with open(CPP_dataFile_badBr, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([heatmap_dic, heatMapNames], f)
    return

def exp4b(v): # good branch scenario

    # ----- good vs. bad branch scenario, 1-step heatmap
    set_acts = ['bus_8','bus_53','bus_57','bus_74'] # has good branch
    addon_acts= ['bus_152'] # arbitrary

    parmObj=hm.configParms()
    parmObj.set_ctrlTypes(['PBC']*len(set_acts+addon_acts))
    parmObj.set_version(1) # 1 for PBC

    # here we set act_locs as the existing actuators
    heatmap_dic,heatMapNames=prc.find_good_colocated(parmObj,v.feeder, set_acts, addon_acts, v.node_index_map, v.substation_name, v.depths,v.file_name, v.Vbase_ll, v.Sbase)
    # result looks like: 'heatmap_colocated' + '_' + file_name

    # Saving results:
    with open(CPP_dataFile_goodBr, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([heatmap_dic,heatMapNames], f)

    # plt.figure()

    return

def make_CPP_heatmaps(v):

    # need load two .pkl files
    with open(CPP_dataFile_goodBr,'rb') as f:  # Python 3: open(..., 'rb')
        heatmap_dic1,k = pickle.load(f)

    with open(CPP_dataFile_badBr,'rb') as f:  # Python 3: open(..., 'rb')
        heatmap_dic2,k = pickle.load(f)


    lst1 = [val[1] for val in heatmap_dic1.values()]  # values are [percent_feas,min_domeig_mag,bestF_asvec,bestF_asmat]
    lst2 = [val[1] for val in heatmap_dic2.values()]  # values are [percent_feas,min_domeig_mag,bestF_asvec,bestF_asmat]
    heatmap_vals=lst1+lst2 # concatenate the min domeig values
    plt.hist(heatmap_vals, density=True, bins=15)  # round to nearest hundredth
    plt.xlabel('RHP: min domeig per config')
    lst_less1 = [round(x,4) for x in heatmap_vals if x < 1]
    vals_range = [min(lst_less1), max(lst_less1)]
    print('stable heatmap_vals range from', round(vals_range[0], 4), ' to ', round(vals_range[1], 4))

    set_acts = ['bus_8','bus_53','bus_57','bus_74'] # has good branch
    addon_acts= ['bus_152'] # arbitrary
    cur_act_locs = set_acts + addon_acts
    level_vals1=vis.make_map(v, cur_act_locs, heatmap_dic1, vals_range, 'CPP_goodBr')
    print('level vals1=',level_vals1)

    set_acts = ['bus_8','bus_53','bus_57','bus_66'] # has bad branch
    addon_acts= ['bus_152'] # arbitrary
    cur_act_locs = set_acts + addon_acts
    level_vals2=vis.make_map(v, cur_act_locs, heatmap_dic2, vals_range, 'CPP_badBr')
    print('level vals2=',level_vals2)


    return
