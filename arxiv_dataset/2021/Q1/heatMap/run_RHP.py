#region Import statements
import sys
import my_heatmapSetup_funcs as hm
import my_process_funcs as prc  # new
import pickle
import time
import matplotlib.pyplot as plt
import my_configVis_funcs as vis

#endregion


NPP_dataFile="NPP_7.2.pkl"

def exp3(v):
    # ------- run runHeatMapProcess (RHP) ---------------
    # 6.18.22
    # run RHP on neighborhood config for 1st to 7th step:
    parmObj = hm.configParms()
    parmObj.set_version(1)  # 1 for PBC

    # run RHP on neighborhood config for 7th step only:
    set_act_locs = ['bus_82', 'bus_87', 'bus_77', 'bus_49', 'bus_41', 'bus_44']
    set_perf = ['bus_77', 'bus_77', 'bus_77', 'bus_44', 'bus_44', 'bus_44']
    addon_act_nodes = ['bus_60'] # arbitrary
    addon_perf_nodes = ['bus_105']

    parmObj.set_ctrlTypes(['PBC'] * len(set_act_locs + addon_act_nodes))

    t = time.time()
    heatmap_dic, heatMapNames = prc.runHeatMapProcess(parmObj, v.feeder, set_act_locs, set_perf, addon_act_nodes,
                                                      addon_perf_nodes, v.node_index_map, v.substation_name,
                                                      v.depths, v.file_name,v.Vbase_ll, v.Sbase)
    with open(NPP_dataFile, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([heatmap_dic, heatMapNames], f)

    # lst_feas_configs is list of dictionaries
    elapsed = time.time() - t
    print('Time elapsed=', elapsed)

def exp3_viewResults(v):
    with open(NPP_dataFile,'rb') as f:  # Python 3: open(..., 'rb')
        heatmap_dic,k = pickle.load(f)

    heatmap_dic['bus_43'][1]=heatmap_dic['bus_46'][1] # todo this is a hack
    heatmap_dic['bus_45'][1]=heatmap_dic['bus_46'][1]


    domeig_lst = [val[1] for val in heatmap_dic.values()]  # values are [percent_feas,min_domeig_mag,bestF_asvec,bestF_asmat]
    domeig_lst_less1 = [round(x,4) for x in domeig_lst if x < 1]
    print('domeig_lst_less1=',domeig_lst_less1)
    # fig, ax1 = plt.subplots()
    # ax1.plot(domeig_lst_less1 , c='green',lw=2,label='min dominant eig')
    # plt.ylim([0.99*min(domeig_lst_less1 ), 1.01*max(domeig_lst_less1 )])
    # ax1.legend(loc='upper left')
    # plt.grid()

    vals_range = [min(domeig_lst_less1), max(domeig_lst_less1)]
    set_act_locs = ['bus_82', 'bus_87', 'bus_77', 'bus_49', 'bus_41', 'bus_44']
    cur_act_locs = set_act_locs
    level_vals=vis.make_map(v, cur_act_locs, heatmap_dic, vals_range, 'RHP_NBHD')
    print('level vals=',level_vals)
    return
