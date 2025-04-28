#region Import statements
import sys
import numpy as np
import my_impedance_funcs as imp
import my_heatmapSetup_funcs as hm
import my_process_funcs as prc  # new
import my_configVis_funcs as vis
import pickle
import my_feeder_funcs as ff
from datetime import date
import os
import scipy.io

#endregion

#exp 1
cfg1 = ['bus_66', 'bus_83', 'bus_56', 'bus_350', 'bus_251']  # bad config 2, evenly spaced
cfg2 = ['bus_86', 'bus_88', 'bus_91', 'bus_96', 'bus_80']  # bad config 1, cluster in most extreme voltage of feeder
cfg3 = ['bus_79', 'bus_103', 'bus_109', 'bus_80','bus_63']  # good config 1, same z2sub as config 1 but greater stability

# exp2
cfg5 = ['bus_87', 'bus_90', 'bus_92', 'bus_95', 'bus_80', 'bus_73', 'bus_70', 'bus_67', 'bus_160', 'bus_61']

# nbhd experiment
cfg6r=['bus_41','bus_44','bus_49','bus_77','bus_82','bus_87','bus_80'] # red node
cfg6o=['bus_41','bus_44','bus_49','bus_77','bus_82','bus_87','bus_62'] # orange node
cfg6y=['bus_41','bus_44','bus_49','bus_77','bus_82','bus_87','bus_104'] # yellow node

# good branch and bad branch
cfg7 = ['bus_8', 'bus_53', 'bus_57', 'bus_66','bus_86']  # has bad branch
cfg8 = ['bus_8','bus_53','bus_57','bus_74','bus_86']

# test2_act=['bus_83','bus_84','bus_79','bus_57','bus_56','bus_62']
# test2_perf=['bus_83','bus_83','bus_83','bus_57','bus_57','bus_57']
test5_act=['bus_85','bus_83','bus_80','bus_41','bus_44','bus_49'] # only on phB
test5_perf=['bus_83','bus_83','bus_83','bus_49','bus_49','bus_49']

current_date = str(date.today())[-5:]  # extract 'MM-DD'
results_foldername = 'results_' + current_date # eval_config results get exported to this dir
main_dir=os.getcwd()

def exp1(v): # eval 3 configs: 2 bad and one good

    exp_name='exp1'
    cfg_lst=[cfg1,cfg2,cfg3]
    test_num_lst=[8,9,10]
    perf_lst=cfg_lst # co-located
    justEval(cfg_lst, test_num_lst,perf_lst, v, exp_name)
    return

def exp2(v):
    # --------- section 3: run eval_config on 1-sensor config --------------
    # Need to design non-colocated configs:

    # 1 sensor deep in feeder, 1 DER on top of it
    # 10 DERs nearby, some above and some below
    # DERs all non-sensor nodes every other node

    perf_node = ['bus_76']
    cfg4 = perf_node
    #cfg6 = v.graphNodes_nosub
    #cfg6.remove(perf_node[0])

    cfg_lst=[cfg4,cfg5]
    perf_lst=[cfg4,perf_node*len(cfg5)]
    cfg_num_lst=[11,12]
    exp_name = 'exp2'
    justEval(cfg_lst, cfg_num_lst,perf_lst,v, exp_name)
    return

def exp3eval(v):
    # --------- validate nbhd config --------------

    cfg_lst=[cfg6r,cfg6o,cfg6y]
    cfg_num_lst=[13,14,15]
    perf_lst=[]
    nbhd_perf=['bus_44']*3+['bus_77']*3
    perf_lst.append(nbhd_perf+['bus_105'])
    perf_lst.append(nbhd_perf+['bus_105'])
    perf_lst.append(nbhd_perf+['bus_105'])
    exp_name = 'exp3'
    justEval(cfg_lst, cfg_num_lst,perf_lst, v, exp_name)
    return

def exp4(v): # eval 3 configs: 2 bad and one good

    exp_name='exp4'
    cfg_lst=[cfg7,cfg8]
    cfg_num_lst=[16,17]
    perf_lst=cfg_lst # co-located
    justEval(cfg_lst, cfg_num_lst,perf_lst, v, exp_name)
    return

def test5(v):
    # --------- validate nbhd config --------------

    cfg_lst=[test5_act]
    cfg_num_lst=[5]
    perf_lst=[test5_perf]
    #perf_lst.append(perf_lst)
    exp_name = 'test5'
    justEval(cfg_lst, cfg_num_lst,perf_lst, v, exp_name)
    return

def exp1_markGraph(v):

    with open('exp1_cfg2_7.3.pkl','rb') as f:  #open results from the good config
        [feas, percent_feas, bestF_asvec, bestF_asmat] = pickle.load(f)

    vis.markMultiple_actConfig([cfg1,cfg2,cfg3], v.feeder, v.file_name)
    return

def exp2_markGraph(v):
    ff.clear_graph(v.feeder)
    vis.markActuatorConfig(cfg5, v.feeder, v.file_name)  # create diagram with actuator locs marked
    return

def justEval(cfg_lst,cfg_num_lst,perf_lst,v,exp_name):

    # --------- section 1.a: run eval_config on CPP config --------------
    parmObj = hm.configParms()
    parmObj.set_version(1)  # 1 for PBC

    results_dir = results_foldername
    if not os.path.exists(results_dir):  # if results folder doesnt exist, create
        os.makedirs(results_dir)

    cfg_num=1
    for i in range(len(cfg_lst)):
        perf=perf_lst[i]
        cfg=cfg_lst[i]
        cfg_num=cfg_num_lst[i]
        # compute sum of z2subs for a given config, show they are about the same
        z2sub = imp.get_z2sub_for_config(v.feeder, cfg, v.depths)
        print('cfg z2sub'+str(cfg_num)+': '+str(np.around(z2sub, 2)))

        parmObj.set_ctrlTypes(['PBC'] * len(cfg))
        os.chdir(main_dir)
        feas, maxError, percent_feas,min_domeig_mag, bestF_asvec,bestF_asmat, indicMat = prc.eval_config(parmObj, v.feeder, cfg, perf, v.node_index_map,
                                                                                          v.substation_name, v.depths, v.file_name, v.Vbase_ll, v.Sbase)
        print('min domeig:',min_domeig_mag)
        # Saving results:
        os.chdir(results_dir)  # changes the current working directory to the given path
        # with open(exp_name+'_cfg'+str(cfg_num)+'_'+current_date+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        #     pickle.dump([feas, percent_feas, bestF_asvec,bestF_asmat], f)

        matdict = {'feas': feas,'percent_feas':percent_feas,'min_domeig_mag':min_domeig_mag,'bestF_asvec':bestF_asvec,'bestF_asmat':bestF_asmat}
        scipy.io.savemat('kgains_test'+str(cfg_num)+'_'+current_date+'.mat', matdict)

        cfg_num=cfg_num+1