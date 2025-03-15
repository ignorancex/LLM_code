# coding: utf-8

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np

def gengraph(logfile,savedir="graph",figname = "train_graph.png",mode="SSD",key_select = None, output_csv = False, ylim = None, m_avr = None):
    with open(logfile) as f:
        # read json file
        ch_log = json.load(f)

    ylim = ylim

    if mode =="SSD":
        # loss = [ep['main/loss'] for ep in ch_log]
        # loss_loc = [ep['main/loss/loc'] for ep in ch_log]
        # loss_conf = [ep['main/loss/conf'] for ep in ch_log]
        keys = ['main/loss','main/loss/loc','main/loss/conf']
    elif mode =="ADDA":
        # loss_t_enc = [ep['loss_t_enc'] for ep in ch_log]
        # loss_dis = [ep['loss_dis'] for ep in ch_log]
        keys = ['loss_t_enc','loss_dis']
    elif mode == "DA_loss":
        keys = ['loss_cls', "loss_t_enc","loss_dis"]
    elif mode == "DA_eval":
        keys = ["validation/main/map", "validation/main/F1/car"]
    elif mode == "DA_eval_s":
        keys = ["validation_s/main/map", "validation_s/main/F1/car"]
    elif mode == "CORAL_loss":
        keys = ['main/cls_loss','main/CORAL_loss_weighted']
    elif mode == "CORAL_loss_":
        keys = ['loss_cls','CORAL_loss_weighted']
    elif mode == "CORAL_eval":
        keys = ['validation_1/main/map','validation_1/main/F1/car']
    elif mode == "Rec_loss":
        keys = ["loss_rec_source","loss_rec_target"]
    elif mode == "Rec_loss_cls":
        keys = ["loss_cls"]
    elif mode == "Rec_loss_sem":
        keys = ["loss_sem"]
    elif mode == "Rec_eval":
        keys = ["validation/main/map", "validation/main/F1/car"]
        mode = "DA_eval"
    elif mode == "Mutual_loss":
        keys = ["loss_cls_diff", "loss_loc_diff", "loss_conf_diff"]
    elif mode == "loss_radv":
        keys = ["loss_rec_fool","loss_dis_raw","loss_dis_src_raw","loss_dis_tgt_raw"]
    elif mode == "loss_radv_cond":
        keys = ["loss_dis_tgt_raw_cond", "loss_gen_src_raw_cond"]
    elif mode == "loss_coGAN":
        keys = ["loss_cls", "loss_cls_org"]
    elif mode == "loss_cls_align":
        keys = ["loss_dis_cls", "loss_dis_cls_src","loss_dis_cls_tgt","loss_dis_cls_enc"]
    elif mode == "RR_FAR":
        keys = ["validation/main/RR/car", "validation/main/FAR/car"]
    elif mode == "entropy":
        keys = ["loss_t_entropy"]
        # ylim = [0, 0.1]
    data = defaultdict(lambda :[[],[]])
    for key in keys:
        for ep in ch_log:
            try:
                data[key][1].append(ep[key])
                data[key][0].append(ep['iteration'])
            except KeyError:
                pass
    # itr = [ep['iteration'] for ep in ch_log]

    if m_avr:
        data_ = defaultdict(lambda: [[], []])
        for key in keys:
            rec_num = len(data[key][0])
            if rec_num > m_avr:
                for i in range(rec_num - m_avr + 1):
                    data_[key][1].append(sum(data[key][1][i:i+m_avr])/m_avr)
                    data_[key][0].append(data[key][0][i+m_avr//2])
        data = data_

    if mode == "CORAL_eval":
        map_ = data['validation_1/main/map'][:]
        f1_ = data['validation_1/main/F1/car'][:]
        difference = set(map_[0]) ^ set(f1_[0])
        for d in difference:
            if d in map_[0]:
                index = map_[0].index(d)
                map_[0].pop(index)
                map_[1].pop(index)
            if d in f1_[0]:
                index = f1_[0].index(d)
                f1_[0].pop(index)
                f1_[1].pop(index)
        for i in range(len(map_[0])):
            data['mean_ap_F1'][0].append(map_[0][i])
            data['mean_ap_F1'][1].append((map_[1][i] + f1_[1][i]) / 2)

    if mode in ["DA_eval","DA_eval_s"]:
        if mode == "DA_eval":
            map_ = data['validation/main/map'][:]
            f1_ =  data['validation/main/F1/car'][:]
        elif mode == "DA_eval_s":
            map_ = data['validation_s/main/map'][:]
            f1_ = data['validation_s/main/F1/car'][:]
        difference = set(map_[0])^set(f1_[0])
        for d in difference:
            if d in map_[0]:
                index = map_[0].index(d)
                map_[0].pop(index)
                map_[1].pop(index)
            if d in f1_[0]:
                index = f1_[0].index(d)
                f1_[0].pop(index)
                f1_[1].pop(index)
        for i in range(len(map_[0])):
            data['mean_ap_F1'][0].append(map_[0][i])
            data['mean_ap_F1'][1].append((map_[1][i]+f1_[1][i])/2)

    if not os.path.isdir(savedir): os.makedirs(savedir)
    savepath = os.path.join(savedir,figname)

    plt.figure(figsize=(10,6))
    #plt.figure()
    if mode in ["DA_eval", "DA_eval_s", "CORAL_eval"]:
        plt.title("Peformance evaluation")
        plt.ylabel("Indicators")
    else:
        plt.title("Training loss")
        plt.ylabel("Loss")
    plt.xlabel("Iteration")

    # if mode == "SSD":
    #     plt.plot(itr, loss, label="Loss")
    #     plt.plot(itr, loss_loc, label="Loss_loc")
    #     plt.plot(itr, loss_conf, label="Loss_conf")
    # elif mode == "ADDA":
    #     plt.plot(itr, loss_t_enc, label="Loss_t_enc")
    #     plt.plot(itr, loss_dis, label="Loss_dis")
    # map = np.array(data["validation/main/map"])
    # f1 = np.array(data["validation/main/F1/car"])
    #index_sorted_
    # try:
    #     map = data["validation/main/map"]
    #     f1 = data["validation/main/F1/car"]
    #     mean = []
    #     for i in range(len(map)):
    #         if map[i] != None and f1[i] != None:
    #             mean.append((i,(map[i]+f1[i])/2))
    #     mean.sort(key=lambda x: x[1],reverse=True)
    #     for j in range(10):
    #         print("map:{0},f1:{1},mean:{2} iter{3}\n".format(map[mean[j][0]],f1[mean[j][0]],mean[j][1],itr[mean[j][0]]))
    # except KeyError:
    #     pass
    # for key in data.keys():
    #     d_array = np.array(data[key])
    #     index_sorted = np.argsort(d_array)[::-1]
    #     print("top arguments:")
    #     print(index_sorted[0:10])
    #     print("top values:")
    #     print(d_array[index_sorted[0:10]])

    if not output_csv:
        if key_select:
            for key in key_select:
                plt.plot(data[key][0], data[key][1],
                         label=key.replace("main/", "").replace("validation/", "").replace("validation_1/", "").replace("validation_s/", ""))
        else:
            for key in data.keys():
                plt.plot(data[key][0], data[key][1], label=key.replace("main/","").replace("validation/","").replace("validation_1/","").replace("validation_s/", ""))

    if output_csv:
        root, ext = os.path.splitext(savepath)
        save_csv = root + ".csv"
        if key_select:
            selected_keys = key_select
        else:
            selected_keys = data.keys()
        max_length = 0
        for key in selected_keys:
            max_length = max(max_length,len(data[key][0]))
        max_length +=2
        write_data = [[] for x in range(max_length)]
        for key in selected_keys:
            write_data[0].extend([key,"",""])
            write_data[1].extend(["iter","data",""])
            for i in range(len(data[key][0])):
                write_data[i+2].extend([data[key][0][i],data[key][1][i],""])

            # temp_data = [key,"iter"]
            # temp_data.extend(data[key][0])
            # write_data.append(temp_data)
            # temp_data = ["", "iter"]
            # temp_data.extend(data[key][1])
            # write_data.append(temp_data)
            # # write_data.append([key, "iter"].extend(data[key][0]))
            # # write_data.append(["", "data"].extend(data[key][1]))
            # write_data.append([])
        with open(save_csv,'w') as f:
            import csv
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(write_data)

    if not output_csv:
        if ylim == None:
            if mode =="ssd" or mode.find("loss")>-1:
                ylim = [0, 1]
                # plt.ylim([0.,1])
                # plt.ylim([0., 1.5])
            elif mode.find("eval")>1:
                ylim = [0, 0.9]
                # plt.ylim([0.0, 0.9])
                # plt.xlim([0.0, 6000])
        plt.ylim(ylim)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(right=0.6)
        plt.savefig(savepath)
        # plt.show()

    plt.close()

if __name__ == "__main__":
    # gengraph("E:/work/DA_vehicle_detection/model/DA/m_thesis/4_CORAL_x5_nmargin_r1_1/log",savedir='E:/work/DA_vehicle_detection/model/DA/m_thesis/4_CORAL_x5_nmargin_r1_1/graph',figname="train_eval.png",mode="CORAL_eval",output_csv=False) #,key_select=('mean_ap_F1',))
    # gengraph("E:/work/experiments/trained_models/ssd_adv_raw_SAGAN/log", savedir='E:/work/experiments/trained_models/ssd_adv_raw_SAGAN/graphs',
    #          figname="rec_loss_sem.png", mode="Rec_loss_sem",output_csv=False)#,key_select=('validation_1/main/map',))
    logfile = "E:/work/experiments/trained_models/cogan_s_train_residual/log"
    savedir = "E:/work/experiments/trained_models/cogan_s_train_residual/graphs"
    gengraph(logfile,
             savedir=savedir,
             figname="eval.png", mode="DA_eval", #m_avr=100,  # ylim=[0.0,1],####
             output_csv=False)  # ,key_select=('validation_1/main/map',))
    gengraph(logfile,
             savedir=savedir,
             figname="eval.png", mode="DA_eval",  # m_avr=100,  # ylim=[0.0,1],####
             output_csv=True)  # ,key_select=('validation_1/main/map',))
    gengraph(logfile,
             savedir=savedir,
             figname="eval_mavr.png", mode="DA_eval", m_avr=100,#ylim=[0.0,1],####
             output_csv=True)  # ,key_select=('validation_1/main/map',))
    # gengraph(logfile,
    #          savedir=savedir,
    #          figname="eval_s.png", mode="DA_eval_s", ylim=[0.0,1], # m_avr=100,  # ,####
    #          output_csv=False)  # ,key_select=('validation_1/main/map',))
    gengraph(logfile,
             savedir=savedir,
             figname="loss_DA.png", mode="DA_loss",  ylim=[0,1],####
             output_csv=False)  # ,key_select=('validation_1/main/map',))
    # gengraph(logfile,
    #          savedir=savedir,
    #          figname="loss_cls_align_2.png", mode="loss_cls_align",   ylim=[1.3,1.5],####
    #          output_csv=False)  # ,key_select=('validation_1/main/map',))
    gengraph(logfile,
             savedir=savedir,
             figname="RR.png", mode="RR_FAR",  ylim=[0,1],####
             output_csv=False)  # ,key_select=('validation_1/main/map',))
    gengraph(logfile,
             savedir=savedir,
             figname="entropy.png", mode="entropy", ylim=[0, 0.05],  ####
             output_csv=False)  # ,key_select=('validation_1/main/map',))