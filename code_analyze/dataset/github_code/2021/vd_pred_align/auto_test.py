from SSD_test import ssd_test
import os, csv
from graph import gengraph

process_dir="cls_align_delta_norm"
# process_dir="plain_adv"
process_dir_base="E:/work/experiments/trained_models/pred_alignment_prod_run"
save_dir="E:/work/experiments/results/pred_alignment_prod_run/ladder2"
# process_iterations = [15000]
# process_iterations = [10000,11000,12000,13000,14000,15000]
# process_iterations = [15000,16000,17000,18000,19000,20000]
process_iterations = [10000]
# process_iterations = ["bestshot"]

imagepath = "E:/work/dataset/raw/DA_images/NTT_scale0.3/2_6"
# dir_list = os.listdir(os.path.join(process_dir_base, process_dir))

# for i in range(len(dir_list))[::-1]:
#     if not os.path.isdir(os.path.join(process_dir_base, process_dir, dir_list[i])):
#         dir_list.pop(i)
#     # else:
#     #     dir_list[i] = os.path.join(process_dir_base, process_dir, dir_list[i])

for k, process_iteration in enumerate(process_iterations):
    summary_stat = []
    summary_stat_ladder = []
    summary_stat_file = os.path.join(save_dir, process_dir, str(process_iteration),
                                "{}_summary_stat_iter{}.csv".format(process_dir, process_iteration))
    summary_stat_ladder_file = os.path.join(save_dir, process_dir, str(process_iteration),
                                     "{}_summary_stat_ladder_iter{}.csv".format(process_dir, process_iteration))
    summary_stat.append([process_dir,process_iteration])
    summary_stat.append(["repeat", "map", "ap", "PR", "RR", "FAR", "F1","mean_ap_F1","entropy","entropy_pos","entropy_neg"])
    print("start iteration {}>>".format(process_iteration))

    for i in range(10):
        save_dir_ = os.path.join(save_dir, process_dir, str(process_iteration), str(i))
        # if process_iteration == "bestshot":
        #     save_dir_ = os.path.join(save_dir_,"bestshot")

        if not os.path.isdir(os.path.join(process_dir_base,process_dir,str(i))):
            print("Directory: {} does not exist.".format(process_dir+"/"+str(i)))
            summary_stat.append(["{}th".format(i), "absent"])
            continue
        if process_iteration == "bestshot":
            dir_ = os.path.join(process_dir_base,process_dir,str(i),"bestshot")
            if os.path.isdir(dir_):
                files = os.listdir(dir_)
            else:
                print("bestshot in {}/{} does not exist.".format(
                    process_dir, i))
                continue
            for j in range(len(files))[::-1]:
                _, ext = os.path.splitext(files[j])
                if not ext == ".npz":
                    files.pop(j)
            modelpath = os.path.join(process_dir_base,process_dir,str(i),"bestshot",files[0])
        else:
            modelpath = os.path.join(process_dir_base, process_dir, str(i),"ssdmodel_iter_{}".format(process_iteration))
        if not os.path.isfile(modelpath):
            print("Snapshot: {} does not exist.".format(process_dir + "/" + str(i) + "/" + "ssdmodel_iter_{}".format(process_iteration)))
            summary_stat.append(["{}th".format(i),"absent"])
            continue

        ssd_test(modelpath,imagepath,procDir=True,resultdir=save_dir_,resolution=0.3,modelsize="ssd300", evalonly=True, iou_threshold=0.4,calc_entropy=True,score_thre_step=[0.9999,1.,0.000001])

        if k == 0:
            logfile = os.path.join(process_dir_base,process_dir,str(i),"log")
            savedir_graph = os.path.join(save_dir, process_dir, "graph", str(i))
            # savedir_graph = os.path.join(save_dir_, "graph")
            if os.path.isfile(logfile):
                gengraph(logfile,
                         savedir=savedir_graph,
                         figname="eval.png", mode="DA_eval",  # m_avr=100,  # ylim=[0.0,1],####
                         output_csv=False)  # ,key_select=('validation_1/main/map',))
                gengraph(logfile,
                         savedir=savedir_graph,
                         figname="eval.png", mode="DA_eval",  # m_avr=100,  # ylim=[0.0,1],####
                         output_csv=True)  # ,key_select=('validation_1/main/map',))
                gengraph(logfile,
                         savedir=savedir_graph,
                         figname="eval_mavr.png", mode="DA_eval", m_avr=100,  # ylim=[0.0,1],####
                         output_csv=True)  # ,key_select=('validation_1/main/map',))
                # gengraph(logfile,
                #          savedir=savedir,
                #          figname="eval_s.png", mode="DA_eval_s", ylim=[0.0,1], # m_avr=100,  # ,####
                #          output_csv=False)  # ,key_select=('validation_1/main/map',))
                gengraph(logfile,
                         savedir=savedir_graph,
                     figname="loss_DA.png", mode="DA_loss", ylim=[0, 5],  ####
                     output_csv=False)  # ,key_select=('validation_1/main/map',))
            else:
                print("no log file:{}".format(logfile))

        with open(os.path.join(save_dir_,"result_stat.csv")) as f:
            reader = csv.reader(f)
            results = [row for row in reader]
            results[-1][0] = "{}th".format(i)
        summary_stat.append(results[-1])

        with open(os.path.join(save_dir_,"result_stat_ladder.csv")) as f:
            reader = csv.reader(f)
            results = [row for row in reader]
            # results[-1][0] = "{}th".format(i)
        summary_stat_ladder.extend(results)
        if i < 9:
            summary_stat_ladder.append([])

    print("<<end iteration {}".format(process_iteration))

    if not os.path.isdir(os.path.join(save_dir, process_dir, str(process_iteration))):
        os.makedirs(os.path.join(save_dir, process_dir, str(process_iteration)))

    with open(summary_stat_file,"w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(summary_stat)

    with open(summary_stat_ladder_file,"w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(summary_stat_ladder)

