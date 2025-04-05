import numpy as np
# from src.evaluation.evaluation_pipeline.evaluate_method import *
from src.evaluation.evaluation_pipeline.evaluate_realizations import *
from src.evaluation.aux.load_results import *


import matplotlib.pyplot as plt
import os
import argparse
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import config as config

my_pal= config.COLOR


parser = argparse.ArgumentParser(description='training advice matrix')

parser.add_argument('-b', default=50, type=int, help='budget')
parser.add_argument('-f', default="addr", type=str, help='folder name')
parser.add_argument("-p", default=True, help='plot CAMS test')
parser.add_argument('-d', default="VERTEBRAL", type=str, help='dataset name', choices=["DRIFT","CIFAR10","BBBP","HIV","VERTEBRAL"])


args = parser.parse_args()
budget = args.b
folder_name = args.f
test_=args.p
dataset_name= args.d

#rename macros

n_RS=config.n_RS
n_Oracle=config.n_CAMS_best_policy
n_QBC = config.n_qbc
n_IWAL = config.n_iwal
n_MP = config.n_mp
n_CQBC = config.n_contextual_qbc
n_CIWAL = config.n_contextual_iwal
n_CAMS = config.n_CAMS_identity
n_test = config.n_CAMS_test

n_entropy ="entropy"
n_variance ="variance"
n_random = "random"

def rename_method_list(methods):
    arr=[]
    for item in methods:
        if item == "rs":
            arr.append(n_RS)
        elif item == "qbc":
            arr.append(n_QBC)
        elif item == "iwal":
            arr.append(n_IWAL)
        elif item == "mp":
            arr.append(n_MP)
        elif item == "contextual_qbc":
            arr.append(n_CQBC)
        elif item == "contextual_iwal":
            arr.append(n_CIWAL)
        elif item == "CAMS_best_policy":
            arr.append(n_Oracle)
        elif item == "CAMS_identity":
            arr.append(n_CAMS)
        elif item == "CAMS_test":
            arr.append(n_test)
        else:
            print("error")
            print(item)
            exit()
    
    return arr



def rename_method(item):

    if item == "rs":
        return n_RS
    elif item == "qbc":
        return n_QBC
    elif item == "iwal":
        return n_IWAL
    elif item == "mp":
        return n_MP
    elif item == "contextual_qbc":
        return n_CQBC
    elif item == "contextual_iwal":
        return n_CIWAL
    elif item == "CAMS_best_policy":
        return n_Oracle
    elif item == "CAMS_identity":
        return n_CAMS
    elif item == "CAMS_test":
        return n_test
    else:
        print("error")
        print(item)
        exit()
    
    return arr


# column : number of realization
# read experiment

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]




#my_pal = {"mp": "tab:blue", "qbc": "darkorange", "CAMS_identity": "r", "CAMS_best_policy": "purple",
#          "sqbc": "y", "rs": "tab:green", "iwal": "tab:pink", "efal": "tab:brown", "contextual_qbc": "orange",
#          "contextual_iwal": "plum", "CAMS_test": "tab:orange"}


def organize_plot(dataset_name, budget, folder_name,my_pal=my_pal):


    path_ = os.getcwd() + "/resources/contextual_data/" + dataset_name

    # Preprocess
    predictions_arr = np.loadtxt(str(path_) + "/predictions.out")
    oracle_arr = np.loadtxt(str(path_) + "/oracle.out")

    oracle_arr = np.asarray(oracle_arr)

    plot_CAMS_test = test_

    path = os.getcwd() + "/resources/results/" + folder_name + "/"

    file_list = os.listdir(path)
    print(file_list)


    # data output
    data = np.load(path + "data.npz")
    num_reals = data["num_reals"]
    print(num_reals)
    num_instances = data["num_instances"]
    num_models = data["num_models"]
    methods = data["methods"]
    budget_raw = data["budgets"]
    experiment_result = np.load(path + "experiment_results_budget" + str(budget) + ".npz")

    idx_log = experiment_result['idx_log']  # labelled_instances: if algo decide to query
    idx_budget_log = experiment_result['idx_budget_log']  # U_t_budget: query under budget
    ct_log = experiment_result['ct_log']  # ct_log: how many instance: all 1
    streaming_instances_log = experiment_result['streaming_instances_log']
    hidden_loss_log = experiment_result['hidden_loss_log']  # loss each query
    posterior_log = experiment_result['posterior_log']
    posterior_log_ap = experiment_result["posterior_log_ap"]
    posterior_log_ap_identity = experiment_result["posterior_log_ap_identity"]
    posterior_log_ap_test = experiment_result["posterior_log_ap_test"]
    posterior_log_contextual_qbc = experiment_result["posterior_log_contextual_qbc"]
    posterior_log_contextual_iwal = experiment_result["posterior_log_contextual_iwal"]
    eval = np.load(path + "eval_results.npz")
    box_budget = eval["box_budget"]
    box_budget_actual = eval["box_budget_actual"]

    eval_regret = eval["regret"]
    eval_cumulative_loss = eval["cumulative_loss"]
    eval_sampled_regret = eval["sampled_regret"]
    #    eval_num_queries = eval["num_queries"]
    #  eval_num_queries = eval["budgets"]
    eval_num_queries = eval["num_queries_under_budget"]
    query_regardles_budget = eval["query_regardles_budget"]
    query_regardles_budget_detail = eval["query_regardles_budget_detail"]


    print(budget_raw)


    eval_cumulative_loss_final = []
    eval_num_queries_final = []
    eval_cumulative_loss_tmp = numpy.transpose(eval_cumulative_loss)

    eval_query_arr = []

    for idx in range(len(eval_num_queries)):
        eval_query_arr.append(np.mean(query_regardles_budget[:, :, idx], axis=0))

        line = eval_num_queries[idx]
        loss_ = eval_cumulative_loss_tmp[idx]
        indicator1 = ((line >= budget_raw - 10) * 1)
        print(indicator1)
        loss_1 = np.nan_to_num(indicator1 * loss_)
        query_1 = np.nan_to_num(indicator1 * eval_num_queries[idx])
        print(query_1)

        idicator2 = ((line < budget_raw - 10) * 1)
        loss_2 = np.nan_to_num(idicator2 * np.sum(idicator2 * eval_cumulative_loss_tmp[idx]) / np.sum(idicator2))
        query_2 = np.nan_to_num(idicator2 * np.sum(idicator2 * eval_num_queries[idx]) / np.sum(idicator2))
        print(query_2)

        eval_num_queries_final.append(query_1 + query_2)
        eval_cumulative_loss_final.append(loss_1 + loss_2)



    eval_query_arr_final = []
    for method_ in eval_query_arr:
        counter = 0
        array_temp = []
        for item in method_:
            counter = counter + item
            array_temp.append(counter)
        eval_query_arr_final.append(array_temp)

    eval_query_arr_final = np.asarray(eval_query_arr_final)
    #eval_cumulative_loss_final=eval_cumulative_loss_tmp
    eval_cumulative_loss_final = np.transpose(np.asarray(eval_cumulative_loss_final))
    eval_num_queries_final = np.asarray(eval_num_queries_final)

    print("before:", eval_num_queries)
    print("after:", eval_num_queries_final)

    print(eval_cumulative_loss_final)
    print(eval_num_queries_final)
    #  exit()

    box_cumulative_loss = eval["box_cumulative_loss"]
    box_method = eval["box_method"]
    box_method = rename_method_list(box_method)
    print(box_method)

    box_df = {"budget": box_budget, "c_regret": box_cumulative_loss, "method": box_method}

    box_df = pd.DataFrame(box_df)

    box_df_shading = {"budget": box_budget_actual,"budget_fixed": box_budget, "c_regret": box_cumulative_loss, "method": box_method}

    box_df_shading = pd.DataFrame(box_df_shading)

    reshape_budget=[]
    reshape_budget_fixed=[]
    horizon_error_bar=[]
    vertical_error_bar=[]
    v_mean=[]


    for index, row in box_df_shading.iterrows():
        print(row['budget'],row['budget_fixed'], row['c_regret'], row['method'])
        reshape_budget.append(row['budget'])
        horizon_error_bar.append(0)
        vertical_error_bar.append(0)
        v_mean.append(0)


        #if row['budget_fixed']-row['budget']>5:
        reshape_budget_fixed.append(find_nearest(budget_raw ,row['budget']))
        #else:
        #    reshape_budget_fixed.append(row['budget_fixed'])


    box_df_shading = {"budget": reshape_budget,"budget_fixed": reshape_budget_fixed, "c_regret": box_cumulative_loss, "method": box_method,"h_err_bar":horizon_error_bar,"v_err_bar":vertical_error_bar,"v_mean":v_mean}

    box_df_shading = pd.DataFrame(box_df_shading)

    print(box_df_shading)

    print("!!!!!",budget_raw)

    for item in methods:
        for budget_ in budget_raw:
            print(item)
            x = np.where((box_df_shading["method"]==item) & (box_df_shading["budget_fixed"]== budget_))
            y = box_df_shading.loc[x]["budget"].mean()
            y_min = box_df_shading.loc[x]["budget"].min()
            y_max = box_df_shading.loc[x]["budget"].max()

            h_bar = np.maximum(abs(y-y_min),abs(y-y_max))

            box_df_shading.iloc[[x], [box_df_shading.columns.get_loc("budget_fixed")]]=y
            box_df_shading.iloc[[x], [box_df_shading.columns.get_loc("h_err_bar")]] = h_bar

            y = box_df_shading.loc[x]["c_regret"].mean()
            std = box_df_shading.loc[x]["c_regret"].std()
            count = box_df_shading.loc[x]["c_regret"].count()
            box_df_shading.iloc[[x], [box_df_shading.columns.get_loc("v_err_bar")]] = 1.95*std/np.sqrt(count)
            box_df_shading.iloc[[x], [box_df_shading.columns.get_loc("v_mean")]] = y

    shade_df_2=box_df_shading.filter(["budget_fixed","h_err_bar","v_err_bar","v_mean","method","c_regret"],axis=1).drop_duplicates().reset_index(drop=True)

    shade_df=box_df_shading.filter(["budget_fixed","h_err_bar","v_err_bar","v_mean","method"],axis=1).drop_duplicates().reset_index(drop=True)
    print(box_df_shading)
    print(shade_df)

    #ap identity
    method_list=shade_df.filter(["method"],axis=1).drop_duplicates()
    print(method_list)

    # Initialize
    loss_true = 0
    loss_winner = 0

    plt.figure(figsize=(10, 10), dpi=300)
    #    sns.set(font_scale = 5)

    if config.q=="arg":
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_RS, data=shade_df_2[shade_df_2["method"]==n_RS],color=my_pal[n_RS],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "arg_q_arg_w_can_E_with_reg", data=shade_df_2[shade_df_2["method"]==n_Oracle],color=my_pal[n_MP],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_QBC, data=shade_df_2[shade_df_2["method"]==n_QBC],color=my_pal[n_QBC],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_IWAL, data=shade_df_2[shade_df_2["method"]==n_IWAL],color=my_pal[n_IWAL],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_MP, data=shade_df_2[shade_df_2["method"]==n_MP],color=my_pal[n_MP],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "arg_q_arg_w_can_E_no_reg", data=shade_df_2[shade_df_2["method"]==n_CQBC],color=my_pal[n_CQBC],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "arg_q_arg_w_raw_E_no_reg", data=shade_df_2[shade_df_2["method"]==n_CIWAL],color=my_pal[n_CIWAL],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "arg_q_arg_w_raw_E_with_reg", data=shade_df_2[shade_df_2["method"]==n_test],color=my_pal[n_random],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_raw_E_with_reg", data=shade_df_2[shade_df_2["method"]==n_CAMS],color=my_pal[n_entropy],ci=63, linewidth=4)

    elif config.q=="weighted":
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_RS, data=shade_df_2[shade_df_2["method"]==n_RS],color=my_pal[n_RS],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_can_E_with_reg", data=shade_df_2[shade_df_2["method"]==n_Oracle],color=my_pal[n_MP],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_QBC, data=shade_df_2[shade_df_2["method"]==n_QBC],color=my_pal[n_QBC],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_IWAL, data=shade_df_2[shade_df_2["method"]==n_IWAL],color=my_pal[n_IWAL],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_MP, data=shade_df_2[shade_df_2["method"]==n_MP],color=my_pal[n_MP],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_can_E_no_reg", data=shade_df_2[shade_df_2["method"]==n_CQBC],color=my_pal[n_CQBC],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_raw_E_no_reg", data=shade_df_2[shade_df_2["method"]==n_CIWAL],color=my_pal[n_CIWAL],ci=63, linewidth=1)
#        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_raw_E_with_reg", data=shade_df_2[shade_df_2["method"]==n_test],color=my_pal[n_random],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_raw_E_with_reg", data=shade_df_2[shade_df_2["method"]==n_CAMS],color=my_pal[n_entropy],ci=63, linewidth=4)

    elif config.q=="random":
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_RS, data=shade_df_2[shade_df_2["method"]==n_RS],color=my_pal[n_RS],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "random_q_arg_w_can_E_with_reg", data=shade_df_2[shade_df_2["method"]==n_Oracle],color=my_pal[n_MP],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_QBC, data=shade_df_2[shade_df_2["method"]==n_QBC],color=my_pal[n_QBC],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_IWAL, data=shade_df_2[shade_df_2["method"]==n_IWAL],color=my_pal[n_IWAL],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_MP, data=shade_df_2[shade_df_2["method"]==n_MP],color=my_pal[n_MP],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "random_q_arg_w_can_E_no_reg", data=shade_df_2[shade_df_2["method"]==n_CQBC],color=my_pal[n_CQBC],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "random_q_arg_w_raw_E_no_reg", data=shade_df_2[shade_df_2["method"]==n_CIWAL],color=my_pal[n_CIWAL],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "random_q_arg_w_raw_E_with_reg", data=shade_df_2[shade_df_2["method"]==n_test],color=my_pal[n_random],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_raw_E_with_reg", data=shade_df_2[shade_df_2["method"]==n_CAMS],color=my_pal[n_entropy],ci=63, linewidth=4)


    elif config.q=="weighted_can_E_forward":
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_RS, data=shade_df_2[shade_df_2["method"]==n_RS],color=my_pal[n_RS],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_can_E_with_reg_can_E_fd", data=shade_df_2[shade_df_2["method"]==n_Oracle],color=my_pal[n_MP],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_QBC, data=shade_df_2[shade_df_2["method"]==n_QBC],color=my_pal[n_QBC],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_IWAL, data=shade_df_2[shade_df_2["method"]==n_IWAL],color=my_pal[n_IWAL],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = n_MP, data=shade_df_2[shade_df_2["method"]==n_MP],color=my_pal[n_MP],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_can_E_no_reg_can_E_fd", data=shade_df_2[shade_df_2["method"]==n_CQBC],color=my_pal[n_CQBC],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_raw_E_no_reg_can_E_fd", data=shade_df_2[shade_df_2["method"]==n_CIWAL],color=my_pal[n_CIWAL],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_raw_E_with_reg_can_E_fd", data=shade_df_2[shade_df_2["method"]==n_test],color=my_pal[n_random],ci=63, linewidth=1)
        sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_raw_E_with_reg_can_E_fd", data=shade_df_2[shade_df_2["method"]==n_CAMS],color=my_pal[n_entropy],ci=63, linewidth=4)


    else:
        print(config.q)
        print("warning!! no config q")
        exit()



    '''
    sns.lineplot(x="budget_fixed", y="c_regret", label = n_RS, data=shade_df_2[shade_df_2["method"]==n_RS],color=my_pal[n_RS],ci=63, linewidth=1)
    sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_raw_E_no_reg", data=shade_df_2[shade_df_2["method"]==n_Oracle],color=my_pal[n_MP],ci=63, linewidth=1)
    sns.lineplot(x="budget_fixed", y="c_regret", label = n_QBC, data=shade_df_2[shade_df_2["method"]==n_QBC],color=my_pal[n_QBC],ci=63, linewidth=1)
    sns.lineplot(x="budget_fixed", y="c_regret", label = n_IWAL, data=shade_df_2[shade_df_2["method"]==n_IWAL],color=my_pal[n_IWAL],ci=63, linewidth=1)
    sns.lineplot(x="budget_fixed", y="c_regret", label = n_MP, data=shade_df_2[shade_df_2["method"]==n_MP],color=my_pal[n_MP],ci=63, linewidth=1)
    sns.lineplot(x="budget_fixed", y="c_regret", label = "sto_q_arg_w_can_E_with_reg", data=shade_df_2[shade_df_2["method"]==n_CQBC],color=my_pal[n_CQBC],ci=63, linewidth=1)
    sns.lineplot(x="budget_fixed", y="c_regret", label = "sto_q_arg_w_can_E_no_reg", data=shade_df_2[shade_df_2["method"]==n_CIWAL],color=my_pal[n_CIWAL],ci=63, linewidth=1)
    sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_can_E_no_reg", data=shade_df_2[shade_df_2["method"]==n_test],color=my_pal[n_random],ci=63, linewidth=1)
    sns.lineplot(x="budget_fixed", y="c_regret", label = "weighted_q_arg_w_raw_E_with_reg", data=shade_df_2[shade_df_2["method"]==n_CAMS],color=my_pal[n_entropy],ci=63, linewidth=4)
    '''

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
#    plt.title(dataset_name, fontsize=25)
    plt.xlabel("Query cost", fontsize=30)
#    plt.ylabel("Cumulative Loss", fontsize=30)
    plt.ylabel("", fontsize=30)
    plt.legend(loc=2)
    plt.legend(fontsize=24,title=None)
    if config.q=="arg":
        plt.savefig("./task8_arg_q/"+dataset_name + "_task8_CAMS_model_selections_shade_line_arg.png", bbox_inches='tight', pad_inches=0.01)
        plt.savefig("./task8_arg_q/"+dataset_name + "_task8_CAMS_model_selections_shade_line_arg.pdf", bbox_inches='tight', pad_inches=0.01)
    elif config.q == "weighted":
        plt.savefig("./task8_weighted/"+dataset_name + "_task8_CAMS_model_selections_shade_line_weighted.png", bbox_inches='tight', pad_inches=0.01)
        plt.savefig("./task8_weighted/"+dataset_name + "_task8_CAMS_model_selections_shade_line_weighted.pdf", bbox_inches='tight', pad_inches=0.01)
    elif config.q == "random":
        plt.savefig("./task8_random/"+dataset_name + "_task8_CAMS_model_selections_shade_line_random.png", bbox_inches='tight', pad_inches=0.01)
        plt.savefig("./task8_random/"+dataset_name + "_task8_CAMS_model_selections_shade_line_random.pdf", bbox_inches='tight', pad_inches=0.01)
    elif config.q == "weighted_can_E_forward":
        plt.savefig("./task8_weighted_can_E_forward/"+dataset_name + "_task8_CAMS_model_selections_shade_line_forwward_can_E.png", bbox_inches='tight', pad_inches=0.01)
        plt.savefig("./task8_weighted_can_E_forward/"+dataset_name + "_task8_CAMS_model_selections_shade_line_forwward_can_E.pdf", bbox_inches='tight', pad_inches=0.01)
    else:
        print("warning!!")
        exit()


#     #   plt.show()
#     regret_t = np.zeros((num_reals, len(methods), num_instances))
#     sampled_regret_t = np.zeros((num_reals, len(methods), num_instances))
#     cumulative_loss_t = np.zeros((num_reals, len(methods), num_instances))

#     regret_t_mean = np.zeros((len(methods), num_instances))
#     sampled_regret_t_mean = np.zeros((len(methods), num_instances))
#     cumulative_loss_t_mean = np.zeros((len(methods), num_instances))

#     #    cumulative_loss_t_all = np.zeros((num_instances, len(data._methods)))

#     relative_shade_methods=[]
#     relative_shade_regret=[]
#     relative_shade_round=[]
#     relative_shade_instance=[]

#     for real_idx in range(num_reals):

#         streaming_first_realizaiton = streaming_instances_log[:, real_idx]

#         predictions = predictions_arr[streaming_first_realizaiton, :]
#         oracle = oracle_arr[streaming_first_realizaiton]

#         true_precisions = compute_precisions(predictions, oracle, num_models)
#         true_winner = np.where(np.equal(true_precisions, np.max(true_precisions)))[0]
#         winner_randint = np.random.randint(len(true_winner))
#         true_winner_random = true_winner[winner_randint]

#         winner_randint = np.random.randint(len(true_winner))
#         true_winner_random = true_winner[winner_randint]

#         for num in range(len(methods)):
#             zt_real = idx_budget_log[:, real_idx, num]  # num method in first realization
#             posterior_real = posterior_log[:, :, real_idx]
#             posterior_real_ap = posterior_log_ap[:, :, real_idx]
#             posterior_real_ap_identity = posterior_log_ap_identity[:, :, real_idx]
#             posterior_real_ap_test = posterior_log_ap_test[:, :, real_idx]
#             posterior_real_contextual_qbc = posterior_log_contextual_qbc[:, :, real_idx]
#             posterior_real_contextual_iwal = posterior_log_contextual_iwal[:, :, real_idx]

#             # labelled_ins = np.squeeze(np.asarray(zt_real.nonzero())) # the indices whose labels are queried
#             labelled_ins = np.ravel(np.asarray(zt_real.nonzero()))  # the indices whose labels are queried
#             num_labelled = np.size(labelled_ins)  # number of queries for this realization ~budget in interest
#             if num_labelled == 0:
#                 labelled_ins = 0
#                 num_labelled = 1

#             cumulative_regrets = []
#             sampled_regret_real = 0
#             regret_real = 0
#             cumulative_loss_real = 0

#             method = methods[num]
#             print("method", method)
#             for t in np.arange(num_instances):

#                 if method == "CAMS_best_policy":
#                     posterior_t = posterior_real_ap[t, :]
#                     arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

#                 elif method == "CAMS_identity":
#                     posterior_t = posterior_real_ap_identity[t, :]
#                     arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

#                 elif method == "CAMS_test":
#                     posterior_t = posterior_real_ap_test[t, :]
#                     arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

#                 elif method == "contextual_qbc":
#                     posterior_t = posterior_real_contextual_qbc[t, :]
#                     arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

#                 elif method == "contextual_iwal":
#                     posterior_t = posterior_real_contextual_iwal[t, :]
#                     arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

#                 elif method == 'mp':  # If MP, use its own posterior
#                     # print(method)
#                     posterior_t = posterior_real[t, :]
#                     arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

#                 else:  # else, check the weighted losses
#                     posterior_t = np.ones(num_models) / num_models
#                     if num_labelled == 1:
#                         labelled_instances_t = 0
#                     else:
#                         idx_labelled_instances_transient = np.where(labelled_ins.reshape(num_labelled, 1) < t)[
#                             0]  # find the location of labelled points that are smaller than t
#                         labelled_instances_t = labelled_ins[
#                             idx_labelled_instances_transient]  # find all labelled points so far

#                     weighted_losses_t = compute_loss(predictions[labelled_instances_t, :], oracle[labelled_instances_t],
#                                                      num_models)
#                     if np.size(labelled_instances_t) > 1:
#                         if np.sum(weighted_losses_t) == 0:  # if no true positive yet, set the posterior uniform
#                             arg_winners_t = np.arange(num_models)
#                         else:
#                             arg_winners_t = \
#                                 np.where(np.equal(weighted_losses_t.reshape(num_models, 1), np.min(weighted_losses_t)))[
#                                     0]
#                     else:
#                         arg_winners_t = np.arange(num_models)

#                 # If multi winners, choose randomly
#                 len_winners = np.size(arg_winners_t)

#                 if len_winners > 1:
#                     idx_winner_t = np.random.choice(len_winners, 1)
#                     winner_t = arg_winners_t[idx_winner_t]
#                 else:
#                     winner_t = arg_winners_t

#                 # Accumulate the error of returned model
#                 loss_winner = int((predictions[t, int(winner_t)] != oracle[t]) * 1)
#                 # Accumulate the error of true winner
#                 loss_true = int((predictions[t, int(true_winner_random)] != oracle[t]) * 1)

#                 # Sampled regret time
#                 m_star = np.random.choice(list(range(num_models)), p=posterior_t)
#                 # Incur hidden loss
#                 loss_sampled = (predictions[t, m_star] != oracle[t]) * 1

#                 orac_rep = np.repeat(int(oracle[t]), len(predictions[t, :]))
#                 val = (predictions[t, :] != orac_rep) * 1

#                 cumulative_loss_real += (loss_winner - np.min(val))
#                 regret_real += (loss_winner - loss_true)
#                 sampled_regret_real += (loss_sampled - loss_true)
#                 # print(regret_real)
#                 regret_t[real_idx, num, t] = regret_real
#                 sampled_regret_t[real_idx, num, t] = sampled_regret_real
#                 cumulative_loss_t[real_idx, num, t] = cumulative_loss_real

#                 relative_shade_methods.append(methods[num])
#                 relative_shade_regret.append(regret_real)
#                 relative_shade_instance.append(real_idx)
#                 relative_shade_round.append(t)

    
#     query_shade_methods=[]
#     query_shade_counts=[]
#     query_shade_round=[]
#     query_shade_instance=[]

#     budget_idx=np.where(budget_raw== budget)[0][0]

#     for num in range(len(methods)):
#         for real_idx in range(num_reals):
#             cnt=0
#             for t in np.arange(num_instances):
#                 cnt = cnt+ query_regardles_budget_detail[budget_idx,t,real_idx,num]
#                 query_shade_methods.append(methods[num])
#                 query_shade_round.append(t)
#                 query_shade_instance.append(real_idx)
#                 query_shade_counts.append(cnt)           

#     relative_shade_methods=rename_method_list(relative_shade_methods)
#     query_shade_methods=rename_method_list(query_shade_methods)

#     shade_relative_loss = {"method": relative_shade_methods,"relative_loss": relative_shade_regret, "round": relative_shade_round,"simulation":relative_shade_instance}
#     shade_relative_loss = pd.DataFrame(shade_relative_loss)

#     shade_query = {"method": query_shade_methods,"counts": query_shade_counts, "round": query_shade_round,"simulation":query_shade_instance}
#     shade_query = pd.DataFrame(shade_query)
# #    shade_query.to_csv("queries.csv")

    
#     for num in range(len(methods)):
#         for real_idx in range(num_reals):
#             regret_t_mean[num, :] = np.mean(regret_t[:, num, :], axis=0)
#             sampled_regret_t_mean[num, :] = np.mean(sampled_regret_t[:, num, :], axis=0)
#             cumulative_loss_t_mean[num, :] = np.mean(cumulative_loss_t[:, num, :], axis=0)
    

#     return eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t_mean, cumulative_loss_t_mean, sampled_regret_t_mean, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query



#arg
date="_Date-2022-03-03_Time-10-06"
config.q="arg"


#lambda
dataset_name="CIFAR10"
budget=200
folder_name="cifar_contextual_streamsize10000_numreals20"+date+"_which_methods00000011111_policy[11]"

organize_plot(dataset_name, budget, folder_name)


dataset_name="VERTEBRAL"
budget=80
folder_name="VERTEBRAL_contextual_streamsize80_numreals200"+date+"_which_methods00000011111_policy[0]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)

dataset_name="DRIFT"
budget=200
folder_name="drift_contextual_streamsize3000_numreals100"+date+"_which_methods00000011111_policy[1]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)

dataset_name="HIV"
budget=100
folder_name="HIV_contextual_streamsize4039_numreals100"+date+"_which_methods00000011111_policy[0]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)


#weighted
date="_Date-2022-03-03_Time-22-06"
config.q="weighted"


#lambda
dataset_name="CIFAR10"
budget=200
folder_name="cifar_contextual_streamsize10000_numreals20"+date+"_which_methods00000011111_policy[11]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)

dataset_name="VERTEBRAL"
budget=80
folder_name="VERTEBRAL_contextual_streamsize80_numreals200"+date+"_which_methods00000011111_policy[0]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)

dataset_name="DRIFT"
budget=200
folder_name="drift_contextual_streamsize3000_numreals100"+date+"_which_methods00000011111_policy[1]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)

dataset_name="HIV"
budget=100
folder_name="HIV_contextual_streamsize4039_numreals100"+date+"_which_methods00000011111_policy[0]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot(dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)



#random
date="_Date-2022-03-03_Time-22-08"
config.q="random"

#lambda
dataset_name="CIFAR10"
budget=200
folder_name="cifar_contextual_streamsize10000_numreals20"+date+"_which_methods00000011111_policy[11]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)

dataset_name="VERTEBRAL"
budget=80
folder_name="VERTEBRAL_contextual_streamsize80_numreals200"+date+"_which_methods00000011111_policy[0]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot(  dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)

dataset_name="DRIFT"
budget=200
folder_name="drift_contextual_streamsize3000_numreals100"+date+"_which_methods00000011111_policy[1]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)

dataset_name="HIV"
budget=100
folder_name="HIV_contextual_streamsize4039_numreals100"+date+"_which_methods00000011111_policy[0]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)




#random
date="_Date-2022-04-01_Time-00-33"
config.q="weighted_can_E_forward"

#lambda
dataset_name="CIFAR10"
budget=200
folder_name="cifar_contextual_streamsize10000_numreals20"+date+"_which_methods00000011111_policy[11]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)

dataset_name="VERTEBRAL"
budget=80
folder_name="VERTEBRAL_contextual_streamsize80_numreals200"+date+"_which_methods00000011111_policy[0]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot(  dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)

dataset_name="DRIFT"
budget=200
folder_name="drift_contextual_streamsize3000_numreals100"+date+"_which_methods00000011111_policy[1]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)

dataset_name="HIV"
budget=100
folder_name="HIV_contextual_streamsize4039_numreals100"+date+"_which_methods00000011111_policy[0]"

# eval_query_arr_final, eval_num_queries_final, eval_cumulative_loss_final, regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries,shade_relative_loss,shade_query = organize_plot( dataset_name, budget, folder_name)
organize_plot(dataset_name, budget, folder_name)



print("budget:", budget)
print("folder_name:", folder_name)

