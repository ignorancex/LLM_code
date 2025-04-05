import  numpy as np
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




# this file is not used



parser = argparse.ArgumentParser(description='training advice matrix')

parser.add_argument('-b', default=200, type=int, help='budget')
parser.add_argument('-f', default="addr", type=str, help='folder name')
parser.add_argument("-p", default=True, help='plot CAMS test')
parser.add_argument('-n', default="DRIFT", type=str, help='dataset name', choices=["DRIFT","CIFAR10","BBBP","HIV"])


args = parser.parse_args()
budget = args.b
folder_name = args.f
test_=args.p
dataset_name= args.n

path_ = os.getcwd()+"/resources/contextual_data/"+dataset_name

# Preprocess
predictions = np.loadtxt(str(path_) + "/predictions.out")
oracle_arr = np.loadtxt(str(path_) + "/oracle.out")


oracle=np.asarray(oracle_arr)

folder_name="drift_contextual_streamsize3000_numreals300_Date-2022-01-07_Time-17-14-14_which_methods00000011100_policy[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]"
plot_active_picker_test=test_

path=os.getcwd()+"/resources/results/"+folder_name+"/"

file_list=os.listdir(path)
print (file_list)

#column : number of realization
#read experiment

def organize_plot(path, budget, predictions, oracle):
    # data output
    data = np.load(path + "data.npz")
    num_reals = data["num_reals"]
    num_instances = data["num_instances"]
    num_models = data["num_models"]
    methods = data["methods"]

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
    eval_regret = eval["regret"]
    eval_cumulative_loss = eval["cumulative_loss"]
    eval_sampled_regret = eval["sampled_regret"]
    #    eval_num_queries = eval["num_queries"]
    #  eval_num_queries = eval["budgets"]
    eval_num_queries = eval["num_queries_under_budget"]
    box_budget=eval["box_budget"]
    box_cumulative_loss=eval["box_cumulative_loss"]
    box_method = eval["box_method"]


    my_dict = {'active_picker_best_policy': "random query strategy", 'active_picker_test': "variance query strategy", 'active_picker_identity': "entropy query strategy"}
    print(box_method)

    box_method = [my_dict[zi] for zi in box_method]

    box_df={"budget":box_budget ,"c_regret":box_cumulative_loss,"method":box_method}

    box_df= pd.DataFrame(box_df)

    # Regret

    # Initialize
    loss_true = 0
    loss_winner = 0
   #    sns.set(rc = {'figure.figsize':(23,8)})
    plt.figure(figsize=(10, 10), dpi=80)
#    sns.set(font_scale = 5)
    ax = sns.boxplot(x="budget", y="c_regret", hue="method", data=box_df,palette=my_pal, linewidth=0.5, width=0.8,showfliers = False)
#    sns.despine(offset=10, trim=False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(dataset_name+", CAMS with various query strategies", fontsize=25)
    plt.xlabel("Budget", fontsize=22)
    plt.ylabel("Regret", fontsize=22)
    plt.legend(loc=2)
    plt.legend(fontsize = 21)
    plt.savefig(dataset_name+"_compare_3_query_strategy_box_plot.png", bbox_inches='tight', pad_inches=0.01)
    plt.savefig(dataset_name+"_compare_3_query_strategy_box_plot.pdf", bbox_inches='tight', pad_inches=0.01)
 #   plt.show()
    regret_t = np.zeros((len(methods), num_instances))
    sampled_regret_t = np.zeros((len(methods), num_instances))
    cumulative_loss_t = np.zeros((len(methods), num_instances))

    true_precisions = compute_precisions(predictions, oracle, num_models)
    true_winner = np.where(np.equal(true_precisions, np.max(true_precisions)))[0]
    winner_randint = np.random.randint(len(true_winner))
    true_winner_random = true_winner[winner_randint]

    streaming_first_realizaiton = streaming_instances_log[:, 0]

    predictions = predictions[streaming_first_realizaiton, :]
    oracle = oracle[streaming_first_realizaiton]

    winner_randint = np.random.randint(len(true_winner))
    true_winner_random = true_winner[winner_randint]

    for num in range(len(methods)):

        zt_real = idx_budget_log[:, 0, num]  # num method in first realization
        posterior_real = posterior_log[:, :, 0]
        posterior_real_ap = posterior_log_ap[:, :, 0]
        posterior_real_ap_identity = posterior_log_ap_identity[:, :, 0]
        posterior_real_ap_test = posterior_log_ap_test[:, :, 0]
        posterior_real_contextual_qbc = posterior_log_contextual_qbc[:, :, 0]
        posterior_real_contextual_iwal = posterior_log_contextual_iwal[:, :, 0]

        # labelled_ins = np.squeeze(np.asarray(zt_real.nonzero())) # the indices whose labels are queried
        labelled_ins = np.ravel(np.asarray(zt_real.nonzero()))  # the indices whose labels are queried
        num_labelled = np.size(labelled_ins)  # number of queries for this realization ~budget in interest
        if num_labelled == 0:
            labelled_ins = 0
            num_labelled = 1

        cumulative_regrets = []
        sampled_regret_real = 0
        regret_real = 0
        cumulative_loss_real = 0

        method = methods[num]
        print("method", method)
        for t in np.arange(num_instances):

            if method == "active_picker_best_policy":
                posterior_t = posterior_real_ap[t, :]
                arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

            elif method == "active_picker_identity":
                posterior_t = posterior_real_ap_identity[t, :]
                arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

            elif method == "active_picker_test":
                posterior_t = posterior_real_ap_test[t, :]
                arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

            elif method == "contextual_qbc":
                posterior_t = posterior_real_contextual_qbc[t, :]
                arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

            elif method == "contextual_iwal":
                posterior_t = posterior_real_contextual_iwal[t, :]
                arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

            elif method == 'mp':  # If MP, use its own posterior
                # print(method)
                posterior_t = posterior_real[t, :]
                arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

            else:  # else, check the weighted losses
                posterior_t = np.ones(num_models) / num_models
                if num_labelled == 1:
                    labelled_instances_t = 0
                else:
                    idx_labelled_instances_transient = np.where(labelled_ins.reshape(num_labelled, 1) < t)[
                        0]  # find the location of labelled points that are smaller than t
                    labelled_instances_t = labelled_ins[
                        idx_labelled_instances_transient]  # find all labelled points so far
                weighted_losses_t = compute_loss(predictions[labelled_instances_t, :], oracle[labelled_instances_t],
                                                 num_models)
                if np.size(labelled_instances_t) > 1:
                    if np.sum(weighted_losses_t) == 0:  # if no true positive yet, set the posterior uniform
                        arg_winners_t = np.arange(num_models)
                    else:
                        arg_winners_t = \
                        np.where(np.equal(weighted_losses_t.reshape(num_models, 1), np.min(weighted_losses_t)))[
                            0]
                else:
                    arg_winners_t = np.arange(num_models)

            # If multi winners, choose randomly
            len_winners = np.size(arg_winners_t)

            if len_winners > 1:
                idx_winner_t = np.random.choice(len_winners, 1)
                winner_t = arg_winners_t[idx_winner_t]
            else:
                winner_t = arg_winners_t

            # Accumulate the error of returned model
            loss_winner = int((predictions[t, int(winner_t)] != oracle[t]) * 1)
            # Accumulate the error of true winner
            loss_true = int((predictions[t, int(true_winner_random)] != oracle[t]) * 1)

            # Sampled regret time
            m_star = np.random.choice(list(range(num_models)), p=posterior_t)
            # Incur hidden loss
            loss_sampled = (predictions[t, m_star] != oracle[t]) * 1

            orac_rep = np.repeat(int(oracle[t]), len(predictions[t, :]))
            val = (predictions[t, :] != orac_rep) * 1

            cumulative_loss_real += (loss_winner - np.min(val))
            regret_real += (loss_winner - loss_true)
            sampled_regret_real += (loss_sampled - loss_true)
            # print(regret_real)
            regret_t[num, t] = regret_real
            sampled_regret_t[num, t] = sampled_regret_real
            cumulative_loss_t[num, t] = cumulative_loss_real

    return regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries


regret_t, cumulative_loss_t, sampled_regret_t, num_instances, methods, eval_regret, eval_cumulative_loss, eval_sampled_regret, eval_num_queries = organize_plot(
    path, budget, predictions, oracle)

rounds = np.linspace(0, num_instances, num_instances)
rounds = np.round(rounds)

for item in methods:
    print(item)


##########################################################
# num of query
##########################################################


plt.figure(figsize=(10, 10))
if "mp" in methods:
    index = np.where(methods == "mp")[0][0]
    plt.plot(eval_num_queries[index, :], eval_regret[:, index], "k-", label=methods[index])

if "qbc" in methods:
    index = np.where(methods == "qbc")[0][0]
    plt.plot(eval_num_queries[index, :], eval_regret[:, index], "b-", label=methods[index])

if "sqbc" in methods:
    index = np.where(methods == "sqbc")[0][0]
    plt.plot(eval_num_queries[index, :], eval_regret[:, index], "g-", label=methods[index])

if "rs" in methods:
    index = np.where(methods == "rs")[0][0]
    plt.plot(eval_num_queries[index, :], eval_regret[:, index], "c-", label=methods[index])

if "iwal" in methods:
    index = np.where(methods == "iwal")[0][0]
    plt.plot(eval_num_queries[index, :], eval_regret[:, index], "m-", label=methods[index])

if "efal" in methods:
    index = np.where(methods == "efal")[0][0]
    plt.plot(eval_num_queries[index, :], eval_regret[:, index], "y-", label=methods[index])

if "active_picker_identity" in methods:
    index = np.where(methods == "active_picker_identity")[0][0]
    plt.plot(eval_num_queries[index, :], eval_regret[:, index], "r--", label="entropy query strategy")

if plot_active_picker_test:
    if "active_picker_test" in methods:
        index = np.where(methods == "active_picker_test")[0][0]
        plt.plot(eval_num_queries[index, :], eval_regret[:, index], "r-.", label="variance query strategy")

if "active_picker_best_policy" in methods:
    index = np.where(methods == "active_picker_best_policy")[0][0]
    plt.plot(eval_num_queries[index, :], eval_regret[:, index], "r-", label="random query strategy")

if "contextual_qbc" in methods:
    index = np.where(methods == "contextual_qbc")[0][0]
    plt.plot(eval_num_queries[index, :], eval_regret[:, index], "y-.", label=methods[index])

if "contextual_iwal" in methods:
    index = np.where(methods == "contextual_iwal")[0][0]
    plt.plot(eval_num_queries[index, :], eval_regret[:, index], "m-.", label=methods[index])



plt.figure(figsize=(10, 10))
if "mp" in methods:
    index = np.where(methods == "mp")[0][0]

    plt.plot(eval_num_queries[index, :], eval_cumulative_loss[:, index], "k-", label=methods[index])

if "qbc" in methods:
    index = np.where(methods == "qbc")[0][0]
    plt.plot(eval_num_queries[index, :], eval_cumulative_loss[:, index], "b-", label=methods[index])

if "sqbc" in methods:
    index = np.where(methods == "sqbc")[0][0]
    plt.plot(eval_num_queries[index, :], eval_cumulative_loss[:, index], "g-", label=methods[index])

if "rs" in methods:
    index = np.where(methods == "rs")[0][0]
    plt.plot(eval_num_queries[index, :], eval_cumulative_loss[:, index], "c-", label=methods[index])

if "iwal" in methods:
    index = np.where(methods == "iwal")[0][0]
    plt.plot(eval_num_queries[index, :], eval_cumulative_loss[:, index], "m-", label=methods[index])

if "efal" in methods:
    index = np.where(methods == "efal")[0][0]
    plt.plot(eval_num_queries[index, :], eval_cumulative_loss[:, index], "y-", label=methods[index])

if "active_picker_identity" in methods:
    index = np.where(methods == "active_picker_identity")[0][0]
    plt.plot(eval_num_queries[index, :], eval_cumulative_loss[:, index], "r-", label="entropy query strategy")

if plot_active_picker_test:
    if "active_picker_test" in methods:
        index = np.where(methods == "active_picker_test")[0][0]
        plt.plot(eval_num_queries[index, :], eval_cumulative_loss[:, index], "tab:orange", label="variance query strategy")

if "active_picker_best_policy" in methods:
    index = np.where(methods == "active_picker_best_policy")[0][0]
    plt.plot(eval_num_queries[index, :], eval_cumulative_loss[:, index], "purple", label="random query strategy")

if "contextual_qbc" in methods:
    index = np.where(methods == "contextual_qbc")[0][0]
    plt.plot(eval_num_queries[index, :], eval_cumulative_loss[:, index], "y-.", label=methods[index])

if "contextual_iwal" in methods:
    index = np.where(methods == "contextual_iwal")[0][0]
    plt.plot(eval_num_queries[index, :], eval_cumulative_loss[:, index], "m-.", label=methods[index])


#mpl.rcParams["font.size"] = 16
plt.xlabel("Query cost", fontsize=22)
plt.ylabel("Cumulative Loss", fontsize=22)
plt.title(dataset_name, fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc=1)
plt.legend(fontsize = 21)
plt.savefig(dataset_name+"_compare_3_query_cumulative_loss.png", bbox_inches='tight', pad_inches=0.01)
plt.savefig(dataset_name+"_compare_3_query_cumulative_loss.pdf", bbox_inches='tight', pad_inches=0.01)
#plt.show()



print("budget:", budget)
print("folder_name:", folder_name)

