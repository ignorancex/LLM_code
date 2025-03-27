import model_clustering
import copy
import numpy as np
import math

import argparse

parser = argparse.ArgumentParser(description="set task")
parser.add_argument('--task', type=str, default="multirc", help='name of a huggingface dataset')
args = parser.parse_args()

if args.task == "mnli":
    with open('leep_score_mnli.txt', 'r') as f:
        lines = f.readlines()
        leep = lines[1].split('\t')
elif args.task == "tweet":
    with open('leep_score_tweet.txt', 'r') as f:
        lines = f.readlines()
        leep = lines[1].split('\t')
elif args.task == "boolq":
    with open('leep_score_boolq.txt', 'r') as f:
        lines = f.readlines()
        leep = lines[1].split('\t')
elif args.task == "multirc":
    with open('leep_score_multirc.txt', 'r') as f:
        lines = f.readlines()
        leep = lines[1].split('\t')
#print(leep)
print("Task: %s" % args.task)
leep_exp = [math.exp(float(leep_score)) for leep_score in leep]

model_clustering_instance = model_clustering.Model_Clustering()
model_clustering_result = model_clustering_instance.do_cluster()
# print(model_clustering_result)

model_name = model_clustering_instance.models

# step 1: None singleton recall
print("---------None Singleton Recall----------")
model_score_avg = copy.deepcopy(model_clustering_instance.model_score_avg)
model_cluster_none_singleton = []
proxy_score_none_singleton = np.zeros(len(model_name)).tolist()
cluster_representatives = []
for cluster in model_clustering_result:
    # select representatives for each none singleton cluster
    if len(cluster) > 1:
        cluster_representative = -1
        cluster_max_score = 0
        for item in cluster:
            model_cluster_none_singleton.append(item)
            if cluster_representative == -1:
                cluster_representative = item
                cluster_max_score = model_score_avg[item]
            else:
                if model_score_avg[item] > cluster_max_score:
                    cluster_representative = item
                    cluster_max_score = model_score_avg[item]
        cluster_representatives.append(cluster_representative)
        print("%d: %s selected as cluster representative, leep score: %s" % (cluster_representative,
                                                                             model_name[cluster_representative],
                                                                             leep[cluster_representative]))

        # calculate proxy score
        for item in cluster:
            proxy_score_none_singleton[item] = model_score_avg[item] * leep_exp[item]

print("----result----")

# step 2: Singleton recall
print("-----------Singleton Recall------------")
for cluster in model_clustering_result:
    if len(cluster) == 1:
        curr_model_idx = cluster[0]
        curr_model_acc = np.array(model_clustering_instance.model_scores[curr_model_idx])
        curr_model_score = 0
        for item in cluster_representatives:
            representative_acc = np.array(model_clustering_instance.model_scores[item])
            cos_similarity = curr_model_acc.dot(representative_acc)/(np.linalg.norm(curr_model_acc) * np.linalg.norm(representative_acc))
            curr_model_score += (cos_similarity * leep_exp[item])
        curr_model_score /= len(cluster_representatives)
        proxy_score_none_singleton[curr_model_idx] = curr_model_score * model_score_avg[curr_model_idx]

recall_idx = []
selected_cnt = 15
while selected_cnt > 0:
    max_idx = proxy_score_none_singleton.index(max(proxy_score_none_singleton))
    recall_idx.append(str(max_idx))
    print("%d: %s selected by proxy score: %s" % (
        max_idx, model_name[max_idx], str(proxy_score_none_singleton[max_idx])))
    selected_cnt -= 1
    proxy_score_none_singleton[max_idx] *= -1

result = ','.join(recall_idx)
with open(args.task + "_recall_result.txt", "w") as f:
    f.write(result)
