import numpy as np
from scipy.spatial import distance
import random
import copy

equivalence_class_num = 15

hamming_file = 'hamming_pool_pred_226.txt'
acc_file = 'acc_lt_226.txt'

hamming_rep = np.loadtxt(hamming_file, delimiter=',')
acc_lt = np.loadtxt(acc_file, delimiter=',')

node_num = len(hamming_rep)

graph = [[0.0 for j in range(node_num)] for i in range(node_num)]

graph = np.array(graph)

for i in range(node_num-1):
    for j in range(i+1, node_num):
        graph[i,j] = distance.hamming(hamming_rep[i], hamming_rep[j])
        graph[j,i] = graph[i,j]

M = {}
for i in range(node_num):
    M[i] = 1.0

head = random.randint(0, node_num-1)

head_lt = [head]

cluster = M.copy()
node_minus_head = M.copy()
del node_minus_head[head]

cluster_lt = []
cluster_lt.append(cluster.copy())

#print(cluster_lt)

def find_farthest_node(head):
    node_dis_lt = []
    for node in node_minus_head:
        dis = graph[head, node]
        node_dis_lt.append((node, dis))

    node_dis_lt.sort(key=lambda x:x[1], reverse=True)

    node = node_dis_lt[0][0]
    return node


node2head = {}
for i in M:
    node2head[i] = head

def partition(farthest_node, head_lt, cluster_lt):
    new_cluster = {}
    for i in range(len(head_lt)):
        head = head_lt[i]
        move_node_lt = []
        for node in cluster_lt[i]:
            head2node_dis = graph[head, node]
            farthest_node2node_dis = graph[farthest_node, node]
            if farthest_node2node_dis <= head2node_dis:
                move_node_lt.append(node)
                #del cluster_lt[i][node]
                new_cluster[node] = 1
        
        for node in move_node_lt:
            del cluster_lt[i][node]
    
    new_cluster[farthest_node] = 1

    cluster_lt.append(new_cluster.copy())
    head_lt.append(farthest_node)
    
    return head_lt, cluster_lt


iter_num = equivalence_class_num-1

for i in range(iter_num):
    head = random.sample(head_lt, 1)
    farthest_node = find_farthest_node(head)
    
    del node_minus_head[farthest_node]
    head_lt, cluster_lt = partition(farthest_node, head_lt, cluster_lt)

fout = open('epsilon_net.txt', 'w') 
for i in range(len(head_lt)):
    head = head_lt[i]
    fout.write(f'{head}:')
    for node in cluster_lt[i]:
        fout.write(f'{node},')
    fout.write('\n')
fout.close()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from sklearn.manifold import TSNE
X = graph
embed_rep = TSNE(n_components=2, perplexity=10, metric='precomputed').fit_transform(X)

np.savetxt( 'tsne.rep', embed_rep)

index_lt = [i for i in range(equivalence_class_num)]
random.shuffle(index_lt)

# find min value
vmin = 1.0
vmax = 0.0
index_acc_lt = []
for i in range(equivalence_class_num):
    cluster_i = i
    cluster = cluster_lt[cluster_i]
    tmp_rep = []
    tmp_acc = []
    for j in cluster:
        tmp_rep.append(embed_rep[j])
        tmp_acc.append(acc_lt[j])
    
    tmp_rep = np.array(tmp_rep)
    tmp_acc = np.array(tmp_acc)

    acc = np.mean(tmp_acc)
    if acc < vmin:
        vmin = acc
    if acc > vmax:
        vmax = acc
    index_acc_lt.append((i, acc))

index_acc_lt.sort(key=lambda x:x[1], reverse=False)

print(index_acc_lt)

print(f'vmin:{vmin}')

colormap = plt.cm.jet
normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
marker_lt = ['*','h','H','+','x',
            '.','o','v','^','<','>','P','p', 
            '4',           
            '1','2','3','8',
            '*','h','H','+','x',
            'X','D','d', '|','_']

cluster_acc_lt = []
cluster_size_lt = []

plt.figure(figsize=(5,5))
for i in range(equivalence_class_num):
    cluster_i = index_acc_lt[i][0]
    cluster = cluster_lt[cluster_i]
    tmp_rep = []
    tmp_acc = []
    for j in cluster:
        tmp_rep.append(embed_rep[j])
        tmp_acc.append(acc_lt[j])
    
    tmp_rep = np.array(tmp_rep)
    tmp_acc = np.array(tmp_acc)
    acc = np.mean(tmp_acc)
    print(f'\ti:{i}, cluster_id: {cluster_i}, acc:{acc}')
    marker=marker_lt[i]
    plt.scatter(tmp_rep[:,0], tmp_rep[:,1], marker=marker, color=colormap(normalize(acc)), label=f'EC-{i}, {acc:.3f}')
    
    cluster_acc_lt.append(acc)
    cluster_size_lt.append(len(cluster)*1.0)

# ground truth

plt.scatter(embed_rep[-1][0], embed_rep[-1][1], marker='s', color=colormap(normalize(vmax)), label=f'true labels')

plt.legend(bbox_to_anchor=(1.0, 1.0),loc='upper left')

handle_colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalize, cmap=colormap), orientation='horizontal', pad=0.08)
handle_colorbar.mappable.set_norm(normalize)
plt.tight_layout()
plt.savefig(f'EC-vis-heatmap.pdf')

cluster_acc_lt = np.array(cluster_acc_lt)
cluster_size_lt = np.array(cluster_size_lt)
cluster_size_lt = cluster_size_lt/np.sum(cluster_size_lt)

plt.close()

plt.figure(figsize=(5,2.5))
for i in range(equivalence_class_num):
    plt.scatter(cluster_acc_lt[i], cluster_size_lt[i], marker=marker_lt[i], color=colormap(normalize(cluster_acc_lt[i])))
plt.xlabel('EC average acc')
plt.ylabel('EC distribution')
plt.tight_layout()
plt.savefig(f'acc_prob.pdf')
