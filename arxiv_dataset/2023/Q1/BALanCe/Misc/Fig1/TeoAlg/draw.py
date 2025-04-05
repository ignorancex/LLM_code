import numpy as np

acc_lt = np.loadtxt('acc_lt_226.txt', delimiter=',')

fin = open('epsilon_net.txt', 'r')

cluster_node_lt = []
head_lt = []
while True:
    line = fin.readline().strip()
    if not line:
        break
    line = line[0:-1]
    array = line.split(':')
    head = array[0]
    rest = array[1]
    array = rest.split(',')
    cluster = []
    for node in array:
        node_int = int(node)
        cluster.append(node_int)
    cluster_node_lt.append(cluster)

    head_lt.append(int(head))

embed_rep = np.loadtxt('tsne.rep')


import numpy as np
from scipy.spatial import distance
import random
import copy

equivalence_class_num = 10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#plt.figure(figsize=(8,8))

index_lt = [i for i in range(equivalence_class_num)]

#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#pca_rep = pca.fit_transform(hamming_rep)

random.shuffle(index_lt)

cluster_lt = cluster_node_lt

# find min value
vmin = 1.0
vmax = 0.0
for i in range(equivalence_class_num):
    cluster_i = index_lt[i]
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

print(f'vmin:{vmin}')

colormap = plt.cm.bwr
normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
marker_lt = ['.',',','o','v','^',
            '<','>','1','2','3',
            '4','8','p','P',
            '*','h','H','+','x',
            'X','D','d', '|','_']
#random.shuffle(marker_lt)

plt.figure(figsize=(8,8))
for i in range(equivalence_class_num):
    #cluster_i = index_lt[i]
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
    marker=marker_lt[i]
    plt.scatter(tmp_rep[:,0], tmp_rep[:,1], marker=marker, color=colormap(normalize(acc)), label=f'EC-{i}, avg acc:{acc:.3f}')
    
# ground truth

plt.scatter(embed_rep[-1][0], embed_rep[-1][1], marker='s', color=colormap(normalize(vmax)), label=f'Hamming dataset labels')

plt.legend(bbox_to_anchor=(1.05, 1.0),loc='upper left')

'''
handle_colorbar = plt.colorbar(orientation='horizontal')
handle_colorbar.mappable.set_norm(normalize)
'''
plt.colorbar(plt.cm.ScalarMappable(norm=normalize, cmap=colormap), orientation='horizontal')
plt.tight_layout()
plt.savefig(f'EC-vis-heatmap.pdf')

