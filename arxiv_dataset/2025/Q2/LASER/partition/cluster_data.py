import json
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import random
import os
import argparse


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def save_json(data, file_path):
    with open(file_path, 'w') as w:
        json.dump(data, w)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def random_data_cluster(dataset_name, n_clusters):
    user_info = np.load(f'checkpoint/lightgcn/lightgcn-{dataset_name}-2023_user.npy')
    print(user_info.shape)
    user_num = user_info.shape[0]

    datamap_path = f'../data/{dataset_name}/proc_data/datamaps.json'
    datamap = load_json(datamap_path)
    id2user = datamap['id2user']
    id2user['0'] = 0
    # values = list(id2user.values())
    # print('in datamap', 'A37U0KM64LZSTE' in values)
    # print(user_num, len(id2user))
    # user2id = {v:k for k, v in id2user.items()}
    # print('key: A37U0KM64LZSTE', 'value:', user2id['A37U0KM64LZSTE'])

    user_list = list(range(user_num))
    random.shuffle(user_list)
    avg = user_num // n_clusters
    remainder = user_num % n_clusters

    sublists = [user_list[i * avg + min(i, remainder):(i + 1) * avg + min(i + 1, remainder)]
                for i in range(n_clusters)]

    label_clf = {id2user[str(i)]: j for j, sublist in enumerate(sublists) for i in sublist}
    # print(label_clf)

    # print('in cluster', 'A37U0KM64LZSTE' in label_clf)
    # exit()

    save_json(label_clf, f'checkpoint/lightgcn/user_class_{n_clusters}_random_{dataset_name}.json')


def data_cluster(dataset_name, n_clusters):

    user_info = np.load(f'checkpoint/lightgcn/lightgcn-{dataset_name}-2023_user.npy')
    print(user_info.shape)

    clf = KMeans(n_clusters=n_clusters)
    ydata = clf.fit_predict(user_info)
    label_clf = clf.labels_
    print(label_clf[:10])

    counter = Counter(label_clf)
    print(counter.items())

    # print(label_clf)
    datamap_path = f'../data/{dataset_name}/proc_data/datamaps.json'
    datamap = load_json(datamap_path)
    id2user = datamap['id2user']
    # print(id2user)

    id2user['0'] = 0
    label_clf = {id2user[str(i)]: int(label) for i, label in enumerate(label_clf)}
    # print(max(label_clf.keys()))

    save_json(label_clf, f'checkpoint/lightgcn/user_class_{n_clusters}_{dataset_name}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amz-new', help='Dataset name')
    parser.add_argument('--n_clusters', type=int, default=3, help='number of clusters')
    args, _ = parser.parse_known_args()
    n_clusters = args.n_clusters
    dataset_name = args.dataset
    data_cluster(dataset_name, n_clusters)
    random_data_cluster(dataset_name, n_clusters)
    # for n_clusters in [3, 4, 5, 6, 7, 8, 9, 20]:
    # #     print('cluster num', n_clusters)
    #     data_cluster(dataset_name, n_clusters)
    #     random_data_cluster(dataset_name, n_clusters)
