import os

from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.datasets import load_unit_test
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from tslearn.datasets import CachedDatasets, UCR_UEA_datasets
import matplotlib.pyplot as plt
from utils import read_UCR_UEA, pickle_save_to_file
from utils.constants import tasks, tasks_new
from sklearn.metrics import mutual_info_score
from collections import defaultdict
tasks = tasks + tasks_new

STC_accs = []


# getThreshold
def calculate_entropy(y):
    # 计算目标变量 y 的熵
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # 加一个小值防止log(0)
    return entropy


def information_gain(y, split_condition):
    # 计算信息增益
    left_split = y[split_condition]
    right_split = y[~split_condition]

    if len(left_split) == 0 or len(right_split) == 0:  # 如果其中一侧为空，跳过此分割点
        return -np.inf

    left_entropy = calculate_entropy(left_split)
    right_entropy = calculate_entropy(right_split)

    weighted_entropy = (len(left_split) * left_entropy + len(right_split) * right_entropy) / len(y)
    return calculate_entropy(y) - weighted_entropy


def find_best_threshold(X, y):
    num_instances, num_features = X.shape
    best_thresholds = {}

    for feature_idx in range(num_features):
        feature_values = X[:, feature_idx]
        sorted_values = np.sort(np.unique(feature_values))  # 对特征值进行排序
        best_gain = -np.inf
        best_threshold = None

        # 遍历每个可能的分割点（相邻值的中点）
        for i in range(1, len(sorted_values)):
            threshold = (sorted_values[i - 1] + sorted_values[i]) / 2

            # 根据当前阈值分割数据
            left_mask = feature_values <= threshold
            right_mask = feature_values > threshold

            # 计算信息增益
            gain = information_gain(y, left_mask)

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

        # 将最佳阈值记录下来
        best_thresholds[feature_idx] = best_threshold

    return best_thresholds

def z_normalize(ts):
    mean = np.mean(ts)
    std = np.std(ts)
    return (ts - mean) / std if std > 0 else ts - mean
def match_shapelet_location(instance, shapelet):
    l = len(shapelet)
    min_dist = float('inf')
    best_loc = -1
    s_z = z_normalize(shapelet)
    min_dist = float('inf')

    for i in range(len(instance) - l + 1):
        window = instance[i:i + l]
        w_z = z_normalize(window)
        dist = np.linalg.norm(w_z - s_z)
        if dist < min_dist:
            min_dist = dist
            best_loc = i
    return best_loc, min_dist

for task in tasks:
    save_dir = f'./output/ST/{task}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    _, test_x, _, test_y, _ = read_UCR_UEA(task, None)
    test_y = np.argmax(test_y, axis=1)
    clf = ShapeletTransformClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_shapelet_samples=100,
        max_shapelet_length=test_x.shape[-1] // 2,
        max_shapelets=10,
        batch_size=20,
        random_state=42,
    )
    clf.fit(test_x, test_y)

    pred_y = clf.predict(test_x)
    acc = accuracy_score(pred_y, test_y)
    STC_accs.append([task, acc])

    shapelets = clf._transformer.shapelets
    ST = clf._transformer
    test_x_transformed = ST.transform(test_x)
    # get thresholds
    best_thresholds = find_best_threshold(test_x_transformed, test_y)
    print(best_thresholds)

    existance_index = defaultdict(list)

    for i in range(len(shapelets)):
        distance_i = test_x_transformed[:, i].flatten()
        existance_index[i] = np.where(distance_i < best_thresholds[i])[0]

    shapelet_locs = []
    for index in existance_index.keys():  #
        print(f'-------Shapelet: {index}---------')
        shapelet_info = shapelets[index]
        normalized_shapelet = shapelet_info[-1]
        # (info_gain, length,start position,dimension, index of instance,class, The z-normalised shapelet array)
        #
        count = 0
        shapelet_length = shapelet_info[1]
        plt.plot(normalized_shapelet, label=f'Shapelet: {index}')
        for instance_index in existance_index[index]:
            best_loc, _ = match_shapelet_location(test_x[instance_index].flatten(), normalized_shapelet)
            # print(best_loc)
            shapelet_loc = [instance_index, None, None, None, best_loc, best_loc + shapelet_length]
            shapelet_locs.append(shapelet_loc)
            count += 1
            if count < 10:
                plt.plot(test_x[instance_index].flatten(), color='gray', lw=1, alpha=0.5)  # ,label=instance_index
                plt.plot(np.arange(best_loc, best_loc + shapelet_length),
                         test_x[instance_index].flatten()[best_loc: best_loc + shapelet_length], color='orange', lw=5,
                         alpha=0.8)
        plt.title(f'Shapelet: {index}')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'existance_{index}.png'), dpi=300)
        plt.close()
        # [i, inst[best_start:best_end + 1], attr[i, best_start:best_end + 1], max_score, best_start,best_end]
        # implet format [index of instance, instance subseq, attr subseq, max_score, best_start, best_end]



    pickle_save_to_file(data={"shapelet": shapelet_locs,
                              "existance_index": existance_index,
                              "best_thresholds": best_thresholds
                              },
                        file_path=os.path.join(save_dir, 'shapelet.pkl'))

df = pd.DataFrame(STC_accs, columns=['task', 'tacc'])
df.to_csv(os.path.join('./output/ST_acc.csv'))