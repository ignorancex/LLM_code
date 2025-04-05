import pickle
import random
import numpy as np

def load_counts(dataset):
    fname = "{}_counts".format(dataset)
    with open(fname, 'rb') as f:
        loaded_cls_counts = pickle.load(f) 
    return loaded_cls_counts

def avg_cls_weights(dataset, num_classes):
    all_class_freq = load_counts(dataset)
    J = 40
    keys = list(all_class_freq)
    random.shuffle(keys)

    averaging_weights = np.zeros((J, num_classes), dtype=np.float32)
    for i in range(num_classes):
        total_num_counts = 0
        worker_class_counts = [0] * J
        for j in range(J):
            w = keys[j]
            if i in all_class_freq[w].keys():
                total_num_counts += all_class_freq[w][i]
                worker_class_counts[j] = all_class_freq[w][i]
            else:
                total_num_counts += 0
                worker_class_counts[j] = 0
        averaging_weights[:, i] = worker_class_counts / total_num_counts

    return averaging_weights, all_class_freq
    
