import os
import re
import pickle
import numpy as np

from baseline_constants import KERNAL_WIDTH, KERNEL_HEIGHT, NUM_INPUT_CHANNEL, NUM_OUTPUT_CHANNEL

# def avg_cls_weights(batch, dataset, num_classes):
#     all_class_freq = load_counts(dataset)
#     J = len(batch)
#     wids = [ c.id for c in batch ]
#     wcnt = [ i for i in range(len(batch))]
#     ret_class_freq = {c.id: all_class_freq[c.id] for c in batch}

#     averaging_weights = np.zeros((J, num_classes), dtype=np.float32)
#     for i in range(num_classes):
#         total_num_counts = 0
#         worker_class_counts = [0] * J
#         for j, c in zip(wids, wcnt):
#             if i in all_class_freq[j].keys():
#                 total_num_counts += all_class_freq[j][i]
#                 worker_class_counts[c] = all_class_freq[j][i]
#             else:
#                 total_num_counts += 0
#                 worker_class_counts[c] = 0
#         denorm = max(1, total_num_counts)
#         if (total_num_counts == 0):
#             print("Batch clients: {}, classes: {}".format(wids, i))
#         averaging_weights[:, i] = np.array(worker_class_counts) / denorm

#     return averaging_weights, ret_class_freq

# def saved_cls_counts(clients, file):
#     net_cls_counts = {}

#     for c in clients:
#         unq, unq_cnt = np.unique(c.train_data['y'], return_counts=True)
#         tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
#         net_cls_counts[c.id] = tmp
        
#     with open(file, 'wb+') as f:
#         pickle.dump(net_cls_counts, f)

# def load_counts(dataset):
#     fname = "{}_counts".format(dataset)
#     with open(fname, 'rb') as f:
#         loaded_cls_counts = pickle.load(f) 
#     return loaded_cls_counts

def pdm_prepare_freq(cls_freqs, n_classes=10):
    freqs = []
    for net_i in sorted(cls_freqs.keys()):
        net_freqs = [0] * n_classes

        for cls_i in cls_freqs[net_i]:
            net_freqs[cls_i] = cls_freqs[net_i][cls_i]

        freqs.append(np.array(net_freqs))

    return freqs


def load_files():
    def get_weightfile():
        sepath = os.path.join(".", "workernn")
        j_files =  [f for f in os.listdir(sepath) if re.match("femnist_.{5}_.{2}\.pb", f)]
        return j_files

    def process_file(f):
        fname = os.path.join("workernn", f)
        with open(fname, 'rb') as file:
            w = pickle.load(file)        
        return w 
    
    return list(map(process_file, get_weightfile()))

def get_cnn_w(value):
    o_shape = value.shape
    width, height = o_shape[KERNAL_WIDTH], o_shape[KERNEL_HEIGHT]
    num_in_chn, num_out_chn = o_shape[NUM_INPUT_CHANNEL], o_shape[NUM_OUTPUT_CHANNEL]
    n_shape = (width * height * num_in_chn, num_out_chn)    
    w = value.reshape(n_shape)
    # IMPORTANAT, here need to invoke transpose , because
    # in orignal paper, they use pytorch, and the order in 
    # pytorch is different from tensorflow
    # the order is (NUM_OUTPUT_CHANNEL, NUM_INPUT_CHANNEL, ker_width, ker_height)
    w = w.transpose() 
    return w
   
def load_local_model_weight_func(model, model_summary):
    all_layers = []
    dense_varname, weight_varname = "dense", "kernel"
    for var_name, value in zip(model_summary, model):
        if var_name.startswith("conv"):
            if var_name.endswith(weight_varname):
                all_layers.append(get_cnn_w(value))
            else:
                all_layers.append(value)
        elif var_name.startswith("batch"):
            pass
        elif var_name.startswith(dense_varname):
            if var_name.endswith(weight_varname):
                all_layers.append(value.transpose())
            else:
                all_layers.append(value)
    return all_layers
