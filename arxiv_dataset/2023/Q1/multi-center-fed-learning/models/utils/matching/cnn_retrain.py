import numpy as np
import copy
import tensorflow as tf

from baseline_constants import KERNAL_WIDTH, KERNEL_HEIGHT, NUM_INPUT_CHANNEL, NUM_OUTPUT_CHANNEL
from utils.matching.cnn_permu import block_patching

def patch_weights(w_j, L_next, assignment_j_c):
    if assignment_j_c is None:
        return w_j
    new_w_j = np.zeros((w_j.shape[0], L_next))
    new_w_j[:, assignment_j_c] = w_j
    indices = [x for x in range(L_next) if x in assignment_j_c]
    return new_w_j[:, indices]


def reconstruct_weights(weight, assignment, model_summary, old_data, layer_identifier=None, slice_dim="filter"):
    res_weights = []
    conv_varname, dense_varname, weight_varname = "conv", "dense", "kernel"
#     print("current layer name ", layer_identifier)
    for var_name, o, value in zip(model_summary, old_data, weight):
#         print("processing {} old shape is {}, new shape is {}".format(var_name, o.shape, value.shape))
        if var_name.startswith(conv_varname):
            if var_name.endswith(weight_varname):
                w = value.transpose()
                if var_name != layer_identifier:
                    w = w.reshape(o.shape)
#                     #w = w.reshape(o.shape)                    
            else:
                w = value
        elif var_name.startswith("batch"):
            w = np.ones(o.shape)
        elif var_name.startswith(dense_varname):
            if var_name.endswith(weight_varname):
                if var_name != layer_identifier: 
                    w = value.transpose()
                else:
                    w = value
            else:
                w = value
        res_weights.append(w)       

    # slice after matched layer's weight from global 
    # (input weight is shared across all nodes, due to
    # the weight is 'global weight')
    #  when we have assignment information
    # what holds in assignment is the identities of 
    # global neurons used by this client
    w_index = model_summary.index(layer_identifier)
    old = old_data[w_index]
    if layer_identifier.startswith(conv_varname) and \
        layer_identifier.endswith(weight_varname):
#         print("Branch executing or not?", w_index)
        if slice_dim == "filter":
            _maw = res_weights[w_index][:, assignment]
            width, height = old.shape[KERNAL_WIDTH], old.shape[KERNEL_HEIGHT]
            num_in_chn, num_out_chn = old.shape[NUM_INPUT_CHANNEL], old.shape[NUM_OUTPUT_CHANNEL]
            _maw = _maw.reshape(width, height, num_in_chn, num_out_chn)            
            res_weights[w_index] = _maw
            res_weights[w_index + 1] = res_weights[w_index + 1][assignment]
        elif slice_dim == "channel":
            pass            
    elif layer_identifier.startswith(dense_varname):
        res_weights[w_index] = res_weights[w_index][:, assignment]
        res_weights[w_index + 1] = res_weights[w_index + 1][assignment]

    return res_weights

def combine_network_after_matching(batch_weights, layer_index, model_summary, model_meta_data,
                                   matched_weight, L_next, assignment, matching_shapes, out_estimator):
    
    def apply_combine(weight, rng):
        return [weight[i] for i in rng]
    
    def apply_addmatch(worker):
        m = copy.deepcopy(matched_weight)
        rng_before_matched = range(2 * layer_index -2) 
        combined = apply_combine(worker, rng_before_matched)
        return combined + m
   
    def apply_remain(worker):       
        rng_remain = range(2 * (layer_index + 1) - 1, len(model_summary)) 
        return apply_combine(worker, rng_remain)
    
    type_of_patched_layer = model_summary[2 * (layer_index + 1) - 2]
    if type_of_patched_layer.startswith("conv"):
        l_type = "conv"
    elif type_of_patched_layer.startswith("dense"):
        l_type = "fc"

    type_of_this_layer = model_summary[2 * layer_index - 2]        
    if (type_of_this_layer == "dense/kernel"):
        fc_pos = layer_index
    else:
        fc_pos = None

    first_worker = True
    fc_outshape = tuple()       
    temp_whole_network = list(map(apply_addmatch, batch_weights))
    for j, worker in enumerate(batch_weights):
        if fc_pos is None:
            if l_type == "conv":
                patched_weight = block_patching(worker[2 * (layer_index + 1) - 2], 
                                    L_next, assignment[j], 
                                    layer_index+1, model_meta_data,
                                    fc_outshape, layer_type=l_type)
            elif l_type == "fc":
                if first_worker:
                    # we just need to compute this once for each of global updated
                    first_worker = False
                    fc_outshape = out_estimator(matching_shapes)
                patched_weight = block_patching(worker[2 * (layer_index + 1) - 2], 
                                    L_next, assignment[j], 
                                    layer_index+1, model_meta_data,
                                    fc_outshape, layer_type=l_type)

        elif layer_index >= fc_pos:
            patched_weight = patch_weights(worker[2 * (layer_index + 1) - 2], L_next, assignment[j])

        temp_whole_network[j].append(patched_weight)

    remain_whole_network = list(map(apply_remain, batch_weights))
    all_combined = [bef + after for bef, after in zip(temp_whole_network, remain_whole_network) ]    
    return all_combined
        

def local_train(clients, whole_network, layer_index, config):
   
    epochs = config["epochs"]
    batch_size = config["batch-size"]
    model_summary = clients[0].model.get_summary()
    layer_identifier = model_summary[layer_index]
    trainables = [i for i in range(layer_index + 2, len(model_summary))]

    # handle the conv layers part which is not changing
    rem_collection = []
    with clients[0].model.graph.as_default():
        trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        all_vars = tf.global_variables()
        
    rem_collection = []
    for seq in reversed(range(len(model_summary))):
        if (seq not in trainables):
            rem_collection.append(trainable_collection[seq])
            
    # i) we need to set trainable variable collections from which those
    # frozen variable can be removed
    for v in rem_collection:
        trainable_collection.remove(v)
    retrained_models = []
    for seq, j in enumerate(clients):
        # ii) we set current matched layers to weights of global, keep the rest untouched
        j.model.set_params(whole_network[seq])
        # iii) we train all locals
        j.model.train(j.train_data, num_epochs=epochs, batch_size=batch_size)
        retrained_models.append(j.model.get_params())
        
    with clients[0].model.graph.as_default():
        collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        for v in rem_collection:
            collection.insert(0, v)

    return retrained_models