import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
import numpy as np
import pandas as pd

from functools import reduce
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from tfkerassurgeon import Surgeon
from sklearn.metrics.pairwise import cosine_similarity


from glob import glob


def get_class_weights(class_names, path_dir, file_ext=".jpg"):
    counts = []
    for class_name, index in class_names.items():
        num_examples = len(glob(os.path.join(path_dir, class_name, f"*{file_ext}")))
        counts.append(num_examples)

    counts = dict(enumerate([sum(counts) / val for val in counts]))
    return counts


def my_get_all_conv_layers(model, filter_thresh=128):
    """
    Arguments:
        model -> your model
    Return:
        List of Indices containing convolution layers
    """

    all_conv_layers = list()
    for i, each_layer in enumerate(model.layers):
        if isinstance(each_layer, layers.Conv2D):
            if each_layer.filters > filter_thresh:
                all_conv_layers.append(i)
    return all_conv_layers


def my_get_all_dense_layers(model):
    """
    Arguments:
        model -> your model
    Return:
        List of Indices containing fully connected layers
    """
    all_dense_layers = list()
    for i, each_layer in enumerate(model.layers):
        # if (each_layer.name[0:5] == 'dense'):
        if isinstance(each_layer, layers.Dense):
            all_dense_layers.append(i)
    return all_dense_layers


def count_conv_params_flops(conv_layer):
    """
    Arguments:
        conv layer
    Return:
        Number of Parameters, Number of Flops
    """
    n_conv_params_total = conv_layer.count_params()
    in_shape = K.int_shape(conv_layer.get_input_at(0))
    out_shape = conv_layer.output_shape
    conv_flops = reduce(lambda x, y: x * y, in_shape[1:] + out_shape[1:])
    return n_conv_params_total, conv_flops


def count_dense_params_flops(dense_layer):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    """
    Arguments:
      dense layer
    Return:
        Number of Parameters, Number of Flops
    """
    n_dense_params_total = dense_layer.count_params()
    in_shape = K.int_shape(dense_layer.get_input_at(0))
    out_shape = dense_layer.output_shape
    dense_flops = reduce(lambda x, y: x * y, in_shape[1:] + out_shape[1:])
    return n_dense_params_total, dense_flops


def count_model_params_flops(model, layers=None):
    """
    Arguments:
        model -> your model
    Return:
        Number of parmaters, Number of Flops
    """
    total_params = 0
    total_flops = 0

    model_layers = [model.layers[layer] for layer in layers] if layers else model.layers

    for _, layer in enumerate(model_layers):
        if any(
            conv_type in str(type(layer))
            for conv_type in ["Conv1D", "Conv2D", "Conv3D"]
        ):

            params, flops = count_conv_params_flops(layer)
            total_params += params
            total_flops += flops
        elif "Dense" in str(type(layer)):
            params, flops = count_dense_params_flops(layer)
            total_params += params
            total_flops += flops
    return total_params, int(total_flops)


def my_get_weights_in_conv_layers(model):
    """
    Arguments:
        model -> your model
    Return:
        List containing weight tensors of each layer
    """
    weights = list()
    all_conv_layers = my_get_all_conv_layers(model)
    for i in all_conv_layers:
        weights.append(model.layers[i].get_weights()[0])
    return weights


def my_get_l1_norms_filters_per_epoch(weight_list_per_epoch):
    """
    Arguments:
        List
    Return:
        Number of parmaters, Number of Flops
    """
    l1_norms_filters_per_epoch = list()
    flattened_filters_per_epoch = list()

    for index in range(len(weight_list_per_epoch)):
        epochs = np.array(weight_list_per_epoch[index]).shape[0]
        h, w, d = (
            np.array(weight_list_per_epoch[index]).shape[1],
            np.array(weight_list_per_epoch[index]).shape[2],
            np.array(weight_list_per_epoch[index]).shape[3],
        )
        l1_norms_filters_per_epoch.append(
            np.sum(
                np.abs(np.array(weight_list_per_epoch[index])).reshape(
                    epochs, h * w * d, -1
                ),
                axis=1,
            )
        )
        flattened_filters_per_epoch.append(
            np.array(weight_list_per_epoch[index]).reshape(epochs, h * w * d, -1)
        )

    return l1_norms_filters_per_epoch, flattened_filters_per_epoch


def my_get_distance_matrix(flat_vects_matrix):
    """
    Arguments:
        l1_norm_matrix:matrix that stores the l1 norms of filters
    Return:
        distance_matrix: matrix that stores the manhattan distance between filters
    """
    flat_vects_matrix = np.reshape(flat_vects_matrix, (flat_vects_matrix.shape[0], -1))
    distance_matrix = cosine_similarity(flat_vects_matrix, flat_vects_matrix)
    return distance_matrix


def my_get_distance_matrix_list(flat_vects_list):
    """
    Arguments:
        l1_norm_matrix_list:
    Return:
        distance_matrix_list:
    """
    distance_matrix_list = []
    for flat_vects_matrix in flat_vects_list:
        distance_matrix_list.append(
            my_get_distance_matrix(flat_vects_matrix.transpose(2, 0, 1))
        )
    return distance_matrix_list


def my_get_episodes(distance_matrix, percentage):
    """
    Arguments:
        distance_matrix:
        percentage:Percentage of filters to be pruned
    Return:
    episodes:list of filter indices
    """
    distance_matrix_flatten = pd.Series(distance_matrix.flatten())
    distance_matrix_flatten = distance_matrix_flatten.sort_values().index.to_list()

    episodes = list()
    n = distance_matrix.shape[0]
    for i in distance_matrix_flatten:
        episodes.append((i // n, i % n))
    k = int((n * percentage) / 100)
    li = list()
    for i in range(2 * k):
        if i % 2 != 0:
            li.append(episodes[n + i])
    return li


def normalize(X_train, X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    return X_train, X_test


def my_get_episodes_for_all_layers(distance_matrix_list, percentage):
    """
    Arguments:
        distance_matrix_list:matrix containing the manhattan distance of all layers
        percentage:percentage of filters to be pruned
    Return:
        all_episodes:all the selected filter pairs
    """
    all_episodes = list()
    for matrix in distance_matrix_list:
        all_episodes.append(my_get_episodes(matrix, percentage))
    return all_episodes


def my_get_filter_pruning_indices(
    episodes_for_all_layers, l1_norms_list, model, weighted_avg=False, model_type=None
):
    """
    Arguments:
        episodes_for_all_layers:list of selected filter pairs
        l1_norm_matrix_list:list of l1 norm matrices of all the filters of each layer
    Return:
        filter_pruning_indices:list of indices of filters to be pruned
    """
    all_conv_layers = my_get_all_conv_layers(model, True)
    filter_pruning_indices = list()
    for layer_index, episode_layer in enumerate(episodes_for_all_layers):
        a = set()
        sum_l1_norms = np.sum(l1_norms_list[layer_index], axis=0, keepdims=True)

        # set of filter indices
        filter_set = set(sum([list(i) for i in episode_layer], []))

        for episode in episode_layer:
            ep1 = sum_l1_norms.T[episode[0]]
            ep2 = sum_l1_norms.T[episode[1]]
            if ep1 <= ep2:
                # checking if pruning a resnet model because of dimension consistency requirement of Add layer
                if (model_type == "resnet") and (episode[0] in a):
                    a.add(episode[1])
                a.add(episode[0])
                greater = episode[1]
            else:
                if (model_type == "resnet") and (episode[1] in a):
                    a.add(episode[0])
                a.add(episode[1])
                greater = episode[0]

            if weighted_avg:
                try:
                    weights, bias = model.layers[
                        all_conv_layers[layer_index]
                    ].get_weights()
                    f1 = weights[:, :, :, episode[0]]
                    f2 = weights[:, :, :, episode[1]]
                    coeff = lambda ep: 1 - (ep / (ep1 + ep2))
                    avg = coeff(ep1) * f1 + coeff(ep2) * f2
                    weights[:, :, :, greater] = avg
                    model.layers[all_conv_layers[layer_index]].set_weights(
                        [weights, bias]
                    )
                except:
                    weights = model.layers[all_conv_layers[layer_index]].get_weights()[
                        0
                    ]
                    f1 = weights[:, :, :, episode[0]]
                    f2 = weights[:, :, :, episode[1]]
                    coeff = lambda ep: 1 - (ep / (ep1 + ep2))
                    avg = coeff(ep1) * f1 + coeff(ep2) * f2
                    weights[:, :, :, greater] = avg
                    model.layers[all_conv_layers[layer_index]].set_weights([weights])

        if (model_type == "resnet") and (len(a) != len(episode_layer)):
            remaining_elements = len(episode_layer) - len(a)
            random_samples = random.sample(list(filter_set - a), remaining_elements)
            a = a.union(set(random_samples))
        a = list(a)
        filter_pruning_indices.append(a)
    return filter_pruning_indices, model


def my_delete_filters(
    model, weight_list_per_epoch, percentage, weighted_avg=False, model_type=None
):
    """
    Arguments:
        model:CNN Model
        wieight_list_per_epoch:History
        percentage:Percentage to be pruned
    Return:
        model_new:input model after pruning
    """
    l1_norms, flattened_filters = my_get_l1_norms_filters_per_epoch(
        weight_list_per_epoch
    )
    distance_matrix_list = my_get_distance_matrix_list(flattened_filters)
    episodes_for_all_layers = my_get_episodes_for_all_layers(
        distance_matrix_list, percentage
    )
    filter_pruning_indices, model = my_get_filter_pruning_indices(
        episodes_for_all_layers,
        l1_norms,
        model,
        weighted_avg=weighted_avg,
        model_type=model_type,
    )
    all_conv_layers = my_get_all_conv_layers(model)

    surgeon = Surgeon(model)
    for index, value in enumerate(all_conv_layers):
        # print(index,value,filter_pruning_indices[index])

        surgeon.add_job(
            "delete_channels",
            model.layers[value],
            channels=filter_pruning_indices[index],
        )

    model_new = surgeon.operate()

    for i in range(len(model_new.layers)):
        if isinstance(model_new.layers[i], layers.Conv2D):
            model_new.layers[i].trainable = False

    # all conv layers but the last layer
    for i in all_conv_layers:
        model_new.layers[i].trainable = True

    return model_new


class Get_Weights(Callback):
    def __init__(self, num_layers=1):
        super(Get_Weights, self).__init__()
        self.weight_list = []  # Using a list of list to store weight tensors per epoch
        self.num_layers = num_layers

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            all_conv_layers = my_get_all_conv_layers(self.model)
            for _ in range(len(all_conv_layers)):
                self.weight_list.append(
                    []
                )  # appending empty lists for later appending weight tensors
        weights_in_conv_layers = my_get_weights_in_conv_layers(self.model)
        for index, each_weight in enumerate(weights_in_conv_layers):
            self.weight_list[index].append(each_weight)


def my_get_l1_norms_filters(model):
    """
    Arguments:
        model:initial model
        weight_list_per_epoch:weight tensors at every epoch
        percentage:percentage of filter to be pruned
    Return:
        regularizer_value
    """

    conv_layers = my_get_all_conv_layers(model)
    l1_norms = list()
    for index, layer_index in enumerate(conv_layers):
        l1_norms.append([])
        weights = model.layers[layer_index].get_weights()[0]
        num_filters = len(weights[0, 0, 0, :])
        for i in range(num_filters):
            weights_sum = np.sum(abs(weights[:, :, :, i]))
            l1_norms[index].append(weights_sum)
    return l1_norms


def my_get_regularizer_value(model, weight_list_per_epoch, percentage):
    """
    Arguments:
        model:initial model
        weight_list_per_epoch:weight tensors at every epoch
        percentage:percentage of filter to be pruned
    Return:
        regularizer_value
    """
    l1_norms_per_epoch, flattened_filters_per_epoch = my_get_l1_norms_filters_per_epoch(
        weight_list_per_epoch
    )
    distance_matrix_list = my_get_distance_matrix_list(flattened_filters_per_epoch)
    episodes_for_all_layers = my_get_episodes_for_all_layers(
        distance_matrix_list, percentage
    )
    l1_norms = my_get_l1_norms_filters(model)
    regularizer_value = 0
    for layer_index, layer in enumerate(episodes_for_all_layers):
        for episode in layer:
            regularizer_value += abs(
                l1_norms[layer_index][episode[1]] - l1_norms[layer_index][episode[0]]
            )
    regularizer_value = np.exp(-1 * (regularizer_value))
    return regularizer_value


def custom_loss(lmbda, regularizer_value):
    def loss(y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred) + lmbda * regularizer_value

    return loss
