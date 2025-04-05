import os
import numpy as np

"""Utils for language models."""

import re
import json
from collections import defaultdict
import pickle

def get_flatten_vec(net, fc_idx):
    d = np.array([])
    layer_range = range(fc_idx, len(net))
    for l in layer_range:
        d = np.concatenate((d, net[l].flatten()), axis=0)
    return d

def get_first_den_indx(layers):
    idx = None
    for i, layer in enumerate(layers):
        if layer.name.startswith("dense"):
            idx = i
            break
    return idx -1

def agg_neurons(assignment_c, ith_glob_neuron, weight, bias):
    num_batch = len(assignment_c)
    neuron_indices = [assgid for assgid in range(num_batch) if ith_glob_neuron in assignment_c[assgid]]
    #print("id_of_global_neuron %d, and assignment %s" % (ith_glob_neuron, neuron_indices))
    cur_weight = []
    cur_bias = []
    for j in neuron_indices:
        cur_weight.append(weight[j][assignment_c[j].index(ith_glob_neuron), :])
        cur_bias.append(bias[j][assignment_c[j].index(ith_glob_neuron)])
        
    avg_w = np.sum(cur_weight, axis=0) / len(neuron_indices) 
    avg_b = sum(cur_bias) / len(neuron_indices)
    
    return avg_w, avg_b

def extend_weight_neurons(prev_weight, weight, l_next, l_prev):
    # this function execute only when 
    # next layer has differ number of global neurons than 
    # client NN next layer
    diff = abs(l_next - l_prev)
    comp = np.arange(l_next)[-diff:]
    padded_w = [prev_weight[i] for i in comp ]
    mean = np.mean(padded_w, axis=1)
    new_weight = np.hstack((weight, mean))
    return new_weight

def get_all_L_next(global_weight_model):
    num_layer = len(global_weight_model)
    L_next = []
    for layer in range(1, num_layer, 2):
        L_next.append(global_weight_model[layer].shape[0])
    return L_next

def agg_layers(batch_weights, global_weight_model, assignment):
    num_worker = len(batch_weights)
    num_layer = len(batch_weights[0])
    
    num_glob_neurons = get_all_L_next(global_weight_model)
    
    manual_global = [None] * num_layer
    # first layer weight and bias we just copy the it
    manual_global[0] = global_weight_model[0]
    
    # we do aggregate in a top-down approach
    for layer in range(2, round(num_layer), 2)[::-1]:
        lay_w = [batch_weights[j][layer] for j in range(num_worker)]
        lay_b = [batch_weights[j][layer -1] for j in range(num_worker)]
        after_w = np.zeros(global_weight_model[layer].shape)
        after_b = np.zeros(global_weight_model[layer -1].shape)
        for k in range(num_glob_neurons[round(layer/2) -1]):
            weight, bias = agg_neurons(assignment[round(layer/2) -1], k, lay_w, lay_b)
            if len(weight) < after_w.shape[1]:
                weight = extend_weight_neurons(prev_weight, weight, after_w.shape[1], len(weight))
                after_w[k, :] = weight
                after_b[k] = bias
            else:
                after_w[k, :] = weight
                after_b[k] = bias
        prev_weight = after_w
        manual_global[layer] = after_w
        manual_global[layer-1] = after_b
    
    manual_global[num_layer -1] = global_weight_model[num_layer-1]
    return manual_global

def reverse_layers(map_out, original_model):
    num_layer = len(map_out)
    temp_shape = [v.shape for v in original_model.get_weights()]
    book_keeping = []
    for layer in range(num_layer):
        # we only want to set bias, dense weight, not cnn
        if len(temp_shape[layer]) in ([2,1]):
            if len(temp_shape[layer]) == 2:
                # weight layer
                x = np.array(map_out[layer])
                t_w = x[-temp_shape[layer][0]:, -temp_shape[layer][1]:]
                book_keeping.append(t_w)
            else:
                x = np.array(map_out[layer])
                t_b = x[-temp_shape[layer][0]:]
                book_keeping.append(t_b)
        else:
            book_keeping.append(map_out[layer])
    return book_keeping
    
def reverse_matched_2_original_weights(global_weight_set, num_clusters, original_model):
    original_weights = [None] * num_clusters
    for c in range(num_clusters):
         original_weights[c] = reverse_layers(global_weight_set[c], original_model)
    return original_weights
'''
In case of the first layer, I cannot just merge them
so I decide to just use the first [orig_shape] of neurons
to send to clients
'''

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data

def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def has_vec_not_init(vec):
    NoneType = type(None)
    return type(vec) == NoneType

# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


# ------------------------
# utils for sent140 dataset


def split_line(line):
    '''split given line/phrase into list of words

    Args:
        line: string representing phrase to be split
    
    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    '''returns index of given word based on given lookup dictionary

    returns the length of the lookup dictionary if word not found

    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    '''
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices
    
    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer 
    representing unknown index to returned list until the list's length is 
    max_words

    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list

    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line) # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab

    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values

    Return:
        integer list
    '''
    bag = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def get_word_emb_arr(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab


def val_to_vec(size, val):
    """Converts target into one-hot.

    Args:
        size: Size of vector.
        val: Integer in range [0, size].
    Returns:
         vec: one-hot vector with a 1 in the val element.
    """
    assert 0 <= val < size
    vec = [0 for _ in range(size)]
    vec[int(val)] = 1
    return vec
