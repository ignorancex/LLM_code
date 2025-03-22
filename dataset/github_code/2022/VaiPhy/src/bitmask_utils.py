import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from scipy.special import logsumexp


def get_total_partitions(bitmask_dict, n_taxa):
    root_mask = [1] * n_taxa
    bitmask_str = "".join(map(str, list(root_mask)))
    num_trees = bitmask_dict[bitmask_str]

    num_total = 0
    for key in bitmask_dict:
        num_total += bitmask_dict[key]
    return num_trees, num_total


def check_valid(bitstr, node_list, max_difference):
    node_vals = []
    for node_i in node_list:
        node_vals.append(int(bitstr[node_i]))
    is_valid = all(ele == node_vals[0] for ele in node_vals)

    if not is_valid:
        return False

    if max_difference is None:
        return is_valid
    else:
        num_others_same = len(bitstr.replace(str(1-int(node_vals[0])), "")) - len(node_list)
        if num_others_same > max_difference:
            #print("not valid. num_oth_same: ", num_others_same, " max_diff: ", max_difference, node_vals, bitstr)
            return False
        #else:
        #    print("is valid. num_oth_same: ", num_others_same, " max_diff: ", max_difference, node_vals, bitstr)
    return True


def retrieve_pair_partitions(bitmask_dict, node_list, max_difference=None, return_LL=False):
    with_LL = False
    for key in bitmask_dict:
        try:
            count = int(bitmask_dict[key])
        except:
            with_LL = True

    if not with_LL:  # dict = {'11111': 10, '00111': 3, etc}
        count = 0
        for bitstr in bitmask_dict:
            is_valid = check_valid(bitstr, node_list, max_difference)
            if is_valid:
                count += bitmask_dict[bitstr]
        if return_LL:
            print("Error! Bitmask doesn't have LL field!", return_LL, with_LL, bitmask_dict)
            sys.exit()
        else:
            return count
    else:    # dict = {'11111': {'count': 10, 'll': -1700}, '00111': {'count': 3, 'll': -1500), etc}
        count = 0
        log_weights = []
        for bitstr in bitmask_dict:
            is_valid = check_valid(bitstr, node_list, max_difference)
            if is_valid:
                count += bitmask_dict[bitstr]["count"]
                log_weights.append(np.log(count) + bitmask_dict[bitstr]["ll"])
        if return_LL:
            if len(log_weights) > 0:
                return count, logsumexp(log_weights)
            else:
                return count, np.NINF
        else:
            return count


def hamming_dist(str1, str2):
    i = 0
    count = 0
    while i < len(str1):
        if str1[i] != str2[i]:
            count += 1
        i += 1
    return count


def retrieve_min_hamming_partition(partitions, input_str, leaf_list):
    min_hdist = None
    closest_bitstr = None

    for bitstr in partitions:
        node_vals = []
        for node_i in leaf_list:
            node_vals.append(int(bitstr[node_i]))
        is_valid = all(ele == node_vals[0] for ele in node_vals)

        if is_valid:
            hdist = hamming_dist(input_str, bitstr)
            if min_hdist is None or hdist < min_hdist:
                min_hdist = hdist
                closest_bitstr = bitstr

        elif bitstr != '1'*len(bitstr):
            bitstr.translate(str.maketrans("01", "10"))  # replaces 0s with 1s
            node_vals = []
            for node_i in leaf_list:
                node_vals.append(int(bitstr[node_i]))
            is_valid = all(ele == node_vals[0] for ele in node_vals)

            if is_valid:
                hdist = hamming_dist(input_str, bitstr)
                if min_hdist is None or hdist < min_hdist:
                    min_hdist = hdist
                    closest_bitstr = bitstr

    return closest_bitstr, min_hdist
