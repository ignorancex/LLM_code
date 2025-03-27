# coding=utf-8
import pickle
import sys
import os
import re
import socket
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score, roc_auc_score, auc
from scipy import stats
import torch
import treelib

def translate_context(model, op, f, ctxt_v, bin_dict):
    # print(f, op)
    if op == '<=' or op == '>=':
        lo, hi = model.feat_range_mappings[f]
        # print(f, lo, hi)
        f_embeddings = model.numeric_embeddings.weight[lo:hi + 1]
        ctxt_vs = ctxt_v.expand_as(f_embeddings)
        le_vs = model.le_layers(torch.cat([f_embeddings, ctxt_vs], dim=-1)).flatten().detach().cpu().numpy()
        ge_vs = model.ge_layers(torch.cat([f_embeddings, ctxt_vs], dim=-1)).flatten().detach().cpu().numpy()
        v_idx = np.argmin(np.abs(le_vs - ge_vs))
        if bin_dict is not None and (f in bin_dict or f[:-2] in bin_dict):
            if f in bin_dict:
                lo, hi = bin_dict[f][v_idx], bin_dict[f][v_idx + 1]
            else:
                lo, hi = bin_dict[f[:-2]][v_idx], bin_dict[f[:-2]][v_idx + 1]
            v_idx = '{}:({},{}]'.format(v_idx, lo, hi)
    else:
        lo, hi = model.feat_range_mappings[f]
        f_embeddings = model.multihot_embeddings.weight[lo:hi + 1]
        ctxt_vs = ctxt_v.expand_as(f_embeddings)
        in_vs = model.blto_layers(torch.cat([f_embeddings, ctxt_vs], dim=-1)).flatten().detach().cpu().numpy()
        # v_idx = in_vs.argsort()[::-1]
        v_idx = in_vs.argsort()[-5:][::-1]
        v_idx = list(v_idx[in_vs[v_idx] > 0.5])
        # if len(v_idx) > 0:
        #     print(in_vs)
        if bin_dict is not None:
            v_idx = '{}:[{}]'.format(str(v_idx), ','.join([bin_dict[f][c] for c in v_idx])).replace(' ', '')
            # print(v_idx)
    return str(v_idx).replace(' ', '')


def translate_voting_weights(model, state_vs):
    states = [state_vs[-1]]
    idx = len(state_vs) - 1
    while idx > 0:
        parent_idx = (idx - 1) // 2
        yn = 0 if idx % 2 == 1 else 1
        parent_state = state_vs[parent_idx][yn].expand_as(states[-1])
        states.insert(0, parent_state)
        idx = parent_idx
    states = torch.cat(states, dim=-1)
    v_w = model.rule_weight_layers(states)
    return v_w.flatten().detach().cpu().numpy()


def draw_tree(model, tree_idx, bin_dict):
    nuf_list = list(model.feat_range_mappings.keys())
    mhf_list = list(model.feat_range_mappings.keys())
    nu_num = len(nuf_list)
    mh_num = len(mhf_list)
    nas_w = model.nas_w[tree_idx]
    state_vs = []
    tree = treelib.Tree()
    for node_idx in range(model.node_n):
        node_w = nas_w[node_idx]
        op_idx = node_w.argmax().cpu().numpy()
        if op_idx < nu_num:
            op = '>='
            f_idx = op_idx
            f = nuf_list[f_idx]
            ctxt_v = model.ge_v[tree_idx][node_idx][f_idx]
            ctxt = translate_context(model, op, f, ctxt_v, bin_dict)
            state_2v = model.ge_state_layers(ctxt_v).view(2, -1)
        elif op_idx < 2 * nu_num:
            op = '<='
            f_idx = op_idx - nu_num
            f = nuf_list[f_idx]
            ctxt_v = model.le_v[tree_idx][node_idx][f_idx]
            ctxt = translate_context(model, op, f, ctxt_v, bin_dict)
            state_2v = model.le_state_layers(ctxt_v).view(2, -1)
        else:
            op = 'in'
            f_idx = op_idx - 2 * nu_num
            f = mhf_list[f_idx]
            ctxt_v = model.blto_v[tree_idx][node_idx][f_idx]
            ctxt = translate_context(model, op, f, ctxt_v, bin_dict)
            state_2v = model.blto_state_layers(ctxt_v).view(2, -1)
        state_vs.append(state_2v)
        node_str = '{} {} {}'.format(f, op, ctxt)
        if node_idx >= model.node_n // 2:
            yes_w, no_w = translate_voting_weights(model, state_vs)
            node_str += ' {:.4f},{:.4f}'.format(yes_w, no_w)
        parent = (node_idx - 1) // 2
        if parent < 0:
            tree.create_node('{} {}'.format(node_idx, node_str), node_idx)
        else:
            tree.create_node('{} {}'.format(node_idx, node_str), node_idx, parent=parent)
    return tree

def reverse_tag(tag):
    if ' <= ' in tag:
        tag = tag.replace(' <= ', ' > ')
    elif ' >= ' in tag:
        tag = tag.replace(' >= ', ' < ')
    else:
        tag = tag.replace(' in ', ' notin ')
    return tag


def draw_rules(model, bin_dict):
    rules = []
    for tree_idx in range(model.rule_n):
        tree = draw_tree(model, tree_idx, bin_dict)
        # print(tree.nodes)
        for leaf_idx in range(model.node_n // 2, model.node_n):
            leaf_node = tree.get_node(leaf_idx)
            tags = leaf_node.tag.split(' ')
            left_w, right_w = tags[-1].split(',')
            left = [' '.join(tags[:-1] + [left_w])]
            right = [reverse_tag(' '.join(tags[:-1] + [right_w]))]
            while leaf_idx > 0:
                parent_idx = (leaf_idx - 1) // 2
                parent_node = tree.get_node(parent_idx)
                parent_tag = parent_node.tag
                if leaf_idx % 2 == 0:
                    parent_tag = reverse_tag(parent_tag)
                # print(leaf_idx, parent_idx, parent_tag)
                left.insert(0, parent_tag)
                right.insert(0, parent_tag)
                leaf_idx = parent_idx
            rules.append((tree_idx, left))
            rules.append((tree_idx, right))
    results = []
    for tree_idx, rule in rules:
        print(tree_idx, rule)
        weight = None
        tree = treelib.Tree()
        for idx, node in enumerate(rule):
            if idx == 0:
                tree.create_node(node, idx)
            else:
                if idx != len(rule) - 1:
                    tree.create_node(node, idx, idx - 1)
                else:
                    node = node.split(' ')
                    weight = float(node[-1])
                    tree.create_node(' '.join(node[:-1]), idx, idx - 1)
        results.append((weight, tree_idx, tree))
    return results


# def main():
#     # # deepest
#     dataset_name = 'Adult'
#     model_version = '4b8d233f9c9fa938d05b_1949' # add version here

#     bin_dict = read_bin_dict(dataset_name)
#     model = read_model(model_name='ENRL', model_version=model_version, dataset_name=dataset_name)
#     tree = draw_tree(model, 0, bin_dict)
#     rules = draw_rules(model, bin_dict)
#     rules = sorted(rules, key=lambda x: x[0], reverse=True)
#     for w, tree, rule in rules[:10] + rules[-10:]:
#         print(w, tree)
#         print(rule)
#     # print(draw_tree(model, 19, bin_dict))
#     # print(draw_tree(model, 22, bin_dict))
#     print(bin_dict)
#     return


# if __name__ == '__main__':
#     main()
