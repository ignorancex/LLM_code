from pathlib import Path
import numpy as np
import pandas as pd
import base64
import sympy
import torch
import random
import time
import click
import os
import shutil
import h5py
import copy
import omegaconf
import json
import timeout_decorator
from tqdm import tqdm
from ControllableNesymres.utils import load_metadata_hdf5, retrofit_word2id, load_eq
from ControllableNesymres.dataset.data_utils import sample_symbolic_constants
from ControllableNesymres.dataset.generator import Generator, UnknownSymPyOperator
from ControllableNesymres.architectures.data import sympify_equation_timeout, remove_rationals_timeout, resolve_problematic_constants, return_costants



def process_equation(eq_string):  # obtain prefix
    try:
        eq_sympy_infix_with_constants = sympify_equation_timeout(eq_string)
        eq_sympy_prefix_with_constants = Generator.sympy_to_prefix(eq_sympy_infix_with_constants, enable_float=True)
    except:
        return None, None
    
    costants, eq_sympy_prefix_with_c = return_costants(eq_sympy_prefix_with_constants)
    
    return eq_sympy_infix_with_constants, eq_sympy_prefix_with_c





def simplify_prefix(prefix):  # delete constants from prefix
    if prefix == None:
        return None
    new_prefix = []
    i = 0
    while i < len(prefix):
        append = True
        if i != len(prefix) - 1:
            if (prefix[i] == "add" or prefix[i] == "mul") and (len(prefix[i + 1]) == 1 or prefix[i + 1][0] == '-'):
                i += 1
                append = False
        if append:
            if len(prefix[i]) == 1 or prefix[i][0] == '-':
                new_prefix.append('c')
            else:
                new_prefix.append(prefix[i])
        i += 1
    return new_prefix




def eq_to_simplified_prefix(eq, cfg): 
    consts, _ = sample_symbolic_constants(eq, cfg.dataset.constants)
    eq_string = eq.expr.format(**consts)
    eq_sympy_infix_with_constants, eq_sympy_prefix_with_c = process_equation(eq_string)
    simplified_prefix = simplify_prefix(eq_sympy_prefix_with_c)
    return simplified_prefix, eq_sympy_infix_with_constants




def return_left_and_right(prefix):  #given prefix should start with "add", and be a complete tree structure
    leaves = ["x_1", "x_2", "x_3", "x_4", "x_5", "c", "pi"]
    left = []
    right = []
    count = 2
    left_done_flag = False
    for i in range(1,len(prefix)):
        if prefix[i] == "add":
            count += 1
        elif prefix[i] in leaves:
            count -= 1
        
        if left_done_flag == False:
            left.append(prefix[i])
        else:
            right.append(prefix[i])

        if count < 2 and left_done_flag == False:
            left_done_flag = True
    return left, right



def standardize_prefix_order(prefix): 
    first_add_index = -1
    for i,token in enumerate(prefix):
        if token == "add":
            first_add_index = i
            break
    if first_add_index == -1:
        return prefix
    else:
        trees = []
        stack = []
        stack.append(prefix[first_add_index:])
        while len(stack) > 0:
            subprefix = stack.pop()
            left, right = return_left_and_right(subprefix)
            if left[0] == "add":
                stack.append(left)
            else:
                trees.append(left)
            if right[0] == "add":
                stack.append(right)
            else:
                trees.append(right)
        
        standardized_trees = []
        for tree in trees:
            standardized_trees.append(standardize_prefix_order(tree))

        standardized_trees = sorted(standardized_trees)

        res = []
        for i in range(first_add_index):
            res.append(prefix[i])
        for i,tree in enumerate(standardized_trees):
            if i != len(standardized_trees) - 1:
                res.append("add")
            for token in tree:
                res.append(token)

        return res
       


        
def return_all_possible_prefixes_(prefix):
    first_add_index = -1
    for i,token in enumerate(prefix):
        if token == "add":
            first_add_index = i
            break
    if first_add_index == -1:
        return [prefix]
    else:
        res = []
        left, right = return_left_and_right(prefix[first_add_index:])
        possible_prefixes_left = return_all_possible_prefixes_(left)
        possible_prefixes_right = return_all_possible_prefixes_(right)
        for possible_prefix_left in possible_prefixes_left:
            res.append(prefix[:first_add_index] + possible_prefix_left)
        for possible_prefix_right in possible_prefixes_right:
            res.append(prefix[:first_add_index] + possible_prefix_right)
        for possible_prefix_left in possible_prefixes_left:
            for possible_prefix_right in possible_prefixes_right:
                res.append(prefix[:first_add_index + 1] + possible_prefix_left + possible_prefix_right)
        return res



def return_all_possible_prefixes(prefix): # sometimes, multiple prefixes are possible from one eq in train data
    res = []
    res_ = return_all_possible_prefixes_(prefix)
    for prefix_ in res_:
        res.append(standardize_prefix_order(prefix_))
    return res





def infix_to_standardized_prefix(infix):
    try:
        prefix = process_equation(infix)[1]
        prefix = simplify_prefix(prefix)
        prefix = standardize_prefix_order(prefix)
        return prefix
    except:
        return None
    





def make_dataset_dict():  # what prefixes are in the train dataset?
    cfg = omegaconf.OmegaConf.load(Path("scripts/config.yaml"))
    dataset_path = f"train_datasets/dataset_training_nopow/datasets/100000"
    metadata_dataset = load_metadata_hdf5(Path(dataset_path))
    eqs_per_hdf = metadata_dataset.eqs_per_hdf

    try:
        with open(f"experiments/nopow_dataset_dict/dataset_dict.json", "r") as f:
            dataset_dict = json.load(f)
    except:
        dataset_dict = {}

    for i in tqdm(range(100000)):
        eq = load_eq(dataset_path, i, eqs_per_hdf)

        simplified_prefix, _ = eq_to_simplified_prefix(eq, cfg)
        if simplified_prefix is None:
            continue

        all_possible_prefixes = return_all_possible_prefixes(simplified_prefix)

        for possible_prefix in all_possible_prefixes:
            if not str(possible_prefix) in dataset_dict:
                dataset_dict[str(possible_prefix)] = 1
            else:
                dataset_dict[str(possible_prefix)] += 1

    print()
    dataset_list = sorted(dataset_dict.items(), key=lambda x:x[1], reverse=True)
    for i, tuple in enumerate(dataset_list):
        print(i, tuple[0], tuple[1])

    with open(f"experiments/nopow_dataset_dict/dataset_dict.json", "w") as f:
        json.dump(dataset_dict, f, indent=4)
    





def make_test_set():   # generate expressions whose prefix is not in train set
    cfg = omegaconf.OmegaConf.load(Path("scripts/config.yaml"))
    mother_dataset_path = f"temp/datasets/100000"
    test_dataset_path = f"test_datasets/original_test_sets/nopow.json"

    metadata_dataset = load_metadata_hdf5(Path(mother_dataset_path))
    eqs_per_hdf = metadata_dataset.eqs_per_hdf

    with open(f"experiments/nopow_dataset_dict/dataset_dict.json", "r") as f:
        dataset_dict = json.load(f)

    test_dataset_list = []
    counter1 = 0
    counter2 = 0

    for i in tqdm(range(1000)):
        eq = load_eq(mother_dataset_path, i, eqs_per_hdf)
        
        simplified_prefix, eq_sympy_infix_with_constants = eq_to_simplified_prefix(eq, cfg)
        if simplified_prefix == None:
            continue

        prefix = standardize_prefix_order(simplified_prefix)
        counter1 += 1
        
        if not str(prefix) in dataset_dict:
            counter2 += 1
            print(counter1, counter2)
            test_dataset_list.append({"eq_string": str(eq_sympy_infix_with_constants)})
        
    with open(test_dataset_path, "w") as f:
        json.dump(test_dataset_list, f, indent=4)

    print(counter1)
    print(counter2)






def is_expression_in_train_data(eq_string, dataset_dict_path = "experiments/nopow_dataset_dict/dataset_dict.json"):
    with open(dataset_dict_path, "r") as f:
        dataset_dict = json.load(f)
    infix = infix_to_standardized_prefix(eq_string)

    if str(infix) in dataset_dict:
        return True
    else:
        return False



    


if __name__ == "__main__":
    is_expression_in_train_data("x_1")







