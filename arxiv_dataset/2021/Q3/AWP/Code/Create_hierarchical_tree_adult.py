import numpy as np
from anytree import AnyNode
import random
import copy
import pandas as pd
import sys



def findNodesOfSize(root, size):
    answer = []
    nodes_to_check = [root]
    while len(nodes_to_check) > 0:
        node = nodes_to_check.pop(0)
        if len(node.examples) > size:
            nodes_to_check = nodes_to_check + list(node.children)
        elif len(node.examples) == size:
            answer = answer + [node]
    return answer

input_string = list(sys.argv)
data = pd.read_csv(input_string[1])
data_size = len(data)
num_nodes = 1
root = AnyNode(id="v" + str(num_nodes), examples=copy.deepcopy(list(data.index)))
nodes_to_split = [root]
min_node_size = 1 # determines the minimum size of nodes in the tree

while len(nodes_to_split) > 0:  # iteratively creates the tree
    node = nodes_to_split.pop(0)
    if len(node.examples) > min_node_size:
        df = data.iloc[node.examples, :].copy()
        for col in df.columns:     # Keep only columns that are relevant to be used as splitting criteria
            if len(df[col].unique()) == 1:
                df.drop(col, inplace=True, axis=1)
        cols = df.columns
        if len(cols) == 0:  # all the example in the node are similar (due to duplicates in data set)
            half = int(np.ceil(len(df) / 2)) # randomly divides them into two groups
            r_son = AnyNode(id="v" + str(num_nodes + 1),
                            examples=copy.deepcopy(list(df.iloc[:half].index)), parent=node)
            l_son = AnyNode(id="v" + str(num_nodes + 2),
                            examples=copy.deepcopy(list(df.iloc[half:].index)), parent=node)
        else:
            col = random.choice(cols)
            if col in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:  # numerical
                median = df[col].median()
                if len(set(df[df[col] < median].index)) == 0:
                    r_son = AnyNode(id="v" + str(num_nodes + 1),
                                    examples=copy.deepcopy(list(set(df[df[col] > median].index))), parent=node)
                    l_son = AnyNode(id="v" + str(num_nodes + 2),
                                    examples=copy.deepcopy(list(set(df[df[col] <= median].index))), parent=node)
                else:
                    r_son = AnyNode(id="v" + str(num_nodes + 1),
                                    examples=copy.deepcopy(list(set(df[df[col] >= median].index))), parent=node)
                    l_son = AnyNode(id="v" + str(num_nodes + 2),
                                    examples=copy.deepcopy(list(set(df[df[col] < median].index))), parent=node)

            else:  # discrete
                values, counts = np.unique(df[col], return_counts=True)
                # sort the unique values list by frequency
                count_sort_ind = np.argsort(-counts)
                counts_sorted = copy.deepcopy(counts[count_sort_ind])
                values_sorted = copy.deepcopy(values[count_sort_ind])
                # greedily finds a balanced split
                sum_A = 0
                sum_B = 0
                split_values = []
                for i in range(len(counts)):
                    if sum_A < sum_B:
                        sum_A += counts_sorted[i]
                        split_values.append(values_sorted[i])
                    else:
                        sum_B += counts_sorted[i]
                r_examples = copy.deepcopy(list(df[df[col].isin(split_values)].index))
                r_son = AnyNode(id="v" + str(num_nodes + 1), examples=r_examples, parent=node)
                l_examples = copy.deepcopy(list(df[~df[col].isin(split_values)].index))
                l_son = AnyNode(id="v" + str(num_nodes + 2), examples=l_examples, parent=node)
        num_nodes += 2
        nodes_to_split = nodes_to_split + [r_son, l_son] # replace node in pruning with its newly created son nodes

print("tree created")
# Saves the tree in the required format for the AWP python program
tree_csv = pd.DataFrame(columns=["LeftSon", "Rightson", "not_relevent", "Size"])
del data
leafs = findNodesOfSize(root, min_node_size)
for leaf in leafs:
    leaf.id = str(leaf.examples[0])

counter = 0
for i in range(data_size - min_node_size):
    nodes = findNodesOfSize(root, i + min_node_size + 1)
    for node in nodes:
        node.id = str(data_size + counter)
        tree_csv.loc[counter] = [node.children[0].id, node.children[1].id, 101, len(node.examples)]
        counter += 1

tree_csv.to_csv(input_string[2])