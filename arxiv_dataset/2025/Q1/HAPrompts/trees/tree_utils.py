import os
import nltk
import pickle

# generic tree related utils
def pretty_print_nltk_tree(tree, tab_len=40):
    print(pretty_print_recursive(tree, 0, '', tab_len))
    return


def pretty_print_recursive(tree, depth, printstr, tab_len=40):
    tab = '|' + ' ' * (tab_len - 1)
    tab2 = '-' + ' ' * (tab_len - 1)

    if not isinstance(tree, str):  # not a leaf node
        word = tree.label()
    else:
        word = tree  # leaf node is represented as str

    # maintain label length to tab_len
    if len(word) > tab_len:
        print(f'truncate node label {word} to {word[:tab_len]}')
        word = word[:tab_len]

    while len(word) < tab_len:
        word = word + ' '

    for _ in range(depth - 1):
        printstr = printstr + tab

    if depth > 0:
        printstr = printstr + tab2

    printstr = printstr + word
    printstr = printstr + '\n'

    if not isinstance(tree, str):
        for node in tree:
            printstr = pretty_print_recursive(node, depth + 1, printstr, tab_len)

    return printstr


def pretty_print_nltk_tree_limited_depth(tree, tab_len=40, depth_limit=5):
    print(pretty_print_recursive_limited_depth(tree, 0, depth_limit, '', tab_len))
    return 


def pretty_print_recursive_limited_depth(tree, depth, depth_limit, printstr, tab_len=13):
    """
        depth-first traversal of the nltk tree
        only print the tree to a certain depth
    """
    if depth >= depth_limit: return printstr

    tab = '|' + ' ' * (tab_len - 1)
    tab2 = '-' + ' ' * (tab_len - 1)

    if not isinstance(tree, str):  # not a leaf node
        word = tree.label()
    else:
        word = tree  # leaf node is represented as str

    # add children counts to classname/word
    if isinstance(tree,str):
        word = word + ' (0)' # leaf
    else:
        word = word + f' ({len(tree)})' # number of children

    # maintain label length to tab_len
    if len(word) > tab_len:
        print(f'truncate node label {word} to {word[:tab_len]}')
        word = word[:tab_len]

    while len(word) < tab_len:
        word = word + ' '

    for i in range(depth - 1):
        printstr = printstr + tab

    if depth > 0:
        printstr = printstr + tab2

    printstr = printstr + word
    printstr = printstr + '\n'

    if not isinstance(tree, str):
        for node in tree:
            printstr = pretty_print_recursive_limited_depth(node, depth + 1, depth_limit, printstr, tab_len)

    return printstr


def root_to_leaf_path(tree, leaf):
    """
        return a list of tree nodes: [root, ..., leaf]
    """
    if type(tree) is nltk.Tree:
        path = []
        for node in tree:
            # recursion
            path = root_to_leaf_path(node, leaf)
            if len(path) != 0:
                break

        if len(path) != 0:
            path = [tree] + path
        else:
            path = []

    elif tree == leaf:  # base case 0
        path = [tree]

    else:  # base case 1
        path = []

    return path    


def lowest_common_ancestor(tree, leaf_a, leaf_b):
    path_a = root_to_leaf_path(tree, leaf_a)
    # print(f"path a:\n{path_a}")
    path_b = root_to_leaf_path(tree, leaf_b)
    # print(f"path b:\n{path_b}")
    max_depth = min(len(path_a), len(path_b))

    cnt = 0
    for i in range(max_depth):
        if path_a[i] == path_b[i]:
            cnt += 1
        else:
            break
    return path_a[cnt - 1]


def hierarchical_distance(tree, leaf_a, leaf_b):
    lca = lowest_common_ancestor(tree, leaf_a, leaf_b)
    try:
        height = lca.height() - 1
    except:
        print(f'exception, lca: {lca}, leaf_a: {leaf_a}, leaf_b: {leaf_b}')
    return height
    
    
def tree_to_distance_dict(tree, viz=False):
    leaves = tree.leaves()
    n_leaves = len(leaves)
    distance_dict = {}

    cnt = 0
    total_cnt = int(n_leaves * (n_leaves-1)/2)

    for i in range(n_leaves-1):  # i in [0, n_leaves-1)
        distance_dict[(leaves[i], leaves[i])] = 0
        for j in range(i+1, n_leaves):  # j in [i+1, n_leaves]
            if leaves[i] == leaves[j]:
                print(f"leaves[{i}]({leaves[i]}) == leaves[{j}]({leaves[j]})")
            hdistance = hierarchical_distance(tree, leaves[i], leaves[j])
            distance_dict[(leaves[i], leaves[j])] = hdistance
            distance_dict[(leaves[j], leaves[i])] = hdistance

            cnt += 1
            if viz:
                print(f'progress {cnt}/{total_cnt} = {cnt*100/total_cnt:.2f}%')

    distance_dict[(leaves[-1], leaves[-1])] = 0
    return distance_dict


# hierarchical distance helper class
class DistanceDict(dict):
    """
    Small helper class implementing a symmetrical dictionary to hold distance data.
    """

    def __init__(self, distances):
        self.distances = {tuple(sorted(t)): v for t, v in distances.items()}

    def __getitem__(self, i):
        if i[0] == i[1]:
            return 0
        else:
            return self.distances[(i[0], i[1]) if i[0] < i[1] else (i[1], i[0])]

    def __setitem__(self, i):
        raise NotImplementedError()


def load_hie_distance(dataset, data_dir):
    if dataset == 'food-101':
        fname = os.path.join(data_dir, dataset, 'food101_hdist_from_ChatGPT.pkl')
    elif dataset == 'ucf-101':
        fname = os.path.join(data_dir, dataset,'UCF101_hdist_from_chatgpt_and_manual.pkl')
    elif dataset == 'cub-200':
        fname = os.path.join(data_dir, dataset, "CUB200_hdist_from_flamingo.pkl")
    elif dataset == 'sun-324':
        fname = os.path.join(data_dir, dataset, 'SUN324_hdist.pkl')
    elif dataset == 'imagenet':
        fname = os.path.join(data_dir, dataset, 'imagenet_hdist_cupl_vcd.pkl')
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    with open(fname, 'rb') as f:
        return DistanceDict(pickle.load(f))


def load_tree(dataset, data_dir):
    if dataset == 'food-101':
        fname = os.path.join(data_dir, dataset, 'food101_tree_from_ChatGPT.pkl')
    elif dataset == 'ucf-101':
        fname = os.path.join(data_dir, dataset, 'UCF101_tree_from_chatgpt_and_manual.pkl')
    elif dataset == 'cub-200':
        fname = os.path.join(data_dir, dataset, "CUB200_tree_from_flamingo.pkl")
    elif dataset == 'sun-324':
        fname = os.path.join(data_dir, dataset, 'SUN324_tree.pkl')
    elif dataset == 'imagenet':
        fname = os.path.join(data_dir, dataset, 'imagenet_tree_cupl_vcd.pkl')
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    with open(fname, 'rb') as f:
        return pickle.load(f)


def dfs_remove_tree_node(tree, node_name):
    """
        tree: nltk tree node
        node_name: string

        node_name can't be the root of the tree
        node_name can't be a leaf node
    """
    target = None
    children = []
    for node in tree:
        if isinstance(node, str): continue
        if node.label() == node_name: # found the node to remove
            # collect its children nodes
            for child in node:
                children.append(child)
            target = node
            break
        # depth-first search node
        dfs_remove_tree_node(node, node_name)
    
    if target is not None:
        # remove the target node in current subtree's children list
        tree.remove(target)

        # add the target nodes children to current subtree's children list
        for c in children:
            tree.append(c)
    return


def dfs_tree_node_by_name(tree, node_name):
    """
        node_name can't be leaf node name
    """
    target_node = None
    if isinstance(tree, str): return target_node
    if tree.label() == node_name: return tree

    for node in tree:
        target_node = dfs_tree_node_by_name(node, node_name)
        if target_node is not None: break
    return target_node


def pop_leaf_node(tree, node_name):
    """
        remove leaf node specified by 'node_name' from the 'tree'
        and return the node removed
    """
    """
        tree: nltk tree node
        node_name: string

        node_name can't be the root of the tree
        node_name can't be a leaf node
    """
    target = None
    match = False
    for node in tree:
        
        if isinstance(node, str): 
            if node == node_name:
                target = node
                match = True
                break
            else:
                continue 
        else:
            target = pop_leaf_node(node, node_name)
            if target is not None:
                break
    
    if target is not None and match == True:
        # remove the target node in current subtree's children list
        tree.remove(target)

    return target


def pop_non_leaf_node(tree, node_name):
    target = None
    match = False
    for node in tree:
        if isinstance(node, str): continue # skip leaf node
        if node.label() == node_name:
            target = node
            match = True
            break
        else:
            target = pop_non_leaf_node(node, node_name)
            if target is not None:
                break
            
    if target is not None and match == True:
        # remove the target node in current subtree's children list
        tree.remove(target)

    return target


def add_children_to_node(tree, parent_node_name, node_list):
    """
        add nodes in 'node_list' as children of 'parent_node'
        in 'tree'
    """
    parent_node = dfs_tree_node_by_name(tree, parent_node_name)
    for node in node_list:
        parent_node.append(node)
    return


def wn_offset_to_cupl_classname(cupl_imagenet_classes, offset_dict):
    wn_to_cupl_name = dict()
    for k, v in offset_dict.items():
        offset = v['id'].split('-')[0] # e.g., '01440764'
        pos = v['id'].split('-')[1] # e.g., 'n'
        posoffset = pos + offset

        cupl_name = cupl_imagenet_classes[k].replace('/', 'or').lower()

        if posoffset == 'n09229709': # bubble -> soap bubble
            wn_to_cupl_name[posoffset] = 'soap bubble'
        else:
            wn_to_cupl_name[posoffset] = cupl_name
    
    return wn_to_cupl_name


def wn_offset_to_cupl_and_vcd_classname(cupl_imagenet_classes, offset_dict):
    wn_to_cupl_name = dict()
    for k, v in offset_dict.items():
        offset = v['id'].split('-')[0] # e.g., '01440764'
        pos = v['id'].split('-')[1] # e.g., 'n'
        posoffset = pos + offset

        cupl_name = cupl_imagenet_classes[k].replace('/', 'or').lower()

        # name adjustment for imagenet-1k
        if cupl_name == 'tights': cupl_name = 'maillot'
        if cupl_name == 'newt': cupl_name = 'eft'
        if cupl_name == 'bubble': cupl_name = 'soap bubble'

        
        wn_to_cupl_name[posoffset] = cupl_name
    
    return wn_to_cupl_name

# EOF