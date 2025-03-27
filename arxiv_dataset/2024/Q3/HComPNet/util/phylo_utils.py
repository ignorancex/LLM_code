from util.node import Node
from util.phylogeny import PhylogenyCUB
from ete3 import TreeNode


def construct_discretized_phylo_tree(phylogeny_path, phyloDistances_string):
    phylo = PhylogenyCUB(phylogeny_path)
    root = Node("root")
    phyloDistances = [float(x) for x in phyloDistances_string.split(',')[::-1]] + [1]
    num_levels = len(phyloDistances)

    ances_lvl_tag_prefix = '_lvl'

    ancestor_lvl_to_spc_groups = {} # maps ancestor levels (int) to spc groups (dict mapping representative_species to a list of species)
    for ancestor_lvl, phylo_dist in enumerate(phyloDistances):
        if ancestor_lvl == len(phyloDistances)-1:
            ancestor_lvl_to_spc_groups[ancestor_lvl] = {spc_group[0]: spc_group \
                                                        for spc_group in phylo.get_species_groups(1-phylo_dist)}
        else:
            ancestor_lvl_to_spc_groups[ancestor_lvl] = {(spc_group[0] + ances_lvl_tag_prefix + str(ancestor_lvl)): spc_group \
                                                        for spc_group in phylo.get_species_groups(1-phylo_dist)}
        if ancestor_lvl == 0:
            children_list = []
            for representative, spc_group in ancestor_lvl_to_spc_groups[ancestor_lvl].items():
                children_list.append(representative)
            root.add_children(children_list)
        else:
            prev_level_representatives = [representative for representative, spc_group in ancestor_lvl_to_spc_groups[ancestor_lvl - 1].items()]
            prev_level_representative_to_children = {representative: [] for representative in prev_level_representatives}
            for representative, spc_group in ancestor_lvl_to_spc_groups[ancestor_lvl].items():
                for prev_lvl_rep in prev_level_representatives:
                    if representative.split(ances_lvl_tag_prefix)[0] in ancestor_lvl_to_spc_groups[ancestor_lvl - 1][prev_lvl_rep]:
                        prev_level_representative_to_children[prev_lvl_rep].append(representative)
                        break
            
            for prev_lvl_rep, children in prev_level_representative_to_children.items():
                root.add_children_to(prev_lvl_rep, children)

    def get_nonsingular_child(node):
        # singular is any internal (parent) node with only one child
        if node.num_children() == 0:
            return node
        if node.num_children() > 1:
            return node
        else:
            return get_nonsingular_child(node.children[0])

    for node in root.nodes_with_children():
        for i in range(len(node.children)):

            # maintaining a reference before replacing node.children[i]
            temp = node.children[i]

            # if node.children[i] is singular it will be replaced with nonsingular descendant 
            node.children[i] = get_nonsingular_child(node.children[i])

            # update the name to label mapping according to the new child
            label = node.children_to_labels[temp.name]
            del node.children_to_labels[temp.name]
            node.children_to_labels[node.children[i].name] = label

    return root

def construct_phylo_tree(phylogeny_path):
    phylo = PhylogenyCUB(phylogeny_path)
    root = Node("root")

    def set_names_to_internal_nodes(node: TreeNode):
        """
        Assumes internal node names are empty strings
        Assumes leaf node names are in the format 'cub_122_Harris_Sparrow'
        """
        if not node.is_leaf():
            child_names = [set_names_to_internal_nodes(child) for child in node.get_children()]
            name = "+".join([name.split("+")[0] for name in child_names])
            if (len(node.get_children()) > 1):
                node.name = name
            return name
        else:
            return node.name[4:7] # assumes cub name in the format cub_122_Harris_Sparrow

    def build_tree(parentnode: Node, ete3_node: TreeNode):
        
        if ete3_node.is_leaf() or (len(ete3_node.get_children()) > 1):
            parentnode.add_children(ete3_node.name)
            
        if (len(ete3_node.get_children()) == 1):
            build_tree(parentnode, ete3_node.get_children()[0])
            
        if not ete3_node.is_leaf() and (len(ete3_node.get_children()) > 1):
            node = parentnode.get_child(ete3_node.name)
            for ete3_child in ete3_node.get_children():
                build_tree(node, ete3_child)

    set_names_to_internal_nodes(phylo.tree.get_tree_root())

    for ete3_child in phylo.tree.get_tree_root().get_children():
        build_tree(root, ete3_child)
        
    return root
