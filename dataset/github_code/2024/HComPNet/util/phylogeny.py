import os
import pandas as pd
import math
import pickle
import pprint
pp = pprint.PrettyPrinter(indent=4)

from opentree import OT
from ete3 import Tree, PhyloTree

# Constants
Fix_Tree = True
format_ = 1 #8

class Phylogeny:
    # Phylogeny class for Fish dataset
    # If node_ids is None, it assumes that the tree already exists. Otherwise, you have to pass node_ids (i.e., list of species names).
    def __init__(self, filePath, node_ids=None, verbose=False):
        # filenames for phylo tree and cached mapping ottid-speciesname
        cleaned_fine_tree_fileName = "cleaned_metadata.tre"
        name_conversion_file = "name_conversion.pkl"
        self.ott_ids = []
        self.ott_id_dict = {}
        self.node_ids = node_ids
        self.treeFileNameAndPath = os.path.join(filePath, cleaned_fine_tree_fileName)
        self.conversionFileNameAndPath = os.path.join(filePath, name_conversion_file)
        self.total_distance = -1 # -1 means we never calculated it before.

        self.distance_matrix = {}
        self.species_groups_within_relative_distance = {}

        self.get_ott_ids(node_ids, verbose=verbose)
        self.get_tree(self.treeFileNameAndPath)
        self.get_total_distance()
    
    # Given two species names, get the phylo distance between them
    def get_distance(self, species1, species2):
        d= None
        if self.distance_matrix[species1][species2] == -1:
            if species1 == species2:
                return 0

            ott_id1 = 'ott' + str(self.ott_id_dict[species1])
            ott_id2 = 'ott' + str(self.ott_id_dict[species2])
            d = self.tree.get_distance(ott_id1, ott_id2)

            self.distance_matrix[species1][species2] = d
        else:
            d = self.distance_matrix[species1][species2]

        return d

    # relative_distance = 0 => species node itself
    # relative_distance = 1 => all species 
    def get_siblings_by_name(self, species, relative_distance, verbose=False):
        #NOTE: This implementation was causing inconsistencies since finding the parent.get_leaves() was not equivalent to get_species_groups 
        # ott_id = 'ott' + str(self.ott_id_dict[species])
        # return self.get_siblings_by_ottid(ott_id, relative_distance, get_ottids, verbose)
        
        self.get_species_groups(relative_distance, verbose)
        for species_group in self.species_groups_within_relative_distance[relative_distance]:
            if species in species_group:
                return species_group
        
        raise species+" was not found in " + self.species_groups_within_relative_distance[relative_distance]
    
    def get_parent_by_name(self, species, relative_distance, verbose=False):
        ott_id = 'ott' + str(self.ott_id_dict[species])
        parent = self.get_parent_by_ottid(ott_id, relative_distance, verbose)
        return parent
    
    def get_distance_between_parents(self, species1, species2, relative_distance):
        parent1 = self.get_parent_by_name(species1, relative_distance)
        parent2 = self.get_parent_by_name(species2, relative_distance)
        return self.tree.get_distance(parent1, parent2)
    
    def get_species_groups(self, relative_distance, verbose=False):
        if relative_distance not in self.species_groups_within_relative_distance.keys():
            groups = {}

            for species in self.getLabelList():
                parent_node = self.get_parent_by_name(species, relative_distance, verbose)
                parent = parent_node.name
                if parent not in groups.keys():
                    groups[parent] = [species]
                else:
                    groups[parent].append(species)
            
            self.species_groups_within_relative_distance[relative_distance] = groups.values()
            
            if verbose:
                print("At relative_distance", relative_distance, ", the groups are:", groups.values())
        
        return self.species_groups_within_relative_distance[relative_distance]
                
            

    def getLabelList(self):
        return list(self.node_ids)


    # ------- privete functions

    def get_total_distance(self):
        if self.node_ids is None:
            self.node_ids = self.ott_id_dict.keys()

        self.init_distance_matrix()

        # For one time, measure distance from all leaves down to root. They all should be equal.
        # Save the value and reuse it.
        
        if self.total_distance==-1:
            for leaf in self.tree.iter_leaves():
                total_distance = self.tree.get_distance(leaf) # gets distance to rootprint
                # print(total_distance)
                assert math.isclose(self.total_distance, total_distance) or self.total_distance==-1
                self.total_distance = total_distance

        return self.total_distance

    def init_distance_matrix(self):
        for i in self.node_ids:
            self.distance_matrix[i] = {}
            for j in self.node_ids:
                self.distance_matrix[i][j] = -1
                
    def get_parent_by_ottid(self, ott_id, relative_distance, verbose=False):
        abs_distance = relative_distance*self.total_distance
        species_node = self.tree.search_nodes(name=ott_id)[0]
        if verbose:
            print('distance to ancestor: ', abs_distance, ". relaive distance: ", relative_distance)

        # keep going up till distance exceeds abs_distance
        distance = 0
        parent = species_node
        while distance < abs_distance:
            if parent.up is None:
                break
            parent = parent.up
            distance = self.tree.get_distance(parent, species_node)
        
        return parent



    #     return ott_id_list
    # node_ids: list of taxa
    # returns: corresponding list of ott_ids
    def get_ott_ids(self, node_ids, verbose=False):
        if not os.path.exists(self.conversionFileNameAndPath):
            if node_ids is None:
                raise TypeError('No existing ottid-speciesnames found. node_ids should be a list of species names.')
            if verbose:
                print('Included taxonomy: ', node_ids, len(node_ids))
                df2 = pd.DataFrame(columns=['in csv', 'in response', 'Same?'])

            # Get the matches
            resp = OT.tnrs_match(node_ids, do_approximate_matching=True)
            matches = resp.response_dict['results']
            unmatched_names = resp.response_dict['unmatched_names']

            # Get the corresponding ott_ids
            ott_ids = set()
            ott_id_dict={}
            assert len(unmatched_names)==0 # everything is matched!
            for match_array in matches:
                match_array_matches = match_array['matches']
                assert len(match_array_matches)==1, match_array['name'] + " has too many matches" + str(list(map(lambda x: x['matched_name'], match_array_matches)))  # we have a single unambiguous match!
                first_match = match_array_matches[0]
                ott_id = first_match['taxon']['ott_id']
                ott_ids.add(ott_id)
                if verbose:
                    #some original and matched names are not exactly the same. Not a bug
                    df2 = df2.append({'in csv':match_array['name'], 'in response': first_match['matched_name'], 'Same?': match_array['name'] == first_match['matched_name']}, ignore_index=True)
                ott_id_dict[match_array['name']] = ott_id
            ott_ids = list(ott_ids)

            if verbose:
                print(df2[df2['Same?']== False])
                pp.pprint(ott_id_dict)

            with open(self.conversionFileNameAndPath, 'wb') as f:
                pickle.dump([ott_ids, ott_id_dict], f)
        else:
            with open(self.conversionFileNameAndPath, 'rb') as f:
                ott_ids, ott_id_dict = pickle.load(f)

        

        self.ott_ids = ott_ids
        self.ott_id_dict = ott_id_dict
        print(self.ott_id_dict)

    def fix_tree(self, treeFileNameAndPath):
        tree = PhyloTree(treeFileNameAndPath, format=format_)

        # Special case for Fish dataset: Fix Esox Americanus.
        D = tree.search_nodes(name="mrcaott47023ott496121")[0]
        D.name = "ott496115"
        tree.write(format=format_, outfile=treeFileNameAndPath)
    
    def get_tree(self, treeFileNameAndPath):
        if not os.path.exists(treeFileNameAndPath):
            output = OT.synth_induced_tree(ott_ids=self.ott_ids, ignore_unknown_ids=False, label_format='id') # name_and_id ott_ids=list(ott_ids),

            output.tree.write(path = treeFileNameAndPath, schema = "newick")

            if Fix_Tree:
                self.fix_tree(treeFileNameAndPath)

        self.tree = PhyloTree(treeFileNameAndPath, format=format_)


class PhylogenyCUB:
    # Phylogeny class for CUB dataset
    def __init__(self, filePath, node_ids=None, verbose=False):
        self.node_ids = node_ids
        self.treeFileNameAndPath = filePath
        self.total_distance = -1 # -1 means we never calculated it before.

        self.distance_matrix = {}
        self.species_groups_within_relative_distance = {}

        self.get_tree(self.treeFileNameAndPath)
        self.get_total_distance()
    
    # Given two species names, get the phylo distance between them
    def get_distance(self, species1, species2):
        d= None
        if self.distance_matrix[species1][species2] == -1:
            if species1 == species2:
                return 0
            d = self.tree.get_distance(species1, species2)

            self.distance_matrix[species1][species2] = d
        else:
            d = self.distance_matrix[species1][species2]

        return d

    # relative_distance = 0 => species node itself
    # relative_distance = 1 => all species 
    def get_siblings_by_name(self, species, relative_distance, verbose=False):
        #NOTE: This implementation was causing inconsistencies since finding the parent.get_leaves() was not equivalent to get_species_groups 
        # ott_id = 'ott' + str(self.ott_id_dict[species])
        # return self.get_siblings_by_ottid(ott_id, relative_distance, get_ottids, verbose)
        
        self.get_species_groups(relative_distance, verbose)
        for species_group in self.species_groups_within_relative_distance[relative_distance]:
            if species in species_group:
                return species_group
        
        raise species+" was not found in " + self.species_groups_within_relative_distance[relative_distance]

    def get_parent_by_name(self, species, relative_distance, verbose=False):
        abs_distance = relative_distance*self.total_distance

        species_node = self.tree.search_nodes(name=species)[0]

        # this line added since not all leaves are at equal distance from root. self.total_distance is max leaf to root length
        abs_distance = abs_distance - (self.total_distance - self.tree.get_distance(species_node))

        if verbose:
            print('distance to ancestor: ', abs_distance, ". relaive distance: ", relative_distance)

        # keep going up till distance exceeds abs_distance
        distance = 0
        parent = species_node
        while distance < abs_distance:
            if parent.up is None:
                break
            parent = parent.up
            distance = self.tree.get_distance(parent, species_node)
        
        return parent
    
    def get_distance_between_parents(self, species1, species2, relative_distance):
        parent1 = self.get_parent_by_name(species1, relative_distance)
        parent2 = self.get_parent_by_name(species2, relative_distance)
        return self.tree.get_distance(parent1, parent2)
    
    def get_species_groups(self, relative_distance, verbose=False):
        if relative_distance not in self.species_groups_within_relative_distance.keys():
            groups = {}

            for species in self.getLabelList():
                parent_node = self.get_parent_by_name(species, relative_distance, verbose)
                parent = parent_node.name
                if parent not in groups.keys():
                    groups[parent] = [species]
                else:
                    groups[parent].append(species)
            
            self.species_groups_within_relative_distance[relative_distance] = groups.values()
            
            if verbose:
                print("At relative_distance", relative_distance, ", the groups are:", groups.values())
        
        return self.species_groups_within_relative_distance[relative_distance]


    def getLabelList(self):
        return list(self.node_ids)


    # ------- privete functions

    def get_total_distance(self):
        if self.node_ids is None:
            self.node_ids = sorted([leaf.name for leaf in self.tree.iter_leaves()])

        self.init_distance_matrix()

        # maximum distance between root and lead node taken as total distance
        leaf_to_root_distances = [self.tree.get_distance(leaf) for leaf in self.tree.iter_leaves()]
        self.total_distance = max(leaf_to_root_distances)

        return self.total_distance

    def init_distance_matrix(self):
        for i in self.node_ids:
            self.distance_matrix[i] = {}
            for j in self.node_ids:
                self.distance_matrix[i][j] = -1
    
    def get_tree(self, treeFileNameAndPath):
        # if not os.path.exists(treeFileNameAndPath):
        #     output = OT.synth_induced_tree(ott_ids=self.ott_ids, ignore_unknown_ids=False, label_format='id') # name_and_id ott_ids=list(ott_ids),

        #     output.tree.write(path = treeFileNameAndPath, schema = "newick")

        self.tree = PhyloTree(treeFileNameAndPath, format=format_)

        # setting a dummy name to the internal nodes if it is unnamed
        for i, node in enumerate(self.tree.traverse("postorder")):
            if not len(node.name) > 0:
                node.name = "temp_" + str(i)
