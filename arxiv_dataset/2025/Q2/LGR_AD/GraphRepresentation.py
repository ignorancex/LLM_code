import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import tree
from itertools import *
from networkx.algorithms.community import *
from networkx.algorithms.clique import*
import copy
import random
import pandas as pd
from collections import defaultdict

class GraphRepresentation:
    
    
    def __init__(self,algorithms_,X):
        self.alg_instance= algorithms_
        self.algorithms=algorithms_.getAlgorithms()
        self.nodes = self.getNodes()
        self.algorithms_chars =algorithms_.characteristics
        self.topk=algorithms_.getTopK()
        self.graphs=[]
        self.graphs1=[]
        self.buildGraph(X)


    def buildGraph(self,X):
        for i,x in X.iterrows():
            G = nx.Graph()
            G_ = nx.Graph()
            for n in self.nodes:
                G.add_node(n)
            for i, n in enumerate(self.nodes):
                for j, n_ in enumerate(self.nodes):
                    if i<j and n is not n_:
                        G.add_edge(n, n_)
                        G_.add_edge(n, n_)
    
            for edge in G.edges():
                weight = self.CCF(edge)
                weight_= self.PCF(edge,x.values)
                G[edge[0]][edge[1]]['weight'] = weight
                G_[edge[0]][edge[1]]['weight'] = weight_
            edges_to_remove = [(u, v) for u, v, data in G_.edges(data=True) if data['weight'] == 0]
            G_.remove_edges_from(edges_to_remove)
            self.graphs.append(G)
            self.graphs1.append(G_)
            
        #nb_votes=self.nodes.count(n)
        #nb_votes=self.nodes.count(n)
    
    def PCF(self,edge,x):
            if len(x.shape) == 1:
                    x = x.reshape(1, -1)
            y1=self.alg_instance.predict_class(self.algorithms[edge[0]],x)
            y2=self.alg_instance.predict_class(self.algorithms[edge[1]],x)

            same_predictions_indices = np.where(y1== y2)[0]

            # Find intersection
            
            # Return the length of the intersection set
            return int(y1== y2)
     # Function to compare two models and find common characteristics
    def compare_models(self,model1, model2, model_dict):
        common_characteristics = {}
        true_count = 0

        # Check if the models exist in the dictionary
        if model1 not in model_dict or model2 not in model_dict:
            return "One or both models are not in the dictionary."
        
        # Loop through the characteristics of the first model
        for characteristic, value1 in model_dict[model1].items():
            
            # Get the value of the same characteristic for the second model
            value2 = model_dict[model2].get(characteristic, None)
            
            # Check if the characteristic exists in both models and if they share the same value
            if value2 is not None and value1 == value2:
                common_characteristics[characteristic] = value1
            if value1 is True:
                true_count += 1

        return true_count
    def CCF(self,edge):
           common_chars = self.compare_models(edge[0], edge[1], self.algorithms_chars)

         
           return common_chars
            
    def getNodes(self):

        return self.algorithms.keys()
    def getGraph(self):
        return self.G
    
    def plotGraph(self):
        nx.spring_layout(self.G)
        nx.draw_networkx(self.G, with_labels = True)
        
    def Adjacency_Matrix(self):
        A = nx.adjacency_matrix(self.G)
        print(A.todense())
    def hamming_distance(self,a, b):
        return sum(abs(e1 - e2) for e1, e2 in zip(a, b)) / len(a)
    def Maximum_Spaning_Tree (self):
        mst = tree.maximum_spanning_edges(self.G_, algorithm="prim", data=True)
        for path in mst:
            print(path)
    def top_k_msts(self, k,G):
        G_copy = copy.deepcopy(G)
        trees = []
        all_edges = list(G.edges())

        for _ in range(k):
            random.shuffle(all_edges)

            # Find the current maximum spanning tree
            T = nx.maximum_spanning_tree(G_copy)
            
            # If no more distinct trees can be found, break
            if not T.edges():
                break
            
            trees.append(T)
            
            # Remove the smallest edge from the current MST
            # This will allow us to find the next highest-weight spanning tree in the next iteration
            min_edge = min(T.edges(data=True), key=lambda x: x[2]['weight'])
            G_copy.remove_edge(min_edge[0], min_edge[1])
    
        return trees
    def diverse_spanning_trees(self,k, G):
        trees = []
        all_edges = list(G.edges())
        
        for _ in range(k):
            # Randomly shuffle edges
            random.shuffle(all_edges)
            
            # Create a new graph with the same nodes but no edges
            T = nx.Graph()
            T.add_nodes_from(G.nodes())
            
            # Add edges in random order, avoiding cycles
            for edge in all_edges:
                T.add_edge(*edge)
                if nx.is_cycle(T):
                    T.remove_edge(*edge)
            
            # Add the tree to the list if it's not identical to an existing tree
            if all(nx.is_isomorphic(T, tree) for tree in trees):
                trees.append(T)
        
        return trees

    def top_k_spanning_trees(self, k):
        trees = []
        for _ in range(k):
            # Find the current maximum spanning tree
            T = nx.maximum_spanning_tree(self.G)
            
            # If no more distinct trees can be found, break
            if not T.edges():
                break
            
            trees.append(T)
            
            # Remove the smallest edge from the current MST
            # This will allow us to find the next highest-weight spanning tree in the next iteration
            min_edge = min(T.edges(data=True), key=lambda x: x[2]['weight'])
            self.G.remove_edge(min_edge[0], min_edge[1])
    
        return trees
    def k_shortest_paths(self , k, weight='weight'):
        nodes = list(self.getNodes())
        paths= list(
            islice(nx.shortest_simple_paths(self.G, nodes[0], nodes[-1], weight=weight), k)
        )
        for path in paths:
            print(path)
        return paths
    def get_depth(self, G, node, parent=None):
        if parent is None:
            return 0
        return 1 + self.get_depth(G, parent)

    def getEncoding(self, k, g):
        # Initialize model set
        trees = self.top_k_msts(k, g)

        # Initialize a dictionary to store feature vectors
        feature_vectors = defaultdict(list)

        # Loop through each tree to calculate features
        for G in trees:
            for node in G.nodes():
                # Get depth (assuming root node has depth 0)
                try:
                    depth = nx.shortest_path_length(G, source=list(G.nodes())[0], target=node)
                except:
                    depth=0
                # Get number of children and multiply by weights
                neighbors = list(G.neighbors(node))
                num_children = len(neighbors)
                child_weight_product = 0
                for neighbor in neighbors:
                    weight = G[node][neighbor]['weight']
                    child_weight_product += num_children * weight

                # Create a feature vector for the current node in the current tree
                feature_vector = [depth, child_weight_product]

                # Append the feature vector to the list corresponding to the node
                feature_vectors[node].append(feature_vector)

        # Convert the dictionary of feature vectors to a DataFrame and average them
        average_features_per_node = {}
        for node, vectors in feature_vectors.items():
            df = pd.DataFrame(vectors, columns=['Depth', 'ChildWeightProduct'])
            average_features = df.mean()
            average_features_per_node[node] = average_features

        # Convert the dictionary of average features to a DataFrame for easier manipulation
        average_features_df = pd.DataFrame.from_dict(average_features_per_node, orient='index')
        flattened_df = average_features_df.stack().to_frame().T

        # Reset the column names
        flattened_df.columns = [f"{col[0]}_{col[1]}" for col in flattened_df.columns]

        return flattened_df

    def children_weights_sum(self,tree):
        weights_sum = {}
    
        def dfs(node, parent):
            if parent is None:
                total_weight = 0

            else:
                for neighbor, data in tree[parent].items():

                    total_weight = data['weight'] 

            for neighbor, data in tree[node].items():
                if neighbor != parent:  # Ensure we don't revisit the parent node
                    total_weight += data['weight'] + dfs(neighbor, node)
            weights_sum[node] = total_weight
            return total_weight
    
        # Start DFS from an arbitrary node
        dfs(next(iter(tree.nodes())), None)
        return weights_sum

    def k_cliques_communities(self,k):
        cliques = list(k_clique_communities(self.G, k))
        for clique in cliques:
            print(clique)
        return cliques
    def cliques(self):
        cliques=find_cliques(self.G)
        for clique in cliques:
            print(clique)
        return cliques
