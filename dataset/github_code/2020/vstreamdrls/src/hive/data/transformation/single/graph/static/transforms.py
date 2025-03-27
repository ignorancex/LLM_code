from hive.data.transformation.utils import relabel_node_ids, generate_adjacency_matrix, generate_ns
from datetime import datetime
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sps
import random
import torch


class Compose(object):
    def __init__(self, dataset_path, dataset_file, device, relabel_nodes = False, weight_threshold=1000):
        custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        data_pd = pd.read_csv(f"{dataset_path}/{dataset_file}", parse_dates=['viewing_minute'], date_parser=custom_date_parser)
        data_pd['partner_node_id'] = data_pd['partner_node_id'].fillna(-1)
        data_pd['partner_node_id'] = data_pd['partner_node_id'].astype('int64')
        data_pd['downloadRate'] = data_pd['downloadRate'].fillna(0)
        data_pd['weight'] = data_pd['downloadRate'].apply(lambda x: x / weight_threshold if x <= weight_threshold else 1)


        data_pd, self.num_nodes = relabel_node_ids(data_pd, relabel_nodes)

        self.data = data_pd[['source_node_id', 'partner_node_id', 'weight']]

        self.static_graph = nx.from_pandas_edgelist(self.data, source='source_node_id', target='partner_node_id', edge_attr=['weight'])
        self.static_graph.remove_node(-1)

        self.all_edges = [(edge[0], edge[1], edge[2]['weight']) for edge in self.static_graph.edges(data=True)]
        self.edges_no_weights = set()
        for edge in self.static_graph.edges():
            self.edges_no_weights.add((edge[0],edge[1]))
        self.train_adj = np.zeros((self.num_nodes, self.num_nodes))
        self.val_adj = np.zeros((self.num_nodes, self.num_nodes))
        self.test_adj = np.zeros((self.num_nodes, self.num_nodes))
        self.device = device



    def get_features(self, node_features = False):
        return torch.tensor(sps.identity(self.num_nodes, dtype=np.float32, format='coo').todense(), dtype=torch.float32, device=self.device)


    def split_data(self, train_perc=0.6, val_perc = 0.5, ns=1):
        random.shuffle(self.all_edges)
        num_train_edges = int(len(self.all_edges) * train_perc)
        num_val_edges = int((len(self.all_edges) - num_train_edges) * val_perc)

        train_edges = self.all_edges[:num_train_edges]
        train_edges.extend(generate_ns(train_edges, self.edges_no_weights, self.num_nodes, ns))
        # print(train_edges)
        train_adj = torch.tensor(generate_adjacency_matrix(train_edges, self.num_nodes), dtype=torch.float32, device=self.device).to_sparse()
        train_edges_indices, train_edges_values = self.__transform_arrays_to_tensor__(train_edges)


        val_edges = self.all_edges[num_train_edges:(num_train_edges + num_val_edges)]
        val_edges.extend(generate_ns(val_edges, self.edges_no_weights, self.num_nodes, ns))
        val_edges_indices, val_edges_values = self.__transform_arrays_to_tensor__(val_edges)

        test_edges = self.all_edges[(num_train_edges + num_val_edges):]
        test_edges.extend(generate_ns(test_edges, self.edges_no_weights, self.num_nodes, ns))
        test_edges_indices,test_edges_values = self.__transform_arrays_to_tensor__(test_edges)

        return train_adj, train_edges_indices, train_edges_values, val_edges_indices, val_edges_values, test_edges_indices, test_edges_values


    def __transform_arrays_to_tensor__(self, edges):
        adj_indices = [torch.tensor([edge[0] for edge in edges], requires_grad=False, device=self.device), torch.tensor([edge[1] for edge in edges], requires_grad=False, device=self.device)]
        adj_values =  torch.reshape(torch.tensor([edge[2] for edge in edges], dtype=torch.float32 , device=self.device), (-1,))
        return adj_indices, adj_values




