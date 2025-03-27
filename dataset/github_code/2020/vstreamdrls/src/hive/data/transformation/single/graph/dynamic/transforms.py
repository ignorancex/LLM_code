from datetime import datetime
from hive.data.transformation.utils import relabel_node_ids, normalize_adj, generate_adjacency_matrix, \
    sparse_mx_to_torch_sparse_tensor, generate_ns
import pandas as pd
import numpy as np
import scipy.sparse as sps
import random
import torch
import networkx as nx



class Compose(object):

    def __init__(self, dataset_path, dataset_file, device, relabel_nodes=False, model='VSTREAMDRLS', weight_threshold=1000):
        custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        self.data_pd = pd.read_csv(f"{dataset_path}/{dataset_file}", parse_dates=['viewing_minute'],
                              date_parser=custom_date_parser)

        self.data_pd['partner_node_id'] = self.data_pd['partner_node_id'].fillna(-1)
        self.data_pd['partner_node_id'] = self.data_pd['partner_node_id'].astype('int64')
        self.data_pd['weight'] = self.data_pd['downloadRate'].apply(lambda x: x / weight_threshold if x <= weight_threshold else 1)
        self.data_pd, self.num_nodes = relabel_node_ids(self.data_pd, relabel_nodes)
        self.data_pd.sort_values(by=['viewing_minute'], inplace=True)
        self.data_pd, self.minutes = self.__relabel_minutes__(self.data_pd)
        self.device = device
        self.model = model



    def split_data(self, val_perc = 0.6, ns=1, start_minute=0, end_minute=10):
        train_data = []
        for i in range(start_minute, end_minute + 1):
            train_df = self.data_pd[self.data_pd['viewing_minute'] == self.minutes[i]]
            if i < (len(self.minutes) - 1):
                next_df = self.data_pd[self.data_pd['viewing_minute'] == self.minutes[i+1]]
                if train_df.shape[0] > 0 and next_df.shape[0]:
                    train_graph = nx.from_pandas_edgelist(train_df, source='source_node_id', target='partner_node_id', edge_attr=['weight'])
                    next_graph = nx.from_pandas_edgelist(next_df, source='source_node_id', target='partner_node_id', edge_attr=['weight'])

                    train_graph.remove_node(-1)
                    next_graph.remove_node(-1)

                    train_adj, \
                    train_adj_norm, \
                    train_edges_with_ns, \
                    val_edges_with_ns, \
                    test_edges_with_ns = self.__generate_val_test_edges__(train_graph, next_graph, val_perc, ns)

                    train_adj_norm_tensor = None
                    train_adj_tensor = None
                    train_edges_indices = []
                    train_edges_values = []
                    train_features = []
                    val_edges_indices = []
                    val_edges_values = []
                    test_edges_indices = []
                    test_edges_values = []
                    if self.model == "EVOLVEGCN":
                        train_adj_norm_tensor = sparse_mx_to_torch_sparse_tensor(train_adj_norm, device=self.device)
                        train_features = self.generate_features()
                    elif self.model == "VSTREAMDRLS":
                        train_adj_tensor = torch.tensor(train_adj, dtype=torch.float32, device=self.device).to_sparse()
                        if i == end_minute:
                            train_adj_norm_tensor = sparse_mx_to_torch_sparse_tensor(train_adj_norm, device=self.device)
                            train_features = self.generate_features()

                    if i == end_minute:
                        train_edges_indices, train_edges_values = self.__transform_arrays_to_tensor__(train_edges_with_ns)
                        val_edges_indices, val_edges_values = self.__transform_arrays_to_tensor__(val_edges_with_ns)
                        test_edges_indices, test_edges_values = self.__transform_arrays_to_tensor__(test_edges_with_ns)

                    train_data.append({'adj': train_adj_tensor, 'adj_norm': train_adj_norm_tensor, 'features': train_features, 'indices': train_edges_indices, 'values': train_edges_values})
        val_data = {'indices': val_edges_indices, 'values':val_edges_values}
        test_data = {'indices': test_edges_indices, 'values':test_edges_values}

        return train_data, val_data, test_data

    def generate_features(self, node_features=False):
        if not node_features:
            features = torch.tensor(sps.identity(self.num_nodes, dtype=np.float32, format='coo').todense(), dtype=torch.float32, device=self.device)
        return features

    def __transform_arrays_to_tensor__(self, edges):
        adj_indices = [torch.tensor([edge[0] for edge in edges], requires_grad=False, device=self.device), torch.tensor([edge[1] for edge in edges], requires_grad=False, device=self.device)]
        adj_values =  torch.reshape(torch.tensor([edge[2] for edge in edges], dtype=torch.float32, device=self.device), (-1,))
        return adj_indices, adj_values



    def __generate_val_test_edges__(self, graph, next_graph, val_perc = 0.6, ns = 1):

        train_graph_edges = set(graph.edges())
        next_graph_edges = set(next_graph.edges())

        all_edges = train_graph_edges.union(next_graph_edges)

        new_edges_list = list(next_graph_edges - train_graph_edges)
        random.shuffle(new_edges_list)

        val_edges_num = int(len(new_edges_list) * val_perc)
        val_graph_edges = new_edges_list[:val_edges_num]
        val_graph_edges_ns = generate_ns(val_graph_edges, all_edges, self.num_nodes, ns)

        test_graph_edges = new_edges_list[val_edges_num + 1:]
        test_graph_edges_ns = generate_ns(test_graph_edges, all_edges, self.num_nodes, ns)

        train_edges_ns = generate_ns(list(train_graph_edges), all_edges, self.num_nodes, ns)
        train_edges_tuples = [(edge[0],edge[1],edge[2]['weight']) for edge in graph.edges(data=True)]

        train_adj = generate_adjacency_matrix(train_edges_tuples, self.num_nodes)

        train_adj_norm = normalize_adj(train_adj)

        train_edges_with_ns = train_edges_tuples
        train_edges_with_ns.extend(train_edges_ns)

        val_edges_with_ns = [(src, dst, next_graph[src][dst]['weight']) for (src,dst) in val_graph_edges]
        val_edges_with_ns.extend(val_graph_edges_ns)

        test_edges_with_ns = [(src, dst, next_graph[src][dst]['weight']) for (src, dst) in test_graph_edges]
        test_edges_with_ns.extend(test_graph_edges_ns)

        return train_adj, train_adj_norm, train_edges_with_ns, val_edges_with_ns, test_edges_with_ns

    def __relabel_minutes__(self, df):
        unique_minutes = df['viewing_minute'].astype(np.int64).unique()
        unique_minutes_sorted = np.sort(unique_minutes)
        minutes_set = {minute:i for i, minute in enumerate(unique_minutes_sorted)}

        assign_minute_index = lambda x: minutes_set[x]

        df['viewing_minute'] = df['viewing_minute'].astype(np.int64).apply(assign_minute_index)
        dict_values = list(minutes_set.values())
        return df, np.sort(dict_values)

