import torch
from cogdl.data import Graph
import copy


class PseudoRanger(torch.utils.data.Dataset):
    def __init__(self, num):
        self.indices = torch.arange(num)
        self.num = num

    def __getitem__(self, item):
        return self.indices[item]

    def __len__(self):
        return self.num

    def shuffle(self):
        rand = torch.randperm(self.num)
        self.indices = self.indices[rand]


class AdjSampler(torch.utils.data.DataLoader):
    def __init__(self, graph, sizes=[2, 2], training=True, *args, **kwargs):
        
        self.graph = copy.deepcopy(graph)
        self.sizes = sizes
        self.degree = graph.degrees()
        self.diag = self._sparse_diagonal_value(graph)
        self.training = training
        if training:
            idx = torch.where(graph['train_mask'])[0]        
        else:
            idx = torch.arange(0, graph.x.shape[0])
        self.dataset = PseudoRanger(idx.shape[0])

        kwargs["collate_fn"] = self.collate_fn
        super(AdjSampler, self).__init__(self.dataset, *args, **kwargs)

    def shuffle(self):
        self.dataset.shuffle()

    def _sparse_diagonal_value(self, adj):
        row, col = adj.edge_index
        value = adj.edge_weight
        return value[row == col]

    def _construct_propagation_matrix(self, sample_adj, sample_id, num_neighbors):
        
        #removing duplicate edges
        edge_index=sample_adj.edge_index
        edge_index=torch.stack(edge_index,dim=0) #need this to convert from tuple
        edges_set = set(map(tuple, edge_index.t().tolist()))
        new_edge_index = torch.tensor(list(edges_set)).t().contiguous()
        row,col=new_edge_index
        
        value=torch.ones(len(row))
        degrees=self.degree #diag of D matrix
        degrees_inv_sqrt=degrees.pow(-0.5)
        degrees_inv_sqrt[degrees_inv_sqrt==float("inf")]=0
        value=degrees_inv_sqrt[sample_id[col]]*value*degrees_inv_sqrt[sample_id[row]]
        
        
        """add self connection"""
        num_row = row.max() + 1  
        row = torch.cat([torch.arange(0, num_row).long(), row], dim=0) #initial samples
        col = torch.cat([torch.arange(0, num_row).long(), col], dim=0)
        value = torch.cat([self.diag[sample_id[:num_row]], value], dim=0) #samples up until n_row
        
        
        
        value = value * self.degree[sample_id[row]] / num_neighbors
        
        ###########################
        
        new_graph = Graph()
        new_graph.edge_index = torch.stack([row, col])
        new_graph.edge_weight = value
        return new_graph

    def collate_fn(self, idx):
        if self.training:
            sample_id = torch.tensor(idx)
            sample_adjs, sample_ids = [], [sample_id]
            full_adjs, full_ids = [], []
            graph_no_self_loops=copy.deepcopy(self.graph)
            graph_no_self_loops.remove_self_loops() #removes the self loops for now so a node doesn't get sampled as its own neighbor
            for size in self.sizes: #each step in this for loop is one hop of neighbors - goes backward, starts from layer K
                full_id, full_adj = self.graph.sample_adj(sample_id, -1) #all neighbors starting from training nodes (if Train==True)
                ############# normalize full_adj to get P ################
                
                #symmetric normalization of the adjacency matrix
                degrees=self.graph.degrees() #diag of D matrix
                degrees_inv_sqrt=degrees.pow(-0.5)
                degrees_inv_sqrt[degrees_inv_sqrt==float("inf")]=0
                edge_weights=full_adj.edge_weight
                row,col=full_adj.edge_index
                full_adj.edge_weight=degrees_inv_sqrt[full_id[col]]*edge_weights*degrees_inv_sqrt[full_id[row]]
                ##############################################################
                
                
               
                sample_id, sample_adj = graph_no_self_loops.sample_adj(sample_id, size, replace=True)

                sample_adj = self._construct_propagation_matrix(sample_adj, sample_id, size)

                sample_adjs = [sample_adj] + sample_adjs #newly generated graph goes at the front of the list
                sample_ids = [sample_id] + sample_ids
                full_adjs = [full_adj] + full_adjs
                full_ids = [full_id] + full_ids

            return torch.tensor(idx), (sample_ids, sample_adjs), (full_ids, full_adjs)
        else:
            sample_id = torch.tensor(idx)
            full_id, full_adj = self.graph.sample_adj(sample_id, -1)
            ############# normalize full_adj to get P ################
            
            #symmetric normalization of the adjacency matrix
            degrees=self.graph.degrees() #diag of D matrix
            degrees_inv_sqrt=degrees.pow(-0.5)
            degrees_inv_sqrt[degrees_inv_sqrt==float("inf")]=0
            edge_weights=full_adj.edge_weight
            row,col=full_adj.edge_index
            full_adj.edge_weight=degrees_inv_sqrt[full_id[col]]*edge_weights*degrees_inv_sqrt[full_id[row]]
            return sample_id, full_id, full_adj
