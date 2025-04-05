import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm
from torch_geometric.utils.convert import from_networkx
import itertools as it


from . import util_data

# print(f"Torch version: {torch.__version__}")
# print(f"Cuda available: {torch.cuda.is_available()}")
# print(f"Torch geometric version: {torch_geometric.__version__}")

def return_data_from_graph_datasets(dataset, index_traindata, num_val):
    data = {}

    data["train"] = dataset['train'].raw_data.iloc[index_traindata[num_val:]]
    data["val"] = dataset['train'].raw_data.iloc[index_traindata[:num_val]]
    data["test"] = dataset['test'].raw_data

    data["y_train"] = torch.tensor((data["train"].iloc[:,-1:]).values)
    data["y_val"] = torch.tensor((data["val"].iloc[:,-1:]).values)
    data["y_test"] = torch.tensor((data['test'].iloc[:,-1:]).values)

    data["x_train"] = torch.tensor((dataset['train'].raw_data.iloc[index_traindata].iloc[num_val:,:-1]).values)
    data["x_val"] = torch.tensor((dataset['train'].raw_data.iloc[index_traindata].iloc[:num_val,:-1]).values)
    data["x_test"] = torch.tensor((dataset['test'].raw_data.iloc[:,:-1]).values)

    return data

class GraphDataset(Dataset):
    def __init__(self, root, filename, subset_cols, graph_type="all", test=False, opposite=True):
        self.graph_type = graph_type
        self.test = test
        self.opposite = opposite

        self.subset_cols = subset_cols
        self.subset_str = util_data.subset_cols_to_str(subset_cols,False)
        if graph_type=="subset":
            if self.opposite:
                self.subset_keys_for_graph = np.array(list(util_data.all_keys.difference(subset_cols)))
            else:
                self.subset_keys_for_graph = np.array(subset_cols)
        else:
            self.subset_keys_for_graph = np.array(list(util_data.all_keys))

        self.subset_rows_for_graph = np.array([x for x in self.subset_keys_for_graph if x in util_data.efficacy_table.index.values])
        self.subset_cols_for_graph = np.array([x for x in self.subset_keys_for_graph if x in util_data.efficacy_table.columns])

        self.filename = filename

        super().__init__(root)

        if os.path.isdir(self.get_save_folder()):
            self.load_all_data()

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered."""
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        save_folder = self.get_save_folder_inside_processed()

        return [os.path.join(save_folder,"data.pt")]

    def download(self):
        pass

    def process(self):
        #print("PROCESSING")
        self.get_raw_data()

        #if graph_type is subset, divide train and test, and process only train (test is the same as "all" case)
        #otherwise processing is done for all samples, then are differently loaded if train or test
        if self.graph_type == "subset":
            self.split_train_test()
        df = self.raw_data

        #subset to df keys
        self.subset_keys_for_graph = np.array([x for x in self.subset_keys_for_graph if x in self.raw_data.columns])
        self.subset_rows_for_graph = np.array([x for x in self.subset_rows_for_graph if x in self.subset_keys_for_graph])
        self.subset_cols_for_graph = np.array([x for x in self.subset_cols_for_graph if x in self.subset_keys_for_graph])

        save_folder = self.get_save_folder()
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        orig_g = nx.DiGraph()
        for i, t in enumerate(self.subset_keys_for_graph):
            orig_g.add_node(i, x=[0, 0])

        for i1, t1 in enumerate(self.subset_keys_for_graph):
            if t1 in self.subset_rows_for_graph:
                for i2, t2 in enumerate(self.subset_keys_for_graph):
                    if t2 in self.subset_cols_for_graph:
                        w = util_data.efficacy_table.loc[t1, t2]
                        if not np.isnan(w):
                            orig_g.add_edge(i1, i2, weight=w)

        self.data = []
        app = df.loc[:,self.subset_keys_for_graph].values

        if self.graph_type == "sample":
            app_r, app_c = np.where(app>0)
            app2 = np.split(app_c, np.flatnonzero(app_r[1:] > app_r[:-1])+1)

        for i, (_,row) in tqdm(enumerate(df.iterrows()), total=df.shape[0], position=0, leave=True):
            attr_dict = dict(zip(range(len(self.subset_keys_for_graph)),[{"x":y} for y in app[i,:]]))
            nx.set_node_attributes(orig_g, attr_dict)

            if self.graph_type == "sample":
                g = orig_g.subgraph(app2[i])
            else:
                g = orig_g

            graph_data_p = from_networkx(g, group_node_attrs=["x"], group_edge_attrs=["weight"])

            self.data.append(Data(x=torch.tensor(graph_data_p.x, dtype=torch.float),
                        edge_index=torch.tensor(graph_data_p.edge_index, dtype=torch.long).view(2, -1),
                        y=torch.tensor([row[-1]], dtype=torch.int64),
                        edge_attr=torch.tensor(graph_data_p.edge_attr, dtype=torch.float).reshape(-1, 1)))
            
        torch.save(self.data, os.path.join(save_folder, 'data.pt'))

    def get_raw_data(self):
        raw_data_loc = os.path.join(self.raw_dir,self.filename+".csv")
        print("RAW DATA LOC",raw_data_loc)
        self.raw_data = pd.read_csv(raw_data_loc)

    def get_save_folder(self, graph_type = None):
        save_folder = os.path.join(self.processed_dir, self.get_save_folder_inside_processed(graph_type))
        return save_folder

    def get_save_folder_inside_processed(self, graph_type = None):
        graph_type = self.graph_type if graph_type is None else graph_type
        save_folder = os.path.join(self.filename, graph_type)
        if self.graph_type=="subset":
            save_folder = os.path.join(save_folder,self.subset_str)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        return save_folder

    def len(self):
        save_folder = self.get_save_folder()
        if not os.path.isdir(save_folder):
            print("¡SAVE FOLDER NOT FOUND!")
            return None
        return len(self.data)

    def load_all_data(self):
        print("LOADING ALL DATA")

        if self.graph_type == "subset" and self.test==True:
            self.graph_type = "all"
        save_folder = self.get_save_folder()
            
        self.data = torch.load(os.path.join(save_folder, 'data.pt'))
        
        self.get_raw_data()
        train_rows, test_rows = self.split_train_test()

        if self.graph_type != "subset":
            self.data = list(it.compress(self.data,[train_rows, test_rows][self.test]))

    def split_train_test(self):
        train_rows, test_rows = util_data.get_subset_rows(self.raw_data, self.subset_cols, self.opposite)
        if not np.any(test_rows): #test not already divided
            num_samples = len(train_rows)
            num_test = int(num_samples * 0.2)
            train_rows[:-num_test] = False
            test_rows[-num_test:] = True
        self.raw_data = self.raw_data.loc[[train_rows, test_rows][self.test]]
        return train_rows, test_rows

    def get(self, idx):
        return self.data[idx]
        
    def y(self):
        return torch.cat([self.get(i).y for i in range(len(self.data))])
        

class EnsembleDataset(GraphDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, idx):
        a = torch.tensor(self.raw_data.iloc[idx,:-1].values)
        b = self.data[idx]
        return a,b

###GNN
def load_data_GNN(data_folder,filename,graph_type,subset_cols):
    dataset = {}
    for sub_set in ["train","test"]:
        dataset[sub_set] = GraphDataset(root = data_folder,
                                                    filename = filename,
                                                    subset_cols = subset_cols,
                                                    graph_type = graph_type,
                                                    test = sub_set=="test")
    return dataset

def load_data_MIX(data_folder,filename,graph_type,subset_cols):
    dataset = {}
    for sub_set in ["train","test"]:
        dataset[sub_set] = EnsembleDataset(root = data_folder,
                                                    filename = filename,
                                                    subset_cols = subset_cols,
                                                    graph_type = graph_type,
                                                    test = sub_set=="test")
    return dataset
