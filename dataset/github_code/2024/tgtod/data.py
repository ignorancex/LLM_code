import torch
import pandas as pd
from torch_geometric.data import (Data, HeteroData,
                                  InMemoryDataset, download_url)
from torch_geometric.datasets import EllipticBitcoinDataset


class EllipticTemporalDataset(EllipticBitcoinDataset):
    def __init__(
            self,
            root,
            transform=None,
            pre_transform=None,
            force_reload=False,
    ):
        super().__init__(root, transform, pre_transform, force_reload)

    @property
    def processed_file_names(self) -> str:
        return f'data.pt'

    def _process_df(self, feat_df, edge_df,
                    class_df):
        txId2t = feat_df.set_index('txId')['time_step']
        edge_df['time_step'] = edge_df['txId1'].map(txId2t)
        return feat_df, edge_df, class_df

    def process(self):
        feat_df = pd.read_csv(self.raw_paths[0], header=None)
        edge_df = pd.read_csv(self.raw_paths[1])
        class_df = pd.read_csv(self.raw_paths[2])

        columns = {0: 'txId', 1: 'time_step'}
        feat_df = feat_df.rename(columns=columns)

        feat_df, edge_df, class_df = self._process_df(
            feat_df,
            edge_df,
            class_df,
        )

        x = torch.from_numpy(feat_df.loc[:, 2:].values).float()

        # There exists 3 different classes in the dataset:
        # 0=licit,  1=illicit, 2=unknown
        mapping = {'unknown': 2, '1': 1, '2': 0}
        class_df['class'] = class_df['class'].map(mapping)
        y = torch.from_numpy(class_df['class'].values)

        mapping = {idx: i for i, idx in enumerate(feat_df['txId'].values)}
        edge_df['txId1'] = edge_df['txId1'].map(mapping)
        edge_df['txId2'] = edge_df['txId2'].map(mapping)
        edge_index = torch.from_numpy(edge_df[['txId1', 'txId2']].values)
        edge_index = edge_index.t().contiguous()
        edge_time = torch.from_numpy(edge_df['time_step'].values)

        # Timestamp based split:
        # train_mask: 1 - 25 time_step,
        # val_mask: 26 - 34 time_step,
        # test_mask: 35 - 49 time_step
        time_step = torch.from_numpy(feat_df['time_step'].values)
        train_mask = (time_step <= 25) & (y != 2)
        val_mask = (time_step > 25) & (time_step < 35) & (y != 2)
        test_mask = (time_step >= 35) & (y != 2)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask, edge_time=edge_time)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


class FiGraphDataset(InMemoryDataset):
    url = 'https://github.com/XiaoguangWang23/FiGraph/raw/main/data'

    def __init__(
            self,
            root,
            to_homo=True,
            station=True,
            transform=None,
            pre_transform=None,
    ):
        self.to_homo = to_homo
        self.station = station
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], HeteroData)

    @property
    def raw_file_names(self):
        return [
            'edges2014.csv', 'edges2015.csv', 'edges2016.csv', 'edges2017.csv',
            'edges2018.csv', 'edges2019.csv', 'edges2020.csv', 'edges2021.csv',
            'edges2022.csv', 'ListedCompanyFeatures.csv'
        ]

    @property
    def processed_file_names(self) -> str:
        if self.station:
            return 'data.pt'
        else:
            return 'data_ns.pt'

    def download(self) -> None:
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self) -> None:
        node_df = pd.read_csv(self.raw_paths[-1])
        feat_mapping = {idx: i for i, idx in enumerate(node_df['nodeID'].values)}

        dfs = []
        for i, path in enumerate(self.raw_paths[:-1]):
            edge_df = pd.read_csv(path, header=None)
            edge_df['edge_time'] = i
            dfs.append(edge_df)

        edge_df = pd.concat(dfs, ignore_index=True)
        columns = {0: 'src', 1: 'dst', 2: 'edge_type'}
        edge_df = edge_df.rename(columns=columns)
        nodes = pd.concat((edge_df['src'], edge_df['dst']),
                          ignore_index=True).drop_duplicates()
        node_map = {}
        for ntype in ['L', 'U', 'H', 'R', 'A']:
            type_nodes = nodes[nodes.str.startswith(ntype)]
            node_map[ntype] = {idx: i for i, idx in enumerate(type_nodes.values)}

        edge_df['stype'] = edge_df['src'].str[0]
        edge_df['dtype'] = edge_df['dst'].str[0]

        edge_df['src'] = edge_df.apply(lambda row:
                                       node_map[row['stype']][row['src']], axis=1)
        edge_df['dst'] = edge_df.apply(lambda row:
                                       node_map[row['dtype']][row['dst']], axis=1)

        data = HeteroData()
        if self.station:
            indices = torch.zeros(len(node_map['L'])).long()
            for k, v in feat_mapping.items():
                indices[node_map['L'][k]] = v

            x = torch.from_numpy(node_df.iloc[:, 2:-1].values).float()[indices]
            y = torch.from_numpy(node_df['Label'].values).long()[indices]

            idx = torch.randperm(x.shape[0])
            train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
            val_mask = torch.zeros(x.shape[0], dtype=torch.bool)
            test_mask = torch.zeros(x.shape[0], dtype=torch.bool)
            train_mask[idx[:int(x.shape[0] * 0.8)]] = True
            val_mask[idx[int(x.shape[0] * 0.8):int(x.shape[0] * 0.9)]] = True
            test_mask[idx[int(x.shape[0] * 0.9):]] = True
            data['L'].train_mask = train_mask
            data['L'].val_mask = val_mask
            data['L'].test_mask = test_mask
        else:
            x, y = [], []
            train_masks, val_masks, test_masks = [], [], []
            group_node = node_df.groupby('Year')
            num_listed = len(node_map['L'])
            num_time = len(group_node)
            for i, (year, group) in enumerate(group_node):
                year_x = torch.zeros((num_listed, group.columns.size - 3)).float()
                year_y = torch.full((num_listed,), -1).long()
                indices = group['nodeID'].map(node_map['L']).values
                year_x[indices] = torch.from_numpy(group.iloc[:, 2:-1].values).float()
                year_y[indices] = torch.from_numpy(group.iloc[:, -1].values).long()
                x.append(year_x)
                y.append(year_y)
                label_mask = year_y != -1
                if i < num_time - 2:
                    train_mask = label_mask
                    val_mask = torch.zeros(num_listed, dtype=torch.bool)
                    test_mask = torch.zeros(num_listed, dtype=torch.bool)
                elif i == num_time - 2:
                    train_mask = torch.zeros(num_listed, dtype=torch.bool)
                    val_mask = label_mask
                    test_mask = torch.zeros(num_listed, dtype=torch.bool)
                else:
                    train_mask = torch.zeros(num_listed, dtype=torch.bool)
                    val_mask = torch.zeros(num_listed, dtype=torch.bool)
                    test_mask = label_mask
                train_masks.append(train_mask)
                val_masks.append(val_mask)
                test_masks.append(test_mask)

            x = torch.stack(x, dim=0).transpose(0, 1)
            y = torch.stack(y, dim=0).transpose(0, 1)
            data['L'].train_mask = torch.stack(train_masks, dim=0).transpose(0, 1)
            data['L'].val_mask = torch.stack(val_masks, dim=0).transpose(0, 1)
            data['L'].test_mask = torch.stack(test_masks, dim=0).transpose(0, 1)

        data['L'].x, data['L'].y = x, y
        for ntype in ['L', 'U', 'H', 'R', 'A']:
            data[ntype].num_nodes = len(node_map[ntype])
        group_edge = edge_df.groupby(['stype', 'edge_type', 'dtype'])
        for name, group in group_edge:
            edge_index = torch.from_numpy(group[['src', 'dst']].values)
            data[name].edge_index = edge_index.t().contiguous()
            data[name].edge_time = torch.from_numpy(group['edge_time'].values)

        if self.to_homo:
            data = data.to_homogeneous()
            data.x[data.x.isnan()] = 0

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        return 2
