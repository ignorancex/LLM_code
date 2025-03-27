import argparse
from torch_geometric.datasets import DGraphFin
import torch_geometric.transforms as T

from data import EllipticTemporalDataset, FiGraphDataset


def arg_parser():
    parser = argparse.ArgumentParser(description='TGTOD')
    parser.add_argument('--model', type=str, default='tgtod')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='dgraph')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batchnorm', type=bool, default=False)
    parser.add_argument('--l2', type=float, default=5e-7)
    parser.add_argument('--timeslot', type=int, default=10)
    parser.add_argument('--num_parts', type=int, default=64)
    parser.add_argument('--use_cformer', type=bool, default=True)
    parser.add_argument('--graph_weight', type=float, default=0.8)
    parser.add_argument('--station', type=bool, default='True')
    args = parser.parse_args()
    print(args)
    return args


def load_data(name, station=True):
    if station:
        if name == 'dgraph':
            dataset = DGraphFin(root='./data/dgraph', transform=T.ToUndirected(reduce='min'))
        elif name == 'elliptic':
            dataset = EllipticTemporalDataset(root='./data/elliptic', transform=T.ToUndirected(reduce='min'))
        elif name == 'figraph':
            dataset = FiGraphDataset(root='./data/figraph', transform=T.ToUndirected(reduce='min'))
        else:
            raise ValueError(f'Unknown dataset in stationary setting: {name}')
    else:
        if name == 'figraph':
            dataset = FiGraphDataset(root='./data/figraph', station=station, transform=T.ToUndirected(reduce='min'))
        else:
            raise ValueError(f'Unknown dataset in non-stationary setting: {name}')
    return dataset
