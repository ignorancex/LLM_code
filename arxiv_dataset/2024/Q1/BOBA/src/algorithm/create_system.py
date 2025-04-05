from .Base import BaseServer
from .FedAvg import FedAvgServer
from .FedProx import FedProxServer
from .SGD import SGDServer


def create_system(client_datasets, server_datasets, test_dataset, args):
    algorithm = args.algorithm

    if algorithm == 'central':
        server = BaseServer(client_datasets, server_datasets, test_dataset, args)
    elif algorithm == 'fedavg':
        server = FedAvgServer(client_datasets, server_datasets, test_dataset, args)
    elif algorithm == 'fedprox':
        server = FedProxServer(client_datasets, server_datasets, test_dataset, args)
    elif algorithm == 'sgd':
        server = SGDServer(client_datasets, server_datasets, test_dataset, args)
    else:
        raise NotImplementedError

    return server