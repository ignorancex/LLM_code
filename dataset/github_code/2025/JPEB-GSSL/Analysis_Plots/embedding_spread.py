from Downstream.model import ContextEncoder
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.nn import global_mean_pool
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
import torch


@torch.no_grad()
def store_embeddings(name):
    z = model(graph.x, graph.edge_index).numpy().flatten()
    # z_mean = global_mean_pool(z, batch=graph.batch).numpy().flatten()

    # Spread plots
    data_dict[name] = z


def plot(data_dict):
    plt.boxplot(data_dict.values(), tick_labels=data_dict.keys(), patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='darkblue'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='green'))
    plt.show()


if __name__ == '__main__':
    load_dotenv('.env')

    cora_path = os.getenv('Cora')
    pubmed_path = os.getenv('Pubmed')
    citeseer_path = os.getenv('CiteSeer')
    computers_path = os.getenv('Computers')
    photos_path = os.getenv('Photo')
    cs_path = os.getenv("CS")

    inp_names = ['Cora', 'Pubmed', 'Citeseer', 'Computers', 'Photos', 'CS']
    data_dict = dict()
    for inp_name in inp_names:

        if inp_name == 'Cora':

            dataset = Planetoid(root=cora_path, name='Cora')
            graph = dataset[0]
            weights_path = os.getenv("cora_encoder_GMM")+"model_100.pt"
        elif inp_name == 'Pubmed':

            dataset = Planetoid(root=pubmed_path, name='PubMed')
            graph = dataset[0]
            weights_path = os.getenv("pubmed_encoder_GMM")+"model_100.pt"
        elif inp_name == 'Citeseer':

            dataset = Planetoid(root=citeseer_path, name='Citeseer')
            graph = dataset[0]
            weights_path = os.getenv("citeseer_encoder_GMM")+"model_150.pt"
        elif inp_name == 'Computers':

            dataset = Amazon(root=computers_path, name='Computers')
            graph = dataset[0]
            # weights_path = os.getenv("computer_encoder_2")+"model_500.pt"
            weights_path = os.getenv("computer_encoder_GMM")+"model_400.pt"
        elif inp_name == 'Photos':

            dataset = Amazon(root=photos_path, name='Photo')
            graph = dataset[0]
            # weights_path = os.getenv("photo_encoder_2")+"model_600.pt"
            weights_path = os.getenv("photo_encoder_GMM")+"model_125.pt"
        elif inp_name == 'CS':

            dataset = Coauthor(root=cs_path, name='CS')
            graph = dataset[0]
            # weights_path = os.getenv("CS_encoder_2")+"model_1400.pt"
            weights_path = os.getenv("CS_encoder_GMM")+"model_100.pt"

        model = ContextEncoder(in_features=graph.x.size(1))
        model.load_state_dict(torch.load(
            weights_path, weights_only=True), strict=True)
        model.eval()

        store_embeddings(inp_name)
    plot(data_dict)
