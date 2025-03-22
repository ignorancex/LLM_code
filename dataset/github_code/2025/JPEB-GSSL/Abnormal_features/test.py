from model import NodeClassifier
from metrics import classification_multiclass_metrics
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
import torch_geometric.transforms as T
import torch.multiprocessing as tmp
import torch
import os
import random
from dotenv import load_dotenv


def abnormal_feature(ratio):
    g = torch.normal(mean=0, std=1, size=(1000, graph.x.size(1)))
    num_nodes_considered = int(ratio*1000)
    num_deactivated = 1000-num_nodes_considered

    mask = [0]*num_deactivated+[1]*num_nodes_considered
    random.shuffle(mask)
    mask_tensor = torch.tensor(mask)
    random_noise_matrix = mask_tensor.unsqueeze(1)*g

    graph.x[graph.test_mask] = random_noise_matrix

    return graph.x


@torch.no_grad()
def test(graph):
    _, probs = model(graph)

    acc, roc, f1 = classification_multiclass_metrics(
        probs[graph.test_mask], graph.y[graph.test_mask], dataset.num_classes)

    return acc.item(), roc.item(), f1.item()


def run(graph, ratio):

    graph.x = abnormal_feature(ratio)

    acc, _, _ = test(graph)

    print("Accuracy: ", 87.4-(acc*100))


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    load_dotenv('.env')

    inp_name = input("Enter dataset to be used: ")
    cora_path = os.getenv('Cora')
    pubmed_path = os.getenv('Pubmed')
    citeseer_path = os.getenv('CiteSeer')
    computers_path = os.getenv('Computers')
    photos_path = os.getenv('Photo')
    cs_path = os.getenv("CS")

    # ratio = eval(input('Enter ratio of test nodes to be distorted: '))
    if inp_name == 'cora':
        dataset = Planetoid(root=cora_path, name='Cora')
        # weights_path = os.getenv("cora_frozen")+"model_2000.pt"
        weights_path = os.getenv("cora_gmm_frozen")+"model_2000.pt"
        graph = dataset[0]

        split = T.RandomNodeSplit(num_test=1000, num_val=500)
        graph = split(graph)
    elif inp_name == 'pubmed':
        dataset = Planetoid(root=pubmed_path, name='PubMed')
        graph = dataset[0]
        # weights_path = os.getenv("pubmed_frozen")+"model_600.pt"
        weights_path = os.getenv("pubmed_gmm_frozen")+"model_3500.pt"
        split = T.RandomNodeSplit(num_test=1000, num_val=500)
        graph = split(graph)
    elif inp_name == 'citeseer':
        dataset = Planetoid(root=citeseer_path, name='CiteSeer')
        graph = dataset[0]
        # weights_path = os.getenv("citeseer_frozen")+"model_2000.pt"
        weights_path = os.getenv("citeseer_gmm_frozen")+"model_1200.pt"
        split = T.RandomNodeSplit(num_test=1000, num_val=500)
        graph = split(graph)
    elif inp_name == 'computers':
        dataset = Amazon(root=computers_path, name='Computers')
        graph = dataset[0]
        # weights_path = os.getenv("computer_frozen")+"model_600.pt"
        weights_path = os.getenv("computer_gmm_frozen")+"model_1250.pt"
        split = T.RandomNodeSplit(num_test=1000, num_val=0.1)
        graph = split(graph)
    elif inp_name == 'photos':
        dataset = Amazon(root=photos_path, name='Photo')
        graph = dataset[0]
        weights_path = os.getenv("photo_gmm_frozen")+"model_5000.pt"
        split = T.RandomNodeSplit(num_test=1000, num_val=0.1)
        graph = split(graph)
    elif inp_name == 'cs':
        dataset = Coauthor(root=cs_path, name='CS')
        graph = dataset[0]
        weights_path = os.getenv('CS_gmm_frozen')+"model_4000.pt"
        split = T.RandomNodeSplit(num_test=1000, num_val=0.1)
        graph = split(graph)
    # Add weights path here

    model = NodeClassifier(features=graph.x.size(1),
                           num_classes=dataset.num_classes)

    # Add weights path here
    ratios = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    for ratio in ratios:
        model.load_state_dict(torch.load(
            weights_path, weights_only=True), strict=True)
        model.eval()

        run(graph, ratio)
