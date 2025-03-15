from model import NodeClassifier
from metrics import classification_multiclass_metrics
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
import torch_geometric.transforms as T
import torch.multiprocessing as tmp
from torch import nn
import torch
import os
from dotenv import load_dotenv


@torch.no_grad()
def test(graph):
    _, probs = model(graph)

    acc, roc, f1 = classification_multiclass_metrics(
        probs[graph.test_mask], graph.y[graph.test_mask], dataset.num_classes)

    return acc.item(), roc.item(), f1.item()


def run(graph):  # Suggested for Amazon Photos, Computers, Coauthor CS
    res = list()
    for e in range(10):
        if inp_name in ['cora', 'pubmed', 'citeseer']:
            split_function = T.RandomNodeSplit(
                num_val=500, num_test=1000)
            graph = split_function(graph)
        else:
            split_function = T.RandomNodeSplit(
                num_val=0.1, num_test=0.2)  # Split each time randomly
            graph = split_function(graph)
        acc, _, _ = test(graph)
        res.append(acc)

        if (e+1) % 10 == 0:
            print(e+1, " runs completed!!!!")

    res = torch.tensor(res)
    u, s = torch.mean(res), torch.std(res)  # with degree of error

    print("Mean Accuracy: ", u.item()*100)
    print("Std. Accuracy: ", s.item()*100)


def single_run(graph):
    if inp_name in ['cora', 'pubmed', 'citeseer']:
        split_function = T.RandomNodeSplit(
            num_val=500, num_test=1000)
        graph = split_function(graph)
    else:
        graph = split_function(graph)
        split_function = T.RandomNodeSplit(
            num_val=0.1, num_test=0.2)  # Split each time randomly
        graph = split_function(graph)

    acc, roc, f1 = test(graph)

    print("Accuracy: ", acc*100)
    print("AUCROC: ", roc)
    print("F1: ", f1)


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

    if inp_name == 'cora':
        dataset = Planetoid(root=cora_path, name='Cora')
        # weights_path = os.getenv("cora_frozen")+"model_2000.pt"
        weights_path = os.getenv("cora_gmm_frozen")+"model_2000.pt"
        graph = dataset[0]
    elif inp_name == 'pubmed':
        dataset = Planetoid(root=pubmed_path, name='PubMed')
        graph = dataset[0]
        # weights_path = os.getenv("pubmed_frozen")+"model_600.pt"
        weights_path = os.getenv("pubmed_gmm_frozen")+"model_3500.pt"
    elif inp_name == 'citeseer':
        dataset = Planetoid(root=citeseer_path, name='CiteSeer')
        graph = dataset[0]
        # weights_path = os.getenv("citeseer_frozen")+"model_2000.pt"
        weights_path = os.getenv("citeseer_gmm_frozen")+"model_1200.pt"
    elif inp_name == 'computers':
        dataset = Amazon(root=computers_path, name='Computers')
        graph = dataset[0]
        # weights_path = os.getenv("computer_frozen")+"model_600.pt"
        weights_path = os.getenv("computer_gmm_frozen")+"model_1250.pt"
    elif inp_name == 'photos':
        dataset = Amazon(root=photos_path, name='Photo')
        graph = dataset[0]
        weights_path = os.getenv("photo_gmm_frozen")+"model_5000.pt"
    elif inp_name == 'cs':
        dataset = Coauthor(root=cs_path, name='CS')
        graph = dataset[0]
        weights_path = os.getenv('CS_gmm_frozen')+"model_4000.pt"
    model = NodeClassifier(features=graph.x.size(1),
                           num_classes=dataset.num_classes)

    # Add weights path here
    model.load_state_dict(torch.load(
        weights_path, weights_only=True), strict=True)
    model.eval()

    run(graph)
