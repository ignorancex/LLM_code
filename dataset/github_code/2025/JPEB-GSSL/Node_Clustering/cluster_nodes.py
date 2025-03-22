from model import ContextEncoder
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score
import torch.multiprocessing as tmp
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import torch
import os
from dotenv import load_dotenv


@torch.no_grad()
def eval_kmeans(graph):
    nmi_score = 0
    ari_score = 0
    for k in range(1, 11):
        z = model(graph.x, graph.edge_index).numpy()

        y_pred = kmeans_transform.fit_predict(
            z.astype('double'))
        y_true = graph.y.numpy()

        nmi_score += v_measure_score(y_true, y_pred)
        ari_score += adjusted_rand_score(y_true, y_pred)

    print(nmi_score/k, ari_score/k)


@torch.no_grad()
def cluster():
    z = model(graph.x, graph.edge_index).numpy()
    projected_2d = tsne_transform.fit_transform(z)
    plt.figure(figsize=(8, 6))
    # plt.contourf(xx, yy, z_kmeans, cmap=cmap_light, alpha=0.6)

    plt.scatter(projected_2d[:, 0], projected_2d[:, 1],
                c=list(graph.y.numpy()), s=50, edgecolor='k', cmap='viridis')

    plt.show()


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
        cmap_light = ListedColormap(
            ["#ADD8E6", "#90EE90", "#F08080", "#FFB6C1", "#FFFFE0", "#E0FFFF", "#D8BFD8"])
        dataset = Planetoid(root=cora_path, name='Cora')
        graph = dataset[0]
        weights_path = os.getenv("cora_encoder_GMM")+"model_100.pt"
    elif inp_name == 'pubmed':
        cmap_light = ListedColormap(['#ADD8E6', '#FFB6C1', '#90EE90'])
        dataset = Planetoid(root=pubmed_path, name='PubMed')
        graph = dataset[0]
        weights_path = os.getenv("pubmed_encoder_GMM")+"model_75.pt"
    elif inp_name == 'citeseer':
        cmap_light = ListedColormap(
            ["#ADD8E6", "#90EE90", "#F08080", "#FFB6C1", "#FFFFE0", "#D8BFD8"])
        dataset = Planetoid(root=citeseer_path, name='Citeseer')
        graph = dataset[0]
        weights_path = os.getenv("citeseer_encoder_GMM")+"model_150.pt"
    elif inp_name == 'computers':
        cmap_light = ListedColormap(['#ADD8E6', '#FFB6C1', '#90EE90', '#FFFFE0',
                                    '#E6E6FA', '#F08080', '#FFDAB9', '#D8BFD8', '#E0FFFF', '#FAFAD2'])
        dataset = Amazon(root=computers_path, name='Computers')
        graph = dataset[0]
        # weights_path = os.getenv("computer_encoder_2")+"model_500.pt"
        weights_path = os.getenv("computer_encoder_GMM")+"model_400.pt"
    elif inp_name == 'photos':
        cmap_light = ListedColormap(['#ADD8E6', '#FFB6C1', '#90EE90', '#FFFFE0', '#E6E6FA',
                                     '#F08080', '#FFDAB9', '#D8BFD8'])
        dataset = Amazon(root=photos_path, name='Photo')
        graph = dataset[0]
        # weights_path = os.getenv("photo_encoder_2")+"model_600.pt"
        weights_path = os.getenv("photo_encoder_GMM")+"model_125.pt"
    elif inp_name == 'cs':
        cmap_light = ListedColormap(['#ADD8E6', '#FFB6C1', '#90EE90', '#FFFFE0', '#E6E6FA',
                                     '#F08080', '#FFDAB9', '#D8BFD8'])
        dataset = Coauthor(root=cs_path, name='CS')
        graph = dataset[0]
        # weights_path = os.getenv("CS_encoder_2")+"model_1400.pt"
        weights_path = os.getenv("CS_encoder_GMM")+"model_100.pt"

    model = ContextEncoder(in_features=graph.x.size(1))
    model.load_state_dict(torch.load(
        weights_path, weights_only=True), strict=True)
    model.eval()

    tsne_transform = TSNE(
        n_components=2, learning_rate='auto', init='random', perplexity=80)
    kmeans_transform = KMeans(n_clusters=dataset.num_classes, random_state=0)

    eval_kmeans(graph)
    cluster()
