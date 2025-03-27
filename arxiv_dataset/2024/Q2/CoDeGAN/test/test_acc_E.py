import sys
import torch
import numpy as np
from dataloader import get_dataloader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI


# mark features by K-means
def k_means(f, num_class):
    label_kmeans = KMeans(n_clusters=num_class, random_state=0).fit_predict(
        f.data.cpu().numpy()
    )
    return torch.from_numpy(label_kmeans)


def calculate_acc(label_kmeans, labels, num, num_class):
    acc = 0
    label_num = []
    for k in range(num_class):
        idx_k = torch.eq(labels, k).float()
        label_num.append(int(torch.sum(idx_k)))

    for i in range(num_class):
        index = torch.eq(label_kmeans, i).float()

        n = []
        j_ = 0
        for j in label_num:
            n.append(torch.sum(index[j_ : (j_ + j)]).view(-1, 1))
            j_ += j
        acc += torch.max(torch.cat(n, dim=0))
    acc = acc / num
    return acc


def test_acc_E(E, epoch_num, save_path, dataset_name="cifar10"):
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = next(E.parameters()).device
    num = 10000
    batch_size = 50
    num_class = 10 if dataset_name != "coil20" else 20
    cal_times = 5
    testset = get_dataloader(batch_size, dataset_name=dataset_name, train=False)
    E.eval()

    # encode test images to features
    features = []
    labels = []
    for idx, (data, label) in enumerate(testset):
        labels.append(label)
        feature = E(data.to(device))
        if isinstance(feature, tuple):
            feature = feature[-1]
        features.append(feature.data)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0).view(-1)
    labels, idx = torch.sort(labels.data)
    features = features.data[idx]

    # calculate acc, nmi, ari
    mean_acc = []
    mean_nmi = []
    mean_ari = []
    for _ in range(cal_times):
        label_kmeans = k_means(features, num_class)  # [num]
        acc = calculate_acc(label_kmeans, labels, num, num_class)
        nmi = NMI(labels.data.cpu().numpy(), label_kmeans.data.cpu().numpy())
        ari = ARI(labels.data.cpu().numpy(), label_kmeans.data.cpu().numpy())
        mean_acc.append(acc)
        mean_nmi.append(nmi)
        mean_ari.append(ari)
    mean_acc = torch.mean(torch.tensor(mean_acc))
    mean_nmi = np.mean(mean_nmi)
    mean_ari = np.mean(mean_ari)

    # write test result to 'test_result.txt'
    list_strings = []
    current_losses = {"mean_acc": mean_acc, "mean_nmi": mean_nmi, "mean_ari": mean_ari}
    for eval_name, eval_value in current_losses.items():
        list_strings.append("%s = %.2f " % (eval_name, eval_value))
    full_string = " ".join(list_strings)
    with open(save_path + "test_result.txt", "a") as f:
        f.write("epoch = {} times={} E {} \n".format(epoch_num, cal_times, full_string))
