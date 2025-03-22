import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from sklearn import metrics
from haven import haven_utils as hu


def get_dataset(dataset_name, train_flag, datadir):
    if dataset_name == "mnist":
        dataset = torchvision.datasets.MNIST(datadir, train=train_flag,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,))
                               ]))

    if dataset_name == "cifar10":
        transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR10(
            root=datadir,
            train=train_flag,
            download=True,
            transform=transform_function)

    if dataset_name == "cifar100":
        transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR100(
            root=datadir,
            train=train_flag,
            download=True,
            transform=transform_function)

    if dataset_name in ["mushrooms", "w8a",
                        "rcv1", "ijcnn","a1a","a2a","a3a","w1a", "phishing"]:

        sigma_dict = {"mushrooms": 0.5,
                      "w8a":20.0,
                      "rcv1":0.25 ,
                      "ijcnn":0.05,
                      "a1a": 1.0,
                      "a2a": 1.0,
                      "a3a": 1.0,
                      "w1a": 1.0,
                      "phishing":1.0}

        X, y = load_libsvm(dataset_name, data_dir=datadir)

        labels = np.unique(y)

        y[y==labels[0]] = 0
        y[y==labels[1]] = 1
		# TODO: (amishkin) splits = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=9513451)
        splits = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        X_train, X_test, Y_train, Y_test = splits


        if train_flag:
            # fname_rbf = "%s/rbf_%s_train.pkl" % (datadir, dataset_name)

            # if os.path.exists(fname_rbf):
            #     k_train_X = hu.load_pkl(fname_rbf)
            # else:
            k_train_X = rbf_kernel(X_train, X_train, sigma_dict[dataset_name])
                # hu.save_pkl(fname_rbf, k_train_X)

            X_train = k_train_X
            X_train = torch.FloatTensor(X_train)
            Y_train = torch.FloatTensor(Y_train)

            dataset = torch.utils.data.TensorDataset(X_train, Y_train)
            return dataset

        else:
            # fname_rbf = "%s/rbf_%s_test.pkl" % (datadir, dataset_name)
            # if os.path.exists(fname_rbf):
            #     k_test_X = hu.load_pkl(fname_rbf)
            # else:
            k_test_X = rbf_kernel(X_test, X_train, sigma_dict[dataset_name])
                # hu.save_pkl(fname_rbf, k_test_X)

            X_test = k_test_X
            X_test = torch.FloatTensor(X_test)
            Y_test = torch.FloatTensor(Y_test)

            dataset = torch.utils.data.TensorDataset(X_test, Y_test)
            return dataset

# ===========================================================
# Helpers
import os
import urllib

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from torchvision.datasets import MNIST


LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"rcv1"       : "rcv1_train.binary.bz2",
                      "mushrooms"  : "mushrooms",
                      "ijcnn"      : "ijcnn1.tr.bz2",
                      "w8a"        : "w8a",
                      "a1a": "a1a",
                      "a2a": "a2a",
                      "a3a":"a3a",
                      "w1a":"w1a",
                      "news": "news20.binary.bz2",
                      "covtype":"covtype.libsvm.binary.scale.bz2",
                      "phishing": "phishing"}



def load_libsvm(name, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(data_dir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    return X, y
#-------------------------------------------------------------------------------------
def rbf_kernel(A, B, sigma=1.0):
    distsq = np.square(metrics.pairwise.pairwise_distances(A, B, metric="euclidean"))
    K = np.exp(-1 * distsq/(2*sigma**2))
    return K
