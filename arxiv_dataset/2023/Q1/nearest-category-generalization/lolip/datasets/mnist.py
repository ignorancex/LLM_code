from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from .feature_extract import extract_nn_feature, cusfet_extract

def dset_to_numpy(dset):
    loader = DataLoader(dset, batch_size=128, shuffle=False, num_workers=8)
    Xs, ys = [], []
    for x, y in tqdm(loader, desc="[dset_to_numpy]"):
        Xs.append(x.numpy().transpose(0, 2, 3, 1))
        ys.append(y.numpy())
    return np.concatenate(Xs, axis=0).astype(np.float32), np.concatenate(ys).reshape(-1)

def pathmnist_leave_one_out(to_left_out):
    from medmnist.dataset import PathMNIST
    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')

    transform = transforms.ToTensor()
    dset = PathMNIST(root='./data/medmnist', split='train', download=True, as_rgb=True, transform=transform)
    trnX, trny = dset_to_numpy(dset)
    dset = PathMNIST(root='./data/medmnist', split='test', download=True, as_rgb=True, transform=transform)
    tstX, tsty = dset_to_numpy(dset)
    rest = (trnX[trny==to_left_out], tstX[tsty==to_left_out],
            to_left_out*np.ones((trny==to_left_out).sum()), to_left_out*np.ones((tsty==to_left_out).sum()))
    trnX, tstX = trnX[trny != to_left_out], tstX[tsty != to_left_out]
    trny, tsty = trny[trny != to_left_out], tsty[tsty != to_left_out]
    trny[trny > to_left_out] -= 1
    tsty[tsty > to_left_out] -= 1

    return trnX, trny, tstX, tsty, rest

def dermamnist_leave_one_out(to_left_out):
    from medmnist.dataset import DermaMNIST
    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')

    transform = transforms.ToTensor()
    dset = DermaMNIST(root='./data/medmnist', split='train', download=True, as_rgb=True, transform=transform)
    trnX, trny = dset_to_numpy(dset)
    dset = DermaMNIST(root='./data/medmnist', split='test', download=True, as_rgb=True, transform=transform)
    tstX, tsty = dset_to_numpy(dset)
    rest = (trnX[trny==to_left_out], tstX[tsty==to_left_out],
            to_left_out*np.ones((trny==to_left_out).sum()), to_left_out*np.ones((tsty==to_left_out).sum()))
    trnX, tstX = trnX[trny != to_left_out], tstX[tsty != to_left_out]
    trny, tsty = trny[trny != to_left_out], tsty[tsty != to_left_out]
    trny[trny > to_left_out] -= 1
    tsty[tsty > to_left_out] -= 1

    return trnX, trny, tstX, tsty, rest

def fashion_leave_one_out(to_left_out):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    rest = (x_train[y_train==to_left_out], x_test[y_test==to_left_out],
            to_left_out*np.ones((y_train==to_left_out).sum()), to_left_out*np.ones((y_test==to_left_out).sum()))
    x_train, x_test = x_train[y_train != to_left_out], x_test[y_test != to_left_out]
    y_train, y_test = y_train[y_train != to_left_out], y_test[y_test != to_left_out]
    y_train[y_train > to_left_out] -= 1
    y_test[y_test > to_left_out] -= 1

    return x_train, y_train, x_test, y_test, rest

def mnist_leave_one_out(to_left_out):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    rest = (x_train[y_train==to_left_out], x_test[y_test==to_left_out],
            to_left_out*np.ones((y_train==to_left_out).sum()), to_left_out*np.ones((y_test==to_left_out).sum()))
    x_train, x_test = x_train[y_train != to_left_out], x_test[y_test != to_left_out]
    y_train, y_test = y_train[y_train != to_left_out], y_test[y_test != to_left_out]
    y_train[y_train > to_left_out] -= 1
    y_test[y_test > to_left_out] -= 1

    return x_train, y_train, x_test, y_test, rest

def mnistwo9resnet18():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
    x_train = extract_nn_feature(x_train, "resnet18")
    x_test = extract_nn_feature(x_test, "resnet18")

    rest = (x_train[y_train==9], x_test[y_test==9],
            9*np.ones((y_train==9).sum()), 9*np.ones((y_test==9).sum()))
    x_train, x_test = x_train[y_train != 9], x_test[y_test != 9]
    y_train, y_test = y_train[y_train != 9], y_test[y_test != 9]

    return x_train, y_train, x_test, y_test, rest

def mnistwo8no9():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    rest = (x_train[y_train==8], x_test[y_test==8],
            8*np.ones((y_train==8).sum()), 8*np.ones((y_test==8).sum()))
    x_train, x_test = x_train[np.logical_and(y_train != 9, y_train != 8)], x_test[np.logical_and(y_test != 8, y_test != 9)]
    y_train, y_test = y_train[np.logical_and(y_train != 9, y_train != 8)], y_test[np.logical_and(y_test != 8, y_test != 9)]

    return x_train, y_train, x_test, y_test, rest

def mnistwo9():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255
    rest = (x_train[y_train==9], x_test[y_test==9],
            9*np.ones((y_train==9).sum()), 9*np.ones((y_test==9).sum()))
    x_train, x_test = x_train[y_train != 9], x_test[y_test != 9]
    y_train, y_test = y_train[y_train != 9], y_test[y_test != 9]

    return x_train, y_train, x_test, y_test, rest

def smallmnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

    trnX, trny, tstX, tsty = [], [], [], []
    for i in np.unique(y_train):
        trnX.append(x_train[y_train==i][:20])
        trny.append(np.ones(20, dtype=np.int) * i)
        tstX.append(x_test[y_test==i][:20])
        tsty.append(np.ones(20, dtype=np.int) * i)

    return np.vstack(trnX), np.concatenate(trny), np.vstack(tstX), np.concatenate(tsty)
