import numpy as np
import torch as th
from torch.utils.data import Dataset
from os.path import join
from sklearn.datasets import olivetti_faces
from PIL import Image


class OlivettiFaces(Dataset):

    def __init__(self, data_dir, split, transform):
        super().__init__()
        data, targets = load_olivetti_faces(data_dir, split)
        self.data = th.from_numpy(data)
        self.targets = th.from_numpy(targets).long()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.data[i]
        img = Image.fromarray(img.numpy(), mode='L')
        img = self.transform(img)
        target = self.targets[i]
        return img, target


def load_olivetti_faces(data_dir, split="train"):
    assert split == "train" or split == "val" or split == "trainval" or split == "test"

    faces = olivetti_faces.fetch_olivetti_faces(
        data_home=join(data_dir),
        shuffle=True,
        random_state=100,
        download_if_missing=True
    )

    X = np.uint8(faces.images * 255.0)
    Y = faces.target

    n_tr = 5  # number of samples for each class in the training set
    n_va = 2  # number of samples for each class in the validation set
    n_te = 3  # number of samples for each class in the test set

    if split == "train":
        X = np.concatenate([X[np.where(Y == c)[0][:n_tr]] for c in range(40)])
        Y = np.concatenate([np.ones([n_tr], 'uint8') * c for c in range(40)])
    elif split == "val":
        X = np.concatenate([X[np.where(Y == c)[0][n_tr:n_va]] for c in range(40)])
        Y = np.concatenate([np.ones([n_va], 'uint8') * c for c in range(40)])
    elif split == "trainval":
        X = np.concatenate([X[np.where(Y == c)[0][:n_tr+n_va]] for c in range(40)])
        Y = np.concatenate([np.ones([n_tr+n_va], 'uint8') * c for c in range(40)])
    else: # test
        X = np.concatenate([X[np.where(Y == c)[0][-n_te:]] for c in range(40)])
        Y = np.concatenate([np.ones([n_te], 'uint8') * c for c in range(40)])

    assert X.shape[0] == Y.shape[0]
    return X, Y


