import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset
from tslearn.datasets import UCR_UEA_datasets



def z_normalization(array):
    # transformed data have zero mean and in unit of standard deviation -- each of the input
    # Input Shape (length, 1, num)
    avg = np.mean(array, axis=-1)
    std = np.std(array, axis=-1)

    return (array - avg[:, :, np.newaxis]) / std[:, :, np.newaxis]


def convert_to_label_if_one_hot(input_data):
    if isinstance(input_data, np.ndarray):
        if input_data.ndim > 1:
            return np.argmax(input_data, axis=1)
        else:
            return input_data
    if isinstance(input_data, list):
        if isinstance(input_data[0], list):
            return [np.argmax(i) for i in input_data]
        else:
            return np.array(input_data)

class CustomDataset(Dataset):
    def __init__(self, feature, target):
        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        item = self.feature[idx]
        label = self.target[idx]

        return item, label


def generate_loader(train_x, test_x, train_y, test_y, batch_size_train=16, batch_size_test=32):
    train_dataset = CustomDataset(train_x.astype(np.float64), train_y.astype(np.int64))
    test_dataset = CustomDataset(test_x.astype(np.float64), test_y.astype(np.int64))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader





def read_UCR_UEA(dataset, UCR_UEA_dataloader):
    if UCR_UEA_dataloader is None:
        UCR_UEA_dataloader = UCR_UEA_datasets()
    X_train, train_y, X_test, test_y = UCR_UEA_dataloader.load_dataset(dataset)
    if X_train is None:
        print(f"{dataset} could not load correctly")
        return None, None, None, None, None
    train_x = X_train.reshape(-1, X_train.shape[-1], X_train.shape[-2])
    test_x = X_test.reshape(-1, X_train.shape[-1], X_train.shape[-2])
    enc1 = sklearn.preprocessing.OneHotEncoder(sparse_output=False).fit(train_y.reshape(-1, 1))
    train_y = enc1.transform(train_y.reshape(-1, 1))
    test_y = enc1.transform(test_y.reshape(-1, 1))
    return train_x, test_x, train_y, test_y, enc1


