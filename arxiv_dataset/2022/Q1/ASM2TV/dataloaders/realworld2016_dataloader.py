import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


class har2016(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, is_single_sensor):
        print("loading HAR RealWorld 2016 Dataset..." + self.name())
        self.data = []
        self.labels = []
        if is_single_sensor:
            task_ids = [1, 3, 8, 9, 10, 11, 12, 13, 15]
            # task_ids = [1, 3, 8, 9, 10]
            data_path = os.path.join(dataroot, 'single_sensor_acc')
        else:
            task_ids = [3, 8, 9, 10, 11, 12, 13, 15]
            # task_ids = [3, 8]
            data_path = os.path.join(dataroot, 'multi_sensors_acc_gyr_mag')
        for task_id in task_ids:
            task_data_path = os.path.join(data_path, mode, 'proband{}_data.npy'.format(task_id))
            task_label_path = os.path.join(data_path, mode, 'proband{}_label.npy'.format(task_id))
            task_data = np.load(task_data_path, allow_pickle=True)
            task_label = np.load(task_label_path, allow_pickle=True)
            self.data.append(task_data[np.newaxis, ...])
            self.labels.append(task_label[np.newaxis, ...])
        self.data = np.concatenate(self.data, axis=0) # Shape: [num_tasks, num_samples, num_views, time_len, dim]
        self.labels = np.concatenate(self.labels, axis=0) # Shape: [num_tasks, num_samples]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        # print(self.data[0].shape)

    def __len__(self):
        return len(self.data[1]) # num_samples

    def __getitem__(self, index):
        return self.data[:, index, ...], self.labels[:, index]
    
    def name(self):
        return "har2016"


class har2016_general(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, is_single_sensor):
        print("loading HAR RealWorld 2016 Dataset for General Model..." + self.name())
        self.data = []
        self.labels = []
        if is_single_sensor:
            task_ids = [1, 3, 8, 9, 10, 11, 12, 13, 15]
            # task_ids = [1, 3, 8, 9, 10]
            data_path = os.path.join(dataroot, 'single_sensor_acc')
        else:
            task_ids = [3, 8, 9, 10, 11, 12, 13, 15]
            # task_ids = [3, 8]
            data_path = os.path.join(dataroot, 'multi_sensors_acc_gyr_mag')
        for task_id in task_ids:
            task_data_path = os.path.join(data_path, mode, 'proband{}_data.npy'.format(task_id))
            task_label_path = os.path.join(data_path, mode, 'proband{}_label.npy'.format(task_id))
            task_data = np.load(task_data_path, allow_pickle=True)
            task_label = np.load(task_label_path, allow_pickle=True)
            self.data.append(task_data)
            self.labels.append(task_label)
        self.data = np.concatenate(self.data, axis=0) # Shape: [num_tasks * num_samples, num_views, time_len, dim]
        self.labels = np.concatenate(self.labels, axis=0) # Shape: [num_tasks * num_samples]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        # print(self.data[0].shape)

    def __len__(self):
        return len(self.data) # num_samples

    def __getitem__(self, index):
        return self.data[index, ...], self.labels[index]
    
    def name(self):
        return "har2016-general"



if __name__ == '__main__':
    trainset_sup = har2016('./data/realworld2016_dataset', mode='train-sup', is_single_sensor=False)
    trainset_unsup = har2016('./data/realworld2016_dataset', mode='train-unsup', is_single_sensor=False)
    trainloader_sup = DataLoader(trainset_sup, batch_size=16, drop_last=False, num_workers=2, shuffle=True)
    trainloader_unsup = DataLoader(trainset_unsup, batch_size=64, drop_last=False, num_workers=2, shuffle=True)
    # trainloader_unsup = repeat_dataloader(trainloader_unsup)
    tk0 = tqdm(trainloader_sup, smoothing=0, mininterval=1.0)
    for i, ((sup_data, sup_labels), (unsup_data, unsup_labels)) in enumerate(zip(tk0, trainloader_unsup)):
        # unsup_data, unsup_labels = next(trainloader_unsup)
        bsz, num_tasks, num_views, seq_len, sensor_dim = sup_data.size()
        print(i, sup_data.size(), unsup_data.size(), sup_labels.size())
        # break
    # data, label = har_dataset[0]
    # print(len(har_dataset))
    # print(har_dataset.labels.size())
    # print(data.size(), data.dtype, label.size(), label.dtype) # shape: [num_tasks, num_views, window_size, dim]
    # from sklearn.metrics import classification_report
    # targets = [1, 2, 2, 1, 1, 3, 4, 5, 6, 6, 6]
    # preds = [1, 1, 2, 2, 1, 3, 4, 5, 5, 6, 6]
    # m = classification_report(targets, preds, output_dict=True)
    # print(m['weighted avg']['f1-score'])
    # x = torch.tensor([1., 2., 3.])
    # y = torch.exp(x)
    # print(y)