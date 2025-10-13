import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class KinFeaturetDataset(Dataset):

    def __init__(self, csv_metadata_file, features_file, isTrain=True, fold_num=1):
        self.isTrain = isTrain
        self.fold_num = fold_num

        csv_data = pd.read_csv(csv_metadata_file)
        if isTrain:
            csv_data = csv_data[csv_data['fold'] != fold_num]
        else:
            csv_data = csv_data[csv_data['fold'] == fold_num]

        csv_data = csv_data.sample(frac=1.0)
        csv_data = csv_data.sample(frac=1.0)
        csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                    'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values)}
        self.df_dataset = pd.DataFrame(csv_data)

        data = np.load(features_file)
        self.features_dataset_array: np.ndarray = data['arr_0']
        features_dataset_labels: np.ndarray = data['arr_1']
        self.features_dataset_labels:list = features_dataset_labels.tolist()

    def __len__(self):
        return len(self.df_dataset.index)

    def __getitem__(self, idx):
        file_parent = self.df_dataset.loc[idx, 'p1']
        file_child = self.df_dataset.loc[idx, 'p2']
        label = int(self.df_dataset.loc[idx, 'label'])
        
        idx1 = self.features_dataset_labels.index(file_parent)
        parent_features = self.features_dataset_array[idx1, :, :]

        idx2 = self.features_dataset_labels.index(file_child)
        child_features = self.features_dataset_array[idx2, :, :]

        parent_id = int(file_parent.split('_')[1])
        child_id = int(file_child.split('_')[1])

        return parent_features, child_features, label, parent_id, child_id
