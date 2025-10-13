import ujson as json
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from . import _utils
import os
from msmtu.utils import sort_nicely


def _parse_delta(masks, dir_):
    masks = np.array(masks)
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []
    seq_len = len(masks)
    data_dim = len(masks[0])

    for month in range(seq_len):
        if month == 0:
            deltas.append(np.ones(data_dim))
        else:
            deltas.append(np.ones(data_dim) + (1 - masks[month-1]) * deltas[-1])

    return np.array(deltas)


class LULCInference(Dataset):
    def __init__(self, json_dir, transform=None, label: int = 0):

        self.json_dir = json_dir
        json_path_pattern = os.path.join(self.json_dir, '*.json')

        self.json_paths = sorted(glob.glob(json_path_pattern))

        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        with open(self.json_paths[idx], 'r') as json_file:
            data = json.load(json_file)

        if self.transform is not None:
            data = self.transform(data)

        rec = data['ts_data']
        rec['json_path'] = self.json_paths[idx]  # Add json_path
        rec['label'] = self.label

        if 'probs' not in rec.keys():
            probs = [0]
            rec['probs'] = probs

        # Fix deltas
        rec['forward']['deltas'] = _parse_delta(rec['forward']['masks'], 'forward')
        rec['backward']['deltas'] = _parse_delta(rec['backward']['masks'], 'backward')

        return rec


def _set_test_imp(rec):
    evals = np.array(rec['forward']['evals'])
    ori_masks = np.array(rec['forward']['masks'])

    shp = evals.shape
    seq_len = shp[0]
    data_dim = shp[1]

    evals = evals.reshape(-1)
    ori_masks = ori_masks.reshape(-1)

    # randomly eliminate 10% values as the imputation ground-truth
    indices = np.where(~(ori_masks == 0))[0].astype(int).tolist()
    indices = np.random.choice(indices, len(indices) // 10).astype(int)

    values = evals.copy()
    values_masks = ori_masks.copy()

    values[indices] = 0.0
    values_masks[indices] = 0

    masks = np.zeros_like(ori_masks)
    masks[~(values_masks == 0)] = 1

    eval_masks = np.zeros_like(ori_masks)
    eval_masks[(~(values_masks == 0)) ^ (~(ori_masks == 0))] = 1

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    # Forward direction
    rec['forward']['values'] = values
    rec['forward']['evals'] = evals
    rec['forward']['masks'] = masks
    rec['forward']['eval_masks'] = eval_masks
    rec['forward']['deltas'] = _parse_delta(masks, 'forward')

    # Backward direction
    rec['backward']['values'] = values[::-1]
    rec['backward']['evals'] = evals[::-1]
    rec['backward']['masks'] = masks[::-1]
    rec['backward']['eval_masks'] = eval_masks[::-1]
    rec['backward']['deltas'] = _parse_delta(masks, 'backward')

    return rec


class LULC(Dataset):
    def __init__(self, directory, transform=None, data_dim=6):
        super(LULC, self).__init__()

        self.data_dim = data_dim

        self.classes, self.class_to_idx, self.idx_to_class = _utils.find_classes(directory)

        if directory[-1] != '/':
            directory = directory + '/'

        self._set_data(directory)

        self.transform = transform

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx: int, test_imp: bool = False):

        with open(self.json_paths[idx], 'r') as json_file:
            data = json.load(json_file)

        if self.transform is not None:
            data = self.transform(data)

        rec = data['ts_data']
        rec['json_path'] = self.json_paths[idx]  # Add json_path
        rec['label'] = self.targets[idx]  # Get numeric label

        if 'probs' not in rec.keys():
            probs = [1 if self.targets[idx] == i else 0 for i in range(len(self.classes))]
            rec['probs'] = probs

        if test_imp:
            rec = _set_test_imp(rec)
        else:
            # Fix deltas
            rec['forward']['deltas'] = _parse_delta(rec['forward']['masks'], 'forward')
            rec['backward']['deltas'] = _parse_delta(rec['backward']['masks'], 'backward')

        return rec

    def _set_data(self, directory):
        classes_dirs = sort_nicely(list(glob.glob(directory + '*/')))
        self.json_paths = []
        self.targets = []
        for class_dir in classes_dirs:

            class_name = class_dir.split(sep='/')[-2]

            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
            else:
                print(f'Class "{class_name}" not in class_to_idx!!')
                class_idx = -1

            class_json_paths = sorted(glob.glob(class_dir + '*.json'))

            self.json_paths.extend(class_json_paths)  # Add the class json to the global json list
            self.targets.extend([class_idx] * len(class_json_paths))  # Add the targets

        self.targets = np.array(self.targets)

    def get_dicts(self):
        return self.class_to_idx, self.idx_to_class


class LULCOptim(Dataset):
    def __init__(
            self,
            directory,
            labels_path,
            transform=None,
            data_dim=6,
            ancillary_path=None,
            ancillary_data=[]
    ):
        super(LULCOptim, self).__init__()

        self.data_dim = data_dim

        self._set_data(directory, labels_path)

        self.transform = transform

        if ancillary_path is not None:
            self.ancillary_df = pd.read_csv(ancillary_path).set_index('square_id').loc[self.labels_df.index]
            self.ancillary_data = ancillary_data

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx: int, test_imp: bool = False):

        with open(self.json_paths[idx], 'r') as json_file:
            data = json.load(json_file)

        square_id = self.labels_df.index[idx]
        rec = {
            'ts_data': data['ts_data'],
            'ancillary_data': {k: self.ancillary_df.loc[square_id][k] for k in self.ancillary_data}
        }

        if self.transform is not None:
            rec = self.transform(rec)

        rec['json_path'] = self.json_paths[idx]  # Add json_path
        rec['label'] = self.targets[idx]  # Get numeric label

        rec['probs'] = self.labels_df.iloc[idx][self.classes].tolist()
        rec['max_prob'] = max(rec['probs'])
        rec['square_id'] = square_id

        # Fix deltas
        rec['ts_data']['forward']['deltas'] = _parse_delta(rec['ts_data']['forward']['masks'], 'forward')
        rec['ts_data']['backward']['deltas'] = _parse_delta(rec['ts_data']['backward']['masks'], 'backward')

        return rec

    def _set_data(self, directory, labels_path):

        self.labels_df = pd.read_csv(labels_path).set_index('square_id').sort_index()
        self.labels_df = self.labels_df[sorted(self.labels_df.columns)]
        self.classes, self.class_to_idx, self.idx_to_class = _utils.find_classes_df(self.labels_df)

        square_ids = self.labels_df.index.to_numpy()
        self.json_paths = [os.path.join(directory, f'{str(square_id).zfill(7)}.json') for square_id in square_ids]

        self.targets = self.labels_df['Class_max'].astype(str).map(self.class_to_idx).tolist()

        assert len(self.json_paths) == len(self.labels_df) == len(self.targets)

    def get_dicts(self):
        return self.class_to_idx, self.idx_to_class


class LULCSubset(Subset):
    def __init__(self, dataset: LULC, indices: np.ndarray, test_imp: bool = False):
        self.dataset = dataset
        self.indices = indices
        self.targets = self.dataset.targets[indices]

        self.test_imp = test_imp

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]

        return self.dataset.__getitem__(self.indices[idx], self.test_imp)

