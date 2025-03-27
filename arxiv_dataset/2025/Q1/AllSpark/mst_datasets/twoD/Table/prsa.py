import glob
from collections import OrderedDict
from os import path

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Vocabulary:
    def __init__(self, adap_thres=10000, target_column_name="Is Fraud?"):
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        self.adap_thres = adap_thres
        self.adap_sm_cols = set()

        self.target_column_name = target_column_name
        self.special_field_tag = "SPECIAL"

        self.special_tokens = [self.unk_token, self.sep_token, self.pad_token,
                               self.cls_token, self.mask_token, self.bos_token, self.eos_token]

        self.token2id = OrderedDict()  # {field: {token: id}, ...}
        self.id2token = OrderedDict()  # {id : [token,field]}
        self.field_keys = OrderedDict()
        self.token2id[self.special_field_tag] = OrderedDict()

        self.filename = ''  # this field is set in the `save_vocab` method

        for token in self.special_tokens:
            global_id = len(self.id2token)
            local_id = len(self.token2id[self.special_field_tag])

            self.token2id[self.special_field_tag][token] = [global_id, local_id]
            self.id2token[global_id] = [token, self.special_field_tag, local_id]

    def set_id(self, token, field_name, return_local=False):
        global_id, local_id = None, None

        if token not in self.token2id[field_name]:
            global_id = len(self.id2token)
            local_id = len(self.token2id[field_name])

            self.token2id[field_name][token] = [global_id, local_id]
            self.id2token[global_id] = [token, field_name, local_id]
        else:
            global_id, local_id = self.token2id[field_name][token]

        if return_local:
            return local_id

        return global_id

    def get_id(self, token, field_name="", special_token=False, return_local=False):
        global_id, local_id = None, None
        if special_token:
            field_name = self.special_field_tag

        if token in self.token2id[field_name]:
            global_id, local_id = self.token2id[field_name][token]

        else:
            raise Exception(f"token {token} not found in field: {field_name}")

        if return_local:
            return local_id

        return global_id

    def set_field_keys(self, keys):

        for key in keys:
            self.token2id[key] = OrderedDict()
            self.field_keys[key] = None

        self.field_keys[self.special_field_tag] = None  # retain the order of columns

    def get_field_ids(self, field_name, return_local=False):
        if field_name in self.token2id:
            ids = self.token2id[field_name]
        else:
            raise Exception(f"field name {field_name} is invalid.")

        selected_idx = 0
        if return_local:
            selected_idx = 1
        return [ids[idx][selected_idx] for idx in ids]

    def get_from_global_ids(self, global_ids, what_to_get='local_ids'):
        device = global_ids.device

        def map_global_ids_to_local_ids(gid):
            return self.id2token[gid][2] if gid != -100 else -100

        def map_global_ids_to_tokens(gid):
            return f'{self.id2token[gid][1]}_{self.id2token[gid][0]}' if gid != -100 else '-'

        if what_to_get == 'local_ids':
            return global_ids.cpu().apply_(map_global_ids_to_local_ids).to(device)
        elif what_to_get == 'tokens':
            vectorized_token_map = np.vectorize(map_global_ids_to_tokens)
            new_array_for_tokens = global_ids.detach().clone().cpu().numpy()
            return vectorized_token_map(new_array_for_tokens)
        else:
            raise ValueError("Only 'local_ids' or 'tokens' can be passed as value of the 'what_to_get' parameter.")

    def save_vocab(self, fname):
        self.filename = fname
        with open(fname, "w") as fout:
            for idx in self.id2token:
                token, field, _ = self.id2token[idx]
                token = "%s_%s" % (field, token)
                fout.write("%s\n" % token)

    def get_field_keys(self, remove_target=True, ignore_special=False):
        keys = list(self.field_keys.keys())

        if remove_target and self.target_column_name in keys:
            keys.remove(self.target_column_name)
        if ignore_special:
            keys.remove(self.special_field_tag)
        return keys

    def get_special_tokens(self):
        special_tokens_map = {}
        # TODO : remove the dependency of re-initializing here. retrieve from field_key = SPECIAL
        keys = ["unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token"]
        for key, token in zip(keys, self.special_tokens):
            token = "%s_%s" % (self.special_field_tag, token)
            special_tokens_map[key] = token

        return AttrDict(special_tokens_map)

    def __len__(self):
        return len(self.id2token)

    def __str__(self):
        str_ = 'vocab: [{} tokens]  [field_keys={}]'.format(len(self), self.field_keys)
        return str_


class PRSADataset(Dataset):
    def __init__(self,
                 data_root,
                 seq_len=10,
                 stride=5,
                 nbins=50,
                 mlm=True,
                 return_labels=False,
                 use_station=False,
                 transform_date=True,
                 flatten=False):

        self.stride = stride
        self.seq_len = seq_len
        self.data_root = data_root
        self.nbins = nbins

        self.mlm = mlm
        self.return_labels = return_labels
        self.use_station = use_station
        self.transform_date = transform_date
        self.flatten = flatten

        self.vocab = Vocabulary()
        self.encoding_fn = {}
        self.target_cols = ['PM2.5']

        self.setup()

    def __getitem__(self, index):
        if self.flatten:
            return_data = torch.tensor(self.samples[index], dtype=torch.long)
        else:
            return_data = torch.tensor(self.samples[index], dtype=torch.long).reshape(self.seq_len, -1)

        if self.return_labels:
            target = self.targets[index]
            return_data = return_data, torch.tensor(target, dtype=torch.float32)

        return return_data

    def __len__(self):
        return len(self.samples)

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.nbins, 1 / self.nbins)
        bin_edges = np.quantile(data, qtls, axis=0)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.nbins) - 1
        return quant_inputs

    @staticmethod
    def time_fit_transform(column):
        mfit = MinMaxScaler()
        mfit.fit(column)
        return mfit, mfit.transform(column)

    @staticmethod
    def timeEncoder(X):
        d = pd.to_datetime(dict(year=X['year'], month=X['month'], day=X['day'], hour=X['hour'])).astype(int)
        return pd.DataFrame(d)

    def setup(self):
        data = self.read_data(self.data_root)

        '''
        year 	month 	day 	hour 	PM2.5 	PM10 	SO2 	NO2 	
        CO 	O3 	TEMP 	PRES 	DEWP 	RAIN 	wd 	WSPM 	station
        '''

        cols_for_bins = []
        if self.transform_date:
            cols_for_bins += ['timestamp']

            data_cols = ['year', 'month', 'day', 'hour']
            timestamp = self.timeEncoder(data[data_cols])
            timestamp_fit, timestamp = self.time_fit_transform(timestamp)
            self.encoding_fn['timestamp'] = timestamp_fit
            data['timestamp'] = timestamp

        cols_for_bins += ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        for col in cols_for_bins:
            col_data = np.array(data[col])
            bin_edges, bin_centers, bin_widths = self._quantization_binning(col_data)
            data[col] = self._quantize(col_data, bin_edges)
            self.encoding_fn[col] = [bin_edges, bin_centers, bin_widths]

        # final_cols = cols_for_bins + ['wd', 'station', 'PM2.5', 'PM10']
        final_cols = cols_for_bins + ['wd', 'station', 'PM2.5']

        self.data = data[final_cols]

        assert len(self.target_cols) == 1
        targets_col = self.target_cols[0]
        self.targets_max = self.data[targets_col].max()
        self.targets_min = self.data[targets_col].min()
        self.data[targets_col] = (self.data[targets_col] - self.targets_min) / (self.targets_max - self.targets_min)

        self.init_vocab()
        self.prepare_samples()
        
    def inv_scale(self, y):
        y = y*(self.targets_max - self.targets_min) + self.targets_min
        return y

    def prepare_samples(self):
        self.samples, self.targets = [], []
        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        groups = self.data.groupby('station')
        for group in tqdm.tqdm(groups):
            station_name, station_data = group

            nrows = station_data.shape[0]
            nrows = nrows - self.seq_len

            print(f"{station_name} : {nrows}")
            for sample_id in range(0, nrows, self.stride):
                sample, target = [], []
                for tid in range(0, self.seq_len):
                    row = station_data.iloc[sample_id + tid]
                    for col_name, col_value in row.items():
                        if not self.use_station:
                            if col_name == "station":
                                continue
                        if col_name not in self.target_cols:
                            vocab_id = self.vocab.get_id(col_value, col_name)
                            sample.append(vocab_id)

                    if self.mlm:
                        sample.append(sep_id)
                    target.append(row[self.target_cols].tolist())

                self.samples.append(sample)
                self.targets.append(target)

        assert len(self.samples) == len(self.targets)
        print(f"total samples {len(self.samples)}")

        self.ncols = len(self.vocab.field_keys)

    def init_vocab(self):
        cols = list(self.data.columns)

        if not self.use_station:
            cols.remove('station')

        for col in self.target_cols:
            cols.remove(col)

        self.vocab.set_field_keys(cols)

        for column in cols:
            unique_values = self.data[column].value_counts(sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.vocab.set_id(val, column)

        print(f"columns used for vocab: {list(cols)}")
        print(f"total vocabulary size: {len(self.vocab.id2token)}")

        for column in cols:
            vocab_size = len(self.vocab.token2id[column])
            print(f"column : {column}, vocab size : {vocab_size}")

    def read_data(self, root):
        all_stations = None
        fnames = glob.glob(f"{root}/*.csv")
        for fname in fnames:
            station_data = pd.read_csv(fname)

            if all_stations is None:
                all_stations = station_data
            else:
                all_stations = all_stations._append(station_data, ignore_index=True)

        all_stations.drop(columns=['No'], inplace=True, axis=1)
        print(f"shape (original)   : {all_stations.shape}")
        all_stations = all_stations.dropna()
        print(f"shape (after nan removed): {all_stations.shape}")
        return all_stations