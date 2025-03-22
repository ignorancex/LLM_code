import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (
    normalize_batch_ts,
    bandpass_filter_func,
)
import warnings
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.signal import resample

from data_provider.dataset_loader.adsz_loader import ADSZLoader
from data_provider.dataset_loader.apava_loader import APAVALoader
from data_provider.dataset_loader.adfsu_loader import ADFSULoader
from data_provider.dataset_loader.cognision_rseeg_loader import COGrsEEGLoader
from data_provider.dataset_loader.cognision_erp_loader import COGERPLoader
from data_provider.dataset_loader.adftd_loader import ADFTDLoader
from data_provider.dataset_loader.cnbpm_loader import CNBPMLoader
from data_provider.dataset_loader.brainlat_loader import BrainLatLoader
from data_provider.dataset_loader.ad_auditory_loader import ADAuditoryLoader
from data_provider.dataset_loader.tdbrain_loader import TBDRAINLoader
from data_provider.dataset_loader.tuep_loader import TUEPLoader
from data_provider.dataset_loader.reeg_pd_loader import REEGPDLoader
from data_provider.dataset_loader.pearl_neuro_loader import PEARLNeuroLoader
from data_provider.dataset_loader.depression_loader import DepressionLoader
from data_provider.dataset_loader.reeg_srm_loader import REEGSRMLoader
from data_provider.dataset_loader.reeg_baca_loader import REEGBACALoader

# data folder dict to loader mapping
data_folder_dict = {
    # should use the same name as the dataset folder
    # For datasets that the raw channel number is 19, there is no -19 suffix in the dataset name

    # datasets with raw channel number for single-dataset supervised learning
    'APAVA': APAVALoader,  # APAVA with 16 channels
    'Cognision-ERP': COGERPLoader,  # Cognision-ERP with 7 channels
    'Cognision-rsEEG': COGrsEEGLoader,  # Cognision-rsEEG with 7 channels
    'BrainLat': BrainLatLoader,  # BrainLat with 128 channels

    # datasets using 19 channels alignments for consistency
    # 5 downstream datasets
    'ADFTD': ADFTDLoader,  # ADFD with 19 channels
    'CNBPM': CNBPMLoader,  # CNBPM with 19 channels
    'Cognision-ERP-19': COGERPLoader,  # Cognision-ERP with 19 channels
    'Cognision-rsEEG-19': COGrsEEGLoader,  # Cognision-rsEEG with 19 channels
    'BrainLat-19': BrainLatLoader,  # BrainLat with 19 channels

    # 11 pretraining datasets
    'ADSZ': ADSZLoader,  # ADSZ with 19 channels
    'ADFSU': ADFSULoader,  # ADFSU with 19 channels
    'APAVA-19': APAVALoader,  # APAVA with 19 channels
    'AD-Auditory': ADAuditoryLoader,  # AD-Auditory
    'TDBRAIN-19': TBDRAINLoader,  # TBDRAIN with 19 channels
    'TUEP': TUEPLoader,  # TUEP with 19 channels
    'REEG-PD-19': REEGPDLoader,  # REEG-PD with 19 channels
    'PEARL-Neuro-19': PEARLNeuroLoader,  # PEARL-Neuro with 19 channels
    'Depression-19': DepressionLoader,  # Depression with 19 channels
    'REEG-SRM-19': REEGSRMLoader,  # REEG-SRM with 19 channels
    'REEG-BACA-19': REEGBACALoader,  # REEG-BACA with 19 channels
}
warnings.filterwarnings('ignore')


class SingleDatasetLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path

        print(f"Loading {flag} samples from single dataset...")
        if flag == 'PRETRAIN':
            data_folder_list = args.pretraining_datasets.split(",")
        elif flag == 'TRAIN':
            data_folder_list = args.training_datasets.split(",")
        elif flag == 'TEST' or flag == 'VAL':
            data_folder_list = args.testing_datasets.split(",")
        else:
            raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")
        if len(data_folder_list) > 1:
            raise ValueError("Only one dataset should be given here")
        print(f"Datasets used ", data_folder_list[0])
        data = data_folder_list[0]
        if data not in data_folder_dict.keys():
            raise Exception("Data not matched, "
                            "please check if the data folder name in data_folder_dict.")
        else:
            Data = data_folder_dict[data]
            data_set = Data(
                root_path=os.path.join(args.root_path, data),
                args=args,
                flag=flag,
            )
            print(f"{data} data shape: {data_set.X.shape}, {data_set.y.shape}")
            # only one dataset, dataset ID is 1
            data_set.y = np.concatenate((data_set.y, np.full(data_set.y[:, 0].shape, 1).reshape(-1, 1)), axis=1)
            self.X, self.y = data_set.X, data_set.y

        self.X, self.y = shuffle(self.X, self.y, random_state=42)
        self.max_seq_len = self.X.shape[1]
        print(f"Unique subjects used in {flag}: ", len(np.unique(self.y[:, 1])))
        print()

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class MultiDatasetsLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path

        print(f"Loading {flag} samples from multiple datasets...")
        if flag == 'PRETRAIN':
            data_folder_list = args.pretraining_datasets.split(",")
        elif flag == 'TRAIN':
            data_folder_list = args.training_datasets.split(",")
        elif flag == 'TEST' or flag == 'VAL':
            data_folder_list = args.testing_datasets.split(",")
        else:
            raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")
        print(f"Datasets used ", data_folder_list)
        self.X, self.y = None, None
        global_sub_num = 1  # count global subject number to avoid duplicate IDs in multiple datasets
        for i, data in enumerate(data_folder_list):
            if data not in data_folder_dict.keys():
                raise Exception("Data not matched, "
                                "please check if the data folder name in data_folder_dict.")
            else:
                Data = data_folder_dict[data]
                data_set = Data(
                    root_path=os.path.join(args.root_path, data),
                    args=args,
                    flag=flag,
                )
                # add dataset ID to the third column of y, id starts from 1
                data_set.y = np.concatenate((data_set.y, np.full(data_set.y[:, 0].shape, i + 1).reshape(-1, 1)), axis=1)
                print(f"{data} data shape: {data_set.X.shape}, {data_set.y.shape}")
                if self.X is None or self.y is None:
                    self.X, self.y = data_set.X, data_set.y
                    global_sub_num = len(data_set.all_ids)
                else:
                    # number of subjects in the current dataset
                    local_sub_num = len(data_set.all_ids)
                    # update subject IDs in the current dataset by adding global_sub_num
                    data_set.y[:, 1] += global_sub_num
                    # update global subject number
                    global_sub_num += local_sub_num
                    # concatenate data from different datasets
                    self.X, self.y = (np.concatenate((self.X, data_set.X), axis=0),
                                      np.concatenate((self.y, data_set.y), axis=0))

        self.X, self.y = shuffle(self.X, self.y, random_state=42)
        self.max_seq_len = self.X.shape[1]
        # print(f"Unique subjects used in {flag}: ", len(np.unique(self.y[:, 1])))
        print()

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)

