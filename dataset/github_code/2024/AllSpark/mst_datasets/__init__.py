import torch

from .oneD.Code.CodeSearchNet import (build_CodeSearchNetDataset,
                                      compute_metrics)
from .oneD.Text.IMDB import build_IMDB_dataloader
from .threeD.PointCloud.ModelNet import ModelNetDataset
from .threeD.PointCloud.FSDatasets import FSLPointDataset, FSLPointBatchSampler
from .twoD.Graph.data_prepare import get_datasets_from_index_data
from .twoD.HSI.Pavia import getPaviaDataset
from .twoD.MSI.BigEarthNet import BigEarthNetDataset
from .twoD.MSI.EuroSAT import EurosatDataset
from .twoD.RGB.NWPU_RESISC45 import NWPURESIS45Dataset
from .twoD.RGB.UCMerced_fewshot import UCMercedFewShotDataset, UCMercedFewShotBatchSampler
from .twoD.RGB.RS19_fewshot import RS19FewShotDataset, RS19FewShotBatchSampler
from .twoD.SAR.MSTAR import MSTARDataset
from .twoD.SAR.SARACD_fewshot import SARACDFewShotDataset, SARACDFewShotBatchSampler
from .twoD.Table.prsa import PRSADataset
from .twoD.Traj.Trajectory import TrajectoryDataset


def build_dataset(dataset_name, dataset_cfg):
    if dataset_name == 'NWPU_RESISC45':
        train_dataset = NWPURESIS45Dataset(metainfo_file=dataset_cfg['metainfo_train'], root_path=dataset_cfg['root_path'], is_train=True)
        test_dataset = NWPURESIS45Dataset(metainfo_file=dataset_cfg['metainfo_test'], root_path=dataset_cfg['root_path'], is_train=False)
    elif dataset_name == 'BigEarthNet_MSI':
        train_dataset = BigEarthNetDataset(metainfo_file=dataset_cfg['metainfo_train'], root_path=dataset_cfg['train_root_path'], is_train=True)
        test_dataset = BigEarthNetDataset(metainfo_file=dataset_cfg['metainfo_test'], root_path=dataset_cfg['test_root_path'], is_train=False)
    elif dataset_name == 'Pavia':
        train_dataset, test_dataset = \
            getPaviaDataset(root_path=dataset_cfg['root_path'], patches=dataset_cfg['patches'], 
                             band_patches=dataset_cfg['band_patches'], num_classes=dataset_cfg['num_classes'])
    
    else:
        raise NotImplementedError
        
    return train_dataset, test_dataset