import os.path

import pytorch_lightning as pl
import numpy as np
from .datasets import LULCOptim
import torch
import torch.utils.data as data

from msmtu.transforms import StandardizeTS

mean_modis = np.array([1778.0370114114635, 2885.055655885381, 1256.2631572253467, 1586.7335998607268,
                       2817.6737163027287, 2420.865232616906, 1763.7376743953382])

std_modis = np.array([2067.9758198306, 1811.1213005051807, 2146.4020578479854, 2059.230258563475,
                      1454.778090999886, 1595.473540093584, 1487.7149351931803])

dict_ancillary = {
    'longitude': {'mean': -4.5726, 'std': 1.42003659},
    'latitude': {'mean': 37.4675, 'std': 0.54427359},
    'altitude': {'mean': 526.941630062964, 'std': 434.20557148483016},
    'slope': {'mean': 9.098918613850193, 'std': 6.635985730650851},
    'precipitation': {'mean': 3398.9591808260107, 'std': 1157.0350863758044},
    'evapotranspiration': {'mean': 5147.744891820799, 'std': 466.2150145975434},
    'temperature_ave': {'mean': 15.593676475326593, 'std': 1.9789136521273902},
    'temperature_max': {'mean': 21.606108, 'std': 2.076332},
    'temperature_min': {'mean': 10.047357, 'std': 2.012799}
}


def collate_fn(recs):
    forward = np.array(list(map(lambda x: x['ts_data']['forward'], recs)))
    backward = np.array(list(map(lambda x: x['ts_data']['backward'], recs)))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(np.array(list(map(lambda r: r['values'], recs))))
        masks = torch.FloatTensor(np.array(list(map(lambda r: r['masks'], recs))))
        deltas = torch.FloatTensor(np.array(list(map(lambda r: r['deltas'], recs))))

        return {'values': values, 'masks': masks, 'deltas': deltas}

    ret_dict = {
        'forward': to_tensor_dict(forward),
        'backward': to_tensor_dict(backward),
        'labels': torch.FloatTensor(np.array(list(map(lambda x: x['label'], recs)))),
        'probs': torch.FloatTensor(np.array(list(map(lambda x: x['probs'], recs)))),
        'ancillary': torch.FloatTensor(np.array(list(map(lambda x: list(x['ancillary_data'].values()), recs)))),
        'square_ids': torch.FloatTensor(np.array(list(map(lambda x: x['square_id'], recs)))),
    }

    return ret_dict


class LULCDataModuleOptim(pl.LightningDataModule):
    def __init__(
            self,
            json_dir: str,
            batch_size: int,
            test_imp: bool = False,
            data_dim: int = 6,
            train_labels_path: str = None,
            val_labels_path: str = None,
            test_labels_path: str = None,
            ancillary_path: str = None,
            ancillary_data: list = [],
    ):
        super(LULCDataModuleOptim, self).__init__()

        self.json_dir = json_dir
        self.batch_size = batch_size
        self.ancillary_data = ancillary_data
        self.transform = StandardizeTS(
            mean=mean_modis,
            std=std_modis,
            dict_ancillary={k: dict_ancillary[k] for k in ancillary_data}
        )

        self.test_imp = test_imp
        self.data_dim = data_dim

        if not os.path.exists(train_labels_path):
            raise ValueError(f'Training labels path does not exists: {train_labels_path}')
        else:
            self.train_labels_path = train_labels_path

        if not os.path.exists(val_labels_path):
            raise ValueError(f'Val labels path does not exists: {val_labels_path}')
        else:
            self.val_labels_path = val_labels_path

        if test_labels_path is not None and not os.path.exists(test_labels_path):
            raise ValueError(f'Test label path does not exists: {test_labels_path}')
        else:
            self.test_labels_path = test_labels_path

        if ancillary_path is not None and not os.path.exists(ancillary_path):
            raise ValueError(f'ancillary path does not exists: {ancillary_path}')
        else:
            self.ancillary_path = ancillary_path

        self.train_set = LULCOptim(
            self.json_dir,
            self.train_labels_path,
            self.transform,
            self.data_dim,
            ancillary_path=ancillary_path,
            ancillary_data=self.ancillary_data
        )

    def _get_val_size(self):
        return self.val_size / (1 - self.test_size)

    def setup(self, stage=None) -> None:
        if stage == 'fit' or stage is None:
            self.val_set = LULCOptim(
                self.json_dir,
                self.val_labels_path,
                self.transform,
                self.data_dim,
                ancillary_path=self.ancillary_path,
                ancillary_data=self.ancillary_data
            )

        if stage == 'test' or stage is None:
            self.test_set = LULCOptim(
                self.json_dir,
                self.test_labels_path,
                self.transform,
                self.data_dim,
                ancillary_path=self.ancillary_path,
                ancillary_data=self.ancillary_data
            )

    def train_dataloader(self):
        train_loader = data.DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return train_loader

    def val_dataloader(self):

        val_loader = data.DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return val_loader

    def test_dataloader(self):
        test_loader = data.DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return test_loader

    def predict_dataloader(self):
        predict_loader = data.DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return predict_loader

    def get_dicts(self):
        return self.train_set.get_dicts()

