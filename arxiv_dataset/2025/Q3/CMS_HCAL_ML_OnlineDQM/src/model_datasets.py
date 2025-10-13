"""
Created on Mon Dec 07 17:33:21 2023

@author: Mulugeta W. Asres, mulugetawa@uia.no

DESMOD: ML Modeling Data Sets Preparation for HCAL DQM Digioccupancy 

Example: given in main function

"""

import os, sys
import seaborn as sns
from functools import reduce
import numpy as np, pandas as pd
from tqdm import tqdm
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import argparse

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)
data_path = os.path.abspath(os.path.dirname(current_path)+"/data")
sys.path.append(data_path)

import utilities as util

def get_model_ts_data_spatial(data_np_nd, **kwargs):
    """
    slices on time window on the first dim=ls
    data_np_nd: 5D [lsxietaxiphixdepthxfeature]
    output: data_np_nd: 6D [slicesxtsxietaxiphixdepthxfeature]

    datatype="ts", timewindow_size=5, horizonsize=None, iststw_overlapped=False, istoslice=True, istrain=False, modeltype="ae"
    """
    
    datatype = kwargs.get("datatype", "ts")
    timewindow_size = kwargs.get("timewindow_size", None)
    timewindow_size = kwargs.get("memorysize", None) if timewindow_size is None else timewindow_size
    iststw_overlapped = kwargs.get("iststw_overlapped", False)
    istoslice = kwargs.get("istoslice", True)

    print("spatial ts slicer: ", datatype, timewindow_size,
          iststw_overlapped, istoslice)

    def slicing_handler(data_np_nd, timewindow_size):
        '''
        input data: data_np_nd -> 3D numpy
        output: sliced_data -> tuple of 3D numpy
        timewindow_size is size of the sliding time window past memory
        '''

        input_shape = data_np_nd.shape
        n = input_shape[0]
        n_slices = (n - timewindow_size)//timewindow_size + 1
        print((n_slices, timewindow_size) + input_shape[1:])
        x_sliced_data = np.zeros((n_slices, timewindow_size) + input_shape[1:])
        for i in range(0, n_slices):
            x_sliced_data[i] = data_np_nd[i *
                                          timewindow_size:i*timewindow_size + timewindow_size].copy()

        return (x_sliced_data, )

    def unslicing_handler(sliced_data, timewindow_size):
        '''
        input data: sliced_data -> 6D numpy
        output: data_np_nd -> 5D
        timewindow_size is size of the sliding time window past memory 
        '''

        input_shape = sliced_data.shape
        n_slices, timewindow_size, n_variables_dim = input_shape[0], input_shape[1], input_shape[2:]

        data_np_nd = sliced_data.reshape(
            (n_slices*timewindow_size, ) + n_variables_dim)

        if n_slices > 1:
            data_np_nd = np.concatenate((data_np_nd[:timewindow_size],
                                         np.concatenate(([data_np_nd[i:i+timewindow_size]
                                                          for i in range(2*timewindow_size-timewindow_size, data_np_nd.shape[0], timewindow_size)]))))

        return data_np_nd

    slice_status = False

    if not isinstance(data_np_nd, np.ndarray):
        if isinstance(data_np_nd, torch.Tensor):
            data_np_nd = data_np_nd.cpu().detach().numpy()
        else:
            raise "data must be an np.ndarray format."

    if datatype == "ts":
        if istoslice:
            print("generating time series slices...")

            if not timewindow_size:
                raise "timewindow_size must be provided to accurately agg slices of horizons!"

           
            print(f"timewindow_size: {timewindow_size}")

            sliced_data = slicing_handler(
                data_np_nd, timewindow_size)

            print(f"sliding timewindow size: {timewindow_size} \ninput shape:{data_np_nd.shape}, output shape: {sliced_data[0].shape}")

            slice_status = True

            return sliced_data, slice_status
        else:
            print("reconstructing time series from slices...")
            '''
            timewindow_size is history window in rec ae or horizonsize for target in forecasting
            '''
            sliced_data = data_np_nd

            if not timewindow_size:
                raise "slicing timewindow_size must be provided to calculate slide_jump and thus, accurately agg slices slide_jump!"

            print(f"timewindow_size: {timewindow_size}")

            data_np_nd = unslicing_handler(
                sliced_data, timewindow_size)

            print(f"sliding timewindow size: {timewindow_size} \ninput shape:{sliced_data.shape}, output shape: {data_np_nd.shape}")

            slice_status = False
            return data_np_nd, slice_status

    else:
        return (data_np_nd, ), slice_status


class NoScaler():
    def __init__(self):
        pass

    def _get_numpy(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        return data

    def fit(self, data):
        pass

    def transform(self, data):
        return self._get_numpy(data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return self._get_numpy(data)
    
class DatasetTemplate(Dataset):

    def __init__(self, data_np_nd=None, **kwargs):
        self.dataset_name = kwargs.get("dataset_name", None)
        self.source_files = kwargs.get("source_files", None)
        self.datatype = kwargs.get("datatype", "iid")
        self.timewindow_size = kwargs.get("timewindow_size", 5)
        self.istrain = kwargs.get("istrain", False)
        self.input_scaling_alg = kwargs.get("input_scaling_alg", None)
        self.input_scaler = kwargs.get("input_scaler", None)
        self.isdata_scaled = kwargs.get("isdata_scaled", False)
        self.mask = kwargs.get("mask", None)

        self.features = ["digioccupancy"]
        self.targets = ["digioccupancy"]
        self.features_idx = [0]
        self.targets_idx = [0]

        print("{}\ndataset preparation...\n{}".format("#"*60, "#"*60))
        print("input data size ([lsxietaxiphxdepthxfeature] or [lsxietaxiphxxfeature]): {}".format(
            data_np_nd.shape))

        self.is_tssliced = False

        self._preprocess(data_np_nd)

        print("{}\ndataset preparation is done: size: {} !\n{}".format(
            "#"*60, self.samples[0].shape, "#"*60))

    def _preprocess(self, data_np_nd):

        print("preprocessing...: {}".format(data_np_nd.shape))

        self.is_tssliced = False

        if isinstance(data_np_nd, np.ndarray):
            data_np_nd = torch.FloatTensor(data_np_nd)

        # data scaling
        data_np_nd = self.pretransform_data(
            data_np_nd, sel_variable_idx=None)  

        if self.datatype != "ts":  # for dim compatability
            self.timewindow_size = 1
            self.datatype = "ts"

        data_np, self.is_tssliced = get_model_ts_data_spatial(data_np_nd, datatype=self.datatype, timewindow_size=self.timewindow_size, istoslice=True)  # return array

        print("after slicing sizes:")
        print(data_np[0].shape)
        data_np = [torch.FloatTensor(d) for d in list(data_np)]

        data_size = data_np[0].shape
        print("prepared sliced data: {}".format(data_size))

        if isinstance(data_np, np.ndarray):
            self.samples = [torch.FloatTensor(d) for d in data_np]
        else:
            self.samples = data_np

        self.samples_len = len(self.samples)
        self.samples_dim_len = len(self.samples[0].shape)
        print("samples size: ", self.samples[0].shape)

    def __len__(self):
        return len(self.samples[0])

    def shape(self, target=False):
        return self.samples[0][..., self.features_idx].shape if not target else self.samples[0][..., self.targets_idx].shape

    def __getitem__(self, idx):
        return self.samples[0][idx, ..., self.features_idx], self.samples[0][idx, ..., self.targets_idx]

    def get_attr(self, key):
        return getattr(self, key)

    def set_attr(self, key, value):
        setattr(self, key, value)

    def pretransform_data(self, data_np_nd, sel_variable_idx=None):
        print(f"data scaling via data transformation: {self.input_scaling_alg}...")

        if self.input_scaling_alg and self.is_tssliced:
            print("scaling time sliced is not allowed.")
            return data_np_nd

        if not self.input_scaling_alg:
            print("no data preprocessing scaler algorithm is not selected or trained.")
            return data_np_nd

        if not self.isdata_scaled:
            if self.istrain:
                self.input_scaler = None
                if self.input_scaling_alg:
                    print(f"fitting data scaling using: {self.input_scaling_alg}...")

                    if self.input_scaling_alg == "minmax":
                        print("{} is being used. make sure first outliers are cleaned in the data.".format(
                            self.input_scaling_alg))
                        self.input_scaler = MinMaxScaler(feature_range=(0, 1))
   
                    elif self.input_scaling_alg == "noscaler":
                        self.input_scaler = NoScaler()
                    
                    else:
                        print("input scaling algorithm is not implemented!")
                        return data_np_nd

                    # change shape into [samplexfeature_to_scale_dim] and fit scaler and scaling training data
                    data_np_nd_reshaped_scaled = torch.tensor(self.input_scaler.fit_transform(
                        data_np_nd.view(data_np_nd.shape[0], -1).detach().numpy().astype("float64")).astype("float64"))
                    
                    # restore original dim
                    data_np_nd = data_np_nd_reshaped_scaled.reshape(
                        data_np_nd.shape)
                    self.isdata_scaled = True

            elif self.input_scaler:
                print(f"transform data scaling using {self.input_scaling_alg}...")
                # change shape into [samplexfeature_to_scale_dim] and scale data
                data_np_nd_reshaped_scaled = torch.tensor(self.input_scaler.transform(
                    data_np_nd.view(data_np_nd.shape[0], -1).detach().numpy().astype("float64")).astype("float64"))
                
                # restore original dim
                data_np_nd = data_np_nd_reshaped_scaled.view( data_np_nd.shape)
                self.isdata_scaled = True
            else:
                print("no data preprocessing scaler algorithm is not selected or trained.")
        else:
            print("data is already transformed using {}.".format(self.input_scaling_alg))

        return data_np_nd

    def get_np(self):
        '''
        returns numpy equivalent of the tensor samples
        '''
        return self.samples[0].detach().numpy(), self.samples[0].detach().numpy()

    def get_tensor(self):
        return self.samples[0][..., self.features_idx].detach().clone(), self.samples[0][..., self.targets_idx].detach().clone()
 

def dqm_dataset_loader(dataset_filedirpath, datanametag="train", **kwargs):
        print("dqm_dataset_loader...")

        raw_data = util.load_npdata(rf"{dataset_filedirpath}\{datanametag}_data.npy")
        raw_data.shape
        meta_data = util.load_json(rf"{dataset_filedirpath}\{datanametag}_meta_data.json")

        data_settings = {}

        # padding between runs, to be added
        data_settings["data_run_ids"] = list(filter(lambda x:x.startswith("Run") and x!="RunId", meta_data.keys()))
        data_settings["meta_data"] = meta_data
        data_settings["variables_sensor"] = meta_data["dqm_variable"]
        data_settings["variables_flag"] = []
        data_settings["sensors_ts"] = data_settings["variables_sensor"]
        data_settings["sensors_iid"] = []
        print("dataset size: {}".format(raw_data.shape))
        return raw_data, data_settings

def prepare_model_temporal_torch_dataset(**kwargs):
    print("prepare_model_temporal_torch_dataset...")

    datatype = kwargs.get("datatype", None)
    subdetector_name = kwargs.get("subdetector_name", "hbhe")
    input_scaling_alg = kwargs.get("input_scaling_alg", None)
    memorysize = kwargs.get("memorysize", None)
    istrain = kwargs.get("istrain", True)
    data_dirpath = kwargs.get("data_dirpath", data_path)
    dataset = kwargs.get("dataset", "")
    data_filename = kwargs.get("data_filename", "train_dataset")
    trainnref_filename = kwargs.get("trainnref_filename", None)
    kwargs_data = json.loads(kwargs.pop("kwargs_data", "{}"))
    print(kwargs_data)

    if istrain:
        print("Training Dataset preparation ....")
    else:
        print("Testing Dataset preparation ....")

    dataset_name = dataset.replace('/', '_')
    train_arg = {
                 "datatype": datatype,
                 "memorysize": memorysize,
                 "istrain": istrain,
                 "input_scaling_alg": input_scaling_alg
                 }

    kwargs.update(kwargs_data)

    train_data, data_settings = dqm_dataset_loader(data_dirpath, datanametag="train" if istrain else "test")
    
    # util.print_dict(data_settings)

    train_arg.update(data_settings)
    train_arg.update({"features_ts": data_settings["sensors_ts"],
                      "targets": data_settings["targets"] if "targets" in data_settings.keys() else [],
                      })

    if not istrain:
        train_data_ref = util.load_pickle("{}/{}_{}_{}.pkl".format(data_path,
            dataset_name, subdetector_name, trainnref_filename))
        
        train_arg["isclean_outlier"] = False
        train_arg["input_scaling_alg"] = train_data_ref.get_attr(
            "input_scaling_alg")
        train_arg["input_scaler"] = train_data_ref.get_attr("input_scaler")
        train_arg["features"] = train_data_ref.get_attr("features")
        train_arg["targets"] = train_data_ref.get_attr("targets")
        train_arg["features_idx"] = train_data_ref.get_attr("features_idx")
        train_arg["targets_idx"] = train_data_ref.get_attr("targets_idx")

    # load subdetector mask
    try:
        subdetector_mask = util.load_npdata(rf"{data_path}\HCAL_CONFIG\{subdetector_name}_segmentation_config_mask.npy").astype(bool)
        print(f"hcal segmentation map is loaded: {subdetector_mask.shape}")
    except Exception as ex:
        print(ex)
        subdetector_mask = None

    mask = ~subdetector_mask

    train_data = DatasetTemplate(
        data_np_nd=train_data, mask=mask, **train_arg)
    
    train_data.dataset_name = dataset_name
    train_data.subdetector_name = subdetector_name

    util.save_pickle("{}/{}_{}_{}.pkl".format(data_path, 
        dataset_name, subdetector_name, data_filename), train_data)


if __name__ == '__main__':
    """Main entry function."""
    # constructing argument parsers
    parser = argparse.ArgumentParser(
        description="modeling dataset preparation")
    
    parser.add_argument('-s', '--subdetector_name', type=str,  
                        default="",
                        help='detector selection')
    parser.add_argument('-d', '--datatype', type=str, choices=['iid', 'ts'], default="ts",
                        help='data type selection')
    parser.add_argument('-m', '--memorysize', type=int, default=5,
                        help='model temporal memory size')
    parser.add_argument('-ts', '--timestep_size', type=int,
                        help='number of overlapping timestep_size jump')
    parser.add_argument('-is', '--input_scaling_alg', type=str,
                        choices=['minmax', "max", None],
                        default=None,
                        help='scaling or transformation alg for the input data')
    parser.add_argument('-dp', '--data_path', type=str, default=data_path,
                        help='data_path')
    parser.add_argument('-ds', '--dataset', type=str,
                        # choices=['HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB'], 
                        default="",
                        help='data source selection')
    parser.add_argument('-np', '--data_dirpath', default=None, type=str,
                        help='data_dirpath path to prepare numpy dataset')
    parser.add_argument('-td', '--data_filename',
                        type=str, default='train_dataset',
                        help='train data in dataset filename')
    parser.add_argument('-it', '--istrain', action='store_true',
                        default=False,
                        help='istrain')
    parser.add_argument('-tr', '--trainnref_filename', type=str, default=None,
                        help='train data in dataset filename for referencing such as scaling')

    args = parser.parse_args()
    print(args)

    args = vars(args)
    prepare_model_temporal_torch_dataset(**args)


    # checking prepare dataset
    model_data_ts = util.load_pickle("{}/{}_{}_{}.pkl".format(data_path,
            args["dataset"].replace('/', '_'), args["subdetector_name"], args["data_filename"]))

    print(model_data_ts.shape())
    print(model_data_ts.mask.shape)
    print(model_data_ts.get_np()[0].shape)

# non temporal
# TRAIN
# python model_datasets.py -it -s he -d ts -m 1 -is minmax -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -np "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data\HBHE\he_train_dataset" -td he_train_dataset_iid 
# TEST
# python model_datasets.py -s he -d ts -m 1 -is minmax -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -np "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data\HBHE\he_test_dataset" -td he_test_dataset_iid -tr he_train_dataset_iid


# temporal
# TRAIN
# python model_datasets.py -it -s he -d ts -m 5 -is minmax -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -np "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data\HBHE\he_train_dataset" -td he_train_dataset_ts 

# TEST
# python model_datasets.py -s he -d ts -m 5 -is minmax -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -np "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data\HBHE\he_test_dataset" -td he_test_dataset_ts -tr he_train_dataset_ts
