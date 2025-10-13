# -*- coding: utf-8 -*-
"""
Created on Fri Feb 7 17:39:37 2025

@author: Mulugeta W.Asres
@email: muleina2000@gmail.com, mulugetawa@uia.no, mulugeta.asres@cern.ch
"""

import os
import sys, copy
import time
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(current_path))
import torch 
import utilities as util
from torch.utils.data import Dataset
import numpy as np
from torchsummaryX import summary

def dqm_ad_detect_processor_online(ad_model_profile: dict, dqmObj: object, prediction_setting: dict={}, **kwargs):
    print("dqm_ad_detect_processor_online...")    
    
    prediction_setting["isstateful"] = prediction_setting['inference_mode'] == "online_stateful"

    isplot = kwargs.pop('isplot', False)

    anomaly_std_th = prediction_setting.get('anomaly_std_th', 25)

    aml_temporal_types = prediction_setting.get('aml_temporal_types', ['_window_spatial_scaled'])

    report_vars = prediction_setting.get(
        'report_vars', ['target_data', 'pred_err_window_spatial_scaled'] + ['pred_err{}_aml'.format(aml_temporal_type) for aml_temporal_type in aml_temporal_types])

    mask = dqmObj.get_attr("mask")
    print("mask", type(mask), mask.sum())
    mask = torch.BoolTensor(mask)

    print("prediction...")
   
    ts = time.time()

    print(ad_model_profile["admodel"].train_data_config["input_scaler"])
    input_data = dqmObj.get_digi_map(norm=True)
    print(input_data.shape, input_data.dtype, type(input_data))

    pred_dict_anml = ModelInterface().predict(ad_model_profile["admodel"], input_data,
                                                        isspatial=True, mask=mask, isplot=isplot, **prediction_setting)

    print("eval time (min): ", (time.time()-ts)/60)    

    pred_err_spatial_scaled = ad_model_profile["admodel"].pred_err_spatial_scaler.transform(pred_dict_anml["pred_err_spatial"], ignore_mean=True)
    pred_err_window_spatial_scaled = ad_model_profile["admodel"].pred_err_window_spatial_scaler.transform(pred_dict_anml["pred_err_window_spatial"], ignore_mean=True)

    pred_err_spatial_scaled = util.TH3_mask_onnx(pred_err_spatial_scaled.squeeze(-1), mask, 0).unsqueeze(-1)
    pred_err_window_spatial_scaled = util.TH3_mask_onnx(pred_err_window_spatial_scaled.squeeze(-1), mask, 0).unsqueeze(-1)

    pred_dict_anml["pred_err_spatial_scaled"] = pred_err_spatial_scaled
    pred_dict_anml["pred_err_window_spatial_scaled"] = pred_err_window_spatial_scaled

    print(report_vars)
    print(type(report_vars))
    print(pred_dict_anml.keys())
    for aml_temporal_type in aml_temporal_types:  
        
        print("AD thresholding...")
        anml_pred_lbl_np = torch.BoolTensor(pred_dict_anml["pred_err{}".format(aml_temporal_type)].abs()>anomaly_std_th).type(torch.uint8)
        print(anml_pred_lbl_np.shape)
        print("total number of anomalies: ", anml_pred_lbl_np.sum())

        aml_var = 'pred_err{}_aml'.format(aml_temporal_type)
        print(aml_var)

        pred_dict_anml[aml_var] = anml_pred_lbl_np

    print(report_vars)
    for key, value in pred_dict_anml.items():
        try:
            print(key, value.shape)
        except Exception as ex:
            print(key, ex)

    return pred_dict_anml

def model_summary(modelObj, inputdata_shape):
    inputdata_shape = inputdata_shape if isinstance(
        inputdata_shape, tuple) else tuple(inputdata_shape)
    print(inputdata_shape)
    dummy_input = torch.zeros(inputdata_shape)
    print("model_input shape: ", dummy_input.shape)
    summary(modelObj, dummy_input.to(modelObj.device))

def load_prod_model(model_filename: str, model_dirpath: str, model_filepath: str = None) -> object:
    
    if model_filepath is None:
        model_filepath = "{}//".format(util.removesuffix(
            model_dirpath, "//")) + "{}.pkl".format(util.removesuffix(model_filename, ".pkl"))

    print("loading prod model model_filepath: ", model_filepath)
    modelProdObj = util.load_pickle(filepath=model_filepath)
    return modelProdObj

def save_prod_model(model_filename, modelProdObj, model_dirpath: str):
    print("saving prod model model_dirpath: ", model_dirpath)
    util.save_pickle(f"{model_dirpath}/{model_filename}.pkl", modelProdObj)

def load_ad_model_wrapper(model_filename, model_dirpath, show_models=False, isplot=False, **kwargs):
   
    print("model_dirpath: ", model_dirpath)

    model_dict = load_prod_model(model_filename=model_filename, model_dirpath=model_dirpath)
    
    train_err_hist = util.load_csv(f"{model_dirpath}/ae_pred_err_spatial_hist.csv", index_col=0)
    train_err_hist.shape
    train_err_hist.head()
    train_err_window_hist = util.load_csv(f"{model_dirpath}/ae_pred_err_window_spatial_hist.csv", index_col=0)
    train_err_window_hist.shape
    train_err_window_hist.head()

    admodel = model_dict["model"]
    admodel.encode_num_outputs = 1
    admodel.get_encoded = False
    admodel.model_interprate = False
    admodel.keep_hidden_states = False
    admodel.model_data_format = False
    admodel = admodel.to("cpu")

    print("admodel.spatial_dims: ", admodel.spatial_dims)
    print("admodel.spatial_dims_full: ", admodel.spatial_dims_full)
    admodel.feature_dim = 1

    # load AD model
    admodel_enc = None

    if show_models:
        if hasattr(admodel, 'spatial_dims'):
            if hasattr(admodel, "spatial_dims_full"): 
                admodel.spatial_dims_sliced = admodel.spatial_dims
                admodel.spatial_dims =  admodel.spatial_dims_full

            model_summary(admodel, [1, admodel.memorysize] + list(
                admodel.spatial_dims) + [admodel.feature_dim])  # from torchsummaryX
        else:
            model_summary(admodel, [1, admodel.memorysize, admodel.feature_dim])

    return {"model_dict": model_dict, "admodel": admodel, "admodel_enc": admodel_enc,
            "train_err_hist": train_err_hist, "train_err_window_hist": train_err_window_hist,
            "model_dirpath": model_dirpath}

def get_model_ts_data_spatial_onnx(data_np_nd, **kwargs):
    """
    slices on time window on the first dim=ls
    data_np_nd: 5D [lsxietaxiphixdepthxfeature]
    output: data_np_nd: 6D [slicesxtsxietaxiphixdepthxfeature]

    datatype="ts", timewindow_size=5, istoslice=True, istrain=False, modeltype="ae"
    """
    
    datatype = kwargs.get("datatype", "ts")
    timewindow_size = kwargs.get("timewindow_size", None)
    timewindow_size = kwargs.get("memorysize", None) if timewindow_size is None else timewindow_size
    istoslice = kwargs.get("istoslice", True)

    print("spatial ts slicer: ", data_np_nd.shape, datatype, timewindow_size, istoslice)

    @torch.jit.script
    def slicing_handler(data_np_nd:torch.Tensor, timewindow_size:int):
        '''
        input data: data_np_nd -> 3D numpy
        output: sliced_data -> tuple of 3D numpy
        timewindow_size is size of the sliding time window past memory
        '''

        print(data_np_nd.shape[0], timewindow_size,  data_np_nd.shape[0] < timewindow_size)
        if data_np_nd.shape[0] < timewindow_size:
            return (data_np_nd.unsqueeze(1).detach().clone(), )
        
        input_shape = data_np_nd.shape
        n = input_shape[0]
        n_slices = (n - timewindow_size)//timewindow_size + 1
        print((n_slices, timewindow_size) + input_shape[1:])
        x_sliced_data = torch.zeros((n_slices, timewindow_size) + input_shape[1:])
        for i in range(0, n_slices):
            x_sliced_data[i] = data_np_nd[i*timewindow_size:i*timewindow_size + timewindow_size].clone()

        return (x_sliced_data, )

    @torch.jit.script
    def unslicing_handler(sliced_data: torch.Tensor, timewindow_size:int):
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
            data_np_nd = torch.cat((data_np_nd[:timewindow_size],
                                         torch.cat(([data_np_nd[i:i+timewindow_size]
                                                          for i in range(2*timewindow_size-timewindow_size, data_np_nd.shape[0], timewindow_size)]))))

        return data_np_nd


    slice_status = False

    if isinstance(data_np_nd, np.ndarray):
        data_np_nd = torch.from_numpy(data_np_nd)
      
    if datatype == "ts":
        if istoslice:
            print("generating time series slices...")

            if not timewindow_size:
                raise "timewindow_size must be provided to accurately agg slices of sliding windows!"

           
            print(f"timewindow_size: {timewindow_size}")

            sliced_data = slicing_handler(
                data_np_nd, timewindow_size)

            print(f"sliding timewindow size: {timewindow_size} \ninput shape:{data_np_nd.shape}, output shape: {sliced_data[0].shape}")

            slice_status = True

            return sliced_data, slice_status
        else:
            print("reconstructing time series from slices...")
            '''
            timewindow_size is history window in rec ae 
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

def mae_torch(y_true, y_pred, multioutput=True, axis=(0)):
    def torch_nanmean(v, *args, inplace=False, **kwargs):
        if not inplace:
            v = v.detach().clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

    if multioutput:
        return torch_nanmean(np.abs(y_true - y_pred), axis=axis)
    else:
        return np.nanmean(np.abs(y_true - y_pred))

def TH3_to_Emap_adjust_4d_torch(TH3Obj_np,  ieta_axis_idx=1):
    '''
    Adjust map to match emap
    '''
    TH3Obj_np_ = TH3Obj_np.clone()
    ieta_idx = TH3Obj_np_.shape[ieta_axis_idx]//2
    TH3Obj_np_[:, ieta_idx:-1, :] = TH3Obj_np_[:, ieta_idx+1:, :].clone()
    
    TH3Obj_np_[:, :, :-1] = TH3Obj_np_[:, :, 1:].clone()
    TH3Obj_np_[:, :, -1] = 0  # unmasked but no data
    
    return TH3Obj_np_

def TH3Obj_normalizer_online_onnx(digi_map_th3_np, **kwargs):
    '''
    normalize raw TH3data with run lumi rec, run_ls_lumi
    '''
    print("TH3Obj_normalizer_online_onnx...", digi_map_th3_np.shape)

    # normalizer_vars = kwargs.get("normalizer_vars", ["Rec. Lumi (pb^{-1})", "NumEvents (10^3)"])
    normalizer_vars = kwargs.get("normalizer_vars", ["NumEvents (10^3)"])
    if not isinstance(normalizer_vars, list):
        normalizer_vars = [normalizer_vars]
    mapnorm_mode = kwargs.get("mapnorm_mode", "self_perdepth_sum")
    normalizer_vector = kwargs.get("normalizer_vector", [0, 0])
   
    print("mapnorm_mode:{} ...".format(mapnorm_mode))
    ts = time.time()

    if isinstance(digi_map_th3_np, np.ndarray):
        digi_map_th3 = torch.tensor(digi_map_th3_np, dtype=torch.float64)
    elif isinstance(digi_map_th3_np, torch.FloatTensor) or torch.is_tensor(digi_map_th3_np):
        digi_map_th3 = digi_map_th3_np
    else:
        raise "Invalid datatype for input data" 

    scaling_factor_afternorm = kwargs.get("scale_norm_depth",  1) # depreciated

    if mapnorm_mode == "raw":
        return digi_map_th3, digi_map_th3.detach().clone()

    elif mapnorm_mode in ["vars_prod", "reg_model_sum", "reg_model_perdepth_sum"]:

        data_np_nd_raw_ = digi_map_th3.detach().clone()
        print('data_np_nd_raw_: ', data_np_nd_raw_.shape)

        isnan_ls = torch.sum(data_np_nd_raw_, axis=(1, 2, 3, 4))
        isnan_ls = isnan_ls==0
        print('data_np_nd_raw_: ', data_np_nd_raw_.shape)

        if mapnorm_mode == "vars_prod":
            print("use vars_prod per ls renormalization...")
            print(normalizer_vars, normalizer_vector)
            if isinstance(normalizer_vector, np.ndarray):
                normalizer_vector = torch.tensor(normalizer_vector, dtype=torch.float64)
            elif torch.is_tensor(normalizer_vector):
                pass
            else:
                raise "type error normalizer_vector. it must be tensor"

            # normalizer_var = "N({})".format("*".join(normalizer_vars) if len(normalizer_vars) > 1 else normalizer_vars[0])
            normalizer_ls = normalizer_vector.prod(axis=1)

        elif mapnorm_mode.startswith("reg_model"):
            print("use regression model normalizer...no longer used!")
            # no longer used
            pass
        
        # avoid error div by zero
        normalizer_ls[normalizer_ls==0] = 1

        digi_map_th3_normalized = torch.div(data_np_nd_raw_, normalizer_ls.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1))

        # scaling
        print(f"scaling normlized values by {scaling_factor_afternorm}...")
        scale_factor = scaling_factor_afternorm.unsqueeze(0).unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        print(digi_map_th3_normalized.sum(), scale_factor)
        digi_map_th3_normalized = digi_map_th3_normalized*scale_factor
        
    else:
        raise "Undefine map renormalization method. Try with valid options."

    print(time.time()-ts)

    print(digi_map_th3.sum(), digi_map_th3_normalized.sum())
    return digi_map_th3, digi_map_th3_normalized

class HCAL_DQM_ONLINE_ONNX:

    def __init__(self, sys_sel="he", mask=None, **kwargs):

        self.monitor_data = {"raw": [], "normalized": []}
        self.anomaly = {}
        self.is_anomaly_predicted = False
        self.sys_sel = sys_sel
        self.segment_config = {}
        self.segment_region = {}
        self.mask = torch.BoolTensor(mask)
        self.isdebug = kwargs.get("isdebug", False)
        self.mon_quantity = kwargs.pop("mon_quantity", "digi_occupancy")
        self.normalizer_vars = kwargs.get("normalizer_vars", [])
        self.mapnorm_mode = kwargs.get("mapnorm_mode", "self_perdepth_sum")
        self.normalizer_model = kwargs.get("normalizer_model", None)
        self.scale_norm_depth = torch.tensor(kwargs.get("scale_norm_depth", 1))

        print("mapnorm_mode: {}, normalizer_vars: {}".format(
            self.mapnorm_mode, self.normalizer_vars))
        print("isdebug mode: ", self.isdebug)

    def load_map_data(self, datainput_dict, **kwargs):
        self.run_id = kwargs.pop("run_id", None)
        self.year = kwargs.pop("year", 2022)
        self.isemap_th3_adjust = kwargs.pop("isemap_th3_adjust", True)
        self.ls_range = kwargs.pop("ls_range", None)
        digi_map_th3_torch, input_data_exo = datainput_dict["input_data"], datainput_dict["input_data_exo"]
        run_ls_setting_dict = {k: v for k, v in zip(
            self.normalizer_vars, input_data_exo.transpose(0, 1))}
        print('mask type: ', type(self.mask), self.mask.dtype)
        print('digi_map_th3_torch type: ', type(digi_map_th3_torch))
        print('mask: ', self.mask.shape)
        print('digi_map_th3_torch: ', digi_map_th3_torch.shape)

        digi_map_th3_torch = digi_map_th3_torch.squeeze(-1)
        print('digi_map_th3_torch: ', digi_map_th3_torch.shape)

        print(digi_map_th3_torch.sum(), self.sys_sel, self.year)
        if self.isemap_th3_adjust:
            digi_map_th3_torch = TH3_to_Emap_adjust_4d_torch(digi_map_th3_torch)
            print('digi_map_th3_torch after isemap_th3_adjust: ', digi_map_th3_torch.shape)

        print(digi_map_th3_torch.sum())

        digi_map_th3_torch = util.TH3_mask_onnx(digi_map_th3_torch, self.mask, 0)

        digi_map_th3_torch = digi_map_th3_torch.unsqueeze(-1)

        print('digi_map_th3_torch: ', digi_map_th3_torch.shape)

        missing_ls = digi_map_th3_torch[:, :, :, :, 0].isnan().to(torch.int8)
        missing_ls = util.TH3_mask_onnx(missing_ls, self.mask, 0)
        missing_ls = missing_ls.unsqueeze(-1)
        missing_ls = missing_ls.sum(axis=(1, 2, 3, 4)) > 0
        self.valid_status_ls = (~missing_ls).clone().to(torch.uint8)
        print("From {} maps, valid_status_ls in preprocessing: {}...".format(
            self.valid_status_ls.sum(), self.valid_status_ls.shape))

        digi_map_th3_torch, digi_map_th3_torch_normalized = TH3Obj_normalizer_online_onnx(digi_map_th3_torch, 
                                                                                             scale_norm_depth=self.scale_norm_depth,
                                                                                             mapnorm_mode=self.mapnorm_mode, 
                                                                                             normalizer_model=self.normalizer_model, 
                                                                                             normalizer_vars=self.normalizer_vars,
                                                                                             normalizer_vector=input_data_exo, 
                                                                                             **kwargs)
        
        digi_map_th3_torch, digi_map_th3_torch_normalized = digi_map_th3_torch.to(torch.float32), digi_map_th3_torch_normalized.to(torch.float32)

        self.dqm_variables = self.mon_quantity
        self.digi_map_th3 = digi_map_th3_torch
        self.digi_map_th3_norm = digi_map_th3_torch_normalized
        self.run_ls_setting = run_ls_setting_dict

    def get_digi_map(self, norm=True):
        if norm:
            return self.digi_map_th3_norm
        else:
            return self.digi_map_th3

    def get_attr(self, key):
        return getattr(self, key)

    def set_attr(self, key, value):
        setattr(self, key, value)

class TorchDatasetTemplate(Dataset):
    '''
    for onnx using fully torch, inclusing data and scaler
    '''
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
        
        self.iststw_overlapped = kwargs.get("iststw_overlapped", False) # no longer used
        self.memorysize = self.timewindow_size

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
        data_np_nd = self.pretransform_data(data_np_nd, sel_variable_idx=None)  

        if self.datatype != "ts":  # for dim compatability
            self.timewindow_size = 1
            self.datatype = "ts"

        data_np, self.is_tssliced = get_model_ts_data_spatial_onnx(data_np_nd.detach(), 
                                                                    datatype=self.datatype, 
                                                                    timewindow_size=self.timewindow_size,
                                                                    istoslice=True)  # return array
  

        # data_np, self.is_tssliced = get_model_ts_data_spatial(data_np_nd, datatype=self.datatype, timewindow_size=self.timewindow_size, istoslice=True)  # return array

        print("after slicing sizes:")
        print(data_np[0].shape)
        data_np = [torch.FloatTensor(d) for d in list(data_np)]

        data_size = data_np[0].shape
        print("prepared sliced data: {}".format(data_size))
       
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
        print("data scaling via data transformation: {}...".format(
            self.input_scaling_alg))

        if self.input_scaling_alg and self.is_tssliced:
            print("scaling sliced times series signal is not allowed.")
            return data_np_nd

        if not self.input_scaling_alg:
            print("no data preprocessing scaler algorithm is not selected or trained.")
            return data_np_nd

        if not self.isdata_scaled:
            if self.input_scaler:
                print("transform data scaling using {}...".format(
                    self.input_scaling_alg))
                # change shape into [samplexfeature_to_scale_dim]
                print(data_np_nd.shape, data_np_nd.dtype, type(data_np_nd))
                data_np_nd_reshaped_scaled = self.input_scaler.transform(
                    data_np_nd.view(data_np_nd.shape[0], -1).detach().to(torch.float64))
                # restore original dim
                data_np_nd = data_np_nd_reshaped_scaled.view(data_np_nd.shape).to(torch.float32)
                #data_np_nd = data_np_nd - 1e-5
                self.isdata_scaled = True
            else:
                print(
                    "no data preprocessing scaler algorithm is not selected or trained.")
        else:
            print("data is already transformed using {}.".format(
                self.input_scaling_alg))
        return data_np_nd

    def get_np(self):
        '''
        returns numpy equivalent of the tensor samples
        '''
        return self.samples[0].detach().numpy(), self.samples[0].detach().numpy()

    def get_tensor(self):
        return self.samples[0][..., self.features_idx].detach().clone(), self.samples[0][..., self.targets_idx].detach().clone()
 
class TorchMinMaxScaler():
    def __init__(self, feature_range=(0,1), skl_scaler=None):
        self.feature_range = feature_range
        self.skl_scaler = skl_scaler
        if self.skl_scaler is not None:
            self._skscaler_to_torch_mapper(self.skl_scaler)
        else:
            if self.feature_range[0] >= self.feature_range[1]:
                raise ValueError(
                        "Minimum of desired feature range must be smaller than maximum. Got %s."
                        % str(feature_range)
                    )
    def _skscaler_to_torch_mapper(self, skl_scaler):
        self.scale_ = torch.tensor(skl_scaler.scale_)
        self.min_ = torch.tensor(skl_scaler.min_)
        self.feature_range = torch.tensor(skl_scaler.feature_range)
        self.scaler_in_min = torch.tensor(skl_scaler.data_min_)
        self.scaler_in_max = torch.tensor(skl_scaler.data_max_)
        
    def fit(self, X):
        if self.skl_scaler is None:
            self.scaler_in_min = X.min(axis=0)
            self.scaler_in_max = X.max(axis=0)
            self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (self.scaler_in_max - self.scaler_in_min)
            self.min_ = self.feature_range[0] - self.scaler_in_min * self.scale_
        else:
            self.skl_scaler.fit(X)
            self._skscaler_to_torch_mapper(self.skl_scaler)

    def transform(self, X):
        return X*self.scale_ + self.min_
        # return X.mul_(self.scale_).add_(self.min_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        # X = X.type(torch.float64) - self.min_
        X = X - self.min_
        X = X/self.scale_
        return X
        # return X.sub_(self.min_).div_(self.scale_)

class TorchMaxScaler():
    def __init__(self, skl_scaler=None):
        self.feature_range = torch.tensor([0, 1])
        if skl_scaler is not None:
            self._skscaler_to_torch_mapper(skl_scaler)

    def _skscaler_to_torch_mapper(self, skl_scaler):
        print("self.skl_scaler.scaler_values: ", skl_scaler.scaler_values.shape)
        self.scaler_in_max = torch.tensor(skl_scaler.scaler_values).to(torch.float64)
        self.scale_ = self.scaler_in_max
        print("self.scale_: ", self.scale_.shape)
                   
    def fit(self, X:torch.Tensor):
        self.scaler_in_max = X.max(axis=0)
        self.scaler_in_max[self.scaler_in_max==0] = 1
        self.scale_ = self.scaler_in_max

    def transform(self, X:torch.Tensor):
        print(X.shape)
        return X/self.scale_
        # return torch.div(X, self.scale_)

    def inverse_transform(self, X:torch.Tensor):
        print(X.shape)
        return X*self.scale_

class PredErrorScaler():
    def __init__(self,  mean_err_train=None, std_err_train=None, **kwargs):
        self.reshape_dim = kwargs.get("reshape_dim", None)
        self.mask = kwargs.get("mask", None)
        self.mean_err_train = mean_err_train
        self.std_err_train = std_err_train
        self.std_err_train[self.std_err_train==0] = torch.mean(self.std_err_train) # avg std scaling iff std is zero

        if self.reshape_dim is not None:
            self.mean_err_train = self.mean_err_train.reshape(self.reshape_dim) if not isinstance(self.mean_err_train, int) else self.mean_err_train
            self.std_err_train = self.std_err_train.reshape(self.reshape_dim) if self.std_err_train is not None else self.std_err_train
        
        if self.mask is not None:
            self.mask =  torch.from_numpy(self.mask).type(torch.bool)

    def transform(self, pred_err: torch.Tensor, ignore_mean: torch.BoolTensor=False):
        pred_err_norm = pred_err

        if self.mean_err_train is not None:
            if not ignore_mean:
                pred_err = pred_err - self.mean_err_train

        if self.std_err_train is not None:
            pred_err_norm = torch.divide(pred_err.abs(), self.std_err_train)

        if self.mask is not None:
            pred_err_norm[:, self.mask, :] = torch.nan

        return pred_err_norm

class ModelInterface():
  
    def __init__(self):
        pass

    def convert_to_unsliced(self, modelObj, X, iststw_overlapped=False, istrain=False, pre_inv_scale=False):
        '''
        return unsliced version, numpy multidim 5D: [lsxietaxiphixdepthxfeature]
        '''
        print(X.sum())
        unsliced = get_model_ts_data_spatial_onnx(X,
                                                      datatype=modelObj.train_data_config["datatype"],
                                                      memorysize=modelObj.train_data_config["memorysize"],
                                                      istoslice=False, istrain=istrain)[0]
        print(unsliced.sum())
        return unsliced

    def scaling_inverse_transform(self, input_scaler, data_np_nd):
        data_np_nd_reshaped_scaled = input_scaler.inverse_transform(
            data_np_nd.detach().to(torch.float64).reshape(data_np_nd.shape[0], -1))
        # restore original dim
        data_np_nd = data_np_nd_reshaped_scaled.reshape(data_np_nd.shape).to(torch.float32)
        #data_np_nd = data_np_nd - 1e-5
        return data_np_nd

    def prepare_inputdata(self, modelObj, data, **kwargs):
        if not isinstance(data, TorchDatasetTemplate):
            if hasattr(modelObj, "train_data_config"):

                data_setting = modelObj.train_data_config.copy()

                [data_setting.pop(k) for k in ["istrain", "iststw_overlapped",
                                               "isdata_scaled", "is_tssliced"] if k in data_setting.keys()]
            else:
                # compatibility to old models

                data_setting = {
                    "datatype": modelObj.datatype,
                    "input_scaler": modelObj.input_scaler,
                    "memorysize": modelObj.memorysize,
                    "data_vars": modelObj.columns,
                    "features": modelObj.columns,
                    "features_ts": modelObj.columns,
                    "targets": modelObj.columns
                }

                data_setting["features_idx"] = [i for i, var in enumerate(
                    data_setting["data_vars"]) if var in data_setting["features"]]
                data_setting["targets_idx"] = [i for i, var in enumerate(
                    data_setting["data_vars"]) if var in data_setting["targets"]]
                modelObj.train_data_config = data_setting.copy()

            data = TorchDatasetTemplate(data, istrain=False,
                                   # istoslice=True,
                                   **data_setting)

        return data
    
    def predict(self, modelObj, data_np_nd, **kwargs):

        keep_hidden_states = kwargs.pop("keep_hidden_states", False)
        skip_unscaling = kwargs.get("skip_unscaling", False)
        mask = kwargs.get("mask", None) # keep zero, hide 1
        isstateful = kwargs.get("isstateful", False) # isstateful for single t-step with states input and output, false: states are kept in the model object

        print("isstateful: ", isstateful)
        
        if mask is None:
            raise "please input mask and mask can not be None!"

        data_np_nd = data_np_nd.squeeze(-1)
        data_np_nd = util.TH3_mask_onnx(data_np_nd, mask, 0)
        data_np_nd = data_np_nd.unsqueeze(-1)
    
        data = self.prepare_inputdata(
            modelObj, data_np_nd, **kwargs)

        features_name = data.get_attr("features")
        targets_name = data.get_attr("targets")

        print(features_name, targets_name)

        input_data, target_data = data.get_tensor()
        print(input_data.shape, target_data.shape)
        input_data_sliced_timewindow = input_data.shape[1]
        
        print("model prediction...")
        
        if not isstateful:
            pred_datas = []
            for i in range(len(input_data)):
                pred = modelObj(input_data[i:i+1], 
                                keep_hidden_states=keep_hidden_states, 
                                return_with_encoded=True, **kwargs)

                if isinstance(pred, tuple):
                    pred_datas.append(pred[0])
                else:
                    pred_datas.append(pred)

            pred_data = torch.cat(pred_datas, dim=0)

            e_rnn_hidden, d_rnn_hidden = None, None
        else:
            pred_datas = []
    
            e_rnn_hidden = kwargs.pop("e_rnn_hidden", [])
            d_rnn_hidden = kwargs.pop("d_rnn_hidden", [])
            print("keep_hidden_states" in kwargs)
    
            for i in range(len(input_data)):
                pred = modelObj.forward(input_data[i:i+1], 
                                e_rnn_hidden, d_rnn_hidden, 
                                keep_hidden_states=keep_hidden_states, 
                                return_with_encoded=True, 
                                **kwargs)
            
                pred_datas.append(pred[0])
                
                e_rnn_hidden, d_rnn_hidden = pred[-2], pred[-1]

                for l, x_layer in enumerate(e_rnn_hidden):    
                    print(e_rnn_hidden[l][0].dtype, e_rnn_hidden[l][0].sum(), e_rnn_hidden[l][1].sum())
                    print(d_rnn_hidden[l][0].dtype, d_rnn_hidden[l][0].sum(), d_rnn_hidden[l][1].sum())

            pred_data = torch.cat(pred_datas, dim=0)

            print("input_data.shape: ", input_data.shape)

        print("done!")
        print("input_data.shape: ", input_data.shape)
        
        isnan_idx = input_data.isnan().sum(axis=tuple(torch.arange(len(input_data.shape))[1:])) > 0
        num_nan = isnan_idx.sum()
        print("window slices with nan size: {}, {:0.3f}\\%".format(
            num_nan, 100*num_nan/input_data.shape[0]))
        
        pred_data_shape = pred_data.shape

        if modelObj.train_data_config["datatype"] == "ts":
            pred_err_window_spatial = torch.zeros(pred_data_shape[0:1] + pred_data_shape[2:])

            for i in range(pred_data_shape[-1]):
                err_window = mae_torch(target_data[:, :, :, :, :, i], pred_data[:, :, :, :, :, i], multioutput=True, axis=1)  # mean across the time window dim

                pred_err_window_spatial[:, :, :, :, i] = err_window
                print("pred_err_window_spatial: ", pred_err_window_spatial.shape)

        # convert into 3D [slice, timewindow, features] into 2D data [time, features]
        if modelObj.train_data_config["datatype"] == "ts":
            print("time slice reconstruction...")

            input_data = self.convert_to_unsliced(
                modelObj, input_data, istrain=False)
            target_data = self.convert_to_unsliced(
                modelObj, target_data, istrain=False, pre_inv_scale=True)
            pred_data = self.convert_to_unsliced(
                modelObj, pred_data, istrain=False, pre_inv_scale=True)
            
            # single windowed recostruction error score for each window

            output_windowsize = input_data_sliced_timewindow # incase the data length is smaller than time window 
                
            pred_err_window_spatial = pred_err_window_spatial.unsqueeze(1)

            pred_err_window_spatial = pred_err_window_spatial.repeat(1, output_windowsize, *tuple([1]*(pred_err_window_spatial.ndim-2)))
            print("pred_err_window_spatial:", pred_err_window_spatial.shape)
            pred_err_window_spatial = self.convert_to_unsliced(
                modelObj, pred_err_window_spatial, istrain=False, pre_inv_scale=True)
            print("pred_err_window_spatial:", pred_err_window_spatial.shape)
            
        print("apply mask to output target_data and pred_data...")
        print("mask: ", mask.shape, mask.sum(), type(mask), mask.dtype)
        print(target_data.shape, pred_data.shape)
          

        target_data = util.TH3_mask_onnx(target_data.squeeze(-1), mask, 0).unsqueeze(-1)
        pred_data = util.TH3_mask_onnx(pred_data.squeeze(-1), mask, 0).unsqueeze(-1)

        pred_err_spatial = (pred_data - target_data).abs()
        
        # mean across the time window dim, keep score accross the spatial and feature dims
        print("pred_err_spatial: ", pred_err_spatial.shape)
        
        
        if ("input_scaler" in modelObj.train_data_config.keys()) and (not skip_unscaling):
            if modelObj.train_data_config["input_scaler"] is not None:
                print("input scaler inverse transforming...")
  
                input_data = self.scaling_inverse_transform(
                    modelObj.train_data_config["input_scaler"], input_data)
                target_data = self.scaling_inverse_transform(
                    modelObj.train_data_config["input_scaler"], target_data)
                pred_data = self.scaling_inverse_transform(
                    modelObj.train_data_config["input_scaler"], pred_data)

        print("input_data: {}, target_data: {}, pred_data: {}".format(
            input_data.shape, target_data.shape, pred_data.shape))
       
        target_data = target_data.squeeze(-1)
        pred_data = pred_data.squeeze(-1)
        pred_err_spatial = pred_err_spatial.squeeze(-1)
        pred_err_window_spatial = pred_err_window_spatial.squeeze(-1)
        target_data = util.TH3_mask_onnx(target_data, mask, 0)
        pred_data = util.TH3_mask_onnx(pred_data, mask, 0)
        pred_err_spatial = util.TH3_mask_onnx(pred_err_spatial, mask, 0)
        pred_err_window_spatial = util.TH3_mask_onnx(pred_err_window_spatial, mask, 0)
        target_data = target_data.unsqueeze(-1)
        pred_data = pred_data.unsqueeze(-1)
        pred_err_spatial = pred_err_spatial.unsqueeze(-1)
        pred_err_window_spatial = pred_err_window_spatial.unsqueeze(-1)

        return {
            # "input_data": input_data, 
                    "target_data": target_data, 
                    "pred_data": pred_data,
                    "pred_err_spatial": pred_err_spatial,
                    "pred_err_window_spatial": pred_err_window_spatial,
                    "e_rnn_hidden": e_rnn_hidden, 
                    "d_rnn_hidden": d_rnn_hidden
                }  

    def get_output_dim(self, modelObj):
        if modelObj.get_encoded:
            return modelObj.latent_dim
        else:
            return modelObj.feature_dim      


class ModelProdNN(torch.nn.Module):

    def __init__(self, model_type: str, model_profile: dict, predict_processor: object, data_kwargs: dict, model_setting: dict):
        super(ModelProdNN, self).__init__()
        self.model_type = model_type
        self.model_profile = model_profile
        self.predict_processor = predict_processor
        self.data_kwargs = data_kwargs
        self.model_setting = model_setting

    def data_pipeline(self, data_dict: dict, **kwargs):
        dataObj = None
        # isonnx = kwargs.get("isonnx", False)
        
        if self.model_type == "DQM_AD_ONLINE":
            print("data_kwargs: ", self.data_kwargs.keys())

            dataObj = HCAL_DQM_ONLINE_ONNX(**self.data_kwargs, **kwargs)

            dataObj.load_map_data(data_dict, **kwargs)
        else:
            print("data_pipeline")
            raise NotImplementedError
        return dataObj

    def output_pipeline(self):
        pass

    def pypredict(self, dataObj, prediction_setting_):
        print("pypredict...")
        model_evals = {}
        prediction_setting = prediction_setting_
        aml_temporal_types = ['_spatial_scaled', '_window_spatial_scaled']
        prediction_setting_static = {
                    'inference_mode': 'online',
                    'iststw_overlapped': False,
                    'use_post_scale': False,
                    'keep_hidden_states': False,
                    'aml_temporal_types': aml_temporal_types,
                    'report_vars': ['target_data', 'pred_data'] + ['pred_err{}_aml'.format(aml_temporal_type) for aml_temporal_type in aml_temporal_types],
                    }
        prediction_setting.update(prediction_setting_static)
        if prediction_setting.get("e_rnn_hidden__layer_0_state_0", None) is not None:
            prediction_setting["inference_mode"] = "online_stateful"
            prediction_setting["e_rnn_hidden"] = [(torch.tensor(prediction_setting['e_rnn_hidden__layer_0_state_0'], dtype=torch.float32), torch.tensor(prediction_setting['e_rnn_hidden__layer_0_state_1'], dtype=torch.float32)),
                                                                (torch.tensor(prediction_setting['e_rnn_hidden__layer_1_state_0'], dtype=torch.float32), torch.tensor(prediction_setting['e_rnn_hidden__layer_1_state_1'], dtype=torch.float32))]
                
            prediction_setting["d_rnn_hidden"] = [(torch.tensor(prediction_setting['d_rnn_hidden__layer_0_state_0'], dtype=torch.float32), torch.tensor(prediction_setting['d_rnn_hidden__layer_0_state_1'], dtype=torch.float32)),
                                                                (torch.tensor(prediction_setting['d_rnn_hidden__layer_1_state_0'], dtype=torch.float32), torch.tensor(prediction_setting['d_rnn_hidden__layer_1_state_1'], dtype=torch.float32))]
            
            # # drop rnn_hidden__layer_ vars
            keys = list(filter(lambda k: "_state_" in k, prediction_setting.keys()))
            [prediction_setting.pop(k) for k in keys]
                
        inference_mode = prediction_setting["inference_mode"]
        print(inference_mode, self.predict_processor)

        if inference_mode == "offline":
            model_evals = self.predict_processor(
                self.model_profile, dataObj, data_kwargs=self.data_kwargs, prediction_setting=prediction_setting)
        elif inference_mode.startswith("online"):
            model_evals = self.predict_processor(
                self.model_profile, dataObj, data_kwargs=self.data_kwargs, prediction_setting=prediction_setting)
            print('self.predict_processor is completed.')
            for key, value in model_evals.items():
                try:
                    print(key, value.shape)
                except Exception as ex:
                    print(key, ex)

            if inference_mode.startswith("online_stateful"):
                rnn_hidden_dict = {}
                if "e_rnn_hidden" in model_evals.keys():
                    for l, x_layer in enumerate(model_evals["e_rnn_hidden"]):
                        for s, x in enumerate(x_layer):
                            print("e_rnn_hidden: ", type(x), len(x))
                            rnn_hidden_dict["e_rnn_hidden__layer_{}_state_{}_o".format(l, s)] = x

                    for l, x_layer in enumerate(model_evals["d_rnn_hidden"]):
                        for s, x in enumerate(x_layer):
                            print("d_rnn_hidden: ", type(x), len(x))
                            rnn_hidden_dict["d_rnn_hidden__layer_{}_state_{}_o".format(l, s)] = x

                    model_evals.update(rnn_hidden_dict)
                    [model_evals.pop(k) for k in ["e_rnn_hidden", "d_rnn_hidden"]]
            else:
                if "e_rnn_hidden" in model_evals.keys():
                    [model_evals.pop(k) for k in ["e_rnn_hidden", "d_rnn_hidden"]]
                    
            for key, value in model_evals.items():
                try:
                    print(key, value.shape)
                except Exception as ex:
                    print(key, ex)

            return model_evals
        else:
            raise "Undefined inference_mode. Choose offline or online"

    def forward(self, input_data, input_data_exo, *args):
        '''
        to be call by c++, cmssw
        '''

        '''
        datainput: is vector of two vectors [input_data, input_data_exo]
        input_data is 3d-arry of digi-occupancy map
        input_data_exo is vectory of ['REC. LUMI (PB^{-1})', 'NUMEVENTS (10^3)'], [rec_lumi_arr, number_of_events_arr]
        prediction_setting
        output: results (dict, json, or arrays based on outpout_mode)
        '''

        print('input: vector or list')
 
        print(len(args))
        datainput_dict = {'input_data': torch.tensor(
            input_data), 'input_data_exo': torch.tensor(
            input_data_exo)}
        prediction_setting_dict_keys = ['anomaly_std_th', 
                                        'e_rnn_hidden__layer_0_state_0', 'e_rnn_hidden__layer_0_state_1', 
                                        'e_rnn_hidden__layer_1_state_0', 'e_rnn_hidden__layer_1_state_1', 
                                        'd_rnn_hidden__layer_0_state_0', 'd_rnn_hidden__layer_0_state_1',
                                        'd_rnn_hidden__layer_1_state_0', 'd_rnn_hidden__layer_1_state_1']

        prediction_setting_dict = {key: value for key, value in zip(
            prediction_setting_dict_keys[:len(args)], args)} if len(args) > 0 else None
        
        print(datainput_dict['input_data'].shape)
        if prediction_setting_dict is None:
            prediction_setting_dict = {}

        dataObj = self.data_pipeline(datainput_dict, isonnx=True)
        results = self.pypredict(dataObj, prediction_setting_dict)
        return list(results.values())




    
