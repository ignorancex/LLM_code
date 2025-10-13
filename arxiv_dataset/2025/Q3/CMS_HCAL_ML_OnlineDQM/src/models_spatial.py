
"""
Created on Mon Dec 07 17:33:21 2023

@author: Mulugeta W. Asres, mulugetawa@uia.no

DESMOD: ML Models and Interfaces for HCAL DQM Digioccupancy 

"""

import os, sys
import seaborn as sns
import pandas as pd
import numpy as np
from functools import reduce
import numpy as np, pandas as pd
from tqdm import tqdm
import copy
import time
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)
data_path = os.path.abspath(os.path.dirname(current_path)+"/data")
sys.path.append(data_path)
result_path = os.path.abspath(os.path.dirname(current_path)+"/results")

import utilities as util
from model_datasets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class ModelInterface():
    
    def __init__(self):
        pass

    def convert_to_unsliced(self, modelObj, data, istrain=False):
        '''
        return unsliced version, numpy multidim 5D: [lsxietaxiphixdepthxfeature]
        '''
        is_np = isinstance(data, np.ndarray)
        unsliced = get_model_ts_data_spatial(data, datatype=modelObj.train_data_config["datatype"],
                                                      timewindow_size=modelObj.train_data_config["memorysize"],
                                                      istoslice=False, istrain=istrain)[0]
        if is_np:
            unsliced
        else:
            return torch.from_numpy(unsliced)

        return unsliced

    def scaling_inverse_transform(self, input_scaler, data):
        is_np = isinstance(data, np.ndarray)
        if is_np:
            data = torch.from_numpy(data)

        x_reshaped_scaled = torch.tensor(input_scaler.inverse_transform(
            data.detach().to(torch.float64).reshape(data.shape[0], -1)))
        
        # restore original dim
        data = x_reshaped_scaled.contiguous().view(data.shape).to(torch.float32)

        if is_np:
            return data.cpu().detach().numpy()

        return data

    def prepare_inputdata(self, modelObj, data, **kwargs):
        if not isinstance(data, DatasetTemplate):
            if hasattr(modelObj, "train_data_config"):
                data_setting = modelObj.train_data_config.copy()
                [data_setting.pop(k) for k in ["istrain",
                                               "isdata_scaled", "is_tssliced"] if k in data_setting.keys()]
            else:
                data_setting = {
                    "datatype": modelObj.datatype,
                    "input_scaler": modelObj.input_scaler,
                    "memorysize": modelObj.memorysize,
                    "timewindow_size": modelObj.memorysize,
                    "features": "digioccupancy",
                    "targets": "digioccupancy",
                    "features_idx": 0,
                    "targets_idx": 0,
                }
                modelObj.train_data_config = data_setting.copy()

            data = DatasetTemplate(data, istrain=False,
                                   **data_setting)

        return data
    
    def direct_predict(self, modelObj, input_data_org, **kwargs):
        print("model prediction...")
        
        pred_data = []
        input_data = input_data_org.clone()
        input_data[input_data.isnan()] = 0 # very important 
        modelObj.eval()
        torch.cuda.empty_cache()
        proc_t = []
        with torch.no_grad():
            # [slice, window_ts, targets_name]
            for i in range(len(input_data)):
                ts = time.time()
                pred = modelObj(
                    input_data[i:i+1], **kwargs)
                proc_t.append(time.time()-ts)
                if isinstance(pred, tuple):
                    pred_data.append(pred[0])
                else:
                    pred_data.append(pred)
        if len(pred_data):
            pred_data = torch.cat(pred_data, dim=0)
        
        proc_t = np.array(proc_t)
        print("data_size={data_size}, inference time (sec): mean={mean}, median={median}, std={std}, min={lower}, max={upper}".format(data_size=pred_data.shape, mean=np.mean(proc_t), median=np.median(proc_t), std=np.std(proc_t), lower=np.min(proc_t), upper=np.max(proc_t)))
        
        return pred_data, 

    def predict(self, modelObj, data_np_nd, **kwargs):

        keep_hidden_states = kwargs.pop("keep_hidden_states", False)
        mask = kwargs.get("mask", None) # keep zero, hide 1

        modelObj = modelObj.to(device)

        if mask is None:
            mask = modelObj.train_data_config.get("mask", None)
            if mask is None:
                raise "please input mask. mask can not be None."

        mask = torch.from_numpy(mask).type(torch.bool)

        data = self.prepare_inputdata(
            modelObj, data_np_nd, **kwargs)

        features_name = data.get_attr("features")
        targets_name = data.get_attr("targets")

        input_data, target_data = data.get_tensor()
        
        input_data = input_data.to(modelObj.device)
        target_data = target_data.to(modelObj.device)

        print(input_data.shape, target_data.shape)
        
        print("model prediction...")
    
        pred_datas = []
        # with torch.no_grad():
        for i in tqdm(range(len(input_data))):
            pred = modelObj(input_data[i:i+1], keep_hidden_states=keep_hidden_states, return_with_encoded=True, **kwargs)
        
            if isinstance(pred, tuple):
                if len(pred) > 1:
                    pred_datas.append(pred[0])
                else:
                    pred_datas.append(pred[0])
            else:
                pred_datas.append(pred)

        pred_data = torch.cat(pred_datas, dim=0).to(modelObj.device)

        print("done!")
        print("input_data.shape: ", input_data.shape)
        
        isnan_idx = input_data.isnan().sum(axis=tuple(torch.arange(len(input_data.shape))[1:])) > 0
        num_nan = isnan_idx.sum()
        print("window slices with nan size: {}, {:0.3f}\\%".format(
            num_nan, 100*num_nan/input_data.shape[0]))
                                                        
        # windowed recostruction error: mean absolute error
        pred_data_shape = pred_data.shape

        if modelObj.train_data_config["datatype"] == "ts":
            pred_err_window_spatial = torch.zeros(pred_data_shape[0:1] + pred_data_shape[2:])

            for i in range(pred_data_shape[-1]):
                # sklearn.mean_absolute_error() # not support nan
                err_window = util.mae_torch(target_data[..., i], pred_data[..., i], multioutput=True, axis=1)  # mean across the time window dim
                pred_err_window_spatial[..., i] = err_window
                print("pred_err_window_spatial: ", pred_err_window_spatial.shape)

        # convert into 3D [slice, timewindow, features] into 2D data [time, features]
        if modelObj.train_data_config["datatype"] == "ts":
            print("time slice reconstruction...")
            # unsliced np_data [lsxietaxiphixdepthxfeature]
            input_data = self.convert_to_unsliced(modelObj, input_data, istrain=False)
            target_data = self.convert_to_unsliced(modelObj, target_data, istrain=False)
            pred_data = self.convert_to_unsliced(modelObj, pred_data,  istrain=False)
            
            # single windowed recostruction error score for each window
            output_windowsize = modelObj.train_data_config["memorysize"]
        
            pred_err_window_spatial = pred_err_window_spatial.unsqueeze(1)
            pred_err_window_spatial = pred_err_window_spatial.repeat(1, output_windowsize, *tuple([1]*(pred_err_window_spatial.ndim-2)))
            print("pred_err_window_spatial:", pred_err_window_spatial.shape)
            pred_err_window_spatial = self.convert_to_unsliced(
                modelObj, pred_err_window_spatial, istrain=False)
            print("pred_err_window_spatial:", pred_err_window_spatial.shape)
                
        print("apply mask to output target_data and pred_data...")
        print("mask: ", mask.shape, mask.sum(), type(mask), mask.dtype)
        print(target_data.shape, pred_data.shape)

        target_data = target_data.squeeze(-1)
        pred_data = pred_data.squeeze(-1)
        target_data = util.TH3_mask_onnx(target_data, mask, 0)
        pred_data = util.TH3_mask_onnx(pred_data, mask, 0)
        target_data = target_data.unsqueeze(-1)
        pred_data = pred_data.unsqueeze(-1)
        
        # absolute reconstruction error on unsliced time series
        # [lsxtwxietaxiphixdepthxfeature] # tw=1 for non-temporal data
        pred_err_spatial = (pred_data - target_data).abs()
        
        # mean across the time window dim, keep score accross the spatial and feature dims
        print("pred_err_spatial: ", pred_err_spatial.shape)

        if "input_scaler" in modelObj.train_data_config.keys():
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
                    "target_data": target_data, 
                    "pred_data": pred_data,
                    "pred_err_spatial": pred_err_spatial,
                    "pred_err_window_spatial": pred_err_window_spatial
                }  

    def train_perf_report(self, modelObj, data, **kwargs):
        tag = kwargs.get("kwargs", "")
        mask = kwargs.get("mask", None)
        model_dirpath = kwargs.get("model_dirpath", result_path)
        # # masking basedto handle sparsing zeros usingmasking for he
        # mask = ~hcal_subsys_emap_mask

        def predict_error_decision_metrics(pred_data, target_data, tag=""):

            pred_err_spatial = np.abs(pred_data - target_data)

            pred_err_depth = np.nanmean(
                pred_err_spatial, axis=tuple(np.arange(pred_err_spatial.ndim)[1:-2]))
            print("pred_err_depth: ", pred_err_depth.shape)

            pred_err_spatial_df = pd.DataFrame(pred_err_spatial.reshape((-1, np.product(pred_err_spatial.shape[2:])))) # exclude window
            pred_err_spatial_df = pred_err_spatial_df.add_prefix("pixel__")

            pred_err_depth_df = pd.DataFrame(pred_err_depth.transpose(0, 2, 1).reshape(pred_err_depth.shape[0], -1), columns=[
                                            "{}__{}".format(var, d) for var in targets_name for d in range(1, pred_err_depth.shape[1]+1)])

            # stat of reconstruction error for each depth and feature dims
            hist_summary_depth = pd.DataFrame(pred_err_depth_df.describe().T)
            # stat of reconstruction error for each feature dim
            hist_summary_spatial = pd.DataFrame(pred_err_spatial_df.describe().T)
            # print(hist_summary)

            if issave:
                util.save_csv(f"{model_dirpath}/ae_pred_err_depth_hist{tag}.csv", hist_summary_depth,
                         index=True)

                util.save_csv(f"{model_dirpath}/ae_pred_err_spatial_hist{tag}.csv", hist_summary_spatial,
                         index=True)

                # util.plot_grid(pred_err_depth_df, kind="hist", isagg=False,
                #             issave=issave, model_dirpath=f"{model_dirpath}/ae_pred_err_depth_hist{tag}")

            if modelObj.train_data_config["datatype"] == "ts":
                # [lsxtwxietaxiphixdepthxfeature] # tw=1 for non-temporal data
                pred_data_shape = input_data.shape

                # windowed recostruction error: mean absolute error
                # mean across the time window dim, keep score accross the spatial and feature dims
                pred_err_window_spatial = np.zeros(
                    pred_data_shape[0:1] + pred_data_shape[2:])
                # [samplexietaxiphixdepthxfeatures]

                for i in range(pred_data_shape[-1]):
                    # sklearn.mean_absolute_error() # not support nan
                    pred_err_window_spatial[:, :, :, :, i] = util.mae(
                        target_data[:, :, :, :, :, i], pred_data[:, :, :, :, :, i], multioutput=True, axis=1)  # mean across the time window dim
                print("pred_err_window_spatial: ", pred_err_window_spatial.shape)

                # mean across sptial dim except the feature and depth dims
                pred_err_window_depth = np.nanmean(
                    pred_err_window_spatial, axis=tuple(np.arange(pred_err_window_spatial.ndim)[1:-2]))  # cross the spatial dims

                pred_err_window_spatial_df = pd.DataFrame(pred_err_window_spatial.reshape((-1, np.product(pred_err_window_spatial.shape[1:]))))
                pred_err_window_spatial_df = pred_err_window_spatial_df.add_prefix("pixel__")
                
                pred_err_window_depth_df = pd.DataFrame(pred_err_window_depth.transpose(0, 2, 1).reshape(
                    pred_err_window_depth.shape[0], -1), columns=["{}__{}".format(var, d) for var in targets_name for d in range(1, pred_err_window_depth.shape[1]+1)])

                # stat of reconstruction error for each feature dim
                # [samplexdepthxfeatures]
                hist_summary_depth = pd.DataFrame(pred_err_window_depth_df.describe().T)
                hist_summary_window_spatial = pd.DataFrame(pred_err_window_spatial_df.describe().T)
                if issave:
                    util.save_csv(f"{model_dirpath}/ae_pred_err_window_depth_hist{tag}.csv",
                                hist_summary_depth, index=True)
                    util.save_csv(f"{model_dirpath}/ae_pred_err_window_spatial_hist{tag}.csv",
                                hist_summary_window_spatial, index=True)

                    # util.plot_grid(pred_err_window_depth_df, kind="hist",
                    #             issave=issave, filepath=f"{model_dirpath}/ae_pred_err_window_depth_hist{tag}")

        modelObj.eval()  # set to predict mode
        modelObj = modelObj.to(device=device)

        keep_hidden_states = kwargs.get("keep_hidden_states", False)

        issave = kwargs.get("issave", False)

        targets_name = data.get_attr("targets")
        if not isinstance(targets_name, list): targets_name=[targets_name]
        input_data, target_data = data.get_tensor()

        input_data = input_data.to(modelObj.device)
        target_data = target_data.to(modelObj.device)
        
        with torch.no_grad():
            # to allow big data memory when iststw_overlapped is true as large number of slices are generated
            pred_data = torch.zeros(target_data.shape)
            encoded_data = []
            print("model prediction...")
            # [slice, window_ts, targets_name]
            for i in range(len(input_data)):
                pred = modelObj(
                    input_data[i:i+1], keep_hidden_states=keep_hidden_states, return_with_encoded=True)
                if isinstance(pred, tuple):
                    if len(pred) > 1:
                        pred_data[i] = pred[0]
                        encoded_data.append(pred[1])
                    else:
                        pred_data[i] = pred[0]
                else:
                    pred_data[i] = pred

        if len(encoded_data):
            encoded_data = torch.cat(encoded_data, dim=0)

        # [lsxtwxietaxiphixdepthxfeature] # tw=1 for non-temporal data
        target_data = target_data.cpu().detach().numpy()
        pred_data = pred_data.cpu().detach().numpy()

        print(target_data.shape, pred_data.shape)

        # for t in range(pred_data.shape[1]):
        #     pred_data[:, t] = util.TH3Obj_np_data_cleaning(pred_data[:, t])
        
        target_data[:, :, mask, ::] = np.nan
        pred_data[:, :, mask, ::] = np.nan

        predict_error_decision_metrics(pred_data, target_data, tag=tag)
        
        return modelObj, encoded_data

    def get_output_dim(self, modelObj):
        if modelObj.get_encoded:
            return modelObj.latent_dim
        else:
            return modelObj.feature_dim
            
            
def get_named_layers(net, tag=""):
    conv_idx = 0
    convT_idx = 0
    linear_idx = 0
    rnn_idx = 0
    pool_idx = 0
    unpool_idx = 0
    unsample_idx = 0
    batchnorm_idx = 0
    layernorm_idx = 0
    instancenorm_idx = 0
    dropout_idx = 0
    act_idx = 0
    gconv_idx = 0
    named_layers = OrderedDict()
    for seg in net:
        for name, mod in seg.named_modules():
            if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                layer_name = '{}_Conv{}_{}_{}'.format(tag,
                                                       conv_idx, mod.in_channels, mod.out_channels
                                                       )
                conv_idx += 1
                named_layers[layer_name] = mod
            elif isinstance(mod, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                layer_name = '{}_ConvT{}_{}_{}'.format(tag,
                                                        convT_idx, mod.in_channels, mod.out_channels
                                                        )
                named_layers[layer_name] = mod
                convT_idx += 1
            elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                layer_name = '{}_BatchNorm{}_{}_{}'.format(tag,
                                                            batchnorm_idx, mod.num_features, mod.num_features)
                named_layers[layer_name] = mod
                batchnorm_idx += 1
            elif isinstance(mod, nn.LayerNorm):
                layer_name = '{}_LayerNorm{}_{}_{}'.format(tag,
                                                            layernorm_idx, mod.num_features, mod.num_features)
                named_layers[layer_name] = mod
                layernorm_idx += 1
            elif isinstance(mod, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                layer_name = '{}_InstanceNorm{}_{}_{}'.format(tag,
                                                            instancenorm_idx, mod.num_features, mod.num_features)
                named_layers[layer_name] = mod
                instancenorm_idx += 1
            elif isinstance(mod, nn.Linear):
                layer_name = '{}_Linear{}_{}_{}'.format(tag,
                                                         linear_idx, mod.in_features, mod.out_features
                                                         )
                named_layers[layer_name] = mod
                linear_idx += 1
            elif isinstance(mod, nn.RNN):
                layer_name = '{}_RNN{}_{}_{}'.format(tag,
                                                      rnn_idx, mod.input_size, mod.hidden_size
                                                      )
                named_layers[layer_name] = mod
                rnn_idx += 1
            elif isinstance(mod, nn.GRU):
                layer_name = '{}_GRU{}_{}_{}'.format(tag,
                                                      rnn_idx, mod.input_size, mod.hidden_size
                                                      )
                named_layers[layer_name] = mod
                rnn_idx += 1
            elif isinstance(mod, nn.LSTM):
                layer_name = '{}_LSTM{}_{}_{}'.format(tag,
                                                       rnn_idx, mod.input_size, mod.hidden_size
                                                       )
                named_layers[layer_name] = mod
                rnn_idx += 1
            elif isinstance(mod, nn.Dropout):
                layer_name = '{}_Dropout{}_{:2.0f}_{}'.format(tag,
                                                              dropout_idx, 100*mod.p, mod.inplace
                                                              )
                named_layers[layer_name] = mod
                dropout_idx += 1
            elif isinstance(mod, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
                layer_name = '{}_Pool{}_{}_{}'.format(tag,
                                                      pool_idx, mod.kernel_size, mod.stride
                                                      )
                named_layers[layer_name] = mod
                pool_idx += 1
            elif isinstance(mod, (nn.Upsample)):
                layer_name = '{}_Upsample{}_{}_{}'.format(tag,
                                                      unsample_idx, mod.size, list([int(x) for x in mod.scale_factor]) if mod.scale_factor is not None else None
                                                      )
                
                # mod.scale_factor is tuple of float

                named_layers[layer_name] = mod
                unsample_idx += 1
            elif isinstance(mod, (nn.MaxUnpool1d, nn.MaxUnpool2d, nn.MaxUnpool3d)):
                layer_name = '{}_Unpool{}_{}_{}'.format(tag,
                                                        unpool_idx, mod.kernel_size, mod.stride
                                                        )
                named_layers[layer_name] = mod
                unpool_idx += 1
            elif isinstance(mod, (nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.Tanh)):
                layer_name = '{}_Activation{}'.format(tag,
                                                      act_idx
                                                      )
                named_layers[layer_name] = mod
                act_idx += 1
            elif isinstance(mod, (dgl.nn.pytorch.GraphConv)):
                layer_name = '{}_gConv{}'.format(tag,
                                                 gconv_idx
                                                 )
                named_layers[layer_name] = mod
                gconv_idx += 1

    return named_layers

def round_up_to_even(f, only_odd=True):
    # f = int(f)
    if only_odd:
      if f%2 != 0:
        return int(np.floor(f) // 2 * 2 + 2)
      else:
        return int(f)
    else:
        return int(np.floor(f) // 2 * 2 + 2)

def round_down_to_odd(f, only_even=True):

    if only_even:
      if f%2 == 0:
        return np.floor(f) // 2 * 2 - 1
      else:
        return f
    else:
        return np.floor(f) // 2 * 2 - 1

def get_optimal_spatial_size(S, p):

  if not isinstance(S, np.ndarray):
    S = np.array(S)
  R = S%p
  D = S//p
  print(S, p, R, D) 
  vfunc = np.vectorize(lambda s, r, d: round_up_to_even(d) if (r>0) and (d>0) else max(d, 1))
  return vfunc(S, R, D)

def get_optimal_kernel_size(S, K):

  if not isinstance(S, np.ndarray):
    S = np.array(S)
  if not isinstance(K, np.ndarray):
    K = np.array(K)
  # squarer = lambda t: t ** 2
  # print(S, K)
  K = np.minimum(S, K)
  vfunc = np.vectorize(lambda k: int(round_down_to_odd(k)))
  return vfunc(K)

def mask_upsampled_3d_with_pooled_indices(x_upsampled,
                                          indices
                                          ) -> torch.FloatTensor:
    '''
    x_upsampled = [bxCxspatial]
    indices = [bxCxspatial//2]
    '''
    indices_ = indices.detach().clone()
    mask = 0*x_upsampled.detach().clone()
    
    mask = mask.reshape(mask.shape[:2] + (-1,))
    indices = indices.reshape(indices.shape[:2] + (-1,))
    mask = mask.scatter(2, indices, 1) 
    mask = mask.reshape(x_upsampled.shape)
    mask = mask.type(torch.bool)
    x_upsampled[~mask] = 0
    return x_upsampled


class AEModelTemplate():
    def __init__(self, **kwargs):
        self.model_type = "ae_ad"
        self.use_sparse = kwargs.get("use_sparse", False)
        self.feature_dim = kwargs.get("feature_dim", None)
        self.target_dim = kwargs.get("target_dim", None)
        self.latent_dim = kwargs.get("latent_dim", 2)
        self.memorysize = kwargs.get("memorysize", 5)
        self.horizonsize = kwargs.get("horizonsize", None)
        self.num_layers = kwargs.get("num_layers", 1)
        self.keep_hidden_states = False
        self.isvariational = kwargs.get("isvariational", False)
        self.reg_beta = kwargs.get("reg_beta", 0.01)
        self.loss_scale = kwargs.get("loss_scale", 100)
        self.random_seed = kwargs.get("random_seed", 123)
        self.get_encoded = False  # for shap
        self.model_interprate = False  # for shap
        self.interp_sample_idx = [-1]
        
        # self.reset_random_init()
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def reset_random_init(self):
        print("reseting random with seed: ", self.random_seed)
        util.set_reproducible(self.random_seed)
        
    def check_reset_state(self):
        try:
            e_rnn_state_len = len(self.e_rnn_hidden)
            d_rnn_state_len = len(self.d_rnn_hidden)
            self.isreset_hstate = {"e": True, "d": True}
            self.isreset_hstate["e"] = True if (not self.keep_hidden_states) or (e_rnn_state_len == 0) else False
            self.isreset_hstate["d"] = True if (not self.keep_hidden_states) or (d_rnn_state_len == 0) else False
        except Exception as ex:
            print("check_reset_state is error: {}!".format(ex))

    def forced_reset_state(self):
        try:
            self.e_rnn_hidden = []
            self.d_rnn_hidden = []
            self.isreset_hstate = {"e": True, "d": True}
            
        except:
            print("state memory is not defined for this model.")
            
    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def select_activation(self, act_func=None, leaky_relu_negative_slope=1):
        if act_func:
            return act_func
        if self.input_scaling_alg:
            if "relu":
                return nn.ReLU()  # cuts lower values when used in the decoder for non +ve outputs
            elif "std" in self.input_scaling_alg:
                return nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
            else:
                raise "activation function for {} is not undefined in self.select_activation().".format(
                    self.input_scaling_alg)
        else:
            return nn.ReLU()

    def init_hidden(self, hidden_size, num_directions, batch_dim, layer):
        if isinstance(layer, nn.GRU):
            h0 = Variable(torch.zeros(layer.num_layers*num_directions,
                                      batch_dim, hidden_size)).to(self.device)
            return h0
        elif isinstance(layer, nn.LSTM):
            h0 = Variable(torch.zeros(layer.num_layers*num_directions,
                                      batch_dim, hidden_size)).to(self.device)
            c0 = Variable(torch.zeros(layer.num_layers*num_directions,
                                      batch_dim, hidden_size)).to(self.device)
            return h0, c0

    def weight_init(self, layer):
        # self.reset_random_init()
        for name, param in layer.named_parameters():
            if 'bias' in name:
                torch.nn.init.zeros_(param)  # has good effect
            if 'weight' in name:
                # torch.nn.init.kaiming_normal_(param)
                torch.nn.init.kaiming_uniform_(param)

    def loss(self, y_true, y_pred, *arg, **kwargs):
        criterion = kwargs.get("criterion", self.criterion)
        mask = kwargs.get("mask", None)
        scale = kwargs.get("loss_scale", self.loss_scale)

        # reconstruction loss
        reg_beta = kwargs.get(
            "reg_beta", self.reg_beta)
        if mask is None:
            rec_loss_neg = criterion(y_pred, y_true)
        else:
            rec_loss_neg = criterion(torch.masked_select(y_pred, ~mask), torch.masked_select(y_true, ~mask))
                

        if self.isvariational:
            latent_mu, latent_logvar = arg
            # KL Divergence score
            reg_beta = kwargs.get(
                "reg_beta", self.reg_beta)

            kl_div = -0.5 * (1 + latent_logvar - latent_mu **
                             2 - torch.exp(latent_logvar)).sum()
            return rec_loss_neg + reg_beta*kl_div

        return scale*rec_loss_neg.abs()

    def pred_loss(self, x, y_true, *arg, **kwargs):
        '''
        self.eval(), self() is only valid when the class is inherited by the nn class
        '''
        pred_criterion = kwargs.get("pred_criterion", self.criterion)
        mask = kwargs.get("mask", None)

        self.eval()
        y_pred = self(x, **kwargs)

        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        if mask is None:
            return pred_criterion(y_pred, y_true)
        else:
            return pred_criterion(torch.masked_select(y_pred, ~mask), torch.masked_select(y_true, ~mask))


class VAEModelTemplate(nn.Module):
    def __init__(self, dim_in, dim_out, variational_beta=0.01):
        super().__init__()
        self.fc_mu = nn.Linear(in_features=dim_in, out_features=dim_out)
        self.fc_logvar = nn.Linear(in_features=dim_in, out_features=dim_out)

        self.normal_dist = MultivariateNormal(
            torch.zeros(dim_out), torch.eye(dim_out))
        self.variational_beta = variational_beta
        
        self.weight_init(self.fc_mu)
        self.weight_init(self.fc_logvar)
        
    def weight_init(self, layer):
        for name, param in layer.named_parameters():
            if 'bias' in name:
                torch.nn.init.zeros_(param)  # has good effect
            if 'weight' in name:
                # torch.nn.init.kaiming_normal_(param)
                torch.nn.init.kaiming_uniform_(param) 
                
    def reparameterize_sampling(self, mu, logvar, istrain_mode=False):
        if istrain_mode:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, istrain_mode=False):
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        # get a latent variable sample with the reparameterization trick
        x = self.reparameterize_sampling(
            x_mu, x_logvar, istrain_mode=istrain_mode)
        return x, x_mu, x_logvar


class SequentialCustom(nn.Sequential):
    def forward(self, inputs, *args, **kwargs):
        for module in self._modules.values():
            # print("module: ", type(module), module)
            if module is None:
                return inputs
            try:
                inputs = module(inputs, *args, **kwargs)
            except:
                inputs = module(inputs)
        return inputs


class CNNAE_MultiDim_SPATIAL_Block(nn.Module, AEModelTemplate):
    '''
    CNN (1d/2d/3d) + AE
    '''

    def __init__(self, feature_dim, latent_dim, target_dim, spatial_dims,
                 memorysize=5, **kwargs):

        super().__init__()

        self.model_type = "ae_ad"
        self.spatial_dims = spatial_dims
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.out_act = kwargs.get("out_act", None)
        self.e_num_conv_layers = kwargs.get("e_num_conv_layers", 4)
        self.d_num_conv_layers = self.e_num_conv_layers
        self.spatial_dim_len = len(spatial_dims)
        self.kernel_size = kwargs.get("kernel_size", (3, 3, 2))
        self.pool_size = kwargs.get("pool_size", (2, 2, 2))
        self.use_res = kwargs.get("use_res", False)
        self.norm_layer = kwargs.get("norm_layer", "bn")
        self.activation = kwargs.get("activation", "leakyrelu")
        self.leakyrelu_negative_slope = kwargs.get("leakyrelu_negative_slope", 0.5)
        self.upsample_layer = kwargs.get("upsample_layer", "unpool")
        
        self.nodropout = kwargs.get("nodropout", ["encoder", "decoder"])
        self.e_isnodropout = "encoder" in self.nodropout
        self.d_isnodropout = "decoder" in self.nodropout
        
        self.act = nn.Sigmoid if self.activation == "sigmoid" else (
                                nn.ReLU if self.activation == "relu" else (
                                nn.LeakyReLU if self.activation == "leakyrelu" else (
                                nn.Tanh if self.activation == "tanh" else nn.Identity)))

        self.e_conv_module = nn.Conv2d if self.spatial_dim_len == 2 else (
            nn.Conv3d if self.spatial_dim_len == 3 else nn.Conv1d)
        self.d_conv_module = nn.ConvTranspose2d if self.spatial_dim_len == 2 else (
            nn.ConvTranspose3d if self.spatial_dim_len == 3 else nn.ConvTranspose1d)
        self.decoder_conv_model = kwargs.get("decoder_conv", "convT")

        self.pooling = nn.MaxPool2d if self.spatial_dim_len == 2 else (
            nn.MaxPool3d if self.spatial_dim_len == 3 else nn.MaxPool1d)
        self.unpooling = nn.MaxUnpool2d if self.spatial_dim_len == 2 else (
            nn.MaxUnpool3d if self.spatial_dim_len == 3 else nn.MaxUnpool1d)
        
        if self.decoder_conv_model == "convT":
            self.d_conv_module = nn.ConvTranspose2d if self.spatial_dim_len == 2 else (
                nn.ConvTranspose3d if self.spatial_dim_len == 3 else nn.ConvTranspose1d)
        else:
            self.d_conv_module = nn.Conv2d if self.spatial_dim_len == 2 else (
                nn.Conv3d if self.spatial_dim_len == 3 else nn.Conv1d)

        if self.norm_layer == "bn":
            # over entire channel
            self.norm_module = nn.BatchNorm2d if self.spatial_dim_len == 2 else (
                nn.BatchNorm3d if self.spatial_dim_len == 3 else nn.BatchNorm1d)
        else:
            self.norm_module = None
            self.norm_layer = None

        print("self.e_num_conv_layers: ", self.e_num_conv_layers)
        print("self.d_num_conv_layers: ", self.d_num_conv_layers)
        print("self.kernel_size: ", self.kernel_size)

        target_dim = self.target_dim if self.target_dim else feature_dim

        print(memorysize, feature_dim, target_dim)
        print("self.spatial_dims: ", self.spatial_dims)
        # CN
        self.no_group_features = 1

        self.e_layer_num_inout = [feature_dim]

        self.e_conv_num_out_basic = 4
        self.e_conv_num_out_descaler = 1
        self.d_conv_num_out_basic = 4
        self.d_conv_num_out_descaler = 1
        self.e_layer_num_inout.extend(
            [np.max([1, self.e_conv_num_out_basic//(self.e_conv_num_out_descaler**i)]) * latent_dim for i in range(self.e_num_conv_layers)])
        self.d_layer_num_inout = [
            np.max([1, self.d_conv_num_out_basic//(self.d_conv_num_out_descaler**i)]) * latent_dim for i in range(self.d_num_conv_layers-1, -1, -1)]

        self.d_layer_num_inout.append(target_dim)

        self.e_layer_num_inout = np.array(self.e_layer_num_inout)
        self.d_layer_num_inout = np.array(self.d_layer_num_inout)

        print("self.e_layer_num_inout: ", self.e_layer_num_inout)
        print("self.d_layer_num_inout: ", self.d_layer_num_inout)

        self.e_layer_spatial_dim = [np.array(self.spatial_dims)]
        self.e_layer_spatial_dim.extend(np.array([[int(np.ceil(np.max([f/(p**i), 1]))) for f, p in zip(self.spatial_dims, self.pool_size)]
                                             for i in range(1, self.e_num_conv_layers+1)]))
                                             
        self.d_layer_spatial_dim = np.array([self.e_layer_spatial_dim[i]
                                             for i in range(self.d_num_conv_layers, -1, -1)])
        print("self.e_layer_spatial_dim: ", self.e_layer_spatial_dim)
        print("self.d_layer_spatial_dim: ", self.d_layer_spatial_dim)

        self.e_kernel_size = [get_optimal_kernel_size(self.e_layer_spatial_dim[i], self.kernel_size) for i in range(self.e_num_conv_layers)]
        self.d_kernel_size = [get_optimal_kernel_size(self.d_layer_spatial_dim[i+1], self.kernel_size) for i in range(self.d_num_conv_layers)]

        self.e_kernel_size = np.array(self.e_kernel_size)
        self.d_kernel_size = np.array(self.d_kernel_size)
        print("self.e_kernel_size: ", self.e_kernel_size)
        print("self.d_kernel_size: ", self.d_kernel_size)

        self.dropout = 0.2
        # encoding part
        self.encoder_stride = 1
        self.decoder_stride = 1
        self.dilation = 1
        # self.bias = True # default
        self.bias = False

        self.e_conv = nn.ModuleList(
            [
                SequentialCustom(
                    self.e_conv_module(in_channels=self.e_layer_num_inout[i], out_channels=self.e_layer_num_inout[i+1],
                                       kernel_size=self.e_kernel_size[i], stride=self.encoder_stride,
                                       dilation=1,
                                       bias=self.bias,
                                    #    padding='same',
                                       padding=tuple(self.e_kernel_size[i]//2),
                                       groups=self.no_group_features),
                    nn.Dropout(self.dropout) if not self.e_isnodropout else None,
                    None if self.norm_layer is None else self.norm_module(self.e_layer_num_inout[i+1]),
                    self.act(inplace=False) if isinstance(self.act(), nn.ReLU) else (self.act(negative_slope=self.leakyrelu_negative_slope) if isinstance(self.act(), nn.LeakyReLU) else self.act()),
                  self.pooling(np.minimum(self.e_layer_spatial_dim[i], self.pool_size).tolist(), return_indices=True, ceil_mode=True),
                )
                for i in range(self.e_num_conv_layers)
            ])

        if self.decoder_conv_model == "convT":
            self.d_conv = nn.ModuleList(
                [
                    SequentialCustom(
                        self.unpooling(np.minimum(self.d_layer_spatial_dim[i+1], self.pool_size).tolist()) if self.upsample_layer == "unpool" else  nn.Upsample(size=tuple(self.d_layer_spatial_dim[i+1])),
                        self.d_conv_module(in_channels=self.d_layer_num_inout[i], out_channels=self.d_layer_num_inout[i+1],
                                           kernel_size=self.d_kernel_size[
                            i], stride=self.decoder_stride,
                            dilation=1,
                            bias=self.bias,
                            padding=tuple(self.d_kernel_size[i]//2),               
                            groups=self.no_group_features),
                        nn.Dropout(self.dropout) if not self.d_isnodropout else None,
                        None if self.norm_layer is None else self.norm_module(self.d_layer_num_inout[i+1]),
                        self.act(inplace=False) if isinstance(self.act(), nn.ReLU) else (self.act(negative_slope=self.leakyrelu_negative_slope) if isinstance(self.act(), nn.LeakyReLU) else self.act()),
                    )
                    for i in range(self.d_num_conv_layers)
                ])
        else:
            self.d_conv = nn.ModuleList(
                [
                    SequentialCustom(
                        self.unpooling(np.minimum(self.d_layer_spatial_dim[i+1], self.pool_size).tolist()) if self.upsample_layer == "unpool" else nn.Upsample(size=tuple(self.d_layer_spatial_dim[i+1])),
                        self.d_conv_module(in_channels=self.d_layer_num_inout[i], out_channels=self.d_layer_num_inout[i+1],
                                           kernel_size=self.d_kernel_size[
                            i], stride=self.decoder_stride,
                            dilation=1,
                            bias=self.bias, 
                            padding=tuple(self.d_kernel_size[i]//2),               
                            groups=self.no_group_features),
                        nn.Dropout(self.dropout) if not self.d_isnodropout else None,
                        None if self.norm_layer is None else self.norm_module(self.d_layer_num_inout[i+1]),
                        self.act(inplace=False) if isinstance(self.act(), nn.ReLU) else (self.act(negative_slope=self.leakyrelu_negative_slope) if isinstance(self.act(), nn.LeakyReLU) else self.act()),
                    )
                    for i in range(self.d_num_conv_layers)
                ])

        self.d_conv_final = SequentialCustom(
            self.d_conv_module(in_channels=self.d_layer_num_inout[-1], out_channels=self.d_layer_num_inout[-1],
                               kernel_size=1,
                               stride=1, dilation=1,
                               padding=0,
                groups=self.no_group_features),
             nn.Sigmoid() if self.out_act == "sigmoid" else (
                                            nn.Tanh() if self.out_act == "tanh" else (nn.ReLU() if self.out_act == "relu" else nn.Identity()))
        )

        self.e_conv = nn.Sequential(get_named_layers(self.e_conv, tag="e"))
        self.d_conv = nn.Sequential(get_named_layers(self.d_conv, tag="d"))
        self.d_conv_final = SequentialCustom(get_named_layers(self.d_conv_final, tag="d"))

    def encoder(self, x, **kwargs):
       
        x_dim_len = len(x.size())
        x = x.permute((0, x_dim_len-1) +
                      tuple(np.arange(1, x_dim_len-1).tolist()))

        self.e_pool_idx = []
        self.e_pool_in_size = []

        x_res = None
        for i, layer in enumerate(self.e_conv):
            x_size = x.size()

            if self.use_res:
                if i>0 and isinstance(layer, self.e_conv_module):
                    x_res = x
                elif x_res is not None and isinstance(layer, self.pooling):
                    x = x + x_res

            x = layer(x)
            if isinstance(layer, self.pooling):
                self.e_pool_in_size.append(x_size)
                self.e_pool_idx.append(x[1])
                x = x[0]

        x = x.permute(
            (0, ) + tuple(np.arange(2, self.spatial_dim_len+2).tolist()) + (1, ))
        
        return x
        
    def decoder(self, x, **kwargs):

        x_dim_len = len(x.size())

        x = x.permute((0, x_dim_len-1) +
                      tuple(np.arange(1, x_dim_len-1).tolist()))

        n_layer = len(self.e_pool_idx) - 1
        j = 0
        x_res = None
        for i, layer in enumerate(self.d_conv):

            if self.use_res:
                if isinstance(layer, self.e_conv_module):
                    x_res = x
                elif x_res is not None and isinstance(layer, self.unpooling):
                    x = x + x_res

            if isinstance(layer, self.unpooling) and self.upsample_layer == "unpool":
                x = layer(
                    x, indices=self.e_pool_idx[n_layer-j], output_size=self.e_pool_in_size[n_layer-j])
                j = j + 1

            else:
                x = layer(x)
                if isinstance(layer, nn.Upsample):
                    x = mask_upsampled_3d_with_pooled_indices(x, self.e_pool_idx[n_layer-j])
                    j = j + 1

        x = self.d_conv_final(x)

        x = x.permute(
            (0, ) + tuple(np.arange(2, self.spatial_dim_len+2).tolist()) + (1, ))

        return x


import dgl
class GNN_MultiDim_SPATIAL_Block(nn.Module, AEModelTemplate):
    '''
    GNN (1d/2d/3d) encoder
    '''

    def __init__(self, feature_dim, latent_dim, target_dim, spatial_dims, node_edge_index,  **kwargs):

        super().__init__()

        self.model_type = "ae_ad"
        self.spatial_dims = spatial_dims
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.e_num_gconv_layers = kwargs.get("e_num_gconv_layers", 4)

        self.e_gconv_module = dgl.nn.pytorch.GraphConv
        self.e_gpool_module = dgl.nn.GlobalAttentionPooling
        
        print(node_edge_index.shape)
        # make sure data input is max node id on the edge
        self.node_max_node_id = node_edge_index.max()
        self.node_edge_index = dgl.graph(
            tuple(node_edge_index), idtype=torch.int32)  # for dgl
        self.node_edge_index = self.node_edge_index.to(self.device)

        # GN
        self.node_feature_dim = feature_dim
        self.e_layer_num_inout_g = [self.node_feature_dim]
        self.e_layer_num_inout_g.extend(
            [np.max([1, 4//(1**i)]) * latent_dim for i in range(self.e_num_gconv_layers)])

        print("self.e_layer_num_inout_g: ", self.e_layer_num_inout_g)

        self.e_gconv = nn.ModuleList(
                [
                    SequentialCustom(
                        self.e_gconv_module(self.e_layer_num_inout_g[i], self.e_layer_num_inout_g[i+1], norm="both",
                                            weight=True, bias=True, allow_zero_in_degree=True, 
                                            ),  
                        nn.ReLU(),
                    )
                    for i in range(self.e_num_gconv_layers)
                ])

        # pooling
        gate_nn = nn.Linear(self.e_layer_num_inout_g[-1], 1)
        self.weight_init(gate_nn)
        self.e_gpool = self.e_gpool_module(gate_nn)

        # curtail to salve senstitveity to random seed ini. has good effect on reconstruction performance
        for layer in self.e_gconv:
            if isinstance(layer, self.e_gconv_module):
                self.weight_init(layer)
            
        self.e_gconv = SequentialCustom(get_named_layers(self.e_gconv, tag="e"))
            
    def encoder(self, x, **kwargs):
        x = util.convert_to_graph_node_data(x.contiguous(), feature_dims=[-1])

        x = x[:, :self.node_max_node_id + 1].contiguous()
        
        x_g_shape = x.size()
        batch = x_g_shape[0]
        g = dgl.batch([self.node_edge_index]*batch)
        
        x = x.view((-1, x.shape[-1]))

        for i, layer in enumerate(self.e_gconv):
            x = layer(g, x) if isinstance(layer, self.e_gconv_module) else layer(x) 
            
        x = self.e_gpool(g, x)  # [bsxfeature]

        return x


class FNNAE_MultiDim_SPATIAL_Block(nn.Module, AEModelTemplate):
    '''
    FC+AE
    '''

    def __init__(self, e_layer_num_inout, e_layer_spatial_dim, d_layer_num_inout, d_layer_spatial_dim, latent_dim, target_dim, 
                **kwargs):

        super().__init__()

        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.e_layer_num_inout = e_layer_num_inout
        self.e_layer_spatial_dim = e_layer_spatial_dim
        self.d_layer_num_inout = d_layer_num_inout
        self.d_layer_spatial_dim = d_layer_spatial_dim
        self.e_num_nn_layers = kwargs.get("e_num_nn_layers", 2)
        self.d_num_nn_layers = self.e_num_nn_layers
    

        # FNN
        self.e_layer_num_inout_fnn = [
            np.prod(self.e_layer_spatial_dim[-1])*self.e_layer_num_inout[-1]]
        self.e_layer_num_inout_fnn.extend(
            [4*latent_dim for i in range(self.e_num_nn_layers-1)])
        self.e_layer_num_inout_fnn.append(latent_dim)
        self.d_layer_num_inout_fnn = [
            self.e_layer_num_inout_fnn[i] for i in range(self.d_num_nn_layers, -1, -1)]

        self.d_layer_num_inout_fnn[-1] = np.prod(self.d_layer_spatial_dim[0])*self.d_layer_num_inout[0]

        self.e_num_deep_layers = len(self.e_layer_num_inout_fnn)
        self.d_num_deep_layers = len(self.d_layer_num_inout_fnn)

        print("self.e_layer_num_inout_fnn: ", self.e_layer_num_inout_fnn)
        print("self.d_layer_num_inout_fnn: ", self.d_layer_num_inout_fnn)


        #  # RNN
        # self.isbidxn = [False]*self.e_num_nn_layers
        # self.isbidxn_scale = [1]*self.e_num_nn_layers

        # self.e_layer_num_inout_rnn = [
        #         np.prod(self.e_layer_spatial_dim[-1])*self.e_layer_num_inout[-1]]  # input from multiple feature extraction blocks

        # self.e_layer_num_inout_rnn.extend(
        #     [4*latent_dim for i in range(self.e_num_nn_layers-1)])
        # self.e_layer_num_inout_rnn.append(latent_dim)
        # self.d_layer_num_inout_rnn = [
        #     self.e_layer_num_inout_rnn[i] for i in range(self.d_num_nn_layers, -1, -1)]

        # self.d_layer_num_inout_rnn[-1] = np.prod(self.d_layer_spatial_dim[0])*self.d_layer_num_inout[0]

        # self.e_num_deep_layers = len(self.e_layer_num_inout_rnn)
        # self.d_num_deep_layers = len(self.d_layer_num_inout_rnn)

        # print("self.e_layer_num_inout_rnn: ", self.e_layer_num_inout_rnn)
        # print("self.d_layer_num_inout_rnn: ", self.d_layer_num_inout_rnn)


        self.dropout = 0.2

        self.e_fnn = nn.ModuleList(
            [
                SequentialCustom(
                    nn.Linear(self.e_layer_num_inout_fnn[i], self.e_layer_num_inout_fnn[i+1],
                              ),
                    nn.Dropout(self.dropout)
                )
                for i in range(self.e_num_nn_layers)
            ])

        self.d_fnn = nn.ModuleList(
            [
                SequentialCustom(
                    nn.Linear(self.d_layer_num_inout_fnn[i], self.d_layer_num_inout_fnn[i+1],
                              ),
                    nn.Dropout(self.dropout)
                )
                for i in range(self.d_num_nn_layers)

            ])

        self.e_fnn = nn.Sequential(get_named_layers(self.e_fnn, tag="e"))
        self.d_fnn = nn.Sequential(get_named_layers(self.d_fnn, tag="d"))

        # has good effect on reconstruction performance
        for layer in self.e_fnn:
            self.weight_init(layer)
        for layer in self.d_fnn:
            self.weight_init(layer)

    def encoder(self, x, **kwargs):
       
        self.conv_last_shape = x.shape
      
        x = x.reshape(self.conv_last_shape[0:2] + (-1, ))

        for i, layer in enumerate(self.e_fnn):
            x = layer(x)

        # x = x[:, -1].unsqueeze(1)
        x = x[:, -1:].unsqueeze(1)
    
        return x

    def decoder(self, x, **kwargs):
        # x = x.squeeze(1)

        for i, layer in enumerate(self.d_fnn):
            x = layer(x)

        x = x.reshape(self.conv_last_shape)

        return x


class RNNAE_MultiDim_SPATIAL_Block(nn.Module, AEModelTemplate):
    '''
    RNN+AE
    '''

    def __init__(self, e_layer_num_inout, e_layer_spatial_dim, d_layer_num_inout, d_layer_spatial_dim, latent_dim, target_dim, 
                 rnn_model="gru", **kwargs):

        super().__init__()

        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.e_layer_num_inout = e_layer_num_inout
        self.e_layer_spatial_dim = e_layer_spatial_dim
        self.d_layer_num_inout = d_layer_num_inout
        self.d_layer_spatial_dim = d_layer_spatial_dim
        self.rnn_model = rnn_model
        self.e_num_nn_layers = kwargs.get("e_num_nn_layers", 2)
        self.e_layer_num_inout_g = kwargs.get("e_layer_num_inout_g", 0) # for gnn fe
        self.d_num_nn_layers = self.e_num_nn_layers
        self.keep_hidden_states = False
        self.num_layers = 1

        rnn_dict = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        self.rnn_module = rnn_dict[self.rnn_model.lower()]

        print(f"self.e_num_nn_layers: {self.e_num_nn_layers}, self.d_num_nn_layers: {self.d_num_nn_layers}")

        # RNN
        self.isbidxn = [False]*self.e_num_nn_layers
        self.isbidxn_scale = [1]*self.e_num_nn_layers

        self.e_layer_num_inout_rnn = [
                np.prod(self.e_layer_spatial_dim[-1])*self.e_layer_num_inout[-1] + self.e_layer_num_inout_g]  # input from multiple feature extraction blocks

        self.e_layer_num_inout_rnn.extend(
            [4*latent_dim for i in range(self.e_num_nn_layers-1)])
        self.e_layer_num_inout_rnn.append(latent_dim)
        self.d_layer_num_inout_rnn = [
            self.e_layer_num_inout_rnn[i] for i in range(self.d_num_nn_layers, -1, -1)]

        self.d_layer_num_inout_rnn[-1] = np.prod(self.d_layer_spatial_dim[0])*self.d_layer_num_inout[0]

        self.e_num_deep_layers = len(self.e_layer_num_inout_rnn)
        self.d_num_deep_layers = len(self.d_layer_num_inout_rnn)

        print("self.e_layer_num_inout_rnn: ", self.e_layer_num_inout_rnn)
        print("self.d_layer_num_inout_rnn: ", self.d_layer_num_inout_rnn)

        self.dropout = 0.2
        self.bias = False

        self.e_rnn = nn.ModuleList(
            [
                SequentialCustom(
                    self.rnn_module(input_size=self.isbidxn_scale[i]*self.e_layer_num_inout_rnn[i], hidden_size=self.e_layer_num_inout_rnn[i+1],
                                    num_layers=self.num_layers, batch_first=True, bidirectional=self.isbidxn[i],
                                    dropout=self.dropout
                                    )
                )
                for i in range(self.e_num_nn_layers)
            ])

        self.d_rnn = nn.ModuleList(
            [
                SequentialCustom(
                    self.rnn_module(input_size=self.d_layer_num_inout_rnn[i], hidden_size=self.d_layer_num_inout_rnn[i+1],
                                    num_layers=self.num_layers, batch_first=True, bidirectional=False,
                                    dropout=self.dropout)
                )
                for i in range(self.d_num_nn_layers)

            ])

        self.e_rnn_hidden = []
        self.d_rnn_hidden = []

        for layer in self.e_rnn:
            self.weight_init(layer)
        for layer in self.d_rnn:
            self.weight_init(layer)

        self.e_rnn = SequentialCustom(get_named_layers(self.e_rnn, tag="e"))
        self.d_rnn = SequentialCustom(get_named_layers(self.d_rnn, tag="d"))

    
    def get_ae_rnn_states_starter(self, batch_dim=1):
        """
        To be called and serve as init input of states during onnx export
        batch_dim=2 as the histogram is splited in tow: hem and hep and make the model to have twice the batch
        """
        e_rnn_hidden = []
        for i, layer in enumerate(self.e_rnn):
            e_rnn_hidden.append(self.init_hidden(layer.hidden_size, int(layer.bidirectional)+1, batch_dim, layer))

        d_rnn_hidden = []
        for i, layer in enumerate(self.d_rnn):
            d_rnn_hidden.append(self.init_hidden(layer.hidden_size, int(layer.bidirectional)+1, batch_dim, layer))
        
        return e_rnn_hidden, d_rnn_hidden
    
    def encoder(self, x, x_exo=None, **kwargs):
       
        self.conv_last_shape = x.shape
      
        x = x.reshape(self.conv_last_shape[0:2] + (-1, ))

        if x_exo is not None:
            x = torch.cat((x, x_exo), 2)  # concat cnn and gnn features

        for i, layer in enumerate(self.e_rnn):
            if self.isreset_hstate["e"]:
                self.e_rnn_hidden.append(self.init_hidden(
                    layer.hidden_size, int(layer.bidirectional)+1, x.size(0), layer))
            x, self.e_rnn_hidden[i] = layer(x, self.e_rnn_hidden[i])

        # compress ts: taking the last output# equivalent of return_false in keras
        x = x[:, -1].unsqueeze(1)
   
        return x

    def decoder(self, x, **kwargs):
        # x = x.squeeze(1)
        x = x.repeat(1, self.conv_last_shape[1], 1)

        # rnn ntk
        for i, layer in enumerate(self.d_rnn):
            if self.isreset_hstate["d"]:
                self.d_rnn_hidden.append(self.init_hidden(
                    layer.hidden_size, int(layer.bidirectional)+1, x.size(0), layer))

            x, self.d_rnn_hidden[i] = layer(x, self.d_rnn_hidden[i])
 
        x = x.reshape(self.conv_last_shape)

        return x


class CNNFNNAE_MultiDim_SPATIAL(nn.Module, AEModelTemplate):
    '''
    CNN(1d/2d/3d)+AE
    '''

    def __init__(self,  feature_dim, latent_dim, target_dim, spatial_dims,
                 memorysize=1, **kwargs):

        # super function is used to use classes of parent class
        super().__init__()
        
        AEModelTemplate.__init__(self, datatype="ts", latent_dim=latent_dim, feature_dim=feature_dim,
                                 memorysize=memorysize, **kwargs)
        
        self.cnn_block = CNNAE_MultiDim_SPATIAL_Block(feature_dim, latent_dim, target_dim, spatial_dims,
                                        memorysize=memorysize, **kwargs)

        self.fnn_block = FNNAE_MultiDim_SPATIAL_Block(
                                     self.cnn_block.e_layer_num_inout, 
                                     self.cnn_block.e_layer_spatial_dim, self.cnn_block.d_layer_num_inout, self.cnn_block.d_layer_spatial_dim, latent_dim, target_dim, 
                                 **kwargs)
        
        if self.isvariational:
            # VAE layer
            print("variational encoder is used...")
            self.variational_layer = VAEModelTemplate(
                dim_in=self.latent_dim, dim_out=latent_dim, variational_beta=kwargs.get("variational_beta", 0.01))
        
    def encoder(self, x, **kwargs):    
        x = x.to(self.device)

        # CNN
        self.e_pool_idx = []
        self.e_pool_in_size = []
        t_size = x.size()[1]  # time window size ==1
        x_over_t = []

        for t in range(t_size):
            x_t = x[:, t]

            x_t = self.cnn_block.encoder(x_t)
            
            x_over_t.append(x_t.unsqueeze(1))
            self.e_pool_idx.append(self.cnn_block.e_pool_idx)
            self.e_pool_in_size.append(self.cnn_block.e_pool_in_size)

        x = torch.cat(tuple(x_over_t), 1)
            
        # FNN
        x = self.fnn_block.encoder(x)

        # VAE
        if self.isvariational:
            x, latent_mu, latent_logvar = self.variational_layer(
                x, istrain_mode=self.training)

            return x, latent_mu, latent_logvar

        return x

    def decoder(self, x, **kwargs):
        # x = x.squeeze(1)

        # FNN
        x = self.fnn_block.decoder(x)

        # CNN
        t_size = x.size()[1]  # time window size==1
        x_over_t = []
        for t in range(t_size):
            x_t = x[:, t]
            self.cnn_block.e_pool_in_size = self.e_pool_in_size[t]
            self.cnn_block.e_pool_idx = self.e_pool_idx[t]
            x_t = self.cnn_block.decoder(x_t)
            x_over_t.append(x_t.unsqueeze(1))
            
        x = torch.cat(tuple(x_over_t), 1)
        
        return x
            
    def forward(self, x, return_with_encoded=False, **kwargs):
        '''
        x: [batchxietaxiphixdepthxfeature]
        y: [batchxietaxiphixdepthxfeature]
        '''

        x_dim_len = x.ndim

        squeeze_unused_dims = tuple(
            np.arange(-(x_dim_len - self.cnn_block.spatial_dim_len - 2), -1).tolist())

        for i, dim in enumerate(squeeze_unused_dims):
            x = x.squeeze(dim)

        if self.isvariational:
            encoded, latent_mu, latent_logvar = self.encoder(x, **kwargs)
        else:
            encoded = self.encoder(x, **kwargs)

    
        if self.get_encoded:
            return encoded
        else:
            x = self.decoder(encoded)

        for i, dim in enumerate(squeeze_unused_dims):
            x = x.unsqueeze(dim)


        if return_with_encoded:
            return x, encoded

        if self.isvariational:
            return x, latent_mu, latent_logvar

        return x

    def predict(self, X, **kwargs):
        self.eval()  # set into evaluation mode
        return ModelInterface().predict(self, X, **kwargs)


class CNNRNNAE_MultiDim_SPATIAL(nn.Module, AEModelTemplate):
    '''
    CNN(1d/2d/3d)+RNN+AE
    '''

    def __init__(self, feature_dim, latent_dim, target_dim, spatial_dims,
                 memorysize=5, **kwargs):
        
        # super function is used to use classes of parent class
        super().__init__()

        AEModelTemplate.__init__(self, datatype="ts", latent_dim=latent_dim, feature_dim=feature_dim,
                                 memorysize=memorysize, **kwargs)
         
        self.cnn_block = CNNAE_MultiDim_SPATIAL_Block(feature_dim, latent_dim, target_dim, spatial_dims,
                                        memorysize=memorysize, **kwargs)

        self.rnn_block = RNNAE_MultiDim_SPATIAL_Block(
                                     self.cnn_block.e_layer_num_inout, 
                                     self.cnn_block.e_layer_spatial_dim, self.cnn_block.d_layer_num_inout, self.cnn_block.d_layer_spatial_dim, 
                                     latent_dim, target_dim, 
                                 **kwargs)
        
        if self.isvariational:
            # VAE layer
            print("variational encoder is used...")
            self.variational_layer = VAEModelTemplate(
                dim_in=self.latent_dim, dim_out=latent_dim, variational_beta=kwargs.get("variational_beta", 0.01))
       
    def encoder(self, x, **kwargs):
        '''
        x: [batchxtxietaxiphixdepthxfeature]
        y: [batchx1xlatent]
        '''
        x = x.to(self.device)

        # CNN
        self.e_pool_idx = []
        self.e_pool_in_size = []
        t_size = x.size()[1]  # time window size
        x_over_t = []
        for t in range(t_size):
            x_t = x[:, t]

            x_t = self.cnn_block.encoder(x_t)
            
            x_over_t.append(x_t.unsqueeze(1))
            self.e_pool_idx.append(self.cnn_block.e_pool_idx)
            self.e_pool_in_size.append(self.cnn_block.e_pool_in_size)

        x = torch.cat(tuple(x_over_t), 1)
            
        # RNN
        x = self.rnn_block.encoder(x)

        # VAE
        if self.isvariational:
            x, latent_mu, latent_logvar = self.variational_layer(
                x, istrain_mode=self.training)

            return x, latent_mu, latent_logvar

        return x

    def decoder(self, x, **kwargs):
        # x = x.squeeze(1)

        # RNN
        x = self.rnn_block.decoder(x)

        # CNN
        t_size = x.size()[1]  # time window size
        x_over_t = []
        for t in range(t_size):
            x_t = x[:, t]
            self.cnn_block.e_pool_in_size = self.e_pool_in_size[t]
            self.cnn_block.e_pool_idx = self.e_pool_idx[t]
            x_t = self.cnn_block.decoder(x_t)
            x_over_t.append(x_t.unsqueeze(1))
            
        x = torch.cat(tuple(x_over_t), 1)
        
        return x

    def forward(self, x, keep_hidden_states=False, return_with_encoded=False, **kwargs):
        '''
        x: [batchxtxietaxiphixdepthxfeature]
        y: [batchxtxietaxiphixdepthxfeature]
        '''

        x_dim_len = x.ndim

        squeeze_unused_dims = tuple(
            np.arange(-(x_dim_len - self.cnn_block.spatial_dim_len - 2), -1).tolist())
        for i, dim in enumerate(squeeze_unused_dims):
            x = x.squeeze(dim)

        if not self.rnn_block.keep_hidden_states or not keep_hidden_states:
            self.rnn_block.e_rnn_hidden = []
            self.rnn_block.d_rnn_hidden = []

        self.rnn_block.keep_hidden_states = keep_hidden_states

        self.rnn_block.check_reset_state()

        if self.isvariational:
            encoded, latent_mu, latent_logvar = self.encoder(x, **kwargs)
        else:
            encoded = self.encoder(x, **kwargs)

        if self.get_encoded:
            return encoded
        else:
            x = self.decoder(encoded)

        for i, dim in enumerate(squeeze_unused_dims):
            x = x.unsqueeze(dim)

        # release memory
        if not self.rnn_block.keep_hidden_states:
            self.rnn_block.e_rnn_hidden = []
            self.rnn_block.d_rnn_hidden = []

        if return_with_encoded:
            return x, encoded

        if self.isvariational:
            return x, latent_mu, latent_logvar

        return x
    
    def predict(self, X, **kwargs):
        self.eval()  # set into evaluation mode
        return ModelInterface().predict(self, X, **kwargs)


# GraphSTAD
class CNNGNNRNNAE_MultiDim_SPATIAL(nn.Module, AEModelTemplate):
    '''
    CNN(1d/2d/3d)+GNN+RNN+AE: GraphSTAD, GNN is not supported by ONNX
    '''

    def __init__(self, feature_dim, latent_dim, target_dim, spatial_dims, node_edge_index,
                 memorysize=5, **kwargs):
        
        # super function is used to use classes of parent class
        super().__init__()

        AEModelTemplate.__init__(self, datatype="ts", latent_dim=latent_dim, feature_dim=feature_dim,
                                 memorysize=memorysize, **kwargs)
        
        self.cnn_block = CNNAE_MultiDim_SPATIAL_Block(feature_dim, latent_dim, target_dim, spatial_dims,
                                        memorysize=memorysize, **kwargs)
        
        self.gnn_block = GNN_MultiDim_SPATIAL_Block(feature_dim, latent_dim, target_dim, spatial_dims, 
                                        node_edge_index,
                                        e_num_gconv_layers=self.cnn_block.e_num_conv_layers,
                                        **kwargs)

        self.rnn_block = RNNAE_MultiDim_SPATIAL_Block(
                                     self.cnn_block.e_layer_num_inout, 
                                     self.cnn_block.e_layer_spatial_dim, self.cnn_block.d_layer_num_inout, self.cnn_block.d_layer_spatial_dim, 
                                     latent_dim, target_dim, e_layer_num_inout_g=self.gnn_block.e_layer_num_inout_g[-1],
                                 **kwargs)
        
        if self.isvariational:
            # VAE layer
            print("variational encoder is used...")
            self.variational_layer = VAEModelTemplate(
                dim_in=self.latent_dim, dim_out=latent_dim, variational_beta=kwargs.get("variational_beta", 0.01))
       
    def encoder(self, x, **kwargs):
        '''
        x: [batchxtxietaxiphixdepthxfeature]
        y: [batchx1xlatent]
        '''
        x = x.to(self.device)

        x_org = x.detach().clone()

        # CNN
        self.e_pool_idx = []
        self.e_pool_in_size = []
        t_size = x.size()[1]  # time window size
        x_c_over_t = []
        for t in range(t_size):
            x_t = x[:, t]
            x_t = self.cnn_block.encoder(x_t)
            x_c_over_t.append(x_t.unsqueeze(1))
            self.e_pool_idx.append(self.cnn_block.e_pool_idx)
            self.e_pool_in_size.append(self.cnn_block.e_pool_in_size)

        x_c = torch.cat(tuple(x_c_over_t), 1)

        # GNN
        x_g_over_t = []
        for t in range(t_size):
            x_t = x_org[:, t]
            x_t = self.gnn_block.encoder(x_t)
            x_g_over_t.append(x_t.unsqueeze(1))

        x_g = torch.cat(tuple(x_g_over_t), 1)

        # RNN
        x = self.rnn_block.encoder(x_c, x_exo=x_g)

        # VAE
        if self.isvariational:
            x, latent_mu, latent_logvar = self.variational_layer(
                x, istrain_mode=self.training)

            return x, latent_mu, latent_logvar

        return x

    def decoder(self, x, **kwargs):
        # x = x.squeeze(1)

        # RNN
        x = self.rnn_block.decoder(x)

        # CNN
        t_size = x.size()[1]  # time window size
        x_over_t = []
        for t in range(t_size):
            x_t = x[:, t]
            self.cnn_block.e_pool_in_size = self.e_pool_in_size[t]
            self.cnn_block.e_pool_idx = self.e_pool_idx[t]
            x_t = self.cnn_block.decoder(x_t)
            x_over_t.append(x_t.unsqueeze(1))
            
        x = torch.cat(tuple(x_over_t), 1)
        
        return x

    def forward(self, x, keep_hidden_states=False, return_with_encoded=False, **kwargs):
        '''
        x: [batchxtxietaxiphixdepthxfeature]
        y: [batchxtxietaxiphixdepthxfeature]
        '''

        x_dim_len = x.ndim

        squeeze_unused_dims = tuple(
            np.arange(-(x_dim_len - self.cnn_block.spatial_dim_len - 2), -1).tolist())
        for i, dim in enumerate(squeeze_unused_dims):
            x = x.squeeze(dim)

        if not self.rnn_block.keep_hidden_states or not keep_hidden_states:
            self.rnn_block.e_rnn_hidden = []
            self.rnn_block.d_rnn_hidden = []

        self.rnn_block.keep_hidden_states = keep_hidden_states

        self.rnn_block.check_reset_state()

        if self.isvariational:
            encoded, latent_mu, latent_logvar = self.encoder(x, **kwargs)
        else:
            encoded = self.encoder(x, **kwargs)

        if self.get_encoded:
            return encoded
        else:
            x = self.decoder(encoded)

        for i, dim in enumerate(squeeze_unused_dims):
            x = x.unsqueeze(dim)

        # release memory
        if not self.rnn_block.keep_hidden_states:
            self.rnn_block.e_rnn_hidden = []
            self.rnn_block.d_rnn_hidden = []

        if return_with_encoded:
            return x, encoded

        if self.isvariational:
            return x, latent_mu, latent_logvar

        return x
    
    def predict(self, X, **kwargs):
        self.eval()  # set into evaluation mode
        return ModelInterface().predict(self, X, **kwargs)


class GraphSTAD_Optimizer():
    '''
    Accuracy and speed optimizer
    '''
    @staticmethod
    def minus_plus_rbx_split(x, mask_split_config):
        '''dimension compression (DC) split spatial map of M and P sides'''

        range_minus = mask_split_config["range_minus"] 
        range_plus = mask_split_config["range_plus"] 
        # print(range_minus, range_plus)
        x_minus = torch.flip(x[:, :, range_minus[0]+32:range_minus[1]+32+1], [2])
        x_plus = x[:, :, range_plus[0]+32-1:range_plus[1]+32-1+1]
        # print(x_minus.shape, x_plus.shape)
        # print(x.shape)
        return x_minus, x_plus

    @staticmethod
    def minus_plus_rbx_split_merge(x, mask_split_config, merge_shape):
        '''stores map dimension, merging the P and M sides'''

        range_minus = mask_split_config["range_minus"] 
        range_plus = mask_split_config["range_plus"] 

        x_minus = x[:x.shape[0]//2]
        x_plus = x[x.shape[0]//2:]
        
        x_merge = torch.zeros((x.shape[0]//2, ) + merge_shape)

        x_minus = torch.flip(x_minus, [2])
        # print(x_minus.shape, x_plus.shape)

        x_merge[:, :, range_minus[0]+32:range_minus[1]+32+1].copy_(x_minus)
        x_merge[:, :, range_plus[0]+32-1:range_plus[1]+32-1+1].copy_(x_plus)
        # print("minus_plus_rbx_merge shape: ")
        # print(x_merge.shape)
        return x_merge
    
    @staticmethod
    def rin_transform(x, use_rnorm_spatial_div):
        '''applies reverse normalization (RIN)'''
        # Reversible normalization to mitigate LHC non-linearity: digi vs number of events
        # use_rnorm_spatial_div = 2 for the iphi axis

        x = x.type(torch.float64)
        mask_nonzero = x != 0
        mask_nonzero_sum = mask_nonzero.sum(dim=use_rnorm_spatial_div+1, keepdim=True)
        mask_nonzero_sum[mask_nonzero_sum==0] = 1 # to void div by zero
        x_med = x.sum(dim=use_rnorm_spatial_div+1, keepdim=True).div(mask_nonzero_sum) # calc average channel digi
        x_med[x_med==0] = 1 # to void div by zero
        x = x.div(x_med) # normalize digi by average channel digi
        x_med = x_med.type(torch.float32)
        x = x.type(torch.float32)
        return x, x_med

    @staticmethod
    def rin_inverse_transform(x, x_med):
        '''restores reverse normalization'''
        x = x.mul(x_med)
        return x

# GraphSTAD+GraphSTAD_Optimizer: RIN + DC
class GraphSTAD_RIN_DC_MultiDim_SPATIAL(nn.Module, AEModelTemplate):
    '''
    CNN(1d/2d/3d)+GNN+RNN+AE with GraphSTAD_Optimizer
    '''

    def __init__(self, model_alg, feature_dim, latent_dim, target_dim, spatial_dims,
                                                        memorysize=5, node_edge_index=None, 
                                                        use_rnorm_spatial_div=2, use_spatial_split=False, subdetector_name=None, **kwargs):
        
        # super function is used to use classes of parent class
        super().__init__()

        AEModelTemplate.__init__(self, **kwargs)

        self.model_alg = model_alg
        self.subdetector_name = subdetector_name
        self.use_rnorm_spatial_div = use_rnorm_spatial_div # 2 for the iphi axis, do not change
        self.use_spatial_split = use_spatial_split

        self.spatial_dims = list(spatial_dims[:])
        self.spatial_dims_full = tuple(spatial_dims[:])

        if self.use_spatial_split:
            if self.subdetector_name == "he":
                self.valid_spatial_ranges_dim = 0
                self.valid_spatial_ranges = [[-29, -16], [16, 29]]
            elif self.subdetector_name == "hb":
                self.valid_spatial_ranges_dim = 0
                self.valid_spatial_ranges = [[-16, -1], [1, 16]]
            else:
                raise f"valid_spatial_ranges is not implemnted for {self.subdetector_name}."
            
            self.mask_split_config= {"dim":self.valid_spatial_ranges_dim, "range_minus":self.valid_spatial_ranges[0], "range_plus":self.valid_spatial_ranges[1]}
            print(f"before DC spatial_dims: {spatial_dims}")
            spatial_dims = list(spatial_dims[:])
            spatial_dims[self.valid_spatial_ranges_dim] = self.valid_spatial_ranges[1][1] - self.valid_spatial_ranges[1][0] + 1
            print(f"after DC spatial_dims: {spatial_dims}")

            self.spatial_dims = spatial_dims

        MODEL_SEL = eval(self.model_alg)
        self.model = MODEL_SEL(feature_dim, latent_dim, target_dim, spatial_dims,
                                                        memorysize=memorysize, node_edge_index=node_edge_index, **kwargs)
        
        # self.isvariational = self.model.isvariational
        # self.reg_beta = self.model.reg_beta
        # # self.criterion = self.model.criterion
        # self.loss = self.model.loss
        # self.loss_scale = self.model.loss_scale
        # self.pred_loss = self.model.pred_loss

    def forward(self, x, *args, keep_hidden_states=False, return_with_encoded=False, isstateful=False, **kwargs):
        '''
        x: [batchxtxietaxiphixdepthxfeature]
        y: [batchxtxietaxiphixdepthxfeature]
        '''
        
        if not hasattr(self, "use_rnorm_spatial_div"):
            self.use_rnorm_spatial_div = None

        # RIN with sum per ls
        if self.use_rnorm_spatial_div is not None:
           x, x_med = GraphSTAD_Optimizer.rin_transform(x, self.use_rnorm_spatial_div)

        x_dim_len = x.ndim

        # split spatial M and P sides
        if self.use_spatial_split:
            x_full_shape = torch.tensor(x.shape[1:])
            x_minus, x_plus = GraphSTAD_Optimizer.minus_plus_rbx_split(x, self.mask_split_config)
            print("use_spatial_split: ", x_full_shape, x.shape, x_minus.shape, x_minus.shape)
            x = torch.cat((x_minus, x_plus), 0).to(self.device)

        squeeze_unused_dims = tuple(
            np.arange(-(x_dim_len - self.model.cnn_block.spatial_dim_len - 2), -1).tolist())
        
        # print(squeeze_unused_dims)
        for i, dim in enumerate(squeeze_unused_dims):
            x = x.squeeze(dim)

        if not isstateful:
            if not self.model.rnn_block.keep_hidden_states or not keep_hidden_states:
                self.model.rnn_block.e_rnn_hidden = []
                self.model.rnn_block.d_rnn_hidden = []

            self.model.rnn_block.keep_hidden_states = keep_hidden_states
        else:
            self.model.rnn_block.keep_hidden_states = True
            self.model.rnn_block.e_rnn_hidden = args[0]
            self.model.rnn_block.d_rnn_hidden = args[1]

        self.model.rnn_block.check_reset_state()

        if self.model.isvariational:
            encoded, latent_mu, latent_logvar = self.model.encoder(x, **kwargs)
        else:
            encoded = self.model.encoder(x, **kwargs)

        # x = self.decoder(encoded)

        if self.model.get_encoded:
            return encoded
        else:
            x = self.model.decoder(encoded)

        for i, dim in enumerate(squeeze_unused_dims):
            x = x.unsqueeze(dim)

        # restore spatial M and P sides
        if self.use_spatial_split:
            x = GraphSTAD_Optimizer.minus_plus_rbx_split_merge(x, self.mask_split_config, tuple(x_full_shape.tolist())).to(self.device)

        # restore RIN
        if self.use_rnorm_spatial_div is not None:
            x = GraphSTAD_Optimizer.rin_inverse_transform(x, x_med).to(self.device)

        if not isstateful:
            # release memory
            if not self.model.rnn_block.keep_hidden_states:
                self.model.rnn_block.e_rnn_hidden = []
                self.model.rnn_block.d_rnn_hidden = []

            if return_with_encoded:
                return x, encoded

            #if self.isvariational and return_with_encoded:
            if self.model.isvariational:
                return x, latent_mu, latent_logvar

            return x
        
        else:
            e_rnn_hidden, d_rnn_hidden = self.model.rnn_block.e_rnn_hidden[:], self.model.rnn_block.d_rnn_hidden[:]
            self.model.rnn_block.e_rnn_hidden = []
            self.model.rnn_block.d_rnn_hidden = []
             
            if return_with_encoded:
                return x, encoded, e_rnn_hidden, d_rnn_hidden

            if self.isvariational:
                return x, latent_mu, latent_logvar, e_rnn_hidden, d_rnn_hidden
     
            return x, e_rnn_hidden, d_rnn_hidden
    
    def predict(self, X, **kwargs):
        self.eval()  # set into evaluation mode
        return ModelInterface().predict(self, X, **kwargs)



