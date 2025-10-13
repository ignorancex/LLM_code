# -*- coding: utf-8 -*-
"""
Created on Fri Feb 7 17:39:37 2025

@author: Mulugeta W.Asres
@email: muleina2000@gmail.com, mulugetawa@uia.no, mulugeta.asres@cern.ch
"""


import torch
import argparse
import sys
import os
import json

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(current_path))

from model_modulizer import *
import utilities as util

# import model_exported_tester as Tester

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)

root_path = os.path.dirname(os.path.dirname(current_path))
result_path = "{}//results".format(root_path)

predict_processors_dict = {
                           'DQM_AD_ONLINE': dqm_ad_detect_processor_online,
                           }

def exporter(**kwargs):
    '''
    exports .pkl mmodel integrating all data pipelines from raw input to predict output
    '''
    util.print_dict(kwargs)
    issave = kwargs.get("issave", True)
    model_type = kwargs.get("model_type", None)
    model_suffix = kwargs.get("model_suffix", None)
    model_dir = kwargs.get("model_dir", None)
    data_kwargs = json.loads(kwargs.get("data_kwargs", "{}"))
    prod_dir = kwargs.get("prod_dir", "prod_model")
    isexport_onnx = kwargs.get("isexport_onnx",  True)

    model_setting = {}
    if model_type in ['DQM_AD_ONLINE']:

        model_system_profile = load_ad_model_wrapper(
           model_filename="AE_MODEL.pkl", model_dirpath=f"{result_path}/{model_dir}/{model_suffix}", isplot=False, isshow=True)

        admodel = model_system_profile["admodel"]
        print(vars(admodel))
        
        model_prod_dir = "{}_m{}".format(model_type, admodel.memorysize)

        if model_type.startswith('DQM_AD'):
                
            admodel.e_pool_idx = []
            admodel.e_pool_in_size = []
            admodel.train_data_config.pop('meta_data', None) 

            if 'cal_mask' in data_kwargs.keys():
                sys_masked_segmentation = util.load_npdata(data_kwargs['cal_mask']['mask_filename'], filepath=util.join_path([
                                                                util.data_path, data_kwargs['cal_mask']['mask_dir']])).astype(bool)
                data_kwargs['mask'] = ~sys_masked_segmentation
            else:
                data_kwargs['mask'] = admodel.train_data_config.pop('mask', None) 

            if data_kwargs['mask'] is None:
                raise "Data mask can not be empty!"
            
            print("mask: ", data_kwargs['mask'][:, -1, :].sum())
            
            if admodel.train_data_config["input_scaling_alg"] == "minmax":
                admodel.train_data_config["input_scaler"] = TorchMinMaxScaler(
                    skl_scaler=admodel.train_data_config["input_scaler"])
            elif admodel.train_data_config["input_scaling_alg"] == "max":
                admodel.train_data_config["input_scaler"] = TorchMaxScaler(
                    skl_scaler=admodel.train_data_config["input_scaler"])
            else:
                raise "Data scaler it not mapped into Torch!"
        
            ae_pred_err_spatial_hist_train = model_system_profile['train_err_hist']
            ae_pred_err_window_spatial_hist_train = model_system_profile['train_err_window_hist']
            print(ae_pred_err_spatial_hist_train.shape)
            print(ae_pred_err_window_spatial_hist_train.shape)
            
            admodel.pred_err_spatial_scaler = PredErrorScaler(mean_err_train=torch.FloatTensor(ae_pred_err_spatial_hist_train["mean"].values),
                                                              std_err_train=torch.FloatTensor(ae_pred_err_spatial_hist_train["std"].values), reshape_dim=(1,) + data_kwargs['mask'].shape+ (1,))
            admodel.pred_err_window_spatial_scaler = PredErrorScaler(mean_err_train=torch.FloatTensor(ae_pred_err_window_spatial_hist_train["mean"].values),
                                                                     std_err_train=torch.FloatTensor(ae_pred_err_window_spatial_hist_train["std"].values), reshape_dim=(1,) + data_kwargs['mask'].shape+ (1,))
            print(vars(admodel))

            admodel.sys_sel = data_kwargs["sys_sel"]
            if not hasattr(admodel, "use_rnorm_sum_window_mean_div"):
                admodel.use_rnorm_sum_window_mean_div = False

            model_system_profile["admodel"] = admodel

            data_kwargs['normalizer_model'] = None


    modelProdObj = ModelProdNN(model_type=model_type, model_profile=model_system_profile,
                             predict_processor=predict_processors_dict[model_type], data_kwargs=data_kwargs, model_setting=model_setting)

    model_dirpath_prod_onnx = model_dirpath_prod = f"{result_path}/{model_dir}/{prod_dir}/{model_prod_dir}"
    os.makedirs(model_dirpath_prod, exist_ok=True)

    if issave:
        from datetime import datetime
        date_time_tag = datetime.now().strftime("%m_%d_%Y_%Hh%M") # get current date and time
        prod_modelname = "{}_v{}".format(model_suffix, date_time_tag)
        save_prod_model(prod_modelname, modelProdObj, model_dirpath=model_dirpath_prod)

        print("#"*60)
        print("Exporting to onnx prod model dir: ")
        print(model_dirpath_prod_onnx)

        onnx_model_gen_script_template = "python cmssw_onnx_exporter.py -mode e_s -pd /{model_dir}/{prod_dir}/{model_prod_dir}/ -ms {prod_modelname}"
        onnx_model_gen_script = onnx_model_gen_script_template.format(model_dir=model_dir, prod_dir=prod_dir, model_prod_dir=model_prod_dir, prod_modelname=prod_modelname)
        print("\n onnx_model_gen_script \n: ", onnx_model_gen_script) # stateful

        if isexport_onnx:  
            os.system(onnx_model_gen_script)

def main(**kwargs):
    mode = kwargs.pop("mode", None)
    if mode == "e":
        exporter(**kwargs)
    else:
        raise "Only export is suppprted. select mode 'e'."

if __name__ == '__main__':
    """Main entry function."""

    parser = argparse.ArgumentParser(description="model exporter source codes")

    parser.add_argument('-mode', '--mode', type=str,
                        help='mode: (e)xport or (t)est', default='e')

    parser.add_argument('-save', '--issave', action='store_true',
                        help='issave: save results', default=False)
    parser.add_argument('-onnx', '--isexport_onnx', action='store_true',
                        help='isexport_onnx: generate ONNX', default=False)
    parser.add_argument('-pd', '--prod_dir', type=str, default="prod_models", 
                        help='production model store directory')
    parser.add_argument('-mt', '--model_type', type=str, choices=['DQM_AD_ONLINE'],
                        # default="",
                        help='model type choose for predict_processor')
    parser.add_argument('-ms', '--model_suffix', type=str,
                        help='model name tag')
    parser.add_argument('-md', '--model_dir', type=str,
                        help='relative model_dir w.r.t ../result/"dataset"/model/')
    parser.add_argument('-kd', '--data_kwargs',  type=str, default="{}",
                        help='data_kwargs such as dict "{\"isclean:\": true, \"down_sample_min\": 10, \"nodropout\": [\"encoder\"]}"')

    args = vars(parser.parse_args())
    print(args)
    main(**args)

# without onnx
# python model_modulized_exporter.py -mt DQM_AD_ONLINE -md HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB/model/he_m_5_is_minmax_ns_1 -ms GraphSTAD_RIN_DC__CNNRNNAE_MultiDim_SPATIAL_HE -kd "{\"sys_sel\": \"hhbhehohfe\", \"year\":2022, \"mapnorm_mode\":\"vars_prod\",\"normalizer_vars\":[\"NumEvents\"],\"scale_norm_depth\":1}" -save

# with onnx
# python model_modulized_exporter.py -mt DQM_AD_ONLINE -md HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB/model/he_m_5_is_minmax_ns_1 -ms GraphSTAD_RIN_DC__CNNRNNAE_MultiDim_SPATIAL_HE -kd "{\"sys_sel\": \"hhbhehohfe\", \"year\":2022, \"mapnorm_mode\":\"vars_prod\",\"normalizer_vars\":[\"NumEvents\"],\"scale_norm_depth\":1}" -save -onnx
