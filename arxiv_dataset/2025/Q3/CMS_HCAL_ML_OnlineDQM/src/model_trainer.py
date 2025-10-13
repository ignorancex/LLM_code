
"""
Created on Mon Dec 07 17:33:21 2023

@author: Mulugeta W. Asres, mulugetawa@uia.no

DESMOD: ML Models Trainer for HCAL DQM Digioccupancy 

"""
import argparse
import os, sys
import seaborn as sns
import pandas as pd
import numpy as np
from functools import reduce
import numpy as np, pandas as pd
from tqdm import tqdm
import copy
import json
import time
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from torchsummaryX import summary
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, roc_auc_score

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)
data_path = os.path.abspath(os.path.dirname(current_path)+"/data")
sys.path.append(data_path)
result_path = os.path.abspath(os.path.dirname(current_path)+"/results")

import utilities as util
from model_datasets import * 
from models_spatial import * 

model_path_template = "{}//{}//model//{}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SEED
SEED = 123

def set_reproducible(SEED):
    if torch.cuda.is_available():
        torch.set_num_threads(1)
        torch.cuda.manual_seed(SEED)
    else:
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

set_reproducible(SEED)

class PredErrorScaler():
    def __init__(self, err_train=None, mean_err_train=None, std_err_train=None, **kwargs):
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

    def transform(self, pred_err: torch.Tensor, ignore_mean=False):
        pred_err_norm = pred_err

        if self.mean_err_train is not None:
            if not ignore_mean:
                pred_err = pred_err - self.mean_err_train

        if self.std_err_train is not None:
            pred_err_norm = torch.divide(pred_err.abs(), self.std_err_train)

        if self.mask is not None:
            pred_err_norm[:, self.mask, :] = torch.nan

        return pred_err_norm
    

class ModelTrainer():

    def __init__(self, shuffle=False, **kwargs):
        self.shuffle = shuffle 
        self.batch_size = None
        self.train_loader = None
        self.valid_loader = None
        self.model = None

    def prepare_dataloader(self, train_dataset, **kwargs):
        print("prepare data loaders....")

        self.batch_size = kwargs.get("batch_size", 0)
        self.valid_size = kwargs.get("valid_size", 0)
        
        if not isinstance(train_dataset, DatasetTemplate):
            raise "the input data must be object of DatasetTemplate."

        
        print("training_data: ", train_dataset.shape())
        print("loader batch size: {}".format(self.batch_size))

        num_train = len(train_dataset)
        indices = list(range(num_train))

        test_idx = []
        num_train = len(indices)

        val_offset = None
        num_data_sources = np.max((len(train_dataset.source_files), 1))
        if not self.shuffle:
            val_offset = train_dataset.memorysize

            # get validation from each data source or runId
            data_size_per_rbx_estimate = num_train//num_data_sources
            train_idx = []
            valid_idx = []
            valid_size_slice = self.valid_size 
            for i in range(num_data_sources):
                train_idx_slice, valid_idx_slice = train_test_split(indices[i*data_size_per_rbx_estimate:(i+1)*data_size_per_rbx_estimate],
                                                                    test_size=valid_size_slice, shuffle=self.shuffle)

                train_idx.extend(train_idx_slice)
                valid_idx.extend(valid_idx_slice[val_offset:])
        else:
            train_idx, valid_idx = train_test_split(
                indices, test_size=self.valid_size, random_state=100, shuffle=self.shuffle)

        print("dataloader sizes (batch_size:{}, val_offset={}, num_data_sources={}): train={}, val={}, test={}".format(self.batch_size, val_offset, num_data_sources, len(train_idx), len(valid_idx), len(test_idx)))
        if val_offset >= len(valid_idx):
            "increase validation data size to be larger than val_offset."
      
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
       
        # DataLoader generator to preserve reproducibility
        g = torch.Generator()
        g.manual_seed(SEED)

        # load training data in batches
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       sampler=train_sampler,
                                       drop_last=True,
                                       generator=g,
                                       )  

        # load validation data in batches
        self.valid_loader = DataLoader(train_dataset,
                                        batch_size=self.batch_size,
                                        drop_last=True,
                                        sampler=valid_sampler,
                                        )
        
        print("indices: {}, train_indics: {}, val_indices: {}".format(
            len(train_dataset), len(train_idx), len(valid_idx)))
        print("train_loader: {}, val_loader: {}".format(
            len(self.train_loader), len(self.valid_loader)))

    def config_model(self, modelObj, X, **kwargs):

        train_data_config = X.__dict__.copy()
        train_data_config.pop('samples', None)
        modelObj.train_data_config = copy.deepcopy(train_data_config)
        modelObj.criterion = nn.MSELoss()
        modelObj = modelObj.to(device=device)

        # display model summary
        data_shape = list(X.shape())
        print(f"data_shape: {data_shape}")
        # make batch 1, to fix memory overflow for large num of batchs
        data_shape[0] = 1
        data_shape = tuple(data_shape)
        print(f"data_shape: {data_shape}")
        try:
            dummy_input = torch.zeros(data_shape).to(device)
            summary(modelObj, dummy_input)
        except Exception as e:
            print("model summary check is failed: {}".format(e))

        return modelObj, X
   
    def get_loss(self, modelObj, optimizer, batch_x, batch_y, **kwargs):

        if modelObj.training:
            input_data = Variable(
                batch_x, requires_grad=False).to(device=device)
            target = Variable(batch_y, requires_grad=False).to(device=device)

            optimizer.zero_grad()  # add to avoid accumulation

            # predict output
            output = modelObj(input_data.contiguous(), **kwargs)

            # calculate loss
            if isinstance(output, tuple):
                loss = modelObj.loss(target, *output, **kwargs)
            else:
                loss = modelObj.loss(target, output, **kwargs)

            # Backward pass
            loss.mean().backward()
            # Update parameters
            optimizer.step()

            return loss.data.item()

        else:
            with torch.no_grad():
                input_data = Variable(batch_x, requires_grad=False).to(
                    device=device)
                target = Variable(
                    batch_y, requires_grad=False).to(device=device)

                # predict output
                output = modelObj(input_data.contiguous(), **kwargs)

                # calculate loss
                if isinstance(output, tuple):
                    loss = modelObj.loss(target, *output, **kwargs)
                else:
                    loss = modelObj.loss(target, output, **kwargs)

                return loss.data.item()
                
    def train_model(self, modelObj, *args, **kwargs):
        print("model training...")

        num_epochs = kwargs.get("num_epochs", 5)
        lr = kwargs.get("learning_rate", 1e-3)
        weight_decay = kwargs.get("weight_decay", 1e-7)
        vae_reg_beta = kwargs.get("vae_reg_beta", None)
        keep_hidden_states = kwargs.get("keep_state", False)

        verbose = kwargs.get("verbose", 1)
        early_stop_epoch = kwargs.get("early_stop_epoch", num_epochs)
       
        hide_mask = torch.BoolTensor(np.expand_dims(kwargs.get(
            "mask", None).astype(bool), axis=(0, 1, -1))).to(device)
    
        print("mask: ", hide_mask.shape)

        # training optimizer
        optimizer = optim.Adam(modelObj.parameters(), lr=lr,weight_decay=weight_decay,
                amsgrad=True)
        
        num_params = sum(p.numel() for p in modelObj.parameters() if p.requires_grad)
        print("Number of parameters: {}".format(num_params))

        train_num_batches = np.max([len(self.train_loader), 1])
        val_num_batches = np.max([len(self.valid_loader), 1])

        print(f"train_num_batches:{train_num_batches}, val_num_batches:{val_num_batches}")
        
        train_loss = []
        valid_loss = []

        t_s = time.time()

        best_model = copy.deepcopy(modelObj.state_dict())
        es_min_loss_tracker = None
        early_stop_counter = 0

        for epoch in range(num_epochs):

            # training
            modelObj.train()  # set to train mode
            train_loss.append(0)

            for i, batch in enumerate(self.train_loader):

                # reconstruction loss
                train_loss[-1] += float(self.get_loss(modelObj, optimizer,
                                                            batch[0], batch[1], vae_reg_beta=vae_reg_beta, mask=hide_mask, keep_hidden_states=keep_hidden_states))
               
            train_loss[-1] /= train_num_batches

            # validation
            modelObj.eval()  # set to predict mode
            valid_loss.append(0)

            # for temporal model
            modelObj.forced_reset_state()
    
            for i, batch in enumerate(self.valid_loader):
               valid_loss[-1] += float(self.get_loss(modelObj, optimizer,
                                                            batch[0], batch[1], vae_reg_beta=vae_reg_beta, mask=hide_mask, keep_hidden_states=keep_hidden_states))

            valid_loss[-1] /= val_num_batches
            
            if not isinstance(es_min_loss_tracker, list):
                es_min_loss_tracker = [
                    epoch, valid_loss[-1], (time.time() - t_s)/60, train_loss[-1]]
                best_model = copy.deepcopy(modelObj.state_dict())
            elif es_min_loss_tracker[1] > valid_loss[-1]:
                es_min_loss_tracker = [
                    epoch, valid_loss[-1], (time.time() - t_s)/60, train_loss[-1]]
                best_model = copy.deepcopy(modelObj.state_dict())
                early_stop_counter = 0
            elif early_stop_counter >= early_stop_epoch:
                break

            early_stop_counter += 1
            if verbose > 0:
                if epoch % verbose == 0:
                    print("Epoch [{:d}]: lr:{:0.6f} train_loss:{:0.6f} val_loss:{:0.6f}, ES:(epoch:{:d} val_loss:{:0.6f} t:{:0.3f})".format(
                        epoch, optimizer.param_groups[0]['lr'], train_loss[-1], valid_loss[-1], es_min_loss_tracker[0], es_min_loss_tracker[1], es_min_loss_tracker[2]))
   
        print("Early stopping at epoch:{:d} best_epoch:{:d} best_val:{:0.6f} best_epoch_train_time:{:0.6f}".format(
            epoch, es_min_loss_tracker[0], es_min_loss_tracker[1], es_min_loss_tracker[2]))

        train_time = (time.time() - t_s)/60
        print("process time: {:0.3f} min".format(train_time))

        es_best_epoch = {"epoch": es_min_loss_tracker[0]+1,
                        "val_loss": np.round(es_min_loss_tracker[1], 5),
                        "time_mint": np.round(es_min_loss_tracker[2], 5),
                        "train_loss": np.round(es_min_loss_tracker[3], 5)
                        }
        print("es_best_epoch: ", es_best_epoch)

        modelObj.load_state_dict(best_model)
        modelObj.es_best_epoch_dict = es_best_epoch.copy()
        train_time = (time.time() - t_s)/60
        print("process time: {:0.3f} min".format(train_time))
        train_history = {"train": np.round(np.array(train_loss), 6),
                         "val": np.round(np.array(valid_loss), 6),
                         "es_best_epoch": es_best_epoch,
                         "train_time_mint": np.round(train_time, 3)
                         }
        
        print("done!")

        # for temporal model
        modelObj.forced_reset_state()
        return modelObj, train_history


def clf_eval_summary(y_true, y_pred, classes=None, issave=False, filepath=None, isplot=True, prefix=""):

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    try:
        cm = cm.ravel() # tn, fp, fn, tp
        
        try:
            auc_score = roc_auc_score(y_true, y_pred)
        except:
            auc_score = np.nan

        clf_report = pd.DataFrame(columns=["score"])
        clf_report.loc["auc"] = auc_score
        clf_report.loc["FPR"] = cm[1]/(cm[1] + cm[0])

        print(clf_report)
    except Exception as ex:
        clf_report = None
        # incase of no anomaly labels.
        print("error in clf_eval_summary. {}".format(ex))

    return clf_report

def anomaly_channel_clf_eval(pred_dict_anml, anml_label, model_ae_pred_err_aml_thr, model_ae_pred_err_window_aml_thr=None, mask=None, sel_index=None, return_report=False, err_tag="_spatial", **kwargs):
    print("#"*70)
    memorysize = anml_label.shape[1]
    if sel_index is None:
        sel_index = slice(0, -1)
    
    clf_report = None
    if model_ae_pred_err_window_aml_thr is None:
        print("anomaly per channel on the last lumisection...")
        anml_pred_lbl_np = torch.BoolTensor(pred_dict_anml["pred_err{}".format(err_tag)]>model_ae_pred_err_aml_thr)
        
        # last time stamp                     
        clf_report = clf_eval_summary(anml_label[sel_index, -1, ~mask, ::].reshape(-1, 1).detach().numpy(), 
                                                anml_pred_lbl_np[memorysize-1::memorysize][sel_index, ~mask, ::].reshape(-1, 1).detach().numpy(),
                                            classes=["normal", "anomaly"], isplot=False)
        if clf_report is not None: 
            clf_report.loc["aml_thr"] = model_ae_pred_err_aml_thr
        
        # clf_report
        print("anomaly per channel on lumisections in rec window...")
  
    else:
        print("anomaly per channel on the last lumisection with windowed score...")
        anml_pred_lbl_np = torch.BoolTensor(pred_dict_anml["pred_err_window{}".format(err_tag)]>model_ae_pred_err_window_aml_thr)

        # last time stamp
        clf_report = clf_eval_summary(anml_label[sel_index, -1, ~mask, ::].reshape(-1, 1).detach().numpy(), 
                                                anml_pred_lbl_np[memorysize-1::memorysize][sel_index, ~mask, ::].reshape(-1, 1).detach().numpy(),
                                            classes=["normal", "anomaly"], isplot=False)
    
        if clf_report is not None: 
            clf_report.loc["aml_thr"] = float(model_ae_pred_err_window_aml_thr)

    print("#"*70)
    return clf_report
    
def model_config_prepare(**kwargs):
    model_config = {
        "model_alg": kwargs.get("model_alg", ""),
        "model_arch": kwargs.get("model_arch", ""),
        "model_kwargs": kwargs.get("model_kwargs", "")
    }
    model_dict = {
        "model_id": kwargs.get("model_id", ""),
        "train": {"arg": kwargs.get("train_arg", ""),
                  "train_history": kwargs.get("train_history", "")},
        "test": {"arg": kwargs.get("test_arg", ""),
                 },
        "model": kwargs.get("model", ""),
        "model_config": model_config
    }
    return model_dict

def create_unique_model_filepath(model_alg, train_data):
    print("create_unique_model_filepath...")

    num_data_sources = np.max((len(train_data.source_files), 1))
    dataset_name = train_data.dataset_name.replace('/', '_')
    model_dir = "//{}_m_{}_is_{}_ns_{}".format(
        train_data.subdetector_name, train_data.memorysize, train_data.input_scaling_alg, num_data_sources)

    # unique model_id
    model_id = np.random.randint(10000, size=1)[0]

    model_folder = "{}//{}_{}".format(model_dir, model_alg, model_id)
    while os.path.exists(model_path_template.format(result_path, dataset_name, model_folder)):
        model_id = np.random.randint(10000, size=1)[0]
        model_folder = "{}//{}_{}".format(model_dir, model_alg, model_id)

    model_dirpath = model_path_template.format(result_path, dataset_name, model_folder)
    os.makedirs(model_dirpath, exist_ok=True)

    print("saving model path: ", model_dirpath)
    return model_dirpath, model_id
    
def plot_train_metric(train_history, **kwargs):
    filename = kwargs.get("filename", None)
    filepath = kwargs.get("filepath", None)
    issave = kwargs.get("issave", False)

    fig, ax = plt.subplots(figsize=(6, 3))
    for key, value in train_history.items():
        if key in ["train", "val"]:
            plt.plot(value, label=key)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()

    if "train_time_mint" in train_history.keys():
        plt.title("training_time (mint): {} \nearly_stop: {}".format(
            train_history["train_time_mint"], train_history["es_best_epoch"]))

    if issave:
        util.save_figure(f"{filepath}/{filename}", fig, isshow=False,
                         issave=issave, dpi=100)
    else:
        plt.show()

def ad_model_generator(**kwargs):
    
    default_setting = {
        "dataset": None,
        "model_alg": None,
        "train_data_filename": "train_dataset",
        "model_dir": "",
        "data_dirpath": "",
        "e_num_conv_layers": 4, 
        "latent_dim": 4, # [2, 4, 8, 16, 32] increase this to improve accuracy and reduce to speed up training
        "rnn_model": "lstm",
        "shuffle": False,
        "isplot": False,
        "num_epochs": 10,
        "early_stop_epoch": 10,
        "verbose": 1,
        # "learning_rate": 0.001, # for cnn and rnn
        "learning_rate": 0.01, # for gnn
        "weight_decay": 1e-7,
        "num_layers": 1,
        "isvariational": True,
        "reg_beta": 0.0001,
        "batch_size": 4,
        "train_size_perc": 1.00,
        "valid_size": 0.20,
        "optimizer": "adam"
    }
    _kwargs = copy.deepcopy(kwargs)
    kwargs.update(default_setting)
    kwargs.update(_kwargs)
    
    data_dirpath = kwargs.get("data_dirpath", rf"{data_path}")
    train_data_filename = kwargs.get("train_data_filename", "train_dataset")

    kwargs_model = json.loads(kwargs.pop("kwargs_model", "{}"))

    print(kwargs_model)
    
    # kwargs = json.loads(kwargs["kwargs"])
    kwargs["train_data_filename"] = train_data_filename
    
    model_alg = kwargs.get("model_alg", "CNNRNNAE_MultiDim_SPATIAL")
    shuffle = kwargs.get("shuffle", False)
    batch_size = kwargs.get("batch_size", 4)
    latent_dim = kwargs.get("latent_dim", 32)
    vae_reg_beta = kwargs.get("vae_reg_beta", 0.001)
    kwargs["vae_reg_beta"] = vae_reg_beta
    valid_size = kwargs.get("valid_size", 0.20)

    # load train_dataset
    print("loading train_data: {}...".format(train_data_filename))
    train_data = util.load_pickle("{}/{}.pkl".format(data_dirpath, train_data_filename))
    train_data.memorysize = train_data.timewindow_size
    print("feature_dim: {} target_dim: {}".format(
        train_data.shape(), train_data.shape(target=True)))
    train_data_config = train_data.__dict__.copy()
    train_data_config.pop('samples', None)

    print(train_data_config)

    feature_dim = train_data.shape()[-1]
    # (6D):[samples X t X ieta X iphi x depth x feature], t=1 for iid datasets
    target_dim = train_data.shape(target=True)[-1]
    spatial_dims = train_data.shape()[2:-1]
    print(f"spatial_dims: {spatial_dims}")
    if spatial_dims[-1] == 1: spatial_dims = spatial_dims[:-1]
        
    print(f"spatial_dims: {spatial_dims}")

    memorysize = train_data.memorysize
   
    args = kwargs.copy()
    print([args.pop(k, None) for k in ["datatype", "latent_dim", "num_layers"]])

    args.update(kwargs_model)

    use_graphstad_optimizer = args.pop("use_graphstad_optimizer", False)
    use_spatial_split = args.pop("use_spatial_split", False)

    # init AD model
    node_edge_index = None
    if "GNN" in model_alg: 
        if use_spatial_split:
            raise "node_edge_index is not generated yet for the GraphSTAD optimizer with use_spatial_split activated. Use use_spatial_split=False."
        
        node_edge_index_filename = args.pop("node_edge_index_filename", None)
        if not isinstance(node_edge_index_filename, str):
            raise f"node_edge_index_filename: {node_edge_index_filename} must is string data, holding the path to node_edge_index data of the detector."
        
        node_edge_index = util.load_npdata(rf"{node_edge_index_filename}")
        node_edge_index.shape
    

    if not use_graphstad_optimizer:
        MODEL_SEL = eval(model_alg)
        ae_model = MODEL_SEL(feature_dim, latent_dim, target_dim, spatial_dims,
                                memorysize=memorysize, node_edge_index=node_edge_index,
                                **args
                                )
    else:
        model_alg = args.pop("model_alg", "CNNRNNAE_MultiDim_SPATIAL")
        ae_model = GraphSTAD_RIN_DC_MultiDim_SPATIAL(model_alg, feature_dim, latent_dim, target_dim, spatial_dims,
                                                        memorysize=memorysize, node_edge_index=node_edge_index, use_rnorm_spatial_div=2, 
                                                     use_spatial_split=use_spatial_split, 
                                                    **args)

    print(ae_model)

    # Number of runid in the training datasets
    train_data.source_files = [None]

    model_dirpath, model_id = create_unique_model_filepath(f"GraphSTAD_RIN_DC__{model_alg}" if use_graphstad_optimizer else model_alg, train_data)

    print(ae_model)
    # model trainer
    modeltrainerObj = ModelTrainer(shuffle=False)

    # model training dataloader preparation
    modeltrainerObj.prepare_dataloader(
        train_data, batch_size=batch_size, valid_size=valid_size)

    # model training configuration
    ae_model, train_data = modeltrainerObj.config_model(ae_model, train_data, **kwargs)
   
    kwargs["mask"] = train_data.mask # is inactive mask
    
    ae_model.mask_dims = train_data.mask.shape
    # model training
    ae_model, train_history = modeltrainerObj.train_model(ae_model, **kwargs)
    
    mask = kwargs.pop("mask", None)
    kwargs.update(kwargs_model)
    kwargs["model_dirpath"] = model_dirpath
    # print(kwargs)

    # prepare model and its config for saving
    model_dict = model_config_prepare(
                        model_id=model_id,
                        model_alg=model_alg,
                        model_arch=str(ae_model),
                        model_kwargs=kwargs,
                        train_arg=train_data_config,
                        model=ae_model.to(torch.device("cpu")),
                        train_history=train_history
                    )

    kwargs["default_setting"] = default_setting
    kwargs["kwargs_model"] = kwargs_model
    kwargs["model_arch"] = str(ae_model)
    kwargs["training_time_mint"] = train_history["train_time_mint"]
    kwargs["es_best_epoch"] = train_history["es_best_epoch"]

    kwargs.update(kwargs_model)

    util.save_json(rf"{model_dirpath}/model_settings.json", kwargs)
    util.save_json(rf"{model_dirpath}/model_train_loss.json", model_dict["train"]["train_history"])
    
    plot_train_metric(
        model_dict["train"]["train_history"], issave=True, filename="model_train_loss", filepath=model_dirpath)
    
    # save model, md models become atributes in train_perf_report
    model_dict["model"] = ae_model.to(torch.device("cpu"))

    util.save_pickle(rf"{model_dirpath}/AE_MODEL.pkl", model_dict)
    
    ae_model.forced_reset_state()
    ae_model, _ = ModelInterface().train_perf_report(ae_model, train_data, mask=mask,
                                                    model_dirpath=model_dirpath,
                                                    keep_hidden_states=False,
                                                    issave=True)
       
if __name__ == '__main__':
    # constructing argument parsers
    parser = argparse.ArgumentParser(description="gpu model trainer")
    parser.add_argument('-ma', '--model_alg', default="",
                        help='model algoritm selection')
    parser.add_argument('-ds', '--dataset', type=str, 
                        # choices=['HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB'], 
                        default="",
                        help='data source selection')
    parser.add_argument('-dp', '--data_dirpath', default=data_path, type=str,
                        help='data_dirpath path to training and val data pickle files')
    parser.add_argument('-td', '--train_data_filename',
                        type=str,
                        help='train data in dataset filename')
    parser.add_argument('-k', '--kwargs', type=str, default="",
                        help='training parameters')
    parser.add_argument('-km', '--kwargs_model', type=str, default="{}",
                        help='kwargs_model such as dict "{\"key1\": value, \"key2\": [\"value21\", \"value22\"]}"')

    args = parser.parse_args()
    print(args)
    args = vars(args)
    ad_model_generator(**args)

# non temporal
# python model_trainer.py -ma CNNFNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_iid

# temporal
# python model_trainer.py -ma CNNRNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_ts

# temporal + graph
# python model_trainer.py -ma CNNGNNRNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -km "{\"node_edge_index_filename\":\"C:\\Users\\mulugetawa\\OneDrive - Universitetet i Agder\\CERN\\InductionProject\\CMS_HCAL_ML_OnlineDQM\\data\\HCAL_CONFIG\\he_graph_edge_array_indices_depth.npy\"}" -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_ts

# temporal + graph + GraphSTAD_Optimizer RIN
# python model_trainer.py -ma CNNGNNRNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -km "{\"use_graphstad_optimizer\":\"true\", \"node_edge_index_filename\":\"C:\\Users\\mulugetawa\\OneDrive - Universitetet i Agder\\CERN\\InductionProject\\CMS_HCAL_ML_OnlineDQM\\data\\HCAL_CONFIG\\he_graph_edge_array_indices_depth.npy\"}" -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_ts

# temporal + GraphSTAD_Optimizer + DC
# python model_trainer.py -ma CNNRNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -km "{\"subdetector_name\":\"he\"}" -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_ts

# temporal + GraphSTAD_Optimizer + DC + RIN
# python model_trainer.py -ma CNNRNNAE_MultiDim_SPATIAL -ds HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc/HEHB -km "{\"subdetector_name\":\"he\", \"use_spatial_split\":\"true\",\"use_graphstad_optimizer\":\"true\"}" -dp "C:\Users\mulugetawa\OneDrive - Universitetet i Agder\CERN\InductionProject\CMS_HCAL_ML_OnlineDQM\data" -td HCAL_ONLINE_DQM__ZeroBias__2022__cut20fc_HEHB_he_train_dataset_ts
