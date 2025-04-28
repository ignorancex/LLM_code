import logging
import os
import warnings
from typing import List, Sequence
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import yaml 
import seaborn as sns 
import matplotlib.pyplot as plt
import umap 
import torch
import torch.nn.functional as F

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.WandbLogger],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    try:
        hparams['run_id'] = trainer.logger.experiment[0].id
    # send hparams to all loggers
        trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
        trainer.logger.log_hyperparams = empty
    except:

        print("Logger does not exist")


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.WandbLogger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


def summarize(eval_dict, prefix): # removes list entries from dictionary for faster logging
    # for set in list(eval_dict) : 
    eval_dict_new = {}
    for key in list(eval_dict) :
        #if type(eval_dict[key]) is not list :
        eval_dict_new[prefix + '/' + key] = eval_dict[key]
    return eval_dict_new

def get_yaml(path): # read yaml 
    with open(path, "r") as stream:
        try:
            file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return file


def get_checkpoints_db(cfg,model_name,current_fold): 
    checkpoint_path = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/"
    all_checkpoints = os.listdir("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/checkpoints/exptname_" + model_name)
    print(all_checkpoints)
    checkpoints = {f'fold-{current_fold}':[]} 
    matching_checkpoints = [c for c in all_checkpoints if "best" in c]
    matching_checkpoints = [c for c in matching_checkpoints if f"fold-{current_fold}" in c]
    print("Matching",matching_checkpoints)
    #matching_checkpoints.sort(key = lambda x: x.split('fold-')[1][0:1])
    for fold in checkpoints:
        for cp in matching_checkpoints:
            checkpoints[fold].append(checkpoint_path + '/checkpoints/exptname_' + model_name + "/" + cp)

    return checkpoints
        
    

def get_checkpoint(cfg, path): 
    checkpoint_path = path
    checkpoint_to_load = cfg.get("checkpoint",'last') # default to last.ckpt 
    all_checkpoints = os.listdir(checkpoint_path + '/checkpoints')
    hparams = get_yaml(path+'/csv//hparams.yaml')
    wandbID = hparams['run_id']
    checkpoints = {'fold-1':[]} # dict to store the checkpoints with their path for different folds

    if checkpoint_to_load == 'last':
        matching_checkpoints = [c for c in all_checkpoints if "last" in c]
        matching_checkpoints.sort(key = lambda x: x.split('fold-')[1][0:1])
        for fold, cp_name in enumerate(matching_checkpoints):
            checkpoints[f'fold-{fold+1}'] = checkpoint_path + '/checkpoints/' + cp_name
    elif 'best' in checkpoint_to_load :
        matching_checkpoints = [c for c in all_checkpoints if "last" not in c]
        matching_checkpoints.sort(key = lambda x: x.split('loss-')[1][0:4]) # sort by loss value -> increasing
        for fold in checkpoints:
            for cp in matching_checkpoints:
                checkpoints[fold].append(checkpoint_path + '/checkpoints/' + cp)
            if not 'best_k' in checkpoint_to_load: # best_k loads the k best checkpoints 
                checkpoints[fold] = checkpoints[fold][0] # get only the best (first) checkpoint of that fold
    return wandbID, checkpoints

"""
#Used to plot bar plots and generate csv that stores the mean and std deviation of cross-fold experiments
def generate_hyperparameter_graph(dictionary, plot_save_path, current_fold,csv_path): 
    
    df = pd.DataFrame.from_dict(dictionary)
    #Plot L1  
    sns_plot = sns.histplot(
        df, x="name", y="l1", hue="condition", legend=False
    )

    sns_plot.figure.savefig(plot_save_path + "histogram_l1.png")

    plt.clf() 

"""

def save_results_autoencoder(log_dict,save_dir,exp_name="Dummy",current_fold=1,cfg=None):

    create_dir(save_dir + f"/{exp_name}/fold_{current_fold}") 
    if cfg is not None:
        #Noise standard deviation of denoising autoencoder 
        noise_std = cfg.datamodule.cfg.noise_std
    for key in log_dict.keys():
        
            
        data_metric = {}
        data_label = {}
        data_roc_curve_comb = {}
        data_roc_curve_l1 = {}
        data_roc_curve_l2 = {}
        data_prc_curve_comb = {}
        data_prc_curve_l1 = {}
        data_prc_curve_l2 = {}

        for second_key in  log_dict[key].keys(): 

            

            if 'AUC' in str(second_key):
                data_metric[str(second_key)] =  [log_dict[key][str(second_key)]]
                data_metric['ExptName'] = [exp_name]

            elif 'AUPRC' in str(second_key):
                data_metric[str(second_key)] =  [log_dict[key][str(second_key)]]

            elif 'Precision' in str(second_key):
                data_metric[str(second_key)] =  [log_dict[key][str(second_key)]]

            elif 'Recall' in str(second_key):
                data_metric[str(second_key)] =  [log_dict[key][str(second_key)]]

            elif 'F1' in str(second_key):
                data_metric[str(second_key)] =  [log_dict[key][str(second_key)]]

            elif 'Specificity' in str(second_key):
                data_metric[str(second_key)] =  [log_dict[key][str(second_key)]]

            elif 'Accuracy' in str(second_key):
                data_metric[str(second_key)] =  [log_dict[key][str(second_key)]]

            elif 'crop_size' in str(second_key):
                data_metric[str(second_key)] =  [log_dict[key][str(second_key)][0]]
                data_label[str(second_key)] =  log_dict[key][str(second_key)]

                if cfg is not None: 
                    data_metric["noise_std"] =  [noise_std]
                    data_label["noise_std"] =  [noise_std] * len(log_dict[key][str(second_key)])
        
            elif 'Prediction' in str(second_key):
                data_label[str(second_key)] =  log_dict[key][str(second_key)]

            elif 'disease_to_patient_id' in str(second_key):
                data_label[str(second_key)] =  log_dict[key][str(second_key)]
            
            elif 'smax' in str(second_key):
                data_label[str(second_key)] =  log_dict[key][str(second_key)] 
            
            elif 'label' in str(second_key):
                data_label[str(second_key)] =  log_dict[key][str(second_key)]

            elif 'ReconError_Comb' in str(second_key):
                data_label[str(second_key)] =  log_dict[key][str(second_key)]

            elif 'ReconError_L1' in str(second_key):
                data_label[str(second_key)] =  log_dict[key][str(second_key)]
            
            elif 'ReconError_L2' in str(second_key):

                data_label[str(second_key)] =  log_dict[key][str(second_key)]
                data_label['ExptName'] =   [exp_name for x in range(len(log_dict[key][str(second_key)]))]

            elif 'FPR_Comb' in str(second_key):
                data_roc_curve_comb[str(second_key)] =  log_dict[key][str(second_key)]
            
            elif 'TPR_Comb' in str(second_key):
                data_roc_curve_comb[str(second_key)] =  log_dict[key][str(second_key)]
                data_roc_curve_comb['crop_size'] =  [log_dict[key]['crop_size'][0]]*len(log_dict[key][str(second_key)])
                data_roc_curve_comb['Fold'] =  [current_fold]*len(log_dict[key][str(second_key)])
                data_roc_curve_comb['ExptName'] =   [exp_name for x in range(len(log_dict[key][str(second_key)]))]
                if cfg is not None: 
                  
                    data_roc_curve_comb["noise_std"] =  [noise_std] *len(log_dict[key][str(second_key)])

            elif 'FPR_L1' in str(second_key):
                data_roc_curve_l1[str(second_key)] =  log_dict[key][str(second_key)]
            
            elif 'TPR_L1' in str(second_key):
                data_roc_curve_l1[str(second_key)] =  log_dict[key][str(second_key)]
                data_roc_curve_l1['crop_size'] =  [log_dict[key]['crop_size'][0]]*len(log_dict[key][str(second_key)])
                data_roc_curve_l1['Fold'] =  [current_fold]*len(log_dict[key][str(second_key)])
                data_roc_curve_l1['ExptName'] =  [exp_name for x in range(len(log_dict[key][str(second_key)]))]

                if cfg is not None: 
                  
                    data_roc_curve_l1["noise_std"] =  [noise_std] *len(log_dict[key][str(second_key)])

            
            elif 'FPR_L2' in str(second_key):
                data_roc_curve_l2[str(second_key)] =  log_dict[key][str(second_key)]
            
            elif 'TPR_L2' in str(second_key):
                data_roc_curve_l2[str(second_key)] =  log_dict[key][str(second_key)]
                data_roc_curve_l2['crop_size'] =  [log_dict[key]['crop_size'][0]]*len(log_dict[key][str(second_key)])
                data_roc_curve_l2['Fold'] =  [current_fold]*len(log_dict[key][str(second_key)])
                data_roc_curve_l2['ExptName'] =  [exp_name for x in range(len(log_dict[key][str(second_key)]))]

                if cfg is not None: 
                  
                    data_roc_curve_l2["noise_std"] =  [noise_std] *len(log_dict[key][str(second_key)])


            

            elif 'Prec_PRCurve_Comb' in str(second_key):

                data_prc_curve_comb[str(second_key)] =  log_dict[key][str(second_key)]
            
            elif 'Rec_PRCurve_Comb' in str(second_key):
                data_prc_curve_comb[str(second_key)] =  log_dict[key][str(second_key)]
                data_prc_curve_comb['crop_size'] =  [log_dict[key]['crop_size'][0]]*len(log_dict[key][str(second_key)])
                data_prc_curve_comb['Fold'] =  [current_fold]*len(log_dict[key][str(second_key)])
                data_prc_curve_comb['ExptName'] =   [exp_name for x in range(len(log_dict[key][str(second_key)]))]

                if cfg is not None: 
                  
                    data_prc_curve_comb["noise_std"] =  [noise_std] *len(log_dict[key][str(second_key)])
            
            elif 'Prec_PRCurve_L1' in str(second_key):
                data_prc_curve_l1[str(second_key)] =  log_dict[key][str(second_key)]
            
            elif 'Rec_PRCurve_L1' in str(second_key):
                data_prc_curve_l1[str(second_key)] =  log_dict[key][str(second_key)]
                data_prc_curve_l1['crop_size'] =  [log_dict[key]['crop_size'][0]]*len(log_dict[key][str(second_key)])
                data_prc_curve_l1['Fold'] =  [current_fold]*len(log_dict[key][str(second_key)])
                data_prc_curve_l1['ExptName'] =   [exp_name for x in range(len(log_dict[key][str(second_key)]))]
                if cfg is not None: 
                  
                    data_prc_curve_l1["noise_std"] =  [noise_std] *len(log_dict[key][str(second_key)])

            elif 'Prec_PRCurve_L2' in str(second_key):
                data_prc_curve_l2[str(second_key)] =  log_dict[key][str(second_key)]
            
            elif 'Rec_PRCurve_L2' in str(second_key):
                data_prc_curve_l2[str(second_key)] =  log_dict[key][str(second_key)]
                data_prc_curve_l2['crop_size'] =  [log_dict[key]['crop_size'][0]]*len(log_dict[key][str(second_key)])
                data_prc_curve_l2['Fold'] =  [current_fold]*len(log_dict[key][str(second_key)])
                data_prc_curve_l2['ExptName'] =   [exp_name for x in range(len(log_dict[key][str(second_key)]))]

                if cfg is not None: 
                  
                    data_prc_curve_l2["noise_std"] =  [noise_std] *len(log_dict[key][str(second_key)])

        """
        for k in data_roc_curve_l2.keys():

            print("data_roc_curve_l2",k,len(data_roc_curve_l2[k]))

        """

        

        df = pd.DataFrame.from_dict(data_metric)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{key}_metric.csv" )

        df = pd.DataFrame.from_dict(data_label)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{key}_prediction.csv")

        df = pd.DataFrame.from_dict(data_prc_curve_comb)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{key}_prcurve_l1_l2.csv")

        df = pd.DataFrame.from_dict(data_prc_curve_l1)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{key}_prcurve_l1.csv")

        df = pd.DataFrame.from_dict(data_prc_curve_l2)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{key}_prcurve_l2.csv")

        df = pd.DataFrame.from_dict(data_roc_curve_comb)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{key}_roccurve_l1_l2.csv")

        df = pd.DataFrame.from_dict(data_roc_curve_l1)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{key}_roccurve_l1.csv")

        df = pd.DataFrame.from_dict(data_roc_curve_l2)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{key}_roccurve_l2.csv")



def save_results(log_dict,save_dir,exp_name="Dummy",current_fold=1,cfg=None,class_dir=''):

    create_dir(save_dir + f"/{exp_name}/fold_{current_fold}/{class_dir}/") 



    for key in log_dict.keys():
        
        if key == "test_without_thresh_calc" or key == "val_without_thresh_calc":
            
            data_metric = {

                
                'AUC': [log_dict[key]['AUC']],
                'AUPRC': [log_dict[key]['AUPRC']],
            
                'AUC_one_instance': [log_dict[key]['AUC_one_instance']],
                'AUPRC_one_instance': [log_dict[key]['AUPRC_one_instance']],
                
                #ROC Curve metrics 
                'Precision_50percent_one_instance': [log_dict[key]['Precision_50percent_one_instance']],
                'Precision_50percent': [log_dict[key]['Precision_50percent']],
                
                'Recall_50percent': [log_dict[key]['Recall_50percent']],
                'Recall_50percent_one_instance': [log_dict[key]['Recall_50percent_one_instance']],

                'F1_50percent': [log_dict[key]['F1_50percent']],
                'F1_50percent_one_instance': [log_dict[key]['F1_50percent_one_instance']],

                'Specificity_50percent': [log_dict[key]['Specificity_50percent']],

                'Specificity_50percent_one_instance': [log_dict[key]['Specificity_50percent_one_instance']],
                
                'Accuracy_50percent': [log_dict[key]['Accuracy_50percent']],
                'Accuracy_50percent_one_instance': [log_dict[key]['Accuracy_50percent_one_instance']],
                'crop_size':  [log_dict[key]['crop_size'][0]],
                'Fold': [current_fold],

                'ExptName': [exp_name],

                }
            
            

            data_label = {

                

                #ROC Curve metrics 
                'Prediction_50percent': log_dict[key]['Prediction_50percent'],
                'Prediction_50percent_one_instance': log_dict[key]['Prediction_50percent_one_instance'],
                'Disease_to_patient_id': log_dict[key]['disease_to_patient_id'],
                'label': log_dict[key]['labels'],
                'disease_label': log_dict[key]['disease_labels'],
                'smax': log_dict[key]['smax'],
                'confidence': log_dict[key]['confidence'],
                
                'confidence_one_instance': log_dict[key]['confidence_one_instance'],
                'confidence_std': log_dict[key]['confidence_std'],
                'thresh_50percent': log_dict[key]['50percent'],
                'crop_size': log_dict[key]['crop_size'],

                'Fold': [current_fold]*len(log_dict[key]['labels']),
                'ExptName': [exp_name]*len(log_dict[key]['labels']),

                }

        else:

            data_metric = {

                

                #ROC Curve metrics 
                'AUC': [log_dict[key]['AUC']],
                'AUPRC': [log_dict[key]['AUPRC']],
                

                #ROC Curve metrics 
                'Precision_thresh_1p_roc': [log_dict[key]['Precision_thresh_1p_roc']],
                'Precision_thresh_5p_roc': [log_dict[key]['Precision_thresh_5p_roc']],
                'Precision_thresh_10p_roc': [log_dict[key]['Precision_thresh_10p_roc']],

                'Recall_thresh_1p_roc': [log_dict[key]['Recall_thresh_1p_roc']],
                'Recall_thresh_5p_roc': [log_dict[key]['Recall_thresh_5p_roc']],
                'Recall_thresh_10p_roc': [log_dict[key]['Recall_thresh_10p_roc']],

                'F1_thresh_1p_roc': [log_dict[key]['F1_thresh_1p_roc']],
                'F1_thresh_5p_roc': [log_dict[key]['F1_thresh_5p_roc']],
                'F1_thresh_10p_roc': [log_dict[key]['F1_thresh_10p_roc']],

                'Specificity_thresh_1p_roc': [log_dict[key]['Specificity_thresh_1p_roc']],
                'Specificity_thresh_5p_roc': [log_dict[key]['Specificity_thresh_5p_roc']],
                'Specificity_thresh_10p_roc': [log_dict[key]['Specificity_thresh_10p_roc']],

                'Accuracy_thresh_1p_roc': [log_dict[key]['Accuracy_thresh_1p_roc']],
                'Accuracy_thresh_5p_roc': [log_dict[key]['Accuracy_thresh_5p_roc']],
                'Accuracy_thresh_10p_roc': [log_dict[key]['Accuracy_thresh_10p_roc']],


                #PRCurve metrics
                'Precision_thresh_1p_prc': [log_dict[key]['Precision_thresh_1p_prc']],
                'Precision_thresh_5p_prc': [log_dict[key]['Precision_thresh_5p_prc']],
                'Precision_thresh_10p_prc': [log_dict[key]['Precision_thresh_10p_prc']],

                'Recall_thresh_1p_prc': [log_dict[key]['Recall_thresh_1p_prc']],
                'Recall_thresh_5p_prc': [log_dict[key]['Recall_thresh_5p_prc']],
                'Recall_thresh_10p_prc': [log_dict[key]['Recall_thresh_10p_prc']],

                'F1_thresh_1p_prc': [log_dict[key]['F1_thresh_1p_prc']],
                'F1_thresh_5p_prc': [log_dict[key]['F1_thresh_5p_prc']],
                'F1_thresh_10p_prc': [log_dict[key]['F1_thresh_10p_prc']],

                'Specificity_thresh_1p_prc': [log_dict[key]['Specificity_thresh_1p_prc']],
                'Specificity_thresh_5p_prc': [log_dict[key]['Specificity_thresh_5p_prc']],
                'Specificity_thresh_10p_prc': [log_dict[key]['Specificity_thresh_10p_prc']],

                'Accuracy_thresh_1p_prc': [log_dict[key]['Accuracy_thresh_1p_prc']],
                'Accuracy_thresh_5p_prc': [log_dict[key]['Accuracy_thresh_5p_prc']],
                'Accuracy_thresh_10p_prc': [log_dict[key]['Accuracy_thresh_10p_prc']],

                'crop_size': [log_dict[key]['crop_size'][0]],
                'Fold': [current_fold],
                'ExptName': [exp_name]


                }

            data_label = {
                
                #ROC Curve metrics 
                'Prediction_thresh_1p_roc': log_dict[key]['Prediction_thresh_1p_roc'],
                'Prediction_thresh_5p_roc': log_dict[key]['Prediction_thresh_5p_roc'],
                'Prediction_thresh_10p_roc': log_dict[key]['Prediction_thresh_10p_roc'],

                'Prediction_thresh_1p_prc': log_dict[key]['Prediction_thresh_1p_prc'],
                'Prediction_thresh_5p_prc': log_dict[key]['Prediction_thresh_5p_prc'],
                'Prediction_thresh_10p_prc': log_dict[key]['Prediction_thresh_10p_prc'],

                'confidence': log_dict[key]['confidence'],
                'confidence_std': log_dict[key]['confidence_std'],

                '1p_roc': log_dict[key]['thresh_1p_roc'],
                '5p_roc': log_dict[key]['thresh_5p_roc'],
                '10p_roc': log_dict[key]['thresh_10p_roc'],
                '1p_prc': log_dict[key]['thresh_1p_prc'],
                '5p_prc': log_dict[key]['thresh_5p_prc'],
                '10p_prc': log_dict[key]['thresh_10p_prc'],

                'Disease_to_patient_id': log_dict[key]['disease_to_patient_id'],
                'label': log_dict[key]['labels'],
                'disease_label': log_dict[key]['disease_labels'],
                'crop_size': log_dict[key]['crop_size'],
                'smax': log_dict[key]['smax'],

                'Fold': [current_fold]*len(log_dict[key]['labels']),
                'ExptName': [exp_name]*len(log_dict[key]['labels']),
                

                }


        data_roc_curve = {
                
                #ROC Curve metrics 
                'FPR': log_dict[key]['FPR'],
                'TPR': log_dict[key]['TPR'],
                'crop_size': [log_dict[key]['crop_size'][0]]*len(log_dict[key]['TPR']),
                'Fold': [current_fold]*len(log_dict[key]['TPR']),
                'ExptName': [exp_name]*len(log_dict[key]['TPR'])
                
            }

        data_prc_curve = {
                
                #ROC Curve metrics 
                'Prec_PRCurve': log_dict[key]['Prec_PRCurve'],
                'Rec_PRCurve': log_dict[key]['Rec_PRCurve'],
                'crop_size': [log_dict[key]['crop_size'][0]]*len(log_dict[key]['Rec_PRCurve']),
                'Fold': [current_fold]*len(log_dict[key]['Rec_PRCurve']),
                'ExptName': [exp_name]*len(log_dict[key]['Rec_PRCurve'])
                
            }
        

    

        df = pd.DataFrame.from_dict(data_metric)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{class_dir}/{key}_metric.csv" )

        df = pd.DataFrame.from_dict(data_label)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{class_dir}/{key}_prediction.csv")

        df = pd.DataFrame.from_dict(data_prc_curve)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{class_dir}/{key}_prcurve.csv")


        df = pd.DataFrame.from_dict(data_roc_curve)  
        df.to_csv(save_dir + f"/{exp_name}/fold_{current_fold}/{class_dir}/{key}_roccurve.csv")



def generate_graphs(pred_dict,log_dict,sets,plot_save_path,current_fold,exp_name,csv_path,csv_path_metric,csv_path_prediction, stage="test"):
    """
    This function is used to render all the graphs that are required for analysis per fold
    """

    create_dir(plot_save_path)
    for set in sets:

        dictionary = pred_dict[stage][set]
        """
        #latent spaces
        latent_space_healthy   = dictionary["latentSpaceHealthy"]
        latent_space_unhealthy = dictionary["latentSpaceUnhealthy"]

        #l1 reconstruction error
        
        l1_recon_healthy   = dictionary["l1recoErrorHealthy"]
        l1_recon_healthy_scaled = [int(x * 100) for x in l1_recon_healthy]
        l1_recon_unhealthy = dictionary["l1recoErrorUnhealthy"]
        l1_recon_unhealthy_scaled = [int(x * 100) for x in l1_recon_unhealthy]

        #l2 reconstruction error
        l2_recon_healthy   = dictionary["l2recoErrorHealthy"]
        l2_recon_healthy_scaled = [int(x * 100) for x in l2_recon_healthy]
        l2_recon_unhealthy = dictionary["l2recoErrorUnhealthy"]
        l2_recon_unhealthy_scaled = [int(x * 100) for x in l2_recon_unhealthy]

        fold_current_h  = [current_fold] * len(l1_recon_healthy)
        fold_current_uh = [current_fold] * len(l1_recon_unhealthy)

        

        exp_names_h     = [exp_name] * len(l1_recon_healthy)
        exp_names_uh    = [exp_name] * len(l1_recon_unhealthy)

        
        
        assert len(l1_recon_healthy) == len(l2_recon_healthy) and len(l1_recon_unhealthy) == len(l2_recon_unhealthy) , "The healthy and unhealthy samples do not match"
 
        #Healthy Labels
        healthy_labels       = ["Healthy"]*len(l1_recon_healthy)
        unhealthy_labels     = ["Unhealthy"]*len(l1_recon_unhealthy)

        #Pandas data frame 
        zipped = list(zip(healthy_labels, l1_recon_healthy,l1_recon_healthy_scaled, l2_recon_healthy, l2_recon_healthy_scaled,fold_current_h,exp_names_h))
        df_h = pd.DataFrame(zipped, columns=['Condition', 'L1', 'L1_scaled','L2','L2_scaled','Fold','ExptName'])

        zipped = list(zip(unhealthy_labels, l1_recon_unhealthy,l1_recon_unhealthy_scaled, l2_recon_unhealthy, l2_recon_unhealthy_scaled,fold_current_uh,exp_names_uh))
        df_uh = pd.DataFrame(zipped, columns=['Condition', 'L1', 'L1_scaled','L2','L2_scaled','Fold','ExptName'])

        df = df_h.append(df_uh, ignore_index=True)

        

        #Plot Histogram 
        sns_plot = sns.histplot(
            df, x="L1_scaled", y="Condition", hue="Condition", legend=False
        )

        sns_plot.figure.savefig(plot_save_path + "histogram_l1.png")

        plt.clf() 
        sns_plot = sns.histplot(
            df, x="L2_scaled", y="Condition", hue="Condition", legend=False
        )
        sns_plot.figure.savefig(plot_save_path + "histogram_l2.png")

        plt.clf() 
        #Plot Bar Chart 
        sns_plot = sns.barplot(
             x="L1", y="Condition", data=df, hue="Condition"
        )

        sns_plot.figure.savefig(plot_save_path + "bar_l1.png")
        plt.clf() 
        plt.close()

        sns_plot = sns.barplot(
             x="L2", y="Condition", data=df, hue="Condition"
        )

        sns_plot.figure.savefig(plot_save_path + "bar_l2.png")

        plt.clf() 
        plt.close()
        #Save results of this experiment to master csv 

        if os.path.isfile(csv_path): 
            #Append new experiment results to existing master csv
            df_master = pd.read_csv(csv_path) 
            df_master = df_master.append(df, ignore_index=True)
            #Overwrite
            df_master.to_csv(csv_path)

        else: 
            #Save new pandas dataframe to csv
            df.to_csv(csv_path)  
        """

        #Save Metrics
        data_metric = {

            'Fold': [current_fold], 
            #L1 
            'Precision_PRC_Comb': [log_dict[stage][set]['Precision_thresh_prc_comb']],
            'Precision_PRC_Reg': [log_dict[stage][set]['Precision_thresh_prc_reg']],
            'Precision_PRC_Reco': [log_dict[stage][set]['Precision_thresh_prc_reco']],

            'Recall_PRC_Comb': [log_dict[stage][set]['Recall_thresh_prc_comb']],
            'Recall_PRC_Reg': [log_dict[stage][set]['Recall_thresh_prc_reg']],
            'Recall_PRC_Reco': [log_dict[stage][set]['Recall_thresh_prc_reco']],


            'F1_PRC_Comb': [log_dict[stage][set]['F1_thresh_prc_comb']],
            'F1_PRC_Reg': [log_dict[stage][set]['F1_thresh_prc_reg']],
            'F1_PRC_Reco': [log_dict[stage][set]['F1_thresh_prc_reco']],

            'Accuracy_PRC_Comb': [log_dict[stage][set]['Accuracy_thresh_prc_comb']],
            'Accuracy_PRC_Reg': [log_dict[stage][set]['Accuracy_thresh_prc_reg']],
            'Accuracy_PRC_Reco': [log_dict[stage][set]['Accuracy_thresh_prc_reco']],

            'AUC_PRC_Comb': [log_dict[stage][set]['AUPRCperVolComb']],
            'AUC_PRC_Reg': [log_dict[stage][set]['AUPRCperVolReg']],
            'AUC_PRC_Reco': [log_dict[stage][set]['AUPRCperVolReco']],

            'AUC_ROC_Comb': [log_dict[stage][set]['AUCperVolComb']],
            'AUC_ROC_Reg': [log_dict[stage][set]['AUCperVolReg']],
            'AUC_ROC_Reco': [log_dict[stage][set]['AUCperVolReco']],
            

            'Precision_PRC_Comb_L2': [log_dict[stage][set]['Precision_thresh_L2_cmb_']],
            
            'Precision_PRC_Reco_L2': [log_dict[stage][set]['Precision_thresh_L2_rcns']],

            'Recall_PRC_Comb_L2': [log_dict[stage][set]['Recall_thresh_L2_cmb_']],
            
            'Recall_PRC_Reco_L2': [log_dict[stage][set]['Recall_thresh_L2_rcns']],

            'F1_PRC_Comb_L2': [log_dict[stage][set]['F1_thresh_L2_cmb_']],
            'F1_PRC_Reco_L2': [log_dict[stage][set]['F1_thresh_L2_rcns']],

            'Accuracy_PRC_Comb_L2': [log_dict[stage][set]['Accuracy_thresh_L2_cmb_']],
            'Accuracy_PRC_Reco_L2': [log_dict[stage][set]['Accuracy_thresh_L2_rcns']],

            'AUC_ROC_CombL2': [log_dict[stage][set]['AUCperVolCombL2']],
            'AUC_ROC_RecoL2': [log_dict[stage][set]['AUCperVolRecoL2']],
            'AUC_PRC_CombL2': [log_dict[stage][set]['AUPRCperVolCombL2']],
            'AUC_PRC_RecoL2': [log_dict[stage][set]['AUPRCperVolRecoL2']],

            'ExptName': [exp_name],

            }
        
        #Creating pandas dataframe 
        df_metric =  pd.DataFrame.from_dict(data_metric)
        print(df_metric)
        print(csv_path_metric)
        #Save results of this experiment to master csv 

        if os.path.isfile(csv_path_metric): 

            #Append new experiment results to existing master csv
            df_m = pd.read_csv(csv_path_metric) 
            df_m = df_m.append(df_metric, ignore_index=True)
            #Overwrite
            df_m.to_csv(csv_path_metric)

        else: 

            #Save new pandas dataframe to csv
            df_metric.to_csv(csv_path_metric) 

        #Save Per Volume Predictions
        data_pred = {}

        for key in list(log_dict[stage][set].keys()):
            if "Prediction" in key: 

                data_pred[key]  = log_dict[stage][set][key]

        
        data_pred["disease_to_patient_id"]   = log_dict[stage][set]["disease_to_patient_id"]
        data_pred["labels"]   = log_dict[stage][set]["labels"]
        data_pred["ExpName"]  = [exp_name]*len(log_dict[stage][set]["labels"])
        data_pred["Fold"]     = [current_fold]*len(log_dict[stage][set]["labels"])
        #Creating pandas dataframe 
        df_pred =  pd.DataFrame.from_dict(data_pred)
        if os.path.isfile(csv_path_prediction): 

            #Append new experiment results to existing master csv
            df_p = pd.read_csv(csv_path_prediction) 
            df_p = df_p.append(df_pred, ignore_index=True)
            #Overwrite
            df_p.to_csv(csv_path_prediction)

        else: 

            #Save new pandas dataframe to csv
            df_pred.to_csv(csv_path_prediction)


def draw_multiple_umaps(dictionary,sets,plot_save_path, stage='test'):

    create_dir(plot_save_path)

    for set in sets: 
        latent_vectors  = dictionary[stage][set]["latentSpaceAll"]
        labels          = np.array(dictionary[stage][set]["labelPerVol"])


        #Normalise 

        lv_tensor = torch.tensor(latent_vectors)
        
        latent_vectors_norm = np.squeeze(F.normalize(lv_tensor.cuda()).cpu().numpy())
        #latent_vectors_norm  = np.squeeze(lv_tensor.numpy())
        #5, 10, 20, 50, 100
        for n in ( 15,30):
            #0.0, 0.1, 0.25, 0.5, 0.8, 0.99
            for d in (0.1,0.25):
                #["euclidean","cosine","mahalanobis","correlation"]
                for m in ["cosine","mahalanobis"]:

                    print("Generating UMAP for n_neighbors = {} min_dist = {} metric = {}".format(n,d,m))
                    draw_umap(latent_vectors_norm,labels, plot_save_path, n_neighbors=n,min_dist=d,metric=m, title="n_neighbors = {} min_dist = {} metric = {}".format(n,d,m))

            

def draw_umap(data, labels,plot_save_path, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )

    u = fit.fit_transform(data)

    fig = plt.figure()
    plt.clf()

    if n_components == 1:
        
        scatter = plt.scatter(u[:,0], range(len(u)), c=labels)
    if n_components == 2:
        #ax = fig.add_subplot(111)
        scatter = plt.scatter(u[:,0], u[:,1], c=labels)
    if n_components == 3:
        #ax = fig.add_subplot(111, projection='3d')
        scatter = plt.scatter(u[:,0], u[:,1], u[:,2], c=labels, s=100)
    classes = ["Healthy","Unhealthy"]
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    #markers = {"Lunch": "s", "Dinner": "X"}
    #plt.title(title, fontsize=18)
    plt.savefig(plot_save_path + title + ".png",dpi=500)
    plt.close()





















        





