
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict
#from pytorch_lightning.plugins import DDPPlugin
import hydra
import logging
from omegaconf import DictConfig
from typing import List, Optional
import wandb 
import os
import warnings
import torch
import pandas
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import copy
from hydra.core.hydra_config import HydraConfig

from utils.utils import generate_graphs, draw_multiple_umaps,save_results
from utils.utils_eval import  redFlagEvaluation_einscanner, calc_thresh, calc_thresh_classification,evaluateModel

from collections.abc import Iterable



os.environ['NUMEXPR_MAX_THREADS'] = '16'
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

# Import modules / files
from utils import utils
#from pytorch_lightning.loggers import LightningLoggerBase

log = utils.get_logger(__name__) # init logger
@hydra.main(config_path='configs', config_name='config') # Hydra decorator
def test(cfg: DictConfig) -> Optional[float]: 
    results = {}

    base = cfg.callbacks.model_checkpoint.monitor # naming of logs
    if 'early_stop' in cfg.callbacks:
        
        base_es = cfg.callbacks.early_stop.monitor # early stop base metric

    if cfg.get('load_checkpoint') : # load stored checkpoint for testing or resuming training
        checkpoints = utils.get_checkpoints_db(cfg, cfg.get('load_checkpoint'),cfg.current_fold) # outputs a Dictionary of checkpoints and the corresponding wandb ID to resume the run 
       
 
    
    cfg.logger.wandb.group = cfg.name  # specify group name in wandb 

    #cfg.logger.wandb.name = f"fold_{cfg.current_fold}"  # specify name of experiment

    # Set plugins for lightning trainer
    if cfg.trainer.get('accelerator',None) == 'ddp': # for better performance in ddp mode
        plugs = pl.plugins.DDPPlugin(find_unused_parameters=False)
    else: 
        plugs = None

    if "seed" in cfg: # for deterministic training (covers pytorch, numpy and python.random)
        log.info(f"Seed specified to {cfg.seed} by config")
        pl.seed_everything(cfg.seed, workers=True)

    
    current_fold = cfg.current_fold
 
    log.info(f"Training Fold {current_fold} of {cfg.get('num_folds',5)} in the WandB group {cfg.logger.wandb.group}")
    prefix = f'{current_fold}/' # naming of logs
    #Set current fold. This is used to find the dataset location
    
    # Init lightning datamodule
    cfg.datamodule._target_ = f'datamodules.data_mod.{cfg.datamodule.cfg.name}'
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule_train: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule_train.prepare_data()
    datamodule_train.setup()
    # Init lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model,prefix=prefix)

    #Initialise lightning convnet model for transformer
    if cfg.get('enable_convnet_model',True): 
        log.info(f"Instantiating additional conv net model for transformer <{cfg.convnet_model._target_}>")
        convnet_model: pl.LightningModule = hydra.utils.instantiate(cfg.convnet_model,prefix=prefix)
        convnet_model.load_state_dict(torch.load(cfg.convnet_model_ckpt_path)['state_dict'])
        model.conv_model = convnet_model
        
        model.create_ViTmodel()
        

    # Init lightning callbacks
    cfg.callbacks.model_checkpoint.monitor = base
    #cfg.callbacks.model_checkpoint.filename = "epoch-{epoch}_step-{step}_loss-{"+f"{prefix}"+"val/loss:.2f}"
    cfg.callbacks.model_checkpoint.filename = f"exptname_{cfg.logger.wandb.name}/"

    # Init lightning callbacks
    
    if 'early_stop' in cfg.callbacks:
        cfg.callbacks.early_stop.monitor = base_es
    
    
    callbacks: List[pl.Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
        
        callbacks[0].FILE_EXTENSION = f'best_fold-{current_fold}.ckpt'

    # Init lightning loggers
    logger = []
    
    if "logger" in cfg and cfg.get("enable_logger",False):
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
    


    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial", plugins=plugs
    )        

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=cfg,
        model=model,
        datamodule=datamodule_train,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )


    # custom K-Fold training loop specifiy num_folds = 1 to only train one (the first) fold
    # trainer.fit_loop = KFoldLoop(cfg.get('num_folds',5), trainer.fit_loop, export_path="./",cfg=cfg, wandblogger=logger[0])
    
    print("LOADING MODEL",checkpoints[f'fold-{cfg.current_fold}'])
    model.load_state_dict(torch.load(checkpoints[f'fold-{cfg.current_fold}'][0])['state_dict'],strict=False)
        # TODO
        #model.z_m=0
        #model.z_std=1
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
    log.info(f"Best checkpoint metric:\n{trainer.checkpoint_callback.best_model_score}")
    validation_metrics = trainer.callback_metrics


    preds_dict = {'val':{},'test':{}} # a dict for each data set
    #Evaluate Validation Set
    trainer.test(model=model,dataloaders=datamodule_train.test_eval_dataloader())
    try: 

        # check if eval_dict is iterable
        if isinstance(trainer.lightning_module.eval_dict, Iterable): 

            print("ITERABLE")

            

        else:

            print("NOT ITERABLE")
            
        
        # evaluation results
        preds_dict['val'] = trainer.lightning_module.eval_dict


    
        #print(preds_dict['val'])
        log_dict = utils.summarize(preds_dict['val'],'val') # sets prefix val/ and removes lists for better logging in wandb
        #Calculate threshold 

        
        thresh = calc_thresh_classification(preds_dict['val'],cfg.model.cfg.plot_save_path) 

        
        # Test steps
        trainer.test(model=model,dataloaders=datamodule_train.test_eval_dataloader())
        
        preds_dict['test'] = trainer.lightning_module.eval_dict
        log_dict.update(utils.summarize(preds_dict['test'],'test')) # sets prefix test/ and removes lists for better logging in wandb
        
        log_dict_redFlag = {'val':{},'val_without_thresh_calc':{},'test':{},'test_without_thresh_calc':{},}
            
        log_dict_redFlag['val'] = evaluateModel(preds_dict['val'].copy(), thresh=thresh) # evaluates the sample-wise detection performance for all data sets
        
        log_dict_redFlag['val_without_thresh_calc'] = evaluateModel(preds_dict['val'].copy(), thresh=None) # evaluates the sample-wise detection performance for all data sets
        
        log_dict_redFlag['test'] = evaluateModel(preds_dict['test'].copy(), thresh=thresh) # evaluates the sample-wise detection performance for all data sets
        
        log_dict_redFlag['test_without_thresh_calc'] = evaluateModel(preds_dict['test'].copy(), thresh=None) # evaluates the sample-wise detection performance for all data sets
        
        save_results(log_dict_redFlag,cfg.results_save_dir,exp_name=HydraConfig.get().job.name,current_fold=current_fold,cfg=cfg)

    except Exception as e:
        print(e)
        print("Error in evaluation")
        pass





    

    
        

        

        




