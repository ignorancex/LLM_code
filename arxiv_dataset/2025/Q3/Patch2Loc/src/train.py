from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from omegaconf import DictConfig, OmegaConf, open_dict
from lightning.pytorch.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig
from typing import List, Optional
import wandb 
import os
import warnings
import torch
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.strategies import DDPStrategy
import pickle

from src.utils import utils
os.environ['NUMEXPR_MAX_THREADS'] = '16'

log = utils.get_logger(__name__) # init logger

@hydra.main(config_path='configs', config_name='config') # Hydra decorator
def train(cfg: DictConfig) -> Optional[float]: 
    results = {}
    # base names for logging
    base = cfg.callbacks.model_checkpoint.monitor # naming of logs
    if 'early_stop' in cfg.callbacks:
        base_es = cfg.callbacks.early_stop.monitor # early stop base metric

    # load checkpoint if specified
    if cfg.get('load_checkpoint'): # load stored checkpoint for testing or resuming training
        wandbID, checkpoints = utils.get_checkpoint(cfg.get('load_checkpoint'), cfg.get("checkpoint",'last'), cfg.get('num_folds',1)) # outputs a Dictionary of checkpoints and the corresponding wandb ID to resume the run 
        if cfg.get('new_wandb_run',False): # If we want to onlyEvaluate a run in a new wandb run
            cfg.logger.wandb.id = wandb.util.generate_id()
        else:
            if cfg.get('resume_wandb',True):
                log.info(f"Resuming wandb run")
                if wandbID is not None:
                    cfg.logger.wandb.resume = wandbID # this will allow resuming the wandb Run 

    cfg.logger.wandb.group = cfg.name  # specify group name in wandb 

    if cfg.trainer.get('accelerator', None) == 'ddp':  # for better performance in ddp mode
        cfg.trainer.accelerator = 'gpu'
        plugs = DDPStrategy(find_unused_parameters=False)
    else:
        plugs = 'auto'


    if "seed" in cfg: # for deterministic training (covers pytorch, numpy and python.random)
        log.info(f"Seed specified to {cfg.seed} by config")
        seed_everything(cfg.seed, workers=True)

    # get start and end fold
    start_fold = cfg.get('start_fold',0)
    end_fold = cfg.get('num_folds',5)
    if start_fold != 0:
        log.info(f'skipping the first {start_fold} fold(s)') 

    # iterate over folds from start_fold to num_fold
    for fold in range(start_fold,end_fold): # iterate over folds 
        
        log.info(f"Training Fold {fold+1} of {end_fold} in the WandB group {cfg.logger.wandb.group}")
        prefix = f'{fold+1}/' # naming of logs


        cfg.datamodule._target_ = f'src.datamodules.Datamodules_train.{cfg.datamodule.cfg.name}' # set datamodule target
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>") 
        datamodule_train: LightningDataModule = hydra.utils.instantiate(cfg.datamodule,fold=fold) # instantiate datamodule

        # Init lightning model
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model,prefix=prefix) # instantiate model

        # setup callbacks
        cfg.callbacks.model_checkpoint.monitor = f'{prefix}' + base # naming of logs for cross validation
        cfg.callbacks.model_checkpoint.filename = "epoch-{epoch}_step-{step}_loss-{"+f"{prefix}"+"val/loss:.2f}" # naming of logs for cross validation

        if 'early_stop' in cfg.callbacks:
            cfg.callbacks.early_stop.monitor = f'{prefix}' + base_es # naming of logs for cross validation

        if 'log_image_predictions' in cfg.callbacks:
            cfg.callbacks.log_image_predictions.prefix = prefix # naming of logs for cross validation
        
        # init callbacks
        callbacks: List[Callback] = []
        if "callbacks" in cfg:
            for _, cb_conf in cfg.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))
            callbacks[0].FILE_EXTENSION = f'_fold-{fold+1}.ckpt' # naming of logs for cross validation callbacks[0] is the model checkpoint callback (this is a hacky way to do this)

        # Init lightning loggers
        logger: List[Logger] = []
        if "logger" in cfg:
            for _, lg_conf in cfg.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))


        target_checkpoint = None
        # Load checkpoint if specified
        if cfg.get('load_checkpoint'): # pass checkpoint to resume from
            with open_dict(cfg):
                target_checkpoint= checkpoints[f"fold-{fold+1}"]
                cfg.ckpt_path=None
            log.info(f"Restoring Trainer State of loaded checkpoint: ",target_checkpoint)

        # Init lightning trainer
        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial", strategy=plugs
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


        if (not cfg.get('onlyEval',False)) : # train model
            trainer.fit(model, datamodule_train, ckpt_path=target_checkpoint)
            validation_metrics = trainer.callback_metrics
        else: # load trained model
            print(checkpoints[f'fold-{fold+1}'])
            model.load_state_dict(torch.load(checkpoints[f'fold-{fold+1}'])['state_dict'])

        # logging
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
        log.info(f"Best checkpoint metric:\n{trainer.checkpoint_callback.best_model_score}")
        trainer.logger.experiment.log({'best_ckpt_path':trainer.checkpoint_callback.best_model_path})
        trainer.logger.experiment.log({'logdir':trainer.log_dir})

        # metrics
        #do we really need this part of the code? it causes an exception when running multiple folds: 4/ad_norm_0 keyerror at the last line. 
        # validation_metrics = trainer.callback_metrics
        # for key in validation_metrics:
        #     key =  key[2:]
        #     valkey= prefix + key
        #     if not 'train' in key and not 'test' in key:
        #         if key not in results:
        #             results[key] = []
        #         results[key].append(validation_metrics[valkey])


    # Evaluate model on test set, using the best or last model from each trained fold 

        if cfg.get("test_after_training"): # and not 'simclr' in  cfg.model._target_.lower():
            log.info(f"Starting evaluation phase of fold {fold+1}!")
            preds_dict = {}
            preds_dict = {'val':{},'test':{}} # a dict for each data set
            
            sets = {
                    't2':['Datamodules_eval.Brats21','Datamodules_eval.MSLUB','Datamodules_train.IXI'],
                    't1':['Datamodules_eval.Brats21','Datamodules_eval.MSLUB','Datamodules_train.IXI','Datamodules_eval.ATLAS_v2','Datamodules_eval.WMH'],
                   }
            
                
            for set in cfg.datamodule.cfg.testsets :
                if not set in sets[cfg.datamodule.cfg.mode]: # skip testsets of different modalities
                    continue    

                cfg.datamodule._target_ = 'src.datamodules.{}'.format(set)
                log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
                datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, fold=fold)
                datamodule.setup()

                # Validation steps
                log.info("Validation of {}!".format(set))

                if 'train' in set:
                    trainer.test(model=model,dataloaders=datamodule.val_eval_dataloader(),ckpt_path=target_checkpoint)
                    # evaluation results
                    preds_dict['val'][set] = trainer.lightning_module.eval_dict
                    log_dict = utils.summarize(preds_dict['val'][set],'val') # sets prefix val/ and removes lists for better logging in wandb
                else:
                    log_dict = None
                # Test steps
                log.info("Test of {}!".format(set))
                if 'train' in set:
                    trainer.test(model=model,dataloaders=datamodule.test_eval_dataloader(),ckpt_path=target_checkpoint)
                else: 
                    trainer.test(model=model,dataloaders=datamodule.test_dataloader(),ckpt_path=target_checkpoint)

                # log to wandb
                preds_dict['test'][set] = trainer.lightning_module.eval_dict
                if log_dict is not None:
                    log_dict.update(utils.summarize(preds_dict['test'][set],'test')) # sets prefix test/ and removes lists for better logging in wandb
                else:
                    log_dict = utils.summarize(preds_dict['test'][set], 'test')
                log_dict = utils.summarize(log_dict,f'{fold+1}/'+set) # sets prefix for each data set
                trainer.logger.experiment.log(log_dict)

                

            # pickle preds_dict for later analysis
            if cfg.get('pickle_preds',True):
                with open(os.path.join(trainer.log_dir,f'{fold+1}_preds_dict.pkl'),'wb') as f:
                    pickle.dump(preds_dict,f)



    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=cfg,
        model=model,
        datamodule=datamodule_train,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

