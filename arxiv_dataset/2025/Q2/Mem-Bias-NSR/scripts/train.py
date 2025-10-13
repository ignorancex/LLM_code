import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from dataclasses import dataclass
from typing import Tuple
from ControllableNesymres.architectures.model import Model
from ControllableNesymres.architectures.data import DataModule, ControllableNesymresDataset
import hydra
from pathlib import Path
from pytorch_lightning import loggers as pl_loggers
import torch
import math
import copy


from lora_pytorch import LoRA

lr_monitor = LearningRateMonitor(logging_interval='step')



class AlternatingDatasetCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        print("next_epoch",current_epoch + 1)
        trainer.datamodule.training_dataset = ControllableNesymresDataset(
            trainer.datamodule.data_train_path,
            trainer.datamodule.cfg.copy(),
            mode="train",
            epoch=current_epoch + 1
        )






@hydra.main(config_name="config")
def main(cfg):


    train_path = Path(hydra.utils.to_absolute_path(cfg.train_path))
    benchmark_path = Path(hydra.utils.to_absolute_path(cfg.benchmark_path))
    data = DataModule(
        train_path,
        benchmark_path,
        cfg
    )

    torch.set_float32_matmul_precision("high")

    cfg.inference.word2id = data.training_dataset.word2id
    cfg.inference.id2word = data.training_dataset.id2word
    cfg.inference.total_variables = data.training_dataset.total_variables


    
    model = Model(cfg=cfg)

    
    
    if cfg.base_model != "None":
        if cfg.base_model == "nsr":
            checkpoint = torch.load(Path(hydra.utils.to_absolute_path('ControllableNeuralSymbolicRegressionWeights/nsr_200000000_epoch=149.ckpt')),map_location=torch.device('cpu'))
        elif cfg.base_model == "nsrwh":
            checkpoint = torch.load(Path(hydra.utils.to_absolute_path('ControllableNeuralSymbolicRegressionWeights/nsrwh_200000000_epoch=149.ckpt')),map_location=torch.device('cpu'))
        elif cfg.base_model == "nopow":
            checkpoint = torch.load(Path(hydra.utils.to_absolute_path('weights/SmallNSR_00000_log_-epoch=999.ckpt')),map_location=torch.device('cpu'))
        state_dict_old = checkpoint['state_dict']
            
        
        state_dict_new = model.state_dict()
        

        

        string1 = "decoder_transfomer.layers."
        string1_len = len(string1)
        for key in state_dict_old:
            
            if key.startswith("decoder_transfomer.layers."):
                num_layer = int(key[string1_len])
                if num_layer < cfg.decoder_top:
                    state_dict_new["decoder_transfomer0.layers." + key[string1_len:]] = state_dict_old[key]
                    #print(key,"decoder_transfomer0.layers." + key[string1_len:])
                elif num_layer > cfg.decoder_bottom:
                    state_dict_new["decoder_transfomer1.layers." + str(num_layer - cfg.decoder_bottom - 1) + key[string1_len + 1:]] = state_dict_old[key]
                else:
                    state_dict_new[string1 + str(num_layer - cfg.decoder_top) + key[string1_len + 1:]] = state_dict_old[key]
                    #print(key,string1 + str(num_layer - cfg.decoder_top) + key[string1_len + 1:])
            
                    
            else:
                state_dict_new[key] = state_dict_old[key]
                #print(key,key)
            
        if cfg.adaptor:
            state_dict_new['adaptor_linear.weight'] = torch.zeros(512,512)
            state_dict_new['adaptor_linear.bias'] = torch.zeros(512)
            
        model.load_state_dict(state_dict_new)

        
        

        
        if cfg.freeze_num_encoder:
            for param in model.enc.parameters():
                param.requires_grad = False

        if cfg.decoder_top > 0:
            for param in model.decoder_transfomer0.parameters():
                param.requires_grad = False

        if cfg.decoder_bottom < 4:
            for param in model.decoder_transfomer1.parameters():
                param.requires_grad = False
        
        if cfg.freeze_decoder:
            for param in model.decoder_transfomer.parameters():
                param.requires_grad = False

        
        if cfg.lora_train:
            for param in model.decoder_transfomer.parameters():
                param.requires_grad = False
            model.decoder_transfomer = LoRA.from_module(model.decoder_transfomer, rank=cfg.lora_r_train)
            
        

    



    data.setup() # Ugly hack in order to create the mapper
    data.val_dataloader()
    model.mapper = data.mapper 
    model.metadata = data.training_dataset.metadata
    model.cfg.inference.id2word = data.training_dataset.id2word

    if cfg.resume_from_checkpoint:
        candidate_path = Path(hydra.utils.to_absolute_path(cfg.resume_from_checkpoint))
        # Check if the path is a file or a directory
        if candidate_path.is_file():
            is_folder = False
            checkpoint_dir_path = Path(hydra.utils.to_absolute_path(cfg.resume_from_checkpoint)).parent
            logs_save_dir_path = Path(hydra.utils.to_absolute_path(cfg.resume_from_checkpoint)).parent.parent / "logs_dir"
        elif candidate_path.is_dir():
            is_folder = True
            checkpoint_dir_path = Path(hydra.utils.to_absolute_path(cfg.resume_from_checkpoint))
            logs_save_dir_path = Path(hydra.utils.to_absolute_path(cfg.resume_from_checkpoint)).parent / "logs_dir"

        logger = pl_loggers.TensorBoardLogger(save_dir=logs_save_dir_path, sub_dir="logs/", name="", version="")
    else:
        logger = pl_loggers.TensorBoardLogger(save_dir="logs_dir/", sub_dir="logs/", name="", version="")
        checkpoint_dir_path = "exp_weights/"
        


    checkpoint_callback = ModelCheckpoint(
        #monitor="train_loss", #/dataloader_idx_0",
        dirpath=checkpoint_dir_path,                 
        filename=train_path.stem+"_log_"+"-{epoch:02d}-{val_loss:.4f}",
        mode="min",
        save_top_k=-1,
        every_n_epochs=cfg.save_checkpoint_every_n_epoch,
    )

    

    if cfg.resume_from_checkpoint:
        print("Resuming from checkpoint")
        if is_folder:
            # Find the latest checkpoint
            checkpoints = list(checkpoint_dir_path.glob("*.ckpt"))
            checkpoints.sort(key=os.path.getmtime)
            path_to_restart = checkpoints[-1]
        else:
            path_to_restart = Path(hydra.utils.to_absolute_path(cfg.resume_from_checkpoint))
    else:
        path_to_restart = None


    alternating_dataset_callback = AlternatingDatasetCallback()
    

    

    trainer = pl.Trainer(
        strategy="ddp",
        gpus=cfg.gpu,
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        precision=cfg.precision,
        callbacks=[checkpoint_callback, lr_monitor, alternating_dataset_callback],
        resume_from_checkpoint=path_to_restart,
        logger=logger,
        reload_dataloaders_every_n_epochs=1
    )



    trainer.fit(model, data)


if __name__ == "__main__":
    main()
