import argparse
import pickle as pkl
import json
from probe import LinearProbe, PL_LinearProbe
import torch
import torch.optim as optim
import os
from random import shuffle
import wandb
import lightning as L
from utils.args_utils import dict_to_object
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split



def parse_args():

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, help="Path to JSON config file")
    pre_args, _ = pre_parser.parse_known_args()

    args = {}
    
    # If --config is provided, load arguments from JSON
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            args.update(json.load(f))
    # Otherwise, use normal argparse
    else:
        parser = argparse.ArgumentParser()
        
        # TODO change the data loading path to multiple arguments
        parser.add_argument("--path_to_train_data", type=str, required=True)
        parser.add_argument("--path_to_test_data", type=str, required=True)
        parser.add_argument("--is_image_probe", type=int, required=True)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--n_epochs", type=int, default=1000)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--eval_interval_epochs", type=int, default=1)
        parser.add_argument("--wandb_dir", type=str, default="./wandb/")
        parser.add_argument("--repr_layer_idx", type=int, default=-1, help="When using mllms, which layer's representation is used to train the probe")

        parser.add_argument("--wandb_project", type=str, default="multimodal_conflict_proc")

        args.update(vars(parser.parse_args()))

    args = dict_to_object(args)

    return args



def load_precomputed_data(path_to_data):
    with open(path_to_data, "rb") as f:
        precomputed_data = pkl.load(f)

    return precomputed_data

def load_training_data(multimodal_data, val_portion=0.2, layer=-1):
        
    datasize = len(multimodal_data)

    train_size = int(datasize * (1 - val_portion))

    all_data = ProbeDataset(multimodal_data, layer=layer)
    
    train_data, val_data = random_split(all_data, [1.0 - val_portion, val_portion])

    return train_data, val_data
    

def get_path_to_save(self_args, train_args, test_args):

    work_dir = train_args.work_dir
    
    model_family = train_args.model_family
    model_name = train_args.model_name
    dataset_name = train_args.dataset
    train_caption_type = train_args.caption_type
    test_caption_type = test_args.caption_type
    with_helper = "with_helper" if (train_args.is_explicit_helper == 1) else "no_helper"
    probe_type = "imageprobe" if self_args.is_image_probe== 1 else "captionprobe"
    seed = self_args.seed

    path_to_save = f"{work_dir}/outputs/probe_evaluation/{model_family}/{dataset_name}/{with_helper}/{probe_type}_layer{self_args.repr_layer_idx}/{train_caption_type}_to_{test_caption_type}_seed{seed}"

    return path_to_save


class ProbeDataset(Dataset):
    def __init__(self, multimodal_data, layer=-1):

        self.multimodal_data = multimodal_data
        self.layer = layer

    def __len__(self):
        return len(self.multimodal_data["image_labels"])

    def __getitem__(self, idx):

        return self.multimodal_data["image_labels"][idx], self.multimodal_data["caption_labels"][idx], self.multimodal_data["representations"][:, self.layer, :][idx] 


if __name__ == "__main__":

    args = parse_args()

    L.seed_everything(args.seed)

    precomputed_train_data = load_precomputed_data(args.path_to_train_data)
    precomputed_test_data = load_precomputed_data(args.path_to_test_data)

    args.result_save_path = get_path_to_save(args, precomputed_train_data["precomputed_data_args"], precomputed_test_data["precomputed_data_args"])

    print(args.result_save_path)

    train_dataset, val_dataset = load_training_data(precomputed_train_data["data"], layer=args.repr_layer_idx)
    test_dataset = ProbeDataset(precomputed_test_data["data"], layer=args.repr_layer_idx)


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = LinearProbe(precomputed_train_data["repr_size"], precomputed_train_data["num_classes"], args.is_image_probe == 1)

    checkpoint = ModelCheckpoint(
        monitor="validation/accuracy",
        filename="epoch:{epoch}-val_acc:{validation/accuracy:.6f}",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    pl_model = PL_LinearProbe(
        model,
        args
    )

    wandb_name = f"|Probing|Model:{precomputed_train_data['precomputed_data_args'].model_name}|Train:{precomputed_train_data['precomputed_data_args'].dataset}-{precomputed_train_data['precomputed_data_args'].caption_type}-{'imageprobe' if (args.is_image_probe == 1) else 'captionprobe'}|Test:{precomputed_test_data['precomputed_data_args'].dataset}-{precomputed_test_data['precomputed_data_args'].caption_type}|seed{args.seed}"
    wandb_logger = WandbLogger(project=args.wandb_project, name=wandb_name, save_dir=args.wandb_dir)

    wandb_logger.experiment.config.update({
        "train_model_family" : precomputed_train_data["precomputed_data_args"].model_family,
        "train_model_name" : precomputed_train_data["precomputed_data_args"].model_name,
        "train_dataset_name" : precomputed_train_data["precomputed_data_args"].dataset,
        "train_caption_type" : precomputed_train_data["precomputed_data_args"].caption_type,
        "train_q_template": precomputed_train_data["precomputed_data_args"].question_template,
        "test_model_family" : precomputed_test_data["precomputed_data_args"].model_family,
        "test_dataset_name" : precomputed_test_data["precomputed_data_args"].dataset,
        "test_caption_type" : precomputed_test_data["precomputed_data_args"].caption_type,
        "repr_layer_idx": args.repr_layer_idx,
        "probe_type": "imageprobe" if args.is_image_probe==1 else "captionprobe",
    })

    trainer = Trainer(
        max_epochs=args.n_epochs, 
        logger=wandb_logger,
        callbacks=[checkpoint],
        check_val_every_n_epoch=args.eval_interval_epochs,
        log_every_n_steps=10,
    )

    trainer.fit(model=pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # load the best model weights
    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path, map_location=lambda storage, loc: storage)
    pl_model.model.state_dict = checkpoint["state_dict"]

    trainer = Trainer(devices=1, num_nodes=1, logger=wandb_logger)
    trainer.test(model=pl_model, dataloaders=test_dataloader)

    torch.save(checkpoint, f"{args.result_save_path}_probe.ckpt")
    
    wandb.finish()