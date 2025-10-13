import argparse
import pickle as pkl
from probe import LinearProbe_consistency, PL_LinearProbe_consistency_gen
import torch
import torch.optim as optim
import os
import wandb
import json
import lightning as L
from utils.args_utils import dict_to_object
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from random import shuffle



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
        parser.add_argument("--path_to_train_data_consistent", type=str, required=True)
        parser.add_argument("--path_to_train_data_inconsistent", type=str, required=True)
        parser.add_argument("--path_to_test_data_consistent", type=str, required=True)
        parser.add_argument("--path_to_test_data_inconsistent", type=str, required=True)

        parser.add_argument("--repr_layer_idx", type=int, default=-1, help="When using mllms, which layer's representation is used to train the probe")
        parser.add_argument("--k_folds", type=int, default=3)
        parser.add_argument("--fold_idx", type=int, required=True)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--n_epochs", type=int, default=10)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--eval_interval_epochs", type=int, default=1)
        parser.add_argument("--wandb_dir", type=str, default="")
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
    

def combine_data(some_data, some_other_data):
    rst_data = {
        "image_labels": np.concatenate([some_data['image_labels'], some_other_data['image_labels']]),
        "caption_labels": np.concatenate([some_data['caption_labels'], some_other_data['caption_labels']]),
        "representations": np.concatenate([some_data['representations'], some_other_data['representations']]),
    }

    return rst_data


def get_path_to_save(self_args, train_args, test_args, current_fold_idx):

    work_dir = train_args.work_dir
    
    model_family = train_args.model_family
    dataset_name = train_args.dataset
    seed = self_args.seed

    path_to_save = f"{work_dir}/outputs/probe_evaluation_consistency/{model_family}/{dataset_name}/layer{self_args.repr_layer_idx}/seed{seed}_fold{current_fold_idx}"

    return path_to_save


class ProbeDataset(Dataset):
    def __init__(self, multimodal_data, layer=-1):

        self.multimodal_data = multimodal_data
        self.layer = layer

    def __len__(self):
        return len(self.multimodal_data["image_labels"])

    def __getitem__(self, idx):

        if self.multimodal_data["image_labels"][idx] == self.multimodal_data["caption_labels"][idx]:
            consistency = 1
        else:
            consistency = 0
        
        return consistency, self.multimodal_data["representations"][:, self.layer, :][idx] 


if __name__ == "__main__":

    args = parse_args()

    L.seed_everything(args.seed)

    consistent_precomputed_train_data = load_precomputed_data(args.path_to_train_data_consistent)
    inconsistent_precomputed_train_data = load_precomputed_data(args.path_to_train_data_inconsistent)

    combined_train_data = combine_data(consistent_precomputed_train_data["data"], inconsistent_precomputed_train_data["data"])

    consistent_precomputed_test_data = load_precomputed_data(args.path_to_test_data_consistent)
    inconsistent_precomputed_test_data = load_precomputed_data(args.path_to_test_data_inconsistent)

    combined_test_data = combine_data(consistent_precomputed_test_data["data"], inconsistent_precomputed_test_data["data"])

    # get kfolds of labels
    num_classes = inconsistent_precomputed_train_data['num_classes']
    all_labels  = np.arange(num_classes)

    np.random.seed(args.seed)
    np.random.shuffle(all_labels)
    folds = np.array_split(all_labels, args.k_folds)


    print(f"=======*** Currently at [{args.fold_idx + 1}/{args.k_folds}] fold ***=======")

    test_labels  = folds[args.fold_idx]

    print("test labels: ", test_labels)

    train_labels = np.concatenate([folds[i] for i in range(args.k_folds) if i != args.fold_idx])

    # set the path to save results
    args.result_save_path = get_path_to_save(args, consistent_precomputed_train_data["precomputed_data_args"], consistent_precomputed_test_data["precomputed_data_args"], args.fold_idx)
    print(f"Saving to {args.result_save_path}")

    # train: both labels in train_labels
    train_mask = (
        np.isin(combined_train_data['image_labels'],   train_labels) &
        np.isin(combined_train_data['caption_labels'], train_labels)
    )

    train_dataset, val_dataset = load_training_data({
            'image_labels': combined_train_data['image_labels'][train_mask],
            'caption_labels': combined_train_data['caption_labels'][train_mask],
            'representations': combined_train_data['representations'][train_mask],
        }, layer=args.repr_layer_idx)

    # in_dist: both labels in train_labels
    in_dist_mask = (
        np.isin(combined_test_data['image_labels'],   train_labels) &
        np.isin(combined_test_data['caption_labels'], train_labels)
    )
    # semi‑in: exactly one label in train_labels
    semi_dist_mask = (
    (np.isin(combined_test_data['image_labels'],   train_labels) &
        ~np.isin(combined_test_data['caption_labels'], train_labels))
    ) | (
    (~np.isin(combined_test_data['image_labels'],   train_labels) &
        np.isin(combined_test_data['caption_labels'], train_labels))
    )
    # out‑of‑distribution: both labels in test_labels
    ood_mask = (
        np.isin(combined_test_data['image_labels'],   test_labels) &
        np.isin(combined_test_data['caption_labels'], test_labels)
    )

    test_in_distribution_dataset = ProbeDataset({
            'image_labels': combined_test_data['image_labels'][in_dist_mask],
            'caption_labels': combined_test_data['caption_labels'][in_dist_mask],
            'representations': combined_test_data['representations'][in_dist_mask],
        }, layer=args.repr_layer_idx)

    test_semi_in_distribution_dataset = ProbeDataset({
            'image_labels': combined_test_data['image_labels'][semi_dist_mask],
            'caption_labels': combined_test_data['caption_labels'][semi_dist_mask],
            'representations': combined_test_data['representations'][semi_dist_mask],
        }, layer=args.repr_layer_idx)

    test_out_of_distribution_dataset = ProbeDataset({
            'image_labels': combined_test_data['image_labels'][ood_mask],
            'caption_labels': combined_test_data['caption_labels'][ood_mask],
            'representations': combined_test_data['representations'][ood_mask],
        }, layer=args.repr_layer_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataloader_id = DataLoader(test_in_distribution_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader_semi_id = DataLoader(test_semi_in_distribution_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader_ood = DataLoader(test_out_of_distribution_dataset, batch_size=args.batch_size, shuffle=False)

    model = LinearProbe_consistency(consistent_precomputed_train_data["repr_size"], 2)

    checkpoint = ModelCheckpoint(
        monitor="validation/accuracy",
        filename="epoch:{epoch}-val_acc:{validation/accuracy:.6f}",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    pl_model = PL_LinearProbe_consistency_gen(
        model,
        args,
        [len(test_in_distribution_dataset), len(test_semi_in_distribution_dataset), len(test_out_of_distribution_dataset)]
    )

    wandb_name = f"|Consistency_Probing|Model:{consistent_precomputed_train_data['precomputed_data_args'].model_name}|seed{args.seed}|fold{args.fold_idx}"
    wandb_logger = WandbLogger(project=args.wandb_project, name=wandb_name, save_dir=args.wandb_dir)

    wandb_logger.experiment.config.update({
        "train_model_name" : consistent_precomputed_train_data["precomputed_data_args"].model_name,
        "train_model_family" : consistent_precomputed_train_data["precomputed_data_args"].model_family,
        "train_dataset_name" : consistent_precomputed_train_data["precomputed_data_args"].dataset,
        "repr_layer_idx": args.repr_layer_idx
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
    trainer.test(model=pl_model, dataloaders=[test_dataloader_id, test_dataloader_semi_id, test_dataloader_ood])

    torch.save(checkpoint, f"{args.result_save_path}_probe.ckpt")

    wandb.finish()

