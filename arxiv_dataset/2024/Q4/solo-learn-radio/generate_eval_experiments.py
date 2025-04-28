import wandb
import yaml
import os
import argparse
import json
import pathlib

def get_local_config(model_path):
    args_file = open(os.path.join(os.path.split(model_path)[0], "args.json"))
    # Inizializza W&B con il tuo API key
    config = json.load(args_file)

    pretrain_method = config["method"]
    name = config["name"]
    epochs = config["max_epochs"]  # Assicurati che questo campo esista nella tua configurazione
    backbone = config["backbone"]  # Assicurati che questo campo esista nella tua configurazione
    
    wandb.finish()  # Chiudi l'esperimento W&B

    return pretrain_method, name, epochs, backbone
# script to generate eval experiment
# finetune
# baseline
# augmentations


def get_wandb_config(run_id):
    # Inizializza W&B con il tuo API key
    api = wandb.Api()
    run = api.run(f"thomas-cecconello/solo-learn/{run_id}")
    config = run.config

    pretrain_method = config["method"]
    name = config["name"]
    epochs = config["max_epochs"]  # Assicurati che questo campo esista nella tua configurazione
    backbone = config["backbone"]["name"]  # Assicurati che questo campo esista nella tua configurazione
    
    wandb.finish()  # Chiudi l'esperimento W&B

    return pretrain_method, name, epochs, backbone

def generate_pretrained_feature_extractor_path(pretrain_method, run_id, name, epochs):
    return f"trained_models/{pretrain_method}/{run_id}/{name}-{run_id}-ep={epochs-1}.ckpt"

def generate_yaml_from_wandb(local_or_wandb, run_id, model_path, dataset, dataset_path, finetune, augmentations, devices, num_workers, K_fold_value, exp_repetitions, datalist, balancing_strategy, sample_size):
    if local_or_wandb == "local":
        pretrain_method, pretrain_name, epochs, backbone = get_local_config(model_path)
        pretrained_feature_extractor = model_path
    elif local_or_wandb == "wandb":
        pretrain_method, pretrain_name, epochs, backbone = get_wandb_config(run_id)
        pretrained_feature_extractor = generate_pretrained_feature_extractor_path(pretrain_method, run_id, pretrain_name, epochs)

    if finetune:
        optimizer= {
            "name": "adamw",
            "batch_size": 256,
            "lr": 5e-4,
            "weight_decay": 0.,
            "layer_decay": 0.
        }
        scheduler= {
            "name": "warmup_cosine",
            "lr_decay_steps": 0.0
        }
        
    else:
        optimizer= {
            "name": "sgd",
            "batch_size": 256,
            "lr": 0.3,
            "weight_decay": 0
        }
        scheduler = {
            "name": "step",
            "lr_decay_steps": [10, 80]
        }
    
    if augmentations == "nomeanstd":
        aug_suffix = ""
    elif augmentations == "meanstd":
        aug_suffix = "_meanstd"

    finetune_text = "finetune" if finetune else "linear"
    name = f"__{finetune_text}__{dataset}__minmax__{pretrain_name}__K{K_fold_value}__{balancing_strategy}__{os.path.splitext(datalist)[0]}__"
    data_path = dataset_path
    #"../2-ROBIN" if dataset == "robin" else "../RGZ-D1-smorph-dataset"
    
    config = {
        "name": name,
        "pretrained_feature_extractor": pretrained_feature_extractor,
        "backbone": {
            "name": backbone
        },
        "finetune": True if finetune else False,
        "pretrain_method": pretrain_method,
        "data": {
            "dataset": dataset,
            "train_path": data_path,
            "val_path": data_path,
            "datalist": datalist,
            "balancing_strategy": balancing_strategy,
            "sample_size": sample_size,
            "format": "image_folder",
            "num_workers": num_workers
        },
        "K_fold": K_fold_value,
        "repetitions": exp_repetitions,
        "devices": devices,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "checkpoint": {
            "enabled": False,
            "dir": "trained_models",
            "frequency": 25
        },
        "auto_resume": {
            "enabled": False
        },
        "performance": {
            "disable_channel_last": True
        },
        "max_epochs": 100,
        "sync_batchnorm": True,
        "accelerator": "gpu",
        "strategy": "auto",
        "precision": 16,
        "defaults": [
            "_self_",
            {"wandb": "private.yaml"},
            {"augmentations": f"asymmetric_minmax{aug_suffix}.yaml"},
            {"override hydra/hydra_logging": "disabled"},
            {"override hydra/job_logging": "disabled"}
        ],
        "hydra": {
            "output_subdir": None,
            "run": {
                "dir": "."
            }
        },
    }
    
    yaml_content = yaml.dump(config)

    output_directory = os.path.join("scripts", "kfold", finetune_text, dataset)
    yaml_filename = f"{name}.yaml"
    yaml_filepath = os.path.join(output_directory, yaml_filename)
    print(yaml_filepath)

    with open(yaml_filepath, "w") as yaml_file:
        yaml_file.write(yaml_content)

    name = f"__{finetune_text}__{dataset}__mixed__{pretrain_name}__K{K_fold_value}__{balancing_strategy}__{os.path.splitext(datalist)[0]}__"

    config = {
        "name": name,
        "pretrained_feature_extractor": pretrained_feature_extractor,
        "backbone": {
            "name": backbone
        },
        "pretrain_method": pretrain_method,
        "finetune": True if finetune else False,
        "data": {
            "dataset": dataset,
            "train_path": data_path,
            "val_path": data_path,
            "datalist": datalist,
            "balancing_strategy": balancing_strategy,
            "sample_size": sample_size,
            "format": "image_folder",
            "num_workers": num_workers
        },
        "K_fold": K_fold_value,
        "repetitions": exp_repetitions,
        "devices": devices,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "checkpoint": {
            "enabled": False,
            "dir": "trained_models",
            "frequency": 25
        },
        "auto_resume": {
            "enabled": False
        },
        "performance": {
            "disable_channel_last": True
        },
        "max_epochs": 100,
        "sync_batchnorm": True,
        "accelerator": "gpu",
        "strategy": "auto",
        "precision": 16,
        "defaults": [
            "_self_",
            {"wandb": "private.yaml"},
            {"augmentations": f"asymmetric_mixed{aug_suffix}.yaml"},
            {"override hydra/hydra_logging": "disabled"},
            {"override hydra/job_logging": "disabled"}
        ],
        "hydra": {
            "output_subdir": None,
            "run": {
                "dir": "."
            }
        },
    }
    
    yaml_content = yaml.dump(config)

    output_directory = os.path.join("scripts", "kfold", finetune_text, dataset)
    yaml_filename = f"{name}.yaml"
    yaml_filepath = os.path.join(output_directory, yaml_filename)
    print(yaml_filepath)

    with open(yaml_filepath, "w") as yaml_file:
        yaml_file.write(yaml_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate yaml config file to validate the pretrained models")
    parser.add_argument("--local_or_wandb", type=str, required=True, help="Local or wandb")
    parser.add_argument("--augmentations", type=str, default="nomeanstd", help="Meanstd or not", choices=["meanstd", "nomeanstd"])
    parser.add_argument("--run_id", required=False, help="wandb experiment ID")
    parser.add_argument("--finetune", action=argparse.BooleanOptionalAction, help="Linear evaluation or finetuning")
    parser.add_argument("--model_path", type=pathlib.Path, default = "", required=False, help="Local model path")
    parser.add_argument("--devices", nargs="+", type=int, required=True, help="GPU devices list")
    parser.add_argument("--num_workers", type=int, default=6, help="CPU workers")
    parser.add_argument("--K", type=int, default=3, help="K value of K-fold cross validation")
    parser.add_argument("--reps", type=int, default=1, help="Number of experiment repetitions")
    parser.add_argument("--dataset", type=str, default="robin", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="../2-ROBIN", help="Dataset path")
    parser.add_argument("--datalist", type=str, default="info.json", help="Datalist name in .json format")
    parser.add_argument("--balancing_strategy", type=str, default="as_is", help="as_is, balanced_downsampled,  balanced_fixed")
    parser.add_argument("--sample_size", type=int, default=1650, help="Number of example per class in balancing phase")
    #parser.add_argument("--meanstd", type=bool, default=False, help="Se usare la normalizzazione meanstd")
    


    args = parser.parse_args()
    print(args.finetune)

    #datasets = ["rgz", "robin"]  # Aggiungi altri dataset se necessario

    #for dataset in datasets:
    #local_or_wandb, run_id, model_path, dataset, dataset_path, finetune, augmentations,
    generate_yaml_from_wandb(args.local_or_wandb, args.run_id, args.model_path, args.dataset, args.dataset_path, args.finetune, args.augmentations, args.devices, args.num_workers, args.K, args.reps, args.datalist, args.balancing_strategy, args.sample_size)

    # python generate_eval_experiments.py --run_id uja9qvb7 --devices 0 1
    # python generate_eval_experiments.py --run_id uja9qvb7 --devices 0 1 --num_workers 16
