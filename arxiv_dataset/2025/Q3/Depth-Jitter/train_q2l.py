import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    TQDMProgressBar, ModelCheckpoint, EarlyStopping
)
from q2l_labeller.data.coco_data_module import COCODataModule
from q2l_labeller.pl_modules.query2label_train_module import Query2LabelTrainModule
from q2l_labeller.data.dataset import SeaThruAugmentation

# Set random seed
pl.seed_everything(100)
torch.backends.cudnn.benchmark = True

# Argument Parser
parser = argparse.ArgumentParser(description="Depth-Jitter Training Script")

# Dataset Selection
parser.add_argument(
    "--dataset", type=str, choices=["UTDAC2020", "FathomNet"], default="FathomNet",
    help="Select dataset: 'UTDAC2020' or 'FathomNet' (default: FathomNet)"
)

# Model Configuration Arguments
parser.add_argument("--backbone", type=str, default="resnest101e", help="Model backbone architecture (default: resnest101e)")
parser.add_argument("--conv_out_dim", type=int, default=2048, help="Convolutional output dimension (default: 2048)")
parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension (default: 256)")
parser.add_argument("--num_encoders", type=int, default=2, help="Number of encoder layers (default: 2)")
parser.add_argument("--num_decoders", type=int, default=3, help="Number of decoder layers (default: 3)")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (default: 8)")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128)")
parser.add_argument("--img_size", type=int, default=384, help="Image size (default: 384)")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer (default: 0.9)")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer (default: 1e-2)")
parser.add_argument("--thresh", type=float, default=0.5, help="Threshold for classification (default: 0.4)")
parser.add_argument("--loss", type=str, choices=["ASL", "BCE", "mll"], default="ASL", help="Loss function (default: ASL)")

# New Arguments: Augmentation and Sampling Strategy
parser.add_argument("--augmentation_strategy", type=str, choices=["baseline", "seathru", "combined","colorjitter"], default="seathru", help="Augmentation strategy: 'baseline', 'seathru', or 'combined' (default: seathru)")
parser.add_argument("--sampling_strategy", type=str, choices=["default", "oversample", "undersample"], default="oversample", help="Sampling strategy: 'default', 'oversample', or 'undersample' (default: oversample)")

args = parser.parse_args()

# Dataset configurations
datasets = {
    "UTDAC2020": {
        "image_folder": "/home/mundus/mrahman528/thesis/thesis_paper/UTDAC2020",
        "depth_image_folder": "/home/mundus/mrahman528/thesis/thesis_paper/UTDAC2020/depth_train",
        "depth_npy_folder": "/home/mundus/mrahman528/thesis/thesis_paper/UTDAC2020/depth_train",
        "seathru_parameters_path": "/home/mundus/mrahman528/thesis/thesis_paper/parameters_train.json",
        "depth_variance_path": "/home/mundus/mrahman528/Depth-Jitter/depth_variance_utdac.json",
        "threshold": 9.49,
        "num_classes": 4,
        "dataset":"UTDAC2020"
    },
    "FathomNet": {
        "image_folder": "/home/mundus/mrahman528/projects/mir/depth_jitter/fathomnet_2023_dataset",
        "depth_image_folder": "/home/mundus/mrahman528/projects/mir/depth_jitter/fathomnet_2023_dataset/depth_vis_train",
        "depth_npy_folder": "/home/mundus/mrahman528/projects/mir/depth_jitter/fathomnet_2023_dataset/depth_vis_train",
        "seathru_parameters_path": "/home/mundus/mrahman528/Depth-Jitter/parameters_train.json",
        "depth_variance_path": "/home/mundus/mrahman528/Depth-Jitter/depth_variance_fathomnet.json",
        "threshold": 3.66,
        "num_classes": 290,
        "dataset":"FathomNet"
    }
}

# Select dataset based on user input
selected_dataset = datasets[args.dataset]

# Initialize SeaThru Augmentation
seathru_transform = SeaThruAugmentation(
    selected_dataset["image_folder"],
    selected_dataset["depth_image_folder"],
    selected_dataset["depth_npy_folder"],
    selected_dataset["seathru_parameters_path"],
    selected_dataset["depth_variance_path"],
    threshold=selected_dataset["threshold"],
    dataset = selected_dataset["dataset"]
)

# Initialize Data Module
coco = COCODataModule(
    data_dir=selected_dataset["image_folder"],
    img_size=args.img_size,
    batch_size=args.batch_size,
    num_workers=8,
    use_cutmix=True,
    cutmix_alpha=1.0,
    train_classes=None,
    sampling_strategy=args.sampling_strategy,
    augmentation_strategy=args.augmentation_strategy,
    num_classes=selected_dataset["num_classes"],
    seathru_transform=seathru_transform
)

# Model Parameters (Updated n_classes Dynamically)
param_dict = {
    "backbone_desc": args.backbone,
    "conv_out_dim": args.conv_out_dim,
    "hidden_dim": args.hidden_dim,
    "num_encoders": args.num_encoders,
    "num_decoders": args.num_decoders,
    "num_heads": args.num_heads,
    "batch_size": args.batch_size,
    "image_dim": args.img_size,
    "learning_rate": args.learning_rate,
    "momentum": args.momentum,
    "weight_decay": args.weight_decay,
    "n_classes": selected_dataset["num_classes"],
    "thresh": args.thresh,
    "use_cutmix": True,
    "use_pos_encoding": True,
    "loss": args.loss,
    "data": coco
}

# Initialize Model
pl_model = Query2LabelTrainModule(**param_dict)

# WandB Logger
wandb_logger = WandbLogger(
    project=f"depth_jitter-experiment-06(ss)-{args.dataset}",
    save_dir="training/logs/depthJitter",
    log_model=True,
    id=f"{args.backbone}-{args.loss}-{args.dataset}-{args.augmentation_strategy}-{args.sampling_strategy}-{args.img_size}-exp09",
    sync_tensorboard=True
)

# Model Checkpoint Callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_mAP@20",
    dirpath=f"training/checkpoints/depth_jitter_{args.dataset}",
    filename="best-checkpoint-{epoch:02d}-{val_mAP:.2f}",
    save_top_k=1,
    mode="min"
)

# Early Stopping Callback
early_stopping_callback = EarlyStopping(
    monitor="val_mAP",
    patience=30,
    verbose=True,
    mode="min"
)

# Trainer Configuration
trainer = pl.Trainer(
    max_epochs=24,
    precision=16,
    accelerator="gpu",
    devices="auto",
    strategy="ddp",
    # gradient_clip_val=0.5,
    logger=wandb_logger,
    default_root_dir=f"training/checkpoints/depth_jitter_{args.dataset}",
    callbacks=[
        TQDMProgressBar(refresh_rate=100),
        checkpoint_callback,
        early_stopping_callback
    ],
    # accumulate_grad_batches=4,
    # detect_anomaly=True,
    # profiler="simple"
)

# Start Training
trainer.fit(pl_model, param_dict["data"])
