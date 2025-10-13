import argparse
import math
import os

import torch
import torch.optim as optim
from monai.data import Dataset
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from bmmae.augmentations import get_train_transforms, get_val_transforms
from bmmae.decoders import Decoder
from bmmae.loops import loop_pretrain
from bmmae.model import BMMAE
from bmmae.tokenizers import MRITokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../rsna/BraTS2021/", help="Path to the data directory")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train the model")
    parser.add_argument(
        "--modalities",
        type=list,
        nargs="+",
        default=["t1", "t1ce", "t2", "flair"],
        help="List of modalities to use for training",
    )
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--n_cpus", type=int, default=20, help="Number of cpus to use for data loading")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for training")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the model")
    parser.add_argument("--img_size", type=int, default=128, help="Image size for the model")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size for the model")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers for the model")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads for the model")
    parser.add_argument("--mlp_dim", type=int, default=1536, help="MLP dimension for the model")
    parser.add_argument("--decoder_hidden_size", type=int, default=384, help="Decoder hidden size for the model")
    parser.add_argument("--decoder_num_layers", type=int, default=3, help="Number of layers for the decoder")
    parser.add_argument("--decoder_num_heads", type=int, default=12, help="Number of heads for the decoder")
    parser.add_argument(
        "--qkv_bias", type=bool, default=True, help="Whether to use bias in qkv projection for the model"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--masking_ratio", type=float, default=0.75, help="Masking ratio for the model")
    parser.add_argument("--norm_pix_loss", type=bool, default=False, help="Whether to normalize the loss or not")
    parser.add_argument("--entity", type=str, default=None, help="Entity for wandb logging")
    parser.add_argument("--project", type=str, default="bmmae", help="Project for wandb logging")
    parser.add_argument("--seed", type=int, default=1999, help="Seed for reproducibility")
    parser.add_argument("--warmup_epochs", type=int, default=50, help="Warmup epochs for the model")
    parser.add_argument("--scaling_factor", type=int, default=1500, help="Scaling factor for the model")
    args = parser.parse_args()
    wandb_logging = True if args.entity is not None else False
    if wandb_logging:
        import wandb

        run = wandb.init(project=args.project, entity=args.entity, reinit=True, config=vars(args), name=f"BM-MAE_pretraining")

    set_determinism(seed=args.seed)
    train_patients = os.listdir(os.path.join(args.data_dir, "archive"))
    val_patients = os.listdir(os.path.join(args.data_dir, "val"))
    
    ## TO WITHDRAW
    # import pandas as pd
    # tcga = pd.read_csv('data/data.csv')
    # # keep only raw where MRI is not NaN
    # tcga = tcga[~tcga['MRI'].isna()]
    # patients_to_withdraw = tcga.MRI.str.split('/').str[-1].values.tolist()
    # train_patients = [patient for patient in train_patients if patient not in patients_to_withdraw]
    # val_patients = [patient for patient in val_patients if patient not in patients_to_withdraw]
    #### END 

    train_files = [
        {
            modality: os.path.join(args.data_dir, "archive", patient, f"{patient}_{modality}.nii.gz")
            for modality in args.modalities
        }
        for patient in train_patients
    ]
    val_files = [
        {
            modality: os.path.join(args.data_dir, "val", patient, f"{patient}_{modality}.nii.gz")
            for modality in args.modalities
        }
        for patient in val_patients
    ]
    transforms_train = get_train_transforms(args.modalities)
    transforms_val = get_val_transforms(args.modalities)

    train_dataset = Dataset(data=train_files, transform=transforms_train)
    val_dataset = Dataset(data=val_files, transform=transforms_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpus,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_cpus,
        pin_memory=True,
        persistent_workers=True,
    )

    tokenizers = {
        modality: MRITokenizer(
            patch_size=(args.patch_size, args.patch_size, args.patch_size),
            img_size=(args.img_size, args.img_size, args.img_size),
            hidden_size=args.hidden_size,
        )
        for modality in args.modalities
    }

    decoder = Decoder(
        decoder_hidden_size=args.decoder_hidden_size,
        num_layers=args.decoder_num_layers,
        mlp_dim=args.mlp_dim,
        dropout_rate=args.dropout_rate,
        num_heads=args.decoder_num_heads,
        qkv_bias=args.qkv_bias,
        img_size=(args.img_size, args.img_size, args.img_size),
        patch_size=(args.patch_size, args.patch_size, args.patch_size),
    )
    

    model = BMMAE(
        modalities=args.modalities,
        tokenizers=tokenizers,
        decoder=decoder,
        masking_ratio=args.masking_ratio,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        qkv_bias=args.qkv_bias
    )
 
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    device = "cuda"
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_func = lambda epoch: min(
        (epoch + 1) / (args.warmup_epochs + 1e-8),
        0.5 * (math.cos(epoch / args.scaling_factor * math.pi) + 1),
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    _, _ = loop_pretrain(
        model=model,
        modalities=args.modalities,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=device,
        norm_pix_loss=args.norm_pix_loss,
        wandb_logging=wandb_logging,
    )
    if wandb_logging:
        run.finish()
