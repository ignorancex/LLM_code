import argparse
import math
import os

import torch
from monai.data import Dataset
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from bmmae.augmentations import get_train_seg_transforms, get_val_seg_transforms
from bmmae.loops import loop_segmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../rsna/BraTS2021/", help="Path to the data directory")
    parser.add_argument("--epochs", type=int, default=70, help="Number of epochs to train the model")
    parser.add_argument("--modalities", nargs="+", default=["t1"], help="List of modalities to use for training")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--n_cpus", type=int, default=12, help="Number of cpus to use for data loading")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for training")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the model")
    parser.add_argument("--img_size", type=int, default=128, help="Image size for the model")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size for the model")
    parser.add_argument("--mlp_dim", type=int, default=1536, help="Number of layers for the model")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads for the model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--entity", type=str, default=None, help="Entity for wandb logging")
    parser.add_argument("--project", type=str, default="bmmae", help="Project for wandb logging")
    parser.add_argument("--seed", type=int, default=1999, help="Seed for reproducibility")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs for the model")
    parser.add_argument("--scaling_factor", type=int, default=70, help="Scaling factor for the model")
    parser.add_argument(
        "--from_scratch", type=bool, default=False, help="Whether to train the model from scratch or not"
    )
    parser.add_argument("--pretraining", type=str, default='bmmae')
    args = parser.parse_args()
    if args.from_scratch:
        args.pretraining = 'fs'

    wandb_logging = True if args.entity is not None else False
    if wandb_logging:
        import wandb

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"{args.pretraining}_{'-'.join(args.modalities)}_{args.from_scratch}",
            reinit=True,
            config=vars(args),
        )

    set_determinism(seed=args.seed)
    train_patients = os.listdir(os.path.join(args.data_path, "archive"))
    val_patients = os.listdir(os.path.join(args.data_path, "val"))
    train_files = [
        {
            "image": [
                os.path.join(args.data_path, "archive", patient, f"{patient}_{modality}.nii.gz")
                for modality in args.modalities
            ],
            "label": os.path.join(args.data_path, "archive", patient, f"{patient}_seg.nii.gz"),
        }
        for patient in train_patients
    ]
    val_files = [
        {
            "image": [
                os.path.join(args.data_path, "val", patient, f"{patient}_{modality}.nii.gz")
                for modality in args.modalities
            ],
            "label": os.path.join(args.data_path, "val", patient, f"{patient}_seg.nii.gz"),
        }
        for patient in val_patients
    ]
    train_dataset = Dataset(data=train_files, transform=get_train_seg_transforms())
    val_dataset = Dataset(data=val_files, transform=get_val_seg_transforms())

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
        shuffle=False,
        num_workers=args.n_cpus,
        pin_memory=True,
        persistent_workers=True,
    )

    model = UNETR(
        in_channels=len(args.modalities),
        out_channels=3,
        img_size=(args.img_size, args.img_size, args.img_size),
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
        proj_type="conv",
        qkv_bias=True,
    )

    if not args.from_scratch:
        if args.pretraining == 'bmmae':
            print("Using pretrained model - BM-MAE")
            from bmmae.model import BMMAEViT
            from bmmae.tokenizers import MRITokenizer

            state_dict = torch.load(f"pretrained_models/bmmae.pth", weights_only=True)
            state_dict = {k: v for k, v in state_dict.items() if "decoder" not in k}
            state_dict = {k: v for k, v in state_dict.items() if "cls_token" not in k}

            tokenizers = {
                modality: MRITokenizer(patch_size=(16, 16, 16), img_size=(128, 128, 128), hidden_size=768)
                for modality in args.modalities
            }

            vit = BMMAEViT(
                modalities=args.modalities,
                tokenizers=tokenizers,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                qkv_bias=True,
                classification=False
            )

            vit.load_state_dict(state_dict, strict=False)

            model.vit = vit
        else:
            print(f"Using {args.pretraining} pretrained model")
            from monai.networks.nets import ViT
            state_dict = torch.load(f"pretrained_models/{args.pretraining}/{'-'.join(args.modalities)}.pth", weights_only=True)
            state_dict = {k: v for k, v in state_dict.items() if "decoder" not in k}
            state_dict = {k: v for k, v in state_dict.items() if "cls_token" not in k}
            # remove also mask tokens
            state_dict = {k: v for k, v in state_dict.items() if "mask_token" not in k}
            vit = ViT(
                in_channels=len(args.modalities),
                img_size=(args.img_size, args.img_size, args.img_size),
                patch_size=(args.patch_size, args.patch_size, args.patch_size),
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_layers=12,
                num_heads=args.num_heads,
                dropout_rate=args.dropout_rate,
                classification=False,
                qkv_bias=True,
            )

            vit.load_state_dict(state_dict, strict=True)

            model.vit = vit

    print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    device = "cuda"
    model = model.to(device)

    loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_func = lambda epoch: min(
        (epoch + 1) / (args.warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / args.scaling_factor * math.pi) + 1)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    # path_to_save = f"pretrained_models/{args.pretraining}/seg_{'-'.join(args.modalities)}_{args.from_scratch}.pth"
    path_to_save = "trolol.pth"
    _, _ = loop_segmentation(
        epochs=args.epochs,
        model=model,
        loss_fn=loss_fn,
        opt=optimizer,
        scheduler=lr_scheduler,
        train_dl=train_loader,
        valid_dl=val_loader,
        device=device,
        path_to_save=path_to_save,
        wandb_logging=wandb_logging,
    )
    if wandb_logging:
        run.finish()
