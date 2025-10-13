import glob
import os
import model
import torch
import wandb
import logging
import argparse
import monai as mn
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

logging.getLogger("monai").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings(
    "ignore",
    ".*pixdim*.",
)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

def add_bg(x):
    return torch.cat([1-x.sum(dim=0, keepdim=True), x], dim=0)

def get_loaders(
    modality,
    batch_size=1,
    device="cpu",
    lowres=False,
    ptch=128,
    pc_data=100,
):
    print(f"Modality: {modality}")

    if modality == "t1":
        data_list = glob.glob("/home/lchalcroft/Data/ARC/PREPROC/sub-*/ses-*/*_T1w_flirt.nii.gz")
        data_list.sort()
        data_dict = [{"image": f, "file": f, "seg": glob.glob(os.path.join(os.path.dirname(f), "*_T2w_desc-lesion_mask.nii.gz"))[0], "dataset": "ARC", "modality": "T1w"} for f in data_list]
    elif modality == "t2":
        data_list = glob.glob("/home/lchalcroft/Data/ARC/PREPROC/sub-*/ses-*/*_T2w.nii.gz")
        data_list.sort()
        data_dict = [{"image": f, "file": f, "seg": glob.glob(os.path.join(os.path.dirname(f), "*_T2w_desc-lesion_mask.nii.gz"))[0], "dataset": "ARC", "modality": "T2w"} for f in data_list]
    elif modality == "flair":
        data_list = glob.glob("/home/lchalcroft/Data/ARC/PREPROC/sub-*/ses-*/*_FLAIR_flirt.nii.gz")
        data_list.sort()
        data_dict = [{"image": f, "file": f, "seg": glob.glob(os.path.join(os.path.dirname(f), "*_T2w_desc-lesion_mask.nii.gz"))[0], "dataset": "ARC", "modality": "FLAIR"} for f in data_list]

    total_samples = len(data_dict)
    train_size = int(0.7 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    train_dict = data_dict[:train_size]
    val_dict = data_dict[train_size:train_size+val_size]
    test_dict = data_dict[train_size+val_size:]

    if pc_data < 100:
        n_samples = max(1, int(len(train_dict) * pc_data / 100))
        train_dict = train_dict[:n_samples]

    print(f"Using {len(train_dict)} training samples, {len(val_dict)} validation samples, and {len(test_dict)} test samples")

    train_transform = mn.transforms.Compose(
        transforms=[
            mn.transforms.LoadImageD(
                keys=["image", "seg"], image_only=True, allow_missing_keys=True
            ),
            mn.transforms.EnsureChannelFirstD(
                keys=["image", "seg"], allow_missing_keys=True
            ),
            mn.transforms.LambdaD(keys="seg", func=add_bg),
            mn.transforms.OrientationD(
                keys=["image", "seg"], axcodes="RAS", allow_missing_keys=True
            ),
            mn.transforms.SpacingD(
                keys=["image", "seg"],
                pixdim=1 if not lowres else 2,
                allow_missing_keys=True,
            ),
            # mn.transforms.ResizeWithPadOrCropD(
            #     keys=["image", "seg"],
            #     spatial_size=(256, 256, 256) if not lowres else (128, 128, 128),
            #     allow_missing_keys=True,
            # ),
            mn.transforms.ToTensorD(
                dtype=float,
                keys=["image", "seg"],
                allow_missing_keys=True,
            ),
            mn.transforms.LambdaD(
                keys=["image", "seg"],
                func=mn.transforms.SignalFillEmpty(),
                allow_missing_keys=True,
            ),
            mn.transforms.RandAffineD(
                keys=["image", "seg"],
                rotate_range=15, shear_range=0.012, scale_range=0.15,
                prob=0.8, 
                # cache_grid=True, spatial_size=(256, 256, 256) if not lowres else (128, 128, 128),
                allow_missing_keys=True,
            ),
            mn.transforms.RandBiasFieldD(keys="image", prob=0.8),
            mn.transforms.RandAxisFlipd(
                keys=["image", "seg"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.RandAxisFlipd(
                keys=["image", "seg"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.RandAxisFlipd(
                keys=["image", "seg"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.ScaleIntensityRangePercentilesD(
                keys=["image"],
                lower=0.5, upper=99.5, b_min=0, b_max=1,
                clip=True, channel_wise=True,
            ),
            mn.transforms.HistogramNormalizeD(keys="image", min=0, max=1, allow_missing_keys=True),
            mn.transforms.NormalizeIntensityD(
                keys="image", nonzero=False, channel_wise=True
            ),
            mn.transforms.RandGaussianNoiseD(keys="image", prob=0.8),
            mn.transforms.RandCropByLabelClassesD(
                keys=["image", "seg"],
                spatial_size=(96, 96, 96) if not lowres else (48, 48, 48),
                label_key="seg",
                num_samples=1,
                ratios=[1, 6],
                allow_missing_keys=True,
            ),
            mn.transforms.ResizeD(
                keys=["image", "seg"],
                spatial_size=(ptch, ptch, ptch),
                allow_missing_keys=True,
            ),
            mn.transforms.ThresholdIntensityD(
                keys=["seg"],
                threshold=0.5,
                above=True,
                cval=0,
                allow_missing_keys=True,
            ),
            mn.transforms.ToTensorD(keys=["image", "seg"], dtype=torch.float32),
        ]
    )

    train_data = mn.data.Dataset(train_dict, transform=train_transform)
    val_data = mn.data.Dataset(val_dict, transform=train_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=8,
    )

    return train_loader, val_loader


def compute_dice(y_pred, y, eps=1e-8):
    y_pred = torch.flatten(y_pred)
    y = torch.flatten(y)
    y = y.float()
    intersect = (y_pred * y).sum(-1)
    denominator = (y_pred * y_pred).sum(-1) + (y * y).sum(-1)
    return 2 * (intersect / denominator.clamp(min=eps))

def run_model(args, device, train_loader, val_loader):
    
    if args.net == "cnn":
        net = model.CNNUNet(
            spatial_dims=3, 
            in_channels=1,
            out_channels=2,
            features=(64, 128, 256, 512, 768, 32),
            act="GELU", 
            norm="instance", 
            bias=True, 
            dropout=0.2,
            upsample="deconv",
        ).to(device)
    elif args.net == "vit":
        net = model.ViTUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            img_size=(96 if args.lowres else 192),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            dropout_rate=0.2,
            qkv_bias=True,
            save_attn=False,
        ).to(device)

    # Load backbone weights if provided
    # Any layers loaded from the backbone will be frozen during training
    cnt_frozen = 0
    cnt_trainable = 0
    if args.backbone_weights is not None:
        checkpoint = torch.load(args.backbone_weights, map_location=device)
        print(f"\nLoading encoder weights from {args.backbone_weights}")
        for name, param in net.named_parameters():
            if name in checkpoint["encoder"]:
                param.data = checkpoint["encoder"][name]
                param.requires_grad = False
                cnt_frozen += 1
            else:
                param.requires_grad = True
                cnt_trainable += 1
        print(f"Frozen layers: {cnt_frozen}, trainable layers: {cnt_trainable}")
    else:
        print("No backbone weights provided, all layers will be trainable.")

    if args.resume or args.resume_best:
        ckpts = glob.glob(
            os.path.join(
                args.logdir,
                args.name,
                "checkpoint.pt" if args.resume else "checkpoint_best.pt",
            )
        )
        if len(ckpts) == 0:
            args.resume = False
            print("\nNo checkpoints found. Beginning from epoch #0")
        else:
            checkpoint = torch.load(ckpts[0], map_location=device)
            print(
                "\nResuming from epoch #{} with WandB ID {}".format(
                    checkpoint["epoch"], checkpoint["wandb"]
                )
            )
    print()

    wandb.init(
        project="contrast-squared",
        entity="atlas-ploras",
        save_code=True,
        name=args.name,
        settings=wandb.Settings(start_method="fork"),
        resume="must" if args.resume else None,
        id=checkpoint["wandb"] if args.resume or args.resume_best else None,
    )
    if not args.resume and not args.resume_best:
        wandb.config.update(args)
    wandb.watch(net)

    crit = mn.losses.DiceCELoss(
        include_background=False,
        to_onehot_y=False,
        sigmoid=False,
        softmax=True,
        other_act=None,
        squared_pred=False,
        jaccard=False,
        reduction="mean",
        smooth_nr=1e-05,
        smooth_dr=1e-05,
        batch=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )

    class WandBID:
        def __init__(self, wandb_id):
            self.wandb_id = wandb_id

        def state_dict(self):
            return self.wandb_id

    class Epoch:
        def __init__(self, epoch):
            self.epoch = epoch

        def state_dict(self):
            return self.epoch

    class Metric:
        def __init__(self, metric):
            self.metric = metric

        def state_dict(self):
            return self.metric

    params = list(net.parameters())
    try:
        opt = torch.optim.AdamW(params, args.lr, fused=torch.cuda.is_available())
    except:
        opt = torch.optim.AdamW(params, args.lr, fused=False)
        
    # Try to load most recent weight
    if args.resume or args.resume_best:
        net.load_state_dict(
            checkpoint["model"], strict=False
        )
        opt.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint["epoch"] + 1
        metric_best = checkpoint["metric"]
        # correct scheduler in cases where max epochs has changed
        def lambda1(epoch):
            return (1 - (epoch + start_epoch) / args.epochs) ** 0.9

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda1])
    else:
        start_epoch = 0
        metric_best = 0

        def lambda1(epoch):
            return (1 - (epoch) / args.epochs) ** 0.9

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda1])

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)

    train_iter = None
    for epoch in range(start_epoch, args.epochs):
        if args.debug:
            saver1 = mn.transforms.SaveImage(
                output_dir=os.path.join(args.logdir, args.name, "debug-train"),
                output_postfix="img",
                separate_folder=False,
                print_log=False,
            )
            saver2 = mn.transforms.SaveImage(
                output_dir=os.path.join(args.logdir, args.name, "debug-train"),
                output_postfix="seg",
                separate_folder=False,
                print_log=False,
            )
        net.train()
        epoch_loss = 0
        step_deficit = -1e-7
        if args.amp:
            ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
            scaler = torch.cuda.amp.GradScaler()
        else:
            ctx = nullcontext()
        progress_bar = tqdm(range(args.epoch_length), total=args.epoch_length, ncols=90)
        progress_bar.set_description(f"Epoch {epoch}")
        if train_iter is None:
            train_iter = iter(train_loader)

        for step in progress_bar:
            try:
                batch = next(train_iter)
            except:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            img = batch[0]["image"].to(device)
            seg = batch[0]["seg"].to(device)
            opt.zero_grad(set_to_none=True)

            if args.debug and step < 5:
                saver1(torch.Tensor(img[0].cpu().float()))
                saver2(torch.Tensor(seg[0].cpu().float()))

            with ctx:
                logits = net(img)
                loss = crit(logits, seg)

            if type(loss) == float or loss.isnan().sum() != 0:
                print("NaN found in loss!")
                step_deficit += 1
            else:
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
                    opt.step()

                epoch_loss += loss.item()

                wandb.log({"train/loss": loss.item()})

            progress_bar.set_postfix(
                {"loss": epoch_loss / (step + 1 - step_deficit)}
            )
        wandb.log({"train/lr": opt.param_groups[0]["lr"]})
        lr_scheduler.step()

        if (epoch + 1) % args.val_interval == 0:
            img_list = []
            seg_list = []
            net.eval()
            with torch.no_grad():
                val_loss = 0
                val_dice = 0
                for i, batch in enumerate(val_loader):
                    img = batch[0]["image"].to(device)
                    seg = batch[0]["seg"].to(device)
                    logits = net(img)
                    loss = crit(logits, seg)
                    val_loss += loss.item()
                    probs = logits.softmax(dim=1)
                    val_dice += compute_dice(probs, seg).item()

                    if i < 16:
                        img_list.append(img[0,...,img.shape[-1]//2])
                        seg_list.append(probs[0,...,seg.shape[-1]//2])
                    elif i == 16:
                        grid_image1 = make_grid(
                                    img_list,
                                    nrow=int(4),
                                    padding=5,
                                    normalize=True,
                                    scale_each=True,
                                )
                        grid_image2 = make_grid(
                                    seg_list,
                                    nrow=int(4),
                                    padding=5,
                                    normalize=True,
                                    scale_each=True,
                                )
                        wandb.log(
                            {
                                "examples": [
                                    wandb.Image(
                                        grid_image1[0].cpu().numpy(), caption="Images"
                                    ),
                                    wandb.Image(
                                        grid_image2[1].cpu().numpy(), caption="Segmentations"
                                    ),
                                ]
                            }
                        )
            
            val_loss /= len(val_loader)
            val_dice /= len(val_loader)

            wandb.log({"val/loss": val_loss, "val/dice": val_dice})


            if val_dice > metric_best:
                metric_best = val_dice
                torch.save(
                    {
                        "model": net.state_dict(),
                        "opt": opt.state_dict(),
                        "lr": lr_scheduler.state_dict(),
                        "wandb": WandBID(wandb.run.id).state_dict(),
                        "epoch": Epoch(epoch).state_dict(),
                        "metric": Metric(metric_best).state_dict(),
                    },
                    os.path.join(args.logdir, args.name, "checkpoint_best.pt"),
                )
        torch.save(
            {
                "model": net.state_dict(),
                "opt": opt.state_dict(),
                "lr": lr_scheduler.state_dict(),
                "wandb": WandBID(wandb.run.id).state_dict(),
                "epoch": Epoch(epoch).state_dict(),
                "metric": Metric(metric_best).state_dict(),
            },
            os.path.join(args.logdir, args.name, "checkpoint.pt"),
        )


def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="Name of WandB run.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs for training.")
    parser.add_argument("--epoch_length", type=int, default=200, help="Number of iterations per epoch.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--val_interval", type=int, default=2, help="Validation interval.")
    parser.add_argument("--batch_size", type=int, default=2, help="Number of subjects to use per batch.")
    parser.add_argument(
        "--net", 
        type=str, 
        help="Encoder network to use. Options: [cnn, vit]. Defaults to cnn.", 
        choices=["cnn", "vit"],
        default="cnn"
    )
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--logdir", type=str, default="./", help="Path to saved outputs")
    parser.add_argument("--modality", type=str, choices=["t1", "t2", "flair"], help="Modality to train on.")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--resume_best", default=False, action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Device to use. If not specified then will check for CUDA.")
    parser.add_argument("--lowres", default=False, action="store_true", help="Train with 2mm resolution images.")
    parser.add_argument("--debug", default=False, action="store_true", help="Save sample images before training.")
    parser.add_argument("--backbone_weights", type=str, default=None, help="Path to encoder weights to load.")
    parser.add_argument("--pc_data", default=100, type=float, help="Percentage of data to use for training.")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)
    print()
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    train_loader, val_loader = get_loaders(modality=args.modality, batch_size=args.batch_size, device=device, lowres=args.lowres, ptch=48 if args.lowres else 96, pc_data=args.pc_data)

    if args.debug:
        saver1 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="img",
            separate_folder=False,
            print_log=False,
        )
        saver2 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="seg",
            separate_folder=False,
            print_log=False,
        )
        for i, batch in enumerate(train_loader):
            if i > 5:
                break
            else:
                print(
                    "Image: ",
                    batch[0]["image"].shape,
                    "min={}".format(batch[0]["image"].min()),
                    "max={}".format(batch[0]["image"].max()),
                )
                saver1(
                    torch.Tensor(batch[0]["image"][0].cpu().float()),
                )
                print(
                    "Segmentation: ",
                    batch[0]["seg"].shape,
                    "min={}".format(batch[0]["seg"].min()),
                    "max={}".format(batch[0]["seg"].max()),
                )
                saver2(
                    torch.Tensor(batch[0]["seg"][0].cpu().float())
                )

    return args, device, train_loader, val_loader


def main():
    args, device, train_loader, val_loader = set_up()
    run_model(args, device, train_loader, val_loader)


if __name__ == "__main__":
    main()
