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
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

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
        data_list = glob.glob("/home/lchalcroft/Data/IXI/guys/t1/preprocessed/p_IXI*-T1.nii.gz")
    elif modality == "t2":
        data_list = glob.glob("/home/lchalcroft/Data/IXI/guys/t2/preprocessed/p_IXI*-T2.nii.gz")
    elif modality == "pd":
        data_list = glob.glob("/home/lchalcroft/Data/IXI/guys/pd/preprocessed/p_IXI*-PD.nii.gz")

    # Load IXI spreadsheet
    ixi_data = pd.read_excel('/home/lchalcroft/Data/IXI/IXI.xls')

    # Load and prepare data
    data_list.sort()
    
    # Sort and split data
    total_samples = len(data_list)
    train_size = int(0.7 * total_samples)
    val_size = int(0.1 * total_samples)

    train_imgs = data_list[:train_size]
    val_imgs = data_list[train_size:train_size+val_size]
    
    train_ids = [int(os.path.basename(f).split("-")[0][5:]) for f in train_imgs]
    val_ids = [int(os.path.basename(f).split("-")[0][5:]) for f in val_imgs]

    # Use ixi_data to find all subjects with a given age
    train_ages = []
    train_genders = []
    train_imgs_new = []
    train_ids_new = []
    for i, id in enumerate(train_ids):
        age = ixi_data.loc[ixi_data["IXI_ID"] == id]["AGE"].values
        if len(age) == 0:
            train_ids.pop(i)
            train_imgs.pop(i)
        elif np.isnan(age[0]):
            train_ids.pop(i)
            train_imgs.pop(i)
        else:
            train_ages.append(age[0])
            train_genders.append(ixi_data.loc[ixi_data["IXI_ID"] == id]["SEX_ID (1=m, 2=f)"].values[0])
            train_imgs_new.append(train_imgs[i])
            train_ids_new.append(train_ids[i])
    val_ages = []
    val_genders = []
    val_imgs_new = []
    val_ids_new = []
    for i, id in enumerate(val_ids):
        age = ixi_data.loc[ixi_data["IXI_ID"] == id]["AGE"].values
        if len(age) == 0:
            val_ids.pop(i)
            val_imgs.pop(i)
        elif np.isnan(age[0]):
            val_ids.pop(i)
            val_imgs.pop(i)
        else:
            val_ages.append(age[0])
            val_genders.append(ixi_data.loc[ixi_data["IXI_ID"] == id]["SEX_ID (1=m, 2=f)"].values[0])
            val_imgs_new.append(val_imgs[i])
            val_ids_new.append(val_ids[i])

    train_imgs = train_imgs_new
    train_ids = train_ids_new
    val_imgs = val_imgs_new
    val_ids = val_ids_new

    train_dict = [{"image": f, "label": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")], "age": train_ages[i], "gender": train_genders[i]} for i, f in enumerate(train_imgs)]
    val_dict = [{"image": f, "label": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")], "age": val_ages[i], "gender": val_genders[i]} for i, f in enumerate(val_imgs)]

    if pc_data < 100:
        train_dict = train_dict[:int(len(train_dict) * pc_data / 100)]

    data_transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(
            keys=["image", "label"], image_only=True, allow_missing_keys=True
        ),
        mn.transforms.EnsureChannelFirstD(
            keys=["image", "label"], allow_missing_keys=True
        ),
        mn.transforms.LambdaD(
            keys=["label"],
            func=add_bg,
            allow_missing_keys=True,
        ),
        mn.transforms.OrientationD(
            keys=["image", "label"], axcodes="RAS", allow_missing_keys=True
        ),
        mn.transforms.SpacingD(
            keys=["image", "label"],
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
            keys=["image", "label"],
            allow_missing_keys=True,
        ),
        mn.transforms.LambdaD(
            keys=["image", "label"],
            func=mn.transforms.SignalFillEmpty(),
            allow_missing_keys=True,
        ),
        mn.transforms.RandAffineD(
            keys=["image", "label"],
            rotate_range=15, shear_range=0.012, scale_range=0.15,
            prob=0.8, 
            # cache_grid=True, spatial_size=(256, 256, 256) if not lowres else (128, 128, 128),
            allow_missing_keys=True,
        ),
        mn.transforms.RandBiasFieldD(keys="image", prob=0.8),
        mn.transforms.RandAxisFlipd(
            keys=["image", "label"],
            prob=0.8,
            allow_missing_keys=True,
        ),
        mn.transforms.RandAxisFlipd(
            keys=["image", "label"],
            prob=0.8,
            allow_missing_keys=True,
        ),
        mn.transforms.RandAxisFlipd(
            keys=["image", "label"],
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
        mn.transforms.RandCropByLabelClassesD(
            keys=["image", "label"],
            spatial_size=(96, 96, 96) if not lowres else (48, 48, 48),
            label_key="label",
            num_samples=1,
            ratios=[1, 5, 5, 5],
            allow_missing_keys=True,
        ),
        mn.transforms.ResizeD(
            keys=["image", "label"],
            spatial_size=(ptch, ptch, ptch),
            allow_missing_keys=True,
        ),
        mn.transforms.ToTensorD(
            dtype=torch.float32, keys=["image", "label", "age", "gender"]
        ),
    ])

    train_data = mn.data.Dataset(train_dict, transform=data_transforms)
    val_data = mn.data.Dataset(val_dict, transform=data_transforms)

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

def run_model(args, device, train_loader, val_loader):
    
    if args.net == "cnn":
        encoder = model.CNNEncoder(
            spatial_dims=3, 
            in_channels=1, 
            features=(64, 128, 256, 512, 768), 
            act="GELU", 
            norm="instance", 
            bias=True, 
            dropout=0.2
        ).to(device)
    elif args.net == "vit":
        encoder = model.ViTEncoder(
            spatial_dims=3,
            in_channels=1,
            img_size=(48 if args.lowres else 96),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            dropout_rate=0.2,
            qkv_bias=True,
            save_attn=False
        ).to(device)

    # Load backbone weights
    checkpoint = torch.load(args.backbone_weights, map_location=device)
    print(f"\nLoading encoder weights from {args.backbone_weights}")
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
        
    class BinaryClassifier(torch.nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_features, 2),  # Output 2 classes for male/female
            )

        def forward(self, x):
            return self.net(x)

    classifier = BinaryClassifier(768).to(device)

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
    wandb.watch(classifier)

    crit = torch.nn.CrossEntropyLoss()

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

    params = list(classifier.parameters())
    params += list(encoder.parameters())
    try:
        opt = torch.optim.AdamW(params, args.lr, fused=torch.cuda.is_available())
    except:
        opt = torch.optim.AdamW(params, args.lr, fused=False)
        
    # Try to load most recent weight
    if args.resume or args.resume_best:
        classifier.load_state_dict(
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
        metric_best = 1e6

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
        encoder.train()
        classifier.train()
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
            img = batch[0]["image"].to(device).float()
            gender = batch[0]["gender"].to(device)
            # Convert gender from 1/2 to 0/1 for binary classification
            gender = (gender - 1).long()
            
            opt.zero_grad(set_to_none=True)

            with ctx:
                features = encoder(img)
                features = features.view(features.shape[0], features.shape[1], -1).mean(dim=-1)
                pred_sex = classifier(features)
                
                loss = crit(pred_sex, gender)

            if type(loss) == float or loss.isnan().sum() != 0:
                print("NaN found in loss!")
                step_deficit += 1
            else:
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 12)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 12)
                    opt.step()

                epoch_loss += loss.item()

                wandb.log({"train/loss": loss.item()})

            progress_bar.set_postfix(
                {"loss": epoch_loss / (step + 1 - step_deficit)}
            )
        wandb.log({"train/lr": opt.param_groups[0]["lr"]})
        lr_scheduler.step()

        if (epoch + 1) % args.val_interval == 0:
            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                val_loss = 0
                val_acc = 0  # Accuracy for sex classification
                for i, batch in enumerate(val_loader):
                    img = batch[0]["image"].to(device).float()
                    gender = (batch[0]["gender"].to(device) - 1).long()
                    
                    features = encoder(img)
                    features = features.view(features.shape[0], features.shape[1], -1).mean(dim=-1)
                    pred_sex = classifier(features)
                    val_loss += crit(pred_sex, gender).item()
                    
                    # Calculate accuracy
                    pred_class = pred_sex.softmax(dim=1).argmax(dim=1)
                    val_acc += (pred_class == gender).float().mean().item()

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)
                
                wandb.log({
                    "val/loss": val_loss, 
                    "val/accuracy": val_acc
                })

            if val_loss < metric_best:
                metric_best = val_loss
                torch.save(
                    {
                        "model": classifier.state_dict(),
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
                "model": classifier.state_dict(),
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
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
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
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--resume_best", default=False, action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Device to use. If not specified then will check for CUDA.")
    parser.add_argument("--lowres", default=False, action="store_true", help="Train with 2mm resolution images.")
    parser.add_argument("--debug", default=False, action="store_true", help="Save sample images before training.")
    parser.add_argument("--backbone_weights", type=str, default=None, help="Path to encoder weights to load.")
    parser.add_argument("--pc_data", default=100, type=float, help="Percentage of data to use for training.")
    parser.add_argument("--modality", type=str, choices=["t1", "t2", "pd"], help="Modality to train on.")
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
    train_loader, val_loader = get_loaders(
        modality=args.modality,
        batch_size=args.batch_size,
        device=device,
        lowres=args.lowres,
        ptch=48 if args.lowres else 96,
        pc_data=args.pc_data
    )

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
                    batch[0]["label"].shape,
                    "min={}".format(batch[0]["label"].min()),
                    "max={}".format(batch[0]["label"].max()),
                )
                saver2(
                    torch.Tensor(batch[0]["label"][0].argmax(dim=0, keepdim=True).cpu().float())
                )

    return args, device, train_loader, val_loader


def main():
    args, device, train_loader, val_loader = set_up()
    run_model(args, device, train_loader, val_loader)


if __name__ == "__main__":
    main()
