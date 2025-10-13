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
    data_dict,
    lowres=False,
):
    print(f"data_dict: {len(data_dict)}")

    test_transform = mn.transforms.Compose(
        transforms=[
            mn.transforms.LoadImageD(
                keys=["image"], image_only=True
            ),
            mn.transforms.EnsureChannelFirstD(
                keys=["image"]
            ),
            mn.transforms.OrientationD(
                keys=["image"], axcodes="RAS"
            ),
            mn.transforms.SpacingD(
                keys=["image"],
                pixdim=1 if not lowres else 2,
            ),
            mn.transforms.CropForegroundD(
                keys=["image"],
                source_key="image",
            ),
            mn.transforms.ToTensorD(
                dtype=float,
                keys=["image"],
            ),
            mn.transforms.LambdaD(
                keys=["image"],
                func=mn.transforms.SignalFillEmpty(),
            ),
            mn.transforms.ScaleIntensityRangePercentilesD(
                keys=["image"],
                lower=0.5, upper=99.5, b_min=0, b_max=1,
                clip=True, channel_wise=True,
            ),
            mn.transforms.HistogramNormalizeD(keys="image", min=0, max=1),
            mn.transforms.NormalizeIntensityD(
                keys="image", nonzero=False, channel_wise=True
            ),
            mn.transforms.CopyItemsD(keys=["image"], names=["noisy_image"]),
            mn.transforms.RandGaussianNoiseD(keys="noisy_image", prob=1.0, mean=0.0, std=0.2, sample_std=False),
            mn.transforms.ToTensorD(keys=["image", "noisy_image"], dtype=torch.float32),
        ]
    )

    data = mn.data.Dataset(data_dict, transform=test_transform)

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )
    return loader

def run_model(args, device):
    
    if args.net == "cnn":
        net = model.CNNUNet(
            spatial_dims=3, 
            in_channels=1,
            out_channels=1,
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
            out_channels=1,
            img_size=(96 if args.lowres else 192),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            dropout_rate=0.2,
            qkv_bias=True,
            save_attn=False,
        ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    print(
        "\nResuming from epoch #{} with WandB ID {}".format(
            checkpoint["epoch"], checkpoint["wandb"]
        )
    )
    print()

    net.load_state_dict(checkpoint["model"], strict=False)
    net.eval()

    odir = os.path.dirname(args.weights)
    
    # Generate training data for the guys t1 modality
    # Load and prepare data
    guys_t1_img_list = glob.glob("/home/lchalcroft/Data/IXI/guys/t1/preprocessed/p_IXI*-T1.nii.gz")    
    guys_t1_img_list.sort()
    total_samples = len(guys_t1_img_list)
    train_size = int(0.7 * total_samples)
    val_size = int(0.1 * total_samples)
    guys_t1_img_list = guys_t1_img_list[train_size+val_size:]

    guys_t2_img_list = glob.glob("/home/lchalcroft/Data/IXI/guys/t2/preprocessed/p_IXI*-T2.nii.gz")
    guys_t2_img_list.sort()
    total_samples = len(guys_t2_img_list)
    train_size = int(0.7 * total_samples)
    val_size = int(0.1 * total_samples)
    guys_t2_img_list = guys_t2_img_list[train_size+val_size:]

    guys_pd_img_list = glob.glob("/home/lchalcroft/Data/IXI/guys/pd/preprocessed/p_IXI*-PD.nii.gz")
    guys_pd_img_list.sort()
    total_samples = len(guys_pd_img_list)
    train_size = int(0.7 * total_samples)
    val_size = int(0.1 * total_samples)
    guys_pd_img_list = guys_pd_img_list[train_size+val_size:]

    # Other site data doesn't need to be split
    hh_t1_img_list = glob.glob("/home/lchalcroft/Data/IXI/hh/t1/preprocessed/p_IXI*-T1.nii.gz")
    hh_t2_img_list = glob.glob("/home/lchalcroft/Data/IXI/hh/t2/preprocessed/p_IXI*-T2.nii.gz")
    hh_pd_img_list = glob.glob("/home/lchalcroft/Data/IXI/hh/pd/preprocessed/p_IXI*-PD.nii.gz")
    iop_t1_img_list = glob.glob("/home/lchalcroft/Data/IXI/iop/t1/preprocessed/p_IXI*-T1.nii.gz")
    iop_t2_img_list = glob.glob("/home/lchalcroft/Data/IXI/iop/t2/preprocessed/p_IXI*-T2.nii.gz")
    iop_pd_img_list = glob.glob("/home/lchalcroft/Data/IXI/iop/pd/preprocessed/p_IXI*-PD.nii.gz")

    guys_t1_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "site": "guys",
            "modality": "T1w",
            "IXI_ID": int(os.path.basename(f).split("-")[0][5:])
        }
        for f in guys_t1_img_list
    ]
    guys_t2_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "site": "guys",
            "modality": "T2w",
            "IXI_ID": int(os.path.basename(f).split("-")[0][5:])
        }
        for f in guys_t2_img_list
    ]
    guys_pd_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "site": "guys",
            "modality": "PDw",
            "IXI_ID": int(os.path.basename(f).split("-")[0][5:])
        }
        for f in guys_pd_img_list
    ]
    hh_t1_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "site": "hh",
            "modality": "T1w",
        }
        for f in hh_t1_img_list
    ]
    hh_t2_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "site": "hh",
            "modality": "T2w",
        }
        for f in hh_t2_img_list
    ]
    hh_pd_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "site": "hh",
            "modality": "PDw",
        }
        for f in hh_pd_img_list
    ]
    iop_t1_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "site": "iop",
            "modality": "T1w",
        }
        for f in iop_t1_img_list
    ]
    iop_t2_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "site": "iop",
            "modality": "T2w",
        }
        for f in iop_t2_img_list
    ]
    iop_pd_test_dict = [
        {
            "image": f,
            "file": f,
            "seg": [f.replace("p_IXI", "c1p_IXI"), f.replace("p_IXI", "c2p_IXI"), f.replace("p_IXI", "c3p_IXI")],
            "dataset": "IXI",
            "site": "iop",
            "modality": "PDw",
        }
        for f in iop_pd_img_list
    ]
    if args.modality == "t1":
        test_dict = guys_t1_test_dict + hh_t1_test_dict + iop_t1_test_dict
    elif args.modality == "t2":
        test_dict = guys_t2_test_dict + hh_t2_test_dict + iop_t2_test_dict
    elif args.modality == "pd":
        test_dict = guys_pd_test_dict + hh_pd_test_dict + iop_pd_test_dict
    test_loader = get_loaders(test_dict, lowres=args.lowres)
    
    window = mn.inferers.SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=2, overlap=0.5, mode="gaussian")

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", total=len(test_loader)):
            image = batch["image"].to(device)
            noisy_image = batch["noisy_image"].to(device)
            
            # Run inference with sliding window
            with torch.cuda.amp.autocast() if args.amp else nullcontext():
                pred_noise = window(noisy_image, net)
                denoised_image = noisy_image - pred_noise
            
            # Calculate MSE
            mse = torch.nn.functional.mse_loss(denoised_image, image).item()
            # Calculate PSNR
            psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse)))
            
            results.append({
                'file': batch["file"][0],
                'dataset': batch["dataset"][0],
                'modality': batch["modality"][0],
                'mse': mse,
                'psnr': psnr.item(),
                'site': batch["site"][0]
            })

            # Convert results to DataFrame and save
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(odir, 'test_results.csv'), index=False)
    
    # Print summary statistics
    print("\nResults Summary:")
    print(df.groupby(['dataset', 'modality', 'site'])[['mse', 'psnr']].mean())

def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weights", type=str, help="Path to model weights.")
    parser.add_argument(
        "--net", 
        type=str, 
        help="Encoder network to use. Options: [cnn, vit]. Defaults to cnn.", 
        choices=["cnn", "vit"],
        default="cnn"
    )
    parser.add_argument("--modality", type=str, help="Modality to test. Options: [t1, t2, pd]. Defaults to t1.")
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Device to use. If not specified then will check for CUDA.")
    parser.add_argument("--lowres", default=False, action="store_true", help="Train with 2mm resolution images.")
    args = parser.parse_args()

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
    return args, device


def main():
    args, device = set_up()
    run_model(args, device)


if __name__ == "__main__":
    main()
