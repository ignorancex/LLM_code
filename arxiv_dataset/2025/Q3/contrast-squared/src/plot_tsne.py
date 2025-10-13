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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

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
                keys=["image", "seg"], image_only=True, allow_missing_keys=True
            ),
            mn.transforms.EnsureChannelFirstD(
                keys=["image", "seg"], allow_missing_keys=True
            ),
            mn.transforms.LambdaD(keys="seg", func=add_bg, allow_missing_keys=True),
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
            mn.transforms.CropForegroundD(
                keys=["image", "seg"],
                source_key="image",
                allow_missing_keys=True,
            ),
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
            mn.transforms.ScaleIntensityRangePercentilesD(
                keys=["image"],
                lower=0.5, upper=99.5, b_min=0, b_max=1,
                clip=True, channel_wise=True, allow_missing_keys=True,
            ),
            mn.transforms.HistogramNormalizeD(keys="image", min=0, max=1, allow_missing_keys=True),
            mn.transforms.NormalizeIntensityD(
                keys="image", nonzero=False, channel_wise=True, allow_missing_keys=True
            ),
            mn.transforms.ToTensorD(keys=["image", "seg"], dtype=torch.float32, allow_missing_keys=True),
        ]
    )

    data = mn.data.Dataset(data_dict, transform=test_transform)

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=8,
    )
    return loader


def compute_dice(y_pred, y, eps=1e-8):
    y_pred = torch.flatten(y_pred)
    y = torch.flatten(y)
    y = y.float()
    intersect = (y_pred * y).sum(-1)
    denominator = (y_pred * y_pred).sum(-1) + (y * y).sum(-1)
    return 2 * (intersect / denominator.clamp(min=eps))

def run_model(args, device):
    # Load model
    if args.net == "cnn":
        net = model.CNNEncoder(
            spatial_dims=3, 
            in_channels=1,
            features=(64, 128, 256, 512, 768),
            act="GELU", 
            norm="instance", 
            bias=True,
        ).to(device)
    elif args.net == "vit":
        net = model.ViTEncoder(
            spatial_dims=3,
            in_channels=1,
            img_size=(96 if args.lowres else 192),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            qkv_bias=True,
            save_attn=False,
        ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    print(f"\nLoading weights from epoch #{checkpoint['epoch']}")
    net.load_state_dict(checkpoint["encoder"], strict=False)
    net.eval()

    # Check to see if features_list, site_labels, and modality_labels already exist
    features_list_path = os.path.join(os.path.dirname(args.weights), 'features_list.npy')
    site_labels_path = os.path.join(os.path.dirname(args.weights), 'site_labels.npy')
    modality_labels_path = os.path.join(os.path.dirname(args.weights), 'modality_labels.npy')
    if os.path.exists(features_list_path) and os.path.exists(site_labels_path) and os.path.exists(modality_labels_path):
        print("\nLoading existing features, site labels, and modality labels.")
        features_list = np.load(features_list_path)
        site_labels = np.load(site_labels_path)
        modality_labels = np.load(modality_labels_path)
    else:
        print("\nComputing features, site labels, and modality labels.")
        # Prepare data
        sites = ['guys', 'hh', 'iop']
        modalities = ['t1', 't2', 'pd']
        features_list = []
        site_labels = []
        modality_labels = []

        window = mn.inferers.SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=2, overlap=0.5, mode="gaussian")
        
        for site in sites:
            for modality in modalities:
                # Skip if combination doesn't exist
                data_path = f"/home/lchalcroft/Data/IXI/{site}/{modality}/preprocessed/p_IXI*-{modality.upper()}.nii.gz"
                files = glob.glob(data_path)
                if not files:
                    continue
                    
                print(f"\nProcessing {site} - {modality}: {len(files)} files")
                
                # Create data dictionary
                data_dict = [{"image": f} for f in files]
                
                # Create data loader
                loader = get_loaders(data_dict, lowres=args.lowres)
                
                # Extract features
                with torch.no_grad():
                    with torch.cuda.amp.autocast() if args.amp else nullcontext():
                        for batch in tqdm(loader, desc=f"{site}-{modality}"):
                            image = batch["image"].to(device)
                            
                            # Get encoder features
                            if args.net == "cnn":
                                features = window(image, net)  # Get last encoder layer features
                            else:  # ViT
                                features = window(image, net)  # Get transformer features
                            
                            # Global average pooling for CNN features
                            if args.net == "cnn":
                                features = torch.mean(features, dim=(2, 3, 4))
                            
                            features_list.append(features.cpu().numpy())
                            site_labels.extend([site] * features.shape[0])
                            modality_labels.extend([modality] * features.shape[0])

        # Save lists to disk
        np.save(features_list_path, features_list)
        np.save(site_labels_path, site_labels)
        np.save(modality_labels_path, modality_labels)

    # Concatenate all features
    features_array = np.concatenate(features_list, axis=0)

    # Compute t-SNE
    print("\nComputing t-SNE...")
    tsne = TSNE(n_components=2, random_state=33, perplexity=30)
    tsne_results = tsne.fit_transform(features_array)
    
    # Create plots
    fig_dir = os.path.join(os.path.dirname(args.weights), 'tsne_plots')
    os.makedirs(fig_dir, exist_ok=True)

    fname = os.path.dirname(args.weights).split('/')[-1]
        
    # Define name mappings
    SITE_NAMES = {
        'guys': 'GST',
        'hh': 'HH',
        'iop': 'IOP',
    }
    
    MODALITY_NAMES = {
        't1': 'T1w',
        't2': 'T2w',
        'pd': 'PDw',
    }
    
    # Map the labels to their standardized names
    site_labels = [SITE_NAMES.get(site, site) for site in site_labels]
    modality_labels = [MODALITY_NAMES.get(mod, mod) for mod in modality_labels]
    
    # Plot by site
    plt.figure(figsize=(10, 10))  # Square figure
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], 
                    hue=site_labels, style=modality_labels, s=100)  # Larger points
    plt.axis('off')  # Remove axes
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,
              fontsize=16, title_fontsize=18,  # Increased font sizes
              markerscale=2)  # Larger legend markers
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'tsne_by_site_{fname}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot by modality
    plt.figure(figsize=(10, 10))  # Square figure
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], 
                    hue=modality_labels, style=site_labels, s=100)  # Larger points
    plt.axis('off')  # Remove axes
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,
              fontsize=16, title_fontsize=18,  # Increased font sizes
              markerscale=2)  # Larger legend markers
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'tsne_by_modality_{fname}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nTSNE plots saved to: {fig_dir}")

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
