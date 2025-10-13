import glob
import os
import model
import torch
import logging
import argparse
import monai as mn
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd
import warnings

logging.getLogger("monai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


def get_loaders(data_dict, lowres=False):
    print(f"data_dict: {len(data_dict)}")

    test_transform = mn.transforms.Compose(
        transforms=[
            mn.transforms.LoadImageD(keys=["image"], image_only=True),
            mn.transforms.EnsureChannelFirstD(keys=["image"]),
            mn.transforms.OrientationD(keys=["image"], axcodes="RAS"),
            mn.transforms.SpacingD(
                keys=["image"],
                pixdim=1 if not lowres else 2,
            ),
            mn.transforms.CropForegroundD(keys=["image"], source_key="image"),
            mn.transforms.ToTensorD(keys=["image"], dtype=torch.float32),
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
            mn.transforms.CenterSpatialCropD(keys=["image"], roi_size=(48, 48, 48) if lowres else (96, 96, 96)),
            mn.transforms.ToTensorD(keys=["image"], dtype=torch.float32),
        ]
    )

    data = mn.data.Dataset(data_dict, transform=test_transform)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8)
    return loader

def run_model(args, device):
    if args.net == "cnn":
        net = model.CNNEncoder(
            spatial_dims=3,
            in_channels=1,
            features=(64, 128, 256, 512, 768),
            act="GELU",
            norm="instance",
            bias=True,
            dropout=0.2,
        ).to(device)
    elif args.net == "vit":
        net = model.ViTEncoder(
            spatial_dims=3,
            in_channels=1,
            img_size=96,  # Always use 96 since we're center cropping
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            dropout_rate=0.2,
            qkv_bias=True,
            save_attn=False,
        ).to(device)

    class BinaryClassifier(torch.nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_features, 2),  # Output 2 classes for male/female
            )

        def forward(self, x):
            return self.net(x)

    classifier = BinaryClassifier(768).to(device)

    checkpoint = torch.load(args.encoder_weights, map_location=device)
    print(
        "\nResuming encoder from epoch #{} with WandB ID {}".format(
            checkpoint["epoch"], checkpoint["wandb"]
        )
    )
    print()

    net.load_state_dict(checkpoint["encoder"], strict=True)
    net.eval()

    checkpoint = torch.load(args.classifier_weights, map_location=device)
    print(
        "\nResuming classifier from epoch #{} with WandB ID {}".format(
            checkpoint["epoch"], checkpoint["wandb"]
        )
    )
    print()
    classifier.load_state_dict(checkpoint["model"], strict=True)
    classifier.eval()

    odir = os.path.dirname(args.classifier_weights)

    # Load IXI data
    ixi_data = pd.read_excel('/home/lchalcroft/Data/IXI/IXI.xls')
    
    # Load and prepare data based on modality
    guys_img_list = glob.glob(f"/home/lchalcroft/Data/IXI/guys/{args.modality}/preprocessed/p_IXI*-{args.modality.upper()}.nii.gz")
    guys_img_list.sort()
    
    # Split data following same ratios as training
    total_samples = len(guys_img_list)
    train_size = int(0.7 * total_samples)
    val_size = int(0.1 * total_samples)
    # Only use test portion
    guys_img_list = guys_img_list[train_size+val_size:]
    
    # Filter out subjects with missing/NaN age values (to match training)
    guys_test_dict = []
    for f in guys_img_list:
        ixi_id = int(os.path.basename(f).split("-")[0][5:])
        age = ixi_data.loc[ixi_data["IXI_ID"] == ixi_id]["AGE"].values
        if len(age) > 0 and not np.isnan(age[0]):
            guys_test_dict.append({
                "image": f,
                "file": f,
                "dataset": "IXI",
                "site": "guys",
                "modality": f"{args.modality.upper()}w",
                "IXI_ID": ixi_id,
                # Convert sex from 1/2 to 0/1 for binary classification
                "sex": ixi_data[ixi_data['IXI_ID'] == ixi_id]['SEX_ID (1=m, 2=f)'].iloc[0] - 1
            })

    # Do the same for other sites
    hh_img_list = glob.glob(f"/home/lchalcroft/Data/IXI/hh/{args.modality}/preprocessed/p_IXI*-{args.modality.upper()}.nii.gz")
    hh_img_list.sort()
    
    hh_test_dict = []
    for f in hh_img_list:
        ixi_id = int(os.path.basename(f).split("-")[0][5:])
        age = ixi_data.loc[ixi_data["IXI_ID"] == ixi_id]["AGE"].values
        if len(age) > 0 and not np.isnan(age[0]):
            hh_test_dict.append({
                "image": f,
                "file": f,
                "dataset": "IXI",
                "site": "hh",
                "modality": f"{args.modality.upper()}w",
                "IXI_ID": ixi_id,
                "sex": ixi_data[ixi_data['IXI_ID'] == ixi_id]['SEX_ID (1=m, 2=f)'].iloc[0] - 1
            })
    
    iop_img_list = glob.glob(f"/home/lchalcroft/Data/IXI/iop/{args.modality}/preprocessed/p_IXI*-{args.modality.upper()}.nii.gz")
    iop_img_list.sort()

    iop_test_dict = []
    for f in iop_img_list:
        ixi_id = int(os.path.basename(f).split("-")[0][5:])
        age = ixi_data.loc[ixi_data["IXI_ID"] == ixi_id]["AGE"].values
        if len(age) > 0 and not np.isnan(age[0]):
            iop_test_dict.append({
                "image": f,
                "file": f,
                "dataset": "IXI",
                "site": "iop",
                "modality": f"{args.modality.upper()}w",
                "IXI_ID": ixi_id,
                "sex": ixi_data[ixi_data['IXI_ID'] == ixi_id]['SEX_ID (1=m, 2=f)'].iloc[0] - 1
            })

    print(f"First subject in guys data: {guys_test_dict[0]['file']}")
    print(f"First subject in hh data: {hh_test_dict[0]['file']}")
    print(f"First subject in iop data: {iop_test_dict[0]['file']}")
    
    test_dict = guys_test_dict + hh_test_dict + iop_test_dict
    test_loader = get_loaders(test_dict, lowres=args.lowres)

    all_preds = []
    all_labels = []
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", total=len(test_loader)):
            images = batch["image"].to(device)
            labels = batch["sex"].to(device)
            
            with torch.cuda.amp.autocast() if args.amp else nullcontext():
                outputs = net(images)
                features = outputs.view(outputs.shape[0], outputs.shape[1], -1).mean(dim=-1)
                preds = classifier(features)
                probs = preds.softmax(dim=1)
                preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Add individual results
            results.append({
                'file': batch["file"][0],
                'dataset': batch["dataset"][0],
                'site': batch["site"][0],
                'modality': batch["modality"][0],
                'true_sex': labels.item(),
                'pred_sex': preds.item(),
                'correct': (labels == preds).item(),
                'prob_male': probs[:, 0].item(),
                'prob_female': probs[:, 1].item()
            })

    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # Calculate ROC curve metrics
    fpr, tpr, thresholds = roc_curve(all_labels, [r['prob_female'] for r in results])
    roc_auc = auc(fpr, tpr)

    # Save detailed results with probabilities
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(odir, f'sex_classification_results_detailed.csv'), index=False)

    # Save ROC curve data
    roc_df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    })
    roc_df.to_csv(os.path.join(odir, f'sex_classification_results_roc.csv'), index=False)

    # Save summary metrics
    summary = pd.DataFrame({
        "modality": [args.modality],
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1],
        "auc_roc": [roc_auc]
    })
    summary.to_csv(os.path.join(odir, f'sex_classification_results_summary.csv'), index=False)

    # Print summary statistics
    print(f"\nResults Summary for {args.modality}:")
    print(summary)
    print("\nResults by site:")
    print(df.groupby('site')['correct'].mean())

def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--encoder_weights", type=str, help="Path to encoder model weights.")
    parser.add_argument("--classifier_weights", type=str, help="Path to classifier model weights.")
    parser.add_argument(
        "--net", 
        type=str, 
        help="Network to use. Options: [cnn, vit]. Defaults to cnn.", 
        choices=["cnn", "vit"],
        default="cnn"
    )
    parser.add_argument(
        "--modality",
        type=str,
        help="Modality to test. Options: [t1, t2, pd].",
        choices=["t1", "t2", "pd"],
        required=True
    )
    parser.add_argument("--amp", default=False, action="store_true", help="Use automatic mixed precision.")
    parser.add_argument("--device", type=str, default=None, help="Device to use. If not specified then will check for CUDA.")
    parser.add_argument("--lowres", default=False, action="store_true", help="Use low resolution images.")
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)
    print()
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