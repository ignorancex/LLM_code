import torch
from torch.utils.data import DataLoader
import torchmetrics
import segmentation_models_pytorch as smp
import numpy as np
from tqdm import tqdm
from torchmetrics.functional import dice

from preprocess import constants
from preprocess.augmentations import basic_transforms
from preprocess.dataset_test import TestDataset
from preprocess.dataset_val import ValDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(backbone, weights):
    # define model
    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights=None,
        in_channels=3,
        classes=5,
        activation=None
    )

    # load weights
    if ".pth" in weights:
        weights = torch.load(weights, map_location=torch.device('cpu'))
        if "model" in weights:
            weights = weights["model"]
        model.load_state_dict(weights)
        return model
    
    checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    return model


def get_dataloaders(names, modes):
    loaders = dict()

    if 'smartphone' in names:
        sp_dataset = TestDataset(dataset='sp', transform_size=(768, 512), modes=modes)
        loaders['smartphone'] = DataLoader(sp_dataset, batch_size=4, shuffle=False)

    if 'low_quality' in names:
        lq_dataset = TestDataset(dataset='lq', transform_size=(768, 512), modes=modes)
        loaders['low_quality'] = DataLoader(lq_dataset, batch_size=4, shuffle=False)

    if 'val' in names:
        val_dataset = ValDataset(transform_size=(768, 512), modes=modes)
        loaders['val'] = DataLoader(val_dataset, batch_size=4, shuffle=False)

    return loaders


def run_one_inference(experiment_name, test_loader, model):
    dice_metric = torchmetrics.Dice(num_classes=5, average='macro').to(device)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=experiment_name):
            # Prepare inputs
            images, masks = batch  # images: (B, N, C, H, W), masks: (B, H, W)
            batch_size, num_versions, _, _, _ = images.shape

            # Flatten images to (B * N, C, H, W)
            images = images.view(batch_size * num_versions, *images.shape[2:]).to(dtype=torch.float32, device=device)
            masks = masks.to(dtype=torch.long, device=device)

            # Forward pass
            outputs = model(images)  # (B * N, num_classes, H, W)

            # Reshape outputs back to (B, N, num_classes, H, W)
            outputs = outputs.view(batch_size, num_versions, outputs.shape[1], outputs.shape[2], outputs.shape[3])

            # Average the outputs (logits) across all versions
            avg_outputs = torch.mean(outputs, dim=1)  # (B, num_classes, H, W)

            # Get predictions
            preds = torch.argmax(avg_outputs, dim=1)  # (B, H, W)

            # Update metric
            dice_metric.update(preds, masks)

    # Compute the final Dice score
    final_score = dice_metric.compute().item()
    rounded_score = round(final_score * 100, 1)
    print(f"{experiment_name}: {rounded_score}")
    print("###########################################")



def run_inference(backbone, weights, names, modes):
    # get model
    model = load_model(backbone, weights)
    model.eval()
    model.to(device)

    # run inference
    loaders = get_dataloaders(names, modes)
    for name, loader in loaders.items():
        run_one_inference(name, loader, model)


if __name__ == "__main__":
    backbone = "efficientnet-b2"

    weight_list = [
        "checkpoints/example.ckpt"
    ]

    for weights in weight_list:
        print(f"Weights: {weights}")
        run_inference(backbone=backbone, weights=weights, names='val', modes=[])
        run_inference(backbone=backbone, \
                    weights=weights, \
                    names=["smartphone", "low_quality"], \
                    modes=[])
        run_inference(backbone=backbone, \
                    weights=weights, \
                    names=["smartphone", "low_quality"], \
                    modes=['dcin_gris'])
        run_inference(backbone=backbone, \
                    weights=weights, \
                    names=["smartphone", "low_quality"], \
                    modes=['dcin_lris'])
        run_inference(backbone=backbone, \
                    weights=weights, \
                    names=["smartphone", "low_quality"], \
                    modes=['dcin_full'])

        print("\n")
