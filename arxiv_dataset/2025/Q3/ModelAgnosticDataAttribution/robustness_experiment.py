from data import CelebDataset, get_subset_celeba_attr
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np 
import torchvision.transforms as T
import argparse
import wandb
from models import AttributeClassifier
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import torchvision
from utils import convert_region_logits_to_pixel_map, get_confidence_measure, map_attributions_to_pixels

# ensure reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a model on the CelebA dataset')
    parser.add_argument('--project', type=str, default="region_classification",
                        help='The name of the wandb project')
    parser.add_argument('--entity', type=str, default="sanderdc",
                        help='The name of the wandb entity')
    parser.add_argument('--epochs', type=int, default=10,
                        help='The number of epochs to train the model')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed used for torch random generator')

    parser.add_argument('--batch_size', type=int,
                        default=128, help='The batch size to use')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='The learning rate to use')
    parser.add_argument('--target', type=str, default='Male',
                        help='The target attribute to predict')
    parser.add_argument('--hidden', type=str, default='Blond_Hair',
                        help='The hidden attribute that biases the data')
    parser.add_argument('--noise', type=str, default='general_noise', choices=['general_noise', 'specific_noise', 'specific_mask', 'general_mask'],
                        help='The type of noise to add to the images')
    parser.add_argument('--quantile', type=float, default=0.7,
                        help='Quantile when using mask quantile noise')
    parser.add_argument('--num_val', type=int, default=182,
                        help='Number of validation samples per class')  
    parser.add_argument('--specific_path', type=str, default=None,
                        help='Path to specific attributions')
    parser.add_argument('--general_path', type=str, default=None,
                        help='Path to general attributions')
    args = parser.parse_args()
    wandb.init(project=args.project, entity=args.entity)

    torch.manual_seed(args.seed)

    # data loading
    transform = T.Compose([
        T.Resize((128, 128), antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.noise == "general_noise" or args.noise == "general_mask":
        # load in general and specific attribution maps
        atts = torch.load(args.general_path)
        atts_gender = torch.stack([att[0] for att in atts])
        map_general = convert_region_logits_to_pixel_map(atts_gender)

    if args.noise == "specific_noise" or args.noise == "specific_mask":
        # this section is quite slow, should look for speedups
        atts = torch.load(args.specific_path)
        atts_gender = torch.stack([att[0] for att in atts])
        confidences = get_confidence_measure(atts_gender, "negative_entropy")
        locations = []
        img_shape = 128
        patch_size = 32
        for i in range(0,img_shape-patch_size + 2,2):
                for j in range(0,img_shape-patch_size + 2,2):
                    # extract patch and interpolate location
                    locations.append([i,j])

        maps_specific = torch.stack([map_attributions_to_pixels(confidences[i], locations, torch.device("cuda")) for i in range(confidences.shape[0])])

    def targeted_noise(images, attr=None, type=args.noise, validation=False, quan=args.quantile):
        # idea, first unorm images and then norm again
        if type == "general_noise":
            return torch.clip(images + ((map_general > torch.quantile(map_general, quan)) * 0.5 * torch.randn_like(images)), 0, 1)
        if type == "specific_noise":
            if validation:
                #return images
                map_quan = torch.mean(maps_specific, axis=0)  > torch.quantile(torch.mean(maps_specific, axis=0) , quan)
                return torch.clip(images + ( torch.randn_like(images) *0.5 * map_quan ), 0, 1)
            maps_quan = torch.stack(
                [maps_specific[q] > torch.quantile(maps_specific[q], quan) for q in attr])
            return torch.clip(images + (maps_quan.unsqueeze(1).tile((1, 3, 1, 1)) * 0.5  * torch.randn_like(images)), 0, 1)
        if type == "specific_mask":
            if validation:
                return images
            maps_quan = torch.stack(
                [maps_specific[q] < torch.quantile(maps_specific[q], quan) for q in attr])
            return images * maps_quan.unsqueeze(1).tile((1, 3, 1, 1))
        if type == "general_mask":
            return images * (map_general < torch.quantile(map_general, quan))
        
    
    indexes, indexes_val = get_subset_celeba_attr(
        'data/celeba/list_attr_celeba.csv', 'data/celeba/list_eval_partition.csv', hidden_attribute=args.hidden, 
        target_attribute=args.target, num_validation_per_class=args.num_val)
    dataset = CelebDataset(
        target=args.target, transforms=transform, return_map=True)
    
    # take a subset of this dataset to allow for faster experiments
    train_subset = torch.utils.data.Subset(dataset, indexes)
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    val_data = CelebDataset(target=args.target, transforms=transform, partition=1)
    val_subset = torch.utils.data.Subset(val_data, indexes_val)
    val_dataloader = DataLoader(val_subset, batch_size=args.batch_size,
                                shuffle=False, pin_memory=True, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AttributeClassifier(num_classes=2)
    model = model.to(device)

    model_robust = copy.deepcopy(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    optimizer_robust = optim.AdamW(
        model_robust.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    scheduler_robust = ExponentialLR(optimizer_robust, gamma=0.95)

    # first show a batch of visualizations to illustrate the noise
    images, attr, labels = next(iter(train_loader))
    images, attr, labels = images.to(
        device), attr.to(device), labels.to(device)
    attr = np.array([np.where(indexes == attr[i].item())[0].item()
                    for i in range(attr.shape[0])])
    images_noise = targeted_noise(images, attr)
    wandb.log({"images": wandb.Image(torchvision.utils.make_grid(images.cpu(), nrow=8).permute(1, 2, 0).numpy()),
               "images_noise": wandb.Image(torchvision.utils.make_grid(images_noise.cpu(), nrow=8).permute(1, 2, 0).numpy())})

    for epoch in range(args.epochs):
        model.train()
        model_robust.train()

        
        running_loss, running_loss_robust = 0, 0
        acc, acc_robust = 0, 0

        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch}"):
            img,  attr, target = data
            target = target.type(torch.long)
            img, target, attr = img.to(device), target.to(
                device), attr.to(device)
            target_oh = torch.nn.functional.one_hot(target, num_classes=2)
            attr = np.array([np.where(indexes == attr[i].item())[
                            0].item() for i in range(attr.shape[0])])

            optimizer.zero_grad()
            optimizer_robust.zero_grad()

            outputs = model(img)

            loss = criterion(outputs, target_oh.float())

            img = targeted_noise(img, attr)

            outputs_robust = model_robust(img)
            loss_robust = criterion(outputs_robust, target_oh.float())

            loss.backward()
            loss_robust.backward()
            optimizer.step()
            optimizer_robust.step()

            running_loss += loss.item()
            running_loss_robust += loss_robust.item()

            # update accuracy
            preds = torch.argmax(outputs, 1)
            preds_robust = torch.argmax(outputs_robust, 1)
            acc += (preds == target).float().mean()
            acc_robust += (preds_robust == target).float().mean()
            if i % 10 == 9:
                wandb.log({"loss": running_loss / 10, "loss_robust": running_loss_robust /
                          10, "train_accuracy": acc / 10, "train_accuracy_robust": acc_robust / 10})
                running_loss = 0.0
                running_loss_robust = 0.0
                acc = 0
                acc_robust = 0
        scheduler.step()
        scheduler_robust.step()

        # do a validation run every epoch
        if epoch % 1 == 0:
            model.eval()
            model_robust.eval()

            all_preds = []
            all_targets = []
            all_preds_robust = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(val_dataloader, 0), total=len(val_dataloader)):
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    images = targeted_noise(images, validation=True)
                    outputs_robust = model_robust(images)

                    preds = torch.argmax(outputs, 1)

                    preds2 = torch.argmax(outputs_robust, 1)

                    all_preds.append(preds)
                    all_preds_robust.append(preds2)
                    all_targets.append(labels)

            all_preds = torch.cat(all_preds)
            all_preds_robust = torch.cat(all_preds_robust)
            all_targets = torch.cat(all_targets)
            sensitive_features = torch.tensor(
                [0 if i < args.num_val *2 else 1 for i in range(args.num_val *4)])

            wandb.log({"val_accuracy": (all_preds == all_targets).float().mean(), "val_accuracy_robust": (all_preds_robust == all_targets).float().mean(),
                       "val_accuracy_men": (all_preds[:args.num_val *2] == all_targets[:args.num_val *2]).float().mean(),
                       "val_accuracy_women:": (all_preds[args.num_val *2:] == all_targets[args.num_val *2:]).float().mean(),
                       "val_accuracy_men_robust:": (all_preds_robust[:args.num_val *2] == all_targets[:args.num_val *2]).float().mean(),
                       "val_accuracy_women_robust:": (all_preds_robust[args.num_val *2:] == all_targets[args.num_val *2:]).float().mean(),
                      })
    wandb.finish()