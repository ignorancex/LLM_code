import wandb
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T
from data import CelebDataset, RandomCropWithFixedCoordinates
from models import ResNet18
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from evaluation import calculate_metrics
from datetime import datetime

# ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)


if __name__ == "__main__":

    dt = datetime.now()
    ts = int(datetime.timestamp(dt))

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--crop_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--step_size_scheduler", type=int, default=40)
    parser.add_argument("--n_regions", type=int, default=2401, help='number of regions to divide the image in, should be a square number')
    parser.add_argument("--attribute", type=str, default="Blond_Hair", help='the attribute for which to train the model')
    parser.add_argument("--num_classes", type=int, default=2, help='number of classes for the attribute')
    parser.add_argument("--wandb_project", type=str, default="region_classification", help='wandb project name')
    parser.add_argument("--wandb_entity", type=str, default="sanderdc", help='wandb entity name')
    parser.add_argument("--save_path", type=str, default=f"models/model_{ts}.pt")



    args = parser.parse_args()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)


    transforms = T.Compose([
        #T.ToTensor(),
        T.Resize(128, antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomCropWithFixedCoordinates(128, args.crop_size, args.n_regions),
    ])

    dataset = CelebDataset(target=args.attribute, transforms=transforms)
    val_dataset = CelebDataset(target=args.attribute, partition=2, transforms=transforms)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ResNet18(num_classes=args.num_classes, n_regions=args.n_regions)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=args.step_size_scheduler, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        model.train()
        metrics_to_log = {"running_loss": 0.0, "accuracy": 0.0,
                          "top20_accuracy": 0.0, "margin_top20": 0.0, "brier_score": 0.0, "ece": 0.0}

        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch}"):
            # load and prepare data
            inputs, loc, labels = data
            inputs, loc, labels = inputs.to(device), loc.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, loc)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log metrics            
            metrics_to_log["running_loss"] += loss.item()
            metrics_to_log = calculate_metrics(
                outputs, labels, metrics_to_log, n_labels=len(dataset.label_categories))

            if i % 20 == 19:    # print every 20 mini-batches
                for key in metrics_to_log:
                    metrics_to_log[key] /= 20.0
                wandb.log(metrics_to_log)
                for key in metrics_to_log:
                    metrics_to_log[key] = 0.0            

        # validation
        if epoch % 2 == 0:
            model.eval()
            metrics_to_log = {"val_running_loss": 0.0, "val_accuracy": 0.0,
                              "val_top20_accuracy": 0.0, "val_margin_top20": 0.0, "val_brier_score": 0.0, "val_ece": 0.0}

            predictions = []
            targets = []
            for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader)):
                inputs, loc, labels = data
                inputs, loc, labels = inputs.to(device), loc.to(
                    device), labels.to(device)
                with torch.no_grad():
                    # forward + backward + optimize
                    outputs = model(inputs, loc)
                    loss = criterion(outputs, labels)

                    predictions.append(
                        torch.nn.functional.softmax(outputs, dim=1))
                    targets.append(labels)
                # calculate statistics
                metrics_to_log["val_running_loss"] += loss.item()
                metrics = calculate_metrics(
                    outputs, labels, metrics_to_log, pretext="val_",  n_labels=len(dataset.label_categories))

                if i % 20 == 19:
                    for key in metrics_to_log:
                        metrics_to_log[key] /= 20.0
                    wandb.log(metrics_to_log)
                    for key in metrics_to_log:
                        metrics_to_log[key] = 0.0

            predictions = torch.cat(predictions, dim=0)
            targets = torch.cat(targets, dim=0)
            wandb.log({"val_confusion_matrix": wandb.plot.confusion_matrix(
                probs=predictions.detach().cpu().numpy(), y_true=targets.detach().cpu().numpy(), class_names=dataset.label_categories)})

        scheduler.step()

    dt = datetime.now()
    ts = int(datetime.timestamp(dt))

    print('Finished Training')
    torch.save(model.state_dict(
    ), f"models/model_{args.attribute}_{ts}.pt")
