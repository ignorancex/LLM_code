import copy
import os
import sys
import time

import torch
import torchvision
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from optimizer.signum import Signum
from utils_shared.run_utils_shared import (
    get_hyperparameter_config_path,
    seed_everything,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np


class WrappedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        data, label = self.subset[index]
        if self.transform:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.subset)


def get_data_sets(dataset_name="flowers102", train_val_split=0.8):
    transforms = {
        "train_flowers102": torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation(45),
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
        "train_fgvc_aircraft": torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
        "valid": torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    if dataset_name == "flowers102":
        dataset_class = torchvision.datasets.Flowers102
    elif dataset_name == "fgvc_aircraft":
        dataset_class = torchvision.datasets.FGVCAircraft
    else:
        raise NotImplementedError
    train_dataset = dataset_class(
        root=f"~/.data/{dataset_name}", download=True, split="train", transform=None
    )
    val_dataset = dataset_class(
        root=f"~/.data/{dataset_name}", download=True, split="val", transform=None
    )
    test_dataset = dataset_class(
        root=f"~/.data/{dataset_name}",
        download=True,
        split="test",
        transform=transforms["valid"],
    )

    train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    train_size = int(train_val_split * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size]
    )
    train_dataset = WrappedSubset(
        train_indices, transform=transforms[f"train_{dataset_name}"]
    )
    val_dataset = WrappedSubset(val_indices, transform=transforms["valid"])

    return {"train": train_dataset, "valid": val_dataset, "test": test_dataset}


def predict(model, data, data_loaders):
    model.eval()
    model.to(device)

    combined_predictions = []
    combined_true_labels = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loaders[data]):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            prediction = (
                torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
            )
            combined_predictions.extend(prediction)
            combined_true_labels.extend(labels.cpu().numpy())

    combined_true_labels = [int(i) for i in combined_true_labels]

    return combined_predictions, combined_true_labels


num_labels_dict = {
    "flowers102": 102,
    "fgvc_aircraft": 100,
}


def calibration_error(predictions, references, n_bins=15):
    predictions, references = np.array(predictions), np.array(references)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = np.max(predictions, axis=1)
    predictions = np.argmax(predictions, axis=1)
    if references.ndim > 1:  # Soft labels
        references = np.argmax(references, axis=1)
    accuracies = predictions == references

    ece = 0.0
    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prob_in_bin = in_bin.mean()
        if prob_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += calibration_error * prob_in_bin
            mce = np.max([calibration_error, mce])
    return ece, mce


def calculate_accuracy(model, data, data_loaders, cli_args):
    predictions, true_labels = predict(model, data, data_loaders)
    predicted_labels = [np.argmax(p) for p in predictions]
    accuracy = sum(
        [
            1 if predicted_labels[i] == true_labels[i] else 0
            for i in range(len(predicted_labels))
        ]
    ) / len(predicted_labels)
    num_labels = num_labels_dict[cli_args.dataset_name]
    calibration_error_score, mce_score = calibration_error(predictions, true_labels)
    per_class_accuracy_metrics = per_class_accuracy(
        predicted_labels, true_labels, num_labels
    )
    accuracy_metrics = {
        "accuracy": accuracy,
        "ece": calibration_error_score,
        "mce": mce_score,
    } | per_class_accuracy_metrics
    return accuracy_metrics


def per_class_accuracy(predictions, true_labels, num_labels=102):
    class_correct = [0 for i in range(num_labels)]
    class_total = [0 for i in range(num_labels)]
    for i in range(len(predictions)):
        label = true_labels[i]
        class_correct[label] += predictions[i] == true_labels[i]
        class_total[label] += 1
    per_class_accuracy = [class_correct[i] / class_total[i] for i in range(num_labels)]
    per_class_accuracy_metrics = {
        "mean_pca": np.mean(per_class_accuracy),
        "min_pca": np.min(per_class_accuracy),
        "max_pca": np.max(per_class_accuracy),
    }
    return per_class_accuracy_metrics


def get_optimizer(cli_args, model, learning_rate=None):
    if learning_rate is None:
        hyperparameter_path = get_hyperparameter_config_path(cli_args)
        with open(
            hyperparameter_path,
            "r",
        ) as f:
            hyperparameter_config = yaml.safe_load(f)
        learning_rate = hyperparameter_config[cli_args.dataset_name][
            cli_args.optimizer_name
        ]["lr"]
    if cli_args.optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif cli_args.optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)
    elif cli_args.optimizer_name == "sgd_momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif cli_args.optimizer_name == "signsgd":
        optimizer = Signum(model.parameters(), lr=learning_rate, momentum=0.0)
    elif cli_args.optimizer_name == "signsgd_momentum":
        optimizer = Signum(model.parameters(), lr=learning_rate, momentum=0.9)
    elif cli_args.optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer name {cli_args.optimizer_name}")
    return optimizer


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    data_loaders,
    dataset_sizes,
    cli_args,
    num_epochs=25,
    log_dir=None,
    epoch_per_val=1,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    log_epoch_loss = {"train": [], "valid": []}
    log_epoch_acc = {"train": [], "valid": []}
    accumulation_steps = cli_args.accumulation_steps
    if accumulation_steps == -1:
        accumulation_steps = len(data_loaders["train"])

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                if epoch % epoch_per_val != 0 and epoch != num_epochs - 1:
                    continue
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            optimizer.zero_grad()
            for n_iter, (inputs, labels) in enumerate(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels) / accumulation_steps
                    if phase == "train":
                        loss.backward()
                        if (n_iter + 1) % accumulation_steps == 0 or (
                            n_iter == len(data_loaders[phase]) - 1
                        ):
                            if cli_args.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), cli_args.max_grad_norm
                                )
                            optimizer.step()
                            optimizer.zero_grad()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            log_epoch_loss[phase].append(epoch_loss)
            log_epoch_acc[phase].append(epoch_acc)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    if log_dir is not None:
        log_loss = np.zeros((num_epochs, 2))
        log_acc = np.zeros((num_epochs, 2))
        for idx, phase in enumerate(["train", "valid"]):
            loss_to_save = np.array(log_epoch_loss[phase])
            acc_to_save = np.array(log_epoch_acc[phase])
            log_loss[: len(loss_to_save), idx] = loss_to_save
            log_acc[: len(acc_to_save), idx] = acc_to_save
        np.save(f"{log_dir}/loss.npy", log_loss)
        np.save(f"{log_dir}/acc.npy", log_acc)
    model_best = copy.deepcopy(model)
    model_best.load_state_dict(best_model_wts)
    result_dict = {
        "model_last": model,
        "model_best": model_best,
        "eval_accuracy": float(best_acc),
        "train_loss": float(log_epoch_loss["train"][-1]),
    }
    return result_dict


def get_model(model_name, num_classes):
    print(f"Model name: {model_name}")
    print(f"Number of classes: {num_classes}")
    if model_name == "resnet18":
        weights = "IMAGENET1K_V1"
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vit-base":
        weights = "IMAGENET1K_V1"
        model = torchvision.models.vit_b_16(weights=weights)
        model.heads[0] = torch.nn.Linear(model.heads[0].in_features, num_classes)
    else:
        raise NotImplementedError
    return model


def setup(cli_args):
    worker_init_fn = seed_everything(cli_args.seed)
    num_classes = {
        "flowers102": 102,
        "fgvc_aircraft": 100,
    }[cli_args.dataset_name]
    model = get_model(cli_args.model_name, num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    model.to(device)

    eps_dict = {
        "flowers102": 50,
        "fgvc_aircraft": 100,
    }
    eps = (
        eps_dict[cli_args.dataset_name]
        if cli_args.num_train_epochs is None
        else cli_args.num_train_epochs
    )

    data_sets = get_data_sets(dataset_name=cli_args.dataset_name)
    dataset_sizes = {x: len(data_sets[x]) for x in ["train", "valid", "test"]}
    data_loaders = {
        x: torch.utils.data.DataLoader(
            data_sets[x],
            batch_size=cli_args.batch_size,
            shuffle=(x == "train"),
            num_workers=4,
            pin_memory=True,
            worker_init_fn=worker_init_fn if x == "train" else None,
        )
        for x in ["train", "valid", "test"]
    }

    criteria = torch.nn.CrossEntropyLoss()
    return model, criteria, data_loaders, dataset_sizes, eps
