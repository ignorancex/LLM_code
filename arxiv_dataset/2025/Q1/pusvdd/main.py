import argparse
import os
import random

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

import dataloaders
from models import detectors, losses, trainers, utils


class Config(argparse.Namespace):
    dataset: str
    normal_class: int
    unseen_anomaly: int
    algorithm: str
    alpha: float
    n_epoch: int
    learning_rate: float
    batch_size: int
    seed: int


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--normal_class", type=int, default=0)
    parser.add_argument("--unseen_anomaly", type=int, default=9)
    parser.add_argument("--algorithm", type=str, default="AE")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    config = parser.parse_args(namespace=Config())

    if config.algorithm not in ["PU", "PUAE", "PUSVDD", "LOE", "SOEL"]:
        config.alpha = 0.0

    print("Setting:", config)

    # Key
    key = "_".join([str(v) for v in vars(config).values()])

    # Seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset
    train_loader, valid_loader, test_loader, test_seen_loader, test_unseen_loader = dataloaders.load(
        name=config.dataset,
        is_gray_scale=False,
        batch_size=config.batch_size,
        normal_class=config.normal_class,
        unseen_anomaly=config.unseen_anomaly,
    )

    # Output Directories
    for path in ["checkpoints", "results", "images"]:
        os.makedirs(path, exist_ok=True)

    # Shallow Model
    if config.algorithm in ["IF", "OCSVM"]:
        X_train, Y_train = dataloaders.to_numpy(train_loader)
        X_test, Y_test = dataloaders.to_numpy(test_loader)
        X_seen_test, Y_seen_test = dataloaders.to_numpy(test_seen_loader)
        X_unseen_test, Y_unseen_test = dataloaders.to_numpy(test_unseen_loader)

        X_train = X_train.reshape(X_train.shape[0], 3 * 32 * 32)
        X_test = X_test.reshape(X_test.shape[0], 3 * 32 * 32)
        X_seen_test = X_seen_test.reshape(X_seen_test.shape[0], 3 * 32 * 32)
        X_unseen_test = X_unseen_test.reshape(X_unseen_test.shape[0], 3 * 32 * 32)

        clf = IsolationForest(random_state=config.seed) if config.algorithm == "IF" else OneClassSVM(gamma="auto")
        clf.fit(X_train[Y_train == 0])

        Y_test_predict = 1 - clf.decision_function(X_test)
        test_score = roc_auc_score(y_true=Y_test, y_score=Y_test_predict)
        print("AUC:", test_score)

        Y_seen_predict = 1 - clf.decision_function(X_seen_test)
        seen_anomaly_score = roc_auc_score(y_true=Y_seen_test, y_score=Y_seen_predict)
        print("AUC (seen):", seen_anomaly_score)

        Y_unseen_predict = 1 - clf.decision_function(X_unseen_test)
        unseen_anomaly_score = roc_auc_score(y_true=Y_unseen_test, y_score=Y_unseen_predict)
        print("AUC (unseen):", unseen_anomaly_score)

        result = utils.Result(
            train_loss=[],
            valid_loss=[],
            test_score=test_score,
            seen_anomaly_score=seen_anomaly_score,
            unseen_anomaly_score=unseen_anomaly_score,
        )
        utils.save_result(result=result, file_name=f"results/{key}.json")

    else:
        # Deep Model
        model = detectors.load_convolutional_model(config.algorithm, epsilon=0)
        model = model.to(device)
        print("Model:", type(model))

        # Criterion
        criterion = losses.load(algorithm=config.algorithm, alpha=config.alpha)

        # Train
        if isinstance(model, detectors.PreTrainableSVDD):
            # Pre-Training DeepSVDD as Autoencoder
            print("Pre-Training:")
            pre_trainer = trainers.Trainer(device)
            pre_trainer.fit(
                model=model,
                criterion=losses.AELoss(),
                train_loader=train_loader,
                valid_loader=valid_loader,
                checkpoint=f"checkpoints/{key}.pt",
                n_epoch=config.n_epoch,
                learning_rate=config.learning_rate,
                weight_decay=1e-3,
            )

        if isinstance(model, detectors.DeepSVDD):
            # Set center for DeepSVDD model
            print("Set center for DeepSVDD:")
            utils.set_center(model=model, train_loader=train_loader, device=device, eps=0.1)

        print("Training:")
        trainer = trainers.Trainer(device=device)
        trainer.fit(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            checkpoint=f"checkpoints/{key}.pt",
            n_epoch=config.n_epoch,
            learning_rate=config.learning_rate,
            weight_decay=1e-3,
        )

        # Test
        test_score = utils.compute_auc(model=model, test_loader=test_loader, device=device)
        seen_anomaly_score = utils.compute_auc(model=model, test_loader=test_seen_loader, device=device)
        unseen_anomaly_score = utils.compute_auc(model=model, test_loader=test_unseen_loader, device=device)
        print("AUC:", test_score)
        print("AUC (seen):", seen_anomaly_score)
        print("AUC (unseen):", unseen_anomaly_score)

        result = utils.Result(
            train_loss=trainer.train_losses,
            valid_loss=trainer.valid_losses,
            test_score=test_score,
            seen_anomaly_score=seen_anomaly_score,
            unseen_anomaly_score=unseen_anomaly_score,
        )

        # Save and Plot Results
        utils.save_result(result=result, file_name=f"results/{key}.json")
        utils.plot_result(result=result, file_name=f"images/{key}.pdf")
