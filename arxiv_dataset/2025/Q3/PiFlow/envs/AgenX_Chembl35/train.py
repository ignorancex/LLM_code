#!/usr/bin/env python
"""
Training script for MoleculeGCN model on ChEMBL pChEMBL values.
"""

import os
import time
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from src.model import MoleculeGCN, Standardizer
from src.preprocessing import MoleculeDataset, get_data_loaders, analyze_dataset


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MoleculeGCN on ChEMBL data")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable dataset caching"
    )
    parser.add_argument(
        "--no_precompute", action="store_true", help="Disable dataset precomputation"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save intermediate model every N epochs",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Add data loading config if it doesn't exist
    if "data_loading" not in config:
        config["data_loading"] = {
            "num_workers": 4,
            "pin_memory": True,
            "use_cached_loader": True,
            "prefetch_size": 2,
            "precompute": True,
            "n_jobs": None,
            "use_cache": True,
        }

    return config


def train_epoch(model, data_loader, optimizer, loss_fn, standardizer, device):
    """
    Train model for one epoch.

    Args:
        model (MoleculeGCN): Model to train
        data_loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        loss_fn (torch.nn.Module): Loss function
        standardizer (Standardizer): Target standardizer
        device (torch.device): Device to use for training

    Returns:
        tuple: (epoch_loss, epoch_mae)
    """
    # Set model to training mode
    model.train()

    # Initialize metrics
    epoch_loss = 0.0
    epoch_mae = 0.0
    n_batches = 0

    # Iterate over batches
    for (node_mats, adj_mats), targets, _ in data_loader:
        # Move data to device (if not already done by CachedLoader)
        if not isinstance(node_mats, torch.Tensor) or node_mats.device != device:
            node_mats = node_mats.to(device)
            adj_mats = adj_mats.to(device)
            targets = targets.to(device)

        # Standardize targets (keep original shape for later MAE calculation)
        targets_std = standardizer.standardize(targets)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(node_mats, adj_mats)

        # Calculate loss - ensure dimensions match (reshape targets to match outputs)
        loss = loss_fn(outputs, targets_std.view(-1, 1))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics
        epoch_loss += loss.item()

        # Calculate MAE on original scale
        predictions = standardizer.unstandardize(outputs.detach())
        mae = torch.mean(torch.abs(predictions - targets)).item()
        epoch_mae += mae

        n_batches += 1

    # Calculate average metrics
    epoch_loss /= n_batches
    epoch_mae /= n_batches

    return epoch_loss, epoch_mae


def evaluate(model, data_loader, loss_fn, standardizer, device):
    """
    Evaluate model on validation or test data.

    Args:
        model (MoleculeGCN): Model to evaluate
        data_loader (DataLoader): Data loader
        loss_fn (torch.nn.Module): Loss function
        standardizer (Standardizer): Target standardizer
        device (torch.device): Device to use for evaluation

    Returns:
        tuple: (loss, mae, rmse, r2, predictions, targets, smiles)
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize metrics
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_smiles = []
    n_batches = 0

    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over batches
        for (node_mats, adj_mats), targets, smiles in data_loader:
            # Move data to device (if not already done by CachedLoader)
            if not isinstance(node_mats, torch.Tensor) or node_mats.device != device:
                node_mats = node_mats.to(device)
                adj_mats = adj_mats.to(device)
                targets = targets.to(device)

            # Standardize targets
            targets_std = standardizer.standardize(targets)

            # Forward pass
            outputs = model(node_mats, adj_mats)

            # Calculate loss - ensure dimensions match
            loss = loss_fn(outputs, targets_std.view(-1, 1))
            total_loss += loss.item()

            # Convert predictions back to original scale
            predictions = standardizer.unstandardize(outputs)

            # Store predictions and targets
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_smiles.extend(smiles)

            n_batches += 1

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics
    loss = total_loss / n_batches
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)

    return loss, mae, rmse, r2, all_predictions, all_targets, all_smiles


def create_plots(
    train_losses,
    val_losses,
    train_maes,
    val_maes,
    train_r2s,
    val_r2s,
    save_dir,
    dpi=300,
):
    """
    Create and save training plots.

    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_maes (list): Training MAEs
        val_maes (list): Validation MAEs
        train_r2s (list): Training R² values
        val_r2s (list): Validation R² values
        save_dir (str): Directory to save plots
        dpi (int): DPI for saved plots
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create epochs array
    epochs = np.arange(1, len(train_losses) + 1)

    # Create loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(save_dir, "loss_curve.pdf"), dpi=dpi)
    plt.close()

    # Create MAE plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_maes, "b-", label="Training MAE")
    plt.plot(epochs, val_maes, "r-", label="Validation MAE")
    plt.title("MAE vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(save_dir, "mae_curve.pdf"), dpi=dpi)
    plt.close()

    # Create R² plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_r2s, "b-", label="Training R²")
    plt.plot(epochs, val_r2s, "r-", label="Validation R²")
    plt.title("R² vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("R² Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(save_dir, "r2_curve.pdf"), dpi=dpi)
    plt.close()


def create_parity_plot(
    predictions, targets, save_dir, dpi=300, filename="parity_plot.pdf"
):
    """
    Create and save parity plot.

    Args:
        predictions (numpy.ndarray): Model predictions
        targets (numpy.ndarray): Ground truth targets
        save_dir (str): Directory to save plot
        dpi (int): DPI for saved plot
        filename (str): Name of the output file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create plot
    plt.figure(figsize=(8, 8))

    # Plot predictions vs targets
    plt.scatter(targets, predictions, alpha=0.5, edgecolor="k")

    # Calculate plot limits
    min_val = min(np.min(predictions), np.min(targets))
    max_val = max(np.max(predictions), np.max(targets))

    # Add identity line
    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    # Add R2 score text
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))

    plt.text(
        0.05,
        0.95,
        f"R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Add labels and title
    plt.xlabel("Actual pChEMBL Values")
    plt.ylabel("Predicted pChEMBL Values")
    plt.title("Parity Plot: Actual vs. Predicted pChEMBL Values")

    # Make plot square
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(save_dir, filename), dpi=dpi)
    plt.close()


def save_model_checkpoint(
    model,
    optimizer,
    standardizer,
    epoch,
    metrics,
    config,
    save_path,
    additional_info=None,
):
    """
    Save model checkpoint.

    Args:
        model (MoleculeGCN): Model to save
        optimizer (torch.optim.Optimizer): Optimizer
        standardizer (Standardizer): Target standardizer
        epoch (int): Current epoch
        metrics (dict): Dictionary of metrics
        config (dict): Configuration
        save_path (str): Path to save checkpoint
        additional_info (dict, optional): Additional information to save
    """
    # Create checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "standardizer": standardizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    # Add additional info if provided
    if additional_info is not None:
        checkpoint.update(additional_info)

    # Save checkpoint
    torch.save(checkpoint, save_path)


def main():
    """Main training function."""
    # Parse arguments and load config
    args = parse_arguments()
    config = load_config(args.config)

    # Override config with command line arguments
    if args.no_cache:
        config["data_loading"]["use_cache"] = False
    if args.no_precompute:
        config["data_loading"]["precompute"] = False
    if args.workers is not None:
        config["data_loading"]["num_workers"] = args.workers

    # Set random seeds
    seed = config["misc"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set device
    use_gpu = (
        config["misc"]["use_gpu"] and torch.cuda.is_available() and not args.no_gpu
    )
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Using device: {device}")

    # Create save directories
    save_dir = config["training"]["save_dir"]
    plot_dir = config["misc"]["plot_dir"]
    intermediate_dir = os.path.join(save_dir, "intermediate")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = MoleculeDataset(
        csv_file=config["data"]["dataset_path"],
        smiles_col=config["data"]["smiles_col"],
        target_col=config["data"]["target_col"],
        node_vec_len=config["model"]["node_vec_len"],
        max_atoms=config["data"]["max_atoms"],
        delimiter=config["data"]["delimiter"],
        use_cache=config["data_loading"]["use_cache"],
        n_jobs=config["data_loading"]["n_jobs"],
        precompute=config["data_loading"]["precompute"],
    )

    # Print dataset statistics
    stats = analyze_dataset(
        csv_file=config["data"]["dataset_path"],
        smiles_col=config["data"]["smiles_col"],
        target_col=config["data"]["target_col"],
        delimiter=config["data"]["delimiter"],
    )

    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset=dataset,
        batch_size=config["training"]["batch_size"],
        train_ratio=config["training"]["train_ratio"],
        val_ratio=config["training"]["val_ratio"],
        test_ratio=config["training"]["test_ratio"],
        seed=seed,
        num_workers=config["data_loading"]["num_workers"],
        pin_memory=config["data_loading"]["pin_memory"],
        use_cached_loader=config["data_loading"]["use_cached_loader"],
        device=device if config["data_loading"]["use_cached_loader"] else None,
        prefetch_size=config["data_loading"]["prefetch_size"],
    )

    # Create standardizer
    all_targets = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    standardizer = Standardizer(all_targets)
    print(
        f"\nTarget standardization: mean={standardizer.mean:.4f}, std={standardizer.std:.4f}"
    )

    # Create model
    print("\nCreating model...")
    model = MoleculeGCN(
        node_vec_len=config["model"]["node_vec_len"],
        node_fea_len=config["model"]["node_fea_len"],
        hidden_fea_len=config["model"]["hidden_fea_len"],
        n_conv=config["model"]["n_conv"],
        n_hidden=config["model"]["n_hidden"],
        n_outputs=config["model"]["n_outputs"],
        p_dropout=config["model"]["p_dropout"],
    )

    # Move model to device
    model = model.to(device)

    # Create optimizer and loss function
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    loss_fn = nn.MSELoss()

    # Print model summary
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Training loop
    print("\nStarting training...")
    n_epochs = config["training"]["n_epochs"]
    patience = config["training"]["early_stopping"]
    save_interval = args.save_interval

    best_val_loss = float("inf")
    best_val_r2 = float("-inf")
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    train_r2s = []
    val_r2s = []

    # Start time
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        # Train for one epoch
        epoch_start_time = time.time()
        train_loss, train_mae = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            standardizer=standardizer,
            device=device,
        )

        # Evaluate on training set to get R²
        _, _, _, train_r2, train_preds, train_targets, _ = evaluate(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            standardizer=standardizer,
            device=device,
        )

        # Evaluate on validation set
        val_loss, val_mae, val_rmse, val_r2, val_preds, val_targets, _ = evaluate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            standardizer=standardizer,
            device=device,
        )

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print epoch stats
        print(
            f"Epoch {epoch}/{n_epochs} ({epoch_time:.2f}s): "
            f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f} | "
            f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}"
        )

        # Save intermediate models at specified intervals
        if epoch % save_interval == 0 or epoch == n_epochs:
            metrics = {
                "train_loss": train_loss,
                "train_mae": train_mae,
                "train_r2": train_r2,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
            }

            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                standardizer=standardizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
                save_path=os.path.join(intermediate_dir, f"model_epoch_{epoch}.pt"),
                additional_info={
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_maes": train_maes,
                    "val_maes": val_maes,
                    "train_r2s": train_r2s,
                    "val_r2s": val_r2s,
                },
            )

            # Create parity plot for this checkpoint
            create_parity_plot(
                predictions=val_preds.flatten(),
                targets=val_targets.flatten(),
                save_dir=os.path.join(plot_dir, "intermediate"),
                filename=f"parity_plot_epoch_{epoch}.pdf",
            )

        # Check for improvement in loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save best loss model
            metrics = {
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
            }

            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                standardizer=standardizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
                save_path=os.path.join(save_dir, "best_loss_model.pt"),
                additional_info={
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_maes": train_maes,
                    "val_maes": val_maes,
                    "train_r2s": train_r2s,
                    "val_r2s": val_r2s,
                },
            )

            print(f"  Saved new best model with validation loss: {val_loss:.4f}")

        # Check for improvement in R²
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2

            # Save best R² model
            metrics = {
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
            }

            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                standardizer=standardizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
                save_path=os.path.join(save_dir, "best_r2_model.pt"),
                additional_info={
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_maes": train_maes,
                    "val_maes": val_maes,
                    "train_r2s": train_r2s,
                    "val_r2s": val_r2s,
                },
            )

            # Create parity plot for best R² model
            create_parity_plot(
                predictions=val_preds.flatten(),
                targets=val_targets.flatten(),
                save_dir=plot_dir,
                filename=f"best_r2_parity_plot.pdf",
            )

            print(f"  Saved new best model with validation R²: {val_r2:.4f}")

            # Reset patience counter if we have improvement in either metric
            patience_counter = 0
        else:
            # Only increment patience counter if neither loss nor R² improved
            if val_loss >= best_val_loss:
                patience_counter += 1

        # Check for early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    # End time
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds.")

    # Create training plots
    create_plots(
        train_losses=train_losses,
        val_losses=val_losses,
        train_maes=train_maes,
        val_maes=val_maes,
        train_r2s=train_r2s,
        val_r2s=val_r2s,
        save_dir=plot_dir,
    )

    # Load best loss model for testing
    loss_checkpoint = torch.load(
        os.path.join(save_dir, "best_loss_model.pt"), weights_only=True
    )
    model.load_state_dict(loss_checkpoint["model_state_dict"])
    standardizer.load_state_dict(loss_checkpoint["standardizer"])
    best_loss_epoch = loss_checkpoint["epoch"]

    print(f"\nLoaded best loss model from epoch {best_loss_epoch}")

    # Evaluate on test set
    print("\nEvaluating best loss model on test set...")
    test_loss, test_mae, test_rmse, test_r2, test_preds, test_targets, test_smiles = (
        evaluate(
            model=model,
            data_loader=test_loader,
            loss_fn=loss_fn,
            standardizer=standardizer,
            device=device,
        )
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # Create parity plot for best loss model
    create_parity_plot(
        predictions=test_preds.flatten(),
        targets=test_targets.flatten(),
        save_dir=plot_dir,
        filename="best_loss_test_parity_plot.pdf",
    )

    # Load best R² model for testing
    r2_checkpoint = torch.load(
        os.path.join(save_dir, "best_r2_model.pt"), weights_only=True
    )
    model.load_state_dict(r2_checkpoint["model_state_dict"])
    standardizer.load_state_dict(r2_checkpoint["standardizer"])
    best_r2_epoch = r2_checkpoint["epoch"]

    print(f"\nLoaded best R² model from epoch {best_r2_epoch}")

    # Evaluate on test set
    print("\nEvaluating best R² model on test set...")
    (
        r2_test_loss,
        r2_test_mae,
        r2_test_rmse,
        r2_test_r2,
        r2_test_preds,
        r2_test_targets,
        r2_test_smiles,
    ) = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        standardizer=standardizer,
        device=device,
    )

    print(f"Test Loss: {r2_test_loss:.4f}")
    print(f"Test MAE: {r2_test_mae:.4f}")
    print(f"Test RMSE: {r2_test_rmse:.4f}")
    print(f"Test R²: {r2_test_r2:.4f}")

    # Create parity plot for best R² model
    create_parity_plot(
        predictions=r2_test_preds.flatten(),
        targets=r2_test_targets.flatten(),
        save_dir=plot_dir,
        filename="best_r2_test_parity_plot.pdf",
    )

    # Compare best loss and best R² models
    print("\nComparison of best models:")
    print(f"Best Loss Model (Epoch {best_loss_epoch}):")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test R²: {test_r2:.4f}")

    print(f"\nBest R² Model (Epoch {best_r2_epoch}):")
    print(f"  Test Loss: {r2_test_loss:.4f}")
    print(f"  Test MAE: {r2_test_mae:.4f}")
    print(f"  Test RMSE: {r2_test_rmse:.4f}")
    print(f"  Test R²: {r2_test_r2:.4f}")

    # Save test results
    test_results = {
        "best_loss_model": {
            "epoch": best_loss_epoch,
            "test_loss": test_loss,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        },
        "best_r2_model": {
            "epoch": best_r2_epoch,
            "test_loss": r2_test_loss,
            "test_mae": r2_test_mae,
            "test_rmse": r2_test_rmse,
            "test_r2": r2_test_r2,
        },
    }

    # Save final model
    torch.save(
        {
            "epoch": n_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "standardizer": standardizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_maes": train_maes,
            "val_maes": val_maes,
            "train_r2s": train_r2s,
            "val_r2s": val_r2s,
            "test_results": test_results,
            "config": config,
        },
        os.path.join(save_dir, "final_model.pt"),
    )

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
