import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data_processor import SuperconDataProcessor
from src.model import TcPredictor
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def train_model(tsv_path, epochs=150, batch_size=32, learning_rate=0.001):
    """
    Train a model to predict critical temperature from superconductor data

    Args:
        tsv_path: Path to TSV file containing superconductor data
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    print(f"Loading and processing data from: {tsv_path}")

    # Initialize data processor and load data
    processor = SuperconDataProcessor()
    X_train, X_test, y_train, y_test = processor.load_and_process_data(tsv_path)

    # Get input size from processed features
    input_size = X_train.shape[1]
    print(f"Input feature size: {input_size}")

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = TcPredictor(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Add learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True, min_lr=1e-5
    )

    # Lists to store metrics for plotting
    train_losses = []
    test_losses = []
    r2_scores = []

    # Training loop
    best_loss = float("inf")
    best_r2 = -float("inf")
    patience = 20000  # Early stopping patience
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_X)

        avg_train_loss = total_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        # Validate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

            # Calculate R² score
            y_pred = test_outputs.numpy().flatten()
            y_true = y_test.numpy().flatten()
            r2 = r2_score(y_true, y_pred)
            r2_scores.append(r2)

        # Update scheduler
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, R²: {r2:.4f}, LR: {current_lr:.6f}"
        )

        # Save best model based on R²
        if r2 > best_r2:
            best_r2 = r2
            if not os.path.exists("./models"):
                os.makedirs("./models")
            torch.save(model.state_dict(), "./models/best_supercon_model_r2.pth")

            # Save model info
            with open("./models/model_info_r2.txt", "w") as f:
                f.write(f"input_size: {input_size}\n")
                f.write(f"best_epoch: {epoch + 1}\n")
                f.write(f"best_r2: {best_r2:.6f}\n")
                f.write(f"test_loss: {test_loss:.6f}\n")

        # Save best model based on loss
        if test_loss < best_loss:
            best_loss = test_loss
            if not os.path.exists("./models"):
                os.makedirs("./models")
            torch.save(model.state_dict(), "./models/best_supercon_model.pth")

            # Save model info
            with open("./models/model_info.txt", "w") as f:
                f.write(f"input_size: {input_size}\n")
                f.write(f"best_epoch: {epoch + 1}\n")
                f.write(f"best_loss: {best_loss:.6f}\n")
                f.write(f"r2_score: {r2:.6f}\n")

            counter = 0  # Reset early stopping counter
        else:
            counter += 1

        # Early stopping check
        if counter >= patience:
            print(f"No improvement after {patience} epochs, stopping training")
            break

        # Save loss curves every epoch to track progress
        save_metrics_curves(train_losses, test_losses, r2_scores, epoch + 1)

    print("Training completed!")
    print(f"Best test loss: {best_loss:.4f}")
    print(f"Best R² score: {best_r2:.4f}")

    # Calculate and display RMSE for model evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy().flatten()
        y_true = y_test.numpy().flatten()
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        print(f"RMSE on test set: {rmse:.4f} K")

        # Show some sample predictions
        print("\nSample predictions:")
        for i in range(min(5, len(y_test))):
            print(f"True: {y_true[i]:.1f}K, Predicted: {y_pred[i]:.1f}K")

    # Create prediction vs true plot
    create_prediction_plot(y_true, y_pred)


def save_metrics_curves(train_losses, test_losses, r2_scores, epochs):
    """Save training metrics (loss and R²) as a publication-quality figure"""
    # Create figures directory if it doesn't exist
    if not os.path.exists("./figures"):
        os.makedirs("./figures")

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Get cividis colormap values
    cmap = plt.cm.cividis
    color1 = cmap(0.2)  # Lighter cividis
    color2 = cmap(0.5)  # Medium cividis
    color3 = cmap(0.8)  # Darker cividis

    # Set up epoch range
    epoch_range = np.arange(1, epochs + 1)

    # Plot training and validation losses
    ax1.plot(
        epoch_range,
        train_losses,
        "o-",
        color=color1,
        markersize=3,
        linewidth=1.5,
        label="Training Loss",
    )
    ax1.plot(
        epoch_range,
        test_losses,
        "s-",
        color=color2,
        markersize=3,
        linewidth=1.5,
        label="Validation Loss",
    )

    # Highlight minimum values
    min_train_epoch = np.argmin(train_losses) + 1
    min_train_loss = min(train_losses)
    min_test_epoch = np.argmin(test_losses) + 1
    min_test_loss = min(test_losses)

    # Add markers at minimum points
    ax1.plot(
        min_train_epoch,
        min_train_loss,
        "o",
        color=color1,
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=1,
    )
    ax1.plot(
        min_test_epoch,
        min_test_loss,
        "s",
        color=color2,
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=1,
    )

    # Add annotations
    ax1.annotate(
        f"Min: {min_train_loss:.4f}",
        (min_train_epoch, min_train_loss),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        color=color1,
    )
    ax1.annotate(
        f"Min: {min_test_loss:.4f}",
        (min_test_epoch, min_test_loss),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=8,
        color=color2,
    )

    # Configure axis 1
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Mean Squared Error", fontsize=11)
    ax1.set_title("Training and Validation Loss", fontsize=12)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot R² scores
    ax2.plot(
        epoch_range,
        r2_scores,
        "d-",
        color=color3,
        markersize=3,
        linewidth=1.5,
        label="R² Score",
    )

    # Highlight maximum R²
    max_r2_epoch = np.argmax(r2_scores) + 1
    max_r2 = max(r2_scores)

    # Add marker at maximum R²
    ax2.plot(
        max_r2_epoch,
        max_r2,
        "d",
        color=color3,
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=1,
    )

    # Add annotation
    ax2.annotate(
        f"Max: {max_r2:.4f}",
        (max_r2_epoch, max_r2),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        color=color3,
    )

    # Configure axis 2
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("R² Score", fontsize=11)
    ax2.set_title("Validation R² Score", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis range to show progress clearly
    if min(r2_scores) > 0:
        ax2.set_ylim([max(0, min(r2_scores) - 0.1), min(1.0, max(r2_scores) + 0.1)])

    # Ensure the figure is saved with tight layout
    plt.tight_layout()

    # Save with high resolution (300 dpi) as PDF
    plt.savefig(
        "./figures/supercon_metrics.pdf", format="pdf", dpi=300, bbox_inches="tight"
    )
    print(f"Loss and R² curves saved to ./figures/supercon_metrics.pdf")
    plt.close()


def create_prediction_plot(y_true, y_pred):
    """Create a scatter plot of predicted vs. true values with error visualization"""
    plt.figure(figsize=(8, 6))

    # Generate scatter plot with error coloration using cividis colormap
    sc = plt.scatter(
        y_true, y_pred, alpha=0.6, c=np.abs(y_pred - y_true), cmap="cividis"
    )

    # Add identity line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5)

    # Calculate R² for title
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Add colorbar for error magnitude
    cbar = plt.colorbar(sc)
    cbar.set_label("Absolute Error (K)")

    plt.xlabel("True Critical Temperature (K)")
    plt.ylabel("Predicted Critical Temperature (K)")
    plt.title(f"Prediction vs. True Values (R² = {r2:.4f}, RMSE = {rmse:.2f} K)")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Make plot square aspect ratio for better comparison
    plt.axis("equal")
    plt.tight_layout()

    # Save with high resolution as PDF
    plt.savefig(
        "./figures/supercon_predictions.pdf", format="pdf", dpi=300, bbox_inches="tight"
    )
    print("Prediction vs. true values plot saved to ./figures/supercon_predictions.pdf")
    plt.close()


if __name__ == "__main__":
    train_model(
        "./data/superconductor_data.tsv", epochs=300, batch_size=32, learning_rate=0.001
    )
