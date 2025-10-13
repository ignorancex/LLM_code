import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def plot_uncertainty(output_npy_path: Path):
    outputs = np.load(output_npy_path, allow_pickle=True)

    var = outputs.item().get("var")
    uncs = np.sqrt(var)
    labels = outputs.item().get("labels")
    preds = outputs.item().get("mean")
    noise = outputs.item().get("noise")

    noisy_mask = noise > 0
    uncs_noisy = uncs[noisy_mask]
    uncs_clean = uncs[~noisy_mask]
    logger.info(f"Mean uncertainty for noisy samples: {np.mean(uncs_noisy)}")
    logger.info(f"Mean uncertainty for clean samples: {np.mean(uncs_clean)}")

    _, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].scatter(labels, preds, c=uncs, cmap="coolwarm")
    ax[0, 0].plot([1, 7], [1, 7], "k--")
    ax[0, 0].set_xlabel("True")
    ax[0, 0].set_ylabel("Predicted")
    ax[0, 1].scatter(labels, preds, c=noise, cmap="coolwarm")
    ax[0, 1].plot([1, 7], [1, 7], "k--")
    ax[0, 1].set_xlabel("True")
    ax[0, 1].set_ylabel("Predicted")

    ax[1, 0].hist(uncs_noisy, bins=50, alpha=0.5, label="Noisy", color="red")
    ax[1, 0].hist(uncs_clean, bins=50, alpha=0.5, label="Clean", color="blue")
    ax[1, 0].legend()
    ax[1, 0].set_xlabel("Uncertainty")
    ax[1, 0].set_ylabel("Frequency")

    ax[1, 1].boxplot([uncs_noisy, uncs_clean], tick_labels=["Noisy", "Clean"])
    ax[1, 1].set_ylabel("Uncertainty")
    ax[1, 1].set_xlabel("Sample type")

    plt.tight_layout()
    plt.savefig(f"{output_npy_path.parent}/uncertainty.pdf")
    logger.info(f"Saved plot at {output_npy_path.parent}/uncertainty.pdf")


if __name__ == "__main__":
    output_npy_path = Path("outputs/2025-08-01/11-12-20_newsemp_cross-prob_lambdas-[1, 9.110462266012783, 5.5635098435909764]_tune-False_single-model_4-passes-label-noise-0.1/seed_0/outputs_new.npy")

    plot_uncertainty(output_npy_path)
