import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from tqdm import tqdm
from typing import Iterable

plt.style.use("./myplots.mplstyle")


# --- Utility function to map model/dataset names ---
def convert_names(name):
    match name:
        case "cifar10":
            return "CIFAR-10"
        case "fashion_mnist":
            return "Fashion"
        case "mnist":
            return "MNIST"
        case "topk":
            return "Top-K"
        case "threshold":
            return "Threshold"
        case "mlp":
            return "MLP"
        case _:
            return name


# --- Plot: one line per run, hue = Model, style = Dataset ---
df_hist = pd.read_csv("histories.csv", index_col=False)
fig, axs = plt.subplots(1, 3, figsize=(7, 1.5))

for i, dataset in enumerate(df_hist.Dataset.unique()):
    df_dataset = df_hist[df_hist["Dataset"] == dataset]
    g = sns.lineplot(
        data=df_dataset,
        x="train/epoch",
        y="val_acc",  # <-- Use 'val accuracy' instead to see raw curves
        hue="Model",
        markers="Model",
        estimator="mean",  # mean across runs
        errorbar="sd",  # shaded standard deviation
        legend=True,
        ax=axs[i],
    )
    axs[i].set_title(f"{dataset}", size=8)
    order = [0, 1, 2, 3, 4]
    handles, labels = axs[i].get_legend_handles_labels()
    handles, labels = [handles[idx] for idx in order], [labels[idx] for idx in order]
    legend = axs[i].legend(
        handles,
        labels,
        ncols=5,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        columnspacing=0.75,
        handlelength=1,
    )
    curr_label = axs[i].get_legend().get_texts()[0].get_text()
    axs[i].get_legend().get_texts()[0].set_text(rf"\textbf{{{curr_label} (ours)}}")
    if i != 1:
        axs[i].get_legend().remove()
    ylabel = "Test Accuracy" if i == 0 else ""
    axs[i].set_ylabel(ylabel, fontsize=8)
    axs[i].set_xlabel("Training Epochs", fontsize=8)

fig.suptitle("Validation Accuracy vs Training Epochs", size=10, y=1.1)
handles, labels = axs[0].get_legend_handles_labels()
plt.savefig("three_lines.pdf", bbox_inches="tight")
plt.close()
