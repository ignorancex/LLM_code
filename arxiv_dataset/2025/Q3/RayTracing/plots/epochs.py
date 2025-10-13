# %%
import wandb
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use("./myplots.mplstyle")


# utility functions
def convert_names(name):
    match name:
        case "cifar10":
            return "CIFAR-10"
        case "fashion_mnist":
            return "Fashion"
        case "mnist":
            return "MNIST"
        # models
        case "topk":
            return "Top-K"
        case "threshold":
            return "Threshold"
        case "mlp":
            return "MLP"
        case _:
            return name


# Get all runs in the sweep
def get_runs_df(sweeps):
    df = pd.DataFrame()
    for sweep in sweeps:
        runs = sweep.runs
        # Collect desired data for each run
        data = []
        for run in runs:
            row = {
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                **run.config,  # hyperparameters
                **run.summary,  # final metrics (like accuracy, loss, etc.)
            }
            data.append(row)
        # Create a pandas DataFrame
        df = pd.concat([df, pd.DataFrame(data)])
    return df


# init WANDB API
api = wandb.Api()
# convert raytracing runs to a dataframe
sweep_names = [
    "silvretta/PaperSweeps/nic7cean",  # mnist
    "silvretta/PaperSweeps/47du0a9z",  # cifar
    "silvretta/PaperSweeps/mfp4wj7b",  # fashion
]
sweeps = [api.sweep(path=r) for r in sweep_names]
df1 = get_runs_df(sweeps)

# same for baseline runs
baseline_sweep_names = [  # <-- CHANGE BASELINE SWEEPS HERE
    "silvretta/BaselineMoE/4ebkz5l8",  # Top-K Fashion
    "silvretta/BaselineMoe/r3veu3o0",  # Top-K MNIST
    "silvretta/BaselineMoe/njmhpwro",  # Top-K CIFAR
    "silvretta/BaselineMoe/lgfjy9bn",  # MLP Big Fashion
    "silvretta/BaselineMoe/pyxqxb29",  # MLP Big MNIST
    "silvretta/BaselineMoe/5vigjidg",  # MLP Big CIFAR
    "silvretta/BaselineMoe/2pbvl4nb",  # MLP Small Fashion
    "silvretta/BaselineMoe/0j749h77",  # MLP Small MNIST
    "silvretta/BaselineMoe/x00bla5t",  # MLP Small CIFAR
    "silvretta/BaselineMoe/x74ozi5g",  # Threshold CIFAR
    "silvretta/BaselineMoE/1esinu88",  # Threshold MNIST
    "silvretta/BaselineMoE/rrd4xcxy",  # Threshold Fashion
]

baseline_sweeps = [api.sweep(path=r) for r in baseline_sweep_names]
df2 = get_runs_df(baseline_sweeps)

# Rename and keep only the relevant columns
df1.rename(columns={"test accuracy": "test/acc"}, inplace=True)
df1["model"] = "RayTracing"
df = pd.concat([df1, df2])
df = df.map(convert_names)
df = df[["model", "test/acc", "train/epoch", "lr", "temp", "dset", "hdim"]]
df.rename(columns={"model": "Model", "dset": "Dataset"}, inplace=True)
# Filter
df = df[df["lr"] == 1e-3]
df = df[(df["temp"] == 50) | (df["Model"] != "RayTracing")]

# Compute the stats
df_aggr = df.groupby(["Model", "hdim", "Dataset"])
df_mean = df_aggr.mean(numeric_only=False)
df_std = df_aggr.std(numeric_only=False)
df_mean.reset_index(inplace=True)
df_std.reset_index(inplace=True)


def rename_mlp(df):
    df.loc[(df["Model"] == "MLP") & (df["hdim"] == 36), "Model"] = "MLP (total)"
    df.loc[(df["Model"] == "MLP") & (df["hdim"] == 24), "Model"] = "MLP (avg)"
    return df


df_mean = rename_mlp(df_mean)
df_std = rename_mlp(df_std)

# %%
plt.style.use("./myplots.mplstyle")
# merge the dataframes to work with both at once
df = df_mean.copy()
df["xerr"] = df_std["train/epoch"]
df["yerr"] = df_std["test/acc"]
# Create Seaborn palette and markers to map hue and style manually
palette = sns.color_palette("tab10", df["Model"].nunique())
markers = ["o", "s", "^", "D", "P", "X"]  # Extend if needed
hue_order = df["Model"].unique()
style_order = df["Dataset"].unique()
# Build dicts for consistent mapping
hue_map = {val: palette[i] for i, val in enumerate(hue_order)}
style_map = {val: markers[i] for i, val in enumerate(style_order)}
# Start plot
fig, axs = plt.subplots(1, 3, figsize=(7, 1.5))
# Plot error bars manually
for i, dset in enumerate(df["Dataset"].unique()):
    df_ = df[df["Dataset"] == dset]
    xrange = df_["train/epoch"].max() - df_["train/epoch"].min()
    yrange = df_["test/acc"].max() - df_["test/acc"].min()
    cross_thickness_x = 0.02 * xrange
    cross_thickness_y = 0.02 * yrange
    axs[i].set_title(f"{dset}", fontsize=8)
    for _, row in df_.iterrows():
        x = row["train/epoch"]
        y = row["test/acc"]
        dx = row["xerr"]
        dy = row["yerr"]
        color = hue_map[row["Model"]]

        axs[i].errorbar(
            row["train/epoch"],
            row["test/acc"],
            xerr=row["xerr"],
            yerr=row["yerr"],
            fmt=",",
            color=hue_map[row["Model"]],
            ecolor=hue_map[row["Model"]],
            linewidth=1,
            alpha=0.7,
            capsize=1,
        )

    # Plot actual points using Seaborn for legend
    sns.scatterplot(
        data=df_,
        x="train/epoch",
        y="test/acc",
        hue="Model",
        palette=hue_map,
        markers=style_map,
        s=20,  # marker size
        legend=True,
        ax=axs[i],
    )
    order = [2, 3, 0, 1, 4]
    handles, labels = axs[i].get_legend_handles_labels()
    handles, labels = [handles[idx] for idx in order], [labels[idx] for idx in order]
    legend = axs[i].legend(
        handles,
        labels,
        ncols=5,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.6),
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


fig.suptitle("Test Accuracy vs Training Epochs", size=10, y=1.1)
handles, labels = axs[0].get_legend_handles_labels()
plt.savefig("three_scatters.pdf", bbox_inches="tight")
# plt.show()
