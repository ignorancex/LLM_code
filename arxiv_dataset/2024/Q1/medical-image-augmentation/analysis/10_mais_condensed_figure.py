# %% Setup
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

BUSI_AGG_PATH = "../results/busi_individual_agg.csv"
BUSI_PATH = "../results/busi_individual.csv"
BUS_BRA_PATH = "../results/bus_bra_individual.csv"
BUS_BRA_BIRADS_PATH = "../results/bus_bra_birads_individual.csv"

BUSI_TRIVIAL_PATH = "../results/busi_trivial_augment_runs.csv"
BUS_BRA_TRIVIAL_PATH = "../results/bus_bra_trivial_augment_runs.csv"
BUS_BRA_BIRADS_TRIVIAL_PATH = "../results/bus_bra_birads_trivial_augment_runs.csv"

C0 = [0.19460784, 0.45343137, 0.63284314, 0.4]
C1 = [0.88186275, 0.50539216, 0.17303922, 0.4]
C2 = ("#2ca02c", 0.4)

TRANSFORM_TO_LABEL = {
    "elastic_transform": "Elastic",
    "center_crop": "None",
    "saturation": "Saturation",
    "equalize": "Equalize",
    "vertical_flip": "Flip V.",
    "gaussian_blur": "Gauss. blur",
    "shear_x": "Shear X",
    "horizontal_flip": "Flip H.",
    "contrast": "Contrast",
    "brightness": "Brightness",
    "median_blur": "Median blur",
    "grid_distortion": "Grid distort",
    "random_crop": "Random crop",
    "translate_x": "Translate X",
    "translate_y": "Translate Y",
    "scaling": "Scale",
    "gaussian_noise": "Gauss. noise",
    "rotation": "Rotate",
    "shear_y": "Shear Y",
    "all": "TrivialAugment",
}

GEOM_ORDER = [
    "elastic_transform",
    "horizontal_flip",
    "vertical_flip",
    "grid_distortion",
    "random_crop",
    "rotation",
    "scaling",
    "shear_x",
    "shear_y",
    "translate_x",
    "translate_y",
]

PHOTO_ORDER = [
    "brightness",
    "contrast",
    "equalize",
    "gaussian_blur",
    "gaussian_noise",
    "median_blur",
    "saturation",
]

# %%
# Load results files
busi_results_df = pd.read_csv(BUSI_PATH)
bus_bra_results_df = pd.read_csv(BUS_BRA_PATH)
bus_bra_birads_results_df = pd.read_csv(BUS_BRA_BIRADS_PATH)

busi_trivial_results_df = pd.read_csv(BUSI_TRIVIAL_PATH)
bus_bra_trivial_results_df = pd.read_csv(BUS_BRA_TRIVIAL_PATH)
bus_bra_birads_trivial_results_df = pd.read_csv(BUS_BRA_BIRADS_TRIVIAL_PATH)

# %%
busi_trivial_results_subset_df = busi_trivial_results_df[
    (busi_trivial_results_df["augmentations"] == "all")
    & (busi_trivial_results_df["ops"] == 4)
].copy()
busi_trivial_results_subset_df.rename(
    columns={"augmentations": "transform"}, inplace=True
)
busi_trivial_results_subset_df.drop(columns=["ops"], inplace=True)

bus_bra_trivial_results_subset_df = bus_bra_trivial_results_df[
    (bus_bra_trivial_results_df["augmentations"] == "all")
    & (bus_bra_trivial_results_df["ops"] == 4)
].copy()
bus_bra_trivial_results_subset_df.rename(
    columns={"augmentations": "transform"}, inplace=True
)
bus_bra_trivial_results_subset_df.drop(columns=["ops"], inplace=True)

bus_bra_birads_trivial_results_subset_df = bus_bra_birads_trivial_results_df[
    (bus_bra_birads_trivial_results_df["augmentations"] == "all")
    & (bus_bra_birads_trivial_results_df["ops"] == 4)
].copy()
bus_bra_birads_trivial_results_subset_df.rename(
    columns={"augmentations": "transform"}, inplace=True
)
bus_bra_birads_trivial_results_subset_df.drop(columns=["ops"], inplace=True)


busi_results_df = pd.concat(
    [busi_results_df, busi_trivial_results_subset_df], ignore_index=True
)
bus_bra_results_df = pd.concat(
    [bus_bra_results_df, bus_bra_trivial_results_subset_df], ignore_index=True
)
bus_bra_birads_results_df = pd.concat(
    [bus_bra_birads_results_df, bus_bra_birads_trivial_results_subset_df],
    ignore_index=True,
)


# %%
# Plot results
def transform_to_category(transform):
    if transform in PHOTO_ORDER:
        return "Photometric"
    elif transform in GEOM_ORDER:
        return "Geometric"
    else:
        return "TrivialAugment"


ordering = PHOTO_ORDER + GEOM_ORDER + ["all"]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 6), sharex=True)

for ax, (dataset, results_df) in zip(
    axes.flat,
    [
        ("BUSI (Pathology)", busi_results_df),
        ("BUS-BRA (Pathology)", bus_bra_results_df),
        ("BUS-BRA (BI-RADS)", bus_bra_birads_results_df),
    ],
):
    plot_df = results_df
    plot_df["type"] = plot_df["transform"].apply(transform_to_category)
    plot_df = plot_df[plot_df["transform"] != "center_crop"]
    plot_df["diff"] = plot_df["diff"] * 100

    # Draw violin plots
    sns.violinplot(
        plot_df,
        ax=ax,
        x="transform",
        y="diff",
        hue="type",
        order=ordering,
        inner="point",
        density_norm="count",
        # cut=2,
    )
    for violin, alpha in zip(ax.collections, [0.4] * len(ax.collections)):
        violin.set_alpha(alpha)

    # Gridlines
    ax.yaxis.grid(True, which="major", color="#EEEEEE")
    ax.yaxis.grid(True, which="minor", color="#EEEEEE", linestyle=":")
    ax.set_axisbelow(True)

    # Significance markers, legend, minor gridlines
    match dataset:
        case "BUSI (Pathology)":
            ax.plot([12], [-17], marker="*", color="k")  # Rotate
            ax.plot([17], [-17], marker="*", color="k")  # Translate Y
            ax.plot([18], [-17], marker="*", color="k")  # TrivialAugment
            ax.legend().remove()
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(2))
        case "BUS-BRA (Pathology)":
            ax.plot([8], [-3], marker="*", color="k")  # Horizontal flip
            ax.plot([12], [-3], marker="*", color="k")  # Rotate
            ax.plot([18], [-3], marker="*", color="k")  # TrivialAugment
            ax.legend().remove()
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
        case "BUS-BRA (BI-RADS)":
            ax.plot([11], [-4], marker="*", color="k")  # Random crop
            ax.plot([13], [-4], marker="*", color="k")  # Scale
            ax.plot([18], [-4], marker="*", color="k")  # TrivialAugment
            legend_elements = [
                Patch(color=C0, label="Photometric"),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor="k",
                    label="Significant",
                    markersize=10,
                ),
                Patch(color=C1, label="Geometric"),
                Patch(color=C2, label="TrivialAugment"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="lower right",
                fontsize=8,
                ncols=3,
                bbox_to_anchor=(1.0075, -0.875),
                frameon=False,
            )
            ax.yaxis.set_minor_locator(MultipleLocator(1))

    # Axes
    ax.set_ylabel("$\Delta$ Bal. Acc. (%)")
    ax.set_xlabel("Transform")
    ax.set_xticks(
        ordering,
        labels=[TRANSFORM_TO_LABEL[x] for x in ordering],
        rotation=22.5,
        ha="right",
    )
    ax.set_title(dataset, fontsize=10)
    fig.align_ylabels(axes)

plt.tight_layout()
plt.savefig(
    "../results/individual_vs_trivial_augment_effects_violins.pdf", bbox_inches="tight"
)
plt.show()

# %%
