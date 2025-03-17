import os
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pylint: disable=no-value-for-parameter


def compute_contrip(x, y, w1, w2):

    return (
        np.minimum(np.full((y.shape[0], x.shape[1]), 5), x + (y - 0.5) * w1)
        - (1 - y) / w2 * x
        - (5 - x) / 100
    )


def scaling(score):
    return (score - 0.61) / (5 - 0.61) * 4 + 1


def generate_dataframe(score, consensus):

    df = pd.DataFrame()
    df["ConTrip Score"] = np.ndarray.flatten(score, "F")
    df["Consensus Score"] = np.asarray([consensus] * 5).flatten()
    df["Rating"] = np.asarray(
        [
            ["Rating: 1"] * 51,
            ["Rating: 2"] * 51,
            ["Rating: 3"] * 51,
            ["Rating: 4"] * 51,
            ["Rating: 5"] * 51,
        ]
    ).flatten()

    return df


def generate_dataframe_2(score, rating):

    df = pd.DataFrame()
    df["ConTrip Score"] = score.flatten()
    df["Rating"] = np.asarray([rating] * 6).flatten()
    df["Consensus Score"] = np.asarray(
        [
            ["Consensus: 0.0"] * 41,
            ["Consensus: 0.2"] * 41,
            ["Consensus: 0.4"] * 41,
            ["Consensus: 0.6"] * 41,
            ["Consensus: 0.8"] * 41,
            ["Consensus: 1.0"] * 41,
        ]
    ).flatten()

    return df


def generate_plots_1(df, output, scale):

    fig, ax = plt.subplots(figsize=(5, 5))

    sns.scatterplot(
        data=df,
        x="Consensus Score",
        y="ConTrip Score",
        hue="Rating",
        palette=["#d7191c", "#fdae61", "#fee08b", "#abdda4", "#2b83ba"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    custom_lines = []
    for el in [
        ["Rating: 1", "#d7191c"],
        ["Rating: 2", "#fdae61"],
        ["Rating: 3", "#fee08b"],
        ["Rating: 4", "#abdda4"],
        ["Rating: 5", "#2b83ba"],
    ]:

        custom_lines.append(
            plt.plot(
                [],
                [],
                marker="o",
                ms=7,
                ls="",
                mec="black",
                mew=0,
                color=el[1],
                label=el[0],
            )[0]
        )

    ax.legend(
        bbox_to_anchor=(0.0, 1.05, 1.0, 0.102),
        handles=custom_lines,
        loc="upper center",
        facecolor="white",
        ncol=3,
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()

    if scale:
        out_file = os.path.join(output, "contrip_score_consensus_scaling.pdf")
    else:
        out_file = os.path.join(output, "contrip_score_consensus.pdf")

    plt.savefig(out_file)
    plt.close()


def generate_plots_2(df, output, scale):

    fig, ax = plt.subplots(figsize=(5, 5))

    sns.scatterplot(
        data=df,
        x="Rating",
        y="ConTrip Score",
        hue="Consensus Score",
        palette=["#d7191c", "#fdae61", "#fee08b", "#abdda4", "#2b83ba", "#5e4fa2"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    custom_lines = []
    for el in [
        ["Consensus: 0.0", "#d7191c"],
        ["Consensus: 0.2", "#fdae61"],
        ["Consensus: 0.4", "#fee08b"],
        ["Consensus: 0.6", "#abdda4"],
        ["Consensus: 0.8", "#2b83ba"],
        ["Consensus: 1.0", "#5e4fa2"],
    ]:

        custom_lines.append(
            plt.plot(
                [],
                [],
                marker="o",
                ms=7,
                ls="",
                mec="black",
                mew=0,
                color=el[1],
                label=el[0],
            )[0]
        )

    ax.legend(
        bbox_to_anchor=(0.0, 1.05, 1.0, 0.102),
        handles=custom_lines,
        loc="upper center",
        facecolor="white",
        ncol=3,
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()

    if scale:
        out_file = os.path.join(output, "contrip_score_rating_scaling.pdf")
    else:
        out_file = os.path.join(output, "contrip_score_rating.pdf")

    plt.savefig(out_file)
    plt.close()


def generate_figure_1(weight_1, weight_2, scale, output):

    rating = np.linspace(1, 5, 5).reshape(1, 5)
    consensus = np.linspace(0, 1, 51).reshape(51, 1)

    score = compute_contrip(rating, consensus, weight_1, weight_2)

    if scale:
        score = scaling(score)

    df = generate_dataframe(score, consensus)

    generate_plots_1(df, output, scale)


def generate_figure_2(weight_1, weight_2, scale, output):

    rating = np.linspace(1, 5, 41).reshape(1, 41)
    consensus = np.linspace(0, 1, 6).reshape(6, 1)

    score = compute_contrip(rating, consensus, weight_1, weight_2)

    if scale:
        score = scaling(score)

    df = generate_dataframe_2(score, rating)

    generate_plots_2(df, output, scale)


# ------------------------------------------------------------------------------
# CLICK
# ------------------------------------------------------------------------------


@click.command(
    short_help="script to study how different parameters change"
    "the behavior of the score of a tripadvisor rating"
)
@click.option(
    "-w1",
    "--weight_1",
    default=0.5,
    help="weight for the importance of the consensus value",
)
@click.option(
    "-w2",
    "--weight_2",
    default=10.0,
    help="weight for the importance of the consensus value 2",
)
@click.option("-sf", "--scaling_flag", is_flag=True, help="flag to perform the scaling")
@click.option("-o", "--output", default="", help="output folder")
def main(weight_1, weight_2, scaling_flag, output):

    generate_figure_1(weight_1, weight_2, scaling_flag, output)
    generate_figure_2(weight_1, weight_2, scaling_flag, output)


if __name__ == "__main__":
    main()
