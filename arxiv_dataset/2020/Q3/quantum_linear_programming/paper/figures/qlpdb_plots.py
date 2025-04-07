import matplotlib.pyplot as plt
from qlpdb.data.models import Data as data_Data
import numpy as np


def single_histogram(datadict):
    """Plots histogram of results
    """
    fig, axs = plt.subplots(
        len(datadict), 1, sharey=True, sharex=True, tight_layout=True, figsize=(5, 15)
    )
    print(np.shape(axs))
    for idx, key in enumerate(datadict):
        query = datadict[key].groupby("energy").count()["id"]
        X = query.index
        height = np.log(query.values)
        axs[idx].bar(x=X, height=height, width=0.9, bottom=0, align="center")
    plt.draw()
    plt.show()


def degenfcn(n):
    if n % 3 == 2:
        return n // 3 + 2
    if n % 3 == 1:
        return 2 * (n // 3 + 1)
    if n % 3 == 0:
        return 1


def plot_prob(datadict, datadict2, datadict3):
    """Plots percentage getting correct answer
    """

    X = [datadict[key]["energy"].min() for key in datadict.keys()]
    X = range(2, len(datadict.keys()) + 2)
    yhmc = np.array(
        [1, 0.738, 0.823, 0.657, 0.271, 0.568, 0.336, 0.1, 0.318, 0.158, 0.029, 0.169]
    )[np.array(list(X))-2]

    y = [
        datadict[key].groupby("energy").count()["id"].iloc[0]
        / datadict[key].count()["id"]
        for key in datadict.keys()
    ]
    y2 = [
        datadict2[key].groupby("energy").count()["id"].iloc[0]
        / datadict2[key].count()["id"]
        for key in datadict2.keys()
    ]
    y3 = [
        datadict3[key].groupby("energy").count()["id"].iloc[0]
        / datadict3[key].count()["id"]
        for key in datadict3.keys()
    ]
    degeneracy = np.array([degenfcn(key) for key in datadict.keys()])
    yscaled = y / degeneracy
    rguess = np.array([1 / 2 ** xi for xi in X]) * degeneracy

    fig = plt.figure("scaling", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.errorbar(x=X, y=rguess, marker="^", label="Random guessing", color="k")
    ax.errorbar(x=X, y=y, marker="o", label="Constant offset", color="#70b741")
    # ax.errorbar(x=X, y=y/rguess, marker="*", label="Constant / Random")

    ax.errorbar(x=X, y=y2, marker="o", label="Linear offset", color="#51a7f9")
    # ax.errorbar(x=X, y=y2/rguess, marker="*", label="Linear / Random")
    ax.errorbar(x=X, y=y3, marker="o", label="Neg. Linear offset", color="#c82506")
    ax.errorbar(x=X, y=yhmc, marker="o", label="HMC", color="cyan")
    ax.set_yscale("log")
    ax.set_xlabel("Nodes in graph")
    ax.set_ylabel("Probability of getting MDS solution")
    plt.legend(fancybox=True, framealpha=0.5)
    plt.savefig("./scaling_plot.pdf")
    plt.draw()
    plt.show()

    fig = plt.figure("ratio", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.errorbar(
        x=X, y=y / rguess, marker="o", label="Constant / Random", color="#70b741"
    )
    ax.errorbar(
        x=X, y=y2 / rguess, marker="*", label="Linear / Random", color="#51a7f9"
    )
    ax.errorbar(
        x=X, y=y3 / rguess, marker="*", label="Neg. Linear / Random", color="#c82506"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Nodes in graph")
    ax.set_ylabel("Improvement over random guessing")
    plt.legend(fancybox=True, framealpha=0.5)
    plt.savefig("./ratio.pdf")
    plt.draw()
    plt.show()

    fig = plt.figure("Lc_ratio", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.errorbar(
        x=X,
        y=100 * (np.array(y2) / np.array(y) - 1.0),
        marker="^",
        label="Linear / Constant",
        color="#51a7f9"
    )
    ax.errorbar(
        x=X,
        y=100 * (np.array(y3) / np.array(y) - 1.0),
        marker="^",
        label="Neg. Linear / Constant",
        color="#c82506"
    )
    #ax.errorbar(
    #    x=X,
    #    y=100*(yhmc / np.array(y) - 1.0),
    #    marker="^",
    #    label="HMC / Constant",
    #    color="cyan"
    #)
    ax.set_xlabel("Nodes in graph")
    ax.set_ylabel("Percent improvement over Constant")
    plt.legend(fancybox=True, framealpha=0.5)
    plt.savefig("./ratio_pct.pdf")
    plt.draw()
    plt.show()


if __name__ == "__main__":
    nndata = {
        key: data_Data.objects.filter(
            experiment__graph__tag=f"NN({key})", experiment__tag="Constant"
        ).to_dataframe()
        for key in range(2, 11)
    }
    nndata2 = {
        key: data_Data.objects.filter(
            experiment__graph__tag=f"NN({key})", experiment__tag="Linear_-0.1_0.12"
        ).to_dataframe()
        for key in range(2, 11)
    }
    nndata3 = {
        key: data_Data.objects.filter(
            experiment__graph__tag=f"NN({key})", experiment__tag="Neglinear_-0.1_0.12"
        ).to_dataframe()
        for key in range(2, 11)
    }

    # single_histogram(nndata)
    plot_prob(nndata, nndata2, nndata3)


