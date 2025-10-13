import argparse
import audmetric
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot MSP-Podcast Tuning Results")
    parser.add_argument("labels", help="Path to MSP-Podcast labels")
    args = parse.parse_args()

    df = pd.read_csv(args.labels)
    df = df.loc[df["Split_Set"] == "Test1"]
    df = df.loc[df["EmoClass"].isin(["A", "H", "N", "S"])]
    df = df.reset_index(drop=True)
    print(df.info())

    def get_CI(y_true, y_pred, metric = audmetric.unweighted_average_recall):
        global_result = metric(y_true, y_pred)

        results = []
        for s in range(1000):
            np.random.seed(s)
            sample = np.random.choice(range(len(y_pred)), len(y_pred), replace=True) #boost with replacement
            sample_preds = y_pred[sample]
            sample_labels = y_true[sample]
            results.append(metric(sample_labels, sample_preds))

        q_0 = pd.DataFrame(np.array(results)).quantile(0.025)[0] #2.5% percentile
        q_1 = pd.DataFrame(np.array(results)).quantile(0.975)[0] #97.5% percentile

        return(f"{global_result:.3f} [{q_0:.3f} - {q_1:.3f}]")

    root = "results/audio/msp-tuning/training"
    models = os.listdir(root)
    models = [f for f in models if os.path.isdir(os.path.join(root, f))]

    MODELS = {
        "w2v2-l-rob": r"\emph{w2v2-L-robust}",
        "w2v2-l-emo": r"\emph{w2v2-L-12-emo}",
        "w2v2-l-100k": r"\emph{w2v2-L-vox}",
        "hubert-l-ll60k": r"\emph{hubert-L}",
        "EfficientNet-B0-T": r"\emph{EfficientNet}"
    }

    results = {}
    for m in models:
        name = m.split("_")[1]
        if name in MODELS:
            if MODELS[name] not in results:
                results[MODELS[name]] = []
            df["pred"] = pd.read_csv(os.path.join(root, m, "_test", "test_results.csv"))["predictions"].values
            results[MODELS[name]].append(audmetric.unweighted_average_recall(df["EmoClass"], df["pred"]))

    res = pd.DataFrame(results)
    res.to_csv("msp.tuning.results.csv")

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Palatino"
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["font.size"] = 12
    plt.set_cmap("viridis")
    fig, ax = plt.subplots(1, 1)
    sns.despine(ax=ax)
    sns.pointplot(
        data=res,
        ax=ax,
        join=False,
        color="k"
    )
    ax.set_title(r"\emph{MSP-Podcast}")
    ax.set_ylabel("UAR")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("msp.png")
    plt.savefig("msp.pdf")