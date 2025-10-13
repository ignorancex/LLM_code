import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



if __name__ == "__main__":
    parser = argparse.ArgumentParser("AIBO Tuning Results")
    parser.add_argument("task", choices=["2cl", "5cl"], help="AIBO task")
    parser.add_argument("root", help="Path to AIBO labels")
    args = parser.parse_args()

    df = pd.read_csv(
        os.path.join(
            args.root, f"chunk_labels_{args.task}_corpus.txt"
        ),
        header=None,
        sep=" ",
    )
    df = df.rename(columns={0: "id", 1: "class", 2: "conf"})
    df["file"] = df["id"].apply(lambda x: x + ".wav")
    df["school"] = df["id"].apply(lambda x: x.split("_")[0])
    df["speaker"] = df["id"].apply(lambda x: x.split("_")[1])
    df = df.loc[df["school"] == "Mont"].reset_index(drop=True)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Palatino"
    plt.rcParams["font.size"] = 12
    plt.rcParams["legend.fontsize"] = 11
    plt.set_cmap("viridis")
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'

    root = f"results/audio/aibo-{args.task}-tuning/training"
    models = os.listdir(root)
    models = [f for f in models if os.path.isdir(os.path.join(root, f))]

    MODELS = {
        "Cnn14-16k-T": r"\emph{CNN14}",
        "Whisper-FFNN-Tiny-T": r"\emph{Whisper}$^{t}$",
        "ResNet-50-T": r"\emph{ResNet50}",
        "w2v2-l-emo": r"\emph{w2v2-L-12-emo}",
        "Seq-FFNN-IS09": r"\emph{IS10-d-lstm}"
    }

    results = {}
    for m in models:
        name = m.split("_")[1]
        if name in MODELS:
            if MODELS[name] not in results:
                results[MODELS[name]] = []
            df["pred"] = pd.read_csv(os.path.join(root, m, "_test", "test_results.csv"))["predictions"].values
            results[MODELS[name]].append(audmetric.unweighted_average_recall(df["class"], df["pred"]))

    res = pd.DataFrame(results)
    res.to_csv(f"aibo.{args.task}.tuning.results.csv", index=False)

    df = pd.read_csv(f"aibo.{args.task}.tuning.results.csv")
    
    df = df.rename(columns={
        "group": "Group",
        "uar": "UAR",
        "year": "Year",
        "macs": "GMACs"
    })
    df["Model"] = df["model"].apply(lambda x: model_dict[x])
    print(df)
    titles = {
        "2cl": r"\emph{FAU-AIBO}$^2$",
        "5cl": r"\emph{FAU-AIBO}$^5$"
    }
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot()
    sns.pointplot(
        data=df,
        x="Model",
        y="UAR",
        join=False,
        color="k"
    )
    sns.despine(ax=ax)
    ax.set_xlabel("")
    ax.set_title(titles[args.task])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results.{args.task}.png")
    plt.savefig(f"results.{args.task}.pdf")