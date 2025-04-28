import numpy as np
import pandas as pd

if __name__ == "__main__":
    # df0 = pd.read_parquet("results/ihdp/dragonnet/results.parquet")
    df1 = pd.read_parquet("results/ihdp/madnet_ablation/results.parquet")
    df2 = pd.read_parquet("results/ihdp/riesznet_ablation/results.parquet")
    df3 = pd.read_parquet("results/bhp/madnet_ablation/results.parquet")
    df4 = pd.read_parquet("results/bhp/riesznet_ablation/results.parquet")
    df = pd.concat([df1, df2, df3, df4])
    df = df[df["estimator"] != "tmle"]
    df["bias"] = (df["estimate"] - df["true_outcome_moment"]) * df["y_std"]
    rename = {
        "plug_in": "Direct",
        "plug_in_tmle": "Direct + SRR",
        "ipw": "IPW",
        "ipw_tmle": "IPW + SRR",
        "double_robust": "DR",
        "double_robust_tmle": "DR + SRR",
        "tmle": "TMLE",
    }
    df["estimator"] = df["estimator"].apply(lambda x: rename[x])
    df["Estimator"] = df["learner"] + " (" + df["estimator"] + ")"
    df["Dataset"] = df["dataset"].apply(lambda s: s.upper())

    desc = df.groupby(["Dataset", "Estimator"])["absolute_error"].describe()
    desc["std_err"] = desc["std"] / np.sqrt(desc["count"])

    print("================")
    print("Summary stastics")
    print("================")
    print(desc)

    # Save to latex table
    desc.rename(
        columns={
            "mean": "Mean Absolute Error (MAE)",
            "std_err": "Standard Error in MAE",
            "50%": "Median Absolute Error",
        }
    ).to_latex(
        "paper/reproduction_results.tex",
        columns=[
            "Mean Absolute Error (MAE)",
            "Standard Error in MAE",
            "Median Absolute Error",
        ],
        float_format="{:.3f}".format,
    )
